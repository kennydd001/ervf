from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p10_cache_family import (
    DOMAINS, EXPERTS, LAYERS, TEST, dynamic_slots, frequency_tables, load_routes,
    concatenate, top_sets, stats, sha256,
)
from scripts.streamq5_moe.run_p2c_physical_h2d import (
    CACHE_BYTES, EXPERT_BYTES, pin_full_bank, copy_record, layer_bases, sample_matches,
)


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P10B_DOMAIN_CACHE_PHYSICAL_PREREGISTRATION.md"
P10 = R / "p10_cache_family.json"
BANK_RESULT = R / "p1d_physical_bank_result.json"
OUTPUT = R / "p10b_domain_cache_physical.json"


def percentile(values):
    return {key: value for key, value in stats(values).items() if key != "sum"}


def profile_data(routes):
    counts, global_counts = frequency_tables(routes, slice(0, 512))
    global12 = [list(values) for values in top_sets(global_counts, 12)]
    ids = np.arange(EXPERTS)
    profile8 = {}
    for domain in DOMAINS:
        profile8[domain] = []
        for layer in range(LAYERS):
            ranking = np.lexsort((ids, -counts[domain][layer]))
            selected = [int(expert) for expert in ranking if int(expert) not in set(global12[layer])][:8]
            profile8[domain].append(selected)
    likelihoods = {domain: (counts[domain] + 1) / (counts[domain].sum(axis=1, keepdims=True) + EXPERTS) for domain in DOMAINS}
    return global12, profile8, likelihoods


def choose_auto(window, likelihoods):
    observed = np.stack(tuple(window))
    scores = {}
    for domain in DOMAINS:
        score = 0.0
        for layer in range(LAYERS):
            score += float(np.log(likelihoods[domain][layer, observed[:, layer]].clip(1e-12)).sum())
        scores[domain] = score
    return max(DOMAINS, key=lambda domain: (scores[domain], -DOMAINS.index(domain)))


def run_policy(name, sequence, labels, pinned, cache_memory, bases, global12, profile8, likelihoods):
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.runtime.memsetAsync(cache_memory.ptr, 0, CACHE_BYTES, stream.ptr)
    universal = name == "universal"
    if universal:
        counts_dummy = None
        # Universal rankings are supplied through global12 as 20 entries.
        for layer in range(LAYERS):
            for slot, expert in enumerate(global12[layer]):
                copy_record(stream, pinned, cache_memory, bases, layer, expert, slot)
    else:
        for layer in range(LAYERS):
            for slot, expert in enumerate(global12[layer]):
                copy_record(stream, pinned, cache_memory, bases, layer, expert, slot)
    stream.synchronize()
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    active = None
    profile_slots = [{} for _ in range(LAYERS)]
    window = deque(maxlen=16)
    misses, copies, event_ms, wall_ms = [], [], [], []
    last_copies = [None] * LAYERS
    for token, route in enumerate(sequence):
        window.append(route)
        chosen = None if universal else (labels[token] if name == "oracle" else choose_auto(window, likelihoods))
        token_misses = 0; token_copies = 0
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        wall_begin = time.perf_counter_ns(); begin.record(stream)
        if not universal and chosen != active:
            for layer in range(LAYERS):
                old_mapping = profile_slots[layer]
                new_set = set(profile8[chosen][layer])
                kept = {expert: slot for expert, slot in old_mapping.items() if expert in new_set}
                free_slots = [slot for slot in range(12, 20) if slot not in kept.values()]
                for expert in profile8[chosen][layer]:
                    if expert in kept: continue
                    slot = free_slots.pop(0)
                    kept[expert] = slot
                    copy_record(stream, pinned, cache_memory, bases, layer, expert, slot)
                    last_copies[layer] = (expert, slot)
                    token_copies += 1
                profile_slots[layer] = kept
            active = chosen
        for layer in range(LAYERS):
            if universal:
                fixed_mapping = {expert: slot for slot, expert in enumerate(global12[layer])}
            else:
                fixed_mapping = {expert: slot for slot, expert in enumerate(global12[layer])}
                fixed_mapping.update(profile_slots[layer])
            lru = dynamic[layer]
            for raw in route[layer]:
                expert = int(raw)
                if expert in fixed_mapping:
                    continue
                if expert in lru:
                    lru.move_to_end(expert)
                else:
                    token_misses += 1; token_copies += 1
                    if len(lru) < dynamic_slots(layer):
                        slot = 20 + len(lru)
                    else:
                        _old, slot = lru.popitem(last=False)
                    lru[expert] = slot
                    copy_record(stream, pinned, cache_memory, bases, layer, expert, slot)
                    last_copies[layer] = (expert, slot)
        end.record(stream); end.synchronize()
        event_ms.append(float(cp.cuda.get_elapsed_time(begin, end)))
        wall_ms.append((time.perf_counter_ns() - wall_begin) / 1e6)
        misses.append(token_misses); copies.append(token_copies)
        if token % 64 == 0:
            print(json.dumps({"policy": name, "token": token, "copies": token_copies, "event_ms": event_ms[-1]}), flush=True)
    integrity_failures = 0
    for layer, item in enumerate(last_copies):
        if item is not None:
            integrity_failures += int(not sample_matches(pinned, cache_memory, bases, layer, item[0], item[1]))
    return {
        "tokens": len(sequence), "misses": misses, "copies": copies,
        "miss_stats": percentile(misses), "copy_stats": percentile(copies),
        "event_ms_stats": percentile(event_ms), "wall_ms_stats": percentile(wall_ms),
        "total_copies": int(sum(copies)), "h2d_bytes": int(sum(copies)) * EXPERT_BYTES,
        "integrity_failures": integrity_failures,
    }


def main():
    p10 = json.loads(P10.read_text(encoding="utf-8"))
    bank = json.loads(BANK_RESULT.read_text(encoding="utf-8"))
    routes, route_hashes = load_routes()
    sequence, labels = concatenate(routes, TEST, [(domain, 128) for domain in DOMAINS[:4]])
    global12, profile8, likelihoods = profile_data(routes)
    _, global_counts = frequency_tables(routes, slice(0, 512))
    universal20 = [list(values) for values in top_sets(global_counts, 20)]
    pinned, pinned_hashes, pin_ms = pin_full_bank(bank)
    cache_memory = cp.cuda.alloc(CACHE_BYTES)
    bases = layer_bases()
    results = {}
    for policy in ("universal", "oracle", "automatic"):
        fixed = universal20 if policy == "universal" else global12
        results[policy] = run_policy(policy, sequence, labels, pinned, cache_memory, bases, fixed, profile8, likelihoods)
    expected = p10["domain_conditioning"]["sequences"]["switch_128"]
    exact_counts = {
        policy: results[policy]["total_copies"] == expected[policy]["copies"]["sum"]
        for policy in results
    }
    automatic, oracle, universal = results["automatic"], results["oracle"], results["universal"]
    ratios = {
        "event_mean_auto_to_oracle": automatic["event_ms_stats"]["mean"] / oracle["event_ms_stats"]["mean"],
        "event_p95_auto_to_oracle": automatic["event_ms_stats"]["p95"] / oracle["event_ms_stats"]["p95"],
        "event_mean_auto_to_universal": automatic["event_ms_stats"]["mean"] / universal["event_ms_stats"]["mean"],
        "event_p95_auto_to_universal": automatic["event_ms_stats"]["p95"] / universal["event_ms_stats"]["p95"],
    }
    gates = {
        "copy_counts_exact": all(exact_counts.values()),
        "integrity": all(result["integrity_failures"] == 0 for result in results.values()),
        "auto_mean_le_110pct_oracle": ratios["event_mean_auto_to_oracle"] <= 1.10,
        "auto_p95_le_110pct_oracle": ratios["event_p95_auto_to_oracle"] <= 1.10,
        "auto_mean_le_95pct_universal": ratios["event_mean_auto_to_universal"] <= 0.95,
        "auto_p95_le_95pct_universal": ratios["event_p95_auto_to_universal"] <= 0.95,
    }
    output = {
        "kind": "streamq5_moe_p10b_domain_cache_physical", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "p10_sha256": sha256(P10), "bank_result_sha256": sha256(BANK_RESULT),
        "route_hashes": route_hashes, "pin_ms": pin_ms, "pinned_hashes": pinned_hashes,
        "results": results, "exact_counts": exact_counts, "ratios": ratios, "gates": gates,
        "overall_pass": all(gates.values()),
        "claim_boundary": "Physical H2D-only test on the fixed switch sequence; full-decoder latency remains untested.",
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"counts": {p: r["total_copies"] for p, r in results.items()}, "exact_counts": exact_counts, "ratios": ratios, "gates": gates, "overall_pass": output["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
