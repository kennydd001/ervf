from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p2b_routes"
PREREG = R / "P10_CACHE_FAMILY_PREREGISTRATION.md"
OUTPUT = R / "p10_cache_family.json"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
EXPERT_BYTES = 3_035_136
VAL = slice(0, 512)
TEST = slice(512, 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dynamic_slots(layer: int) -> int:
    return 15 if layer < 8 else 14


def stats(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
        "max": float(x.max()), "sum": int(x.sum()),
    }


def load_routes():
    result = {domain: [] for domain in DOMAINS}
    hashes = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                result[domain].append(handle.get_tensor(f"{domain}_router_ids").astype(np.int16))
    return {domain: np.stack(values, axis=1) for domain, values in result.items()}, hashes


def frequency_tables(routes, split):
    tables = {domain: np.zeros((LAYERS, EXPERTS), dtype=np.int64) for domain in DOMAINS}
    for domain in DOMAINS:
        for layer in range(LAYERS):
            tables[domain][layer] = np.bincount(routes[domain][split, layer].reshape(-1), minlength=EXPERTS)
    global_table = sum(tables.values())
    return tables, global_table


def top_sets(table, count):
    ids = np.arange(EXPERTS)
    return [tuple(int(x) for x in np.lexsort((ids, -table[layer]))[:count]) for layer in range(LAYERS)]


def simulate_sequence(sequence, fixed, capacities, policy="lru", pair=None):
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    frequencies = [Counter() for _ in range(LAYERS)]
    probation = [OrderedDict() for _ in range(LAYERS)]
    protected = [OrderedDict() for _ in range(LAYERS)]
    request_misses = np.zeros(len(sequence), dtype=np.int64)
    copies = np.zeros(len(sequence), dtype=np.int64)
    prefetch_copies = np.zeros(len(sequence), dtype=np.int64)
    for token, route in enumerate(sequence):
        for layer in range(route.shape[0]):
            fixed_set = fixed[layer]
            capacity = capacities[layer]
            requested = [int(x) for x in route[layer]]
            for expert in requested:
                frequencies[layer][expert] += 1
                if expert in fixed_set:
                    continue
                if policy == "2q":
                    protected_cap = max(1, capacity * 3 // 4)
                    probation_cap = max(1, capacity - protected_cap)
                    if expert in protected[layer]:
                        protected[layer].move_to_end(expert)
                        continue
                    if expert in probation[layer]:
                        probation[layer].pop(expert)
                        protected[layer][expert] = None
                        if len(protected[layer]) > protected_cap:
                            old, _ = protected[layer].popitem(last=False)
                            probation[layer][old] = None
                            while len(probation[layer]) > probation_cap:
                                probation[layer].popitem(last=False)
                        continue
                    request_misses[token] += 1; copies[token] += 1
                    probation[layer][expert] = None
                    if len(probation[layer]) > probation_cap:
                        probation[layer].popitem(last=False)
                    continue
                cache = dynamic[layer]
                if expert in cache:
                    cache.move_to_end(expert)
                    continue
                request_misses[token] += 1
                admit = True
                if policy == "tinylfu" and len(cache) >= capacity:
                    victim = next(iter(cache))
                    admit = frequencies[layer][expert] >= frequencies[layer][victim]
                if admit and capacity:
                    copies[token] += 1
                    if len(cache) >= capacity:
                        if policy == "lfu":
                            victim = min(cache, key=lambda item: (frequencies[layer][item], cache[item]))
                            del cache[victim]
                        else:
                            cache.popitem(last=False)
                    cache[expert] = token
            if policy == "pair" and capacity:
                cache = dynamic[layer]
                candidates = []
                for expert in requested:
                    candidate = pair[layer, expert]
                    if candidate >= 0 and candidate not in fixed_set and candidate not in cache and candidate not in requested:
                        candidates.append(int(candidate))
                if candidates:
                    candidate = candidates[0]
                    if len(cache) >= capacity:
                        cache.popitem(last=False)
                    cache[candidate] = token
                    copies[token] += 1; prefetch_copies[token] += 1
    return {
        "request_misses": request_misses, "copies": copies,
        "prefetch_copies": prefetch_copies,
    }


def summarize(result):
    return {
        "request_misses": stats(result["request_misses"]),
        "copies": stats(result["copies"]),
        "prefetch_copies": stats(result["prefetch_copies"]),
        "h2d_bytes": int(result["copies"].sum()) * EXPERT_BYTES,
    }


def concatenate(routes, split, chunks):
    cursors = {domain: split.start for domain in DOMAINS}
    pieces, labels = [], []
    for domain, length in chunks:
        begin = cursors[domain]; end = begin + length
        if end > split.stop:
            begin = split.start; end = begin + length
        pieces.append(routes[domain][begin:end])
        labels.extend([domain] * length)
        cursors[domain] = end
    return np.concatenate(pieces), labels


def pair_table(routes, split):
    counts = np.zeros((LAYERS, EXPERTS, EXPERTS), dtype=np.int32)
    for domain in DOMAINS:
        values = routes[domain][split]
        for token in range(len(values) - 1):
            for layer in range(LAYERS):
                for current in values[token, layer]:
                    counts[layer, current, values[token + 1, layer]] += 1
    ids = np.arange(EXPERTS)
    selected = np.empty((LAYERS, EXPERTS), dtype=np.int16)
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            selected[layer, expert] = np.lexsort((ids, -counts[layer, expert]))[0]
    return selected, hashlib.sha256(counts.tobytes()).hexdigest()


def aggregate(results):
    return {
        key: np.concatenate([value[key] for value in results])
        for key in ("request_misses", "copies", "prefetch_copies")
    }


def waterfill(routes, fixed_by_domain):
    base = [4] * LAYERS
    remaining = 680 - sum(base)
    curves = np.zeros((LAYERS, 33), dtype=np.float64)
    for layer in range(LAYERS):
        for capacity in range(4, 33):
            misses = 0
            for domain in DOMAINS:
                partial = simulate_sequence(
                    routes[domain][VAL, layer:layer + 1],
                    [fixed_by_domain[domain][layer]], [capacity], policy="lru",
                )
                misses += int(partial["request_misses"].sum())
            curves[layer, capacity] = misses
    capacities = base[:]
    while remaining:
        candidates = []
        for layer in range(LAYERS):
            current = capacities[layer]
            benefit = -math.inf if current >= 32 else curves[layer, current] - curves[layer, current + 1]
            candidates.append((benefit, -layer))
        _, neg_layer = max(candidates)
        capacities[-neg_layer] += 1
        remaining -= 1
    return capacities, curves


def auto_profile(sequence, labels, global12, profile8, likelihoods, oracle=False):
    capacities = [dynamic_slots(layer) for layer in range(LAYERS)]
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    active = None
    window = deque(maxlen=16)
    misses = np.zeros(len(sequence), dtype=np.int64)
    copies = np.zeros(len(sequence), dtype=np.int64)
    profile_copies = np.zeros(len(sequence), dtype=np.int64)
    for token, route in enumerate(sequence):
        window.append(route)
        if oracle:
            chosen = labels[token]
        else:
            scores = {}
            observed = np.stack(tuple(window))
            for domain in DOMAINS:
                score = 0.0
                for layer in range(LAYERS):
                    score += float(np.log(likelihoods[domain][layer, observed[:, layer]].clip(1e-12)).sum())
                scores[domain] = score
            chosen = max(DOMAINS, key=lambda domain: (scores[domain], -DOMAINS.index(domain)))
        if chosen != active:
            for layer in range(LAYERS):
                old = set() if active is None else set(profile8[active][layer])
                new = set(profile8[chosen][layer])
                changed = len(new - old)
                copies[token] += changed; profile_copies[token] += changed
            active = chosen
        fixed = [set(global12[layer]) | set(profile8[active][layer]) for layer in range(LAYERS)]
        for layer in range(LAYERS):
            cache = dynamic[layer]
            for raw in route[layer]:
                expert = int(raw)
                if expert in fixed[layer]:
                    continue
                if expert in cache:
                    cache.move_to_end(expert)
                else:
                    misses[token] += 1; copies[token] += 1
                    if len(cache) >= capacities[layer]: cache.popitem(last=False)
                    cache[expert] = None
    return {"request_misses": misses, "copies": copies, "prefetch_copies": profile_copies}


def main():
    routes, route_hashes = load_routes()
    domain_counts, global_counts = frequency_tables(routes, VAL)
    fixed_by_domain = {domain: [set(x) for x in top_sets(domain_counts[domain], 20)] for domain in DOMAINS}
    universal20 = [set(x) for x in top_sets(global_counts, 20)]
    global12 = [set(x) for x in top_sets(global_counts, 12)]
    ids = np.arange(EXPERTS)
    profile8 = {}
    for domain in DOMAINS:
        profile8[domain] = []
        for layer in range(LAYERS):
            ranking = np.lexsort((ids, -domain_counts[domain][layer]))
            selected = [int(expert) for expert in ranking if int(expert) not in global12[layer]][:8]
            profile8[domain].append(set(selected))
    likelihoods = {domain: (domain_counts[domain] + 1) / (domain_counts[domain].sum(axis=1, keepdims=True) + EXPERTS) for domain in DOMAINS}
    pairs, pair_hash = pair_table(routes, VAL)
    capacities_base = [dynamic_slots(layer) for layer in range(LAYERS)]
    water_capacities, curves = waterfill(routes, fixed_by_domain)

    validation_raw = {}
    for policy in ("lru", "lfu", "tinylfu", "2q", "pair"):
        per_domain = []
        for domain in DOMAINS:
            per_domain.append(simulate_sequence(routes[domain][VAL], fixed_by_domain[domain], capacities_base, policy, pairs))
        validation_raw[policy] = aggregate(per_domain)
    validation_raw["waterfill"] = aggregate([
        simulate_sequence(routes[domain][VAL], fixed_by_domain[domain], water_capacities, "lru")
        for domain in DOMAINS
    ])
    validation = {policy: summarize(value) for policy, value in validation_raw.items()}
    baseline = validation["lru"]
    eligible = []
    for policy, value in validation.items():
        if policy == "lru": continue
        mean_ratio = value["request_misses"]["mean"] / baseline["request_misses"]["mean"]
        p95_ratio = value["request_misses"]["p95"] / baseline["request_misses"]["p95"]
        copy_ratio = value["copies"]["sum"] / baseline["copies"]["sum"]
        value["ratios_to_lru"] = {"mean_miss": mean_ratio, "p95_miss": p95_ratio, "copies": copy_ratio}
        value["validation_eligible"] = bool(mean_ratio <= 0.95 and p95_ratio <= 0.95 and copy_ratio <= 1.0)
        if value["validation_eligible"]: eligible.append(policy)
    selected = None if not eligible else min(eligible, key=lambda policy: (
        validation[policy]["request_misses"]["p95"], validation[policy]["request_misses"]["mean"], policy))

    test = None
    if selected:
        test_raw = {}
        for policy in ("lru", selected):
            caps = water_capacities if policy == "waterfill" else capacities_base
            base_policy = "lru" if policy == "waterfill" else policy
            test_raw[policy] = aggregate([
                simulate_sequence(routes[domain][TEST], fixed_by_domain[domain], caps, base_policy, pairs)
                for domain in DOMAINS
            ])
        test = {policy: summarize(value) for policy, value in test_raw.items()}
        selected_value, test_baseline = test[selected], test["lru"]
        ratios = {
            "mean_miss": selected_value["request_misses"]["mean"] / test_baseline["request_misses"]["mean"],
            "p95_miss": selected_value["request_misses"]["p95"] / test_baseline["request_misses"]["p95"],
            "copies": selected_value["copies"]["sum"] / test_baseline["copies"]["sum"],
        }
        selected_value["ratios_to_lru"] = ratios
        selected_value["test_pass"] = bool(ratios["mean_miss"] <= 0.97 and ratios["p95_miss"] <= 0.97 and ratios["copies"] <= 1.0)

    sequences = {
        **{f"pure_{domain}": (routes[domain][TEST], [domain] * 512) for domain in DOMAINS},
        "mixed_64": concatenate(routes, TEST, [(domain, 64) for _ in range(2) for domain in DOMAINS]),
        "switch_128": concatenate(routes, TEST, [(domain, 128) for domain in DOMAINS[:4]]),
    }
    domain_results = {}
    for name, (sequence, labels) in sequences.items():
        universal = simulate_sequence(sequence, universal20, capacities_base, "lru")
        oracle = auto_profile(sequence, labels, global12, profile8, likelihoods, oracle=True)
        automatic = auto_profile(sequence, labels, global12, profile8, likelihoods, oracle=False)
        domain_results[name] = {"universal": summarize(universal), "oracle": summarize(oracle), "automatic": summarize(automatic)}
        for policy in ("oracle", "automatic"):
            domain_results[name][policy]["classifier_or_oracle_accuracy"] = float(np.nan)
    aggregate_domain = {}
    for policy in ("universal", "oracle", "automatic"):
        copies = sum(result[policy]["copies"]["sum"] for result in domain_results.values())
        misses = sum(result[policy]["request_misses"]["sum"] for result in domain_results.values())
        aggregate_domain[policy] = {"copies": copies, "request_misses": misses, "h2d_bytes": copies * EXPERT_BYTES}
    auto_oracle_ratio = aggregate_domain["automatic"]["copies"] / aggregate_domain["oracle"]["copies"]
    auto_universal_ratio = aggregate_domain["automatic"]["copies"] / aggregate_domain["universal"]["copies"]
    switch = domain_results["switch_128"]
    switch_auto_oracle = switch["automatic"]["copies"]["sum"] / switch["oracle"]["copies"]["sum"]
    switch_auto_universal = switch["automatic"]["copies"]["sum"] / switch["universal"]["copies"]["sum"]
    domain_pass = bool(auto_oracle_ratio <= 1.05 and auto_universal_ratio <= 0.95 and switch_auto_oracle <= 1.05 and switch_auto_universal <= 0.95)

    output = {
        "kind": "streamq5_moe_p10_cache_family", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "route_hashes": route_hashes, "pair_count_sha256": pair_hash,
        "budget": {"static_per_layer": 20, "dynamic_total": sum(capacities_base), "slots_total": 48 * 20 + sum(capacities_base), "expert_bytes": EXPERT_BYTES},
        "waterfill_capacities": water_capacities,
        "validation": validation, "selected": selected, "test": test,
        "domain_conditioning": {
            "sequences": domain_results, "aggregate": aggregate_domain,
            "ratios": {"automatic_to_oracle": auto_oracle_ratio, "automatic_to_universal": auto_universal_ratio,
                       "switch_automatic_to_oracle": switch_auto_oracle, "switch_automatic_to_universal": switch_auto_universal},
            "pass": domain_pass,
        },
        "cache_family_pass": bool(selected and test and test[selected]["test_pass"]),
        "overall_pass": bool(selected and test and test[selected]["test_pass"] and domain_pass),
        "claim_boundary": "Causal simulation on real routes with exact record-copy accounting; selected wins require physical H2D validation before runtime integration.",
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "validation_ratios": {p: v.get("ratios_to_lru") for p, v in validation.items() if p != "lru"},
        "test_selected": None if not selected else test[selected],
        "domain_ratios": output["domain_conditioning"]["ratios"],
        "cache_family_pass": output["cache_family_pass"], "domain_pass": domain_pass,
        "overall_pass": output["overall_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
