from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
TOKENS, CONTEXT_TOKENS = 32_768, 1_024
CAPACITY, EXPERT_MIB = 4_280, 9
PARAMETERS_PER_EXPERT = 4_718_592
NONEXPERT_PARAMETERS = 1_541_093_376
RATE_BPP = 1.930708991156684

PREREG = ROOT / "reports/ldhera_moe/P0A_LAYER_CACHE_PREREGISTRATION.md"
DOMAIN_LOCK = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"
ALLOCATION_LOCK = ROOT / "reports/ldhera_moe/p0a_layer_allocation_lock.json"
ROUTE_CAPTURE = ROOT / "reports/dhera_moe/p0_route_capture.json"
RESULT = ROOT / "reports/ldhera_moe/p0a_layer_cache_result.json"
REPORT = ROOT / "reports/ldhera_moe/P0A_LAYER_CACHE_REPORT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest(values: np.ndarray, probability: float) -> float:
    return float(np.sort(values)[math.ceil(probability * len(values)) - 1])


def simulate(
    routes: np.ndarray,
    base_rows: list[dict[str, object]],
    allocation: list[int],
    switch_mib: float,
) -> dict[str, object]:
    if len(allocation) != LAYERS or sum(allocation) != 56:
        raise ValueError("invalid layer allocation")
    base = np.zeros((LAYERS, EXPERTS), dtype=np.bool_)
    for row in base_rows:
        base[row["layer"], row["expert"]] = True
    layer_axis = np.arange(LAYERS)[None, :, None]
    is_base = base[layer_axis, routes]
    cold = (~is_base).reshape(TOKENS, LAYERS * TOP_K)
    flat = routes.reshape(TOKENS, LAYERS * TOP_K)
    misses = np.zeros(TOKENS, dtype=np.int16)
    misses_by_layer = np.zeros(LAYERS, dtype=np.int64)
    hits_by_layer = np.zeros(LAYERS, dtype=np.int64)
    for start in range(0, TOKENS, CONTEXT_TOKENS):
        caches = [OrderedDict() for _ in range(LAYERS)]
        for token in range(start, start + CONTEXT_TOKENS):
            for flat_index in np.flatnonzero(cold[token]):
                index = int(flat_index)
                layer = index // TOP_K
                expert = int(flat[token, index])
                cache = caches[layer]
                if expert in cache:
                    cache.move_to_end(expert)
                    hits_by_layer[layer] += 1
                    continue
                misses[token] += 1
                misses_by_layer[layer] += 1
                if allocation[layer] > 0:
                    cache[expert] = None
                    if len(cache) > allocation[layer]:
                        cache.popitem(last=False)
    base_calls = int(is_base.sum(dtype=np.int64))
    total_calls = TOKENS * LAYERS * TOP_K
    cold_calls = total_calls - base_calls
    hit_count, miss_count = int(hits_by_layer.sum()), int(misses_by_layer.sum())
    if hit_count + miss_count != cold_calls or miss_count != int(misses.sum()):
        raise AssertionError("event conservation failed")
    traffic = misses.astype(np.float64) * EXPERT_MIB
    traffic[::CONTEXT_TOKENS] += switch_mib
    return {
        "base_invocations": base_calls,
        "base_invocation_fraction": base_calls / total_calls,
        "cold_invocations": cold_calls,
        "layer_local_hits": hit_count,
        "misses": miss_count,
        "cold_hit_fraction": hit_count / cold_calls if cold_calls else 1.0,
        "traffic_mib_per_token": {
            "mean": float(traffic.mean()),
            "p50": nearest(traffic, 0.50),
            "p95": nearest(traffic, 0.95),
            "p99": nearest(traffic, 0.99),
            "maximum": float(traffic.max()),
        },
        "allocation": allocation,
        "misses_by_layer": misses_by_layer.tolist(),
        "hits_by_layer": hits_by_layer.tolist(),
        "base_switches": TOKENS // CONTEXT_TOKENS,
        "base_switch_mib_each": switch_mib,
    }


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite LDHERA result")
    domain_lock = json.loads(DOMAIN_LOCK.read_text(encoding="utf-8"))
    allocation_lock = json.loads(ALLOCATION_LOCK.read_text(encoding="utf-8"))
    capture = json.loads(ROUTE_CAPTURE.read_text(encoding="utf-8"))
    if allocation_lock["preregistration_sha256"] != sha256(PREREG):
        raise ValueError("preregistration hash mismatch")
    if allocation_lock["domain_base_lock_sha256"] != sha256(DOMAIN_LOCK):
        raise ValueError("domain lock hash mismatch")

    entropy_gib = CAPACITY * PARAMETERS_PER_EXPERT * RATE_BPP / 8 / 2**30
    trunk_gib = NONEXPERT_PARAMETERS * 4 / 8 / 2**30
    cache_gib = 56 * EXPERT_MIB / 1024
    resident_gib = entropy_gib + trunk_gib + cache_gib
    cold_gib = (LAYERS * EXPERTS - CAPACITY) * EXPERT_MIB / 1024
    switch_mib = entropy_gib * 1024

    routes = {domain: [] for domain in DOMAINS}
    integrity = True
    for layer in range(LAYERS):
        item = capture["artifacts"][str(layer)]
        path = ROOT / item["artifact"]
        integrity &= sha256(path) == item["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain].append(tensors[f"{domain}_router_ids"].numpy())
    domains, gates = {}, {}
    for domain in DOMAINS:
        domains[domain] = simulate(
            np.stack(routes[domain], axis=1),
            domain_lock["bases"][domain],
            allocation_lock["allocations"][domain],
            switch_mib,
        )
        traffic = domains[domain]["traffic_mib_per_token"]
        gates[domain] = {
            "mean_le_64": traffic["mean"] <= 64.0,
            "p95_le_144": traffic["p95"] <= 144.0,
            "p99_le_288": traffic["p99"] <= 288.0,
        }
        gates[domain]["all_traffic_gates"] = all(gates[domain].values())
        print(json.dumps({"domain": domain, "traffic": traffic, "gate": gates[domain]}), flush=True)
    memory_gate = resident_gib <= 5.75 and cold_gib <= 24.0
    all_traffic = all(row["all_traffic_gates"] for row in gates.values())
    positive = bool(memory_gate and all_traffic and integrity)
    verdict = (
        "p0a_exploratory_positive_pending_verification"
        if positive
        else "p0a_exploratory_negative_pending_verification"
    )
    payload = {
        "kind": "ldhera_moe_p0a_layer_cache_result",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "exploratory_opened_routes": True,
        "p0b_authorized": positive,
        "p1_authorized": False,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "domain_base_lock_sha256": sha256(DOMAIN_LOCK),
            "allocation_lock_sha256": sha256(ALLOCATION_LOCK),
            "route_capture_sha256": sha256(ROUTE_CAPTURE),
            "route_artifact_hashes_valid": integrity,
        },
        "policy": {
            "total_layer_local_slots": 56,
            "allocation_objective": "minimum HERA-training LRU misses",
            "context_tokens": CONTEXT_TOKENS,
            "full_base_switches_per_context": 1,
            "base_switch_mib_each": switch_mib,
        },
        "memory_projection": {
            "resident_weight_gib": resident_gib,
            "active_cold_bf16_host_gib": cold_gib,
            "memory_gate_pass": memory_gate,
        },
        "domains": domains,
        "gates": {
            "memory": memory_gate,
            "traffic_by_domain": gates,
            "all_traffic": all_traffic,
            "independent_verification": "pending",
        },
        "claim_boundary": (
            "Exploration on opened routes. Positive permits only fresh blind P0B; "
            "no pack, quality, latency, classifier, or speed claim."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for domain in DOMAINS:
        row = domains[domain]
        traffic = row["traffic_mib_per_token"]
        rows.append(
            f"| {domain} | {row['cold_hit_fraction']:.3%} | {traffic['mean']:.3f} | "
            f"{traffic['p95']:.0f} | {traffic['p99']:.0f} | "
            f"{'PASS' if gates[domain]['all_traffic_gates'] else 'FAIL'} |"
        )
    REPORT.write_text(
        "\n".join(
            [
                "# LDHERA-MoE P0A — training-geleerde laagcache",
                "",
                f"Voorlopige exploratieve uitkomst: **{verdict}**.",
                "",
                "| Domein | Cold hitrate | Gem. MiB/token | p95 | p99 | Gate |",
                "|---|---:|---:|---:|---:|:---:|",
                *rows,
                "",
                "Alle allocaties zijn vóór validation uit trainingroutes gelockt; "
                "één volledige basewissel per context is meegerekend.",
                "P0A gebruikt geopende routes en kan alleen verse P0B openen.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "p0b_authorized": positive}, indent=2))
