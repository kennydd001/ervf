from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

R = ROOT / "reports/streamq5_moe"
ROUTES = ROOT / "reports/runs/streamq5_moe/p4d_routes"
PREREG = R / "N3B_VRAM_KV_CACHE_PARETO_PREREGISTRATION.md"
OUTPUT = R / "n3b_vram_kv_cache_pareto.json"
PHYSICAL = R / "p7c_ervf_end_to_end_smoke.json"
CAPTURE = R / "p4d_route_capture_result.json"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
EXPERT_BYTES = 3_035_136
RESERVE_BYTES = 402_653_184
CONTEXTS = (4096, 8192, 16384, 32768, 65536)
PARTITIONS = {"calibration": (0, 512), "validation": (512, 768), "test": (768, 1024)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kv_bytes(context: int) -> int:
    return LAYERS * 2 * 4 * context * 128 * 2


def load_routes():
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    result = {domain: np.empty((1024, LAYERS, TOP_K), dtype=np.int16) for domain in DOMAINS}
    hashes = {}
    for layer in range(LAYERS):
        path = ROUTES / f"layer_{layer:02d}.safetensors"
        hashes[str(layer)] = sha256(path)
        if hashes[str(layer)] != capture["manifests"][str(layer)]["artifact_sha256"]:
            raise RuntimeError("route hash mismatch")
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                result[domain][:, layer] = handle.get_tensor(f"{domain}_router_ids").astype(np.int16)
    return result, hashes


def fixed_sets(routes, fixed_count: int):
    begin, end = PARTITIONS["calibration"]
    result = {domain: [] for domain in DOMAINS}
    for domain in DOMAINS:
        for layer in range(LAYERS):
            counts = np.bincount(routes[domain][begin:end, layer].reshape(-1), minlength=EXPERTS)
            order = np.lexsort((np.arange(EXPERTS), -counts))
            result[domain].append(frozenset(int(x) for x in order[:fixed_count]))
    return result


def simulate(routes, capacities, fixed_count: int, partition: str):
    fixed = fixed_sets(routes, fixed_count)
    begin, end = PARTITIONS[partition]
    misses = 0; accesses = 0; token_misses = []
    for domain in DOMAINS:
        lrus = [OrderedDict() for _ in range(LAYERS)]
        for token in range(begin, end):
            token_total = 0
            for layer in range(LAYERS):
                static = fixed[domain][layer]
                dynamic_capacity = max(0, capacities[layer] - len(static))
                lru = lrus[layer]
                for raw in routes[domain][token, layer]:
                    expert = int(raw); accesses += 1
                    if expert in static:
                        continue
                    if expert in lru:
                        lru.move_to_end(expert)
                        continue
                    misses += 1; token_total += 1
                    if dynamic_capacity:
                        if len(lru) >= dynamic_capacity:
                            lru.popitem(last=False)
                        lru[expert] = None
            token_misses.append(token_total)
    values = np.asarray(token_misses, dtype=np.float64)
    return {"misses": misses, "accesses": accesses, "miss_ratio": misses / accesses,
            "token_misses_mean": float(values.mean()), "token_misses_p50": float(np.percentile(values, 50)),
            "token_misses_p95": float(np.percentile(values, 95))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    physical = json.loads(PHYSICAL.read_text(encoding="utf-8"))["physical"]
    free_before = int(physical["free_before_bytes"])
    trunk = int(physical["trunk_device_bytes"])
    routes, hashes = load_routes()
    rows = {}
    for context in CONTEXTS:
        kv = kv_bytes(context)
        cache_budget = free_before - trunk - kv - RESERVE_BYTES
        total_slots = max(0, min(LAYERS * EXPERTS, cache_budget // EXPERT_BYTES))
        capacities = [total_slots // LAYERS + (layer < total_slots % LAYERS) for layer in range(LAYERS)]
        row = {"context": context, "kv_bytes": kv, "cache_budget_bytes": cache_budget,
               "total_slots": int(total_slots), "layer_capacity_min": int(min(capacities)),
               "layer_capacity_max": int(max(capacities)), "minimum_top8_compatible": min(capacities) >= TOP_K,
               "current_static20_compatible": min(capacities) >= 20,
               "allocation_slack_bytes": int(cache_budget - total_slots * EXPERT_BYTES)}
        if min(capacities) >= TOP_K:
            candidates = []
            for fixed_count in range(0, min(capacities) + 1):
                metric = simulate(routes, capacities, fixed_count, "validation")
                candidates.append({"fixed_count": fixed_count, **metric})
            best = min(candidates, key=lambda x: (x["miss_ratio"], x["fixed_count"]))
            test = simulate(routes, capacities, best["fixed_count"], "test")
            row["validation_selection"] = best
            row["test"] = test
            row["candidate_count"] = len(candidates)
        rows[str(context)] = row
        print(json.dumps({"context": context, "slots": total_slots, "min_per_layer": min(capacities),
                          "best_fixed": row.get("validation_selection", {}).get("fixed_count"),
                          "test_miss_ratio": row.get("test", {}).get("miss_ratio")}), flush=True)
    gates = {"allocation_exact": all(row["allocation_slack_bytes"] < EXPERT_BYTES for row in rows.values()),
             "4k_minimum_top8": rows["4096"]["minimum_top8_compatible"],
             "8k_minimum_top8": rows["8192"]["minimum_top8_compatible"],
             "32k_static20_compatible": rows["32768"]["current_static20_compatible"]}
    result = {"kind": "streamq5_moe_n3b_vram_kv_cache_pareto", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"preregistration_sha256": sha256(PREREG), "physical_sha256": sha256(PHYSICAL),
                         "route_capture_sha256": sha256(CAPTURE), "route_hashes": hashes,
                         "free_before_bytes": free_before, "trunk_device_bytes": trunk,
                         "reserve_bytes": RESERVE_BYTES, "expert_bytes": EXPERT_BYTES},
              "partitions": PARTITIONS, "contexts": rows, "gates": gates,
              "overall_pass": gates["allocation_exact"] and gates["4k_minimum_top8"] and gates["8k_minimum_top8"],
              "claim_boundary": "Exact capacity arithmetic plus P4D held-out route simulation; no physical long-context decode, quality or throughput result."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
