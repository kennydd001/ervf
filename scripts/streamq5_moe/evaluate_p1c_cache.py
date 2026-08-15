from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


REPORT_DIR = ROOT / "reports/streamq5_moe"
PREREG = REPORT_DIR / "P1C_CORRECTED_ROUTE_CACHE_PREREGISTRATION.md"
INPUT_LOCK = REPORT_DIR / "p1c_route_input_lock.json"
CAPTURE = REPORT_DIR / "p1c_route_capture_result.json"
EVALUATOR_LOCK = REPORT_DIR / "p1c_cache_evaluator_lock.json"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p1c_routes"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
STATIC_SLOTS = 32
EXPERT_BYTES = 3_035_136
BANDWIDTH_GB_S = 26.158915272090432
TRUNK_GIB = 1.4352550506591797
KV_GIB, RESERVE_GIB, GPU_GIB = 0.375, 0.75, 7.9599609375
BANK_BYTES = 18_647_875_584


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dynamic_slots(layer: int) -> int:
    return 8 if layer <= 37 else 7


def load_routes() -> tuple[dict, dict]:
    routes = {domain: [] for domain in DOMAINS}
    hashes = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                value = handle.get_tensor(f"{domain}_router_ids").astype(np.int64)
                if value.shape != (1024, TOP_K) or value.min() < 0 or value.max() >= EXPERTS:
                    raise ValueError(f"invalid routes at layer {layer}, {domain}")
                routes[domain].append(value)
    return {domain: np.stack(values, axis=1) for domain, values in routes.items()}, hashes


def static_sets(routes: dict) -> tuple[dict, dict]:
    selected = {}
    counts_hashes = {}
    for domain in DOMAINS:
        selected[domain] = []
        packed_counts = []
        for layer in range(LAYERS):
            counts = np.bincount(routes[domain][:512, layer, :].reshape(-1), minlength=EXPERTS)
            order = np.lexsort((np.arange(EXPERTS), -counts))
            chosen = tuple(int(value) for value in order[:STATIC_SLOTS])
            selected[domain].append(frozenset(chosen))
            packed_counts.append(counts.astype(np.int64))
        counts_hashes[domain] = hashlib.sha256(np.stack(packed_counts).tobytes()).hexdigest()
    return selected, counts_hashes


def evaluate_domain(route: np.ndarray, static: list[frozenset], begin: int, end: int) -> dict:
    dynamic = [OrderedDict() for _ in range(LAYERS)]
    misses = np.zeros(end - begin, dtype=np.int16)
    static_hits = np.zeros(end - begin, dtype=np.int16)
    dynamic_hits = np.zeros(end - begin, dtype=np.int16)
    for local, token in enumerate(range(begin, end)):
        for layer in range(LAYERS):
            fixed = static[layer]
            lru = dynamic[layer]
            for expert_value in route[token, layer]:
                expert = int(expert_value)
                if expert in fixed:
                    static_hits[local] += 1
                elif expert in lru:
                    dynamic_hits[local] += 1
                    lru.move_to_end(expert)
                else:
                    misses[local] += 1
                    lru[expert] = None
                    if len(lru) > dynamic_slots(layer):
                        lru.popitem(last=False)
            if len(lru) > dynamic_slots(layer) or any(expert in fixed for expert in lru):
                raise RuntimeError("cache invariant failed")
    bytes_per_token = misses.astype(np.int64) * EXPERT_BYTES
    ms = bytes_per_token / (BANDWIDTH_GB_S * 1e9) * 1000
    total_calls = (end - begin) * LAYERS * TOP_K
    return {
        "tokens": end - begin,
        "mean_misses": float(misses.mean()), "p95_misses": float(np.percentile(misses, 95)), "p99_misses": float(np.percentile(misses, 99)), "max_misses": int(misses.max()),
        "mean_h2d_bytes": float(bytes_per_token.mean()), "p95_h2d_bytes": float(np.percentile(bytes_per_token, 95)), "p99_h2d_bytes": float(np.percentile(bytes_per_token, 99)),
        "mean_h2d_ms": float(ms.mean()), "p95_h2d_ms": float(np.percentile(ms, 95)), "p99_h2d_ms": float(np.percentile(ms, 99)), "max_h2d_ms": float(ms.max()),
        "static_hit_rate": float(static_hits.sum() / total_calls), "dynamic_hit_rate": float(dynamic_hits.sum() / total_calls), "miss_rate": float(misses.sum() / total_calls),
        "misses": misses.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = REPORT_DIR / f"p1c_cache_{args.split}.json"
    report = REPORT_DIR / f"P1C_CACHE_{args.split.upper()}.md"
    if output.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite P1C cache {args.split}")
    validation_path = REPORT_DIR / "p1c_cache_validation.json"
    if args.split == "test" and not validation_path.exists():
        raise RuntimeError("validation must precede test")
    evaluator_lock = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != evaluator_lock["evaluator_sha256"] or sha256(INPUT_LOCK) != evaluator_lock["input_lock_sha256"] or sha256(CAPTURE) != evaluator_lock["route_capture_sha256"]:
        raise ValueError("P1C cache evaluator lock mismatch")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    if sha256(PREREG) != lock["preregistration_sha256"] or capture.get("status") != "route_capture_complete":
        raise ValueError("P1C provenance or route capture failure")

    routes, route_hashes = load_routes()
    static, calibration_count_hashes = static_sets(routes)
    begin, end = lock["partitions"][args.split]
    per_domain = {domain: evaluate_domain(routes[domain], static[domain], begin, end) for domain in DOMAINS}
    all_misses = np.concatenate([np.asarray(per_domain[domain]["misses"], dtype=np.int64) for domain in DOMAINS])
    all_bytes = all_misses * EXPERT_BYTES
    all_ms = all_bytes / (BANDWIDTH_GB_S * 1e9) * 1000
    cache_slots = sum(STATIC_SLOTS + dynamic_slots(layer) for layer in range(LAYERS))
    cache_bytes = cache_slots * EXPERT_BYTES
    resident_gib = TRUNK_GIB + KV_GIB + RESERVE_GIB + cache_bytes / 2**30
    preload_bytes = STATIC_SLOTS * LAYERS * EXPERT_BYTES
    preload_ms = preload_bytes / (BANDWIDTH_GB_S * 1e9) * 1000
    aggregate = {
        "tokens": int(all_misses.size), "mean_misses": float(all_misses.mean()), "p95_misses": float(np.percentile(all_misses, 95)), "p99_misses": float(np.percentile(all_misses, 99)), "max_misses": int(all_misses.max()),
        "mean_h2d_bytes": float(all_bytes.mean()), "p95_h2d_bytes": float(np.percentile(all_bytes, 95)), "p99_h2d_bytes": float(np.percentile(all_bytes, 99)),
        "mean_h2d_ms": float(all_ms.mean()), "p95_h2d_ms": float(np.percentile(all_ms, 95)), "p99_h2d_ms": float(np.percentile(all_ms, 99)), "max_h2d_ms": float(all_ms.max()),
    }
    gates = {
        "mean_h2d_ms_le_25": aggregate["mean_h2d_ms"] <= 25.0,
        "p95_h2d_ms_le_35": aggregate["p95_h2d_ms"] <= 35.0,
        "all_domain_mean_h2d_ms_le_25": all(row["mean_h2d_ms"] <= 25.0 for row in per_domain.values()),
        "all_domain_p95_h2d_ms_le_35": all(row["p95_h2d_ms"] <= 35.0 for row in per_domain.values()),
        "static_preload_ms_le_250": preload_ms <= 250.0,
        "resident_gib_le_gpu": resident_gib <= GPU_GIB,
        "host_bank_gib_le_17_45": BANK_BYTES / 2**30 <= 17.45,
        "host_bank_gib_le_24": BANK_BYTES / 2**30 <= 24.0,
    }
    all_pass = all(gates.values())
    if args.split == "validation":
        status = "p1c_validation_pass_test_authorized" if all_pass else "p1c_validation_closed"
        p1c_pass = False
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "p1c_validation_pass_test_authorized":
            raise RuntimeError("test not authorized by validation")
        status = "p1c_cache_pass" if all_pass and all(validation["gates"].values()) else "p1c_cache_closed"
        p1c_pass = status == "p1c_cache_pass"
    payload = {
        "kind": "streamq5_moe_p1c_cache_evaluation", "completed_utc": datetime.now(timezone.utc).isoformat(), "split": args.split, "status": status,
        "inputs": {"preregistration_sha256": sha256(PREREG), "input_lock_sha256": sha256(INPUT_LOCK), "route_capture_sha256": sha256(CAPTURE), "evaluator_lock_sha256": sha256(EVALUATOR_LOCK), "evaluator_sha256": sha256(Path(__file__)), "route_artifact_sha256": route_hashes},
        "policy": {"static_slots_per_layer": STATIC_SLOTS, "dynamic_slots_layers_0_37": 8, "dynamic_slots_layers_38_47": 7, "total_slots": cache_slots, "calibration_tokens": [0, 512], "split_tokens": [begin, end], "calibration_count_sha256": calibration_count_hashes},
        "physical_accounting": {"matrix_record_bytes": 1_011_712, "expert_record_bytes": EXPERT_BYTES, "host_bank_bytes": BANK_BYTES, "host_bank_gib": BANK_BYTES / 2**30, "cache_bytes": cache_bytes, "cache_gib": cache_bytes / 2**30, "resident_gib": resident_gib, "gpu_gib": GPU_GIB, "static_preload_bytes": preload_bytes, "static_preload_ms": preload_ms, "bandwidth_gb_s": BANDWIDTH_GB_S},
        "per_domain": per_domain, "aggregate": aggregate, "gates": gates, "p1c_pass": p1c_pass, "next_phase_authorized": p1c_pass,
        "claim_boundary": "Route-cache simulation and exact byte accounting only; no physical Q5 bank, transfer timing, kernel, overlap, or end-to-end wall-clock.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(
        f"# STREAMQ5-MoE P1C - {args.split} corrected-semantics cache\n\nUitkomst: **{status}**.\n\n"
        f"Gemiddelde/p95 geprojecteerde dynamische H2D: {aggregate['mean_h2d_ms']:.3f}/{aggregate['p95_h2d_ms']:.3f} ms/token. "
        f"Statische preload: {preload_ms:.3f} ms. Resident: {resident_gib:.6f} GiB.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "aggregate": aggregate, "resident_gib": resident_gib, "preload_ms": preload_ms, "gates": gates}, indent=2))
