from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from datetime import datetime

import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P2C_STATIC_DYNAMIC_REBALANCE_PREREGISTRATION.md"
SELECTION_PATH = R / "p2c_policy_selection.json"
LOCK_PATH = R / "p2b_route_input_lock.json"
CAPTURE_PATH = R / "p2b_route_capture_result.json"
EVALUATOR_LOCK = R / "p2c_h2d_evaluator_lock.json"
EVALUATOR = ROOT / "scripts/streamq5_moe/run_p2c_physical_h2d.py"
BANK_RESULT_PATH = R / "p1d_physical_bank_result.json"
VALIDATION_PATH = R / "p2c_physical_h2d_validation.json"
TEST_PATH = R / "p2c_physical_h2d_test.json"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p2b_routes"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
EXPERT_BYTES = 3_035_136


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b, tolerance=1e-10):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def slots(layer):
    return 35 if layer <= 7 else 34


def simulate(route, static_count, begin, end):
    fixed = []
    for layer in range(48):
        counts = np.bincount(route[:512, layer, :].reshape(-1), minlength=128)
        order = np.lexsort((np.arange(128), -counts))
        fixed.append(frozenset(int(value) for value in order[:static_count]))
    dynamic = [OrderedDict() for _ in range(48)]
    misses = np.zeros(end - begin, dtype=np.int64)
    for local, token in enumerate(range(begin, end)):
        for layer in range(48):
            lru = dynamic[layer]
            for raw in route[token, layer]:
                expert = int(raw)
                if expert in fixed[layer]:
                    continue
                if expert in lru:
                    lru.move_to_end(expert)
                else:
                    misses[local] += 1; lru[expert] = None
                    if len(lru) > slots(layer) - static_count:
                        lru.popitem(last=False)
    return misses


def stat(values, name):
    if name == "mean": return float(values.mean())
    if name == "max": return float(values.max())
    return float(np.percentile(values, int(name[1:])))


if __name__ == "__main__":
    checks = []
    add = lambda name, passed: checks.append({"name": name, "pass": bool(passed)})
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    evaluator_lock = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    bank = json.loads(BANK_RESULT_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    add("preregistration hash", sha256(PREREG) == selection["inputs"]["preregistration_sha256"])
    add("selection hash", sha256(SELECTION_PATH) == evaluator_lock["selection_sha256"])
    add("evaluator hash", sha256(EVALUATOR) == evaluator_lock["evaluator_sha256"])
    add("input lock hash", sha256(LOCK_PATH) == evaluator_lock["input_lock_sha256"])
    add("route capture hash", sha256(CAPTURE_PATH) == evaluator_lock["route_capture_sha256"])
    add("P2B test remained unopened", not (R / "p2b_physical_h2d_test.json").exists())
    chosen = min(selection["rows"], key=lambda row: (row["worst_domain_p95_misses"], row["aggregate_mean_misses"], -row["static_slots"]))
    add("selection rule", selection["selected"] == chosen and chosen["static_slots"] == 20)

    routes = {domain: [] for domain in DOMAINS}; route_hashes = {}
    for layer in range(48):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"; route_hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                routes[domain].append(handle.get_tensor(f"{domain}_router_ids").astype(np.int64))
    routes = {domain: np.stack(values, axis=1) for domain, values in routes.items()}

    recomputed_aggregate = {}
    for split, begin, end, result in (("validation", 512, 768, validation), ("test", 768, 1024, test)):
        all_misses, all_wall, all_event = [], [], []
        per_domain_exact = True
        for domain in DOMAINS:
            expected = simulate(routes[domain], 20, begin, end)
            row = result["per_domain"][domain]
            observed = np.asarray(row["misses"], dtype=np.int64)
            wall = np.asarray(row["wall_ms"], dtype=np.float64)
            event = np.asarray(row["event_ms"], dtype=np.float64)
            per_domain_exact &= np.array_equal(expected, observed) and row["simulation_exact"] is True
            per_domain_exact &= row["sample_integrity_failures"] == 0 and len(wall) == len(event) == 256
            per_domain_exact &= all(close(stat(wall, key), row["wall_ms_stats"][key]) for key in ("mean", "p50", "p95", "p99", "max"))
            per_domain_exact &= all(close(stat(event, key), row["event_ms_stats"][key]) for key in ("mean", "p50", "p95", "p99", "max"))
            per_domain_exact &= all(int(value) == int(miss) * EXPERT_BYTES for value, miss in zip(row["h2d_bytes"], observed))
            all_misses.append(observed); all_wall.append(wall); all_event.append(event)
        add(f"{split} per-domain samples and simulation", per_domain_exact)
        misses = np.concatenate(all_misses); wall = np.concatenate(all_wall); event = np.concatenate(all_event)
        aggregate = result["aggregate"]
        aggregate_exact = aggregate["tokens"] == 1280 and all(close(stat(wall, key), aggregate["wall_ms"][key]) for key in ("mean", "p50", "p95", "p99", "max")) and all(close(stat(event, key), aggregate["event_ms"][key]) for key in ("mean", "p50", "p95", "p99", "max")) and close((misses * EXPERT_BYTES).mean(), aggregate["mean_h2d_bytes"]) and close(np.percentile(misses * EXPERT_BYTES, 95), aggregate["p95_h2d_bytes"])
        add(f"{split} aggregate arithmetic", aggregate_exact)
        expected_gates = {
            "full_bank_pinned_and_hash_exact": result["physical"]["pinned_bank_bytes"] == 18_647_875_584 and result["physical"]["pinned_layer_sha256"] == {str(layer): bank["manifests"][str(layer)]["artifact_sha256"] for layer in range(48)},
            "device_cache_trunk_kv_co_resident": result["physical"]["free_before_bytes"] >= 4_977_623_040 + 1_541_093_376 + 402_653_184 + 402_653_184 and result["physical"]["free_after_bytes"] >= 402_653_184,
            "exact_physical_cache_bytes": result["physical"]["cache_bytes"] == 1640 * EXPERT_BYTES,
            "aggregate_mean_wall_h2d_ms_le_25": wall.mean() <= 25,
            "aggregate_p95_wall_h2d_ms_le_35": np.percentile(wall, 95) <= 35,
            "all_domain_mean_wall_h2d_ms_le_25": all(result["per_domain"][domain]["wall_ms_stats"]["mean"] <= 25 for domain in DOMAINS),
            "all_domain_p95_wall_h2d_ms_le_35": all(result["per_domain"][domain]["wall_ms_stats"]["p95"] <= 35 for domain in DOMAINS),
            "all_domain_preload_wall_ms_le_250": all(result["per_domain"][domain]["preload_wall_ms"] <= 250 for domain in DOMAINS),
            "all_miss_simulations_exact": per_domain_exact,
            "all_sampled_transfers_exact": all(result["per_domain"][domain]["sample_integrity_failures"] == 0 for domain in DOMAINS),
        }
        add(f"{split} physical gates", result["gates"] == expected_gates and all(expected_gates.values()))
        add(f"{split} route hashes", result["inputs"]["route_artifact_sha256"] == route_hashes)
        add(f"{split} policy", result["policy"]["total_slots"] == 1640 and result["policy"]["static_slots_per_layer"] == 20 and result["policy"]["dynamic_slots_layers_0_7"] == 15 and result["policy"]["dynamic_slots_layers_8_47"] == 14)
        recomputed_aggregate[split] = {"mean_wall_ms": float(wall.mean()), "p95_wall_ms": float(np.percentile(wall, 95))}

    add("split decisions", validation["status"] == "p2c_validation_pass_test_authorized" and validation["p2c_pass"] is False and test["status"] == "p2c_physical_h2d_pass" and test["p2c_pass"] is True and test["next_phase_authorized"] is True)
    add("test chronology", datetime.fromisoformat(test["started_utc"]) > datetime.fromisoformat(validation["completed_utc"]))
    passed = sum(row["pass"] for row in checks); total = len(checks)
    status = "p2c_physical_h2d_verification_pass" if passed == total else "p2c_physical_h2d_verification_fail"
    payload = {"kind": "streamq5_moe_p2c_independent_physical_h2d_verification", "status": status, "checks_passed": passed, "checks_total": total, "checks": checks, "validation": recomputed_aggregate["validation"], "test": recomputed_aggregate["test"], "claim_boundary": test["claim_boundary"]}
    (R / "p2c_physical_h2d_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (R / "P2C_PHYSICAL_H2D_VERIFICATION.md").write_text(f"# STREAMQ5-MoE P2C - onafhankelijke verificatie\n\nUitkomst: **{status}** ({passed}/{total}). Test wall mean/p95: {payload['test']['mean_wall_ms']:.3f}/{payload['test']['p95_wall_ms']:.3f} ms/token.\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": f"{passed}/{total}", "validation": payload["validation"], "test": payload["test"]}, indent=2))
    if status.endswith("fail"):
        raise SystemExit(1)
