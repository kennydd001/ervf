from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict

import cupy as cp
import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P3A_INTEGRATED_EXPERT_DATAPLANE_PREREGISTRATION.md"
LOCK_PATH = R / "p3a_dataplane_input_lock.json"
EVALUATOR_LOCK = R / "p3a_benchmark_evaluator_lock.json"
EVALUATOR = ROOT / "scripts/streamq5_moe/run_p3a_integrated_expert.py"
SMOKE = R / "p3a_dataplane_smoke.json"
CAPTURE = R / "p3a_route_capture_result.json"
VALIDATION_PATH = R / "p3a_integrated_expert_validation.json"
TEST_PATH = R / "p3a_integrated_expert_test.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
BANK_DIR = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p3a_routes"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
EXPERT_BYTES, LAYER_BYTES = 3_035_136, 388_497_408


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b, tolerance=1e-10):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def dynamic(layer):
    return 15 if layer <= 7 else 14


def simulate(route, begin, end):
    fixed = []
    for layer in range(48):
        counts = np.bincount(route[:512, layer, :].reshape(-1), minlength=128)
        order = np.lexsort((np.arange(128), -counts))
        fixed.append(frozenset(int(value) for value in order[:20]))
    lru = [OrderedDict() for _ in range(48)]; misses = np.zeros(end - begin, dtype=np.int64)
    for local, token in enumerate(range(begin, end)):
        for layer in range(48):
            cache = lru[layer]
            for raw in route[token, layer]:
                expert = int(raw)
                if expert in fixed[layer]: continue
                if expert in cache: cache.move_to_end(expert)
                else:
                    misses[local] += 1; cache[expert] = None
                    if len(cache) > dynamic(layer): cache.popitem(last=False)
    return misses


def stats(values):
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def transfer_integrity(routes):
    device = cp.empty(EXPERT_BYTES, dtype=cp.uint8); stream = cp.cuda.Stream(); failures = 0; checked = 0; started = time.perf_counter()
    for layer in range(48):
        for token in (512, 768):
            expert = int(routes["general"][token, layer, 0]); memory = cp.cuda.alloc_pinned_memory(EXPERT_BYTES); host = np.frombuffer(memory, dtype=np.uint8, count=EXPERT_BYTES)
            with (BANK_DIR / f"layer_{layer:02d}.q5bin").open("rb") as handle:
                handle.seek(expert * EXPERT_BYTES)
                if handle.readinto(host) != EXPERT_BYTES: failures += 1; continue
            with stream: cp.cuda.runtime.memcpyAsync(device.data.ptr, memory.ptr, EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
            stream.synchronize(); observed = cp.asnumpy(device); failures += int(not np.array_equal(observed, host)); checked += 1
    return {"records_checked": checked, "failures": failures, "seconds": time.perf_counter() - started}


if __name__ == "__main__":
    checks = []; add = lambda name, passed: checks.append({"name": name, "pass": bool(passed)})
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8")); evaluator_lock = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8")); smoke = json.loads(SMOKE.read_text(encoding="utf-8")); capture = json.loads(CAPTURE.read_text(encoding="utf-8")); validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8")); test = json.loads(TEST_PATH.read_text(encoding="utf-8")); p1d = json.loads(P1D_VERIFY.read_text(encoding="utf-8"))
    add("preregistration hash", sha256(PREREG) == lock["preregistration_sha256"])
    add("evaluator hash", sha256(EVALUATOR) == evaluator_lock["evaluator_sha256"])
    add("input lock hash", sha256(LOCK_PATH) == evaluator_lock["input_lock_sha256"])
    add("smoke hash and status", sha256(SMOKE) == evaluator_lock["smoke_sha256"] and smoke["status"] == "smoke_pass")
    add("smoke stage correctness", set(smoke["errors"]) == {"gate", "up", "swiglu", "down", "reduced"} and all(row["finite"] and row["max_abs"] <= 0.02 and row["relative_l2"] <= 1e-4 for row in smoke["errors"].values()))
    add("fresh route capture", capture["status"] == "route_capture_complete" and all(capture["controls"].values()) and capture["scale_semantics"].startswith("codes selected with FP32"))
    add("P1D physical bank verification", p1d["status"] == "p1d_physical_bank_verification_pass" and p1d["counters"]["records"] == 18432 and p1d["counters"]["code_source_mismatches"] == p1d["counters"]["scale_source_mismatches"] == 0)
    source = EVALUATOR.read_text(encoding="utf-8")
    add("fused physical CUDA stages", all(name in source for name in ("q5_gate_up_8", "swiglu_8", "q5_down_8", "reduce_experts")) and "float* dequantized" not in source)

    routes = {domain: [] for domain in DOMAINS}; route_hashes = {}
    for layer in range(48):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"; route_hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS: routes[domain].append(handle.get_tensor(f"{domain}_router_ids").astype(np.int64))
    routes = {domain: np.stack(values, axis=1) for domain, values in routes.items()}

    for split, begin, end, result in (("validation", 512, 768, validation), ("test", 768, 1024, test)):
        all_times, all_misses = [], []; exact = True
        for domain in DOMAINS:
            expected = simulate(routes[domain], begin, end); row = result["per_domain"][domain]; observed = np.asarray(row["misses"], dtype=np.int64); times = np.asarray(row["wall_ms"], dtype=np.float64)
            exact &= np.array_equal(expected, observed) and len(times) == 256 and row["finite_outputs"] is True
            expected_stats = stats(times); exact &= all(close(expected_stats[key], row["wall_ms_stats"][key]) for key in expected_stats)
            all_times.append(times); all_misses.append(observed)
        add(f"{split} exact miss reconstruction", exact)
        times = np.concatenate(all_times); misses = np.concatenate(all_misses); aggregate = result["aggregate"]
        add(f"{split} aggregate timing arithmetic", all(close(stats(times)[key], aggregate["wall_ms"][key]) for key in stats(times)) and all(close(stats(misses.astype(np.float64))[key], aggregate["misses"][key]) for key in stats(misses.astype(np.float64))) and aggregate["tokens"] == 1280)
        expected_gates = {"full_bank_pinned": result["physical"]["pinned_bank_bytes"] == 18_647_875_584, "device_co_resident_and_scratch": result["physical"]["free_after_bytes"] >= 402_653_184, "aggregate_mean_le_60": times.mean() <= 60, "aggregate_p95_le_75": np.percentile(times, 95) <= 75, "all_domain_mean_le_60": all(result["per_domain"][domain]["wall_ms_stats"]["mean"] <= 60 for domain in DOMAINS), "all_domain_p95_le_75": all(result["per_domain"][domain]["wall_ms_stats"]["p95"] <= 75 for domain in DOMAINS), "all_outputs_finite": all(result["per_domain"][domain]["finite_outputs"] for domain in DOMAINS)}
        add(f"{split} gate recomputation", result["gates"] == expected_gates and all(expected_gates.values()))
        add(f"{split} route provenance", result["inputs"]["route_artifact_sha256"] == route_hashes)
        add(f"{split} physical accounting", result["physical"]["cache_bytes"] == 4_977_623_040 and result["physical"]["trunk_bytes"] == 1_541_093_376 and result["physical"]["kv_bytes"] == 402_653_184 and result["physical"]["free_before_bytes"] >= 7_385_120_768)
    add("split decisions", validation["status"] == "p3a_validation_pass_test_authorized" and validation["p3a_pass"] is False and test["status"] == "p3a_integrated_expert_pass" and test["p3a_pass"] is True)
    transfer = transfer_integrity(routes)
    add("96 physical H2D records byte-exact", transfer["records_checked"] == 96 and transfer["failures"] == 0)
    passed = sum(row["pass"] for row in checks); total = len(checks); status = "p3a_integrated_expert_verification_pass" if passed == total else "p3a_integrated_expert_verification_fail"
    payload = {"kind": "streamq5_moe_p3a_independent_integrated_expert_verification", "status": status, "checks_passed": passed, "checks_total": total, "checks": checks, "validation_mean_ms": validation["aggregate"]["wall_ms"]["mean"], "validation_p95_ms": validation["aggregate"]["wall_ms"]["p95"], "test_mean_ms": test["aggregate"]["wall_ms"]["mean"], "test_p95_ms": test["aggregate"]["wall_ms"]["p95"], "test_max_ms": test["aggregate"]["wall_ms"]["max"], "independent_transfer_integrity": transfer, "claim_boundary": test["claim_boundary"]}
    (R / "p3a_integrated_expert_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (R / "P3A_INTEGRATED_EXPERT_VERIFICATION.md").write_text(f"# STREAMQ5-MoE P3A - onafhankelijke verificatie\n\nUitkomst: **{status}** ({passed}/{total}). Test mean/p95/max: {payload['test_mean_ms']:.3f}/{payload['test_p95_ms']:.3f}/{payload['test_max_ms']:.3f} ms.\n", encoding="utf-8"); print(json.dumps({"status": status, "checks": f"{passed}/{total}", "transfer": transfer, "test_mean_ms": payload["test_mean_ms"], "test_p95_ms": payload["test_p95_ms"]}, indent=2)); raise SystemExit(0 if status.endswith("pass") else 1)
