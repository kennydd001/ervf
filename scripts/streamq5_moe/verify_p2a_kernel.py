from __future__ import annotations

import hashlib
import json
import math

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P2A_Q5_KERNEL_PREREGISTRATION.md"
LOCK_PATH = R / "p2a_kernel_input_lock.json"
EVALUATOR_LOCK = R / "p2a_kernel_benchmark_evaluator_lock.json"
EVALUATOR = ROOT / "scripts/streamq5_moe/run_p2a_q5_kernel.py"
P1D = R / "p1d_physical_bank_verification.json"
SMOKE = R / "p2a_kernel_smoke.json"
RESULT = R / "p2a_kernel_benchmark.json"
FULL_WEIGHTS = 1_811_939_328


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b, tolerance=1e-10):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


if __name__ == "__main__":
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    evaluator_lock = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = []
    add = lambda name, passed: checks.append({"name": name, "pass": bool(passed)})
    add("preregistration hash", sha256(PREREG) == lock["preregistration_sha256"])
    add("P1D verification hash", sha256(P1D) == lock["p1d_verification_sha256"])
    add("evaluator hash", sha256(EVALUATOR) == evaluator_lock["evaluator_sha256"])
    add("input lock hash", sha256(LOCK_PATH) == evaluator_lock["input_lock_sha256"])
    add("smoke hash", sha256(SMOKE) == evaluator_lock["smoke_sha256"])
    expected_keys = {
        f"layer_{layer:02d}_expert_{expert:03d}_{name}"
        for layer in lock["layers"] for expert in lock["experts"][str(layer)] for name in ("gate", "up", "down")
    }
    cases = result["cases"]
    add("72 fixed cases", len(cases) == 72 and {row["key"] for row in cases} == expected_keys)
    tolerance = lock["tolerances"]
    correct = all(row["correct"] and row["error"]["finite"] and row["error"]["max_abs"] <= tolerance["max_abs"] and row["error"]["relative_l2"] <= tolerance["relative_l2"] for row in cases)
    add("all correctness metrics", correct)
    add("physical shapes and bytes", all(row["rows"] * row["columns"] == row["weights"] == 1_572_864 and row["physical_bytes"] == 1_007_616 for row in cases))
    total_weights = sum(row["weights"] for row in cases)
    p50_seconds = sum(row["timing"]["physical_q5"]["p50_ms"] for row in cases) / 1000
    p95_seconds = sum(row["timing"]["physical_q5"]["p95_ms"] for row in cases) / 1000
    bf16_seconds = sum(row["timing"]["bf16_dequantized_baseline"]["p50_ms"] for row in cases) / 1000
    aggregate = result["aggregate"]
    add("sample weight arithmetic", total_weights == aggregate["sample_weights"] == 113_246_208)
    add("p50 throughput arithmetic", close(total_weights / p50_seconds, aggregate["q5_p50_weight_applications_per_second"]))
    add("p95 throughput arithmetic", close(total_weights / p95_seconds, aggregate["q5_summed_p95_weight_applications_per_second"]))
    add("BF16 baseline arithmetic", close(total_weights / bf16_seconds, aggregate["bf16_p50_weight_applications_per_second"]))
    add("full-token projection arithmetic", close(FULL_WEIGHTS / aggregate["q5_p50_weight_applications_per_second"] * 1000, aggregate["full_token_q5_p50_compute_ms"]) and close(FULL_WEIGHTS / aggregate["q5_summed_p95_weight_applications_per_second"] * 1000, aggregate["full_token_q5_summed_p95_compute_ms"]))
    expected_gates = {
        "all_72_correct": correct,
        "p50_throughput_ge_27_2_gweights_s": aggregate["q5_p50_weight_applications_per_second"] >= lock["gates"]["weights_per_second"],
        "summed_p95_throughput_ge_27_2_gweights_s": aggregate["q5_summed_p95_weight_applications_per_second"] >= lock["gates"]["weights_per_second"],
        "full_token_p95_compute_ms_le_66_615": aggregate["full_token_q5_summed_p95_compute_ms"] <= lock["gates"]["full_token_p95_compute_ms_max"],
        "direct_physical_q5_no_dequantized_candidate_matrix": True,
    }
    add("gate recomputation", result["gates"] == expected_gates and all(expected_gates.values()))
    add("result decision", result["status"] == "p2a_kernel_pass")
    passed = sum(row["pass"] for row in checks); total = len(checks)
    status = "p2a_kernel_verification_pass" if passed == total else "p2a_kernel_verification_fail"
    payload = {"kind": "streamq5_moe_p2a_independent_kernel_verification", "status": status, "checks_passed": passed, "checks_total": total, "checks": checks, "p50_gweights_s": aggregate["q5_p50_weight_applications_per_second"] / 1e9, "summed_p95_gweights_s": aggregate["q5_summed_p95_weight_applications_per_second"] / 1e9, "full_token_p95_compute_ms": aggregate["full_token_q5_summed_p95_compute_ms"], "claim_boundary": result["claim_boundary"]}
    (R / "p2a_kernel_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (R / "P2A_KERNEL_VERIFICATION.md").write_text(f"# STREAMQ5-MoE P2A - onafhankelijke verificatie\n\nUitkomst: **{status}** ({passed}/{total}). P50/p95: {payload['p50_gweights_s']:.3f}/{payload['summed_p95_gweights_s']:.3f} Gweight/s.\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": f"{passed}/{total}", "p50_gweights_s": payload["p50_gweights_s"], "p95_gweights_s": payload["summed_p95_gweights_s"]}, indent=2))
    if status.endswith("fail"):
        raise SystemExit(1)
