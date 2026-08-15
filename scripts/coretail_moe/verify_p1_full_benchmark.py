from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/coretail_moe/P1_FUSED_KERNEL_PREREGISTRATION.md"
LOCK = ROOT / "reports/coretail_moe/p1_fused_kernel_input_lock.json"
P0 = ROOT / "reports/coretail_moe/p0_full_bank_format_verification.json"
SMOKE = ROOT / "reports/coretail_moe/p1a_kernel_smoke_result.json"
FORMAT = ROOT / "reports/coretail_moe/p0_full_bank_format_result.json"
RESULT = ROOT / "reports/coretail_moe/p1_full_benchmark_result.json"
OUT_JSON = ROOT / "reports/coretail_moe/p1_full_benchmark_verification.json"
OUT_MD = ROOT / "reports/coretail_moe/P1_FULL_BENCHMARK_VERIFICATION.md"
DOMAINS = ("general", "instruction", "code", "math", "multilingual")
MATRICES = ("gate", "up", "down")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= max(1e-9, abs(expected) * 1e-12)


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite P1 verification")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    physical = json.loads(FORMAT.read_text(encoding="utf-8"))
    checks = {
        "preregistration_hash": sha256(PREREG) == result["inputs"]["preregistration_sha256"] == lock["preregistration_sha256"],
        "lock_hash": sha256(LOCK) == result["inputs"]["lock_sha256"],
        "p0_hash": sha256(P0) == result["inputs"]["p0_verification_sha256"] == lock["p0_verification_sha256"],
        "smoke_hash": sha256(SMOKE) == result["inputs"]["smoke_sha256"],
        "record_count_and_keys": True,
        "all_record_correct_flags": True,
        "all_error_tolerances": True,
        "all_timings_finite_positive": True,
        "aggregate_arithmetic": True,
        "tail_route_hashes": True,
        "tail_distribution_arithmetic": True,
        "reported_gates_recomputed": True,
        "final_status_consistent": True,
    }
    expected_keys = {
        f"{layer}:{expert}:{matrix}"
        for layer in lock["layers"]
        for expert in lock["experts"][str(layer)]
        for matrix in MATRICES
    }
    records = result["records"]
    checks["record_count_and_keys"] &= len(records) == 72 and {record["key"] for record in records} == expected_keys
    tolerance = lock["tolerances"]
    total_weights = 0
    sums = {"bf16": 0.0, "fixed": 0.0, "core_p50": 0.0, "core_p95": 0.0}
    for record in records:
        checks["all_record_correct_flags"] &= record["correct"] is True
        for name in ("fixed", "coretail", "cross"):
            error = record["errors"][name]
            checks["all_error_tolerances"] &= (
                error["finite"] is True
                and error["max_abs"] <= tolerance["max_abs"]
                and error["relative_l2"] <= tolerance["relative_l2"]
            )
        for timing in ("bf16_dequantized_reference", "fixed_uint2", "coretail"):
            values = record["timing"][timing]
            checks["all_timings_finite_positive"] &= all(
                np.isfinite(values[key]) and values[key] > 0
                for key in ("p50_ms", "p95_ms", "p99_ms", "p50_weight_applications_per_second", "p95_weight_applications_per_second")
            )
        weight = record["weights"]
        total_weights += weight
        sums["bf16"] += record["timing"]["bf16_dequantized_reference"]["p50_ms"] / 1000
        sums["fixed"] += record["timing"]["fixed_uint2"]["p50_ms"] / 1000
        sums["core_p50"] += record["timing"]["coretail"]["p50_ms"] / 1000
        sums["core_p95"] += record["timing"]["coretail"]["p95_ms"] / 1000
    recomputed_aggregate = {
        "bf16_p50_weight_applications_per_second": total_weights / sums["bf16"],
        "fixed_uint2_p50_weight_applications_per_second": total_weights / sums["fixed"],
        "coretail_p50_weight_applications_per_second": total_weights / sums["core_p50"],
        "coretail_p95_weight_applications_per_second": total_weights / sums["core_p95"],
    }
    checks["aggregate_arithmetic"] &= all(
        close(result["aggregate"][key], value) for key, value in recomputed_aggregate.items()
    )

    record_map = {record["key"]: record for record in physical["records"]}
    raw_cost = np.zeros((48, 128), dtype=np.int64)
    compressed_cost = np.zeros((48, 128), dtype=np.int64)
    for layer in range(48):
        route_path = ROOT / f"reports/runs/qwen_gptq_bank/p0_supplement_routes/layer_{layer:02d}.safetensors"
        checks["tail_route_hashes"] &= sha256(route_path) == result["tail"]["route_artifact_sha256"][str(layer)]
        for expert in range(128):
            for matrix in MATRICES:
                tail = record_map[f"{layer}:{expert}:{matrix}"]["tail"]
                raw_cost[layer, expert] += tail["raw_flag_bytes"] + tail["rows"] + 4 * (tail["rows"] + 1)
                compressed_cost[layer, expert] += tail["header_bytes"] + tail["index_bytes"] + tail["payload_bytes"]
    token_count = lock["tail_trace"]["tokens_per_domain"]
    all_raw = []
    all_compressed = []
    for domain in DOMAINS:
        domain_raw = np.zeros(token_count, dtype=np.int64)
        domain_compressed = np.zeros(token_count, dtype=np.int64)
        for layer in range(48):
            route_path = ROOT / f"reports/runs/qwen_gptq_bank/p0_supplement_routes/layer_{layer:02d}.safetensors"
            with safe_open(route_path, framework="pt", device="cpu") as handle:
                ids = handle.get_tensor(f"{domain}_router_ids")[:token_count].numpy().astype(np.int64)
            domain_raw += raw_cost[layer][ids].sum(axis=1)
            domain_compressed += compressed_cost[layer][ids].sum(axis=1)
        all_raw.append(domain_raw); all_compressed.append(domain_compressed)
    all_raw = np.concatenate(all_raw); all_compressed = np.concatenate(all_compressed)
    expected_raw = {name: int(np.percentile(all_raw, percentile, method="higher")) for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}
    expected_compressed = {name: int(np.percentile(all_compressed, percentile, method="higher")) for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}
    checks["tail_distribution_arithmetic"] &= (
        result["tail"]["tokens"] == 5120
        and result["tail"]["raw_runtime_bytes_conservative_upper_bound"] == expected_raw
        and result["tail"]["compressed_physical_bytes"] == expected_compressed
    )
    recomputed_gates = {
        "all_72_correct": bool(checks["record_count_and_keys"] and checks["all_record_correct_flags"] and checks["all_error_tolerances"]),
        "coretail_p50_throughput_ge_27_2_gweights_s": recomputed_aggregate["coretail_p50_weight_applications_per_second"] >= lock["gates"]["weights_per_second"],
        "coretail_p95_throughput_ge_27_2_gweights_s": recomputed_aggregate["coretail_p95_weight_applications_per_second"] >= lock["gates"]["weights_per_second"],
        "tail_decode_h2d_p95_le_33_3_ms": result["tail"]["p95_size_copy_timing_ms"]["p95"] <= lock["gates"]["tail_decode_h2d_p95_ms"],
        "no_full_dequantized_matrix_in_custom_kernels": result["gates"]["no_full_dequantized_matrix_in_custom_kernels"],
    }
    checks["reported_gates_recomputed"] &= result["gates"] == recomputed_gates
    checks["final_status_consistent"] &= result["status"] == ("p1_pass" if all(recomputed_gates.values()) else "p1_fail")
    passed = all(checks.values())
    payload = {
        "kind": "coretail_moe_p1_full_benchmark_independent_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p1_verification_pass" if passed else "p1_verification_fail",
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "recomputed_aggregate": recomputed_aggregate,
        "recomputed_gates": recomputed_gates,
        "claim_boundary": "Independent provenance, arithmetic, route-distribution and gate audit; CUDA event samples themselves remain the primary measurement artifact.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# CORETAIL-MoE P1 — onafhankelijke benchmarkaudit", "",
        f"Uitkomst: **{payload['status']}** ({payload['passed_checks']}/{payload['total_checks']}).", "",
        f"Herberekende CORETAIL-throughput: p50 {recomputed_aggregate['coretail_p50_weight_applications_per_second']/1e9:.3f} en p95 {recomputed_aggregate['coretail_p95_weight_applications_per_second']/1e9:.3f} Gweight/s.",
        f"De 5.120-token routed tailverdeling, alle inputhashes, 72 correctheidsgevallen en vijf gatebeslissingen zijn onafhankelijk herberekend.", "",
    ]), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}", "aggregate": recomputed_aggregate, "gates": recomputed_gates}, indent=2))
