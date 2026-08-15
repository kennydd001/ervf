from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
PREREG = R / "N4B_SYNTHETIC_80B_GPU_SHAPE_PREREGISTRATION.md"
EVALUATOR = ROOT / "scripts/streamq5_moe/run_n4b_synthetic_80b_gpu_shape.py"
RESULT = R / "n4b_synthetic_80b_gpu_shape.json"
N4A = R / "n4a_synthetic_80b_shape_capacity.json"
N1C = R / "n1c_generalized_exact_reduction_autotuner.json"
OUTPUT = R / "n4b_synthetic_80b_gpu_shape_verification.json"

LAYERS, ACTIVE, HIDDEN, INTER = 48, 11, 2048, 512
WIDTHS = (8, 16, 32)
HEADER, ALIGNMENT, GROUP = 64, 4096, 128
SOURCE_Q8_BYTES = 1_248_931_840


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def bf16_round(value: float) -> float:
    data = np.asarray([value], dtype=np.float32)
    bits = data.view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits &= np.uint32(0xFFFF0000)
    return float(data[0])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    n4a = json.loads(N4A.read_text(encoding="utf-8"))
    n1c = json.loads(N1C.read_text(encoding="utf-8"))
    evaluator_source = EVALUATOR.read_text(encoding="utf-8")

    weights = HIDDEN * INTER
    code_bytes = weights * 5 // 8
    scale_bytes = weights // GROUP * 2
    matrix_bytes = math.ceil((HEADER + code_bytes + scale_bytes) / ALIGNMENT) * ALIGNMENT
    padding_bytes = matrix_bytes - HEADER - code_bytes - scale_bytes
    expert_bytes = matrix_bytes * 3
    slots = LAYERS * ACTIVE
    bank_bytes = slots * expert_bytes
    outputs_per_layer = ACTIVE * (INTER + INTER + HIDDEN)
    outputs_total = LAYERS * outputs_per_layer

    recomputed_stats = {}
    raw_stats_exact = True
    sample_counts_exact = True
    for split, expected_count in (("validation", 30), ("test", 120)):
        recomputed_stats[split] = {}
        for width, row in result[split].items():
            calculated = stat(row["event_ms"])
            recomputed_stats[split][width] = calculated
            raw_stats_exact &= all(calculated[key] == row["stats"][key] for key in calculated)
            sample_counts_exact &= len(row["event_ms"]) == expected_count

    exact_widths_reported = all(
        result["correctness"][str(width)] == {
            "bitwise_equal": True,
            "elements": outputs_total,
            "different": 0,
            "max_abs": 0.0,
            "finite": True,
        }
        for width in WIDTHS
    )
    eligible = [width for width in WIDTHS if result["correctness"][str(width)]["bitwise_equal"]]
    selected = min(eligible, key=lambda width: recomputed_stats["validation"][str(width)]["p50"])
    selected_test_key = str(selected) if str(selected) in result["test"] else "selected16"
    expert_test = recomputed_stats["test"][selected_test_key]

    q8_source_p95 = n1c["test"]["q8"]["candidate"]["p95"]
    shell_bytes = n4a["device_budget"]["q8_device_shell_bytes"]
    dense_p95 = q8_source_p95 * shell_bytes / SOURCE_Q8_BYTES
    dense_2x_p95 = dense_p95 * 2
    resident_compute_p95 = expert_test["p95"]
    all_cold_h2d_ms = n4a["expert_accounting"]["all_cold_h2d_ms_at_local_26_16_decimal_gbps"]
    ideal_overlap_expert_ms = max(resident_compute_p95, all_cold_h2d_ms)
    serial_expert_ms = resident_compute_p95 + all_cold_h2d_ms

    # The proven P6/P7 SwiGLU rounds SiLU to BF16 before the multiply. N4B omits it.
    canonical_swiglu_present = "float silu=round_bf16" in evaluator_source.replace(" ", "")
    n4b_expression_present = "gate[i]=round_bf16((g/(1.0f+expf(-g)))*u)" in evaluator_source.replace(" ", "")
    example_g, example_u = -7.9375, 1.5
    silu = example_g / (1.0 + float(np.exp(np.float32(-example_g))))
    canonical_example = bf16_round(bf16_round(silu) * example_u)
    n4b_example = bf16_round(silu * example_u)

    reported_gate_recalculation = {
        "all_widths_exact": exact_widths_reported,
        "expert_p95_le_50ms": resident_compute_p95 <= 50,
        "dense_byte_linear_p95_le_40ms": dense_p95 <= 40,
        "dense_2x_p95_le_40ms": dense_2x_p95 <= 40,
        "projected_total_p95_le_90ms": resident_compute_p95 + dense_2x_p95 <= 90,
        "n4a_host_le_58g": n4a["gates"]["host_with_1gib_reserve_le_58_gib"],
        "n4a_4k_cache": n4a["gates"]["4k_cache_at_least_32_per_layer"],
        "n4a_32k_cache": n4a["gates"]["32k_cache_at_least_32_per_layer"],
    }
    transfer_aware_projection_gates = {
        "ideal_overlap_expert_le_50ms": ideal_overlap_expert_ms <= 50,
        "conservative_serial_expert_le_50ms": serial_expert_ms <= 50,
        "ideal_overlap_total_with_dense_2x_le_90ms": ideal_overlap_expert_ms + dense_2x_p95 <= 90,
        "serial_total_with_dense_2x_le_90ms": serial_expert_ms + dense_2x_p95 <= 90,
    }

    checks = {
        "preregistration_hash_current": result["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "n4a_hash_current": result["inputs"]["n4a_sha256"] == sha256(N4A),
        "n1c_hash_current": result["inputs"]["n1c_sha256"] == sha256(N1C),
        "record_code_bytes_exact": code_bytes == 655_360,
        "record_scale_bytes_exact": scale_bytes == 16_384,
        "record_padding_bytes_exact": padding_bytes == 4_032,
        "matrix_record_bytes_exact": matrix_bytes == 675_840,
        "expert_record_bytes_exact": expert_bytes == 2_027_520,
        "resident_slot_count_exact": slots == 528,
        "resident_bank_bytes_exact": bank_bytes == result["physical"]["q5_bank_bytes"] == 1_070_530_560,
        "output_elements_exact": outputs_total == 1_622_016,
        "reported_width_comparisons_structurally_exact": exact_widths_reported,
        "raw_event_statistics_exact": raw_stats_exact,
        "raw_event_sample_counts_exact": sample_counts_exact,
        "selected_width_recomputed": selected == result["selected_width"] == 8,
        "dense_projection_exact": all((
            q8_source_p95 == result["dense_projection"]["n1c_q8_p95_ms"],
            SOURCE_Q8_BYTES == result["dense_projection"]["source_bytes"],
            shell_bytes == result["dense_projection"]["official_80b_shell_bytes"],
            dense_p95 == result["dense_projection"]["byte_linear_p95_ms"],
            dense_2x_p95 == result["dense_projection"]["conservative_2x_p95_ms"],
            resident_compute_p95 + dense_2x_p95 == result["dense_projection"]["projected_total_p95_ms"],
        )),
        "reported_gates_recomputed_exact": reported_gate_recalculation == result["gates"],
        "reported_overall_pass_recomputed": all(reported_gate_recalculation.values()) == result["overall_pass"],
        "transfer_aware_analytical_gates_pass": all(transfer_aware_projection_gates.values()),
        "evaluator_hash_recorded_in_result": "script_sha256" in result["inputs"],
        "raw_width_outputs_or_digests_archived": any(
            key in result for key in ("output_sha256", "width_output_sha256", "raw_outputs")
        ),
        "canonical_streamq5_swiglu_rounding_present": canonical_swiglu_present,
    }
    methodological_findings = {
        "silu_rounding_contract_breach": {
            "present": n4b_expression_present and not canonical_swiglu_present,
            "canonical_contract": "round_bf16(round_bf16(g/(1+exp(-g))) * u)",
            "n4b_contract": "round_bf16((g/(1+exp(-g))) * u)",
            "counterexample": {
                "gate_bf16": example_g,
                "up_bf16": example_u,
                "canonical_float32": canonical_example,
                "n4b_float32": n4b_example,
                "different": canonical_example != n4b_example,
            },
            "impact": (
                "All widths share the same non-canonical SwiGLU, so mutual equality does not establish "
                "bit-exactness against the proven STREAMQ5 runtime."
            ),
        },
        "independent_width_exactness_limit": (
            "The JSON archives only comparison summaries, not raw outputs or per-width digests; "
            "without a prohibited GPU rerun, bitwise equality cannot be independently recomputed."
        ),
        "artifact_binding_limit": (
            "The result records preregistration/N4A/N1C hashes but omits the evaluator SHA256 and input digest."
        ),
        "projection_limit": (
            "The measured 7.7067 ms p95 is resident Q5 compute. All-cold H2D and the dense shell are analytical; "
            "DeltaNet recurrence, routing/aggregation, real cache behavior and their p95 variability are not measured."
        ),
    }
    critical_audit_checks = (
        checks["evaluator_hash_recorded_in_result"],
        checks["raw_width_outputs_or_digests_archived"],
        checks["canonical_streamq5_swiglu_rounding_present"],
    )
    verification = {
        "kind": "streamq5_moe_n4b_synthetic_80b_gpu_shape_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "evaluator_sha256_current": sha256(EVALUATOR),
            "result_sha256": sha256(RESULT),
            "n4a_sha256": sha256(N4A),
            "n1c_sha256": sha256(N1C),
            "gpu_rerun_performed": False,
        },
        "recomputed_record_contract": {
            "weights_per_matrix": weights,
            "code_bytes": code_bytes,
            "scale_bytes": scale_bytes,
            "header_bytes": HEADER,
            "padding_bytes": padding_bytes,
            "matrix_record_bytes": matrix_bytes,
            "expert_record_bytes": expert_bytes,
            "resident_slots": slots,
            "resident_bank_bytes": bank_bytes,
        },
        "recomputed_output_contract": {
            "outputs_per_layer": outputs_per_layer,
            "outputs_all_layers": outputs_total,
        },
        "recomputed_event_stats": recomputed_stats,
        "width_exactness": {
            "reported_summaries_consistent": exact_widths_reported,
            "independently_recomputable_without_gpu": False,
            "reason": methodological_findings["independent_width_exactness_limit"],
        },
        "recomputed_selection": {"selected_width": selected, "expert_test_stats": expert_test},
        "recomputed_dense_projection": {
            "n1c_q8_p95_ms": q8_source_p95,
            "source_bytes": SOURCE_Q8_BYTES,
            "official_80b_shell_bytes": shell_bytes,
            "byte_linear_p95_ms": dense_p95,
            "conservative_2x_p95_ms": dense_2x_p95,
            "reported_formula_total_p95_ms": resident_compute_p95 + dense_2x_p95,
        },
        "transfer_aware_sensitivity": {
            "resident_compute_p95_ms": resident_compute_p95,
            "all_cold_h2d_analytical_ms": all_cold_h2d_ms,
            "ideal_overlap_expert_ms": ideal_overlap_expert_ms,
            "conservative_serial_expert_ms": serial_expert_ms,
            "ideal_overlap_total_with_dense_2x_ms": ideal_overlap_expert_ms + dense_2x_p95,
            "serial_total_with_dense_2x_ms": serial_expert_ms + dense_2x_p95,
            "gates": transfer_aware_projection_gates,
        },
        "reported_gate_recalculation": reported_gate_recalculation,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "methodological_findings": methodological_findings,
        "reported_numerical_pass_confirmed": all(reported_gate_recalculation.values()),
        "independent_verification_pass": all(checks.values()),
        "verdict": (
            "reported_numeric_shape_timing_gates_recompute_but_independent_exact_port_gate_fails"
            if not all(critical_audit_checks)
            else "independently_verified"
        ),
        "claim_boundary": (
            "No GPU rerun. Arithmetic, raw event summaries and projections were independently recomputed. "
            "Width outputs cannot be independently reconstructed from the archived artifact."
        ),
    }
    OUTPUT.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verification["verdict"],
        "reported_numerical_pass_confirmed": verification["reported_numerical_pass_confirmed"],
        "independent_verification_pass": verification["independent_verification_pass"],
        "passed_checks": verification["passed_checks"],
        "total_checks": verification["total_checks"],
        "failed_checks": [key for key, value in checks.items() if not value],
        "recomputed_selection": verification["recomputed_selection"],
        "recomputed_dense_projection": verification["recomputed_dense_projection"],
        "transfer_aware_sensitivity": verification["transfer_aware_sensitivity"],
    }, indent=2))


if __name__ == "__main__":
    main()
