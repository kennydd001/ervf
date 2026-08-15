from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
PREREG = R / "N4BR_SYNTHETIC_80B_EXACT_REPLICATION_PREREGISTRATION.md"
EVALUATOR = ROOT / "scripts/streamq5_moe/run_n4br_synthetic_80b_exact_replication.py"
RESULT = R / "n4br_synthetic_80b_exact_replication.json"
N4B = R / "n4b_synthetic_80b_gpu_shape.json"
N4A = R / "n4a_synthetic_80b_shape_capacity.json"
N1C = R / "n1c_generalized_exact_reduction_autotuner.json"
OUTPUT = R / "n4br_synthetic_80b_exact_replication_verification.json"

LAYERS, ACTIVE, HIDDEN, INTER = 48, 11, 2048, 512
WIDTHS, SEED = (8, 16, 32), 120825
HEADER, ALIGNMENT, GROUP = 64, 4096, 128
SOURCE_Q8_BYTES = 1_248_931_840


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def bf16_round(value) -> np.float32:
    data = np.asarray([value], dtype=np.float32)
    bits = data.view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits &= np.uint32(0xFFFF0000)
    return data[0]


def synthetic_q5_weights() -> np.ndarray:
    word = sum(0x55 << (8 * index) for index in range(5))
    codes = np.asarray([((word >> (5 * item)) & 31) - 15 for item in range(8)], dtype=np.int32)
    scale = np.asarray(np.uint32(0x3C00) << np.uint32(16)).view(np.float32)
    return np.asarray([bf16_round(np.float32(code) * scale) for code in codes], dtype=np.float32)


def q5_ervf16_row(inputs: np.ndarray, cols: int) -> np.float32:
    """CPU reconstruction of the frozen CUDA width-16 virtual reduction graph."""
    width, virtual = 16, 16
    packs = cols // 8
    weights = synthetic_q5_weights()
    partial = np.zeros((width, virtual), dtype=np.float32)
    for lane in range(width):
        for virtual_index in range(virtual):
            tid = lane + width * virtual_index
            total = np.float32(0)
            for pack in range(tid, packs, 256):
                column = pack * 8
                for item in range(8):
                    total = np.float32(total + np.float32(weights[item] * inputs[column + item]))
            partial[lane, virtual_index] = total
    for stride in (128, 64, 32, 16):
        for lane in range(width):
            for index in range(stride // width):
                partial[lane, index] = np.float32(
                    partial[lane, index] + partial[lane, index + stride // width]
                )
    values = partial[:, 0].copy()
    for offset in (8, 4, 2, 1):
        previous = values.copy()
        for lane in range(width - offset):
            values[lane] = np.float32(previous[lane] + previous[lane + offset])
    return bf16_round(values[0])


def reconstruct_input_and_output() -> tuple[str, str, dict[str, float]]:
    rng = np.random.default_rng(SEED)
    hidden = rng.standard_normal(HIDDEN, dtype=np.float32)
    input_digest = hashlib.sha256(hidden.tobytes()).hexdigest()
    raw_gate = q5_ervf16_row(hidden, HIDDEN)
    raw_up = raw_gate
    silu = bf16_round(np.float32(raw_gate / np.float32(1.0 + np.exp(np.float32(-raw_gate)))))
    gate = bf16_round(np.float32(silu * raw_up))
    activation = np.full(INTER, gate, dtype=np.float32)
    down = q5_ervf16_row(activation, INTER)
    layer = np.concatenate((
        np.full(ACTIVE * INTER, gate, dtype=np.float32),
        np.full(ACTIVE * INTER, raw_up, dtype=np.float32),
        np.full(ACTIVE * HIDDEN, down, dtype=np.float32),
    ))
    output = np.tile(layer, (LAYERS, 1))
    output_digest = hashlib.sha256(output.tobytes()).hexdigest()
    values = {
        "raw_gate_up": float(raw_gate),
        "rounded_silu": float(silu),
        "post_swiglu_gate": float(gate),
        "down": float(down),
    }
    return input_digest, output_digest, values


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    n4b = json.loads(N4B.read_text(encoding="utf-8"))
    n4a = json.loads(N4A.read_text(encoding="utf-8"))
    n1c = json.loads(N1C.read_text(encoding="utf-8"))
    source = EVALUATOR.read_text(encoding="utf-8").replace(" ", "")

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

    input_digest, independent_output_digest, reconstructed_values = reconstruct_input_and_output()
    digest_values = list(result["output_digests"].values())
    digests_valid_sha256 = all(len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in digest_values)

    recomputed_stats = {}
    raw_stats_exact = True
    sample_counts_exact = True
    for split, expected_count in (("validation", 30), ("test", 120)):
        recomputed_stats[split] = {}
        for width, row in result[split].items():
            calculated = summary(row["event_ms"])
            recomputed_stats[split][width] = calculated
            raw_stats_exact &= all(calculated[key] == row["stats"][key] for key in calculated)
            sample_counts_exact &= len(row["event_ms"]) == expected_count

    output_summary_exact = all(
        result["correctness"][str(width)] == {
            "bitwise_equal": True,
            "elements": outputs_total,
            "different": 0,
            "max_abs": 0.0,
            "finite": True,
        }
        for width in WIDTHS
    )
    selected = min(WIDTHS, key=lambda width: recomputed_stats["validation"][str(width)]["p50"])
    selected_test = recomputed_stats["test"][str(selected)]

    q8_source_p95 = n1c["test"]["q8"]["candidate"]["p95"]
    shell_bytes = n4a["device_budget"]["q8_device_shell_bytes"]
    dense_p95 = q8_source_p95 * shell_bytes / SOURCE_Q8_BYTES
    dense_2x_p95 = dense_p95 * 2
    resident_expert_p95 = selected_test["p95"]
    reported_total = resident_expert_p95 + dense_2x_p95
    all_cold_h2d_ms = n4a["expert_accounting"]["all_cold_h2d_ms_at_local_26_16_decimal_gbps"]
    ideal_overlap_expert = max(resident_expert_p95, all_cold_h2d_ms)
    serial_expert = resident_expert_p95 + all_cold_h2d_ms

    recomputed_gates = {
        "all_widths_exact": output_summary_exact,
        "expert_p95_le_50ms": resident_expert_p95 <= 50,
        "dense_byte_linear_p95_le_40ms": dense_p95 <= 40,
        "dense_2x_p95_le_40ms": dense_2x_p95 <= 40,
        "projected_total_p95_le_90ms": reported_total <= 90,
        "n4a_host_le_58g": n4a["gates"]["host_with_1gib_reserve_le_58_gib"],
        "n4a_4k_cache": n4a["gates"]["4k_cache_at_least_32_per_layer"],
        "n4a_32k_cache": n4a["gates"]["32k_cache_at_least_32_per_layer"],
        "all_output_digests_equal": len(set(digest_values)) == 1,
    }
    transfer_gates = {
        "ideal_overlap_expert_le_50ms": ideal_overlap_expert <= 50,
        "serial_expert_le_50ms": serial_expert <= 50,
        "ideal_overlap_total_le_90ms": ideal_overlap_expert + dense_2x_p95 <= 90,
        "serial_total_le_90ms": serial_expert + dense_2x_p95 <= 90,
    }

    canonical_swiglu = (
        "floatsilu=round_bf16(g/(1.0f+expf(-g)))" in source
        and "gate[i]=round_bf16(silu*u)" in source
    )
    old_noncanonical_swiglu_absent = "gate[i]=round_bf16((g/(1.0f+expf(-g)))*u)" not in source
    checks = {
        "preregistration_hash_exact": result["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "evaluator_hash_exact": result["inputs"]["evaluator_sha256"] == sha256(EVALUATOR),
        "n4b_hash_exact": result["inputs"]["n4b_sha256"] == sha256(N4B),
        "n4a_hash_exact": result["inputs"]["n4a_sha256"] == sha256(N4A),
        "n1c_hash_exact": result["inputs"]["n1c_sha256"] == sha256(N1C),
        "seed_exact": result["inputs"]["seed"] == SEED,
        "input_digest_independently_recomputed": result["inputs"]["input_sha256"] == input_digest,
        "canonical_two_stage_swiglu_present": canonical_swiglu,
        "old_noncanonical_swiglu_absent": old_noncanonical_swiglu_absent,
        "record_code_bytes_exact": code_bytes == 655_360,
        "record_scale_bytes_exact": scale_bytes == 16_384,
        "record_padding_bytes_exact": padding_bytes == 4_032,
        "matrix_record_bytes_exact": matrix_bytes == 675_840,
        "expert_record_bytes_exact": expert_bytes == 2_027_520,
        "slot_count_exact": slots == 528,
        "bank_bytes_exact": bank_bytes == result["physical"]["q5_bank_bytes"] == 1_070_530_560,
        "physical_shape_exact": all((
            result["physical"]["layers"] == LAYERS,
            result["physical"]["active_experts"] == ACTIVE,
            result["physical"]["hidden"] == HIDDEN,
            result["physical"]["intermediate"] == INTER,
        )),
        "output_element_count_exact": outputs_total == 1_622_016,
        "comparison_summaries_exact": output_summary_exact,
        "digest_values_valid_sha256": digests_valid_sha256,
        "all_archived_digests_equal": len(set(digest_values)) == 1,
        "output_digest_independently_recomputed": set(digest_values) == {independent_output_digest},
        "raw_event_stats_exact": raw_stats_exact,
        "raw_event_sample_counts_exact": sample_counts_exact,
        "selected_width_recomputed": selected == result["selected_width"] == 8,
        "expert_test_stats_exact": all(selected_test[key] == result["expert_test_stats"][key] for key in selected_test),
        "fresh_test_arrays_not_reused_from_n4b": result["test"] != n4b["test"],
        "dense_source_p95_exact": q8_source_p95 == result["dense_projection"]["n1c_q8_p95_ms"],
        "dense_source_bytes_exact": SOURCE_Q8_BYTES == result["dense_projection"]["source_bytes"],
        "dense_shell_bytes_exact": shell_bytes == result["dense_projection"]["official_80b_shell_bytes"],
        "dense_projection_exact": all((
            dense_p95 == result["dense_projection"]["byte_linear_p95_ms"],
            dense_2x_p95 == result["dense_projection"]["conservative_2x_p95_ms"],
            reported_total == result["dense_projection"]["projected_total_p95_ms"],
        )),
        "all_reported_gates_exact": recomputed_gates == result["gates"],
        "reported_overall_pass_exact": all(recomputed_gates.values()) == result["overall_pass"],
        "transfer_sensitivity_gates_pass": all(transfer_gates.values()),
    }
    verification = {
        "kind": "streamq5_moe_n4br_synthetic_80b_exact_replication_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "evaluator_sha256": sha256(EVALUATOR),
            "result_sha256": sha256(RESULT),
            "n4b_sha256": sha256(N4B),
            "n4a_sha256": sha256(N4A),
            "n1c_sha256": sha256(N1C),
            "gpu_rerun_performed": False,
        },
        "record_contract": {
            "weights_per_matrix": weights,
            "code_bytes": code_bytes,
            "scale_bytes": scale_bytes,
            "header_bytes": HEADER,
            "padding_bytes": padding_bytes,
            "matrix_record_bytes": matrix_bytes,
            "expert_record_bytes": expert_bytes,
            "slots": slots,
            "bank_bytes": bank_bytes,
        },
        "output_contract": {
            "outputs_per_layer": outputs_per_layer,
            "outputs_all_layers": outputs_total,
            "independent_input_sha256": input_digest,
            "independent_output_sha256": independent_output_digest,
            "reconstructed_uniform_values": reconstructed_values,
        },
        "event_stats": recomputed_stats,
        "selection": {"selected_width": selected, "expert_test_stats": selected_test},
        "dense_projection": {
            "n1c_q8_p95_ms": q8_source_p95,
            "source_bytes": SOURCE_Q8_BYTES,
            "official_80b_shell_bytes": shell_bytes,
            "byte_linear_p95_ms": dense_p95,
            "conservative_2x_p95_ms": dense_2x_p95,
            "reported_total_p95_ms": reported_total,
        },
        "transfer_aware_sensitivity": {
            "all_cold_h2d_analytical_ms": all_cold_h2d_ms,
            "ideal_overlap_expert_ms": ideal_overlap_expert,
            "serial_expert_ms": serial_expert,
            "ideal_overlap_total_ms": ideal_overlap_expert + dense_2x_p95,
            "serial_total_ms": serial_expert + dense_2x_p95,
            "gates": transfer_gates,
        },
        "recomputed_gates": recomputed_gates,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "overall_pass": all(checks.values()),
        "verdict": "independently_verified_exact_synthetic_shape_timing_pass" if all(checks.values()) else "verification_failed",
        "claim_boundary": (
            "CPU-only independent artifact audit, including exact reconstruction of the deterministic synthetic "
            "input and 48-layer output digest. No GPU rerun and no expansion beyond N4B-R's claim boundary."
        ),
    }
    OUTPUT.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verification["verdict"],
        "passed_checks": verification["passed_checks"],
        "total_checks": verification["total_checks"],
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "independent_output_sha256": independent_output_digest,
        "selection": verification["selection"],
        "dense_projection": verification["dense_projection"],
        "transfer_aware_sensitivity": verification["transfer_aware_sensitivity"],
    }, indent=2))


if __name__ == "__main__":
    main()
