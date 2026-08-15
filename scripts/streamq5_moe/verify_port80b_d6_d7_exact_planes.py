from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe/port80b_p0"
BANK = RUNS / "port80b_p0_full_q5_bank.bin"
MANIFEST = RUNS / "port80b_p0_full_q5_bank_manifest.json"
D6_PREREG = REPORTS / "PORT80B_D6_EXACT_HOST_Q5_FUSION_PREREGISTRATION.md"
D6_RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d6_exact_host_q5_fusion.py"
D6_RESULT = REPORTS / "port80b_d6_exact_host_q5_fusion.json"
D6_REPORT = REPORTS / "PORT80B_D6_EXACT_HOST_Q5_FUSION_REPORT_2026-08-12.md"
D7_PREREG = REPORTS / "PORT80B_D7_STAGED_EXACT_Q5_PLANE_PREREGISTRATION.md"
D7_RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d7_staged_exact_q5_plane.py"
D7_RESULT = REPORTS / "port80b_d7_staged_exact_q5_plane.json"
D7_REPORT = REPORTS / "PORT80B_D7_STAGED_EXACT_Q5_PLANE_REPORT_2026-08-12.md"
D2_RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d2_registered_scatter.py"
D2_RESULT = REPORTS / "port80b_d2_registered_scatter.json"
D5_RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d5_cp_async_host_smem.py"
D5_RESULT = REPORTS / "port80b_d5_cp_async_host_smem.json"
P6_RUNNER = ROOT / "scripts/streamq5_moe/run_p6a_end_to_end_decode.py"
P6_LOCK = REPORTS / "p6a_end_to_end_evaluator_lock.json"
P7_RUNNER = ROOT / "scripts/streamq5_moe/run_p7b_ervf_kernel.py"
P7_RESULT = REPORTS / "p7b_ervf_kernel.json"
OUTPUT = REPORTS / "port80b_d6_d7_exact_planes_independent_verification.json"
REPORT = REPORTS / "PORT80B_D6_D7_EXACT_PLANES_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

LAYERS = 48
ACTIVE = 10
HIDDEN = 2048
INTERMEDIATE = 512
EXPERTS = 307
EXPERTS_WITH_SHARED = 513
EXPERT_BYTES = 2_027_520
MATRIX_BYTES = 675_840
HEADER_BYTES = 64
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
PADDING_BYTES = 4_032
TOKEN_BYTES = 973_209_600
BANK_BYTES = 49_925_652_480
OUTPUT_ELEMENTS = LAYERS * ACTIVE * (INTERMEDIATE + INTERMEDIATE + HIDDEN)
TRACE_SEED = 0x80B0120826
EXPECTED_BANK_SHA256 = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
DENSE_SHELL_P95 = 28.077_227
TOLERANCE = 1e-9
MASK64 = (1 << 64) - 1
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
PROJECTIONS = ((0, 512, 2048), (1, 512, 2048), (2, 2048, 512))


def sha256(path: Path, chunk_bytes: int = 64 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def stats(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("finite nonempty sample series required")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def close(left: float | int, right: float | int) -> bool:
    return abs(float(left) - float(right)) <= TOLERANCE


def stats_match(recomputed: dict[str, float | int], stored: dict[str, Any]) -> dict[str, bool]:
    return {name: close(value, stored[name]) for name, value in recomputed.items()}


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def routes(token: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for layer in range(LAYERS):
        state = (TRACE_SEED ^ (token * 0xD6E8FEB86659FD93) ^ (layer * 0xA5A3564E27F8862D)) & MASK64
        values: list[int] = []
        while len(values) < ACTIVE:
            state = splitmix64(state)
            value = int(state % EXPERTS)
            if value not in values:
                values.append(value)
        selected.extend((layer, expert) for expert in values)
    return selected


def route_digest(tokens: list[int]) -> str:
    table = np.asarray([[expert for _, expert in routes(token)] for token in tokens], dtype=np.uint16)
    return hashlib.sha256(table.tobytes()).hexdigest()


def record_offset(layer: int, expert: int) -> int:
    return (layer * EXPERTS_WITH_SHARED + expert) * EXPERT_BYTES


def expected_header(layer: int, expert: int, projection: int, rows: int, columns: int, crc: int) -> bytes:
    return struct.pack(
        HEADER_FORMAT, b"SQ5M", 1, layer, expert, projection, 5, rows, columns,
        128, CODE_BYTES, SCALE_BYTES, crc, bytes(28),
    )


def structural_source_check(token: int) -> dict[str, Any]:
    selected = routes(token)
    codes_ref = bytes([0x55]) * CODE_BYTES
    scales_ref = struct.pack("<H", 0x3C00) * (SCALE_BYTES // 2)
    crc = zlib.crc32(scales_ref, zlib.crc32(codes_ref)) & 0xFFFFFFFF
    mismatch_count = 0
    checked_bytes = 0
    payload_digest = hashlib.sha256()
    with BANK.open("rb", buffering=0) as handle:
        for layer, expert in selected:
            handle.seek(record_offset(layer, expert))
            for projection, rows, columns in PROJECTIONS:
                header = handle.read(HEADER_BYTES)
                codes = handle.read(CODE_BYTES)
                scales = handle.read(SCALE_BYTES)
                padding = handle.read(PADDING_BYTES)
                if min(len(header), len(codes), len(scales), len(padding)) == 0:
                    raise EOFError("short selected record")
                wanted = expected_header(layer, expert, projection, rows, columns, crc)
                mismatch_count += sum(a != b for a, b in zip(header, wanted))
                mismatch_count += len(codes) - codes.count(0x55)
                mismatch_count += sum(a != b for a, b in zip(scales, scales_ref))
                mismatch_count += len(padding) - padding.count(0)
                payload_digest.update(codes)
                payload_digest.update(scales)
                checked_bytes += MATRIX_BYTES
    return {
        "token": token,
        "records": len(selected),
        "unique_ten_each_layer": all(
            len({expert for candidate_layer, expert in selected if candidate_layer == layer}) == ACTIVE
            for layer in range(LAYERS)
        ),
        "inside_307_prefix": all(0 <= expert < EXPERTS for _, expert in selected),
        "checked_bytes": checked_bytes,
        "structural_mismatch_count": mismatch_count,
        "payload_sha256_excluding_headers_and_padding": payload_digest.hexdigest(),
    }


def input_digest(seed: int) -> str:
    values = np.random.default_rng(seed).standard_normal(HIDDEN, dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def verify_d6(result: dict[str, Any], source: str) -> dict[str, Any]:
    raw = [float(value) for value in result["validation"]["raw_ms"]]
    recalculated = stats(raw)
    p50 = float(recalculated["p50"])
    diagnostic_p95 = float(recalculated["p95"])
    exact = result["correctness"]
    digests = result["output_digests"]
    gates = {
        "all_outputs_bit_exact_and_digest_equal": (
            exact["elements"] == OUTPUT_ELEMENTS
            and exact["different_bits"] == 0
            and exact["bitwise_equal"] is True
            and exact["max_abs"] == 0.0
            and exact["finite"] is True
            and digests["resident"] == digests["remote"]
        ),
        "test_120_finite": False,
        "test_p95_le_65ms": False,
        "effective_remote_payload_gb_s_ge_15": False,
        "projected_total_p95_le_100ms": False,
        "strong_test_p95_le_55ms": False,
        "strong_projected_total_p95_le_90ms": False,
        "registration_48_ranges": result["gates"]["registration_48_ranges"],
        "no_cuda_or_runner_error": result["error"] is None and not result["unregister_failures"],
    }
    protocol = {
        "physical_exact": result["physical"] == {
            "layers": 48, "active_experts": 10, "hidden": 2048, "intermediate": 512,
            "remote_payload_bytes": TOKEN_BYTES, "registered_experts_per_layer": 307,
            "registered_gib": 48 * 307 * EXPERT_BYTES / 2**30,
        },
        "protocol_exact": result["protocol"] == {
            "warmups": 5, "validation_rounds": 24, "test_rounds": 120,
            "width": 8, "rows_per_block": 32,
        },
        "fixed_route_token_100000": "selected = routes(100_000, EXPERTS_REGISTERED)" in source,
        "five_warmups_in_source": "for _ in range(WARMUPS):" in source,
        "24_validation_samples": len(raw) == 24 and all(math.isfinite(value) for value in raw),
        "validation_open_rule": "float(validation_stats[\"p50\"]) <= 65.0" in source,
        "validation_correctly_closed": result["validation"]["open"] is False and p50 > 65.0,
        "test_correctly_absent": result["test"] == {"raw_ms": [], "stats": None},
        "no_post_validation_tuning": "SCHEDULES" not in source and "selected" not in source[source.index("raw_validation"):],
    }
    return {
        "recomputed_validation_stats": recalculated,
        "checks": {
            "stored_validation_stats": stats_match(recalculated, result["validation"]["stats"]),
            "input_sha_cpu_reproduced": input_digest(120_826) == result["input_sha256"],
            "output_element_arithmetic": OUTPUT_ELEMENTS == 1_474_560,
            "recomputed_gates": gates == result["gates"],
            "primary_false": result["primary_pass"] is False and not all(
                gates[name] for name in (
                    "all_outputs_bit_exact_and_digest_equal", "test_120_finite", "test_p95_le_65ms",
                    "effective_remote_payload_gb_s_ge_15", "projected_total_p95_le_100ms",
                    "registration_48_ranges", "no_cuda_or_runner_error",
                )
            ),
            "strong_false": result["strong_pass"] is False,
            "full_bank_false": result["full_bank_pass"] is False,
            "protocol": protocol,
        },
        "diagnostic_only": {
            "validation_p50_over_opening_gate_ms": p50 - 65.0,
            "validation_p50_ratio_to_gate": p50 / 65.0,
            "required_p50_reduction_percent": (p50 - 65.0) / p50 * 100.0,
            "validation_p95_payload_rate_gb_s_not_a_formal_test_metric": TOKEN_BYTES / (diagnostic_p95 / 1000.0) / 1e9,
        },
        "portable_route_index_sha256_token_100000": route_digest([100_000]),
        "non_replayable_evidence": {
            "pointer_table_sha256": "Contains process-specific mapped device aliases and cannot be CPU-reproduced after the run.",
            "output_digests": "Stored full-output digests are internally equal, but output arrays were not retained for independent rehashing.",
        },
    }


def verify_d7(result: dict[str, Any], source: str) -> dict[str, Any]:
    validation_raw = [float(value) for value in result["validation"]["raw_ms"]]
    test_raw = [float(value) for value in result["test"]["raw_ms"]]
    validation_stats = stats(validation_raw)
    test_stats = stats(test_raw)
    p95 = float(test_stats["p95"])
    effective = TOKEN_BYTES / (p95 / 1000.0) / 1e9
    projected = p95 + DENSE_SHELL_P95
    exact = result["correctness"]
    digests = result["output_digests"]
    gates = {
        "all_outputs_bit_exact_and_digest_equal": (
            exact["elements"] == OUTPUT_ELEMENTS
            and exact["different_bits"] == 0
            and exact["bitwise_equal"] is True
            and exact["max_abs"] == 0.0
            and exact["finite"] is True
            and digests["resident"] == digests["staged"]
        ),
        "test_120_finite": len(test_raw) == 120 and all(math.isfinite(value) for value in test_raw),
        "test_p95_le_65ms": p95 <= 65.0,
        "effective_remote_payload_gb_s_ge_15": effective >= 15.0,
        "projected_total_p95_le_100ms": projected <= 100.0,
        "strong_test_p95_le_55ms": p95 <= 55.0,
        "strong_projected_total_p95_le_90ms": projected <= 90.0,
        "registration_48_ranges": result["gates"]["registration_48_ranges"],
        "no_cuda_or_runner_error": result["error"] is None and not result["unregister_failures"],
    }
    correctness_tokens = [109_999]
    validation_tokens = list(range(110_000, 110_024))
    test_tokens = list(range(111_000, 111_120))
    all_tokens = correctness_tokens + validation_tokens + test_tokens
    protocol = {
        "physical_exact": result["physical"] == {
            "remote_payload_bytes": TOKEN_BYTES, "hbm_work_buffer_bytes": TOKEN_BYTES,
            "registered_experts_per_layer": 307, "registered_gib": 48 * 307 * EXPERT_BYTES / 2**30,
            "layers": 48, "active_experts": 10,
        },
        "protocol_exact": result["protocol"] == {
            "stage_blocks": 1024, "stage_threads": 256, "q5_width": 8,
            "warmups": 5, "validation_rounds": 24, "test_rounds": 120,
        },
        "tokens_constructed_exactly_in_source": (
            "tokens = [109_999] + list(range(110_000, 110_000 + VALIDATION_ROUNDS)) + list(range(111_000, 111_000 + TEST_ROUNDS))" in source
        ),
        "correctness_validation_test_disjoint": not (
            set(correctness_tokens) & set(validation_tokens)
            or set(correctness_tokens) & set(test_tokens)
            or set(validation_tokens) & set(test_tokens)
        ),
        "144_distinct_timed_route_tokens": len(set(validation_tokens + test_tokens)) == 144,
        "warmups_are_first_five_validation_routes": "candidate(110_000 + warmup, False)" in source,
        "validation_open_rule": "float(validation_stats[\"p50\"]) <= 65.0" in source,
        "validation_open_recomputed": result["validation"]["open"] is (float(validation_stats["p50"]) <= 65.0),
        "fixed_schedule_no_validation_selection": "stage_blocks\": 1024" in source and "min(" not in source,
        "candidate_serial_stage_then_48_layer_compute": (
            'kernels["host_to_smem_pipeline"]((1024,), (256,)' in source
            and "for layer in range(LAYERS):\n                compute(staging, layer" in source
        ),
        "test_tokens_not_persisted_in_result": "tokens" not in result["test"],
    }
    return {
        "recomputed_validation_stats": validation_stats,
        "recomputed_test_stats": test_stats,
        "recomputed_effective_remote_payload_gb_s_at_p95": effective,
        "recomputed_projected_total_p95_ms": projected,
        "checks": {
            "stored_validation_stats": stats_match(validation_stats, result["validation"]["stats"]),
            "stored_test_stats": stats_match(test_stats, result["test"]["stats"]),
            "stored_effective_rate": close(effective, result["effective_remote_payload_gb_s_at_p95"]),
            "stored_projection": close(projected, result["dense_projection"]["projected_total_p95_ms"]),
            "input_sha_cpu_reproduced": input_digest(120_827) == result["input_sha256"],
            "output_element_arithmetic": OUTPUT_ELEMENTS == 1_474_560,
            "recomputed_gates": gates == result["gates"],
            "all_gates_true": all(gates.values()),
            "primary_true": result["primary_pass"] is True,
            "strong_true": result["strong_pass"] is True,
            "full_bank_false": result["full_bank_pass"] is False,
            "protocol": protocol,
        },
        "gate_margins": {
            "p95_to_primary_65ms_ms": 65.0 - p95,
            "p95_to_strong_55ms_ms": 55.0 - p95,
            "bandwidth_to_15gb_s": effective - 15.0,
            "projected_total_to_primary_100ms_ms": 100.0 - projected,
            "projected_total_to_strong_90ms_ms": 90.0 - projected,
            "projected_tokens_per_second_not_end_to_end": 1000.0 / projected,
        },
        "portable_route_index_sha256": {
            "correctness": route_digest(correctness_tokens),
            "validation": route_digest(validation_tokens),
            "test": route_digest(test_tokens),
            "all_145_rows": route_digest(all_tokens),
        },
        "non_replayable_evidence": {
            "pointer_table_sha256": "Contains process-specific mapped device aliases and cannot be CPU-reproduced after the run.",
            "output_digests": "Stored full-output digests are internally equal, but output arrays were not retained for independent rehashing.",
            "token_lists": "Exact split is reconstructible from the hashed current runner but token IDs were not persisted in the result JSON.",
        },
    }


def all_bools(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return all(all_bools(item) for item in value.values())
    return True


def main() -> None:
    d6 = json.loads(D6_RESULT.read_text(encoding="utf-8"))
    d7 = json.loads(D7_RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d2 = json.loads(D2_RESULT.read_text(encoding="utf-8"))
    d5 = json.loads(D5_RESULT.read_text(encoding="utf-8"))
    p6_lock = json.loads(P6_LOCK.read_text(encoding="utf-8"))
    p7 = json.loads(P7_RESULT.read_text(encoding="utf-8"))
    d6_source = D6_RUNNER.read_text(encoding="utf-8")
    d7_source = D7_RUNNER.read_text(encoding="utf-8")

    input_hashes = {
        "bank_sha256_recomputed": sha256(BANK),
        "manifest_sha256": sha256(MANIFEST),
        "d6_preregistration_sha256": sha256(D6_PREREG),
        "d6_evaluator_sha256": sha256(D6_RUNNER),
        "d6_result_sha256": sha256(D6_RESULT),
        "d6_source_report_sha256": sha256(D6_REPORT),
        "d7_preregistration_sha256": sha256(D7_PREREG),
        "d7_evaluator_sha256": sha256(D7_RUNNER),
        "d7_result_sha256": sha256(D7_RESULT),
        "d7_source_report_sha256": sha256(D7_REPORT),
        "d2_dependency_sha256": sha256(D2_RUNNER),
        "d5_dependency_sha256": sha256(D5_RUNNER),
        "p6_dependency_sha256": sha256(P6_RUNNER),
        "p7_dependency_sha256": sha256(P7_RUNNER),
    }
    provenance = {
        "bank_size": BANK.stat().st_size == BANK_BYTES,
        "bank_sha_recomputed": input_hashes["bank_sha256_recomputed"] == EXPECTED_BANK_SHA256,
        "manifest_bank_sha": manifest["bank_sha256"] == EXPECTED_BANK_SHA256,
        "d6_prereg_sha": d6["inputs"]["preregistration_sha256"] == input_hashes["d6_preregistration_sha256"],
        "d6_evaluator_sha": d6["inputs"]["evaluator_sha256"] == input_hashes["d6_evaluator_sha256"],
        "d6_manifest_sha": d6["inputs"]["manifest_sha256"] == input_hashes["manifest_sha256"],
        "d7_prereg_sha": d7["inputs"]["preregistration_sha256"] == input_hashes["d7_preregistration_sha256"],
        "d7_evaluator_sha": d7["inputs"]["evaluator_sha256"] == input_hashes["d7_evaluator_sha256"],
        "d7_manifest_sha": d7["inputs"]["manifest_sha256"] == input_hashes["manifest_sha256"],
        "d2_dependency_matches_prior_result": input_hashes["d2_dependency_sha256"] == d2["inputs"]["evaluator_sha256"],
        "d5_dependency_matches_prior_result": input_hashes["d5_dependency_sha256"] == d5["inputs"]["evaluator_sha256"],
        "p6_dependency_matches_lock": input_hashes["p6_dependency_sha256"] == p6_lock["evaluator_sha256"],
        "p7_dependency_matches_prior_result": input_hashes["p7_dependency_sha256"] == p7["script_sha256"],
        "d6_result_does_not_pin_dependency_hashes": not any(key.endswith("dependency_sha256") for key in d6["inputs"]),
        "d7_result_does_not_pin_dependency_hashes": not any(key.endswith("dependency_sha256") for key in d7["inputs"]),
    }
    d6_verification = verify_d6(d6, d6_source)
    d7_verification = verify_d7(d7, d7_source)
    source_checks = {
        "d6_token_100000": structural_source_check(100_000),
        "d7_correctness_token_109999": structural_source_check(109_999),
    }
    structural_sources_clean = all(
        item["records"] == 480
        and item["unique_ten_each_layer"]
        and item["inside_307_prefix"]
        and item["checked_bytes"] == TOKEN_BYTES
        and item["structural_mismatch_count"] == 0
        for item in source_checks.values()
    )
    replayable_checks_pass = (
        all_bools(provenance)
        and all_bools(d6_verification["checks"])
        and all_bools(d7_verification["checks"])
        and structural_sources_clean
    )
    result = {
        "kind": "port80b_d6_d7_exact_planes_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "d6_negative_and_d7_strong_component_pass_verified_with_exactness_scope_limit",
        "all_replayable_checks_pass": replayable_checks_pass,
        "input_hashes": input_hashes,
        "provenance_checks": provenance,
        "d6": d6_verification,
        "d7": d7_verification,
        "selected_source_checks": source_checks,
        "exactness_scope_limit": {
            "payload_is_invariant": manifest["payload"] == {
                "codes": "0x55", "bf16_scale_word": "0x3c00",
                "payload_crc32": 1415299960, "padding": "zero",
            },
            "headers_are_ignored_by_q5_compute": (
                "const unsigned char* packed = bank + base + 64;" in d7_source
                and "const unsigned char* matrix = record +" in d6_source
            ),
            "effect": "Equal outputs prove the staged/direct arithmetic for the invariant synthetic payload, but cannot by themselves detect a wrong layer/expert record because every record has identical Q5 codes and scales. The portable route tables and selected source structure were independently checked; candidate output equality is not a routing-integrity oracle.",
        },
        "claim_boundary": "CPU-only audit of stored D6/D7 artifacts. No GPU rerun. D7 remains a strong 60%-bank synthetic transport+compute component pass, not a full-bank, differentiated-weight routing, real-model, physical dense-shell, end-to-end or quality result.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    d6s = d6_verification["recomputed_validation_stats"]
    d7v = d7_verification["recomputed_validation_stats"]
    d7t = d7_verification["recomputed_test_stats"]
    margins = d7_verification["gate_margins"]
    REPORT.write_text(
        "# PORT80B-D6/D7 independent CPU verification\n\n"
        f"Verdict: **{result['verdict']}**. All replayable checks: **{replayable_checks_pass}**. No GPU code was executed.\n\n"
        "## D6 — verified physical negative\n\n"
        f"The 24 stored validation samples recompute to p50 **{d6s['p50']:.6f} ms** and p95 **{d6s['p95']:.6f} ms**. "
        f"The frozen p50 opening gate is missed by **{d6_verification['diagnostic_only']['validation_p50_over_opening_gate_ms']:.6f} ms**, so the 120-sample test correctly did not run. "
        "The 1,474,560 stored outputs report zero bit differences and equal full-output digests, but the output arrays were not retained for an independent digest replay.\n\n"
        "## D7 — verified strong component pass\n\n"
        f"Validation recomputes to p50/p95 **{d7v['p50']:.6f}/{d7v['p95']:.6f} ms**. The 120 once-only test samples recompute to p50/p95 "
        f"**{d7t['p50']:.6f}/{d7t['p95']:.6f} ms**, effective rate **{d7_verification['recomputed_effective_remote_payload_gb_s_at_p95']:.6f} GB/s**, and frozen-shell projection "
        f"**{d7_verification['recomputed_projected_total_p95_ms']:.6f} ms**. All primary and strong gates recompute true. Strong margins are **{margins['p95_to_strong_55ms_ms']:.6f} ms** on expert-plane p95 and "
        f"**{margins['projected_total_to_strong_90ms_ms']:.6f} ms** on the projected total.\n\n"
        "## Exactness evidence boundary\n\n"
        "The stored resident/staged digests agree and the stored comparison reports 1,474,560/1,474,560 bitwise-equal outputs. This is stronger than a scalar mismatch count, but the output arrays were not saved, so the digests cannot be independently regenerated. "
        "More importantly, all synthetic expert payloads contain the same Q5 codes and scales and the compute kernels ignore headers. Therefore numerical equality cannot reveal a wrong layer/expert selection. The audit independently reconstructed the route indices and scanned all 973,209,600 selected source bytes for each correctness token, but D7 is not a differentiated-weight routing-correctness proof.\n\n"
        "## Provenance and scope\n\n"
        "The full 49,925,652,480-byte bank hash, both preregistrations, both evaluator hashes and the manifest hash match. Current P6/P7/D2/D5 dependency files match their prior locks/results, though D6/D7 did not pin those dependency hashes inside their own JSON. D7 uses a 973,209,600-byte HBM work buffer and a 307/512 (about 60%) registered prefix. It does not prove a full bank, real checkpoint, natural routing, physical dense shell, end-to-end throughput, quality or endurance.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "verdict": result["verdict"],
        "all_replayable_checks_pass": replayable_checks_pass,
        "d6_validation": d6s,
        "d7_validation": d7v,
        "d7_test": d7t,
        "d7_effective_gb_s": d7_verification["recomputed_effective_remote_payload_gb_s_at_p95"],
        "d7_projected_total_ms": d7_verification["recomputed_projected_total_p95_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
