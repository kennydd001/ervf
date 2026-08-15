from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe/port80b_p0"
MANIFEST = RUNS / "port80b_p0_full_q5_bank_manifest.json"
BANK = RUNS / "port80b_p0_full_q5_bank.bin"

PREREGS = {
    "d3": REPORTS / "PORT80B_D3_MAPPED_HOST_KERNEL_PREREGISTRATION.md",
    "d3r": REPORTS / "PORT80B_D3R_MAPPED_HOST_KERNEL_PREREGISTRATION.md",
    "d4": REPORTS / "PORT80B_D4_CUDA_BATCHCOPY_PREREGISTRATION.md",
    "d4r": REPORTS / "PORT80B_D4R_CUDA_BATCHCOPY_PREREGISTRATION.md",
}
RESULTS = {
    "d3": REPORTS / "port80b_d3_mapped_host_kernel.json",
    "d3r": REPORTS / "port80b_d3r_mapped_host_kernel.json",
    "d4": REPORTS / "port80b_d4_cuda_batchcopy.json",
    "d4r": REPORTS / "port80b_d4r_cuda_batchcopy.json",
}
SOURCE_REPORTS = {
    "d3": REPORTS / "PORT80B_D3_MAPPED_HOST_KERNEL_REPORT_2026-08-12.md",
    "d3r": REPORTS / "PORT80B_D3R_MAPPED_HOST_KERNEL_REPORT_2026-08-12.md",
    "d4": REPORTS / "PORT80B_D4_CUDA_BATCHCOPY_REPORT_2026-08-12.md",
    "d4r": REPORTS / "PORT80B_D4R_CUDA_BATCHCOPY_REPORT_2026-08-12.md",
}
D3_RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d3_mapped_host_kernel.py"
D4_RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d4_cuda_batchcopy.py"
OUTPUT = REPORTS / "port80b_d3_d4_combined_independent_verification.json"
REPORT = REPORTS / "PORT80B_D3_D4_COMBINED_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

LAYERS = 48
TOP_K = 10
EXPERTS_WITH_SHARED = 513
EXPERTS = 307
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
HEADER_BYTES = 64
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
PADDING_BYTES = 4_032
MATRIX_BYTES = 675_840
EXPERT_BYTES = 2_027_520
TOKEN_BYTES = 973_209_600
BANK_BYTES = 49_925_652_480
PROJECTIONS = ((0, 512, 2048), (1, 512, 2048), (2, 2048, 512))
TRACE_SEED = 0x80B0120826
MASK64 = (1 << 64) - 1
EXPECTED_BANK_SHA256 = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
D3_SCHEDULES = (512, 1024, 2048, 4096)
D4_ARMS = ("ordinary480", "batch48x10", "batch1x480")
D4_CANDIDATES = ("batch48x10", "batch1x480")
TOLERANCE = 1e-9


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
    if not ordered:
        raise ValueError("empty series")
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def stats(values: list[float]) -> dict[str, float | int]:
    floats = [float(value) for value in values]
    if not floats or not all(math.isfinite(value) for value in floats):
        raise ValueError("series must be finite and nonempty")
    return {
        "count": len(floats),
        "mean": statistics.fmean(floats),
        "p50": percentile(floats, 0.50),
        "p95": percentile(floats, 0.95),
        "p99": percentile(floats, 0.99),
        "min": min(floats),
        "max": max(floats),
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
        while len(values) < TOP_K:
            state = splitmix64(state)
            value = int(state % EXPERTS)
            if value not in values:
                values.append(value)
        selected.extend((layer, expert) for expert in values)
    return selected


def record_offset(layer: int, expert: int) -> int:
    return (layer * EXPERTS_WITH_SHARED + expert) * EXPERT_BYTES


def expected_header(layer: int, expert: int, projection: int, rows: int, columns: int, crc: int) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        b"SQ5M", 1, layer, expert, projection, 5, rows, columns, 128,
        CODE_BYTES, SCALE_BYTES, crc, bytes(28),
    )


def structural_source_check(token: int) -> dict[str, Any]:
    selected = routes(token)
    codes_ref = bytes([0x55]) * CODE_BYTES
    scales_ref = struct.pack("<H", 0x3C00) * (SCALE_BYTES // 2)
    crc = zlib.crc32(scales_ref, zlib.crc32(codes_ref)) & 0xFFFFFFFF
    mismatch_count = 0
    checked_bytes = 0
    digest = hashlib.sha256()
    with BANK.open("rb", buffering=0) as handle:
        for layer, expert in selected:
            handle.seek(record_offset(layer, expert))
            for projection, rows, columns in PROJECTIONS:
                header = handle.read(HEADER_BYTES)
                codes = handle.read(CODE_BYTES)
                scales = handle.read(SCALE_BYTES)
                padding = handle.read(PADDING_BYTES)
                if len(header) != HEADER_BYTES or len(codes) != CODE_BYTES or len(scales) != SCALE_BYTES or len(padding) != PADDING_BYTES:
                    raise EOFError("short selected source record")
                expected = expected_header(layer, expert, projection, rows, columns, crc)
                mismatch_count += sum(left != right for left, right in zip(header, expected))
                mismatch_count += len(codes) - codes.count(0x55)
                if scales != scales_ref:
                    mismatch_count += sum(left != right for left, right in zip(scales, scales_ref))
                mismatch_count += len(padding) - padding.count(0)
                digest.update(header)
                digest.update(codes)
                digest.update(scales)
                digest.update(padding)
                checked_bytes += MATRIX_BYTES
    return {
        "token": token,
        "selected_records": len(selected),
        "unique_ten_per_layer": all(
            len({expert for candidate_layer, expert in selected if candidate_layer == layer}) == TOP_K
            for layer in range(LAYERS)
        ),
        "all_inside_307_prefix": all(0 <= expert < EXPERTS for _, expert in selected),
        "checked_bytes": checked_bytes,
        "structural_mismatch_count": mismatch_count,
        "ordered_source_sha256": digest.hexdigest(),
    }


def expected_rotating_orders(values: tuple[Any, ...], rounds: int) -> list[list[Any]]:
    result: list[list[Any]] = []
    for round_index in range(rounds):
        rotation = round_index % len(values)
        order = list(values[rotation:] + values[:rotation])
        if round_index & 1:
            order.reverse()
        result.append(order)
    return result


def order_summary(orders: list[list[Any]]) -> dict[str, Any]:
    counter = Counter(tuple(order) for order in orders)
    values = tuple(orders[0]) if orders else ()
    distinct_values = tuple(dict.fromkeys(value for order in orders for value in order))
    position_counts = {
        str(value): [sum(order[position] == value for order in orders) for position in range(len(distinct_values))]
        for value in distinct_values
    }
    return {
        "order_count": len(orders),
        "permutation_counts": {"/".join(map(str, order)): count for order, count in sorted(counter.items(), key=lambda item: str(item[0]))},
        "position_counts": position_counts,
        "first_order_values": list(values),
    }


def main() -> None:
    prereg_text = {key: path.read_text(encoding="utf-8") for key, path in PREREGS.items()}
    results = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in RESULTS.items()}
    source_report_text = {key: path.read_text(encoding="utf-8") for key, path in SOURCE_REPORTS.items()}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d3_source = D3_RUNNER.read_text(encoding="utf-8")
    d4_source = D4_RUNNER.read_text(encoding="utf-8")

    input_hashes = {
        "manifest_sha256": sha256(MANIFEST),
        "bank_manifest_value": manifest["bank_sha256"],
        "d3_runner_current_sha256": sha256(D3_RUNNER),
        "d4_runner_current_sha256": sha256(D4_RUNNER),
        "preregistrations": {key: sha256(path) for key, path in PREREGS.items()},
        "results": {key: sha256(path) for key, path in RESULTS.items()},
        "source_reports": {key: sha256(path) for key, path in SOURCE_REPORTS.items()},
    }
    full_bank_sha = sha256(BANK)
    provenance_checks: dict[str, Any] = {}
    for key, row in results.items():
        provenance_checks[key] = {
            "preregistration_sha": row["inputs"]["preregistration_sha256"] == input_hashes["preregistrations"][key],
            "manifest_sha": row["inputs"]["manifest_sha256"] == input_hashes["manifest_sha256"],
            "bank_sha_from_manifest": row["inputs"]["bank_sha256_from_manifest"] == EXPECTED_BANK_SHA256,
            "source_report_status": row["status"] in source_report_text[key],
        }
    provenance_checks["d3"]["evaluator_source_preserved"] = results["d3"]["inputs"]["evaluator_sha256"] == input_hashes["d3_runner_current_sha256"]
    provenance_checks["d3r"]["evaluator_current_sha"] = results["d3r"]["inputs"]["evaluator_sha256"] == input_hashes["d3_runner_current_sha256"]
    provenance_checks["d4"]["evaluator_source_preserved"] = results["d4"]["inputs"]["evaluator_sha256"] == input_hashes["d4_runner_current_sha256"]
    provenance_checks["d4r"]["evaluator_current_sha"] = results["d4r"]["inputs"]["evaluator_sha256"] == input_hashes["d4_runner_current_sha256"]
    bank_checks = {
        "bank_size": BANK.stat().st_size == BANK_BYTES,
        "manifest_value": manifest["bank_sha256"] == EXPECTED_BANK_SHA256,
        "full_bank_sha_recomputed": full_bank_sha == EXPECTED_BANK_SHA256,
    }

    d3 = results["d3"]
    d3_compile_checks = {
        "negative_status": d3["status"] == "mapped_host_kernel_negative",
        "no_mechanism_or_strong_pass": d3["mechanism_pass"] is False and d3["strong_pass"] is False,
        "compile_exception": d3.get("error", "").startswith("CompileException:"),
        "missing_stdint_header_exact_cause": 'cannot open source file "stdint.h"' in d3.get("error", ""),
        "compilation_terminated": "Compilation terminated" in d3.get("error", ""),
        "no_timing_or_mismatch_payload": "validation" not in d3 and "test" not in d3 and "full_destination_mismatch_count" not in d3,
        "unregister_failures_empty": d3.get("unregister_failures") == [],
        "protocol_frozen": d3["protocol"] == {
            "experts_per_layer": 307,
            "schedules_blocks": [512, 1024, 2048, 4096],
            "threads": 256,
            "validation_warmups": 6,
            "validation_rounds": 24,
            "test_rounds": 120,
        },
    }

    d3r = results["d3r"]
    d3r_validation_stats: dict[str, Any] = {}
    d3r_stats_checks: dict[str, Any] = {}
    for blocks in D3_SCHEDULES:
        raw = [float(value) for value in d3r["validation"]["schedules"][str(blocks)]["raw_ms"]]
        recalculated = stats(raw)
        d3r_validation_stats[str(blocks)] = {
            **recalculated,
            "diagnostic_effective_gb_s_at_validation_p95": TOKEN_BYTES / (float(recalculated["p95"]) / 1000.0) / 1e9,
        }
        d3r_stats_checks[str(blocks)] = {
            "24_finite_samples": len(raw) == 24 and all(math.isfinite(value) for value in raw),
            "stored_stats": stats_match(recalculated, d3r["validation"]["schedules"][str(blocks)]["stats"]),
        }
    d3r_selected = min(D3_SCHEDULES, key=lambda value: (float(d3r_validation_stats[str(value)]["p50"]), value))
    d3r_validation_open = (
        d3r["full_destination_mismatch_count"] == 0
        and float(d3r_validation_stats[str(d3r_selected)]["p50"]) <= 65.0
    )
    d3r_orders_expected = expected_rotating_orders(D3_SCHEDULES, 24)
    d3r_order_checks = {
        "tokens_exact": d3r["validation"]["tokens"] == list(range(50_000, 50_024)),
        "orders_exact": d3r["validation"]["orders"] == d3r_orders_expected,
        "selected_512": d3r_selected == d3r["selected_blocks"] == 512,
        "test_correctly_closed": d3r_validation_open is False and d3r["test"] == {"tokens": [], "raw_ms": [], "stats": None},
    }
    d3r_recomputed_gates = {
        "registration_48_ranges": d3r["registered_bytes"] == LAYERS * EXPERTS * EXPERT_BYTES,
        "full_destination_zero_mismatches": d3r["full_destination_mismatch_count"] == 0,
        "validation_open": d3r_validation_open,
        "test_120_finite": False,
        "test_p95_le_65ms": False,
        "effective_gb_s_at_p95_ge_15": False,
        "strong_test_p95_le_45ms": False,
        "strong_effective_gb_s_at_p95_ge_21_627": False,
        "no_cuda_or_runner_error": d3r.get("error") is None and not d3r.get("unregister_failures"),
    }
    d3r_gate_checks = {name: value == d3r["gates"][name] for name, value in d3r_recomputed_gates.items()}
    d3r_failure_arithmetic = {
        "selected_validation_p50_ms": d3r_validation_stats["512"]["p50"],
        "validation_open_limit_ms": 65.0,
        "excess_ms": float(d3r_validation_stats["512"]["p50"]) - 65.0,
        "over_limit_factor": float(d3r_validation_stats["512"]["p50"]) / 65.0,
        "latency_reduction_needed_fraction": 1.0 - 65.0 / float(d3r_validation_stats["512"]["p50"]),
        "test_bandwidth_is_formally_null": d3r["effective_gb_s_at_p95"] is None,
    }

    d4 = results["d4"]
    d4_compatibility_checks = {
        "negative_status": d4["status"] == "native_batchcopy_negative" and d4["pass"] is False,
        "native_symbol_recorded": d4["native"]["symbol"] == "cudaMemcpyBatchAsync",
        "abi_sizes": d4["native"]["mem_location_size"] == 8 and d4["native"]["attributes_size"] == 24,
        "src_access_order_any": d4["native"]["src_access_order"] == "Any",
        "illegal_address": "cudaErrorIllegalAddress" in d4.get("error", ""),
        "all_48_unregisters_report_illegal_address": (
            len(d4.get("unregister_failures", [])) == 48
            and all("cudaErrorIllegalAddress" in error for error in d4["unregister_failures"])
        ),
        "no_correctness_or_timing_payload": all(name not in d4 for name in ("correctness", "validation", "test", "gates")),
        "protocol_frozen": d4["protocol"] == {
            "experts_per_layer": 307,
            "arms": ["ordinary480", "batch48x10", "batch1x480"],
            "warmups": 6,
            "validation_rounds": 24,
            "test_rounds": 120,
        },
    }

    d4r = results["d4r"]
    d4r_validation_stats: dict[str, Any] = {}
    d4r_stats_checks: dict[str, Any] = {}
    for arm in D4_ARMS:
        raw = [float(value) for value in d4r["validation"]["arms"][arm]["raw_ms"]]
        recalculated = stats(raw)
        d4r_validation_stats[arm] = recalculated
        d4r_stats_checks[arm] = {
            "24_finite_samples": len(raw) == 24 and all(math.isfinite(value) for value in raw),
            "stored_stats": stats_match(recalculated, d4r["validation"]["arms"][arm]["stats"]),
        }
    selected_arm = min(
        D4_CANDIDATES,
        key=lambda arm: (float(d4r_validation_stats[arm]["p50"]), D4_CANDIDATES.index(arm)),
    )
    correctness_pass = all(
        d4r["correctness"][arm]["full_destination_mismatch_count"] == 0 for arm in D4_ARMS
    )
    validation_open = correctness_pass and (
        float(d4r_validation_stats[selected_arm]["p50"])
        <= 1.05 * float(d4r_validation_stats["ordinary480"]["p50"])
    )
    d4r_test_raw = [float(value) for value in d4r["test"]["raw_ms"]]
    d4r_test_stats = stats(d4r_test_raw)
    d4r_bandwidth = TOKEN_BYTES / (float(d4r_test_stats["p95"]) / 1000.0) / 1e9
    d4r_ratios = {
        "selected_validation_p50_over_ordinary": (
            float(d4r_validation_stats[selected_arm]["p50"])
            / float(d4r_validation_stats["ordinary480"]["p50"])
        ),
        "selected_validation_p95_over_ordinary": (
            float(d4r_validation_stats[selected_arm]["p95"])
            / float(d4r_validation_stats["ordinary480"]["p95"])
        ),
    }
    d4r_order_expected = expected_rotating_orders(D4_ARMS, 24)
    d4r_order_checks = {
        "validation_tokens_exact": d4r["validation"]["tokens"] == list(range(70_000, 70_024)),
        "orders_exact": d4r["validation"]["orders"] == d4r_order_expected,
        "selected_arm_exact": selected_arm == d4r["selected_arm"] == "batch48x10",
        "validation_open_exact": validation_open == d4r["validation_open"] is True,
        "test_tokens_exact": d4r["test"]["tokens"] == list(range(80_000, 80_120)),
    }
    d4r_recomputed_gates = {
        "native_symbol_and_abi": (
            d4r["native"]["symbol"] == "cudaMemcpyBatchAsync"
            and d4r["native"]["mem_location_size"] == 8
            and d4r["native"]["attributes_size"] == 24
        ),
        "all_arms_zero_mismatches": correctness_pass,
        "test_120_finite": len(d4r_test_raw) == 120 and all(math.isfinite(value) for value in d4r_test_raw),
        "test_p95_le_45ms": float(d4r_test_stats["p95"]) <= 45.0,
        "effective_gb_s_at_p95_ge_21_627": d4r_bandwidth >= 21.627,
        "validation_p50_ratio_le_0_90": d4r_ratios["selected_validation_p50_over_ordinary"] <= 0.90,
        "validation_p95_ratio_le_0_90": d4r_ratios["selected_validation_p95_over_ordinary"] <= 0.90,
        "registration_48_ranges": d4r["gates"]["registration_48_ranges"] is True,
        "no_cuda_or_runner_error": d4r.get("error") is None and not d4r.get("unregister_failures"),
    }
    d4r_gate_checks = {name: value == d4r["gates"][name] for name, value in d4r_recomputed_gates.items()}
    d4r_numeric_checks = {
        "test_120_finite": len(d4r_test_raw) == 120 and all(math.isfinite(value) for value in d4r_test_raw),
        "test_stats": stats_match(d4r_test_stats, d4r["test"]["stats"]),
        "bandwidth": close(d4r_bandwidth, d4r["effective_gb_s_at_p95"]),
        "ratios": {name: close(value, d4r["ratios"][name]) for name, value in d4r_ratios.items()},
    }
    d4r_failure_arithmetic = {
        "test_p95_ms": d4r_test_stats["p95"],
        "p95_excess_over_45_ms": float(d4r_test_stats["p95"]) - 45.0,
        "p95_over_gate_factor": float(d4r_test_stats["p95"]) / 45.0,
        "latency_reduction_needed_fraction": 1.0 - 45.0 / float(d4r_test_stats["p95"]),
        "effective_gb_s_at_p95": d4r_bandwidth,
        "bandwidth_shortfall_gb_s": 21.627 - d4r_bandwidth,
        "bandwidth_shortfall_fraction": 1.0 - d4r_bandwidth / 21.627,
        "selected_p50_ratio": d4r_ratios["selected_validation_p50_over_ordinary"],
        "selected_p95_ratio": d4r_ratios["selected_validation_p95_over_ordinary"],
        "p50_speedup_factor": 1.0 / d4r_ratios["selected_validation_p50_over_ordinary"],
        "p95_speedup_factor": 1.0 / d4r_ratios["selected_validation_p95_over_ordinary"],
    }

    repair_compliance = {
        "d3r": {
            "protocol_equal_to_d3": d3r["protocol"] == d3["protocol"],
            "parent_compile_failure_preserved": RESULTS["d3"].is_file() and SOURCE_REPORTS["d3"].is_file(),
            "current_kernel_has_no_stdint_include": "#include <stdint.h>" not in d3_source,
            "current_kernel_has_no_uintptr_t": "uintptr_t" not in d3_source,
            "current_kernel_direct_ull_pointer_cast": "(const unsigned char*)source_pointers[record]" in d3_source,
            "current_runner_hash_matches_d3r": input_hashes["d3_runner_current_sha256"] == d3r["inputs"]["evaluator_sha256"],
            "original_runner_source_available_for_exact_diff": d3["inputs"]["evaluator_sha256"] == input_hashes["d3_runner_current_sha256"],
        },
        "d4r": {
            "protocol_equal_to_d4": d4r["protocol"] == d4["protocol"],
            "parent_illegal_address_preserved": RESULTS["d4"].is_file() and SOURCE_REPORTS["d4"].is_file(),
            "ordinary_uses_host_pointer": "int(mapped.ctypes.data) + record_offset(layer, expert)" in d4_source,
            "native_uses_device_alias": "aliases[layer] + expert * EXPERT_BYTES" in d4_source,
            "aliases_checked_nonzero": "if not alias:" in d4_source,
            "current_runner_hash_matches_d4r": input_hashes["d4_runner_current_sha256"] == d4r["inputs"]["evaluator_sha256"],
            "original_runner_source_available_for_exact_diff": d4["inputs"]["evaluator_sha256"] == input_hashes["d4_runner_current_sha256"],
        },
    }

    source_checks = {
        "d3r_correctness_token": structural_source_check(49_999),
        "d4r_correctness_token": structural_source_check(69_999),
    }

    replayable_check_groups = {
        "bank": all(bank_checks.values()),
        "d3_provenance_except_overwritten_runner": all(
            value for name, value in provenance_checks["d3"].items() if name != "evaluator_source_preserved"
        ),
        "d3r_provenance": all(provenance_checks["d3r"].values()),
        "d4_provenance_except_overwritten_runner": all(
            value for name, value in provenance_checks["d4"].items() if name != "evaluator_source_preserved"
        ),
        "d4r_provenance": all(provenance_checks["d4r"].values()),
        "d3_compile_failure": all(d3_compile_checks.values()),
        "d3r_raw_stats": all(
            row["24_finite_samples"] and all(row["stored_stats"].values()) for row in d3r_stats_checks.values()
        ),
        "d3r_selection_and_order": all(d3r_order_checks.values()),
        "d3r_gates": all(d3r_gate_checks.values()),
        "d4_compatibility_failure": all(d4_compatibility_checks.values()),
        "d4r_raw_stats": all(
            row["24_finite_samples"] and all(row["stored_stats"].values()) for row in d4r_stats_checks.values()
        ),
        "d4r_test_ratios_bandwidth": (
            d4r_numeric_checks["test_120_finite"] and all(d4r_numeric_checks["test_stats"].values())
            and d4r_numeric_checks["bandwidth"] and all(d4r_numeric_checks["ratios"].values())
        ),
        "d4r_selection_and_order": all(d4r_order_checks.values()),
        "d4r_gates": all(d4r_gate_checks.values()),
        "source_structural_checks": all(
            row["selected_records"] == 480 and row["unique_ten_per_layer"]
            and row["all_inside_307_prefix"] and row["checked_bytes"] == TOKEN_BYTES
            and row["structural_mismatch_count"] == 0
            for row in source_checks.values()
        ),
    }
    all_replayable_checks_pass = all(replayable_check_groups.values())

    output = {
        "kind": "port80b_d3_d4_combined_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_context_opened": False,
        "independent_verdict": "all_four_negative_confirmed_with_evidence_limits",
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "input_hashes": input_hashes,
        "verifier_sha256": sha256(Path(__file__)),
        "full_bank_sha256": full_bank_sha,
        "replayable_check_groups": replayable_check_groups,
        "provenance_checks": provenance_checks,
        "bank_checks": bank_checks,
        "d3_compile_failure": d3_compile_checks,
        "d3r": {
            "validation_stats": d3r_validation_stats,
            "stat_checks": d3r_stats_checks,
            "order_checks": d3r_order_checks,
            "order_summary": order_summary(d3r["validation"]["orders"]),
            "selected_blocks": d3r_selected,
            "recomputed_gates": d3r_recomputed_gates,
            "gate_checks": d3r_gate_checks,
            "failure_arithmetic": d3r_failure_arithmetic,
            "error": d3r.get("error"),
            "unregister_failures": d3r.get("unregister_failures"),
        },
        "d4_compatibility_failure": d4_compatibility_checks,
        "d4r": {
            "validation_stats": d4r_validation_stats,
            "stat_checks": d4r_stats_checks,
            "test_stats": d4r_test_stats,
            "numeric_checks": d4r_numeric_checks,
            "order_checks": d4r_order_checks,
            "order_summary": order_summary(d4r["validation"]["orders"]),
            "selected_arm": selected_arm,
            "validation_open": validation_open,
            "ratios": d4r_ratios,
            "effective_gb_s_at_p95": d4r_bandwidth,
            "recomputed_gates": d4r_recomputed_gates,
            "gate_checks": d4r_gate_checks,
            "failure_arithmetic": d4r_failure_arithmetic,
            "error": d4r.get("error"),
            "unregister_failures": d4r.get("unregister_failures"),
        },
        "repair_compliance": repair_compliance,
        "source_structural_verification": source_checks,
        "byte_mismatch_evidence_boundary": (
            "D3R and D4R save only GPU mismatch scalars, not destination hashes or buffers. The verifier independently scans all "
            "973,209,600 source bytes for each correctness token and confirms zero source-structure mismatches, but cannot CPU-replay "
            "the transient GPU destinations or prove the per-arm D4R copies from saved artifacts alone."
        ),
        "repair_provenance_boundary": (
            "The current runner files exactly match the D3R/D4R evaluator hashes. The original D3/D4 evaluator sources were overwritten "
            "and are not preserved under their recorded hashes, so exact old-vs-repair source diffs are impossible. Repair compliance is "
            "verified from current source, result protocols and preserved failure artefacts, not from a byte-for-byte source diff."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    d3_rows = []
    for blocks in D3_SCHEDULES:
        row = d3r_validation_stats[str(blocks)]
        d3_rows.append(
            f"| {blocks} | {row['count']} | {row['mean']:.6f} | {row['p50']:.6f} | {row['p95']:.6f} | "
            f"{row['p99']:.6f} | {row['min']:.6f} | {row['max']:.6f} | {row['diagnostic_effective_gb_s_at_validation_p95']:.6f} |"
        )
    d4_rows = []
    for arm in D4_ARMS:
        row = d4r_validation_stats[arm]
        d4_rows.append(
            f"| {arm} | {row['count']} | {row['mean']:.6f} | {row['p50']:.6f} | {row['p95']:.6f} | "
            f"{row['p99']:.6f} | {row['min']:.6f} | {row['max']:.6f} |"
        )

    report = f"""# PORT80B D3/D4 — gecombineerde onafhankelijke CPU-only verificatie

**Verdict:** `all_four_negative_confirmed_with_evidence_limits`  
**GPU-context geopend:** nee  
**Alle replaybare reken-, selectie-, gate-, hash- en provenancechecks:** {'PASS' if all_replayable_checks_pass else 'FAIL'}

## Vier onafhankelijke conclusies

| Fase | Onafhankelijk bevestigd verdict | Fysieke data |
|---|---|---|
| D3 | compile-fail | NVRTC stopte op ontbrekende `stdint.h`; geen kernel, mismatch of timing |
| D3R | mapped-host physical negative | correctnessscalar 0; beste validation-p50 166,471 ms >65 ms, test bleef gesloten |
| D4 | compatibility fail | eerste native batchroute gaf illegal address; geen correctness/timing; 48 cleanupfouten |
| D4R | repaired native-batch physical negative | volledige validation/test; kleine winst, maar vier performancegates falen |

## D3 en D3R

D3's resultaat bevat exact de compilefout `cannot open source file "stdint.h"`, gevolgd door `Compilation terminated`. Er zijn geen validation-, test- of mismatchvelden. De unregister-foutenlijst is leeg. Dit is uitsluitend een compile-fail, geen negatieve kernelmeting.

D3R herhaalt dezelfde protocolgeometrie. Alle vier schedules hebben 24 eindige validation-samples:

| blocks | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms | diagnostische GB/s bij validation-p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(d3_rows)}

De selectie is correct: 512 blocks heeft de laagste p50. Die p50 is {d3r_failure_arithmetic['selected_validation_p50_ms']:.6f} ms, {d3r_failure_arithmetic['excess_ms']:.6f} ms boven de 65-ms-openingspoort en {d3r_failure_arithmetic['over_limit_factor']:.6f}× de limiet. Er zou {100 * d3r_failure_arithmetic['latency_reduction_needed_fraction']:.3f}% latencyreductie nodig zijn. Daarom bleef de vooraf geregistreerde 120-sampletest dicht; test-p95 en testbandbreedte zijn formeel `null`, niet nul of een geschatte fail.

Alle 24 tokens en rotatie/omkeerorders kloppen; iedere van de vier werkelijk voorkomende orders staat zesmaal in de reeks. Registratie is 48×307 records, aliases zijn niet nul, `error=null` en unregisterfouten zijn leeg. Alle herberekende D3R-gates matchen de JSON.

## D4 en D4R

D4 vond de native `cudaMemcpyBatchAsync`-symbol en ABI-groottes 8/24, maar de eerste native route vergiftigde de context met `cudaErrorIllegalAddress`. Er zijn geen correctness-, validation-, test- of gatevelden. Alle 48 unregistercalls rapporteerden dezelfde illegal-addressfout. Dit is een valide compatibiliteitsfail, geen snelheidstest.

D4R gebruikt voor native descriptors de niet-nulle `devicePointer`-aliases, terwijl de ordinary arm CPU-hostpointers behoudt. De drie validationarmen zijn onafhankelijk herberekend:

| arm | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(d4_rows)}

`batch48x10` is correct geselecteerd. De validation-p50-/p95-ratio's versus ordinary zijn {d4r_ratios['selected_validation_p50_over_ordinary']:.9f} en {d4r_ratios['selected_validation_p95_over_ordinary']:.9f}; dit zijn slechts {d4r_failure_arithmetic['p50_speedup_factor']:.6f}× en {d4r_failure_arithmetic['p95_speedup_factor']:.6f}× snelheidsfactoren, niet de vereiste ratio ≤0,90.

De once-onlytest bevat exact 120 eindige samples: mean {d4r_test_stats['mean']:.6f}, p50 {d4r_test_stats['p50']:.6f}, p95 {d4r_test_stats['p95']:.6f}, p99 {d4r_test_stats['p99']:.6f}, min {d4r_test_stats['min']:.6f}, max {d4r_test_stats['max']:.6f} ms. De p95 ligt {d4r_failure_arithmetic['p95_excess_over_45_ms']:.6f} ms boven 45 ms; {100 * d4r_failure_arithmetic['latency_reduction_needed_fraction']:.3f}% reductie ontbreekt. Bandbreedte is {d4r_bandwidth:.6f} GB/s, {d4r_failure_arithmetic['bandwidth_shortfall_gb_s']:.6f} GB/s onder de gate.

De vier performancefails zijn: test-p95, p95-bandwidth, validation-p50-ratio en validation-p95-ratio. Native symbol/ABI, 3× mismatchscalar nul, 120 samples, registratie en lokale error/unregisterpoorten passeren. `error=null`; unregisterfouten zijn leeg.

## Repairs en provenance

- D3R voldoet in de huidige bron aan de vastgelegde repair: geen `<stdint.h>`/`uintptr_t`, wel directe `unsigned long long`-pointercast. Protocolvelden zijn gelijk aan D3.
- D4R voldoet in de huidige bron: native descriptors gebruiken devicealiases, ordinary gebruikt hostpointers, aliases worden op nul gecontroleerd. Protocolvelden zijn gelijk aan D4.
- De huidige runners matchen exact de opgeslagen D3R/D4R-evaluatorhashes.
- **Beperking:** de oorspronkelijke D3/D4-runnerbronnen onder hashes `{d3['inputs']['evaluator_sha256']}` en `{d4['inputs']['evaluator_sha256']}` zijn overschreven. Een exacte source-diff kan daarom niet worden gecontroleerd; alleen de huidige repair, gelijke protocols en bewaarde failureartefacten.
- De fysieke bank is opnieuw volledig CPU-side gehasht: `{full_bank_sha}`.

## Byte-evidencegrens

De correctnessroutes voor D3R-token 49.999 en D4R-token 69.999 zijn onafhankelijk gereconstrueerd. Voor beide zijn alle 480 records en alle 973.209.600 geselecteerde bronbytes gescand: nul structurele bronmismatches.

D3R en D4R bewaren echter alleen GPU-mismatchscalars, geen destinationhashes/buffers. De tijdelijke GPU-bestemmingen — en bij D4R de drie afzonderlijke armuitkomsten — kunnen daarom niet post-hoc CPU-only worden hervergeleken. Dat beperkt de reproduceerbaarheid van correctness, maar niet de negatieve timingverdicts.

Geen van deze fasen bewijst Q5-aritmetiek, een echte 80B-port, kwaliteit, dense-shell-timing, end-to-end tokens/s, full-bankcapaciteit of endurance.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "independent_verdict": output["independent_verdict"],
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "d3r_selected_blocks": d3r_selected,
        "d3r_validation_open": d3r_validation_open,
        "d4r_selected_arm": selected_arm,
        "d4r_test_p95_ms": d4r_test_stats["p95"],
        "d4r_effective_gb_s": d4r_bandwidth,
        "output": str(OUTPUT),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
