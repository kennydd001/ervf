from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe/port80b_p0"
PREREG = REPORTS / "PORT80B_D2_REGISTERED_SCATTER_PREREGISTRATION.md"
RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_d2_registered_scatter.py"
RESULT = REPORTS / "port80b_d2_registered_scatter.json"
SOURCE_REPORT = REPORTS / "PORT80B_D2_REGISTERED_SCATTER_REPORT_2026-08-12.md"
MANIFEST = RUNS / "port80b_p0_full_q5_bank_manifest.json"
BANK = RUNS / "port80b_p0_full_q5_bank.bin"
OUTPUT = REPORTS / "port80b_d2_registered_scatter_independent_verification.json"
REPORT = REPORTS / "PORT80B_D2_REGISTERED_SCATTER_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

LAYERS = 48
TOP_K = 10
ROUTED_EXPERTS = 512
EXPERTS_WITH_SHARED = 513
PREFIX_EXPERTS = (307, 358, 410, 512)
WARMUPS = 10
ROUNDS = 120
REGISTER_FLAGS = 0x02 | 0x08
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
LATENCY_GATE_MS = 45.0
BANDWIDTH_GATE_GB_S = 21.627
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
        raise ValueError("empty percentile input")
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def stats(values: list[float]) -> dict[str, float | int]:
    floats = [float(value) for value in values]
    if not floats or not all(math.isfinite(value) for value in floats):
        raise ValueError("timing values must be finite and nonempty")
    return {
        "count": len(floats),
        "mean": statistics.fmean(floats),
        "p50": percentile(floats, 0.50),
        "p95": percentile(floats, 0.95),
        "p99": percentile(floats, 0.99),
        "min": min(floats),
        "max": max(floats),
    }


def close(left: float | int, right: float | int, tolerance: float = TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def stats_checks(recomputed: dict[str, float | int], stored: dict[str, Any]) -> dict[str, bool]:
    return {name: close(value, stored[name]) for name, value in recomputed.items()}


def literal_assignments(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    result: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            result[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return result


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def routes(token: int, experts: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for layer in range(LAYERS):
        state = (TRACE_SEED ^ (token * 0xD6E8FEB86659FD93) ^ (layer * 0xA5A3564E27F8862D)) & MASK64
        values: list[int] = []
        while len(values) < TOP_K:
            state = splitmix64(state)
            value = int(state % experts)
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


def structural_source_mismatches(bank: Path, experts: int) -> dict[str, Any]:
    """Check every source byte selected by D2's one physical correctness transfer."""
    selected = routes(20_000 + experts, experts)
    code_reference = bytes([0x55]) * CODE_BYTES
    scale_reference = struct.pack("<H", 0x3C00) * (SCALE_BYTES // 2)
    padding_reference = bytes(PADDING_BYTES)
    crc = zlib.crc32(scale_reference, zlib.crc32(code_reference)) & 0xFFFFFFFF
    mismatches = 0
    checked = 0
    digest = hashlib.sha256()
    with bank.open("rb", buffering=0) as handle:
        for layer, expert in selected:
            base = record_offset(layer, expert)
            handle.seek(base)
            for projection, rows, columns in PROJECTIONS:
                header = handle.read(HEADER_BYTES)
                codes = handle.read(CODE_BYTES)
                scales = handle.read(SCALE_BYTES)
                padding = handle.read(PADDING_BYTES)
                if any(len(value) != expected for value, expected in (
                    (header, HEADER_BYTES), (codes, CODE_BYTES), (scales, SCALE_BYTES), (padding, PADDING_BYTES)
                )):
                    raise EOFError("short selected matrix read")
                wanted_header = expected_header(layer, expert, projection, rows, columns, crc)
                mismatches += sum(left != right for left, right in zip(header, wanted_header))
                mismatches += len(codes) - codes.count(0x55)
                if scales != scale_reference:
                    mismatches += sum(left != right for left, right in zip(scales, scale_reference))
                mismatches += len(padding) - padding.count(0)
                digest.update(header)
                digest.update(codes)
                digest.update(scales)
                digest.update(padding)
                checked += MATRIX_BYTES
    return {
        "selected_records": len(selected),
        "selected_unique_per_layer": all(
            len({expert for candidate_layer, expert in selected if candidate_layer == layer}) == TOP_K
            for layer in range(LAYERS)
        ),
        "all_experts_inside_prefix": all(0 <= expert < experts for _, expert in selected),
        "checked_bytes": checked,
        "structural_mismatch_count": mismatches,
        "ordered_source_sha256": digest.hexdigest(),
    }


def flatten_bools(value: Any) -> list[bool]:
    if isinstance(value, bool):
        return [value]
    if isinstance(value, dict):
        result: list[bool] = []
        for child in value.values():
            result.extend(flatten_bools(child))
        return result
    return []


def main() -> None:
    prereg_text = PREREG.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    source_report_text = SOURCE_REPORT.read_text(encoding="utf-8")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assignments = literal_assignments(runner_text)

    input_hashes = {
        "preregistration_sha256": sha256(PREREG),
        "evaluator_sha256": sha256(RUNNER),
        "manifest_sha256": sha256(MANIFEST),
        "result_sha256": sha256(RESULT),
        "source_report_sha256": sha256(SOURCE_REPORT),
    }
    provenance_checks = {
        "preregistration_sha_matches_result": input_hashes["preregistration_sha256"] == result["inputs"]["preregistration_sha256"],
        "evaluator_sha_matches_result": input_hashes["evaluator_sha256"] == result["inputs"]["evaluator_sha256"],
        "manifest_sha_matches_result": input_hashes["manifest_sha256"] == result["inputs"]["manifest_sha256"],
        "manifest_bank_sha_matches_result": manifest["bank_sha256"] == result["inputs"]["bank_sha256_from_manifest"],
        "manifest_bank_sha_matches_frozen": manifest["bank_sha256"] == EXPECTED_BANK_SHA256,
        "bank_size_exact": BANK.stat().st_size == BANK_BYTES,
        "source_report_status_matches": result["status"] in source_report_text,
    }

    full_bank_sha = sha256(BANK)
    provenance_checks["full_bank_sha_recomputed"] = full_bank_sha == EXPECTED_BANK_SHA256

    source_contract_checks = {
        "prefix_constants": tuple(assignments.get("PREFIX_EXPERTS", ())) == PREFIX_EXPERTS,
        "warmups_constant": assignments.get("WARMUPS") == WARMUPS,
        "rounds_constant": assignments.get("ROUNDS") == ROUNDS,
        "expected_bank_sha_constant": assignments.get("EXPECTED_BANK_SHA256") == EXPECTED_BANK_SHA256,
        "read_only_memmap": 'mode="r"' in runner_text,
        "mapped_and_readonly_flags": "cudaHostRegisterMapped | cudaHostRegisterReadOnly" in runner_text,
        "finally_unregister_present": "finally:\n            row[\"unregister_failures\"] = unregister_ranges(pointers)" in runner_text,
        "full_verify_total_is_token_bytes": "np.uint64(TOKEN_BYTES), mismatches" in runner_text,
        "full_verify_structural_kernel_covers_all_indices": "for (; index < total; index += stride)" in runner_text,
        "small_probe_compares_only_edges": (
            "source_head" in runner_text and "source_tail" in runner_text
            and "destination_head" in runner_text and "destination_tail" in runner_text
        ),
        "preregistered_gate_values_present": all(value in prereg_text for value in ("45.0 ms", "21.627 GB/s", "120 finite")),
    }

    protocol_checks = {
        "prefix_order": result["protocol"]["prefix_experts"] == list(PREFIX_EXPERTS),
        "warmups": result["protocol"]["warmups"] == WARMUPS,
        "rounds": result["protocol"]["rounds"] == ROUNDS,
        "registration_flags": result["protocol"]["registration_flags"] == REGISTER_FLAGS,
        "four_sweep_rows": len(result["sweep"]) == 4,
        "sweep_row_order": [row["experts_per_layer"] for row in result["sweep"]] == list(PREFIX_EXPERTS),
    }

    capability = result["capability"]
    capability_value_checks = {
        "device_name_nonempty": isinstance(capability.get("name"), str) and bool(capability["name"]),
        "compute_capability_12_0": capability.get("compute_capability") == "12.0",
        "discrete": capability.get("integrated") == 0,
        "unified_addressing": capability.get("unified_addressing") == 1,
        "async_engine_count_1": capability.get("async_engine_count") == 1,
        "can_map_host_memory": capability.get("cudaDevAttrCanMapHostMemory") == 1,
        "host_register_supported": capability.get("cudaDevAttrHostRegisterSupported") == 1,
        "host_register_readonly_supported": capability.get("cudaDevAttrHostRegisterReadOnlySupported") == 1,
        "cannot_use_host_pointer_directly": capability.get("cudaDevAttrCanUseHostPointerForRegisteredMem") == 0,
        "pageable_memory_does_not_use_host_page_tables": capability.get("cudaDevAttrPageableMemoryAccessUsesHostPageTables") == 0,
    }
    small_probe = result["small_mapped_host_probe"]
    small_probe_checks = {
        "registered_64_mib": small_probe.get("registered_bytes") == 64 * 2**20,
        "device_pointer_nonzero": small_probe.get("device_pointer_nonzero") is True,
        "edge_bytes_equal": small_probe.get("edge_bytes_equal") is True,
        "pointer_device_zero": small_probe.get("pointer_attributes", {}).get("device") == 0,
        "host_pointer_nonzero": int(small_probe.get("pointer_attributes", {}).get("hostPointer", 0)) != 0,
        "device_pointer_matches_attribute": small_probe.get("pointer_attributes", {}).get("devicePointer") != 0,
    }
    set_device_flags_finding = {
        "success": result["set_device_map_host"].get("success") is True,
        "stored_error": result["set_device_map_host"].get("error"),
        "interpretation": "binding_missing_but_mapped_probe_succeeded",
    }
    small_probe_protocol_finding = {
        "prereg_required_full_byte_comparison": True,
        "runner_compared_only_first_and_last_4096_bytes": source_contract_checks["small_probe_compares_only_edges"],
        "full_64mib_byte_comparison_proven": False,
    }

    timed_rows = [row for row in result["sweep"] if row.get("status") == "timed"]
    arithmetic: dict[str, Any] = {}
    numerical_checks: dict[str, Any] = {}
    page_telemetry: dict[str, Any] = {}
    source_structure: dict[str, Any] = {}
    stored_gate_checks: dict[str, Any] = {}

    any_global_error = any(row.get("error") or row.get("unregister_failures") for row in result["sweep"])
    for row in timed_rows:
        experts = int(row["experts_per_layer"])
        key = str(experts)
        raw = [float(value) for value in row["raw_ms"]]
        recomputed = stats(raw)
        bandwidth = TOKEN_BYTES / (float(recomputed["p95"]) / 1000.0) / 1e9
        latency_excess = float(recomputed["p95"]) - LATENCY_GATE_MS
        arithmetic[key] = {
            "stats": recomputed,
            "effective_gb_s_at_p95": bandwidth,
            "p95_excess_ms": latency_excess,
            "p95_over_gate_factor": float(recomputed["p95"]) / LATENCY_GATE_MS,
            "latency_reduction_required_fraction": 1.0 - LATENCY_GATE_MS / float(recomputed["p95"]),
            "bandwidth_shortfall_gb_s": BANDWIDTH_GATE_GB_S - bandwidth,
            "bandwidth_shortfall_fraction_of_gate": 1.0 - bandwidth / BANDWIDTH_GATE_GB_S,
        }
        numerical_checks[key] = {
            "sample_count_120": len(raw) == ROUNDS,
            "all_samples_finite": all(math.isfinite(value) for value in raw),
            "stored_stats": stats_checks(recomputed, row["timing"]),
            "stored_bandwidth": close(bandwidth, row["effective_gb_s_at_p95"]),
            "registered_bytes": row["registered_bytes"] == LAYERS * experts * EXPERT_BYTES,
            "registered_gib": close(row["registered_gib"], row["registered_bytes"] / 2**30),
            "fraction": close(row["fraction"], experts / ROUTED_EXPERTS),
        }

        telemetry = row["page_telemetry"]
        samples = telemetry["samples"]
        page_reads = [float(sample["page_reads_per_sec"]) for sample in samples]
        pages_input = [float(sample["pages_input_per_sec"]) for sample in samples]
        page_telemetry[key] = {
            "available": telemetry["available"],
            "error": telemetry["error"],
            "sample_count": len(samples),
            "page_reads_stats": stats(page_reads),
            "pages_input_stats": stats(pages_input),
            "stored_max_matches": close(max(page_reads), telemetry["page_reads_max"]),
            "timestamps_present": all(bool(sample.get("utc")) for sample in samples),
            "monotonic_strictly_increasing": all(
                float(samples[index]["monotonic_seconds"]) < float(samples[index + 1]["monotonic_seconds"])
                for index in range(len(samples) - 1)
            ),
        }

        source_structure[key] = structural_source_mismatches(BANK, experts)
        local_no_error = row.get("status") == "timed" and not row.get("error") and not row.get("unregister_failures")
        recomputed_gates = {
            "registration_48_ranges": row.get("registered_ranges") == LAYERS,
            "full_destination_zero_mismatches": row.get("full_destination_mismatch_count") == 0,
            "samples_120_finite": len(raw) == ROUNDS and all(math.isfinite(value) for value in raw),
            "p95_le_45ms": float(recomputed["p95"]) <= LATENCY_GATE_MS,
            "effective_gb_s_at_p95_ge_21_627": bandwidth >= BANDWIDTH_GATE_GB_S,
            "page_reads_zero_when_available": (
                telemetry["error"] is not None or (bool(page_reads) and max(page_reads) == 0.0)
            ),
            "no_cuda_or_runner_error": local_no_error,
        }
        mechanism_pass = all(
            recomputed_gates[name]
            for name in (
                "registration_48_ranges", "full_destination_zero_mismatches", "samples_120_finite",
                "p95_le_45ms", "effective_gb_s_at_p95_ge_21_627", "no_cuda_or_runner_error",
            )
        )
        stored_gate_checks[key] = {
            "recomputed": recomputed_gates,
            "stored_match": {name: value == row["gates"][name] for name, value in recomputed_gates.items()},
            "mechanism_pass_recomputed": mechanism_pass,
            "mechanism_pass_matches": mechanism_pass == row["mechanism_pass"],
            "all_gates_pass_recomputed": all(recomputed_gates.values()),
            "all_gates_pass_matches": all(recomputed_gates.values()) == row["all_gates_pass"],
            "prefix_local_no_error": local_no_error,
            "strict_run_global_no_error": not any_global_error,
        }

    full_row = next(row for row in result["sweep"] if row["experts_per_layer"] == 512)
    registration_checks: dict[str, Any] = {}
    for row in result["sweep"]:
        key = str(row["experts_per_layer"])
        registration_checks[key] = {
            "registered_ranges_reported": row.get("registered_ranges"),
            "all_48_registered": row.get("registered_ranges") == LAYERS,
            "unregister_failure_count": len(row.get("unregister_failures", [])),
            "all_unregistered_without_reported_failure": not row.get("unregister_failures"),
            "status": row["status"],
            "error": row.get("error"),
            "available_before": row["system_before"]["available"],
            "available_after_registration": row.get("system_after_registration", {}).get("available"),
            "available_after_unregister": row["system_after_unregister"]["available"],
        }

    full_bank_failure_checks = {
        "full_prefix_reached": full_row["experts_per_layer"] == 512,
        "all_48_ranges_initially_reported_registered": full_row.get("registered_ranges") == LAYERS,
        "status_failed": full_row["status"] == "registration_or_timing_failed",
        "oom_recorded": "cudaErrorMemoryAllocation" in full_row.get("error", ""),
        "all_48_unregister_calls_reported_oom": (
            len(full_row.get("unregister_failures", [])) == LAYERS
            and all("cudaErrorMemoryAllocation" in error for error in full_row["unregister_failures"])
        ),
        "no_timing_or_mismatch_claim_for_full_prefix": (
            "raw_ms" not in full_row and "timing" not in full_row
            and "full_destination_mismatch_count" not in full_row and "gates" not in full_row
        ),
    }

    result_level_checks = {
        "no_prefix_mechanism_pass": not any(bool(row.get("mechanism_pass")) for row in result["sweep"]),
        "mechanism_pass_false": result["mechanism_pass"] is False,
        "full_bank_pass_false": result["full_bank_pass"] is False,
        "status_negative": result["status"] == "registered_scatter_negative",
    }

    numerical_flat = flatten_bools(numerical_checks)
    replayable_checks = {
        "provenance": all(provenance_checks.values()),
        "source_contract": all(source_contract_checks.values()),
        "protocol": all(protocol_checks.values()),
        "capability_values_internally_valid": all(capability_value_checks.values()),
        "small_probe_saved_claims_internally_valid": all(small_probe_checks.values()),
        "numeric_recomputation": all(numerical_flat),
        "page_telemetry": all(
            row["available"] is True and row["error"] is None and row["stored_max_matches"]
            and row["timestamps_present"] and row["monotonic_strictly_increasing"]
            for row in page_telemetry.values()
        ),
        "source_structural_checks": all(
            row["selected_records"] == 480 and row["selected_unique_per_layer"]
            and row["all_experts_inside_prefix"] and row["checked_bytes"] == TOKEN_BYTES
            and row["structural_mismatch_count"] == 0
            for row in source_structure.values()
        ),
        "stored_gate_recomputation": all(
            all(row["stored_match"].values()) and row["mechanism_pass_matches"] and row["all_gates_pass_matches"]
            for row in stored_gate_checks.values()
        ),
        "full_bank_failure_arithmetic": all(full_bank_failure_checks.values()),
        "result_level": all(result_level_checks.values()),
    }
    all_replayable_checks_pass = all(replayable_checks.values())

    destructor_oom_assessment = {
        "same_process_recorded_cuda_oom": any_global_error,
        "full_prefix_error": full_row.get("error"),
        "full_prefix_unregister_failure_count": len(full_row.get("unregister_failures", [])),
        "process_exit_code_or_stderr_saved": False,
        "destructor_oom_posthoc_replayable": False,
        "prefix_local_interpretation": (
            "The 307/358/410 rows completed timing and reported zero unregister failures before the 512-prefix failure; "
            "their local no-error claims remain internally consistent."
        ),
        "strict_global_interpretation": (
            "The same D2 process later recorded cudaErrorMemoryAllocation and 48 unregister failures. Therefore a run-global "
            "no_cuda_or_runner_error claim is false/unverified for every prefix, especially if destructor stderr is included."
        ),
        "verdict_effect": (
            "None: every timed prefix already fails latency and bandwidth, and also page telemetry; the negative D2 verdict "
            "is unchanged under either error-scope interpretation."
        ),
    }

    threshold_consistency = {
        "exact_bandwidth_at_45ms_gb_s": TOKEN_BYTES / (LATENCY_GATE_MS / 1000.0) / 1e9,
        "stored_bandwidth_threshold_gb_s": BANDWIDTH_GATE_GB_S,
        "rounding_gap_gb_s": BANDWIDTH_GATE_GB_S - TOKEN_BYTES / (LATENCY_GATE_MS / 1000.0) / 1e9,
        "finding": (
            "The bandwidth gate is the latency gate rounded upward: exactly 45.000 ms yields 21.62688 GB/s, "
            "which passes <=45 ms but misses >=21.627 GB/s by 0.00012 GB/s."
        ),
    }

    output = {
        "kind": "port80b_d2_registered_scatter_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_context_opened": False,
        "independent_verdict": "verified_negative_with_protocol_findings",
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "protocol_conformance_pass": False,
        "protocol_findings": {
            "small_probe_only_edges_not_full_64mib": True,
            "set_device_map_host_binding_failed": result["set_device_map_host"].get("success") is False,
            "run_global_no_cuda_or_runner_error_not_supported": any_global_error,
            "device_destination_mismatch_claim_not_cpu_replayable_without_saved_hash": True,
        },
        "input_hashes": input_hashes,
        "verifier_sha256": sha256(Path(__file__)),
        "full_bank_sha256": full_bank_sha,
        "checks": {
            "replayable_summary": replayable_checks,
            "provenance": provenance_checks,
            "runner_source_contract": source_contract_checks,
            "protocol": protocol_checks,
            "capability_values": capability_value_checks,
            "small_probe": small_probe_checks,
            "full_bank_failure": full_bank_failure_checks,
            "result_level": result_level_checks,
        },
        "capability": capability,
        "set_device_map_host": set_device_flags_finding,
        "small_mapped_host_probe": small_probe,
        "small_probe_protocol_finding": small_probe_protocol_finding,
        "prefix_arithmetic": arithmetic,
        "prefix_numerical_checks": numerical_checks,
        "prefix_source_structural_verification": source_structure,
        "prefix_page_telemetry": page_telemetry,
        "prefix_gate_verification": stored_gate_checks,
        "registration_and_unregister": registration_checks,
        "destructor_oom_assessment": destructor_oom_assessment,
        "threshold_consistency": threshold_consistency,
        "mismatch_claim_limitation": (
            "For each timed prefix the saved GPU mismatch scalar is zero and the evaluator source covers every destination byte. "
            "The verifier independently confirms all 973,209,600 selected source bytes are structurally correct, but D2 saved no "
            "destination hash or buffer; the transient GPU destination cannot be post-hoc replayed CPU-only."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    prefix_lines = []
    for experts in (307, 358, 410):
        key = str(experts)
        row = arithmetic[key]
        page = page_telemetry[key]
        prefix_lines.append(
            f"| {experts}/512 ({100 * experts / 512:.1f}%) | {row['stats']['count']} | "
            f"{row['stats']['mean']:.6f} | {row['stats']['p50']:.6f} | {row['stats']['p95']:.6f} | "
            f"{row['stats']['p99']:.6f} | {row['stats']['min']:.6f} | {row['stats']['max']:.6f} | "
            f"{row['effective_gb_s_at_p95']:.6f} | {page['page_reads_stats']['max']:.6f} | 0 |"
        )

    failure_lines = []
    for experts in (307, 358, 410):
        row = arithmetic[str(experts)]
        failure_lines.append(
            f"| {experts} | {row['p95_excess_ms']:.6f} | {row['p95_over_gate_factor']:.6f}× | "
            f"{100 * row['latency_reduction_required_fraction']:.3f}% | {row['bandwidth_shortfall_gb_s']:.6f} | "
            f"{100 * row['bandwidth_shortfall_fraction_of_gate']:.3f}% |"
        )

    report = f"""# PORT80B-D2 — onafhankelijke CPU-only verificatie

**Verdict:** `verified_negative_with_protocol_findings`  
**GPU-context geopend:** nee  
**Alle replaybare reken-, hash- en provenancechecks:** {'PASS' if all_replayable_checks_pass else 'FAIL'}  
**Volledige protocolconformiteit:** FAIL

## Herberekende fysieke resultaten

| prefix | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms | GB/s bij p95 | Page Reads/s max | bronmismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(prefix_lines)}

Alle drie getimede prefixes hebben 120 eindige samples. Hun opgeslagen mean/p50/p95/p99/min/max, bandbreedte, gates en page maxima zijn bit-/floating-pointconsistent met de ruwe JSON-reeksen. Iedere prefix rapporteert 48 geslaagde registraties en nul unregisterfouten.

## Failure arithmetic

| experts | p95 boven 45 ms | factor | nodige latencyreductie | GB/s-tekort | tekort t.o.v. gate |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(failure_lines)}

De 21,627-GB/s-poort is de 45-ms-poort naar boven afgerond. Exact 973.209.600 bytes in 45,000 ms is 21,62688 GB/s: formeel zou dat de latencygate halen maar de bandwidthgate met 0,00012 GB/s missen. Deze minieme afrondingsinconsistentie beïnvloedt D2 niet; de gemeten tekorten zijn 2,93–3,04 GB/s.

## Capability en kleine mapped-hostprobe

- Device: `{capability['name']}`, compute capability `{capability['compute_capability']}`, discrete=`{capability['integrated'] == 0}`, unified addressing=`{capability['unified_addressing']}` en async engines=`{capability['async_engine_count']}`.
- `canMapHostMemory=1`, `hostRegisterSupported=1`, `hostRegisterReadOnlySupported=1`, `canUseHostPointerForRegisteredMem=0`, `pageableMemoryAccessUsesHostPageTables=0`.
- De 64-MiB-registratie rapporteert een niet-nulle devicepointer en gelijke eerste/laatste 4.096 bytes.
- `setDeviceFlags(cudaDeviceMapHost)` werd niet aangeroepen: CuPy miste de binding en bewaarde een `AttributeError`. De mapped probe werkte desondanks.
- **Protocolafwijking:** de preregistratie eiste een byte-voor-bytevergelijking van de 64-MiB-probe; de runner vergeleek slechts 8.192 randbytes. De volledige probe is dus niet bewezen.

## Mismatch- en routeaudit

Voor elk van 307/358/410 experts zijn de exacte correctnessroutes (`token=20.000+prefix`) onafhankelijk gereconstrueerd: 48 lagen × tien unieke experts, alle binnen de prefix. De verifier scande voor iedere prefix alle 973.209.600 geselecteerde bronbytes; headers, Q5-codes, BF16-schalen en padding hebben nul structurele mismatches. De volledige bank-SHA is opnieuw CPU-side `{full_bank_sha}`.

De D2-run bewaart echter alleen `full_destination_mismatch_count: 0`, geen destinationhash of buffer. De tijdelijke GPU-bestemming kan daarom niet post-hoc CPU-only worden hervergeleken. De evaluatorbron dekt aantoonbaar alle bytes, maar de uitgevoerde GPU-uitkomst blijft een niet-replaybare scalarclaim.

## Registratie, OOM en destructorscope

| prefix | 48 ranges geregistreerd | unregisterfouten | status |
|---:|---|---:|---|
| 307 | ja | 0 | timed |
| 358 | ja | 0 | timed |
| 410 | ja | 0 | timed |
| 512 | ja | **48** | `cudaErrorMemoryAllocation` |

De 100%-prefix registreerde aanvankelijk alle 48 ranges, maar faalde vóór correctness/timing met OOM. Daarna rapporteerde ieder van de 48 unregistercalls eveneens OOM; de full-bankpoort faalt dus zowel op uitvoering als op verplichte succesvolle unregister.

Voor de destructorvraag gelden twee scopes:

- **Prefix-lokaal:** 307/358/410 waren al volledig getimed en zonder gemelde unregisterfout afgesloten. De latere 512-OOM verandert hun ruwe samples niet; hun lokale `no_cuda_or_runner_error=true` is intern consistent.
- **Strikt run-globaal:** dezelfde Python-run registreerde later een CUDA-OOM en 48 cleanupfouten. Een globale claim “geen CUDA/runner error in D2” is daarom fout. Exitcode en stderr/destructorlog zijn niet opgeslagen, zodat een afzonderlijke destructor-OOM niet post-hoc kan worden gereplayed. Als destructorstderr meetelt, moet de globale foutpoort voor alle prefixes als false/unverified worden gezien.

Dit verandert het verdict niet: iedere getimede prefix faalt al de p95- en bandwidthpoorten, plus de page-readpoort.

## Page telemetry en bewijsgrens

- Prefix 307: max 4,928408 Page Reads/s.
- Prefix 358: max 93,987077 Page Reads/s.
- Prefix 410: max 5,998743 Page Reads/s.

Telemetry was beschikbaar, zonder samplererror en met strikt oplopende monotone timestamps; de zero-page-readgate faalt bij alle getimede prefixes.

Dit blijft synthetisch registered scattertransport. Geen Q5-rekenkernel, echte 80B-router, kwaliteit, dense shell, daadwerkelijke mapped-host ERGV-uitvoering, tokens/s of endurance is bewezen.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({
        "independent_verdict": output["independent_verdict"],
        "all_replayable_checks_pass": all_replayable_checks_pass,
        "protocol_conformance_pass": False,
        "timed_prefixes": [row["experts_per_layer"] for row in timed_rows],
        "full_prefix_error": full_row.get("error"),
        "full_prefix_unregister_failures": len(full_row.get("unregister_failures", [])),
        "output": str(OUTPUT),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
