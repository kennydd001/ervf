#!/usr/bin/env python3
"""CPU-only independent adjudication of immutable exploratory CAP0X-R2 evidence."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports/runs/streamq5_moe/het_next_cap0x_r2_intel_usm_sentinel"
COORDINATOR_RESULT = RUN / "cap0x_r2_result.json"
INTEL_RESULT = RUN / "intel_usm_sentinel.json"
NVIDIA_RESULT = RUN / "nvidia_d7.json"
INTEL_STDOUT = RUN / "intel.stdout.txt"
INTEL_STDERR = RUN / "intel.stderr.txt"
NVIDIA_STDOUT = RUN / "nvidia.stdout.txt"
NVIDIA_STDERR = RUN / "nvidia.stderr.txt"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_CAP0X_R2_INTEL_USM_SENTINEL_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md"
COORDINATOR_SOURCE = ROOT / "scripts/streamq5_moe/run_het_next_cap0x_r2_intel_usm_sentinel.py"
BASE_COORDINATOR_SOURCE = ROOT / "scripts/streamq5_moe/run_het_next_cap0x_existing_runner_diagnostic.py"
INTEL_DEPENDENCY = ROOT / "scripts/streamq5_moe/run_st2_mini_host_usm_q5.py"
NVIDIA_SOURCE = ROOT / "scripts/streamq5_moe/run_port80b_d7_staged_exact_q5_plane.py"
OUTPUT = ROOT / "reports/streamq5_moe/het_next_cap0x_r2_independent_diagnostic.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": linear_quantile(values, 0.50),
        "p95": linear_quantile(values, 0.95),
        "p99": linear_quantile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def intel_arrays() -> tuple[list[int], list[int]]:
    source: list[int] = []
    state = 0xC0A080B1
    for index in range(1024):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        source.append(state ^ ((index * 0x45D9F3B) & 0xFFFFFFFF))
    expected = [(((value ^ 0x9E3779B9) * 1664525) + 1013904223) & 0xFFFFFFFF for value in source]
    return source, expected


def word_sha(values: list[int]) -> str:
    return hashlib.sha256(struct.pack("<1024I", *values)).hexdigest()


def main() -> int:
    coordinator = json.loads(COORDINATOR_RESULT.read_text(encoding="utf-8"))
    intel = json.loads(INTEL_RESULT.read_text(encoding="utf-8"))
    nvidia = json.loads(NVIDIA_RESULT.read_text(encoding="utf-8"))
    source, expected = intel_arrays()
    input_sha = word_sha(source)
    expected_sha = word_sha(expected)

    binding_paths = {
        "coordinator_result": COORDINATOR_RESULT,
        "intel_result": INTEL_RESULT,
        "nvidia_result": NVIDIA_RESULT,
        "intel_stdout": INTEL_STDOUT,
        "intel_stderr": INTEL_STDERR,
        "nvidia_stdout": NVIDIA_STDOUT,
        "nvidia_stderr": NVIDIA_STDERR,
        "preregistration": PREREG,
        "coordinator_source": COORDINATOR_SOURCE,
        "base_coordinator_source": BASE_COORDINATOR_SOURCE,
        "intel_dependency": INTEL_DEPENDENCY,
        "nvidia_source": NVIDIA_SOURCE,
    }
    hashes = {name: sha256_file(path) for name, path in binding_paths.items()}

    intel_source_text = COORDINATOR_SOURCE.read_text(encoding="utf-8")
    intel_checks = {
        "stored_status_exact": intel.get("status") == "intel_host_usm_sentinel_exact" and intel.get("error") is None,
        "stored_shape": intel.get("launches") == 1000 and intel.get("words") == 1024,
        "source_frozen_launch_count": "LAUNCHES = 1000" in intel_source_text and "for _ in range(LAUNCHES):" in intel_source_text,
        "input_hash_recomputed": intel.get("input_sha256") == input_sha,
        "expected_hash_recomputed": intel.get("expected_sha256") == expected_sha,
        "observed_equals_expected_digest": intel.get("observed_sha256") == expected_sha,
        "stored_exactness": intel.get("correctness") == {"bitwise_equal": True, "different_bits": 0},
        "host_usm_contract": intel.get("capability") == {
            "type": 0x4197,
            "type_is_host": True,
            "size": 4096,
            "base_pointer_matches": True,
            "alignment": 4096,
        },
        "no_explicit_input_copy": intel.get("explicit_input_copy_api_calls") == 0,
        "positive_qpc_interval": int(intel.get("complete_qpc_ns", 0)) > int(intel.get("submit_qpc_ns", 0)),
    }
    intel_interval_ms = (int(intel["complete_qpc_ns"]) - int(intel["submit_qpc_ns"])) / 1e6

    validation_raw = [float(value) for value in nvidia["validation"]["raw_ms"]]
    test_raw = [float(value) for value in nvidia["test"]["raw_ms"]]
    validation_stats = stats(validation_raw)
    test_stats = stats(test_raw)
    effective = float(nvidia["physical"]["remote_payload_bytes"]) / (float(test_stats["p95"]) / 1000.0) / 1e9
    projected = float(test_stats["p95"]) + float(nvidia["dense_projection"]["frozen_dense_shell_p95_ms"])
    recomputed_gates = {
        "all_outputs_bit_exact_and_digest_equal": bool(
            nvidia["correctness"]["elements"] == 1_474_560
            and nvidia["correctness"]["different_bits"] == 0
            and nvidia["correctness"]["bitwise_equal"] is True
            and nvidia["correctness"]["max_abs"] == 0.0
            and nvidia["correctness"]["finite"] is True
            and nvidia["output_digests"]["resident"] == nvidia["output_digests"]["staged"]
        ),
        "test_120_finite": len(test_raw) == 120 and all(math.isfinite(value) for value in test_raw),
        "test_p95_le_65ms": float(test_stats["p95"]) <= 65.0,
        "effective_remote_payload_gb_s_ge_15": effective >= 15.0,
        "projected_total_p95_le_100ms": projected <= 100.0,
        "strong_test_p95_le_55ms": float(test_stats["p95"]) <= 55.0,
        "strong_projected_total_p95_le_90ms": projected <= 90.0,
        "registration_48_ranges": nvidia["gates"].get("registration_48_ranges") is True,
        "no_cuda_or_runner_error": nvidia.get("error") is None and nvidia.get("unregister_failures") == [],
    }
    nvidia_checks = {
        "protocol_counts": len(validation_raw) == 24 and len(test_raw) == 120,
        "stored_validation_stats": all(close(validation_stats[key], nvidia["validation"]["stats"][key]) for key in validation_stats),
        "stored_test_stats": all(close(test_stats[key], nvidia["test"]["stats"][key]) for key in test_stats),
        "stored_effective_rate": close(effective, nvidia["effective_remote_payload_gb_s_at_p95"]),
        "stored_projection": close(projected, nvidia["dense_projection"]["projected_total_p95_ms"]),
        "stored_gates": recomputed_gates == nvidia["gates"],
        "strong_component_pass": nvidia.get("primary_pass") is True and nvidia.get("strong_pass") is True and all(recomputed_gates.values()),
        "full_bank_not_claimed": nvidia.get("full_bank_pass") is False,
    }

    process = coordinator["processes"]
    intel_process = process["intel"]
    nvidia_process = process["nvidia"]
    interval_start = max(int(intel_process["start_qpc_ns"]), int(nvidia_process["start_qpc_ns"]))
    interval_end = min(int(intel_process["end_qpc_ns"]), int(nvidia_process["end_qpc_ns"]))
    both_alive = [
        row for row in coordinator["monitor_samples"]
        if row.get("alive", {}).get("intel") is True and row.get("alive", {}).get("nvidia") is True
    ]
    both_alive_during_intel = [
        row for row in both_alive
        if int(intel["submit_qpc_ns"]) <= int(row["qpc_ns"]) <= int(intel["complete_qpc_ns"])
    ]
    monitor_span_ms = (both_alive[-1]["qpc_ns"] - both_alive[0]["qpc_ns"]) / 1e6 if len(both_alive) >= 2 else 0.0
    process_checks = {
        "distinct_pids": intel_process["pid"] != nvidia_process["pid"],
        "strict_stored_process_interval_overlap": interval_start < interval_end,
        "monitor_confirms_both_alive": len(both_alive) > 0,
        "both_exit_zero": all(process[role].get("exit_code") == process[role].get("final_exit_code") == 0 for role in ("intel", "nvidia")),
        "no_surviving_children": all(process[role].get("alive_after_wait") is False for role in ("intel", "nvidia")),
        "stdout_hashes_bound": coordinator.get("intel_result_sha256") == hashes["intel_result"] and coordinator.get("nvidia_result_sha256") == hashes["nvidia_result"] and intel_process.get("stdout_sha256") == hashes["intel_stdout"] and nvidia_process.get("stdout_sha256") == hashes["nvidia_stdout"],
        "stderr_empty_and_bound": hashes["intel_stderr"] == hashes["nvidia_stderr"] == hashlib.sha256(b"").hexdigest(),
    }

    all_replayable = all(intel_checks.values()) and all(nvidia_checks.values()) and all(process_checks.values())
    output = {
        "kind": "het_next_cap0x_r2_independent_cpu_diagnostic",
        "verdict": "exploratory_process_lifetime_overlap_with_exact_component_results_verified",
        "all_replayable_checks_pass": all_replayable,
        "bindings": {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": hashes[name]} for name, path in binding_paths.items()},
        "intel": {
            "checks": intel_checks,
            "recomputed_input_sha256": input_sha,
            "recomputed_expected_sha256": expected_sha,
            "launches": 1000,
            "words": 1024,
            "submit_qpc_ns": intel["submit_qpc_ns"],
            "complete_qpc_ns": intel["complete_qpc_ns"],
            "interval_ms": intel_interval_ms,
        },
        "nvidia_d7": {
            "checks": nvidia_checks,
            "recomputed_validation_stats": validation_stats,
            "recomputed_test_stats": test_stats,
            "recomputed_effective_remote_payload_gb_s_at_p95": effective,
            "recomputed_projected_total_p95_ms": projected,
            "recomputed_gates": recomputed_gates,
            "correctness_elements": nvidia["correctness"]["elements"],
            "stored_equal_output_digest": nvidia["output_digests"]["resident"],
        },
        "processes": {
            "checks": process_checks,
            "stored_interval_intersection_ms": (interval_end - interval_start) / 1e6,
            "both_alive_monitor_sample_count": len(both_alive),
            "both_alive_monitor_sample_span_ms": monitor_span_ms,
            "both_alive_samples_during_intel_device_interval": len(both_alive_during_intel),
            "intel_device_interval_inside_nvidia_stored_process_interval": int(nvidia_process["start_qpc_ns"]) < int(intel["submit_qpc_ns"]) < int(intel["complete_qpc_ns"]) < int(nvidia_process["end_qpc_ns"]),
        },
        "evidence_limits": {
            "nvidia_submit_complete_qpc_absent": True,
            "kernel_or_device_work_overlap_proven": False,
            "same_process_coexistence_proven": False,
            "formal_cap0_gate_proven": False,
            "performance_or_speedup_proven": False,
            "model_or_quality_proven": False,
            "nvidia_full_output_arrays_retained": False,
            "intel_observed_output_words_retained": False,
            "intel_observed_digest_independently_rehashable": False,
            "stored_process_end_qpc_is_coordinator_observation_after_monitor_loop_not_exact_child_exit": True,
        },
        "claim_boundary": "Retrospective CPU-only verification of an exploratory run: Intel 4-KiB host-USM sentinel exactness, stored NVIDIA D7 strong synthetic component gates, and overlapping child-process lifetimes. NVIDIA submit/complete QPC was not retained, so kernel/device-work interval overlap is not proven. No formal CAP0, same-process, performance, model, quality, deployment, or breakthrough claim.",
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": all_replayable, "output": str(OUTPUT), "verdict": output["verdict"]}, sort_keys=True))
    return 0 if all_replayable else 1


if __name__ == "__main__":
    raise SystemExit(main())
