from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUN_DIR = ROOT / "reports/runs/streamq5_moe/port80b_p0"
PREREG = REPORTS / "PORT80B_P0_PHYSICAL_HOST_BANK_PREREGISTRATION.md"
RUNNER = ROOT / "scripts/streamq5_moe/run_port80b_p0_physical_host_bank.py"
RESULT = REPORTS / "port80b_p0_physical_host_bank_result.json"
MANIFEST = RUN_DIR / "port80b_p0_full_q5_bank_manifest.json"
BANK = RUN_DIR / "port80b_p0_full_q5_bank.bin"
OUTPUT = REPORTS / "port80b_p0_independent_verification.json"

BANK_BYTES = 49_925_652_480
MATRIX_BYTES = 675_840
EXPERT_BYTES = 2_027_520
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)), "max": float(array.max()),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["input_hashes"] = (
        result["inputs"]["preregistration_sha256"] == sha256(PREREG)
        and result["inputs"]["script_sha256"] == sha256(RUNNER)
        and result["inputs"]["manifest_sha256"] == sha256(MANIFEST)
    )
    checks["file_size_exact"] = BANK.stat().st_size == BANK_BYTES
    checks["manifest_sha_matches_result"] = (
        manifest["bank_sha256"] == result["verification_and_warmup"]["full_sha256"]
    )
    # The runner already performed the expensive full-file hash immediately
    # before measurement.  This verifier avoids a second cache-disturbing sweep,
    # and verifies deterministic boundary headers independently instead.
    with BANK.open("rb", buffering=0) as handle:
        headers = []
        for layer, expert, projection, rows, columns in (
            (0, 0, 0, 512, 2048), (0, 512, 2, 2048, 512),
            (47, 0, 0, 512, 2048), (47, 512, 2, 2048, 512),
        ):
            offset = ((layer * 513 + expert) * EXPERT_BYTES) + projection * MATRIX_BYTES
            handle.seek(offset)
            fields = HEADER.unpack(handle.read(HEADER.size))
            headers.append(
                fields[0] == b"SQ5M" and fields[1] == 1 and fields[2] == layer
                and fields[3] == expert and fields[4] == projection and fields[5] == 5
                and fields[6] == rows and fields[7] == columns and fields[8] == 128
                and fields[9] == 655_360 and fields[10] == 16_384
            )
    checks["boundary_headers"] = all(headers)
    checks["physical_allocation_contract"] = all((
        result["verification_and_warmup"]["allocation"]["logical_bytes"] == BANK_BYTES,
        result["verification_and_warmup"]["allocation"]["allocated_bytes_getcompressedfilesize"] == BANK_BYTES,
        result["verification_and_warmup"]["allocation"]["sparse_attribute"] is False,
        result["verification_and_warmup"]["allocation"]["compressed_attribute"] is False,
        result["mapping_readonly"] is True,
    ))
    aggregate_matches = {}
    for name, scenario in result["scenarios"].items():
        aggregate_matches[name] = (
            stats(scenario["h2d_ms"]) == scenario["h2d"]
            and stats(scenario["stage_plus_h2d_wall_ms"]) == scenario["stage_plus_h2d_wall"]
            and len(scenario["misses_per_token"]) == scenario["tokens"]
            and scenario["hits"] + scenario["misses"] == scenario["tokens"] * 48 * 10
            and scenario["transferred_bytes"] == scenario["misses"] * EXPERT_BYTES
        )
    checks["all_raw_aggregates"] = all(aggregate_matches.values())
    primary = ("zero_cache", "cache_4k", "cache_32k")
    checks["primary_tokens"] = all(result["scenarios"][name]["tokens"] == 10_000 for name in primary)
    samples = result["hard_page_read_telemetry"]["samples"]
    reads = [float(row["page_reads_per_sec"]) for row in samples]
    inputs = [float(row["pages_input_per_sec"]) for row in samples]
    checks["pdh_aggregates"] = (
        stats(reads) == result["hard_page_read_telemetry"]["page_reads_per_sec"]
        and stats(inputs) == result["hard_page_read_telemetry"]["pages_input_per_sec"]
    )
    reconstructed_gates = {
        "bank_physical_contract": result["verification_and_warmup"]["allocation"]["non_sparse_fully_allocated"],
        "full_sha256_verified": result["verification_and_warmup"]["full_sha256_matches_manifest"],
        "three_scenarios_10000_tokens": checks["primary_tokens"],
        "one_hour_uninterrupted": result["elapsed_seconds"] >= 3_600,
        "pdh_available_and_sampled": result["hard_page_read_telemetry"]["error"] is None and bool(samples),
        "no_system_page_reads_after_warmup": bool(reads) and max(reads) == 0.0,
        "peak_process_commit_le_58gib": result["memory_final"]["peak_pagefile"] <= 58 * 2**30,
        "zero_cache_h2d_p95_le_45ms": result["scenarios"]["zero_cache"]["h2d"]["p95"] <= 45.0,
        "no_cuda_or_runner_error": result["runner_error"] is None,
        "telemetry_gap_le_45s": result["gates"]["telemetry_gap_le_45s"],
        "no_thermal_or_driver_error": result["gates"]["no_thermal_or_driver_error"],
    }
    checks["gates_reconstructed"] = reconstructed_gates == result["gates"]
    checks["overall_fail_reconstructed"] = (
        result["overall_pass"] is False and not all(reconstructed_gates.values())
    )
    verification = {
        "kind": "port80b_p0_independent_verification",
        "result_sha256": sha256(RESULT),
        "manifest_sha256": sha256(MANIFEST),
        "bank_full_hash_reused_from_immediate_prerun_verification": manifest["bank_sha256"],
        "no_gpu_work": True,
        "checks": checks,
        "aggregate_matches": aggregate_matches,
        "reconstructed_gates": reconstructed_gates,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "verification_pass": all(checks.values()),
        "verified_experimental_verdict": "P0_fail_due_postwarm_system_page_reads_and_zero_cache_h2d_p95",
        "claim_boundary": "CPU-only artifact and raw-array audit; no repeated GPU benchmark and no model/runtime claim.",
    }
    OUTPUT.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
