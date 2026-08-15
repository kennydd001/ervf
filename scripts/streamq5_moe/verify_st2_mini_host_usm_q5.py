from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
PRIMARY = R / "st2_mini_host_usm_q5_result.json"
WIDTH8 = R / "st2_mini_ergv_w8_result.json"
CAPABILITY = R / "st2_mini_opencl_capability_probe.json"
P1D = R / "p1d_physical_bank_result.json"
PRIMARY_PREREG = R / "ST2_MINI_PREREGISTRATION_2026-08-12.md"
WIDTH8_PREREG = R / "ST2_MINI_ERVG_W8_CONFIRMATION_PREREGISTRATION_2026-08-12.md"
PRIMARY_RUNNER = ROOT / "scripts/streamq5_moe/run_st2_mini_host_usm_q5.py"
WIDTH8_RUNNER = ROOT / "scripts/streamq5_moe/run_st2_mini_ergv_w8.py"
OUTPUT = R / "st2_mini_independent_verification.json"

RECORD_BYTES = 1_011_712
RING_RECORDS = 531
EFFECTIVE_BYTES = 1_007_616
BATCH = 16
ITERATIONS = 1_000
THRESHOLD = 21.63


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def close_dict(left: dict, right: dict, tolerance: float = 1e-12) -> bool:
    return all(abs(float(left[key]) - float(right[key])) <= tolerance for key in right)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    width8 = json.loads(WIDTH8.read_text(encoding="utf-8"))
    capability = json.loads(CAPABILITY.read_text(encoding="utf-8"))
    p1d = json.loads(P1D.read_text(encoding="utf-8"))
    checks = []

    def check(name: str, passed: bool, observed=None, expected=None) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed, "expected": expected})

    check("capability pass", capability["capability_pass"] is True, capability["capability_pass"], True)
    check("Intel selected", "Intel(R) Arc(TM) Pro 140T" in capability["selected"]["name"], capability["selected"]["name"], "Intel Arc Pro 140T")
    check("primary prereg hash", primary["preregistration_sha256"] == sha256(PRIMARY_PREREG), primary["preregistration_sha256"], sha256(PRIMARY_PREREG))
    check("width8 prereg hash", width8["preregistration_sha256"] == sha256(WIDTH8_PREREG), width8["preregistration_sha256"], sha256(WIDTH8_PREREG))
    check("capability hash primary", primary["capability_probe_sha256"] == sha256(CAPABILITY), primary["capability_probe_sha256"], sha256(CAPABILITY))
    check("primary runner hash", primary["runner_sha256"] == sha256(PRIMARY_RUNNER), primary["runner_sha256"], sha256(PRIMARY_RUNNER))
    check("width8 runner hash", width8["runner_sha256"] == sha256(WIDTH8_RUNNER), width8["runner_sha256"], sha256(WIDTH8_RUNNER))
    check("width8 imported runner hash", width8["imported_runner_sha256"] == sha256(PRIMARY_RUNNER), width8["imported_runner_sha256"], sha256(PRIMARY_RUNNER))
    check("P1D hash both", primary["p1d_manifest_sha256"] == width8["p1d_manifest_sha256"] == sha256(P1D), [primary["p1d_manifest_sha256"], width8["p1d_manifest_sha256"]], sha256(P1D))
    check("width8 binds primary result", width8["primary_st2_result_sha256"] == sha256(PRIMARY), width8["primary_st2_result_sha256"], sha256(PRIMARY))

    descriptors = []
    for layer in range(3):
        for expert in range(128):
            for projection in (0, 1):
                if len(descriptors) < RING_RECORDS:
                    descriptors.append((layer, expert, projection))
    selection_hash = hashlib.sha256(json.dumps(descriptors, separators=(",", ":")).encode("ascii")).hexdigest()
    check("selection digest both", primary["loaded"]["selection_sha256"] == width8["loaded"]["selection_sha256"] == selection_hash, [primary["loaded"]["selection_sha256"], width8["loaded"]["selection_sha256"]], selection_hash)
    check("USM content same", primary["loaded"]["usm_content_sha256"] == width8["loaded"]["usm_content_sha256"], primary["loaded"]["usm_content_sha256"], width8["loaded"]["usm_content_sha256"])
    check("ring >=512 MiB", primary["loaded"]["ring_record_bytes"] == width8["loaded"]["ring_record_bytes"] == RING_RECORDS * RECORD_BYTES and RING_RECORDS * RECORD_BYTES >= 512 * 2**20, primary["loaded"]["ring_record_bytes"], RING_RECORDS * RECORD_BYTES)
    for layer in range(3):
        expected_hash = p1d["manifests"][str(layer)]["artifact_sha256"]
        actual_hash = sha256(ROOT / p1d["manifests"][str(layer)]["artifact"])
        check(f"source layer {layer} physical SHA", primary["loaded"]["verified_layer_sha256"][str(layer)] == width8["loaded"]["verified_layer_sha256"][str(layer)] == expected_hash == actual_hash, actual_hash, expected_hash)

    for name, result in (("primary", primary), ("width8", width8)):
        allocation = result["allocation"]
        check(f"{name} host USM attested", allocation["type_is_host"] and allocation["base_pointer_matches"] and allocation["size"] == 532 * RECORD_BYTES, allocation, {"host": True, "size": 532 * RECORD_BYTES})
        audit = result["hidden_copy_audit"]
        hidden_zero = all(audit[key] == 0 for key in ("weight_cl_mem_buffers_created", "weight_enqueue_write_calls", "weight_enqueue_copy_calls", "weight_enqueue_migrate_calls")) and audit["private_weight_copy_requested"] is False and audit["host_usm_allocation_attested"] is True
        check(f"{name} no requested hidden weight copy", hidden_zero, audit, "all zero/false and host-USM true")
        rows = result["correctness"]["rows"]
        correct = len(rows) == 9 and result["correctness"]["bit_differences"] == 0 and all(row["bit_differences"] == 0 and row["expected_sha256"] == row["observed_sha256"] for row in rows)
        check(f"{name} 9 Q5 exact probes", correct, {"rows": len(rows), "diffs": result["correctness"]["bit_differences"]}, {"rows": 9, "diffs": 0})
        check(f"{name} no NVIDIA execution", result["nvidia_gpu_kernel_or_transfer_calls"] == 0, result["nvidia_gpu_kernel_or_transfer_calls"], 0)
        perf = result["performance"]
        event_ms = perf["raw_event_ms"]
        wall_ms = perf["raw_wall_ms"]
        check(f"{name} 1000 raw timings", len(event_ms) == len(wall_ms) == ITERATIONS, [len(event_ms), len(wall_ms)], [ITERATIONS, ITERATIONS])
        event_gbps = [BATCH * EFFECTIVE_BYTES / (value * 1e6) for value in event_ms]
        wall_gbps = [BATCH * EFFECTIVE_BYTES / (value * 1e6) for value in wall_ms]
        check(f"{name} event latency stats", close_dict(summary(event_ms), perf["event_ms"]), summary(event_ms), perf["event_ms"])
        check(f"{name} wall latency stats", close_dict(summary(wall_ms), perf["wall_ms"]), summary(wall_ms), perf["wall_ms"])
        check(f"{name} event bandwidth stats", close_dict(summary(event_gbps), perf["event_gbps"]), summary(event_gbps), perf["event_gbps"])
        check(f"{name} wall bandwidth stats", close_dict(summary(wall_gbps), perf["wall_gbps"]), summary(wall_gbps), perf["wall_gbps"])
        conservative = min(float(np.percentile(event_gbps, 5)), float(np.percentile(wall_gbps, 5)))
        check(f"{name} conservative recomputation", abs(conservative - perf["conservative_p95_latency_side_gbps"]) <= 1e-12, conservative, perf["conservative_p95_latency_side_gbps"])
        check(f"{name} throughput correctly fails", conservative < THRESHOLD and perf["throughput_pass"] is False, conservative, f"< {THRESHOLD}")
    primary_page = primary["performance"]["pdh"]
    check("primary PDH sampled zero", len(primary_page["samples"]) >= 1 and primary_page["all_page_reads_zero"] is True and max(row["page_reads_per_sec"] for row in primary_page["samples"]) == 0, primary_page, "at least one zero sample")
    width_page = width8["performance"]["pdh"]
    check("width8 PDH correctly not claimed", len(width_page["samples"]) == 0 and width_page["all_page_reads_zero"] is False, width_page, "no sample => no page pass")
    check("both formal statuses negative", primary["status"] == width8["status"] == "negative_throughput_or_page_gate", [primary["status"], width8["status"]], "negative_throughput_or_page_gate")

    passed = sum(row["passed"] for row in checks)
    result = {
        "kind": "streamq5_moe_st2_mini_independent_cpu_verification",
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_pass": passed == len(checks),
        "checks": checks,
        "reconstructed_verdict": "exact_host_usm_q5_pass_but_21_63_gbps_p95_side_fails_for_source_tree_and_ergv_width8",
        "primary_conservative_gbps": primary["performance"]["conservative_p95_latency_side_gbps"],
        "width8_conservative_gbps": width8["performance"]["conservative_p95_latency_side_gbps"],
        "claim_boundary": "CPU-only reconstruction from saved hashes and raw arrays; no GPU work repeated.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

