from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
RAW = R / "port80b_d10br_heldout_10000_endurance_revision.json"
RAW_REPORT = R / "PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_REPORT_2026-08-13.md"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10br_heldout_10000_endurance_revision.py"
PREREG = R / "PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_PREREGISTRATION.md"
PREFLIGHT = R / "port80b_d10br_heldout_10000_endurance_revision_preflight.json"
OUT = R / "port80b_d10br_heldout_10000_endurance_independent_verification.json"
REPORT = R / "PORT80B_D10BR_HELDOUT_10000_ENDURANCE_INDEPENDENT_VERIFICATION_REPORT_2026-08-13.md"

EXPECTED = {
    "raw": "f4e01b4aa810f26a09835bfa8dd2b7e9153be432523841a35ac1fc7d6671fc9a",
    "raw_report": "c5cbb93e15928a5850cdfdb6cb19a308d3c13da72bf83d54b3f50ecc463c111d",
    "runner": "b91eed7dda9930859a8c95b1f0116c017e0f7d00651e4d0632f5530b840e55e9",
    "prereg": "236587175a33e39ac3052e3e7b537938ce62a5842d61fb3d67a407273205b928",
    "preflight": "d1f0bdcb2e41dbd1c8734130acb2bacd740f2e39e908cd3f37747d77155fbf1f",
}
ROUTE_SHA = "85f12fb0020bb8568dfc3683662e8251b29bf83684beb296dbb6d8734f5ffd20"
DIGEST_STEPS = [0] + list(range(99, 10_000, 100))
ARRAY_NAMES = {
    "routed_capture", "routed_down", "shared_down", "attention", "delta",
    "kv_state", "recurrent_state", "conv_state", "composed_state",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def stats_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["count"] == right["count"] and all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-9)
        for key in ("mean", "p50", "p95", "p99", "min", "max")
    )


def main() -> None:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite independent D10B-R verification")
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    walls = raw["latency"]["wall_ms"]
    events = raw["latency"]["cuda_event_ms"]
    telemetry = raw["telemetry"]
    checkpoints = raw["checkpoint_evidence"]
    arrays = [value for row in checkpoints for value in row["arrays"].values()]
    page_rates = [float(row["page_reads_per_sec"]) for row in raw["page_reads"]["samples"]]
    wall_replay = stats(walls)
    event_replay = stats(events)
    first_replay = stats(walls[:1000])
    last_replay = stats(walls[-1000:])
    drift_replay = last_replay["p95"] / first_replay["p95"]
    memory_loss = telemetry[0]["available_ram"] - telemetry[-1]["available_ram"]
    composed = [row["arrays"]["composed_state"]["sha256"] for row in checkpoints]
    hashes = {
        "raw": sha256(RAW),
        "raw_report": sha256(RAW_REPORT),
        "runner": sha256(RUNNER),
        "prereg": sha256(PREREG),
        "preflight": sha256(PREFLIGHT),
    }
    checks = {
        "immutable_hashes_exact": hashes == EXPECTED,
        "raw_input_locks_replay": (
            raw["inputs"]["runner_sha256"] == hashes["runner"]
            and raw["inputs"]["preregistration_sha256"] == hashes["prereg"]
            and raw["inputs"]["preflight_sha256"] == hashes["preflight"]
        ),
        "canonical_status_and_all_19_gates": (
            raw["status"] == "heldout_10000_endurance_pass"
            and raw["overall_pass"] is True
            and len(raw["gates"]) == 19
            and all(raw["gates"].values())
            and raw["error"] is None
        ),
        "heldout_route_contract_exact": (
            raw["route_contract"] == {
                "label": "p4d_shaped_synthetic_proxy",
                "partition": [768, 1024],
                "route_sha256": ROUTE_SHA,
                "steps": 10_000,
                "warmups": 8,
            }
        ),
        "latency_vectors_10000_finite_positive": (
            len(walls) == len(events) == 10_000
            and all(math.isfinite(value) and value > 0.0 for value in walls + events)
        ),
        "latency_statistics_replay": (
            stats_equal(wall_replay, raw["latency"]["wall_stats"])
            and stats_equal(event_replay, raw["latency"]["cuda_event_stats"])
            and stats_equal(first_replay, raw["latency"]["first_1000_wall_stats"])
            and stats_equal(last_replay, raw["latency"]["last_1000_wall_stats"])
            and math.isclose(drift_replay, raw["latency"]["last_first_p95_ratio"], rel_tol=0.0, abs_tol=1e-12)
        ),
        "timing_and_drift_hard_gates": wall_replay["p95"] <= 150.0 and wall_replay["p99"] <= 200.0 and drift_replay <= 1.20,
        "all_10000_state_checks_true": len(raw["state_checks"]) == 10_000 and all(raw["state_checks"]),
        "telemetry_10000_and_state_true": len(telemetry) == 10_000 and all(row["state_finite_and_written"] for row in telemetry),
        "ram_vram_and_loss_replay": (
            min(row["available_ram"] for row in telemetry) >= 1_610_612_736
            and min(row["free_vram"] for row in telemetry) >= 536_870_912
            and memory_loss <= 1_073_741_824
            and raw["physical"]["available_ram_after_first_touch"] >= 2_147_483_648
        ),
        "checkpoint_schedule_exact_101": len(checkpoints) == 101 and [row["step"] for row in checkpoints] == DIGEST_STEPS,
        "checkpoint_schema_exact_9_each": all(set(row["arrays"]) == ARRAY_NAMES for row in checkpoints),
        "all_909_arrays_finite_digested_no_poison": (
            len(arrays) == 909
            and all(value["finite"] and value["poison_count"] == 0 and len(value["sha256"]) == 64 for value in arrays)
        ),
        "all_101_composed_digests_unique": len(set(composed)) == 101,
        "paging_replay": bool(page_rates) and raw["page_reads"]["error"] is None and max(page_rates) <= 2048.0,
        "dense_and_runtime_exact": (
            raw["dense_runtime"]["dense_checksum_observed"] == raw["dense_runtime"]["dense_checksum_expected"]
            and raw["dense_runtime"]["runtime_sentinels"] == [0xA5, 0xA5]
        ),
        "registration_rows_48_exact": (
            len(raw["registration_attempts"]) == 48
            and [row["layer"] for row in raw["registration_attempts"]] == list(range(48))
            and all(row["attempted"] and row["success"] and row["error"] is None and row["device_alias"] for row in raw["registration_attempts"])
        ),
        "unregister_rows_48_clean": (
            len(raw["unregister_attempts"]) == 48
            and [row["layer"] for row in raw["unregister_attempts"]] == list(range(48))
            and all(row["attempted"] and row["success"] and row["error"] is None for row in raw["unregister_attempts"])
            and raw["unregister_failures"] == []
        ),
        "claim_boundary_remains_synthetic": all(
            phrase in raw["claim_boundary"]
            for phrase in ("Synthetic", "held-out P4D-shaped proxy", "not checkpoint", "not", "breakthrough")
        ),
    }
    passed = all(checks.values())
    result = {
        "kind": "port80b_d10br_heldout_10000_endurance_independent_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verification_pass": passed,
        "check_count": len(checks),
        "checks": checks,
        "inputs": hashes,
        "replay": {
            "wall_stats": wall_replay,
            "cuda_event_stats": event_replay,
            "first_1000_wall_stats": first_replay,
            "last_1000_wall_stats": last_replay,
            "last_first_p95_ratio": drift_replay,
            "telemetry_min_ram": min(row["available_ram"] for row in telemetry),
            "telemetry_min_vram": min(row["free_vram"] for row in telemetry),
            "telemetry_memory_loss": memory_loss,
            "page_read_samples": len(page_rates),
            "page_reads_per_sec_max": max(page_rates),
            "checkpoints": len(checkpoints),
            "array_evidence_rows": len(arrays),
            "unique_composed_digests": len(set(composed)),
            "registration_rows": len(raw["registration_attempts"]),
            "unregister_rows": len(raw["unregister_attempts"]),
        },
        "claim_boundary": raw["claim_boundary"],
        "physical_actions": {
            "gpu_run": False,
            "host_registration": False,
            "bank_scan": False,
            "registry_edit": False,
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# PORT80B-D10B-R independent endurance verification\n\n"
        f"Verdict: **{'PASS' if passed else 'FAIL'}** ({sum(checks.values())}/{len(checks)} checks).\n\n"
        f"The exact 10,000-step held-out raw result was replayed CPU-only. Wall p50/p95/p99: "
        f"{wall_replay['p50']:.6f}/{wall_replay['p95']:.6f}/{wall_replay['p99']:.6f} ms; "
        f"last/first-1,000 p95 ratio {drift_replay:.6f}. Exactly {len(checkpoints)} checkpoints, "
        f"{len(arrays)} array records, {len(set(composed))} unique composed-state digests, "
        f"48/48 register rows and 48/48 clean unregister rows were verified.\n\n"
        f"Claim boundary: {raw['claim_boundary']}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

