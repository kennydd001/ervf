"""Phase55 target-ubatch sweep for quantized Ornith DFlash correctness."""
from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase54_ornith_greedy_repro import _run_once


RESULTS = REPO / "pro_research" / "results" / "s100_phase55"
PREREG = REPO / "pro_research" / "S100_PHASE55_ORNITH_UBATCH_ISOLATION_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase55_ornith_ubatch_isolation.py"
PHASE54 = RESULTS.parent / "s100_phase54" / "S100_PHASE54_ORNITH_GREEDY_REPRO.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=10)
    parser.add_argument("--port-base", type=int, default=18100)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE55_ORNITH_UBATCH_ISOLATION.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase55_ornith_ubatch_isolation",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        for path in (args.server, args.target, args.draft, PHASE54):
            if not path.is_file():
                raise FileNotFoundError(path)
        phase54 = json.loads(PHASE54.read_text(encoding="utf-8"))
        frozen_baseline = phase54["arms"]["baseline_r1"]["records"][0]["text"]
        version = subprocess.run(
            [str(args.server), "--version"],
            cwd=args.server.parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        arms: dict[str, dict[str, Any]] = {}
        ubatches = (4, 8, 16, 32, 64, 128, 256, 512)
        port_offset = 0
        for ubatch in ubatches:
            baseline_name = f"baseline_ub{ubatch}"
            try:
                arms[baseline_name] = _run_once(
                    baseline_name,
                    args.server.resolve(),
                    args.target.resolve(),
                    None,
                    0,
                    args.port_base + port_offset,
                    args.gpu_layers,
                    server_extra_args=("--ubatch-size", str(ubatch)),
                )
            except Exception as exc:
                arms[baseline_name] = {
                    "status": "technical_failure",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            port_offset += 1
            dflash_name = f"dflash_k8_ub{ubatch}"
            if arms[baseline_name]["status"] != "served":
                arms[dflash_name] = {
                    "status": "skipped_baseline_invalid",
                    "reason": baseline_name,
                }
                continue
            try:
                arms[dflash_name] = _run_once(
                    dflash_name,
                    args.server.resolve(),
                    args.target.resolve(),
                    args.draft.resolve(),
                    8,
                    args.port_base + port_offset,
                    args.gpu_layers,
                    server_extra_args=("--ubatch-size", str(ubatch)),
                )
            except Exception as exc:
                arms[dflash_name] = {
                    "status": "technical_failure",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            port_offset += 1
        valid_baseline_ubatches = [
            ubatch for ubatch in ubatches
            if arms[f"baseline_ub{ubatch}"]["status"] == "served"
        ]
        valid_paired_ubatches = [
            ubatch for ubatch in valid_baseline_ubatches
            if arms[f"dflash_k8_ub{ubatch}"]["status"] == "served"
        ]
        exact_baseline_vs_frozen = {
            str(ubatch): (
                arms[f"baseline_ub{ubatch}"]["records"][0]["text"] == frozen_baseline
            )
            for ubatch in valid_baseline_ubatches
        }
        exact_same_ubatch = {
            str(ubatch): (
                arms[f"dflash_k8_ub{ubatch}"]["records"][0]["text"]
                == arms[f"baseline_ub{ubatch}"]["records"][0]["text"]
            )
            for ubatch in valid_paired_ubatches
        }
        matching_k8_ubatches = [
            ubatch for ubatch in valid_paired_ubatches if exact_same_ubatch[str(ubatch)]
        ]
        all_valid_accept = all(
            arms[f"dflash_k8_ub{ubatch}"].get("accepted") is not None
            and arms[f"dflash_k8_ub{ubatch}"]["accepted"] > 0
            for ubatch in valid_paired_ubatches
        )
        if matching_k8_ubatches:
            adjudication = "bounded_ubatch_restores_same_geometry_lossless"
        elif valid_paired_ubatches:
            adjudication = "divergence_survives_all_valid_target_ubatches"
        else:
            adjudication = "no_valid_paired_ubatch"
        gates = {
            "P55_G1_default_ub512_pair_serves": (
                arms["baseline_ub512"]["status"] == "served"
                and arms["dflash_k8_ub512"]["status"] == "served"
            ),
            "P55_G2_default_reference_matches_phase54_baseline": (
                exact_baseline_vs_frozen.get("512") is True
            ),
            "P55_G3_all_valid_dflash_cells_accept_positive": all_valid_accept,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "llama_version": version,
            "arms": arms,
            "valid_baseline_ubatches": valid_baseline_ubatches,
            "valid_paired_ubatches": valid_paired_ubatches,
            "exact_baseline_vs_phase54": exact_baseline_vs_frozen,
            "exact_dflash_vs_same_ubatch_baseline": exact_same_ubatch,
            "matching_k8_ubatches": matching_k8_ubatches,
            "largest_matching_k8_ubatch": max(matching_k8_ubatches) if matching_k8_ubatches else None,
            "adjudication": adjudication,
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    payload["environment"] = environment_snapshot((SCRIPT, PREREG))
    write_json_atomic(out, payload, archive=True)
    arms = payload.get("arms") or {}
    print(json.dumps({
        "status": payload.get("status"),
        "adjudication": payload.get("adjudication"),
        "valid_baseline_ubatches": payload.get("valid_baseline_ubatches"),
        "valid_paired_ubatches": payload.get("valid_paired_ubatches"),
        "baseline_exact": payload.get("exact_baseline_vs_phase54"),
        "same_ubatch_exact": payload.get("exact_dflash_vs_same_ubatch_baseline"),
        "matching_k8_ubatches": payload.get("matching_k8_ubatches"),
        "tok_s": {
            name: arm["records"][0]["wall_tok_s"] for name, arm in arms.items()
            if arm.get("status") == "served"
        },
        "acceptance": {
            name: {"accepted": arm.get("accepted"), "drafted": arm.get("drafted")}
            for name, arm in arms.items()
            if name.startswith("dflash") and arm.get("status") == "served"
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
