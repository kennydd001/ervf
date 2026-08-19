from __future__ import annotations

import json

from common import write_json_atomic, utc_now
from s100_phase14r_common import RESULTS, ensure_results

def load(name):
    path = RESULTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    ensure_results()
    component = load("S100_PHASE14R_NATIVE_ZERO_COPY.json")
    validation = load("S100_PHASE14R_NATIVE_VALIDATION.json")
    heldout = load("S100_PHASE14R_NATIVE_HELDOUT.json")
    subspace = load("S100_PHASE14R_OUTPUT_SUBSPACE.json")
    expert = load("S100_PHASE14R_EXPERT_BASIS.json")

    component_valid = bool(
        component
        and component.get("status") == "measured"
        and component.get("b4_measurement_valid")
    )
    if not component_valid:
        native_flag = None
    elif not component.get("b4_component_gate_pass"):
        native_flag = False
    elif not validation or validation.get("status") != "measured":
        native_flag = None
    elif not validation.get("strict_pass"):
        native_flag = False
    elif not heldout or heldout.get("status") != "measured":
        native_flag = None
    else:
        native_flag = bool(heldout.get("official_pass"))

    if not subspace or subspace.get("status") != "measured":
        subspace_flag = None
    else:
        subspace_flag = bool(
            subspace.get("SUBSPACE_RUNTIME_BUILD_OPEN")
        )

    if not expert or expert.get("status") not in {
        "measured", "incomplete"
    }:
        expert_flag = None
    elif expert.get("status") == "incomplete":
        expert_flag = None
    else:
        expert_flag = bool(
            expert.get("EXPERT_BASIS_RUNTIME_BUILD_OPEN")
        )

    out = {
        "kind": "s100_phase14r_summary",
        "created_utc": utc_now(),
        "dflash2_frozen": {
            "DFLASH2_CURRENT_VERIFIER_S100_OPEN": False,
            "DFLASH2_RESIDENT_MEMORY_OPEN_4K": False,
            "DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN": False,
            "DFLASH2_TRAINING_BUILD_OPEN": False,
            "provenance": (
                "completed Phase-14F run on "
                "agent/s100-phase14-dflash2-hardware"
            ),
        },
        "NATIVE_BLOCK_RUNTIME_BUILD_OPEN": native_flag,
        "SUBSPACE_RUNTIME_BUILD_OPEN": subspace_flag,
        "EXPERT_BASIS_RUNTIME_BUILD_OPEN": expert_flag,
        "instrumentation": {
            "native_component_valid": component_valid,
            "native_validation_measured": bool(
                validation and validation.get("status") == "measured"
            ),
            "native_heldout_measured": bool(
                heldout and heldout.get("status") == "measured"
            ),
            "subspace_measured": bool(
                subspace and subspace.get("status") == "measured"
            ),
            "expert_measured": bool(
                expert and expert.get("status") == "measured"
            ),
        },
        "native_b4": (
            component.get("per_block", {}).get("4")
            if component else None
        ),
        "native_validation": (
            validation.get("summary") if validation else None
        ),
        "native_heldout": (
            heldout.get("summary") if heldout else None
        ),
        "subspace_open_families": (
            subspace.get("open_families") if subspace else None
        ),
        "expert_layers": expert.get("layers") if expert else None,
        "s100_single_achieved": False,
        "next_action": (
            "BUILD_NATIVE_LAYER_MAJOR_BLOCK_RUNTIME"
            if native_flag is True else
            "BUILD_SUBSPACE_RUNTIME"
            if subspace_flag is True else
            "BUILD_EXPERT_BASIS_RUNTIME"
            if expert_flag is True else
            "ALL_REPAIRED_SURVIVORS_CLOSED"
            if all(flag is False for flag in (
                native_flag, subspace_flag, expert_flag
            )) else
            "REPAIR_REMAINING_TECHNICAL_EVIDENCE"
        ),
    }
    write_json_atomic(
        RESULTS / "S100_PHASE14R_SUMMARY.json", out, archive=True
    )
    text = (
        "S100 PHASE 14R — REPAIRED SURVIVORS\n"
        f"NATIVE_BLOCK_RUNTIME_BUILD_OPEN: {native_flag}\n"
        f"SUBSPACE_RUNTIME_BUILD_OPEN: {subspace_flag}\n"
        f"EXPERT_BASIS_RUNTIME_BUILD_OPEN: {expert_flag}\n"
        "DFLASH2_TRAINING_BUILD_OPEN: False (frozen completed result)\n"
        f"Next action: {out['next_action']}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS / "S100_PHASE14R_SUMMARY.txt").write_text(
        text, encoding="utf-8"
    )
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
