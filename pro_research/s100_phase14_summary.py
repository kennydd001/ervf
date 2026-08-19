from __future__ import annotations

import json

from common import write_json_atomic, utc_now
from s100_phase14_common import RESULTS, ensure_results


def load(name):
    path = RESULTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def tri_gate(payload: dict, name: str):
    if payload.get("status") != "measured":
        return None
    value = (payload.get("gates") or {}).get(name)
    return bool(value) if value is not None else None


def dflash_projection(economics: dict, proxy: dict):
    if economics.get("status") != "measured" or proxy.get("status") != "measured":
        return None
    measured_b8 = [
        row for row in (economics.get("verifier_sources") or [])
        if row.get("kind") == "measured" and int(row.get("block", 0)) == 8
    ]
    if not measured_b8:
        return None
    verifier = min(measured_b8, key=lambda row: float(row["verify_cycle_ms"]))
    verify_ms = float(verifier["verify_cycle_ms"])
    corrected = (
        proxy.get("validation_candidates", {}).get("corrected") or {}
    )
    selector = proxy.get("selector_proxy") or {}
    acceptance = {
        "independent": corrected.get(
            "mean_acceptance_independent_including_anchor"
        ),
        "oracle_lattice": corrected.get(
            "mean_acceptance_oracle_lattice_including_anchor"
        ),
        "frozen_selector_proxy": selector.get(
            "validation_mean_acceptance_including_anchor"
        ),
    }
    projections = {}
    for name, value in acceptance.items():
        if value is None:
            projections[name] = None
            continue
        accepted = float(value)
        projections[name] = {
            "mean_acceptance_including_anchor": accepted,
            "zero_cost_drafter_upper_bound_tok_s": 1000.0 * accepted / verify_ms,
            "remaining_draft_plus_selector_budget_for_s100_ms": (
                10.0 * accepted - verify_ms
            ),
        }
    return {
        "verifier_source": verifier.get("name"),
        "verify_cycle_ms": verify_ms,
        "projections": projections,
        "claim_boundary": (
            "combines measured verifier cost with proxy acceptance; zero-cost "
            "drafter upper bound, not end-to-end throughput"
        ),
    }


def main():
    ensure_results()
    component = load("S100_PHASE14D_NATIVE_EXTENDED.json") or {}
    validation = load("S100_PHASE14D_NATIVE_VALIDATION.json") or {}
    heldout = load("S100_PHASE14D_NATIVE_HELDOUT.json") or {}
    subspace = load("S100_PHASE14B_OUTPUT_SUBSPACE.json") or {}
    expert = load("S100_PHASE14E_DECODED_BASIS.json") or {}
    dflash_economics = load("S100_PHASE14F_DFLASH2_ECONOMICS.json") or {}
    dflash_proxy = load("S100_PHASE14F_DFLASH2_PROXY.json") or {}

    native_complete = (
        component.get("status") == "measured"
        and validation.get("status") == "measured"
        and (
            not validation.get("strict_pass")
            or heldout.get("status") == "measured"
        )
    )
    if not native_complete:
        native_open = None
    else:
        native_open = bool(
            component.get("b4_component_gate_pass")
            and validation.get("strict_pass")
            and heldout.get("official_pass")
        )

    subspace_open = (
        bool(subspace.get("SUBSPACE_RUNTIME_BUILD_OPEN"))
        if subspace.get("status") == "measured" else None
    )
    expert_open = (
        bool(expert.get("EXPERT_BASIS_RUNTIME_BUILD_OPEN"))
        if expert.get("status") == "measured" else None
    )

    current_verifier_open = tri_gate(
        dflash_economics, "CURRENT_VERIFIER_PERFECT_DRAFT_S100_OPEN"
    )
    dflash_memory_open = tri_gate(
        dflash_economics, "RESIDENT_DRAFTER_MEMORY_OPEN_4K_BF16_KV"
    )
    dflash_transfer_open = tri_gate(
        dflash_proxy, "DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN"
    )
    dflash_complete = (
        dflash_economics.get("status") == "measured"
        and dflash_proxy.get("status") == "measured"
    )
    dflash_inputs = (
        current_verifier_open, dflash_memory_open, dflash_transfer_open
    )
    # Three-valued conjunction: one measured hard false is sufficient to close
    # the combined gate; otherwise missing evidence stays null.
    if any(value is False for value in dflash_inputs):
        dflash_training_open = False
    elif any(value is None for value in dflash_inputs):
        dflash_training_open = None
    else:
        dflash_training_open = True
    dflash_perf = dflash_projection(dflash_economics, dflash_proxy)

    out = {
        "kind": "s100_phase14_summary",
        "created_utc": utc_now(),
        "closed_without_rerun": {
            "13A_entropy": True,
            "13C_temporal_delta": True,
        },
        "instrumentation": {
            "14D_complete": native_complete,
            "14B2_complete": subspace.get("status") == "measured",
            "14E2_complete": expert.get("status") == "measured",
            "14F_economics_complete": dflash_economics.get("status") == "measured",
            "14F_transfer_proxy_complete": dflash_proxy.get("status") == "measured",
        },
        "NATIVE_BLOCK_RUNTIME_BUILD_OPEN": native_open,
        "SUBSPACE_RUNTIME_BUILD_OPEN": subspace_open,
        "EXPERT_BASIS_RUNTIME_BUILD_OPEN": expert_open,
        "DFLASH2_CURRENT_VERIFIER_S100_OPEN": current_verifier_open,
        "DFLASH2_RESIDENT_MEMORY_OPEN_4K": dflash_memory_open,
        "DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN": dflash_transfer_open,
        "DFLASH2_TRAINING_BUILD_OPEN": dflash_training_open,
        "native_component_b4": (
            component.get("per_block", {}).get("4", {})
        ),
        "native_validation": validation.get("summary"),
        "native_heldout": heldout.get("summary"),
        "subspace_open_families": subspace.get("open_families"),
        "expert_layers": expert.get("layers"),
        "dflash2_economics_decision": dflash_economics.get("decision"),
        "dflash2_proxy_decision": dflash_proxy.get("decision"),
        "dflash2_proxy_candidates": (
            dflash_proxy.get("validation_candidates", {}).get("corrected")
        ),
        "dflash2_measured_verifier_projection": dflash_perf,
        "s100_single_achieved": False,
        "claim_boundary": (
            "survivor adjudication plus DFlash2 train-or-kill screen; flags "
            "authorize later implementation work only; null means incomplete "
            "evidence and never an automatic no-go"
        ),
    }
    write_json_atomic(
        RESULTS / "S100_PHASE14_SUMMARY.json", out, archive=True
    )
    projection_lines = ""
    if dflash_perf:
        for name, row in (dflash_perf.get("projections") or {}).items():
            if row:
                projection_lines += (
                    f"DFLASH2_{name.upper()}_ZERO_COST_UPPER_TOK_S: "
                    f"{row['zero_cost_drafter_upper_bound_tok_s']:.3f}\n"
                )
    text = (
        "S100 PHASE 14 — SURVIVOR + DFLASH2 ADJUDICATION\n"
        f"NATIVE_BLOCK_RUNTIME_BUILD_OPEN: {native_open}\n"
        f"SUBSPACE_RUNTIME_BUILD_OPEN: {subspace_open}\n"
        f"EXPERT_BASIS_RUNTIME_BUILD_OPEN: {expert_open}\n"
        f"DFLASH2_CURRENT_VERIFIER_S100_OPEN: {current_verifier_open}\n"
        f"DFLASH2_RESIDENT_MEMORY_OPEN_4K: {dflash_memory_open}\n"
        f"DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN: {dflash_transfer_open}\n"
        f"DFLASH2_TRAINING_BUILD_OPEN: {dflash_training_open}\n"
        f"{projection_lines}"
        f"14D instrumentation complete: {native_complete}\n"
        f"14B2 status: {subspace.get('status')}\n"
        f"14E2 status: {expert.get('status')}\n"
        f"14F economics status: {dflash_economics.get('status')}\n"
        f"14F proxy status: {dflash_proxy.get('status')}\n"
        "13A entropy: CLOSED\n"
        "13C temporal delta: CLOSED\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS / "S100_PHASE14_SUMMARY.txt").write_text(
        text, encoding="utf-8"
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
