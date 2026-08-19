from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REPO, write_json_atomic, utc_now

R = REPO / "pro_research" / "results" / "s100_phase15"

ARMS = [
    ("mm_fp32out", "all"),
    ("mm_fp32out", "attention"),
    ("mm_fp32out", "mamba"),
    ("mm_fp32out_comp2", "all"),
]

def safe_scope(scope):
    return scope.replace(":", "_").replace("/", "_").upper()

def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def arm_file(variant, scope, split):
    return R / (
        f"S100_PHASE15B_{variant.upper()}_"
        f"{safe_scope(scope)}_{split.upper()}.json"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("preheldout", "final"), required=True)
    args = ap.parse_args()

    component = load(R / "S100_PHASE15A_COMPONENT_VARIANTS.json") or {}
    matrix = load(R / "S100_PHASE15C_MATRIX_SENSITIVITY.json") or {}
    h1 = load(R / "S100_PHASE15D_HORIZON_MM_FP32OUT.json") or {}
    h2 = load(R / "S100_PHASE15D_HORIZON_MM_FP32OUT_COMP2.json") or {}

    validation = []
    for variant, scope in ARMS:
        d = load(arm_file(variant, scope, "validation"))
        if d and d.get("status") == "measured":
            validation.append({
                "variant": variant,
                "scope": scope,
                "strict_pass": bool(d.get("strict_pass")),
                "summary": d.get("summary"),
            })

    green = [x for x in validation if x["strict_pass"]]
    # Prefer one-pass all, then selective one-pass, then comp2 all.
    preference = {
        ("mm_fp32out", "all"): 0,
        ("mm_fp32out", "attention"): 1,
        ("mm_fp32out", "mamba"): 2,
        ("mm_fp32out_comp2", "all"): 3,
    }
    green.sort(key=lambda x: preference[(x["variant"], x["scope"])])
    selected = green[0] if green else None

    b1 = (
        component.get("per_B", {})
        .get("1", {})
        .get("mm_fp32out", {})
        .get("aggregate_speedup")
    )
    b4c = (
        component.get("per_B", {})
        .get("4", {})
        .get("mm_fp32out_comp2", {})
        .get("aggregate_speedup")
    )
    direct_component = b1 is not None and b1 >= 1.10
    comp_block_component = b4c is not None and b4c >= 1.25

    block_go = bool(
        comp_block_component
        and h2.get("H4_BLOCK_RESEARCH_GO") is True
    )
    onepass_block_go = bool(
        h1.get("H4_BLOCK_RESEARCH_GO") is True
    )

    heldout = None
    heldout_pass = None
    if args.stage == "final" and selected:
        heldout = load(arm_file(
            selected["variant"], selected["scope"], "heldout"
        ))
        heldout_pass = (
            bool(heldout.get("strict_pass"))
            if heldout and heldout.get("status") == "measured"
            else None
        )

    selective_open = bool(
        not green and int(matrix.get("safe_count") or 0) >= 3
    )

    if selected and args.stage == "preheldout":
        next_route = "RUN_LOCKED_HELDOUT"
    elif selected and heldout_pass is True and direct_component:
        next_route = "BUILD_GRAPH_NATIVE_BF16_SELECTED_SCOPE_B1"
    elif selected and heldout_pass is True and (block_go or onepass_block_go):
        next_route = "BUILD_EXACT_STATE_REFRESH_BLOCK_DRAFT_VERIFIER"
    elif selective_open:
        next_route = "BUILD_SAFE_MATRIX_SUBSET_AND_VALIDATE"
    elif block_go:
        next_route = "BLOCK_DRAFT_PROMISING_BUT_FULL_TRAJECTORY_NOT_GREEN"
    else:
        next_route = "LOCALIZE_OR_CLOSE_NATIVE_BF16_DIRECT_SUBSTITUTION"

    out = {
        "kind": "s100_phase15_summary",
        "stage": args.stage,
        "created_utc": utc_now(),
        "torch_mm_bf16_fp32_out_supported": component.get(
            "torch_mm_bf16_fp32_out_supported"
        ),
        "B1_mm_fp32out_speedup": b1,
        "B4_comp2_speedup": b4c,
        "validation_arms": validation,
        "selected_validation_arm": selected,
        "RUN_HELDOUT": bool(selected and args.stage == "preheldout"),
        "heldout_strict_pass": heldout_pass,
        "matrix_safe_count": matrix.get("safe_count"),
        "safe_cases_ranked": matrix.get("safe_cases_ranked"),
        "H4_mm_fp32out_go": h1.get("H4_BLOCK_RESEARCH_GO"),
        "H4_mm_fp32out_comp2_go": h2.get("H4_BLOCK_RESEARCH_GO"),
        "DIRECT_NATIVE_BF16_RUNTIME_BUILD_OPEN": bool(
            args.stage == "final"
            and heldout_pass is True
            and direct_component
        ),
        "EXACT_STATE_BLOCK_DRAFT_BUILD_OPEN": bool(
            args.stage == "final"
            and heldout_pass is True
            and (block_go or onepass_block_go)
        ),
        "SELECTIVE_MATRIX_RUNTIME_RESEARCH_OPEN": selective_open,
        "NEXT_ROUTE": next_route,
        "S100_SINGLE_ACHIEVED": False,
        "claim_boundary": (
            "Phase15 integration authorization only; no production tok/s claim"
        ),
    }
    R.mkdir(parents=True, exist_ok=True)
    name = (
        "S100_PHASE15_PREHELDOUT_SUMMARY.json"
        if args.stage == "preheldout"
        else "S100_PHASE15_SUMMARY.json"
    )
    write_json_atomic(R / name, out, archive=True)

    text = (
        f"S100 PHASE 15 — {args.stage.upper()}\n"
        f"BF16->FP32 torch.mm supported: "
        f"{out['torch_mm_bf16_fp32_out_supported']}\n"
        f"B1 mm_fp32out speedup: {b1}\n"
        f"B4 comp2 speedup: {b4c}\n"
        f"Selected validation arm: {selected}\n"
        f"Heldout strict pass: {heldout_pass}\n"
        f"Matrix safe count: {out['matrix_safe_count']}\n"
        f"H4 one-pass go: {out['H4_mm_fp32out_go']}\n"
        f"H4 comp2 go: {out['H4_mm_fp32out_comp2_go']}\n"
        f"DIRECT_NATIVE_BF16_RUNTIME_BUILD_OPEN: "
        f"{out['DIRECT_NATIVE_BF16_RUNTIME_BUILD_OPEN']}\n"
        f"EXACT_STATE_BLOCK_DRAFT_BUILD_OPEN: "
        f"{out['EXACT_STATE_BLOCK_DRAFT_BUILD_OPEN']}\n"
        f"SELECTIVE_MATRIX_RUNTIME_RESEARCH_OPEN: "
        f"{out['SELECTIVE_MATRIX_RUNTIME_RESEARCH_OPEN']}\n"
        f"NEXT_ROUTE: {next_route}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    txt_name = (
        "S100_PHASE15_PREHELDOUT_SUMMARY.txt"
        if args.stage == "preheldout"
        else "S100_PHASE15_SUMMARY.txt"
    )
    (R / txt_name).write_text(text, encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
