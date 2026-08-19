from __future__ import annotations

import json

from common import write_json_atomic, utc_now
from s100_lightning15_common import RESULTS, ensure_results

def load(name):
    path = RESULTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def candidate_file(mode, families, split):
    return (
        f"S100_LIGHTNING15_QUALITY_{mode}_{families}_{split}.json"
        .upper()
    )

def main():
    ensure_results()
    identity = load("S100_LIGHTNING15_IDENTITY_AUDIT.json") or {}
    component = load("S100_LIGHTNING15_COMPONENT.json") or {}
    selection = load("S100_LIGHTNING15_SELECTION.json") or {}
    parent_a = load("S100_LIGHTNING15_PARENT_TIMING_A.json")
    parent_b = load("S100_LIGHTNING15_PARENT_TIMING_B.json")

    selected_results = []
    quality_open = False
    family_open = False
    for row in selection.get("selected_for_validation", []):
        mode = row["mode"].upper()
        families = row["families"].upper()
        validation = load(candidate_file(
            mode, families, "VALIDATION"
        ))
        heldout = load(candidate_file(
            mode, families, "HELDOUT"
        ))
        record = {
            "mode": row["mode"],
            "families": row["families"],
            "validation": validation,
            "heldout": heldout,
        }
        selected_results.append(record)
        if (
            validation and validation.get("strict_pass")
            and heldout and heldout.get("official_pass")
        ):
            quality_open = True
            family_open = row["families"] != "kvo"

    baseline_ms = None
    baseline_tok_s = None
    if (
        parent_a and parent_a.get("status") == "measured"
        and parent_b and parent_b.get("status") == "measured"
    ):
        baseline_ms = (
            float(parent_a["timing"]["p50"])
            + float(parent_b["timing"]["p50"])
        ) / 2
        baseline_tok_s = 1000.0 / baseline_ms

    trace_green = bool(
        selection.get("LIGHTNING_TRACE_PROVENANCE_GREEN")
    )
    cold_open = bool(component.get("BF16X2_COLD_STREAM_OPEN"))
    block_rerun = bool(trace_green and cold_open and quality_open)

    payload = {
        "kind": "s100_lightning15_summary",
        "created_utc": utc_now(),
        "identity": identity.get("identity"),
        "inherited_trace_quarantined": bool(
            identity.get("INHERITED_TRACE_QUARANTINED")
        ),
        "LIGHTNING_TRACE_PROVENANCE_GREEN": trace_green,
        "BF16X2_COLD_STREAM_OPEN": cold_open,
        "BF16X2_QUALITY_OPEN": quality_open,
        "BF16X2_FAMILY_SELECTIVE_OPEN": family_open,
        "LIGHTNING_BLOCK_VERIFIER_RERUN_OPEN": block_rerun,
        "parent_baseline_ms": baseline_ms,
        "parent_baseline_tok_s": baseline_tok_s,
        "component_b4_tc2": (
            component.get("per_block", {})
            .get("4", {}).get("terms", {}).get("2")
        ),
        "selected_results": selected_results,
        "nano_results_status": (
            "quarantined for Lightning model-dependent claims"
        ),
        "s100_single_achieved": False,
        "next_action": (
            "BUILD_GRAPH_CAPTURABLE_BF16X2_AND_RERUN_LIGHTNING_BLOCK_VERIFIER"
            if block_rerun else
            "LOCALIZE_OR_CLOSE_BF16X2_QUALITY"
            if trace_green and cold_open else
            "REPAIR_LIGHTNING_PROVENANCE_OR_COMPONENT"
        ),
    }
    write_json_atomic(
        RESULTS / "S100_LIGHTNING15_SUMMARY.json",
        payload, archive=True,
    )
    text = (
        "S100 LIGHTNING PHASE 15 — BF16X2\n"
        f"LIGHTNING_TRACE_PROVENANCE_GREEN: {trace_green}\n"
        f"BF16X2_COLD_STREAM_OPEN: {cold_open}\n"
        f"BF16X2_QUALITY_OPEN: {quality_open}\n"
        f"BF16X2_FAMILY_SELECTIVE_OPEN: {family_open}\n"
        f"LIGHTNING_BLOCK_VERIFIER_RERUN_OPEN: {block_rerun}\n"
        f"Lightning parent baseline: {baseline_ms} ms / {baseline_tok_s} tok/s\n"
        f"Next action: {payload['next_action']}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS / "S100_LIGHTNING15_SUMMARY.txt").write_text(
        text, encoding="utf-8"
    )
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
