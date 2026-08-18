"""Frozen morning adjudicator for the Arc Downflow Engine overnight campaign."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

OFFSETS = (0, 1, 4, 16)
QFAST_MS = 18.75165
ACT_BYTES = 6 * 1856 * 4
OUT_BYTES = 2688 * 4
BRIDGE_PROBE_BYTES = ACT_BYTES + OUT_BYTES


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def strict6_summary(doc):
    if not doc or doc.get("status") != "measured":
        return None
    rows = [r for r in doc.get("records", [])
            if int(r.get("nexperts", -1)) == 6 and not bool(r.get("fast_math", True))]
    if not rows:
        return None
    by_layer = {}
    for r in rows:
        c = r.get("correctness") or {}
        ok = bool(c.get("finite")) and float(c.get("cosine", -1)) >= 0.999 and float(c.get("nrmse", 1e9)) <= 0.02
        by_layer[int(r["layer"])] = {
            "wall_ms": float(r["best"]["wall_median_ms"]),
            "event_ms": float(r["best"]["event_median_ms"]),
            "correct": ok,
            "correctness": c,
            "local": int(r["best"]["local"]),
        }
    return {
        "layer_count": len(by_layer),
        "all_correct": len(by_layer) == 23 and all(x["correct"] for x in by_layer.values()),
        "wall_sum_ms": float(sum(x["wall_ms"] for x in by_layer.values())),
        "event_sum_ms": float(sum(x["event_ms"] for x in by_layer.values())),
        "layers": by_layer,
    }


def median_or_none(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    results = {
        "kind": "s100_phase8_overnight_adjudication",
        "decision": "ADE_BORDERLINE",
        "decision_reasons": [],
        "qfast_reference_ms": QFAST_MS,
        "offsets": {},
    }
    required_complete = True
    raw_arc_baselines = {}
    adjusted_advantages = []
    route_match_all = True
    correctness_all = True

    # Bridge is intentionally conservative: the existing probe sends BRIDGE_PROBE_BYTES
    # in each direction rather than the asymmetric 44.5 KB + 10.5 KB true payload.
    bridge_vals = []
    bridge_rows = []
    for p in sorted(d.glob("bridge_*.json")):
        doc = load(p)
        if not doc or doc.get("status") != "measured":
            continue
        row = next((r for r in doc.get("rows", []) if int(r.get("bytes", -1)) == BRIDGE_PROBE_BYTES), None)
        if row and row.get("correct"):
            bridge_vals.append(float(row["median_ms"]))
            bridge_rows.append({"file": p.name, **row})
    bridge_med = median_or_none(bridge_vals)
    results["bridge"] = {
        "probe_bytes_each_direction": BRIDGE_PROBE_BYTES,
        "actual_design_total_payload_bytes": BRIDGE_PROBE_BYTES,
        "note": "probe is conservative because it transfers this size in both PCIe directions",
        "samples": bridge_rows,
        "median_ms": bridge_med,
    }
    if bridge_med is None:
        required_complete = False
        results["decision_reasons"].append("missing valid repeated CUDA-pinned/OpenCL bridge data")

    # Full-bank pressure establishes how optimistic the six-record tiny buffer is.
    full_docs = []
    for p in sorted(d.glob("full_bank_*.json")):
        doc = load(p)
        if doc and doc.get("status") == "measured":
            full_docs.append((p, doc))
    full_cold = []
    full_warm = []
    full_correct = True
    for p, doc in full_docs:
        full_cold.append(float(doc["cold_summary"]["median_ms"]))
        full_warm.append(float(doc["warm_actual_route"]["median_ms"]))
        for c in doc.get("correctness", []):
            ok = bool(c.get("finite")) and float(c.get("cosine", -1)) >= .999 and float(c.get("nrmse", 1e9)) <= .02
            full_correct = full_correct and ok
    cold_med = median_or_none(full_cold)
    warm_med = median_or_none(full_warm)
    pressure_factor = max(1.0, cold_med / warm_med) if cold_med is not None and warm_med not in (None, 0) else None
    results["full_bank_pressure"] = {
        "runs": len(full_docs),
        "cold_median_ms": cold_med,
        "warm_median_ms": warm_med,
        "cold_over_warm_factor": pressure_factor,
        "all_correct": full_correct,
        "note": "factor is applied conservatively to small-snapshot Arc kernel sums",
    }
    if not full_docs or pressure_factor is None:
        required_complete = False
        results["decision_reasons"].append("missing full-bank cold-residency pressure result")
    if not full_correct:
        correctness_all = False
        results["decision_reasons"].append("full-bank Arc NVFP4 correctness failed")

    for off in OFFSETS:
        arc = strict6_summary(load(d / f"arc_real_nvfp4_offset_{off}.json"))
        rtx = load(d / f"rtx_live_downflow_offset_{off}.json")
        snap = load(d / f"snapshot_offset_{off}.json")
        if arc is None or not rtx or rtx.get("status") != "measured" or not snap or snap.get("status") != "measured":
            required_complete = False
            results["offsets"][str(off)] = {"complete": False}
            continue
        raw_arc_baselines[off] = arc["wall_sum_ms"]
        correctness_all = correctness_all and arc["all_correct"]
        rtx_down = float(rtx["all_layer_sum_down_only_ms"])
        # Cross-check exact route identity between the independently rebuilt Arc snapshot
        # and RTX measurement.  A mismatch invalidates the comparison.
        snap_by = {int(x["layer"]): x["ids"] for x in snap.get("layers", [])}
        rtx_by = {int(x["layer"]): x["ids"] for x in rtx.get("records", [])}
        route_match = snap_by == rtx_by and len(snap_by) == 23
        route_match_all = route_match_all and route_match
        adjusted_kernel = arc["wall_sum_ms"] * (pressure_factor if pressure_factor is not None else 1.0)
        arc_total = adjusted_kernel + (23 * bridge_med if bridge_med is not None else 0.0)
        advantage = (rtx_down - arc_total) / rtx_down if rtx_down > 0 else None
        if advantage is not None:
            adjusted_advantages.append(advantage)
        results["offsets"][str(off)] = {
            "complete": True,
            "route_match": route_match,
            "arc_strict6": arc,
            "rtx_down_only_sum_ms": rtx_down,
            "rtx_serial_plane_fetch_plus_down_sum_ms": float(rtx["all_layer_sum_serial_ms"]),
            "arc_raw_kernel_sum_ms": arc["wall_sum_ms"],
            "arc_pressure_adjusted_kernel_sum_ms": adjusted_kernel,
            "bridge_per_layer_ms_conservative": bridge_med,
            "arc_adjusted_total_sum_ms": arc_total,
            "advantage_vs_rtx_down_fraction": advantage,
            "component_substitution_projected_qfast_ms": QFAST_MS - rtx_down + arc_total,
            "component_substitution_projected_tok_s": 1000.0 / (QFAST_MS - rtx_down + arc_total),
        }

    # Replication drift normalized by each offset's own first baseline.
    rep_ratios = []
    rep_rows = []
    for p in sorted(d.glob("replicate_*.json")):
        doc = load(p)
        s = strict6_summary(doc)
        if not s:
            continue
        # Filename ..._offsetX.json
        try:
            off = int(p.stem.rsplit("offset", 1)[1])
        except Exception:
            continue
        base = raw_arc_baselines.get(off)
        if base:
            ratio = s["wall_sum_ms"] / base
            rep_ratios.append(ratio)
            rep_rows.append({"file": p.name, "offset": off, "sum_ms": s["wall_sum_ms"], "ratio_to_initial": ratio, "all_correct": s["all_correct"]})
            correctness_all = correctness_all and s["all_correct"]
    drift = None
    if len(rep_ratios) >= 3:
        q = max(1, len(rep_ratios)//4)
        first = statistics.median(rep_ratios[:q])
        last = statistics.median(rep_ratios[-q:])
        drift = (last - first) / first
    results["overnight_replication"] = {
        "runs": len(rep_rows),
        "rows": rep_rows,
        "normalized_first_to_last_drift_fraction": drift,
    }
    if len(rep_rows) < 3:
        required_complete = False
        results["decision_reasons"].append("fewer than 3 independent Arc replication runs")

    # Contention from the real full-bank kernel, not synthetic OpenVINO work.
    inter = []
    for p in sorted(d.glob("interference_*.json")):
        doc = load(p)
        if doc and doc.get("regression_fraction") is not None:
            inter.append(float(doc["regression_fraction"]))
    inter_med = median_or_none(inter)
    results["contention"] = {
        "runs": len(inter),
        "regressions": inter,
        "median_regression_fraction": inter_med,
        "worst_regression_fraction": max(inter) if inter else None,
    }
    if len(inter) < 2:
        required_complete = False
        results["decision_reasons"].append("fewer than 2 real Arc-vs-QFAST contention runs")

    median_adv = median_or_none(adjusted_advantages)
    worst_adv = min(adjusted_advantages) if adjusted_advantages else None
    results["speed_decision"] = {
        "median_advantage_vs_rtx_down_fraction": median_adv,
        "worst_snapshot_advantage_fraction": worst_adv,
        "go_threshold": 0.10,
    }
    results["instrumentation"] = {
        "required_complete": required_complete,
        "all_arc_correct": correctness_all,
        "all_snapshot_routes_match": route_match_all,
    }

    # Frozen decision hierarchy.
    no_go = False
    if not correctness_all or not route_match_all:
        no_go = True
    if median_adv is not None and median_adv <= -0.10:
        no_go = True
        results["decision_reasons"].append("pressure-adjusted Arc routed-down is >=10% slower than current RTX downflow")
    if inter_med is not None and inter_med > 0.15:
        no_go = True
        results["decision_reasons"].append("Arc contention slows QFAST by >15%")

    go = (
        required_complete and correctness_all and route_match_all and
        median_adv is not None and median_adv >= 0.10 and
        (worst_adv is None or worst_adv >= 0.0) and
        inter_med is not None and inter_med <= 0.05 and
        drift is not None and abs(drift) <= 0.10
    )
    if go:
        results["decision"] = "ADE_GO"
        results["decision_reasons"].append("Arc routed-down clears correctness, full-bank pressure, speed, contention and overnight stability gates")
    elif no_go:
        results["decision"] = "ADE_NO_GO"
    else:
        results["decision"] = "ADE_BORDERLINE"
        if median_adv is not None:
            results["decision_reasons"].append(f"median pressure-adjusted Arc advantage is {median_adv:.3%}; GO requires >=10% with no negative snapshot")
        if inter_med is not None:
            results["decision_reasons"].append(f"median QFAST contention regression is {inter_med:.3%}; GO requires <=5%")
        if drift is not None:
            results["decision_reasons"].append(f"overnight normalized Arc drift is {drift:.3%}; GO requires |drift|<=10%")

    # What the experiment can and cannot claim.
    projected = [v.get("component_substitution_projected_qfast_ms") for v in results["offsets"].values() if isinstance(v, dict) and v.get("component_substitution_projected_qfast_ms")]
    results["projection_only"] = {
        "median_component_substitution_qfast_ms": median_or_none(projected),
        "median_component_substitution_tok_s": (1000.0 / median_or_none(projected)) if median_or_none(projected) else None,
        "not_end_to_end": True,
        "s100_single_achieved": False,
    }

    outj = d / "S100_PHASE8_OVERNIGHT_SUMMARY.json"
    outt = d / "S100_PHASE8_OVERNIGHT_SUMMARY.txt"
    outj.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "S100 PHASE 8 ARC OVERNIGHT ADJUDICATION",
        f"DECISION: {results['decision']}",
        f"Instrumentation complete: {required_complete}",
        f"All correctness green: {correctness_all}",
        f"All route snapshots match: {route_match_all}",
        f"Median Arc advantage vs current RTX downflow: {median_adv}",
        f"Worst snapshot advantage: {worst_adv}",
        f"Median QFAST contention regression: {inter_med}",
        f"Normalized first-to-last Arc drift: {drift}",
        f"Full-bank cold/warm factor: {pressure_factor}",
        f"Projected QFAST ms (component substitution only): {results['projection_only']['median_component_substitution_qfast_ms']}",
        f"Projected tok/s (NOT end-to-end): {results['projection_only']['median_component_substitution_tok_s']}",
        "S100 SINGLE ACHIEVED: False (component suite cannot make this claim)",
        "",
        "REASONS:",
    ] + [f"- {x}" for x in results["decision_reasons"]]
    outt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
