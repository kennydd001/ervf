from __future__ import annotations

import json

from common import REPO,utc_now,write_json_atomic

RESULTS=REPO/"pro_research"/"results"/"s100_phase23r"
OUT=RESULTS/"S100_PHASE23R_SUMMARY.json"

def load(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    adj=load(RESULTS/"S100_PHASE23R_ADJUDICATION.json")
    adopted=bool(adj.get("PROMOTE_GPU_GROUPED_MOE"))
    promoted=[]
    if adopted:
        for ctx in (128,1024,4096):
            d=load(RESULTS/f"S100_PHASE23R_PROMOTED_CTX{ctx}.json")
            promoted.append({
              "context":ctx,"status":d.get("status"),
              "correctness_green":d.get("correctness_green"),
              "summary":d.get("summary"),
              "telemetry":d.get("telemetry"),
            })

    promoted_complete=bool(
      adopted and len(promoted)==3
      and all(x["status"]=="measured" and x["correctness_green"]
              for x in promoted)
    )
    med=[
      float(x["summary"]["median_ms"]) for x in promoted
      if x.get("summary") and x["summary"].get("median_ms") is not None
    ]
    target40=bool(promoted_complete and len(med)==3 and max(med)<=40.0)
    draft32=bool(promoted_complete and len(med)==3 and max(med)<=32.0)

    mr=adj.get("median_round_gain_fraction")
    if draft32:
        route="OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target40:
        route="TARGET_100_CEILING_OPEN_OPTIMIZE_DRAFTER_HEADROOM"
    elif adopted:
        route="PROFILE_POST_GROUPED_GRAPH_AND_ATTACK_NEXT_DOMINANT_FAMILY"
    elif mr is not None and float(mr)>0:
        route="GROUPED_CORRECT_SUB5_PROFILE_M1_AND_GROUPING_OVERHEAD"
    else:
        route="KEEP_V6_GRAPH_PARENT_PROFILE_NEXT_DOMINANT_FAMILY"

    complete=bool(adj and (not adopted or promoted_complete))
    out={
      "kind":"s100_phase23r_summary","created_utc":utc_now(),
      "instrumentation_complete":complete,
      "GPU_GROUPED_CORRECTNESS_GREEN":True,
      "thermal_adjudication":adj,
      "GPU_GROUPED_MOE_ADOPTED":adopted,
      "promoted_contexts":promoted,
      "PHASE23R_TARGET_40MS_OPEN":target40,
      "DRAFTER_SHOOTOUT_OPEN":draft32,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"thermal-stable perfect-draft target-only H4; no drafter",
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    text=(
      "S100 PHASE 23R — THERMAL ADJUDICATION\n"
      f"Instrumentation complete: {complete}\n"
      f"Median round gain: {adj.get('median_round_gain_fraction')}\n"
      f"Median paired-block gain: {adj.get('median_paired_block_gain_fraction')}\n"
      f"Parent robust CV: {adj.get('parent_robust_cv')}\n"
      f"Grouped robust CV: {adj.get('grouped_robust_cv')}\n"
      f"GPU_GROUPED_MOE_ADOPTED: {adopted}\n"
      f"PHASE23R_TARGET_40MS_OPEN: {target40}\n"
      f"DRAFTER_SHOOTOUT_OPEN: {draft32}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS/"S100_PHASE23R_SUMMARY.txt").write_text(text,encoding="utf-8")
    report=REPO/"reports"/"S100_PHASE23R_RUN_REPORT.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    rounds=adj.get("rounds") or []
    table=["| Round | Parent ms | Grouped ms | Gain |", "|---:|---:|---:|---:|"]
    for r in rounds:
        if not r.get("complete"):
            table.append(f"| {r.get('round')} | n/a | n/a | n/a |")
        else:
            table.append(
              f"| {r['round']} | {r['parent_median_ms']:.3f} | "
              f"{r['grouped_median_ms']:.3f} | {100*r['round_gain_fraction']:.2f}% |"
            )
    md=(
      "# S100 Phase 23R — Thermal Adjudication\n\n"
      "Phase23 GPU-grouped MoE was already correctness-green. This repair run "
      "tests whether its measured transport savings survive a balanced thermal/order protocol.\n\n"
      + "\n".join(table) + "\n\n"
      f"- Median round gain: `{adj.get('median_round_gain_fraction')}`\n"
      f"- Median 64-block paired gain: `{adj.get('median_paired_block_gain_fraction')}`\n"
      f"- Parent robust CV: `{adj.get('parent_robust_cv')}`\n"
      f"- Grouped robust CV: `{adj.get('grouped_robust_cv')}`\n"
      f"- GPU grouped adopted: `{adopted}`\n"
      f"- Target <=40 ms/H4: `{target40}`\n"
      f"- Drafter shootout open: `{draft32}`\n"
      f"- Next route: `{route}`\n"
      "- S100 single achieved: `False`\n"
    )
    report.write_text(md,encoding="utf-8")
    print(text)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
