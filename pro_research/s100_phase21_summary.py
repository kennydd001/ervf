from __future__ import annotations

import json
from common import write_json_atomic,utc_now
from s100_phase21_common import RESULTS

OUT=RESULTS/"S100_PHASE21_SUMMARY.json"
CONTEXTS=(128,1024,4096)

def load(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    prof=load(RESULTS/"S100_PHASE21_PROFILE.json")
    sel=load(RESULTS/"S100_PHASE21_SELECTION.json")
    arm=sel.get("selected_arm")
    promoted=[]
    if arm:
        for ctx in CONTEXTS:
            p=RESULTS/f"S100_PHASE21_PROMOTED_{arm.upper()}_CTX{ctx}.json"
            d=load(p)
            promoted.append({
              "context":ctx,"status":d.get("status"),
              "correctness_green":d.get("correctness_green"),
              "summary":d.get("summary"),
              "path":str(p),
            })

    complete=bool(
      prof.get("status")=="measured"
      and arm
      and len(promoted)==3
      and all(x["status"]=="measured" for x in promoted)
    )
    correct=bool(complete and all(x["correctness_green"] for x in promoted))
    medians=[
      float(x["summary"]["median_ms"]) for x in promoted
      if x.get("summary") and x["summary"].get("median_ms") is not None
    ]
    target40=bool(correct and len(medians)==3 and max(medians)<=40.0)
    draft32=bool(correct and len(medians)==3 and max(medians)<=32.0)

    gain=sel.get("gain_fraction_vs_current")
    device=arm in ("v6_device_rows","v18_device_rows")
    graph_open=bool(correct and device and gain is not None and float(gain)>=0.10)

    rc=prof.get("route_census") or {}
    repeat=float(rc.get("median_repeat_rate") or 0.0)
    current=next((x for x in sel.get("arms",[]) if x.get("arm")=="current_grouped"),{})
    chosen=sel.get("selected") or {}
    grouped_repair=bool(
      device and repeat>=0.20
      and current.get("median_ms") is not None
      and chosen.get("median_ms") is not None
      and float(chosen["median_ms"]) < float(current["median_ms"])
    )

    family=prof.get("family_ms") or {}
    candidates={k:float(family.get(k,0.0)) for k in ("mamba","moe","attention","lm_head")}
    dominant=max(candidates,key=candidates.get) if candidates else None

    if draft32:
        route="OPEN_OFFICIAL_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target40:
        route="TARGET_100_CEILING_OPEN_OPTIMIZE_DRAFTER_HEADROOM"
    elif graph_open:
        route="BUILD_SINGLE_H4_CUDA_GRAPH_ON_DEVICE_MOE_PARENT"
    elif grouped_repair:
        route="BUILD_GPU_RESIDENT_H4_GROUPED_MOE_DISPATCHER"
    elif dominant=="moe":
        route="PROFILE_AND_REPAIR_MOE_DATAPLANE"
    elif dominant=="attention":
        route="BUILD_H4_FP32_CAUSAL_ATTENTION_BLOCK"
    elif dominant=="mamba":
        route="RECHECK_MAMBA_BLOCK_INTEGRATION"
    else:
        route="PROFILE_LMHEAD_AND_GLUE"

    out={
      "kind":"s100_phase21_summary","created_utc":utc_now(),
      "instrumentation_complete":complete,
      "selected_arm":arm,
      "selection_gain_fraction_vs_current":gain,
      "profile":{
        "ordinary_block_ms":prof.get("ordinary_block_ms"),
        "family_ms":family,
        "route_census":rc,
        "dominant_family":dominant,
      },
      "promoted_contexts":promoted,
      "FULL_VERIFIER_CORRECTNESS_GREEN":correct,
      "PHASE21_TARGET_40MS_OPEN":target40,
      "DRAFTER_SHOOTOUT_OPEN":draft32,
      "PHASE21_GRAPH_BUILD_OPEN":graph_open,
      "PHASE21_GROUPED_MOE_REPAIR_OPEN":grouped_repair,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"target-only H4 optimization; no drafter cost",
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    text=(
      "S100 PHASE 21 — V18/V19 PARENT + H4\n"
      f"Instrumentation complete: {complete}\n"
      f"Selected arm: {arm}\n"
      f"Gain vs current grouped @1024: {gain}\n"
      f"Dominant profiled family: {dominant}\n"
      f"FULL_VERIFIER_CORRECTNESS_GREEN: {correct}\n"
      f"PHASE21_TARGET_40MS_OPEN: {target40}\n"
      f"DRAFTER_SHOOTOUT_OPEN: {draft32}\n"
      f"PHASE21_GRAPH_BUILD_OPEN: {graph_open}\n"
      f"PHASE21_GROUPED_MOE_REPAIR_OPEN: {grouped_repair}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS/"S100_PHASE21_SUMMARY.txt").write_text(text,encoding="utf-8")
    print(text)
    return 0

if __name__=="__main__":raise SystemExit(main())
