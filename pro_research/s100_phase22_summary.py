from __future__ import annotations
import json
from common import REPO,write_json_atomic,utc_now
from s100_phase22_common import RESULTS

OUT=RESULTS/"S100_PHASE22_SUMMARY.json"

def load(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    head=load(RESULTS/"S100_PHASE22_HEAD_SELECTION.json")
    state=load(RESULTS/"S100_PHASE22_GRAPH_STATE_CHECK.json")
    screen=load(RESULTS/"S100_PHASE22_SCREEN.json")
    p21=load(
      REPO/"pro_research"/"results"/"s100_phase21"/"S100_PHASE21_SUMMARY.json"
    )
    promoted=[]
    if screen.get("RUN_PROMOTED_GRAPH"):
        for ctx in (128,1024,4096):
            d=load(RESULTS/f"S100_PHASE22_GRAPH_CTX{ctx}_PROMOTED.json")
            promoted.append({
              "context":ctx,"status":d.get("status"),
              "correctness_green":d.get("correctness_green"),
              "summary":d.get("summary"),
            })

    complete=bool(
      head.get("status")=="measured"
      and state.get("status")=="measured"
      and screen
      and (
        not screen.get("RUN_PROMOTED_GRAPH")
        or (len(promoted)==3 and all(x["status"]=="measured" for x in promoted))
      )
    )
    graph_green=bool(state.get("GRAPH_CORRECTNESS_GREEN"))
    promoted_green=bool(
      promoted and all(x["correctness_green"] for x in promoted)
    )
    medians=[
      float(x["summary"]["median_ms"]) for x in promoted
      if x.get("summary") and x["summary"].get("median_ms") is not None
    ]
    target40=bool(promoted_green and len(medians)==3 and max(medians)<=40.0)
    draft32=bool(promoted_green and len(medians)==3 and max(medians)<=32.0)

    m1=float(((p21.get("profile") or {}).get("route_census") or {}).get(
      "m1_fraction",0.0
    ))
    phase23=bool(graph_green and not target40 and m1>=0.50)

    if draft32:
        route="OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target40:
        route="TARGET_100_CEILING_OPEN_OPTIMIZE_DRAFTER_HEADROOM"
    elif phase23 and screen.get("RUN_PROMOTED_GRAPH"):
        route="BUILD_GPU_RESIDENT_M1_M2_M3_M4_GROUPED_MOE_ON_H4_GRAPH"
    elif graph_green and not screen.get("RUN_PROMOTED_GRAPH"):
        route="GRAPH_CORRECT_BUT_NO_GAIN_PROFILE_GRAPH_NODES_AND_HEAD"
    elif not graph_green:
        route="REPAIR_H4_GRAPH_TECHNICAL_OR_NUMERICAL_PARITY"
    else:
        route="REPAIR_INCOMPLETE_PHASE22_EVIDENCE"

    out={
      "kind":"s100_phase22_summary","created_utc":utc_now(),
      "instrumentation_complete":complete,
      "head_selection":{
        "mode":head.get("selected_mode"),
        "generic_ms":head.get("generic_midpoint_ms"),
        "selected_ms":head.get("selected_median_ms"),
        "gain_fraction":head.get("selected_gain_fraction_vs_generic"),
      },
      "GRAPH_CORRECTNESS_GREEN":graph_green,
      "graph_state":state.get("state"),
      "graph_capture":state.get("capture_info"),
      "same_era_screen":screen,
      "promoted_contexts":promoted,
      "PHASE22_TARGET_40MS_OPEN":target40,
      "DRAFTER_SHOOTOUT_OPEN":draft32,
      "GPU_GROUPED_MOE_PHASE23_OPEN":phase23,
      "phase21_m1_fraction":m1,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"perfect-draft target-only H4 graph; no drafter",
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    text=(
      "S100 PHASE 22 — SINGLE H4 CUDA GRAPH\n"
      f"Instrumentation complete: {complete}\n"
      f"Head mode: {out['head_selection']['mode']}\n"
      f"Head gain fraction: {out['head_selection']['gain_fraction']}\n"
      f"GRAPH_CORRECTNESS_GREEN: {graph_green}\n"
      f"Graph screen gain fraction: {screen.get('graph_gain_fraction')}\n"
      f"PHASE22_TARGET_40MS_OPEN: {target40}\n"
      f"DRAFTER_SHOOTOUT_OPEN: {draft32}\n"
      f"GPU_GROUPED_MOE_PHASE23_OPEN: {phase23}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS/"S100_PHASE22_SUMMARY.txt").write_text(text,encoding="utf-8")
    print(text);return 0
if __name__=="__main__":raise SystemExit(main())
