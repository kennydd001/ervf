from __future__ import annotations
import json
from common import write_json_atomic,utc_now
from s100_phase23_common import RESULTS

OUT=RESULTS/"S100_PHASE23_SUMMARY.json"
def load(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    st=load(RESULTS/"S100_PHASE23_STATE_CHECK.json")
    sc=load(RESULTS/"S100_PHASE23_SCREEN.json")
    promoted=[]
    if sc.get("RUN_PROMOTED_GROUPED"):
        for ctx in (128,1024,4096):
            d=load(RESULTS/f"S100_PHASE23_GROUPED_CTX{ctx}_PROMOTED.json")
            promoted.append({"context":ctx,"status":d.get("status"),
              "correctness_green":d.get("correctness_green"),"summary":d.get("summary")})
    complete=bool(
      st.get("status")=="measured" and sc
      and (not sc.get("RUN_PROMOTED_GROUPED")
           or (len(promoted)==3 and all(x["status"]=="measured" for x in promoted)))
    )
    correct=bool(st.get("GPU_GROUPED_CORRECTNESS_GREEN"))
    pg=bool(promoted and all(x["correctness_green"] for x in promoted))
    med=[float(x["summary"]["median_ms"]) for x in promoted
         if x.get("summary") and x["summary"].get("median_ms") is not None]
    target40=bool(pg and len(med)==3 and max(med)<=40.0)
    draft32=bool(pg and len(med)==3 and max(med)<=32.0)
    gain=sc.get("grouped_gain_fraction")
    p24=bool(correct and gain is not None and float(gain)>=.05 and not target40)

    if draft32:route="OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target40:route="TARGET_100_CEILING_OPEN_OPTIMIZE_DRAFTER_HEADROOM"
    elif p24:route="PROFILE_POST_GROUPED_GRAPH_AND_ATTACK_NEXT_DOMINANT_FAMILY"
    elif correct and sc.get("measurement_stable"):
        route="GPU_GROUPING_CORRECT_BUT_UNECONOMIC_KEEP_V6_GRAPH_PARENT"
    elif correct:
        route="REPEAT_THERMALLY_STABLE_GROUPED_GRAPH_SCREEN"
    else:
        route="REPAIR_GPU_GROUPED_MOE_CORRECTNESS"

    out={"kind":"s100_phase23_summary","created_utc":utc_now(),
      "instrumentation_complete":complete,
      "GPU_GROUPED_CORRECTNESS_GREEN":correct,
      "census":st.get("census"),
      "same_era_screen":sc,
      "promoted_contexts":promoted,
      "PHASE23_TARGET_40MS_OPEN":target40,
      "DRAFTER_SHOOTOUT_OPEN":draft32,
      "PHASE24_MOE_NEXT_OPEN":p24,
      "NEXT_ROUTE":route,
      "S100_SINGLE_ACHIEVED":False,
      "claim_boundary":"GPU grouped MoE perfect-draft target-only H4; no drafter"}
    RESULTS.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,out,archive=True)
    text=(
      "S100 PHASE 23 — GPU GROUPED MOE\n"
      f"Instrumentation complete: {complete}\n"
      f"GPU_GROUPED_CORRECTNESS_GREEN: {correct}\n"
      f"Grouped gain fraction @1024: {gain}\n"
      f"PHASE23_TARGET_40MS_OPEN: {target40}\n"
      f"DRAFTER_SHOOTOUT_OPEN: {draft32}\n"
      f"PHASE24_MOE_NEXT_OPEN: {p24}\n"
      f"NEXT_ROUTE: {route}\n"
      "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS/"S100_PHASE23_SUMMARY.txt").write_text(text,encoding="utf-8")
    print(text);return 0
if __name__=="__main__":raise SystemExit(main())
