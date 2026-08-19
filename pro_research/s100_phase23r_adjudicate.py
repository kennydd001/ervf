from __future__ import annotations

import json
import math
from statistics import median

import numpy as np

from common import REPO,utc_now,write_json_atomic

RESULTS=REPO/"pro_research"/"results"/"s100_phase23r"
OUT=RESULTS/"S100_PHASE23R_ADJUDICATION.json"

ROUNDS={
  1:("R1_PARENT","R1_GROUPED"),
  2:("R2_PARENT","R2_GROUPED"),
  3:("R3_PARENT","R3_GROUPED"),
  4:("R4_PARENT","R4_GROUPED"),
}

def load(tag):
    p=RESULTS/f"S100_PHASE23R_{tag}.json"
    try:return json.loads(p.read_text(encoding="utf-8"))
    except Exception:return {}

def robust_cv(vals):
    a=np.asarray(vals,np.float64)
    med=float(np.median(a))
    mad=float(np.median(np.abs(a-med)))
    return float(1.4826*mad/max(abs(med),1e-30))

def telemetry_compact(d):
    t=d.get("telemetry") or {}
    out={}
    for key in ("after_graph_setup","after_prefill","after_warmup","after_measure"):
        row=t.get(key) or {}
        out[key]={k:row.get(k) for k in (
          "temperature.gpu","pstate","clocks.sm","clocks.mem","power.draw",
          "utilization.gpu","memory.used","error"
        ) if k in row}
    return out

def main():
    round_rows=[]
    pair_gains=[]
    parent_meds=[]
    grouped_meds=[]
    correctness=True
    position_alignment=True

    for r,(ptag,gtag) in ROUNDS.items():
        p=load(ptag);g=load(gtag)
        ok=bool(
          p.get("status")=="measured" and g.get("status")=="measured"
          and p.get("correctness_green") and g.get("correctness_green")
        )
        correctness &= ok
        pm=(p.get("summary") or {}).get("median_ms")
        gm=(g.get("summary") or {}).get("median_ms")
        if pm is None or gm is None:
            round_rows.append({"round":r,"complete":False})
            continue
        pm=float(pm);gm=float(gm)
        parent_meds.append(pm);grouped_meds.append(gm)

        prec=p.get("records") or [];grec=g.get("records") or []
        posp=[int(x["pos"]) for x in prec]
        posg=[int(x["pos"]) for x in grec]
        aligned=(posp==posg and len(prec)==16 and len(grec)==16)
        position_alignment &= aligned
        local=[]
        if aligned:
            for a,b in zip(prec,grec):
                local.append(1.0-float(b["ms"])/float(a["ms"]))
                pair_gains.append(local[-1])
        gain=1.0-gm/pm
        round_rows.append({
          "round":r,"complete":True,
          "parent_median_ms":pm,"grouped_median_ms":gm,
          "round_gain_fraction":gain,
          "positions_aligned":aligned,
          "paired_block_gain_median":(
             float(np.median(local)) if local else None
          ),
          "parent_telemetry":telemetry_compact(p),
          "grouped_telemetry":telemetry_compact(g),
        })

    complete=(len(parent_meds)==4 and len(grouped_meds)==4
              and len(pair_gains)==64 and position_alignment)
    rg=[x["round_gain_fraction"] for x in round_rows if x.get("complete")]
    median_round=float(np.median(rg)) if rg else None
    median_pair=float(np.median(pair_gains)) if pair_gains else None
    positive_rounds=sum(1 for x in rg if x>0)
    pcv=robust_cv(parent_meds) if len(parent_meds)==4 else None
    gcv=robust_cv(grouped_meds) if len(grouped_meds)==4 else None

    gates={
      "complete_4_rounds_64_pairs":complete,
      "all_correctness_green":bool(correctness),
      "positions_aligned":bool(position_alignment),
      "median_round_gain_ge_5pct":bool(median_round is not None and median_round>=.05),
      "median_block_pair_gain_ge_5pct":bool(median_pair is not None and median_pair>=.05),
      "positive_rounds_ge_3of4":positive_rounds>=3,
      "parent_robust_cv_le_5pct":bool(pcv is not None and pcv<=.05),
      "grouped_robust_cv_le_5pct":bool(gcv is not None and gcv<=.05),
    }
    promote=all(gates.values())

    if promote:
        route="PROMOTE_GPU_GROUPED_AND_RUN_CONTEXTS"
    elif complete and median_round is not None and median_round>0:
        route="GROUPED_CORRECT_BUT_SUB5_PROFILE_M1_AND_GROUPING_OVERHEAD"
    elif complete:
        route="KEEP_V6_PARENT_PROFILE_NEXT_DOMINANT_FAMILY"
    else:
        route="REPAIR_OR_REPEAT_INCOMPLETE_THERMAL_SCREEN"

    out={
      "kind":"s100_phase23r_adjudication","created_utc":utc_now(),
      "rounds":round_rows,
      "parent_process_medians_ms":parent_meds,
      "grouped_process_medians_ms":grouped_meds,
      "round_gain_fractions":rg,
      "median_round_gain_fraction":median_round,
      "paired_block_count":len(pair_gains),
      "median_paired_block_gain_fraction":median_pair,
      "paired_block_gain_p10":(
        float(np.percentile(pair_gains,10)) if pair_gains else None
      ),
      "paired_block_gain_p90":(
        float(np.percentile(pair_gains,90)) if pair_gains else None
      ),
      "positive_rounds":positive_rounds,
      "parent_robust_cv":pcv,
      "grouped_robust_cv":gcv,
      "gates":gates,
      "PROMOTE_GPU_GROUPED_MOE":promote,
      "NEXT_ROUTE":route,
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
