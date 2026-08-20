from __future__ import annotations

import json
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase24_common import RESULTS

OUT=RESULTS/"S100_PHASE24_THERMAL_ADJUDICATION.json"
ROUNDS={
  1:("R1_BASELINE","R1_SELECTED"),
  2:("R2_BASELINE","R2_SELECTED"),
  3:("R3_BASELINE","R3_SELECTED"),
  4:("R4_BASELINE","R4_SELECTED"),
}

def load(tag):
    try:
        return json.loads(
          (RESULTS/f"S100_PHASE24_{tag}.json").read_text(encoding="utf-8")
        )
    except Exception:
        return {}

def robust_cv(vals):
    a=np.asarray(vals,np.float64);m=float(np.median(a))
    return float(1.4826*np.median(np.abs(a-m))/max(abs(m),1e-30))

def main():
    rounds=[];pairs=[];bm=[];sm=[];correct=True;aligned=True
    for r,(bt,st) in ROUNDS.items():
        b=load(bt);s=load(st)
        ok=bool(
          b.get("status")=="measured" and s.get("status")=="measured"
          and b.get("correctness_green") and s.get("correctness_green")
        )
        correct &= ok
        bmed=(b.get("summary") or {}).get("median_ms")
        smed=(s.get("summary") or {}).get("median_ms")
        if bmed is None or smed is None:
            rounds.append({"round":r,"complete":False});continue
        bmed=float(bmed);smed=float(smed);bm.append(bmed);sm.append(smed)
        br=b.get("records") or [];sr=s.get("records") or []
        same=(
          len(br)==len(sr)==16
          and [int(x["pos"]) for x in br]==[int(x["pos"]) for x in sr]
        )
        aligned &= same
        local=[]
        if same:
            for x,y in zip(br,sr):
                local.append(1.0-float(y["ms"])/float(x["ms"]))
                pairs.append(local[-1])
        rounds.append({
          "round":r,"complete":True,
          "baseline_median_ms":bmed,"selected_median_ms":smed,
          "round_gain_fraction":1.0-smed/bmed,
          "positions_aligned":same,
          "paired_gain_median":float(np.median(local)) if local else None,
          "baseline_telemetry":b.get("telemetry"),
          "selected_telemetry":s.get("telemetry"),
        })

    rg=[x["round_gain_fraction"] for x in rounds if x.get("complete")]
    complete=bool(len(bm)==4 and len(sm)==4 and len(pairs)==64 and aligned)
    mr=float(np.median(rg)) if rg else None
    mp=float(np.median(pairs)) if pairs else None
    pos=sum(1 for x in rg if x>0)
    bcv=robust_cv(bm) if len(bm)==4 else None
    scv=robust_cv(sm) if len(sm)==4 else None
    gates={
      "complete":complete,"correctness":bool(correct),
      "positions_aligned":bool(aligned),
      "median_round_gain_ge_5pct":bool(mr is not None and mr>=.05),
      "median_pair_gain_ge_5pct":bool(mp is not None and mp>=.05),
      "positive_rounds_ge_3":pos>=3,
      "baseline_cv_le_5pct":bool(bcv is not None and bcv<=.05),
      "selected_cv_le_5pct":bool(scv is not None and scv<=.05),
    }
    adopt=all(gates.values())
    out={"kind":"s100_phase24_thermal_adjudication",
      "created_utc":utc_now(),"rounds":rounds,
      "baseline_medians_ms":bm,"selected_medians_ms":sm,
      "round_gain_fractions":rg,
      "median_round_gain_fraction":mr,
      "paired_block_count":len(pairs),
      "median_paired_block_gain_fraction":mp,
      "paired_gain_p10":float(np.percentile(pairs,10)) if pairs else None,
      "paired_gain_p90":float(np.percentile(pairs,90)) if pairs else None,
      "positive_rounds":pos,
      "baseline_robust_cv":bcv,"selected_robust_cv":scv,
      "gates":gates,"BEST_OF_ALL_ADOPTED":adopt}
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2));return 0

if __name__=="__main__":
    raise SystemExit(main())
