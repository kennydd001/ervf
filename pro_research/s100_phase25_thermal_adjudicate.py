from __future__ import annotations

import json
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase25_common import RESULTS,ADOPTION_ABS_MS,S100_MS
OUT=RESULTS/"S100_PHASE25_THERMAL_ADJUDICATION.json"
ROUNDS={1:("R1_PARENT","R1_SELECTED"),2:("R2_PARENT","R2_SELECTED"),3:("R3_PARENT","R3_SELECTED"),4:("R4_PARENT","R4_SELECTED")}

def load(tag):
    try:return json.loads((RESULTS/f"S100_PHASE25_{tag}.json").read_text(encoding="utf-8"))
    except Exception:return {}
def robust_cv(vals):
    a=np.asarray(vals,np.float64);m=float(np.median(a));return float(1.4826*np.median(np.abs(a-m))/max(abs(m),1e-30))
def main():
    rounds=[];pairs=[];pm=[];sm=[];correct=True;aligned=True;variant=None
    for r,(pt,st) in ROUNDS.items():
        p=load(pt);s=load(st);variant=variant or s.get("variant")
        ok=bool(p.get("status")=="measured" and s.get("status")=="measured" and p.get("correctness_green") and s.get("correctness_green"));correct&=ok
        pmed=(p.get("summary") or {}).get("median_ms");smed=(s.get("summary") or {}).get("median_ms")
        if pmed is None or smed is None:rounds.append({"round":r,"complete":False});continue
        pmed=float(pmed);smed=float(smed);pm.append(pmed);sm.append(smed);pr=p.get("records") or [];sr=s.get("records") or []
        same=(len(pr)==len(sr)==16 and [int(x["pos"]) for x in pr]==[int(x["pos"]) for x in sr]);aligned&=same;local=[]
        if same:
            for x,y in zip(pr,sr):local.append(1.0-float(y["ms"])/float(x["ms"]));pairs.append(local[-1])
        rounds.append({"round":r,"complete":True,"parent_median_ms":pmed,"selected_median_ms":smed,
          "round_gain_fraction":1.0-smed/pmed,"positions_aligned":same,"paired_gain_median":float(np.median(local)) if local else None,
          "parent_telemetry":p.get("telemetry"),"selected_telemetry":s.get("telemetry")})
    rg=[x["round_gain_fraction"] for x in rounds if x.get("complete")];complete=bool(len(pm)==4 and len(sm)==4 and len(pairs)==64 and aligned)
    mr=float(np.median(rg)) if rg else None;mp=float(np.median(pairs)) if pairs else None;pos=sum(1 for x in rg if x>0)
    pcv=robust_cv(pm) if len(pm)==4 else None;scv=robust_cv(sm) if len(sm)==4 else None;selected_med=float(np.median(sm)) if sm else None
    gates={"complete":complete,"correctness":bool(correct),"positions_aligned":bool(aligned),
      "median_round_gain_ge_5pct":bool(mr is not None and mr>=.05),"median_pair_gain_ge_5pct":bool(mp is not None and mp>=.05),
      "positive_rounds_ge_3":pos>=3,"parent_cv_le_5pct":bool(pcv is not None and pcv<=.05),
      "selected_cv_le_5pct":bool(scv is not None and scv<=.05),"absolute_adoption_ms":bool(selected_med is not None and selected_med<=ADOPTION_ABS_MS)}
    adopt=all(gates.values());out={"kind":"s100_phase25_thermal_adjudication","created_utc":utc_now(),"variant":variant,"rounds":rounds,
      "parent_medians_ms":pm,"selected_medians_ms":sm,"selected_median_of_rounds_ms":selected_med,"round_gain_fractions":rg,
      "median_round_gain_fraction":mr,"paired_block_count":len(pairs),"median_paired_block_gain_fraction":mp,
      "paired_gain_p10":float(np.percentile(pairs,10)) if pairs else None,"paired_gain_p90":float(np.percentile(pairs,90)) if pairs else None,
      "positive_rounds":pos,"parent_robust_cv":pcv,"selected_robust_cv":scv,"gates":gates,"H8_ADOPTED":adopt,
      "S100_TARGET_ONLY_THERMAL":bool(adopt and selected_med is not None and selected_med<=S100_MS)}
    write_json_atomic(OUT,out,archive=True);print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
