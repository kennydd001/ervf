from __future__ import annotations

import json
import numpy as np

from common import utc_now, write_json_atomic
from s100_phase26_common import RESULTS, robust_cv

OUT=RESULTS/"S100_PHASE26_THERMAL_ADJUDICATION.json"

def load(tag):
    try:
        return json.loads(
          (RESULTS/f"S100_PHASE26_{tag}.json").read_text(encoding="utf-8")
        )
    except Exception:
        return {}

def main():
    screen=json.loads(
      (RESULTS/"S100_PHASE26_SCREEN.json").read_text(encoding="utf-8")
    )
    h=screen.get("selected_horizon")
    if h not in (4,8):
        out={
          "kind":"s100_phase26_thermal_adjudication",
          "created_utc":utc_now(),
          "status":"not_run_no_selected_candidate",
          "ADOPT_OVERLAP":False,
        }
        write_json_atomic(OUT,out,archive=True)
        print(json.dumps(out,indent=2))
        return 0

    rounds=[]
    pair_gains=[]
    parent_meds=[]
    cand_meds=[]
    correct=True
    aligned=True

    for r in (1,2,3,4):
        p=load(f"THERMAL_R{r}_PARENT")
        c=load(f"THERMAL_R{r}_CANDIDATE")
        ok=bool(
          p.get("status")=="measured"
          and c.get("status")=="measured"
          and p.get("correctness_green")
          and c.get("correctness_green")
          and int(p.get("horizon",0))==h
          and int(c.get("horizon",0))==h
        )
        correct &= ok
        pm=(p.get("summary") or {}).get("median_ms")
        cm=(c.get("summary") or {}).get("median_ms")
        if pm is None or cm is None:
            rounds.append({"round":r,"complete":False})
            continue
        pm=float(pm);cm=float(cm)
        parent_meds.append(pm);cand_meds.append(cm)
        pr=p.get("records") or [];cr=c.get("records") or []
        same=(
          len(pr)==len(cr)==16
          and [int(x["pos"]) for x in pr]
              ==[int(x["pos"]) for x in cr]
        )
        aligned &= same
        local=[]
        if same:
            for a,b in zip(pr,cr):
                g=1.0-float(b["ms"])/float(a["ms"])
                local.append(g);pair_gains.append(g)
        rounds.append({
          "round":r,"complete":True,
          "parent_median_ms":pm,
          "candidate_median_ms":cm,
          "round_gain_fraction":1.0-cm/pm,
          "positions_aligned":same,
          "paired_gain_median":(
            float(np.median(local)) if local else None
          ),
          "parent_telemetry":p.get("telemetry"),
          "candidate_telemetry":c.get("telemetry"),
        })

    rg=[x["round_gain_fraction"] for x in rounds if x.get("complete")]
    complete=bool(
      len(parent_meds)==4 and len(cand_meds)==4
      and len(pair_gains)==64 and aligned
    )
    mr=float(np.median(rg)) if rg else None
    mp=float(np.median(pair_gains)) if pair_gains else None
    positive=sum(1 for x in rg if x>0)
    pcv=robust_cv(parent_meds) if len(parent_meds)==4 else None
    ccv=robust_cv(cand_meds) if len(cand_meds)==4 else None

    gates={
      "complete":complete,
      "correctness":bool(correct),
      "positions_aligned":bool(aligned),
      "median_round_gain_ge_5pct":bool(mr is not None and mr>=.05),
      "median_pair_gain_ge_5pct":bool(mp is not None and mp>=.05),
      "positive_rounds_ge_3":positive>=3,
      "parent_cv_le_5pct":bool(pcv is not None and pcv<=.05),
      "candidate_cv_le_5pct":bool(ccv is not None and ccv<=.05),
    }
    adopt=all(gates.values())
    out={
      "kind":"s100_phase26_thermal_adjudication",
      "status":"measured","created_utc":utc_now(),
      "selected_horizon":h,"rounds":rounds,
      "parent_medians_ms":parent_meds,
      "candidate_medians_ms":cand_meds,
      "round_gain_fractions":rg,
      "median_round_gain_fraction":mr,
      "paired_window_count":len(pair_gains),
      "median_paired_window_gain_fraction":mp,
      "paired_gain_p10":(
        float(np.percentile(pair_gains,10)) if pair_gains else None
      ),
      "paired_gain_p90":(
        float(np.percentile(pair_gains,90)) if pair_gains else None
      ),
      "positive_rounds":positive,
      "parent_robust_cv":pcv,
      "candidate_robust_cv":ccv,
      "gates":gates,
      "ADOPT_OVERLAP":adopt,
    }
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
