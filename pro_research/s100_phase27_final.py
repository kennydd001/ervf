from __future__ import annotations

import json
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase27_common import RESULTS,robust_cv


def load(tag):
    try:
        return json.loads(
          (RESULTS/f"S100_PHASE27_{tag.upper()}.json").read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def med(d):
    try:return float(d["summary"]["median_ms"])
    except Exception:return None


def exact(d):
    return bool(
      d.get("status")=="measured"
      and d.get("correctness_green")
      and d.get("summary",{}).get("all_token_exact")
    )


def screen():
    pa,ca,cb,pb=(
      load("FINAL_PARENT_A"),load("FINAL_CANDIDATE_A"),
      load("FINAL_CANDIDATE_B"),load("FINAL_PARENT_B")
    )
    vals={
      "parent_a":med(pa),"candidate_a":med(ca),
      "candidate_b":med(cb),"parent_b":med(pb),
    }
    complete=all(v is not None for v in vals.values())
    state={}
    try:
        state=json.loads(
          (RESULTS/"S100_PHASE27_STATE_CHECK.json").read_text(encoding="utf-8")
        )
    except Exception:pass
    correctness=all(exact(x) for x in (pa,ca,cb,pb))
    if complete:
        pm=(vals["parent_a"]+vals["parent_b"])/2.0
        cm=(vals["candidate_a"]+vals["candidate_b"])/2.0
        pd=abs(vals["parent_a"]-vals["parent_b"])/pm
        cd=abs(vals["candidate_a"]-vals["candidate_b"])/cm
        gain=1.0-cm/pm
    else:
        pm=cm=pd=cd=gain=None
    stable=bool(complete and pd<=.07 and cd<=.07)
    open_thermal=bool(
      correctness and state.get("SELECTED_STATE_GREEN")
      and stable and gain is not None and gain>=.02
    )
    out={
      "kind":"s100_phase27_final_screen",
      "created_utc":utc_now(),
      "medians_ms":vals,
      "parent_midpoint_ms":pm,
      "candidate_midpoint_ms":cm,
      "parent_relative_drift":pd,
      "candidate_relative_drift":cd,
      "gain_fraction":gain,
      "candidate_ms_per_useful_token":None if cm is None else cm/4.0,
      "candidate_target_only_tok_s":None if cm is None else 4000.0/cm,
      "correctness_green":correctness,
      "state_green":bool(state.get("SELECTED_STATE_GREEN")),
      "stable":stable,
      "RUN_THERMAL_ADOPTION":open_thermal,
    }
    write_json_atomic(
      RESULTS/"S100_PHASE27_FINAL_SCREEN.json",out,archive=True
    )
    print(json.dumps(out,indent=2))
    return out


def thermal():
    screen_d={}
    try:
        screen_d=json.loads(
          (RESULTS/"S100_PHASE27_FINAL_SCREEN.json").read_text(encoding="utf-8")
        )
    except Exception:pass
    if not screen_d.get("RUN_THERMAL_ADOPTION"):
        out={
          "kind":"s100_phase27_thermal_adjudication",
          "created_utc":utc_now(),
          "status":"not_run_screen_closed",
          "ADOPT_PHASE27":False,
        }
        write_json_atomic(
          RESULTS/"S100_PHASE27_THERMAL_ADJUDICATION.json",out,archive=True
        )
        print(json.dumps(out,indent=2))
        return out

    rounds=[]
    paired=[]
    pmed=[]
    cmed=[]
    correctness=True
    aligned=True
    for r in (1,2,3,4):
        p=load(f"THERMAL_R{r}_PARENT")
        c=load(f"THERMAL_R{r}_CANDIDATE")
        correctness &= exact(p) and exact(c)
        pm,cm=med(p),med(c)
        if pm is None or cm is None:
            rounds.append({"round":r,"complete":False})
            continue
        pmed.append(pm);cmed.append(cm)
        pr=p.get("records") or []
        cr=c.get("records") or []
        same=(
          len(pr)==len(cr)==16
          and [int(x["pos"]) for x in pr]==[int(x["pos"]) for x in cr]
        )
        aligned &= same
        local=[]
        if same:
            for a,b in zip(pr,cr):
                g=1.0-float(b["ms"])/float(a["ms"])
                local.append(g);paired.append(g)
        rounds.append({
          "round":r,"complete":True,
          "parent_median_ms":pm,
          "candidate_median_ms":cm,
          "round_gain_fraction":1.0-cm/pm,
          "positions_aligned":same,
          "paired_gain_median":float(np.median(local)) if local else None,
          "parent_telemetry":p.get("telemetry"),
          "candidate_telemetry":c.get("telemetry"),
        })

    gains=[x["round_gain_fraction"] for x in rounds if x.get("complete")]
    complete=bool(
      len(pmed)==4 and len(cmed)==4 and len(paired)==64 and aligned
    )
    mr=float(np.median(gains)) if gains else None
    mp=float(np.median(paired)) if paired else None
    pos=sum(1 for x in gains if x>0)
    pcv=robust_cv(pmed) if len(pmed)==4 else None
    ccv=robust_cv(cmed) if len(cmed)==4 else None

    gates={
      "complete":complete,
      "correctness":bool(correctness),
      "positions_aligned":bool(aligned),
      "median_round_gain_ge_5pct":bool(mr is not None and mr>=.05),
      "median_pair_gain_ge_5pct":bool(mp is not None and mp>=.05),
      "positive_rounds_ge_3":pos>=3,
      "parent_cv_le_5pct":bool(pcv is not None and pcv<=.05),
      "candidate_cv_le_5pct":bool(ccv is not None and ccv<=.05),
    }
    adopt=all(gates.values())
    out={
      "kind":"s100_phase27_thermal_adjudication",
      "status":"measured","created_utc":utc_now(),
      "rounds":rounds,
      "parent_medians_ms":pmed,
      "candidate_medians_ms":cmed,
      "round_gain_fractions":gains,
      "median_round_gain_fraction":mr,
      "paired_window_count":len(paired),
      "median_paired_window_gain_fraction":mp,
      "paired_gain_p10":float(np.percentile(paired,10)) if paired else None,
      "paired_gain_p90":float(np.percentile(paired,90)) if paired else None,
      "positive_rounds":pos,
      "parent_robust_cv":pcv,
      "candidate_robust_cv":ccv,
      "gates":gates,
      "ADOPT_PHASE27":adopt,
    }
    write_json_atomic(
      RESULTS/"S100_PHASE27_THERMAL_ADJUDICATION.json",out,archive=True
    )
    print(json.dumps(out,indent=2))
    return out


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("screen","thermal"),required=True)
    args=ap.parse_args()
    out=screen() if args.stage=="screen" else thermal()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
