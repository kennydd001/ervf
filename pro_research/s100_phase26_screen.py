from __future__ import annotations

import json
import numpy as np

from common import utc_now, write_json_atomic
from s100_phase26_common import RESULTS

OUT=RESULTS/"S100_PHASE26_SCREEN.json"

def load(tag):
    try:
        return json.loads(
          (RESULTS/f"S100_PHASE26_{tag}.json").read_text(encoding="utf-8")
        )
    except Exception:
        return {}

def med(d):
    try:return float(d["summary"]["median_ms"])
    except Exception:return None

def evaluate(h):
    p1=load(f"H{h}_PARENT_A")
    c1=load(f"H{h}_OVERLAP_A")
    c2=load(f"H{h}_OVERLAP_B")
    p2=load(f"H{h}_PARENT_B")
    vals={
      "parent_a":med(p1),"candidate_a":med(c1),
      "candidate_b":med(c2),"parent_b":med(p2),
    }
    complete=all(x is not None for x in vals.values())
    correct=bool(all(
      x.get("status")=="measured" and x.get("correctness_green")
      for x in (p1,c1,c2,p2)
    ))
    if complete:
        pm=(vals["parent_a"]+vals["parent_b"])/2.0
        cm=(vals["candidate_a"]+vals["candidate_b"])/2.0
        pd=abs(vals["parent_a"]-vals["parent_b"])/pm
        cd=abs(vals["candidate_a"]-vals["candidate_b"])/cm
        gain=1.0-cm/pm
        candidate_ms_per_token=cm/float(h)
    else:
        pm=cm=pd=cd=gain=candidate_ms_per_token=None
    stable=bool(complete and pd<=.07 and cd<=.07)
    eligible=bool(correct and stable and gain is not None and gain>=.02)
    return {
      "horizon":h,"medians_ms":vals,
      "parent_midpoint_ms":pm,"candidate_midpoint_ms":cm,
      "parent_relative_drift":pd,"candidate_relative_drift":cd,
      "gain_fraction":gain,
      "candidate_ms_per_useful_token":candidate_ms_per_token,
      "correctness_green":correct,"stable":stable,
      "eligible_for_thermal":eligible,
    }

def main():
    rows=[evaluate(4),evaluate(8)]
    eligible=[x for x in rows if x["eligible_for_thermal"]]
    selected=min(
      eligible,key=lambda x:x["candidate_ms_per_useful_token"]
    ) if eligible else None
    out={
      "kind":"s100_phase26_screen","created_utc":utc_now(),
      "horizons":rows,
      "selected":selected,
      "selected_horizon":None if selected is None else selected["horizon"],
      "RUN_THERMAL_ADOPTION":selected is not None,
      "selection_rule":"lowest candidate ms/useful among stable >=2% screens",
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
