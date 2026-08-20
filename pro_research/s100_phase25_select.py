from __future__ import annotations

import json

from common import utc_now,write_json_atomic
from s100_phase25_common import (
    RESULTS,VARIANTS,OFFICIAL_PARENT_H8_MS,ADOPTION_ABS_MS,STRONG_MS,BREAKTHROUGH_MS,S100_MS,
)
OUT=RESULTS/"S100_PHASE25_SELECTION.json"

def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    parent=load(RESULTS/"S100_PHASE25_SCREEN_PARENT_CTX1024.json")
    pmed=(parent.get("summary") or {}).get("median_ms")
    arms=[]
    for v in VARIANTS:
        m=load(RESULTS/f"S100_PHASE25_SCREEN_{v.upper()}_CTX1024.json")
        s=load(RESULTS/f"S100_PHASE25_STATE_CHECK_{v.upper()}.json")
        med=(m.get("summary") or {}).get("median_ms")
        rec={"variant":v,"status":m.get("status"),"state_green":bool(s.get("H8_STATE_GREEN")),
             "median_ms":med,"tok_s":None if med is None else 8000.0/float(med),
             "fresh_gain_fraction":None if med is None or pmed is None else 1.0-float(med)/float(pmed),
             "official_gain_fraction":None if med is None else 1.0-float(med)/OFFICIAL_PARENT_H8_MS,
             "meets_abs_adoption":bool(med is not None and float(med)<=ADOPTION_ABS_MS)}
        arms.append(rec)
    eligible=[x for x in arms if x["status"]=="measured" and x["state_green"] and x["median_ms"] is not None]
    eligible.sort(key=lambda x:float(x["median_ms"]))
    selected=eligible[0] if eligible else None
    thermal_open=bool(selected and selected["meets_abs_adoption"] and selected["fresh_gain_fraction"] is not None
                      and selected["fresh_gain_fraction"]>=.05)
    med=None if selected is None else float(selected["median_ms"])
    out={"kind":"s100_phase25_selection","created_utc":utc_now(),"parent_fresh_median_ms":pmed,
      "official_parent_h8_ms":OFFICIAL_PARENT_H8_MS,"adoption_abs_ms":ADOPTION_ABS_MS,
      "arms":arms,"selected":selected,"H8_STATE_GREEN_ANY":bool(eligible),
      "THERMAL_ADOPTION_OPEN":thermal_open,
      "SCREEN_STRONG":bool(med is not None and med<=STRONG_MS),
      "SCREEN_BREAKTHROUGH":bool(med is not None and med<=BREAKTHROUGH_MS),
      "SCREEN_S100":bool(med is not None and med<=S100_MS)}
    write_json_atomic(OUT,out,archive=True);print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
