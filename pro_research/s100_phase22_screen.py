from __future__ import annotations
import json
from common import write_json_atomic,utc_now
from s100_phase22_common import RESULTS

OUT=RESULTS/"S100_PHASE22_SCREEN.json"

def load(name):
    try:return json.loads((RESULTS/name).read_text(encoding="utf-8"))
    except Exception:return {}

def med(d):
    try:return float(d["summary"]["median_ms"])
    except Exception:return None

def main():
    ea=load("S100_PHASE22_EAGER_CTX1024_EAGER_A.json")
    ga=load("S100_PHASE22_GRAPH_CTX1024_GRAPH_A.json")
    gb=load("S100_PHASE22_GRAPH_CTX1024_GRAPH_B.json")
    eb=load("S100_PHASE22_EAGER_CTX1024_EAGER_B.json")
    state=load("S100_PHASE22_GRAPH_STATE_CHECK.json")

    vals={"eager_a":med(ea),"graph_a":med(ga),"graph_b":med(gb),"eager_b":med(eb)}
    complete=all(x is not None for x in vals.values())
    correct=bool(
      state.get("GRAPH_CORRECTNESS_GREEN")
      and all(d.get("correctness_green") for d in (ea,ga,gb,eb))
    )
    if complete:
        emid=(vals["eager_a"]+vals["eager_b"])/2.0
        gmid=(vals["graph_a"]+vals["graph_b"])/2.0
        edrift=abs(vals["eager_a"]-vals["eager_b"])/emid
        gdrift=abs(vals["graph_a"]-vals["graph_b"])/gmid
        gain=(emid-gmid)/emid
    else:
        emid=gmid=edrift=gdrift=gain=None

    stable=bool(
      complete and edrift<=0.05 and gdrift<=0.05
    )
    promote=bool(
      correct and stable and gmid is not None and emid is not None and gmid<=emid
    )
    out={
      "kind":"s100_phase22_screen","created_utc":utc_now(),
      "medians_ms":vals,
      "eager_midpoint_ms":emid,
      "graph_midpoint_ms":gmid,
      "eager_relative_drift":edrift,
      "graph_relative_drift":gdrift,
      "graph_gain_fraction":gain,
      "correctness_green":correct,
      "measurement_stable":stable,
      "RUN_PROMOTED_GRAPH":promote,
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
