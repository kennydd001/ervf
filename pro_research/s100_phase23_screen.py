from __future__ import annotations
import json
from common import write_json_atomic,utc_now
from s100_phase23_common import RESULTS

OUT=RESULTS/"S100_PHASE23_SCREEN.json"
def load(name):
    try:return json.loads((RESULTS/name).read_text(encoding="utf-8"))
    except Exception:return {}
def med(d):
    try:return float(d["summary"]["median_ms"])
    except Exception:return None

def main():
    pa=load("S100_PHASE23_PARENT_CTX1024_PARENT_A.json")
    ga=load("S100_PHASE23_GROUPED_CTX1024_GROUPED_A.json")
    gb=load("S100_PHASE23_GROUPED_CTX1024_GROUPED_B.json")
    pb=load("S100_PHASE23_PARENT_CTX1024_PARENT_B.json")
    vals={"parent_a":med(pa),"grouped_a":med(ga),"grouped_b":med(gb),"parent_b":med(pb)}
    complete=all(x is not None for x in vals.values())
    correct=bool(all(d.get("correctness_green") for d in (pa,ga,gb,pb)))
    if complete:
        pmid=(vals["parent_a"]+vals["parent_b"])/2
        gmid=(vals["grouped_a"]+vals["grouped_b"])/2
        pdr=abs(vals["parent_a"]-vals["parent_b"])/pmid
        gdr=abs(vals["grouped_a"]-vals["grouped_b"])/gmid
        gain=(pmid-gmid)/pmid
    else:
        pmid=gmid=pdr=gdr=gain=None
    stable=bool(complete and pdr<=.07 and gdr<=.07)
    promote=bool(correct and stable and gain is not None and gain>=.05)
    out={"kind":"s100_phase23_screen","created_utc":utc_now(),
      "medians_ms":vals,"parent_midpoint_ms":pmid,"grouped_midpoint_ms":gmid,
      "parent_relative_drift":pdr,"grouped_relative_drift":gdr,
      "grouped_gain_fraction":gain,"correctness_green":correct,
      "measurement_stable":stable,"RUN_PROMOTED_GROUPED":promote}
    RESULTS.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
