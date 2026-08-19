from __future__ import annotations

import json
from common import write_json_atomic,utc_now
from s100_phase21_common import RESULTS,ARMS

OUT=RESULTS/"S100_PHASE21_SELECTION.json"

def load(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except Exception:return None

def main():
    rows=[]
    for arm in ARMS:
        p=RESULTS/f"S100_PHASE21_SCREEN_{arm.upper()}_CTX1024.json"
        d=load(p)
        row={"arm":arm,"path":str(p),"status":None if d is None else d.get("status")}
        if d and d.get("status")=="measured":
            s=d.get("summary") or {}
            row.update({"correctness_green":bool(d.get("correctness_green")),
                        "median_ms":s.get("median_ms"),
                        "tok_s":s.get("target_only_tok_s")})
        rows.append(row)

    green=[x for x in rows if x.get("correctness_green") and x.get("median_ms") is not None]
    preference={"v18_device_rows":0,"v6_device_rows":1,
                "selective_grouped":2,"current_grouped":3}
    selected=None
    if green:
        best=min(float(x["median_ms"]) for x in green)
        tied=[x for x in green if float(x["median_ms"])<=best*1.01]
        selected=min(tied,key=lambda x:preference[x["arm"]])

    current=next((x for x in green if x["arm"]=="current_grouped"),None)
    gain=None
    if selected and current:
        gain=(float(current["median_ms"])-float(selected["median_ms"]))/float(current["median_ms"])

    out={
      "kind":"s100_phase21_selection","created_utc":utc_now(),
      "arms":rows,
      "selected":selected,
      "selected_arm":None if selected is None else selected["arm"],
      "gain_fraction_vs_current":gain,
      "RUN_PROMOTED_CONTEXTS":selected is not None,
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
