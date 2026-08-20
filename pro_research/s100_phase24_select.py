from __future__ import annotations

import json

from common import utc_now,write_json_atomic
from s100_phase24_common import RESULTS

OUT=RESULTS/"S100_PHASE24_SELECTION.json"
KS=(0,4,8,12,16,23)

def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}

def main():
    rows=[]
    b=load(RESULTS/"S100_PHASE24_SCREEN_BASELINE_CTX1024.json")
    rows.append({
      "label":"baseline","arm":"baseline","k":None,
      "status":b.get("status"),
      "correctness_green":b.get("correctness_green"),
      "median_ms":(b.get("summary") or {}).get("median_ms"),
      "tok_s":(b.get("summary") or {}).get("target_only_tok_s"),
      "actual_plane_bytes":0,
      "config":b.get("config"),
    })
    for k in KS:
        d=load(RESULTS/f"S100_PHASE24_SCREEN_SYNTH_K{k}_CTX1024.json")
        rows.append({
          "label":f"synth_k{k}","arm":"synth","k":k,
          "status":d.get("status"),
          "correctness_green":d.get("correctness_green"),
          "median_ms":(d.get("summary") or {}).get("median_ms"),
          "tok_s":(d.get("summary") or {}).get("target_only_tok_s"),
          "actual_plane_bytes":d.get("actual_plane_bytes"),
          "config":d.get("config"),
          "error":(d.get("error") or {}).get("message"),
        })

    green=[
      x for x in rows
      if x.get("status")=="measured"
      and x.get("correctness_green")
      and x.get("median_ms") is not None
    ]
    selected=None
    if green:
        best=min(float(x["median_ms"]) for x in green)
        tied=[x for x in green if float(x["median_ms"])<=best*1.01]
        selected=min(
          tied,
          key=lambda x:(
            int(x.get("actual_plane_bytes") or 0),
            0 if x["label"]=="baseline" else 1,
          )
        )

    base=next((x for x in green if x["label"]=="baseline"),None)
    gain=None
    if selected and base:
        gain=(
          float(base["median_ms"])-float(selected["median_ms"])
        )/float(base["median_ms"])

    out={"kind":"s100_phase24_selection","created_utc":utc_now(),
      "arms":rows,"selected":selected,
      "selected_label":None if selected is None else selected["label"],
      "selected_gain_fraction_vs_baseline":gain,
      "RUN_STATE_CHECK":bool(selected and selected["label"]!="baseline"),
      "RUN_THERMAL_ADJUDICATION":bool(
        selected and selected["label"]!="baseline"
      )}
    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
