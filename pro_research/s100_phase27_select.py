from __future__ import annotations

import argparse
import json

from common import utc_now,write_json_atomic
from s100_phase27_common import RESULTS


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


def anchors(a,b):
    ma,mb=med(a),med(b)
    if ma is None or mb is None:
        return None,None,None
    mid=(ma+mb)/2.0
    drift=abs(ma-mb)/mid
    return mid,drift,(ma,mb)


def measured_exact(d):
    return bool(
      d.get("status")=="measured"
      and d.get("correctness_green")
      and d.get("summary",{}).get("all_token_exact")
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("geometry","pipeline","combo"),required=True)
    args=ap.parse_args()

    if args.stage=="geometry":
        pa,pb=load("GEOM_PARENT_A"),load("GEOM_PARENT_B")
        pm,pd,pr=anchors(pa,pb)
        rows=[]
        for y in (4,8,16,32):
            d=load(f"GEOM_G{y}")
            m=med(d)
            rows.append({
              "gather_y":y,"median_ms":m,
              "ms_per_useful_token":None if m is None else m/4.0,
              "gain_vs_parent":None if m is None or pm is None else 1.0-m/pm,
              "measured_exact":measured_exact(d),
              "telemetry":d.get("telemetry"),
            })
        valid=[x for x in rows if x["measured_exact"] and x["median_ms"] is not None]
        sel=min(valid,key=lambda x:x["median_ms"]) if valid else None
        ctrl=next((x for x in rows if x["gather_y"]==32),None)
        control_delta=(
          None if ctrl is None or ctrl["median_ms"] is None or pm is None
          else ctrl["median_ms"]/pm-1.0
        )
        out={
          "kind":"s100_phase27_geometry_selection",
          "created_utc":utc_now(),
          "parent_midpoint_ms":pm,
          "parent_relative_drift":pd,
          "parent_anchors_ms":pr,
          "parent_stable":bool(pd is not None and pd<=.07),
          "arms":rows,
          "selected":sel,
          "selected_gather_y":None if sel is None else sel["gather_y"],
          "y32_control_delta_fraction":control_delta,
          "GEOMETRY_SELECTION_GREEN":bool(
            sel is not None and pd is not None and pd<=.07
            and control_delta is not None and abs(control_delta)<=.03
          ),
        }
        path=RESULTS/"S100_PHASE27_GEOMETRY_SELECTION.json"

    elif args.stage=="pipeline":
        geom=json.loads(
          (RESULTS/"S100_PHASE27_GEOMETRY_SELECTION.json").read_text(encoding="utf-8")
        )
        gy=geom.get("selected_gather_y")
        if gy not in (4,8,16,32):
            raise RuntimeError("geometry selection missing")
        pa,pb=load("PIPE_PARENT_A"),load("PIPE_PARENT_B")
        pm,pd,pr=anchors(pa,pb)
        rows=[]
        for b in (1,2,3,4):
            d=load(f"PIPE_B{b}")
            m=med(d)
            rows.append({
              "batches":b,"gather_y":gy,"median_ms":m,
              "ms_per_useful_token":None if m is None else m/4.0,
              "gain_vs_parent":None if m is None or pm is None else 1.0-m/pm,
              "measured_exact":measured_exact(d),
              "telemetry":d.get("telemetry"),
            })
        valid=[x for x in rows if x["measured_exact"] and x["median_ms"] is not None]
        sel=min(valid,key=lambda x:x["median_ms"]) if valid else None
        out={
          "kind":"s100_phase27_pipeline_selection",
          "created_utc":utc_now(),
          "selected_gather_y":gy,
          "parent_midpoint_ms":pm,
          "parent_relative_drift":pd,
          "parent_anchors_ms":pr,
          "parent_stable":bool(pd is not None and pd<=.07),
          "arms":rows,
          "selected":sel,
          "selected_batches":None if sel is None else sel["batches"],
          "PIPELINE_SELECTION_GREEN":bool(
            sel is not None and pd is not None and pd<=.07
          ),
        }
        path=RESULTS/"S100_PHASE27_PIPELINE_SELECTION.json"

    else:
        pipe=json.loads(
          (RESULTS/"S100_PHASE27_PIPELINE_SELECTION.json").read_text(encoding="utf-8")
        )
        gy=pipe.get("selected_gather_y")
        batches=pipe.get("selected_batches")
        if gy not in (4,8,16,32) or batches not in (1,2,3,4):
            raise RuntimeError("pipeline selection missing")
        pa,pb=load("COMBO_PARENT_A"),load("COMBO_PARENT_B")
        pm,pd,pr=anchors(pa,pb)
        rows=[]
        for name,ovl in (("pipe",False),("pipe_overlap",True)):
            d=load("COMBO_PIPE" if not ovl else "COMBO_OVL")
            m=med(d)
            rows.append({
              "name":name,"gather_y":gy,"batches":batches,
              "shared_overlap":ovl,"median_ms":m,
              "ms_per_useful_token":None if m is None else m/4.0,
              "gain_vs_parent":None if m is None or pm is None else 1.0-m/pm,
              "measured_exact":measured_exact(d),
              "telemetry":d.get("telemetry"),
            })
        valid=[x for x in rows if x["measured_exact"] and x["median_ms"] is not None]
        sel=min(valid,key=lambda x:x["median_ms"]) if valid else None
        out={
          "kind":"s100_phase27_selection",
          "created_utc":utc_now(),
          "parent_midpoint_ms":pm,
          "parent_relative_drift":pd,
          "parent_anchors_ms":pr,
          "parent_stable":bool(pd is not None and pd<=.07),
          "arms":rows,
          "selected":sel,
          "selected_variant":None if sel is None else {
            "gather_y":sel["gather_y"],
            "batches":sel["batches"],
            "shared_overlap":sel["shared_overlap"],
          },
          "COMBINATION_SELECTION_GREEN":bool(
            sel is not None and pd is not None and pd<=.07
          ),
        }
        path=RESULTS/"S100_PHASE27_SELECTION.json"

    RESULTS.mkdir(parents=True,exist_ok=True)
    write_json_atomic(path,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
