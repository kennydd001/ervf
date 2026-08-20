from __future__ import annotations

from collections import Counter,defaultdict
import json
import time
import traceback

import numpy as np

from common import REPO,utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,expected_for_block,release
from s100_phase24_common import (
    RESULTS,SynthesisConfig,make_synth,
)

OUT=RESULTS/"S100_PHASE24_PROFILE.json"
CTX=1024

def combine_h8(a,b):
    rows=[]
    for x,y in zip(a,b):
        if int(x["layer"])!=int(y["layer"]):
            raise RuntimeError("profile layer order drift")
        ids=np.asarray(x["ids"]+y["ids"],np.int32).reshape(-1)
        c=Counter(int(v) for v in ids)
        hist=Counter(c.values())
        rows.append({
          "layer":int(x["layer"]),
          "h4_streams_first":int(x["ngroups"]),
          "h4_streams_second":int(y["ngroups"]),
          "two_h4_streams":int(x["ngroups"]+y["ngroups"]),
          "h8_unique_experts":len(c),
          "h8_m_histogram":{str(k):int(v) for k,v in sorted(hist.items())},
          "h8_additional_stream_reduction_fraction":(
             1.0-len(c)/float(x["ngroups"]+y["ngroups"])
          ),
        })
    return rows

def main():
    payload={"kind":"s100_phase24_profile","status":"started",
      "context":CTX,"started_utc":utc_now(),
      "claim_boundary":"synchronized post-grouped diagnostic; not throughput"}
    rt=None
    try:
        import cupy as cp
        tr=load_trace();tokens=tr["tokens"]
        rt,g,keep=make_synth(CTX,SynthesisConfig(),diagnostic=True)
        prefill_to(rt,tokens,CTX)

        blocks=[];censuses=[]
        for bi in range(2):
            pos=int(rt.pos);draft,expected=expected_for_block(tokens,pos)
            t0=time.perf_counter_ns()
            got,census=g.v.block(draft.tolist(),True)
            cp.cuda.get_current_stream().synchronize()
            wall=(time.perf_counter_ns()-t0)/1e6
            if not np.array_equal(got,expected):
                raise RuntimeError(f"profile block {bi} diverged")
            blocks.append({"block":bi,"pos":pos,"wall_ms":wall})
            censuses.append(census)

        # profile_rows contains the same two blocks x 23 layers.
        by_layer=defaultdict(list)
        for row in g.gmoe.profile_rows:
            by_layer[int(row["layer"])].append(row)

        layer_rows=[]
        stage_totals=defaultdict(float)
        for layer in sorted(by_layer):
            rows=by_layer[layer]
            avg={}
            for key in (
              "group_scale_bytes","group_code_bytes","group_total_down_bytes",
              "resident_plane_bytes","cache_miss_routes",
            ):
                avg[key]=float(np.mean([r[key] for r in rows]))
            stage={}
            for r in rows:
                for k,v in (r.get("stage_ms") or {}).items():
                    stage[k]=stage.get(k,0.0)+float(v)/len(rows)
            for k,v in stage.items():stage_totals[k]+=v
            plane=max(avg["resident_plane_bytes"],1.0)
            layer_rows.append({
              "layer":layer,
              "avg":avg,
              "avg_stage_ms":stage,
              "scale_bytes_per_plane_vram_byte":avg["group_scale_bytes"]/plane,
              "scale_bytes_per_block":avg["group_scale_bytes"],
              "plane_vram_bytes":avg["resident_plane_bytes"],
            })

        ranking=sorted(
          layer_rows,
          key=lambda r:(
            -r["scale_bytes_per_plane_vram_byte"],
            -r["scale_bytes_per_block"],
          )
        )
        h8=combine_h8(censuses[0],censuses[1])
        ratios=[
          r["h8_unique_experts"]/float(r["two_h4_streams"]) for r in h8
        ]

        payload.update({
          "status":"measured",
          "blocks":blocks,
          "stage_totals_ms_per_h4":dict(stage_totals),
          "layers":layer_rows,
          "sres_layer_ranking":[
            {"rank":i+1,**r} for i,r in enumerate(ranking)
          ],
          "h8_route_census":{
            "layers":h8,
            "median_h8_over_two_h4_streams":float(np.median(ratios)),
            "median_additional_reduction_fraction":float(
              np.median([1.0-x for x in ratios])
            ),
            "total_two_h4_streams":int(sum(r["two_h4_streams"] for r in h8)),
            "total_h8_unique_experts":int(sum(r["h8_unique_experts"] for r in h8)),
            "total_additional_reduction_fraction":(
              1.0-sum(r["h8_unique_experts"] for r in h8)/
                  float(sum(r["two_h4_streams"] for r in h8))
            ),
          },
          "all_token_exact":True,
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),
      "blocks":payload.get("blocks"),
      "stage_totals_ms_per_h4":payload.get("stage_totals_ms_per_h4"),
      "top_sres_layers":[
        (r["layer"],r["scale_bytes_per_plane_vram_byte"])
        for r in payload.get("sres_layer_ranking",[])[:8]
      ],
      "h8_route_census":payload.get("h8_route_census"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
