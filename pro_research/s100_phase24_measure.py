from __future__ import annotations

import argparse
import json
import traceback

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,release
from s100_phase22_common import (
    make_v6,selected_head_mode,timed_graph_blocks,
)
from s100_phase23_common import GraphH4VerifierGrouped
from s100_phase24_common import (
    RESULTS,config_for_k,make_synth,timed_synth_blocks,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=("baseline","synth"),required=True)
    ap.add_argument("--k",type=int,default=0)
    ap.add_argument("--context",type=int,default=1024)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=8)
    ap.add_argument("--warmup",type=int,default=4)
    args=ap.parse_args()

    label=("baseline" if args.arm=="baseline" else f"synth_k{args.k}")
    out=RESULTS/f"S100_PHASE24_{args.tag.upper()}_{label.upper()}_CTX{args.context}.json"
    payload={"kind":"s100_phase24_measure","status":"started",
      "arm":args.arm,"k":args.k,"label":label,"context":args.context,
      "tag":args.tag,"blocks":args.blocks,"warmup":args.warmup,
      "started_utc":utc_now(),
      "claim_boundary":"fresh-process best-of-all target-only H4 timing"}
    rt=None
    try:
        tr=load_trace();tokens=tr["tokens"]
        need=args.context+4*(args.blocks+args.warmup)+1
        if need>len(tokens):
            raise RuntimeError(f"trace too short: need={need}, have={len(tokens)}")

        if args.arm=="baseline":
            rt,keep=make_v6(args.context)
            g=GraphH4VerifierGrouped(rt,selected_head_mode())
            cap=g.setup_graph()
            records,summary=timed_graph_blocks(
              rt,g,tokens,args.context,args.blocks,args.warmup
            )
            config={
              "attention_m4":False,"router_m4":False,
              "shared_m4":False,"sres_layers":[],
            }
            plane_bytes=0
        else:
            config_obj=config_for_k(args.k)
            rt,g,keep=make_synth(args.context,config_obj)
            cap=g.setup_graph()
            records,summary=timed_synth_blocks(
              rt,g,tokens,args.context,args.blocks,args.warmup
            )
            config=config_obj.as_dict()
            plane_bytes=int(g.gmoe.actual_plane_bytes)

        payload.update({
          "status":"measured","config":config,
          "actual_plane_bytes":plane_bytes,
          "capture_info":cap,"records":records,"summary":summary,
          "correctness_green":bool(summary.get("all_token_exact")),
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        # A true allocation failure is feasibility evidence only.
        msg=str(exc).lower()
        oom=(
          "out of memory" in msg or "cuda_error_out_of_memory" in msg
          or type(exc).__name__.lower() in ("outofmemoryerror","memoryerror")
        )
        payload.update({
          "status":"infeasible_vram" if oom else "technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now(),
        })
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    out.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"label":label,"context":args.context,
      "config":payload.get("config"),
      "actual_plane_mib":(
        payload.get("actual_plane_bytes",0)/2**20
        if payload.get("actual_plane_bytes") is not None else None
      ),
      "summary":payload.get("summary"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out)},indent=2))
    return 0 if payload.get("status") in ("measured","infeasible_vram") else 2

if __name__=="__main__":
    raise SystemExit(main())
