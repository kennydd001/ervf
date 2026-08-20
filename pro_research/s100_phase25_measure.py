from __future__ import annotations

import argparse
import json
import traceback

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,release
from s100_phase24_common import selected_config,make_synth
from s100_phase25_common import (
    RESULTS,VARIANTS,make_h8,timed_h8_blocks,timed_parent_h8_windows,phase24_gate,
    OFFICIAL_PARENT_H8_MS,ADOPTION_ABS_MS,
)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("parent","candidate"),required=True)
    ap.add_argument("--variant",choices=tuple(VARIANTS),default="split4_route")
    ap.add_argument("--context",type=int,default=1024);ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=8);ap.add_argument("--warmup",type=int,default=4)
    args=ap.parse_args();label="parent" if args.arm=="parent" else args.variant
    out=RESULTS/f"S100_PHASE25_{args.tag.upper()}_{label.upper()}_CTX{args.context}.json"
    payload={"kind":"s100_phase25_measure","status":"started","arm":args.arm,"variant":args.variant,
      "context":args.context,"tag":args.tag,"blocks":args.blocks,"warmup":args.warmup,
      "official_parent_h8_ms":OFFICIAL_PARENT_H8_MS,"adoption_abs_ms":ADOPTION_ABS_MS,
      "started_utc":utc_now(),"claim_boundary":"fresh-process exact target-only H8 window timing"}
    rt=None
    try:
        tr=load_trace();tokens=tr["tokens"];need=args.context+8*(args.blocks+args.warmup)+1
        if need>len(tokens):raise RuntimeError(f"trace too short need={need} have={len(tokens)}")
        if args.arm=="parent":
            cfg,_,_,_=phase24_gate();rt,g,keep=make_synth(args.context,cfg);cap=g.setup_graph()
            records,summary=timed_parent_h8_windows(rt,g,tokens,args.context,args.blocks,args.warmup)
            config={"phase24_parent":True,"phase24_config":cfg.as_dict()};plane_bytes=int(g.gmoe.actual_plane_bytes)
        else:
            rt,g,keep=make_h8(args.context,args.variant);cap=g.setup_graph()
            records,summary=timed_h8_blocks(rt,g,tokens,args.context,args.blocks,args.warmup)
            config={"variant":args.variant,"up_mode":g.up_mode,"down_mode":g.down_mode,
                    "phase24_config":g.config.as_dict()};plane_bytes=int(g.gmoe.actual_plane_bytes)
        payload.update({"status":"measured","config":config,"actual_plane_bytes":plane_bytes,
          "capture_info":cap,"records":records,"summary":summary,"correctness_green":True,
          "beats_official_parent":bool(summary["median_ms"]<OFFICIAL_PARENT_H8_MS) if args.arm=="candidate" else None,
          "meets_adoption_abs":bool(summary["median_ms"]<=ADOPTION_ABS_MS) if args.arm=="candidate" else None,
          "completed_utc":utc_now()})
    except Exception as exc:
        msg=str(exc).lower();oom=("out of memory" in msg or "cuda_error_out_of_memory" in msg or
          type(exc).__name__.lower() in ("outofmemoryerror","memoryerror"))
        payload.update({"status":"infeasible_vram" if oom else "technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass
    out.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(out,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"label":label,"context":args.context,
      "summary":payload.get("summary"),"actual_plane_mib":payload.get("actual_plane_bytes",0)/2**20,
      "error":(payload.get("error") or {}).get("message"),"output":str(out)},indent=2))
    return 0 if payload.get("status") in ("measured","infeasible_vram") else 2
if __name__=="__main__":raise SystemExit(main())
