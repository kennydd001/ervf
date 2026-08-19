from __future__ import annotations

import argparse
import json
import traceback

from common import write_json_atomic,utc_now
from s100_phase21_common import (
    RESULTS,ARMS,identity_gate,load_trace,make_rt,verifier_for,
    measure_blocks,release,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=ARMS,required=True)
    ap.add_argument("--context",type=int,required=True)
    ap.add_argument("--blocks",type=int,default=6)
    ap.add_argument("--warmup",type=int,default=1)
    ap.add_argument("--tag",default="screen")
    args=ap.parse_args()

    safe=f"{args.tag}_{args.arm}_ctx{args.context}"
    out=RESULTS/f"S100_PHASE21_{safe.upper()}.json"
    payload={
      "kind":"s100_phase21_hybrid",
      "status":"started",
      "arm":args.arm,
      "context":args.context,
      "blocks":args.blocks,
      "warmup":args.warmup,
      "started_utc":utc_now(),
      "claim_boundary":"fresh-process H4 target-only hybrid timing; no drafter",
    }
    rt=None
    try:
        identity_gate();tr=load_trace()
        if args.context+4*(args.blocks+args.warmup)+1>len(tr["tokens"]):
            raise RuntimeError("canonical trace too short")
        rt,keep=make_rt(args.context,args.arm)
        v=verifier_for(rt,args.arm)
        records,summary=measure_blocks(
            rt,v,tr["tokens"],args.context,args.blocks,args.warmup
        )
        payload.update({
          "status":"measured",
          "summary":summary,
          "records":records,
          "correctness_green":bool(summary["all_token_exact"]),
          "cache_stats":dict(getattr(rt,"cache_stats",{})),
          "device_cache":bool(getattr(rt,"device_cache",False)),
          "cache_capacity_by_layer":{
             str(i):int(c["cap"]) for i,c in getattr(rt,"cache",{}).items()
          },
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try: release(rt)
            except Exception: pass

    out.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"arm":args.arm,"context":args.context,
      "summary":payload.get("summary"),"cache_stats":payload.get("cache_stats"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
