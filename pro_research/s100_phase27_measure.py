from __future__ import annotations

import argparse
import json
import subprocess
import traceback

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,release
from s100_phase24_common import make_synth,timed_synth_blocks
from s100_phase27_common import (
    RESULTS,Variant,phase27_gate,make_candidate,timed_candidate,
)


def telemetry():
    fields=(
      "timestamp","temperature.gpu","pstate","clocks.sm","clocks.mem",
      "power.draw","utilization.gpu","memory.used",
    )
    try:
        raw=subprocess.check_output(
          ["nvidia-smi",f"--query-gpu={','.join(fields)}",
           "--format=csv,noheader,nounits"],
          text=True,stderr=subprocess.STDOUT,timeout=10,
        ).strip().splitlines()[0]
        vals=[x.strip() for x in raw.split(",")]
        d=dict(zip(fields,vals))
        for k in fields:
            if k in ("timestamp","pstate"):continue
            try:d[k]=float(d[k])
            except Exception:pass
        return d
    except Exception as exc:
        return {"error":f"{type(exc).__name__}: {exc}"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=("parent","candidate"),required=True)
    ap.add_argument("--gather-y",type=int,choices=(4,8,16,32),default=32)
    ap.add_argument("--batches",type=int,choices=(1,2,3,4),default=1)
    ap.add_argument("--shared-overlap",action="store_true")
    ap.add_argument("--context",type=int,default=1024)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=8)
    ap.add_argument("--warmup",type=int,default=4)
    args=ap.parse_args()

    out=RESULTS/f"S100_PHASE27_{args.tag.upper()}.json"
    variant=Variant(args.gather_y,args.batches,args.shared_overlap)
    payload={
      "kind":"s100_phase27_measure","status":"started",
      "arm":args.arm,"variant":variant.as_dict() if args.arm=="candidate" else None,
      "context":args.context,"tag":args.tag,
      "blocks":args.blocks,"warmup":args.warmup,
      "started_utc":utc_now(),
      "claim_boundary":"fresh-process exact target-only H4 timing",
    }
    rt=None
    try:
        cfg,_,_=phase27_gate()
        pf=json.loads(
          (RESULTS/"S100_PHASE27_PREFLIGHT.json").read_text(encoding="utf-8")
        )
        if not pf.get("PREFLIGHT_GREEN"):
            raise RuntimeError("Phase27 preflight not green")

        tr=load_trace();tokens=tr["tokens"]
        need=args.context+4*(args.blocks+args.warmup)+1
        if need>len(tokens):
            raise RuntimeError(
              f"canonical trace too short need={need} have={len(tokens)}"
            )

        if args.arm=="parent":
            rt,g,keep=make_synth(args.context,cfg)
        else:
            rt,g,keep=make_candidate(args.context,variant)

        cap=g.setup_graph()
        tel0=telemetry()
        if args.arm=="parent":
            records,summary=timed_synth_blocks(
              rt,g,tokens,args.context,args.blocks,args.warmup
            )
        else:
            records,summary=timed_candidate(
              rt,g,tokens,args.context,args.blocks,args.warmup
            )
        tel1=telemetry()

        payload.update({
          "status":"measured","capture_info":cap,
          "records":records,"summary":summary,
          "correctness_green":bool(summary.get("all_token_exact")),
          "ms_per_useful_token":float(summary["median_ms"]/4.0),
          "target_only_tok_s":float(4000.0/summary["median_ms"]),
          "telemetry":{"after_setup":tel0,"after_measure":tel1},
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        msg=str(exc).lower()
        oom=(
          "out of memory" in msg or "cuda_error_out_of_memory" in msg
          or type(exc).__name__.lower() in ("outofmemoryerror","memoryerror")
        )
        payload.update({
          "status":"infeasible_vram" if oom else "technical_failure",
          "error":{
            "type":type(exc).__name__,
            "message":str(exc),
            "traceback":traceback.format_exc(),
          },
          "completed_utc":utc_now(),
        })
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    out.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),
      "arm":args.arm,
      "variant":payload.get("variant"),
      "context":args.context,
      "tag":args.tag,
      "summary":payload.get("summary"),
      "ms_per_useful_token":payload.get("ms_per_useful_token"),
      "target_only_tok_s":payload.get("target_only_tok_s"),
      "telemetry":payload.get("telemetry"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out),
    },indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
