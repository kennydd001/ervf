from __future__ import annotations

import argparse
import json
import subprocess
import traceback

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace, release
from s100_phase24_common import selected_config, make_synth, timed_synth_blocks
from s100_phase25_common import timed_parent_h8_windows, timed_h8_blocks
from s100_phase26_common import (
    RESULTS, phase26_gate, make_h4_overlap, make_h8_overlap,
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
    ap.add_argument("--arm",choices=("parent","overlap"),required=True)
    ap.add_argument("--horizon",type=int,choices=(4,8),required=True)
    ap.add_argument("--context",type=int,default=1024)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=12)
    ap.add_argument("--warmup",type=int,default=8)
    args=ap.parse_args()

    out=RESULTS/f"S100_PHASE26_{args.tag.upper()}.json"
    payload={
      "kind":"s100_phase26_measure","status":"started",
      "arm":args.arm,"horizon":args.horizon,
      "context":args.context,"tag":args.tag,
      "blocks":args.blocks,"warmup":args.warmup,
      "started_utc":utc_now(),
      "claim_boundary":"exact target-only shared/routed overlap timing",
    }
    rt=None
    try:
        cfg,p24,p25=phase26_gate()
        st=json.loads(
          (RESULTS/"S100_PHASE26_STATE_CHECK.json").read_text(encoding="utf-8")
        )
        if args.arm=="overlap":
            key=("H4_OVERLAP_STATE_GREEN" if args.horizon==4
                 else "H8_OVERLAP_STATE_GREEN")
            if not st.get(key):
                raise RuntimeError(f"{key} is not green")

        tr=load_trace();tokens=tr["tokens"]
        need=args.context+args.horizon*(args.blocks+args.warmup)+1
        if need>len(tokens):
            raise RuntimeError(f"canonical trace too short: need={need}, have={len(tokens)}")

        if args.horizon==4:
            if args.arm=="parent":
                rt,g,keep=make_synth(args.context,cfg)
            else:
                rt,g,keep=make_h4_overlap(args.context)
            cap=g.setup_graph()
            payload["telemetry"]={"after_setup":telemetry()}
            records,summary=timed_synth_blocks(
                rt,g,tokens,args.context,args.blocks,args.warmup
            )
        else:
            if args.arm=="parent":
                rt,g,keep=make_synth(args.context,cfg)
                cap=g.setup_graph()
                payload["telemetry"]={"after_setup":telemetry()}
                records,summary=timed_parent_h8_windows(
                    rt,g,tokens,args.context,args.blocks,args.warmup
                )
            else:
                rt,g,keep=make_h8_overlap(args.context)
                cap=g.setup_graph()
                payload["telemetry"]={"after_setup":telemetry()}
                records,summary=timed_h8_blocks(
                    rt,g,tokens,args.context,args.blocks,args.warmup
                )

        payload["telemetry"]["after_measure"]=telemetry()
        payload.update({
          "status":"measured","capture_info":cap,
          "records":records,"summary":summary,
          "correctness_green":bool(summary.get("all_token_exact")),
          "ms_per_useful_token":float(summary["median_ms"]/args.horizon),
          "target_only_tok_s":float(1000.0/(summary["median_ms"]/args.horizon)),
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
            "type":type(exc).__name__,"message":str(exc),
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
      "status":payload.get("status"),"arm":args.arm,
      "horizon":args.horizon,"context":args.context,"tag":args.tag,
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
