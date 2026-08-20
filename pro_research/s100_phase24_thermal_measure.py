from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback

import numpy as np

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,expected_for_block,release
from s100_phase22_common import make_v6,selected_head_mode
from s100_phase23_common import GraphH4VerifierGrouped
from s100_phase24_common import RESULTS,selected_config,make_synth

def telemetry():
    fields=(
      "timestamp","temperature.gpu","pstate","clocks.sm","clocks.mem",
      "power.draw","utilization.gpu","memory.used"
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

def summarize(rows):
    v=np.asarray([x["ms"] for x in rows],np.float64)
    return {
      "count":len(rows),"median_ms":float(np.median(v)),
      "p10_ms":float(np.percentile(v,10)),
      "p90_ms":float(np.percentile(v,90)),
      "mean_ms":float(v.mean()),
      "mad_ms":float(np.median(np.abs(v-np.median(v)))),
      "ms_per_useful_token":float(np.median(v)/4.0),
      "target_only_tok_s":float(4000.0/np.median(v)),
      "all_token_exact":True,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("baseline","selected"),required=True)
    ap.add_argument("--context",type=int,required=True)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=16)
    ap.add_argument("--warmup",type=int,default=8)
    args=ap.parse_args()

    out=RESULTS/f"S100_PHASE24_{args.tag.upper()}.json"
    payload={"kind":"s100_phase24_thermal_measure","status":"started",
      "mode":args.mode,"context":args.context,"tag":args.tag,
      "blocks":args.blocks,"warmup":args.warmup,
      "started_utc":utc_now(),
      "claim_boundary":"balanced thermal best-of-all H4 timing"}
    rt=None
    try:
        tr=load_trace();tokens=tr["tokens"]
        need=args.context+4*(args.blocks+args.warmup)+1
        if need>len(tokens):
            raise RuntimeError(f"trace too short need={need}")

        if args.mode=="baseline":
            rt,keep=make_v6(args.context)
            g=GraphH4VerifierGrouped(rt,selected_head_mode())
            config={"baseline":True}
            cap=g.setup_graph()
            prefill_to(rt,tokens,args.context)
            g.set_pos_from_host()
        else:
            check=json.loads(
              (RESULTS/"S100_PHASE24_STATE_CHECK.json").read_text(encoding="utf-8")
            )
            if not check.get("BEST_OF_ALL_STATE_GREEN"):
                raise RuntimeError("selected state gate not green")
            cfg=selected_config()
            if cfg is None:
                raise RuntimeError("selected arm is baseline")
            rt,g,keep=make_synth(args.context,cfg)
            config=cfg.as_dict()
            cap=g.setup_graph()
            prefill_to(rt,tokens,args.context)
            g.prepare_after_prefill()

        payload["capture_info"]=cap
        payload["config"]=config
        payload["telemetry"]={"after_setup_prefill":telemetry()}

        for wi in range(args.warmup):
            pos=int(rt.pos);draft,expected=expected_for_block(tokens,pos)
            got=g.launch(draft.tolist())
            if not np.array_equal(got,expected):
                raise RuntimeError(f"warmup mismatch pos={pos}")
        payload["telemetry"]["after_warmup"]=telemetry()

        rows=[]
        for bi in range(args.blocks):
            pos=int(rt.pos);draft,expected=expected_for_block(tokens,pos)
            t0=time.perf_counter_ns()
            got=g.launch(draft.tolist())
            ms=(time.perf_counter_ns()-t0)/1e6
            if not np.array_equal(got,expected):
                raise RuntimeError(f"measure mismatch pos={pos}")
            rows.append({"block":bi,"pos":pos,"ms":ms,
                         "got":got.tolist(),"expected":expected.tolist()})
        payload["telemetry"]["after_measure"]=telemetry()
        payload.update({
          "status":"measured","records":rows,"summary":summarize(rows),
          "correctness_green":True,
          "actual_plane_bytes":(
            int(g.gmoe.actual_plane_bytes)
            if args.mode=="selected" else 0
          ),
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
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass

    out.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"mode":args.mode,
      "context":args.context,"tag":args.tag,
      "summary":payload.get("summary"),
      "actual_plane_mib":payload.get("actual_plane_bytes",0)/2**20,
      "telemetry":payload.get("telemetry"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out)},indent=2))
    return 0 if payload.get("status") in ("measured","infeasible_vram") else 2

if __name__=="__main__":
    raise SystemExit(main())
