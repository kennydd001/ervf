from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback

import numpy as np

from common import REPO,utc_now,write_json_atomic
from s100_phase21_common import (
    identity_gate,load_trace,prefill_to,expected_for_block,release,
)
from s100_phase22_common import make_v6,GraphH4Verifier,selected_head_mode
from s100_phase23_common import GraphH4VerifierGrouped

RESULTS=REPO/"pro_research"/"results"/"s100_phase23r"

def gpu_telemetry():
    fields=[
      "timestamp","temperature.gpu","pstate","clocks.sm","clocks.mem",
      "power.draw","utilization.gpu","memory.used",
    ]
    try:
        out=subprocess.check_output(
          ["nvidia-smi",f"--query-gpu={','.join(fields)}",
           "--format=csv,noheader,nounits"],
          text=True,stderr=subprocess.STDOUT,timeout=10,
        ).strip().splitlines()[0]
        vals=[x.strip() for x in out.split(",")]
        row=dict(zip(fields,vals))
        for k in ("temperature.gpu","clocks.sm","clocks.mem","power.draw",
                  "utilization.gpu","memory.used"):
            try:row[k]=float(row[k])
            except Exception:pass
        return row
    except Exception as exc:
        return {"error":f"{type(exc).__name__}: {exc}"}

def summarize(rows):
    vals=np.asarray([x["ms"] for x in rows],np.float64)
    return {
      "count":len(rows),
      "median_ms":float(np.median(vals)),
      "p10_ms":float(np.percentile(vals,10)),
      "p90_ms":float(np.percentile(vals,90)),
      "mean_ms":float(vals.mean()),
      "mad_ms":float(np.median(np.abs(vals-np.median(vals)))),
      "ms_per_useful_token":float(np.median(vals)/4.0),
      "target_only_tok_s":float(4000.0/np.median(vals)),
      "all_token_exact":True,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("parent","grouped"),required=True)
    ap.add_argument("--context",type=int,default=1024)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=16)
    ap.add_argument("--warmup",type=int,default=8)
    args=ap.parse_args()

    out=RESULTS/f"S100_PHASE23R_{args.tag.upper()}.json"
    payload={
      "kind":"s100_phase23r_measure","status":"started",
      "mode":args.mode,"context":args.context,"tag":args.tag,
      "blocks":args.blocks,"warmup":args.warmup,
      "started_utc":utc_now(),
      "claim_boundary":"thermal-order-balanced perfect-draft H4 graph timing",
    }
    rt=None
    try:
        identity_gate()
        p23=json.loads(
          (REPO/"pro_research"/"results"/"s100_phase23"/"S100_PHASE23_SUMMARY.json")
          .read_text(encoding="utf-8")
        )
        if not p23.get("GPU_GROUPED_CORRECTNESS_GREEN"):
            raise RuntimeError("Phase23 correctness gate not green")

        tr=load_trace();tokens=tr["tokens"]
        need=args.context+4*(args.blocks+args.warmup)+1
        if need>len(tokens):
            raise RuntimeError(f"canonical trace too short: need={need}, have={len(tokens)}")

        rt,keep=make_v6(args.context)
        head=selected_head_mode()
        g=(GraphH4VerifierGrouped(rt,head)
           if args.mode=="grouped" else GraphH4Verifier(rt,head))
        capture=g.setup_graph()
        payload["capture_info"]=capture
        payload["telemetry"]={"after_graph_setup":gpu_telemetry()}

        prefill_to(rt,tokens,args.context)
        g.set_pos_from_host()
        payload["telemetry"]["after_prefill"]=gpu_telemetry()

        # Warmup advances the exact same canonical target chain. No measured
        # block is reused and P/G measured positions remain aligned.
        warm_ids=[]
        for wi in range(args.warmup):
            pos=int(rt.pos)
            drafts,expected=expected_for_block(tokens,pos)
            got=g.launch(drafts.tolist())
            if not np.array_equal(got,expected):
                raise RuntimeError(
                    f"warmup divergence wi={wi} pos={pos} "
                    f"got={got.tolist()} expected={expected.tolist()}"
                )
            warm_ids.append({"pos":pos,"ids":got.tolist()})
        payload["telemetry"]["after_warmup"]=gpu_telemetry()

        rows=[]
        for bi in range(args.blocks):
            pos=int(rt.pos)
            drafts,expected=expected_for_block(tokens,pos)
            t0=time.perf_counter_ns()
            got=g.launch(drafts.tolist())
            ms=(time.perf_counter_ns()-t0)/1e6
            if not np.array_equal(got,expected):
                raise RuntimeError(
                    f"measured divergence bi={bi} pos={pos} "
                    f"got={got.tolist()} expected={expected.tolist()}"
                )
            rows.append({
              "block":bi,"pos":pos,"ms":ms,
              "got":got.tolist(),"expected":expected.tolist(),
            })
        payload["telemetry"]["after_measure"]=gpu_telemetry()
        payload.update({
          "status":"measured","head_mode":head,
          "warmup_records":warm_ids,
          "records":rows,"summary":summarize(rows),
          "correctness_green":True,
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({
          "status":"technical_failure",
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
      "status":payload.get("status"),"mode":args.mode,"tag":args.tag,
      "summary":payload.get("summary"),"telemetry":payload.get("telemetry"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(out)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
