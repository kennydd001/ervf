from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
import numpy as np

from common import utc_now,write_json_atomic
from s100_phase21_common import load_trace,prefill_to,release
from s100_phase24_common import selected_config,make_synth
from s100_phase25_common import RESULTS,make_h8,expected_for_h8,summarize_h8,phase24_gate

def telemetry():
    fields=("timestamp","temperature.gpu","pstate","clocks.sm","clocks.mem","power.draw","utilization.gpu","memory.used")
    try:
        raw=subprocess.check_output(["nvidia-smi",f"--query-gpu={','.join(fields)}","--format=csv,noheader,nounits"],
            text=True,stderr=subprocess.STDOUT,timeout=10).strip().splitlines()[0]
        vals=[x.strip() for x in raw.split(",")];d=dict(zip(fields,vals))
        for k in fields:
            if k in ("timestamp","pstate"):continue
            try:d[k]=float(d[k])
            except Exception:pass
        return d
    except Exception as exc:return {"error":f"{type(exc).__name__}: {exc}"}

def selected_variant():
    d=json.loads((RESULTS/"S100_PHASE25_SELECTION.json").read_text(encoding="utf-8"));s=d.get("selected")
    if not s or not d.get("THERMAL_ADOPTION_OPEN"):raise RuntimeError("Phase25 selection has not opened thermal adoption")
    v=str(s["variant"]);st=json.loads((RESULTS/f"S100_PHASE25_STATE_CHECK_{v.upper()}.json").read_text(encoding="utf-8"))
    if not st.get("H8_STATE_GREEN"):raise RuntimeError("selected H8 state gate not green")
    return v

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("parent","selected"),required=True)
    ap.add_argument("--context",type=int,default=1024);ap.add_argument("--tag",required=True)
    ap.add_argument("--blocks",type=int,default=16);ap.add_argument("--warmup",type=int,default=8);args=ap.parse_args()
    out=RESULTS/f"S100_PHASE25_{args.tag.upper()}.json";payload={"kind":"s100_phase25_thermal_measure","status":"started",
      "mode":args.mode,"context":args.context,"tag":args.tag,"blocks":args.blocks,"warmup":args.warmup,
      "started_utc":utc_now(),"claim_boundary":"balanced thermal exact H8-window timing"};rt=None
    try:
        tr=load_trace();tokens=tr["tokens"];need=args.context+8*(args.blocks+args.warmup)+1
        if need>len(tokens):raise RuntimeError(f"trace too short need={need}")
        if args.mode=="parent":
            cfg,_,_,_=phase24_gate();rt,g,keep=make_synth(args.context,cfg);cap=g.setup_graph();prefill_to(rt,tokens,args.context);g.prepare_after_prefill()
            variant=None;plane=int(g.gmoe.actual_plane_bytes)
            def one(draft):
                a=g.launch(draft[:4].tolist());b=g.launch(draft[4:].tolist());return np.concatenate([a,b])
        else:
            variant=selected_variant();rt,g,keep=make_h8(args.context,variant);cap=g.setup_graph();prefill_to(rt,tokens,args.context);g.prepare_after_prefill()
            plane=int(g.gmoe.actual_plane_bytes)
            def one(draft):return g.launch(draft.tolist())
        payload.update({"capture_info":cap,"variant":variant,"actual_plane_bytes":plane,
                        "telemetry":{"after_setup_prefill":telemetry()}})
        for _ in range(args.warmup):
            pos=int(rt.pos);draft,expected=expected_for_h8(tokens,pos);got=one(draft)
            if not np.array_equal(got,expected):raise RuntimeError(f"warmup mismatch pos={pos}")
        payload["telemetry"]["after_warmup"]=telemetry();rows=[]
        for bi in range(args.blocks):
            pos=int(rt.pos);draft,expected=expected_for_h8(tokens,pos);t0=time.perf_counter_ns();got=one(draft);ms=(time.perf_counter_ns()-t0)/1e6
            if not np.array_equal(got,expected):raise RuntimeError(f"measure mismatch pos={pos}")
            rows.append({"block":bi,"pos":pos,"ms":ms,"got":got.tolist(),"expected":expected.tolist()})
        payload["telemetry"]["after_measure"]=telemetry();payload.update({"status":"measured","records":rows,
          "summary":summarize_h8(rows),"correctness_green":True,"completed_utc":utc_now()})
    except Exception as exc:
        msg=str(exc).lower();oom=("out of memory" in msg or "cuda_error_out_of_memory" in msg or type(exc).__name__.lower() in ("outofmemoryerror","memoryerror"))
        payload.update({"status":"infeasible_vram" if oom else "technical_failure","error":{"type":type(exc).__name__,
          "message":str(exc),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass
    out.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(out,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"mode":args.mode,"tag":args.tag,"variant":payload.get("variant"),
      "summary":payload.get("summary"),"telemetry":payload.get("telemetry"),"error":(payload.get("error") or {}).get("message"),
      "output":str(out)},indent=2));return 0 if payload.get("status") in ("measured","infeasible_vram") else 2
if __name__=="__main__":raise SystemExit(main())
