from __future__ import annotations
import json,statistics,traceback
import numpy as np
from common import write_json_atomic,utc_now
from s100_phase22_common import (
    RESULTS,identity_gate,load_trace,make_v6,eager_verifier,prefill_to,
    expected_for_block,release,nrmse,
)

OUT=RESULTS/"S100_PHASE22_HEAD_SELECTION.json"

def event_times(cp,fn,reps=12):
    vals=[]
    for _ in range(3):fn()
    cp.cuda.get_current_stream().synchronize()
    for _ in range(reps):
        a=cp.cuda.Event();b=cp.cuda.Event()
        a.record();fn();b.record();b.synchronize()
        vals.append(float(cp.cuda.get_elapsed_time(a,b)))
    return {
      "median_ms":float(statistics.median(vals)),
      "p10_ms":float(np.percentile(vals,10)),
      "p90_ms":float(np.percentile(vals,90)),
      "raw_ms":vals,
    }

def main():
    payload={"kind":"s100_phase22_head_selection","status":"started",
             "context":1024,"started_utc":utc_now(),
             "claim_boundary":"lm_head component selection only"}
    rt=None
    try:
        import cupy as cp
        identity_gate();tr=load_trace();tokens=tr["tokens"]
        rt,keep=make_v6(1024)
        v=eager_verifier(rt,"generic_m4")
        prefill_to(rt,tokens,1024)
        draft,expected=expected_for_block(tokens,rt.pos)
        got,_=v.block(draft.tolist(),False)
        cp.cuda.get_current_stream().synchronize()
        if not np.array_equal(got,expected):
            raise RuntimeError("real H4 capture block diverged")

        x=v.final_normed.copy()
        generic=cp.empty_like(v.logits)
        prod=cp.empty_like(v.logits)

        def f_generic():
            v.bk.nvfp4(
              rt.lm_head_codes,rt.lm_head_scales,rt.fused.e2m1,rt.fused.e4m3,
              x,generic,rt.lm_head_g,rt.vocab,rt.hidden,4,False
            )
        def f_prod():
            for t in range(4):
                rt.fused.gemv_into(
                  prod[t],rt.lm_head_codes,rt.lm_head_scales,x[t],
                  rt.lm_head_g,rt.vocab,rt.hidden
                )

        f_generic();f_prod();cp.cuda.get_current_stream().synchronize()
        gn=cp.asnumpy(generic);pn=cp.asnumpy(prod)
        err=nrmse(pn,gn)
        arg=float(np.mean(
          np.argmax(gn,axis=1)==np.argmax(pn,axis=1)
        ))
        finite=bool(np.isfinite(gn).all() and np.isfinite(pn).all())

        ga=event_times(cp,f_generic)
        pt=event_times(cp,f_prod)
        gb=event_times(cp,f_generic)
        gmid=(ga["median_ms"]+gb["median_ms"])/2.0
        prod_green=bool(finite and err<=5e-4 and arg==1.0)

        candidates=[
          {"mode":"generic_m4","green":True,"median_ms":gmid,
           "a":ga,"b":gb,"nrmse_vs_generic":0.0,"argmax_agreement":1.0},
          {"mode":"production_x4","green":prod_green,"median_ms":pt["median_ms"],
           "timing":pt,"nrmse_vs_generic":err,"argmax_agreement":arg},
        ]
        green=[x for x in candidates if x["green"]]
        best=min(x["median_ms"] for x in green)
        tied=[x for x in green if x["median_ms"]<=best*1.01]
        selected=(
          next((x for x in tied if x["mode"]=="production_x4"),tied[0])
        )
        payload.update({
          "status":"measured","correctness":{
            "production_vs_generic_nrmse":err,
            "row_argmax_agreement":arg,"finite":finite,
            "production_green":prod_green,
          },
          "candidates":candidates,
          "selected_mode":selected["mode"],
          "selected_median_ms":selected["median_ms"],
          "generic_midpoint_ms":gmid,
          "selected_gain_fraction_vs_generic":(
             (gmid-selected["median_ms"])/gmid
          ),
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
      "correctness":payload.get("correctness"),
      "selected_mode":payload.get("selected_mode"),
      "generic_midpoint_ms":payload.get("generic_midpoint_ms"),
      "selected_median_ms":payload.get("selected_median_ms"),
      "gain":payload.get("selected_gain_fraction_vs_generic"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
