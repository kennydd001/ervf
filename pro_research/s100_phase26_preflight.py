from __future__ import annotations

import json
import traceback
import numpy as np

from common import REPO, utc_now, write_json_atomic
from s100_phase26_common import RESULTS, phase26_gate

OUT=RESULTS/"S100_PHASE26_PREFLIGHT.json"

SRC=r"""
extern "C" __global__ void p26_a(float* a,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)a[i]=(float)i;
}
extern "C" __global__ void p26_b(float* b,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)b[i]=2.0f*(float)i;
}
extern "C" __global__ void p26_sum(const float* a,const float* b,float* c,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)c[i]=a[i]+b[i];
}
"""

def main():
    payload={
      "kind":"s100_phase26_preflight","status":"started",
      "started_utc":utc_now(),
      "claim_boundary":"synthetic CUDA cross-stream graph capture only",
    }
    try:
        import cupy as cp
        cfg,p24,p25=phase26_gate()

        mod=cp.RawModule(
            code=SRC,options=("-std=c++14",),
            name_expressions=("p26_a","p26_b","p26_sum"),
        )
        ka=mod.get_function("p26_a")
        kb=mod.get_function("p26_b")
        ks=mod.get_function("p26_sum")

        n=4096
        a=cp.empty(n,cp.float32)
        b=cp.empty(n,cp.float32)
        c=cp.empty(n,cp.float32)
        main_stream=cp.cuda.Stream(non_blocking=True)
        side_stream=cp.cuda.Stream(non_blocking=True)
        fork=cp.cuda.Event()
        done=cp.cuda.Event()
        grid=((n+255)//256,)
        block=(256,)

        # Compile/warm before capture.
        with main_stream:
            ka(grid,block,(a,np.int32(n)))
        with side_stream:
            kb(grid,block,(b,np.int32(n)))
        main_stream.synchronize();side_stream.synchronize()

        main_stream.begin_capture()
        with main_stream:
            ka(grid,block,(a,np.int32(n)))
            fork.record(main_stream)
            with side_stream:
                side_stream.wait_event(fork)
                kb(grid,block,(b,np.int32(n)))
                done.record(side_stream)
            main_stream.wait_event(done)
            ks(grid,block,(a,b,c,np.int32(n)))
        graph=main_stream.end_capture()
        graph.launch(main_stream)
        main_stream.synchronize()

        got=cp.asnumpy(c)
        ref=3.0*np.arange(n,dtype=np.float32)
        exact=bool(np.array_equal(got,ref))

        payload.update({
          "status":"measured",
          "cross_stream_capture_exact":exact,
          "phase24_sres_layers":len(cfg.sres_layers),
          "phase25_h8_adopted":bool(p25.get("H8_ADOPTED")),
          "PREFLIGHT_GREEN":exact,
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({
          "status":"technical_failure",
          "PREFLIGHT_GREEN":False,
          "error":{
            "type":type(exc).__name__,"message":str(exc),
            "traceback":traceback.format_exc(),
          },
          "completed_utc":utc_now(),
        })

    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2))
    return 0 if payload.get("PREFLIGHT_GREEN") else 2

if __name__=="__main__":
    raise SystemExit(main())
