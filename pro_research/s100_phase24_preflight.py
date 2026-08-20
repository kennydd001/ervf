from __future__ import annotations

import json
import traceback

import numpy as np

from common import REPO,utc_now,write_json_atomic
from s100_phase24_dense_kernels import DenseM4Kernels
from s100_phase24_sres_grouped import GroupedScaleResidentKernels

OUT=REPO/"pro_research"/"results"/"s100_phase24"/"S100_PHASE24_PREFLIGHT.json"

def bf16_bits(x):
    u=np.asarray(x,np.float32).view(np.uint32)
    u=u+np.uint32(0x7FFF)+((u>>np.uint32(16))&np.uint32(1))
    return (u>>np.uint32(16)).astype(np.uint16)

def decode_bf16(u):
    return (np.asarray(u,np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)

def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30))

def main():
    payload={"kind":"s100_phase24_preflight","status":"started",
             "started_utc":utc_now()}
    try:
        import cupy as cp
        dense=DenseM4Kernels()
        sres=GroupedScaleResidentKernels()  # forces NVRTC get_function

        rng=np.random.default_rng(2400)
        rows,cols=17,64
        x=rng.standard_normal((4,cols),dtype=np.float32)
        wf=rng.standard_normal((rows,cols),dtype=np.float32)
        wb=bf16_bits(wf)
        wbd=decode_bf16(wb)

        xg=cp.asarray(x)
        bg=cp.asarray(wb)
        fg=cp.asarray(wf)
        bo=cp.empty((4,rows),cp.float32)
        fo=cp.empty_like(bo)
        dense.bf16(bg,xg,bo,rows,cols)
        dense.f32(fg,xg,fo,rows,cols)
        cp.cuda.get_current_stream().synchronize()

        bref=(x@wbd.T).astype(np.float32)
        fref=(x@wf.T).astype(np.float32)
        be=nrmse(cp.asnumpy(bo),bref)
        fe=nrmse(cp.asnumpy(fo),fref)
        payload.update({
          "status":"measured",
          "device":str(cp.cuda.runtime.getDeviceProperties(0).get("name")),
          "bf16_synthetic_nrmse":be,
          "f32_synthetic_nrmse":fe,
          "dense_module_green":bool(be<=2e-6 and fe<=2e-6),
          "sres_module_compiled":True,
          "PREFLIGHT_GREEN":bool(be<=2e-6 and fe<=2e-6),
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})

    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2))
    return 0 if payload.get("PREFLIGHT_GREEN") else 2

if __name__=="__main__":
    raise SystemExit(main())
