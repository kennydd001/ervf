from __future__ import annotations

import json
import traceback
import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_phase17_kernels import Phase17Kernels

OUT = (
    REPO / "pro_research" / "results" / "s100_phase17"
    / "S100_PHASE17_PREFLIGHT.json"
)

def nrmse(a, b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30))

def main():
    payload={
        "kind":"s100_phase17_preflight",
        "status":"started",
        "started_utc":utc_now(),
    }
    try:
        # No torch import: Phase17 is CuPy/CUDA only.
        import cupy as cp

        kernels=Phase17Kernels()
        props=cp.cuda.runtime.getDeviceProperties(0)

        T=4;Hh=4;P=4;N=8;hpg=2;G=Hh//hpg
        rng=np.random.default_rng(17)
        state0=rng.standard_normal(Hh*P*N).astype(np.float32)
        x=rng.standard_normal((T,Hh,P)).astype(np.float32)
        dt=np.abs(rng.standard_normal((T,Hh)).astype(np.float32))*.1
        Alog=rng.standard_normal(Hh).astype(np.float32)
        B=rng.standard_normal((T,G,N)).astype(np.float32)

        decay=np.exp(-np.exp(Alog[None,:])*dt).astype(np.float32)
        dx=(dt[:,:,None]*x).astype(np.float32)
        ref=np.empty((T,Hh,P,N),np.float32)
        s=state0.reshape(Hh,P,N).copy()
        for t in range(T):
            for h in range(Hh):
                g=h//hpg
                s[h]=(decay[t,h]*s[h]+dx[t,h,:,None]*B[t,g,None,:]).astype(np.float32)
            ref[t]=s

        ds=cp.asarray(state0)
        dxg=cp.asarray(dx.reshape(T,Hh*P))
        Bg=cp.asarray(B)
        dg=cp.asarray(decay)
        prefix=cp.empty((T,Hh*P*N),cp.float32)
        serial=cp.empty_like(prefix)

        kernels.scan("prefix",T,ds,dxg,Bg,dg,prefix,Hh,P,N,hpg)
        kernels.scan("serial",T,ds,dxg,Bg,dg,serial,Hh,P,N,hpg)
        cp.cuda.get_current_stream().synchronize()

        pe=nrmse(cp.asnumpy(prefix).reshape(ref.shape),ref)
        se=nrmse(cp.asnumpy(serial).reshape(ref.shape),ref)
        payload.update({
            "status":"measured",
            "device":{
                "name":props.get("name",b"").decode(errors="replace")
                       if isinstance(props.get("name"),bytes)
                       else str(props.get("name")),
                "capability":list(cp.cuda.Device(0).compute_capability),
            },
            "prefix_synthetic_nrmse":pe,
            "serial_synthetic_nrmse":se,
            "PREFLIGHT_GREEN":bool(pe<=2e-6 and se<=2e-6),
            "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status":"technical_failure",
            "error":{"type":type(exc).__name__,"message":str(exc),
                     "traceback":traceback.format_exc()},
            "completed_utc":utc_now(),
        })
    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
