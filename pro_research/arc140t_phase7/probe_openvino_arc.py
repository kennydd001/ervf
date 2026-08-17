from __future__ import annotations
import json, statistics, time, traceback
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]/"results"/"arc140t_phase7"/"openvino_arc_probe.json"

SHAPES=[
 ("q_like_f16",1,2688,4096,"f16"),("q_like_M8_f16",8,2688,4096,"f16"),
 ("mamba_in_f16",1,2688,10304,"f16"),("mamba_in_M8_f16",8,2688,10304,"f16"),
 ("expert_down_f16",1,3072,2688,"f16"),("expert_down_M6_f16",6,3072,2688,"f16"),
 ("q_like_i8",1,2688,4096,"i8"),("q_like_M8_i8",8,2688,4096,"i8"),
 ("expert_down_i8",1,3072,2688,"i8"),("expert_down_M6_i8",6,3072,2688,"i8"),
]

def main():
    payload={"kind":"arc140t_phase7_openvino_probe","status":"started","records":[]}
    try:
        import openvino as ov
        from openvino import opset13 as ops
        core=ov.Core()
        payload["openvino_version"]=getattr(ov,"__version__","unknown")
        payload["available_devices"]=list(core.available_devices)
        gpu=next((d for d in core.available_devices if d.upper().startswith("GPU")),None)
        if not gpu:
            payload["status"]="no_gpu_device"
        else:
            payload["device"]=gpu
            props={}
            for key in ("FULL_DEVICE_NAME","DEVICE_TYPE","OPTIMIZATION_CAPABILITIES","GPU_DEVICE_TOTAL_MEM_SIZE","GPU_UARCH_VERSION"):
                try: props[key]=str(core.get_property(gpu,key))
                except Exception: pass
            payload["properties"]=props
            rng=np.random.default_rng(260817)
            for label,m,k,n,dtype in SHAPES:
                rec={"label":label,"m":m,"k":k,"n":n,"dtype":dtype}
                try:
                    # Geometry only. Raw i8 tests whether an integer matrix path exists;
                    # it is not the current checkpoint quantization contract.
                    if dtype=="f16":
                        W=(rng.standard_normal((k,n),dtype=np.float32)*0.01).astype(np.float16)
                        x=(rng.standard_normal((m,k),dtype=np.float32)*0.01).astype(np.float16)
                        A=ops.parameter([m,k],np.float16,name="A")
                    else:
                        W=rng.integers(-7,8,size=(k,n),dtype=np.int8)
                        x=rng.integers(-7,8,size=(m,k),dtype=np.int8)
                        A=ops.parameter([m,k],np.int8,name="A")
                    C=ops.constant(W)
                    Y=ops.matmul(A,C,False,False)
                    model=ov.Model([Y],[A],label)
                    compiled=core.compile_model(model,gpu,{"PERFORMANCE_HINT":"LATENCY"})
                    req=compiled.create_infer_request()
                    for _ in range(5): req.infer({0:x})
                    vals=[]
                    reps=20 if n<6000 else 12
                    for _ in range(reps):
                        t=time.perf_counter(); req.infer({0:x}); vals.append((time.perf_counter()-t)*1e3)
                    rec.update({"status":"measured","median_ms":statistics.median(vals),"min_ms":min(vals),"max_ms":max(vals),"samples_ms":vals})
                    del W,model,compiled,req
                except Exception as e:
                    rec.update({"status":"technical_failure","error":f"{type(e).__name__}: {e}"})
                payload["records"].append(rec)
            payload["status"]="measured"
    except Exception as e:
        payload.update({"status":"technical_failure","error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}})
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(json.dumps(payload,indent=2,allow_nan=False))
    return 0 if payload["status"] in {"measured","no_gpu_device"} else 2
if __name__=="__main__": raise SystemExit(main())
