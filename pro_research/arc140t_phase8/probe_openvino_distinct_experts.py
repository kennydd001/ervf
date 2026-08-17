from __future__ import annotations
import argparse,json,statistics,time,traceback
from pathlib import Path
import numpy as np

def bench(req, inputs, reps=30, share=False):
    for _ in range(6):
        req.infer(inputs,share_inputs=share,share_outputs=share)
    vals=[]
    for _ in range(reps):
        t=time.perf_counter()
        req.infer(inputs,share_inputs=share,share_outputs=share)
        vals.append((time.perf_counter()-t)*1e3)
    return {"median_ms":statistics.median(vals),"p95_ms":float(np.percentile(vals,95)),
            "min_ms":min(vals),"max_ms":max(vals),"samples_ms":vals}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shape',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    shape=json.loads(Path(args.shape).read_text())
    if "shape_contract" in shape: shape=shape["shape_contract"]
    hidden=int(shape["hidden"]);inter=int(shape["moe_inter"])
    out=Path(args.out)
    p={"kind":"s100_phase8_openvino_distinct_experts","status":"started",
       "hidden":hidden,"moe_inter":inter,"records":[]}
    try:
        import openvino as ov
        from openvino import opset13 as ops
        core=ov.Core()
        gpu=next(d for d in core.available_devices if d.upper().startswith("GPU"))
        p["device"]=gpu;p["openvino_version"]=ov.__version__
        rng=np.random.default_rng(260818)

        # Same-weight M scaling: batch rows share one matrix.
        for dtype in ("f16","i8"):
            for n in (1,2,4,6):
                rec={"kind":"same_weight_batch","dtype":dtype,"experts_or_rows":n}
                try:
                    if dtype=="f16":
                        W=(rng.standard_normal((inter,hidden),dtype=np.float32)*.01).astype(np.float16)
                        X=(rng.standard_normal((n,inter),dtype=np.float32)*.01).astype(np.float16)
                        A=ops.parameter([n,inter],np.float16,name="A")
                    else:
                        W=rng.integers(-7,8,size=(inter,hidden),dtype=np.int8)
                        X=rng.integers(-7,8,size=(n,inter),dtype=np.int8)
                        A=ops.parameter([n,inter],np.int8,name="A")
                    Y=ops.matmul(A,ops.constant(W),False,False)
                    model=ov.Model([Y],[A],f"same_{dtype}_{n}")
                    comp=core.compile_model(model,gpu,{"PERFORMANCE_HINT":"LATENCY"})
                    req=comp.create_infer_request()
                    rec["standard"]=bench(req,{0:X},20,False)
                    try: rec["shared_io"]=bench(req,{0:X},20,True)
                    except Exception as e: rec["shared_io"]={"status":"unsupported","error":str(e)}
                    rec["status"]="measured"
                except Exception as e: rec.update({"status":"technical_failure","error":f"{type(e).__name__}: {e}"})
                p["records"].append(rec)

        # N distinct down matrices in one graph, already route-weighted to one hidden output.
        for dtype in ("f16","i8"):
            for n in (1,2,4,6):
                rec={"kind":"distinct_down_weighted_sum","dtype":dtype,"experts_or_rows":n}
                try:
                    params=[];feeds={};summed=None
                    route=np.linspace(.7,1.3,n,dtype=np.float32);route/=route.sum()
                    for s in range(n):
                        if dtype=="f16":
                            W=(rng.standard_normal((inter,hidden),dtype=np.float32)*.01).astype(np.float16)
                            x=(rng.standard_normal((1,inter),dtype=np.float32)*.01).astype(np.float16)
                            A=ops.parameter([1,inter],np.float16,name=f"A{s}")
                            y=ops.matmul(A,ops.constant(W),False,False)
                            yf=ops.convert(y,np.float32)
                        else:
                            W=rng.integers(-7,8,size=(inter,hidden),dtype=np.int8)
                            x=rng.integers(-7,8,size=(1,inter),dtype=np.int8)
                            A=ops.parameter([1,inter],np.int8,name=f"A{s}")
                            y=ops.matmul(A,ops.constant(W),False,False)
                            yf=ops.convert(y,np.float32)
                        term=ops.multiply(yf,ops.constant(np.float32(route[s])))
                        summed=term if summed is None else ops.add(summed,term)
                        params.append(A);feeds[s]=x
                    model=ov.Model([summed],params,f"distinct_down_{dtype}_{n}")
                    comp=core.compile_model(model,gpu,{"PERFORMANCE_HINT":"LATENCY"})
                    req=comp.create_infer_request()
                    rec["standard"]=bench(req,feeds,20,False)
                    try: rec["shared_io"]=bench(req,feeds,20,True)
                    except Exception as e: rec["shared_io"]={"status":"unsupported","error":str(e)}
                    rec["status"]="measured"
                except Exception as e: rec.update({"status":"technical_failure","error":f"{type(e).__name__}: {e}"})
                p["records"].append(rec)

        # Full FP16 routed experts: hidden -> inter -> ReLU^2 -> hidden -> weighted sum.
        for n in (1,2,4,6):
            rec={"kind":"distinct_full_expert","dtype":"f16","experts_or_rows":n}
            try:
                params=[];feeds={};summed=None
                route=np.linspace(.7,1.3,n,dtype=np.float32);route/=route.sum()
                for s in range(n):
                    Wu=(rng.standard_normal((hidden,inter),dtype=np.float32)*.01).astype(np.float16)
                    Wd=(rng.standard_normal((inter,hidden),dtype=np.float32)*.01).astype(np.float16)
                    x=(rng.standard_normal((1,hidden),dtype=np.float32)*.01).astype(np.float16)
                    A=ops.parameter([1,hidden],np.float16,name=f"A{s}")
                    u=ops.matmul(A,ops.constant(Wu),False,False)
                    r=ops.relu(u); r2=ops.multiply(r,r)
                    y=ops.matmul(r2,ops.constant(Wd),False,False)
                    term=ops.multiply(ops.convert(y,np.float32),ops.constant(np.float32(route[s])))
                    summed=term if summed is None else ops.add(summed,term)
                    params.append(A);feeds[s]=x
                model=ov.Model([summed],params,f"full_expert_{n}")
                comp=core.compile_model(model,gpu,{"PERFORMANCE_HINT":"LATENCY"})
                req=comp.create_infer_request()
                rec["standard"]=bench(req,feeds,16,False)
                try: rec["shared_io"]=bench(req,feeds,16,True)
                except Exception as e: rec["shared_io"]={"status":"unsupported","error":str(e)}
                rec["status"]="measured"
            except Exception as e: rec.update({"status":"technical_failure","error":f"{type(e).__name__}: {e}"})
            p["records"].append(rec)
        p["status"]="measured"
    except Exception as e:
        p.update({"status":"technical_failure","error":{"type":type(e).__name__,
                  "message":str(e),"traceback":traceback.format_exc()}})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n")
    print(json.dumps(p,indent=2,allow_nan=False))
    return 0 if p["status"]=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
