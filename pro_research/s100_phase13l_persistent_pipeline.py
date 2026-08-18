from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
BLOCKS=(4,8)

def median_time(fn, reps, torch):
    for _ in range(3): fn()
    torch.cuda.synchronize(); vals=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); vals.append((time.perf_counter()-t)*1000)
    return float(np.median(vals))

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",default="models/nemotron_3_5_lightning"); ap.add_argument("--rank",type=int,default=256); ap.add_argument("--reps",type=int,default=16); ap.add_argument("--output",type=Path,default=Path("pro_research/results/s100_phase13l/S100_PHASE13L_PERSISTENT_PIPELINE.json")); args=ap.parse_args()
    sys.path.insert(0,str(REPO/"src")); os.environ["LS_MODEL_DIR"]=str(Path(args.model_dir).resolve())
    import torch
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    rt=LightningRuntime(Path(args.model_dir).resolve(),contexts_max=4096,embed_on_host=True,fp8_kv=True,verbose=False); layer=4; d=rt.layer[layer]; rows,cols=int(rt.proj.size),int(rt.hidden)
    W=torch.utils.dlpack.from_dlpack(d["in_w"]).view(torch.bfloat16).reshape(rows,cols).clone().float()
    U,_=torch.linalg.qr(torch.randn(cols,args.rank,device="cuda",dtype=torch.float32),mode="reduced"); WU=torch.mm(W,U)
    records=[]
    for B in BLOCKS:
        x=torch.randn(B,cols,device="cuda",dtype=torch.float32); z=x@U; sub=z@WU.t(); exact=x@W.t(); residual=torch.linalg.vector_norm(x-z@U.t(),dim=1); threshold=torch.median(residual); fast=residual<=threshold
        def exact_fn(): torch.mm(x,W.t())
        def sub_fn():
            zz=torch.mm(x,U); torch.mm(zz,WU.t())
        def gated_fn():
            zz=torch.mm(x,U); candidate=torch.mm(zz,WU.t()); full=torch.mm(x,W.t()); torch.where((torch.linalg.vector_norm(x-zz@U.t(),dim=1)<=threshold)[:,None],candidate,full)
        exact_ms=median_time(exact_fn,args.reps,torch); sub_ms=median_time(sub_fn,args.reps,torch); gated_ms=median_time(gated_fn,args.reps,torch)
        records.append({"B":B,"rank":args.rank,"fast_fraction":float(fast.float().mean()),"subspace_output_nrmse":float(torch.linalg.vector_norm(sub-exact)/torch.linalg.vector_norm(exact).clamp_min(1e-12)),"exact_ms":exact_ms,"subspace_ms":sub_ms,"gated_fallback_ms":gated_ms,"subspace_speedup":exact_ms/sub_ms,"gated_speedup":exact_ms/gated_ms})
    result={"kind":"s100_phase13l_persistent_compressed_pipeline","status":"measured","created_utc":datetime.now(timezone.utc).isoformat(),"model_dir":str(Path(args.model_dir).resolve()),"claim_boundary":"GPU eager-operator pipeline prototype; not a persistent custom CUDA kernel and no quality claim","pipeline":["U^T x","WU projection","residual norm gate","exact fallback"],"records":records,"gates":{"persistent_kernel_measured":False,"official_quality_green":False,"promotion_open":False}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"status":result["status"],"records":len(records),"promotion_open":False},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
