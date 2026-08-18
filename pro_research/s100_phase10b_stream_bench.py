from __future__ import annotations
import json,statistics,subprocess
import numpy as np
from common import REPO,require_model_dir,write_json_atomic
from ervf_dense import DenseERVF
from s100_phase10b_mamba_kernels import MambaERVF2,VARIANTS
def smi():
 p=subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate","--format=csv,noheader,nounits"],capture_output=True,text=True);return p.stdout.strip()
def main():
 import cupy as cp
 from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
 rt=LightningRuntime(require_model_dir(),contexts_max=4096,embed_on_host=True,fp8_kv=True,verbose=False)
 base=DenseERVF();cand=MambaERVF2()
 xh=cp.random.standard_normal(rt.hidden,dtype=cp.float32);xo=cp.random.standard_normal(rt.d_inner,dtype=cp.float32)
 cases=[]
 for l in rt.mamba_layers:
  d=rt.layer[l]
  if d["in_k"]=="fp8_tensor":cases.append((l,"in",d["in_w8"],float(d["in_s"]),int(rt.proj.size),rt.hidden,xh))
  if d["out_k"]=="fp8_tensor":cases.append((l,"out",d["out_w8"],float(d["out_s"]),rt.hidden,rt.d_inner,xo))
 total=sum(int(W.nbytes) for _,_,W,_,_,_,_ in cases);exact={}
 for name in VARIANTS:
  ok=True;mx=0.0
  for l,f,W,sc,r,c,x in cases:
   a=cp.empty(r,cp.float32);b=cp.empty(r,cp.float32);base.mv_fp8_tensor(a,W,x,sc,r,c);cand.run(name,b,W,x,sc,r,c);cp.cuda.Stream.null.synchronize()
   if not bool(cp.array_equal(a,b).item()):ok=False;mx=max(mx,float(cp.max(cp.abs(a-b)).item()))
  exact[name]={"bitexact":ok,"max_abs":mx}
 def measure(name):
  for _ in range(3):
   for l,f,W,sc,r,c,x in cases:
    o=cp.empty(r,cp.float32)
    if name=="baseline":base.mv_fp8_tensor(o,W,x,sc,r,c)
    else:cand.run(name,o,W,x,sc,r,c)
  cp.cuda.Stream.null.synchronize();vals=[]
  for _ in range(24):
   s=cp.cuda.Event();e=cp.cuda.Event();s.record()
   for l,f,W,sc,r,c,x in cases:
    o=cp.empty(r,cp.float32)
    if name=="baseline":base.mv_fp8_tensor(o,W,x,sc,r,c)
    else:cand.run(name,o,W,x,sc,r,c)
   e.record();e.synchronize();vals.append(float(cp.cuda.get_elapsed_time(s,e)))
  med=statistics.median(vals);return {"median_ms":med,"p95_ms":float(np.percentile(vals,95)),"gb_s":total/(med*1e-3)/1e9,"raw_ms":vals}
 before=smi();b=measure("baseline");rows=[]
 for name in VARIANTS:
  t=measure(name);rows.append({"variant":name,"exact":exact[name],"timing":t,"speedup":b["median_ms"]/t["median_ms"]})
 after=smi();eligible=sorted([x for x in rows if x["exact"]["bitexact"] and x["speedup"]>=1.05],key=lambda x:x["timing"]["median_ms"]);selected=[x["variant"] for x in eligible[:2]]
 out={"kind":"s100_phase10b_mamba_stream","status":"measured","matrix_count":len(cases),"weight_bytes_per_token_equivalent":total,"baseline":b,"rows":rows,"selected_for_integration":selected,"smi_before":before,"smi_after":after,"claim_boundary":"real-weight cold stream, not end-to-end"}
 R=REPO/"pro_research"/"results"/"s100_phase10b";R.mkdir(parents=True,exist_ok=True);write_json_atomic(R/"S100_PHASE10B_STREAM.json",out,archive=True);print(json.dumps({"baseline":b,"selected":selected,"top":eligible[:3]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
