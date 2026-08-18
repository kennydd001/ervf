from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
import numpy as np
from common import REPO,write_json_atomic,require_model_dir
from diag_component_marginals_graph import _prefill,_reset_exact_state
from s100_phase3_fidelity import _advance
from s100_phase5_quality import load_trace
from s100_phase10a_runtime import build
from s100_phase10a_panel_kernels import ROWHALF,CODE_PANEL_BYTES
BUDGETS=(8,16,24,32,40,48)
def collect(b,kind):
 import cupy as cp
 from transformers import AutoTokenizer
 rt=b.rt;ps,ix,n,d,m=load_trace(kind);t=d["target_ids"].astype(np.int32);layers=[int(x) for x in rt.moe_layers]
 tok=AutoTokenizer.from_pretrained(str(require_model_dir()),local_files_only=True,trust_remote_code=True,use_fast=True)
 stat=defaultdict(int);total=0
 for pi,p in enumerate(ps):
  ids0=tok.encode(p["prompt"],add_special_tokens=False);_reset_exact_state(rt);_prefill(rt,ids0)
  for ti in range(n):
   _advance(rt,int(t[pi,ti]));rt._graph_stream.synchronize()
   ids=cp.asnumpy(cp.stack([rt._dev_cache[l]["ids"][:rt.top_k] for l in layers]))
   masks=cp.asnumpy(cp.stack([b.state[l]["masks"] for l in layers])).reshape(len(layers),rt.top_k,-1)
   for li,l in enumerate(layers):
    for s in range(rt.top_k):
     e=int(ids[li,s]);row=masks[li,s]
     for pno in np.flatnonzero(row):
      bb=int(int(row[pno]).bit_count()*ROWHALF);stat[(l,e,int(pno))]+=bb;total+=bb
 return stat,total
def main():
 a=argparse.ArgumentParser();a.add_argument("--alpha",type=float,default=.0003);q=a.parse_args()
 R=REPO/"pro_research"/"results"/"s100_phase10a";R.mkdir(parents=True,exist_ok=True)
 b=build(q.alpha,expose=True);cal,ct=collect(b,"calibration");rank=sorted(cal.items(),key=lambda z:(-z[1],z[0]))
 sel={}
 for mib in BUDGETS:
  n=(mib*1024**2)//CODE_PANEL_BYTES;items=[{"layer":k[0],"expert":k[1],"panel":k[2],"calibration_saved_bytes":int(v)} for k,v in rank[:n]]
  h=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest()
  sel[str(mib)]={"budget_mib":mib,"panel_count":len(items),"panel_bytes":len(items)*CODE_PANEL_BYTES,"selection_sha256":h,"items":items}
 val,vt=collect(b,"validation")
 for s in sel.values():
  keys={(x["layer"],x["expert"],x["panel"]) for x in s["items"]};saved=sum(v for k,v in val.items() if k in keys)
  s["validation_active_code_bytes"]=int(vt);s["validation_saved_bytes"]=int(saved);s["validation_byte_coverage"]=saved/vt if vt else 0
 out={"kind":"s100_phase10a_panel_profile","status":"measured","alpha":q.alpha,"code_panel_bytes":CODE_PANEL_BYTES,"calibration_active_code_bytes":int(ct),"selections":sel}
 write_json_atomic(R/"S100_PHASE10A_PANEL_PROFILE.json",out,archive=True);print(json.dumps({"status":"measured","validation_coverage":{k:v["validation_byte_coverage"] for k,v in sel.items()}},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
