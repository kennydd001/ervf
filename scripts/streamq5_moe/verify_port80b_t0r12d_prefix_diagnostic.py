#!/usr/bin/env python3
"""Independent recomputation of R12-D raw diagnostic evidence."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";D=ROOT/"reports/runs/streamq5_moe/port80b_t0r12d_prefix_diagnostic";RUN=ROOT/"scripts/streamq5_moe/run_port80b_t0r12d_prefix_diagnostic.py";LOCK=R/"port80b_t0r12d_runner_lock.json";VL=R/"port80b_t0r12d_verifier_lock.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*2**20),b''):h.update(x)
 return h.hexdigest()
def tsha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def ordered(t):
 x=t.contiguous().view(torch.uint16).numpy();return np.where((x&0x8000)!=0,0x8000-(x&0x7fff),0x8000+x).astype(np.int32)
def met(a,b):
 d=a.float()-b.float();den=float(torch.linalg.vector_norm(a.float()));return {"diff_words":int((a.view(torch.uint16)!=b.view(torch.uint16)).sum()),"max_bf16_ulp":int(np.max(np.abs(ordered(a)-ordered(b)))),"max_abs":float(d.abs().max()),"relative_l2":float(torch.linalg.vector_norm(d))/max(den,1e-30),"finite":bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()),"a_sha256":tsha(a),"b_sha256":tsha(b)}
def preflight():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());c={"runner":sha(RUN)==l["runner_sha256"],"verifier":sha(__file__)==l["verifier_sha256"]==v["verifier_sha256"],"vlock":sha(VL)==l["verifier_lock_sha256"],"outputs_absent":not D.exists()};return {"kind":"r12d_verifier_preflight","pass":all(c.values()),"checks":c}
def verify():
 r=json.loads((D/"t0r12d_result.json").read_text());x={}
 with safe_open(D/"t0r12d_raw.safetensors",framework="pt",device="cpu") as f:
  for k in f.keys():x[k]=f.get_tensor(k)
 rows=[{"prompt":p,"length":n,**met(x[f"p{p}_whole_output"][:,n-1:n],x[f"p{p}_prefix{n}_final"])} for p in range(4) for n in range(1,17)]
 caches=[{"prompt":p,"conv":met(x[f"p{p}_whole_cache_conv"],x[f"p{p}_prefix16_cache_conv"]),"recurrent":met(x[f"p{p}_whole_cache_recurrent"],x[f"p{p}_prefix16_cache_recurrent"])} for p in range(4)]
 repeats={"whole_prompt1":met(x["p1_whole_output"],x["p1_whole_repeat_output"]),"prefix3_prompt1":met(x["p1_prefix3_final"],x["p1_prefix3_repeat_final"])}
 manifest={k:{"semantic_key":k,"dtype":str(v.dtype),"shape":list(v.shape),"bytes":v.numel()*v.element_size(),"sha256":tsha(v)} for k,v in sorted(x.items())}
 c={"raw_sha":sha(D/"t0r12d_raw.safetensors")==r["raw_sha256"],"metrics_exact":rows==r["whole_prefix_metrics"],"cache_exact":caches==r["cache_metrics"],"repeats_exact":repeats==r["same_length_repeat"],"manifest_exact":manifest==r["raw_manifest"],"finite":all(torch.isfinite(v.float()).all() for v in x.values()),"diagnostic_only":r["status"]=="diagnostic_only_not_pass" and r["cuda_initialized"] is False};return {"kind":"r12d_independent_verification","pass":False,"valid_diagnostic":all(c.values()),"checks":c,"claim_boundary":"Diagnostic validity only; never a pass."}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preflight','verify'),required=True);a=p.parse_args();r=preflight() if a.phase=='preflight' else verify();print(json.dumps(r,indent=2));return 0 if r.get('pass') else (3 if r.get('valid_diagnostic') else 2)
if __name__=='__main__':raise SystemExit(main())
