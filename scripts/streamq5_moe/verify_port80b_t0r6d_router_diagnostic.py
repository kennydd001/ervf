#!/usr/bin/env python3
"""Independent verifier for R6-D router diagnostics; never returns a scientific pass."""
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";D=ROOT/"reports/runs/streamq5_moe/port80b_t0r6d_router_diagnostic";RUNNER=ROOT/"scripts/streamq5_moe/run_port80b_t0r6d_router_diagnostic.py";LOCK=R/"port80b_t0r6d_runner_lock.json";VLOCK=R/"port80b_t0r6d_verifier_lock.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""):h.update(b)
 return h.hexdigest()
def preflight():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());c={"runner":sha(RUNNER)==l["runner_sha256"],"verifier":sha(__file__)==l["verifier_sha256"]==v["verifier_sha256"],"vlock":sha(VLOCK)==l["verifier_lock_sha256"],"outputs_absent":not D.exists()};return {"kind":"port80b_t0r6d_verifier_preflight","pass":all(c.values()),"checks":c}
def verify():
 result=json.loads((D/"t0r6d_router_diagnostic.json").read_text());raw={}
 with safe_open(D/"t0r6d_router_raw.safetensors",framework="pt",device="cpu") as h:
  for k in h.keys():raw[k]=h.get_tensor(k)
 logits=raw["official_logits"];probs=torch.softmax(logits.float(),dim=-1);v11,i11=torch.topk(probs,11,dim=-1);pre=v11[:,:10];pre=pre/pre.sum(-1,keepdim=True);bf=pre.to(torch.bfloat16);margin=v11[:,9]-v11[:,10];l11=torch.gather(logits,1,i11);rows=[]
 for r in range(16):
  ties=torch.nonzero(probs[r]==v11[r,9]).flatten();selected=raw["official_ids"][r];bits=l11[r].contiguous().view(torch.uint16);checks={"ids_equal":torch.equal(selected,i11[r,:10]),"weights_equal":torch.equal(raw["official_weights"][r],bf[r]),"finite":bool(torch.isfinite(probs[r]).all() and torch.isfinite(raw["official_weights"][r].float()).all()),"positive":bool((pre[r]>0).all() and (raw["official_weights"][r]>0).all()),"monotonic":bool((pre[r,:-1]>=pre[r,1:]).all() and (raw["official_weights"][r,:-1]>=raw["official_weights"][r,1:]).all()),"unique":torch.unique(selected).numel()==10,"bounds":bool((selected>=0).all() and (selected<512).all())}
  rows.append({"row":r,"checks":checks,"fp32_sum_error":float((pre[r].sum()-1).abs()),"bf16_sum_error":float((bf[r].float().sum()-1).abs()),"probability_margin":float(margin[r]),"rank10":{"expert":int(i11[r,9]),"native_bf16_logit":float(l11[r,9]),"u16_bits":int(bits[9])},"rank11":{"expert":int(i11[r,10]),"native_bf16_logit":float(l11[r,10]),"u16_bits":int(bits[10])},"boundary_tie_expert_ids":[int(x) for x in ties],"selected_boundary_subset":[int(x) for x in selected if bool((ties==x).any())]})
 failed={k for x in rows for k,v in x["checks"].items() if not v};verdict="strict_margin_negative" if bool((margin==0).any()) else ("failed_conjunct:"+",".join(sorted(failed)) if failed else "no_failure_reproduced")
 c={"raw_sha":sha(D/"t0r6d_router_raw.safetensors")==result["raw_sha256"],"rows_exact":rows==result["rows"],"verdict_exact":verdict==result["verdict"],"two_calls_equal":torch.equal(raw["official_logits"],raw["second_logits"]) and torch.equal(raw["official_weights"],raw["second_weights"]) and torch.equal(raw["official_ids"],raw["second_ids"]),"diagnostic_only":result["status"]=="diagnostic_negative_not_pass" and result["bank_built"] is False and result["cuda_initialized"] is False};return {"kind":"port80b_t0r6d_independent_verification","pass":False,"valid_diagnostic":all(c.values()),"verdict":verdict,"checks":c,"claim_boundary":"Valid diagnostic is not a scientific pass."}
def main():
 p=argparse.ArgumentParser();p.add_argument("--phase",choices=("preflight","verify"),required=True);a=p.parse_args();r=preflight() if a.phase=="preflight" else verify();print(json.dumps(r,indent=2));return 0 if a.phase=="preflight" and r["pass"] else (3 if r.get("valid_diagnostic") else 2)
if __name__=="__main__":raise SystemExit(main())
