from __future__ import annotations
import hashlib,json,statistics
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,cupy as cp
from safetensors import safe_open
import run_port80b_t0y_canonical_router_accumulation as base
ROOT=base.ROOT;REPORTS=base.REPORTS;RUN=ROOT/'reports/runs/streamq5_moe/port80b_t0yp_canonical_router_performance';PREREG=REPORTS/'PORT80B_T0YP_CANONICAL_ROUTER_PERFORMANCE_PREREGISTRATION_2026-08-13.md';LOCK=REPORTS/'port80b_t0yp_canonical_router_performance_lock.json';VERIFY=ROOT/'scripts/streamq5_moe/verify_port80b_t0yp_canonical_router_performance.py';T0YR=ROOT/'reports/runs/streamq5_moe/port80b_t0yr_canonical_router/t0yr_canonical_router_raw.safetensors'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def pct(xs,q):return float(np.percentile(np.asarray(xs,np.float64),q,method='linear'))
def main():
 l=json.loads(LOCK.read_text());s={'runner':sha(__file__),'verifier':sha(VERIFY),'prereg':sha(PREREG)}
 if s!=l['source_sha256'] or sha(T0YR)!=l['t0yr_raw_sha256']:raise RuntimeError('lock mismatch pre-CUDA')
 with safe_open(str(base.RAW6),framework='pt',device='cpu') as f:x=f.get_tensor('official_gate_input').contiguous()
 with safe_open(str(base.SHARD),framework='pt',device='cpu') as f:w=f.get_tensor('model.layers.0.mlp.gate.weight').contiguous()
 with safe_open(str(T0YR),framework='np') as f:oracle_logits=f.get_tensor('gpu1_logits');oracle_ids=f.get_tensor('gpu1_ids')
 inc=ROOT/'.venv/Lib/site-packages/nvidia/cu13/include';mod=cp.RawModule(code=base.CUDA,options=('--std=c++11',f'--include-path={inc}'),name_expressions=('canonical_logits','stable_top10'));kl=mod.get_function('canonical_logits');kt=mod.get_function('stable_top10')
 xt=x.cuda();wt=w.cuda();xg=cp.asarray(x.view(torch.uint16).numpy());wg=cp.asarray(w.view(torch.uint16).numpy());lg=cp.empty((16,512),cp.float32);ig=cp.empty((16,10),cp.int64)
 kl((2,16),(256,),(xg,wg,lg));kt((16,),(1,),(lg,ig));cp.cuda.runtime.deviceSynchronize()
 exact=np.array_equal(cp.asnumpy(lg).view(np.uint32),oracle_logits.view(np.uint32)) and np.array_equal(cp.asnumpy(ig),oracle_ids)
 if not exact:raise RuntimeError('candidate drifted from T0Y-R exact oracle')
 def ref():
  z=F.linear(xt,wt);p=torch.softmax(z,dtype=torch.float32,dim=-1);v,i=torch.topk(p,10,dim=-1);v=v/v.sum(-1,keepdim=True);return i,v.to(z.dtype)
 def cand():kl((2,16),(256,),(xg,wg,lg));kt((16,),(1,),(lg,ig))
 for _ in range(30):ref();cand()
 torch.cuda.synchronize();cp.cuda.runtime.deviceSynchronize()
 def timeone(fn,torch_arm):
  if torch_arm:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();return float(a.elapsed_time(b))
  a=cp.cuda.Event();b=cp.cuda.Event();a.record();fn();b.record();b.synchronize();return float(cp.cuda.get_elapsed_time(a,b))
 def phase(n):
  rr=[];cc=[];raw=[]
  for i in range(n):
   order=('reference','candidate') if i%2==0 else ('candidate','reference')
   vals={}
   for arm in order:vals[arm]=timeone(ref,True) if arm=='reference' else timeone(cand,False)
   rr.append(vals['reference']);cc.append(vals['candidate']);raw.append({'pair':i,'order':order,'reference_ms':vals['reference'],'candidate_ms':vals['candidate']})
  return {'reference':rr,'candidate':cc,'raw':raw,'ratio_p50':pct(cc,50)/pct(rr,50),'ratio_p95':pct(cc,95)/pct(rr,95),'reference_p50_ms':pct(rr,50),'reference_p95_ms':pct(rr,95),'candidate_p50_ms':pct(cc,50),'candidate_p95_ms':pct(cc,95)}
 val=phase(80);opened=val['ratio_p50']<=4 and val['ratio_p95']<=5;test=phase(240) if opened else None;passed=bool(test and test['ratio_p50']<=2 and test['ratio_p95']<=2.5)
 out={'kind':'port80b_t0yp_canonical_router_performance','status':'performance_pass' if passed else 'performance_negative','overall_pass':passed,'correctness_replay_pass':exact,'validation':val,'test_opened':opened,'test':test,'sources':s,'t0yr_raw_sha256':sha(T0YR),'claim_boundary':'Resident 16-row component timing; candidate omits probability/weight output and is not full router parity.'};RUN.mkdir(parents=True,exist_ok=True);(RUN/'t0yp_canonical_router_performance.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('validation','test')}|{'validation_summary':{k:v for k,v in val.items() if k not in ('raw','reference','candidate')},'test_summary':None if test is None else {k:v for k,v in test.items() if k not in ('raw','reference','candidate')}},indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
