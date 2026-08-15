#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2_full_stage_diagnostic';RUN=ROOT/'scripts/streamq5_moe/run_port80b_t0r12d2_full_stage_diagnostic.py';LOCK=R/'port80b_t0r12d2_runner_lock.json';VL=R/'port80b_t0r12d2_verifier_lock.json';PR=R/'PORT80B_T0R12D2_FULL_STAGE_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md';BASE=ROOT/'scripts/streamq5_moe/run_port80b_t0r12_official_cpu_reference_only.py';FAIL=ROOT/'reports/runs/streamq5_moe/port80b_t0r12_official_cpu_reference_only/t0r12_capture_1_failure.json'
STAGES={'input_norm':('torch.bfloat16',2048),'gdn':('torch.bfloat16',2048),'post_norm':('torch.bfloat16',2048),'official_router_logits':('torch.bfloat16',512),'official_router_weights':('torch.bfloat16',10),'official_router_ids':('torch.int64',10),'experts':('torch.bfloat16',2048),'shared':('torch.bfloat16',2048),'shared_gate':('torch.bfloat16',1),'layer_output':('torch.bfloat16',2048)}
ROUTES={'diagnostic_router_logits':('torch.bfloat16',512),'diagnostic_router_weights':('torch.bfloat16',10),'diagnostic_router_ids':('torch.int64',10),'router_logits_fp32':('torch.float32',512),'router_probs_fp32':('torch.float32',512),'router_top10_ids_recomputed':('torch.int64',10),'router_weights_precast_fp32':('torch.float32',10),'router_weights_recomputed_bf16':('torch.bfloat16',10),'router_top10_top11_margin_fp32':('torch.float32',None),'router_boundary_tie_mask':('torch.bool',512),'router_selected_boundary_mask':('torch.bool',10),'router_top11_ids':('torch.int64',11),'router_top11_native_bf16_logits':('torch.bfloat16',11)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*2**20),b''):h.update(z)
 return h.hexdigest()
def tb(t):return t.contiguous().view(torch.uint8).numpy().tobytes()
def schema():
 s={'token_ids':('torch.int64',[4,16]),'embedding':('torch.bfloat16',[4,16,2048])}
 prefixes=[f'p{p}_whole' for p in range(4)]+[f'p{p}_n{n}' for p in range(4) for n in range(1,17)]+['p1_whole_repeat','p1_n3_repeat']
 for x in prefixes:
  length=16 if ('whole' in x) else (3 if x=='p1_n3_repeat' else int(x.rsplit('n',1)[1]))
  for k,(d,w) in STAGES.items():s[f'{x}_{k}']=(d,[1,length,w] if k in ('input_norm','gdn','post_norm','layer_output') else [length,w])
  for k,(d,w) in ROUTES.items():s[f'{x}_{k}']=(d,[length] if w is None else [length,w])
  s[f'{x}_cache_conv']=('torch.bfloat16',[1,8192,4]);s[f'{x}_cache_recurrent']=('torch.float32',[1,32,128,128])
 return s
def preflight():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());actual={'runner_sha256':sha(RUN),'verifier_sha256':sha(__file__),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'base_sha256':sha(BASE),'failure_sha256':sha(FAIL)};c={'bindings':all(l.get(k)==x for k,x in actual.items()),'verifier_bound':v['verifier_sha256']==actual['verifier_sha256'],'outputs_absent':not D.exists()};return {'kind':'d2_verifier_preflight','pass':all(c.values()),'checks':c}
def ordered(t):
 q=t.contiguous().view(torch.uint16).to(torch.int32);return torch.where((q&0x8000)!=0,0x8000-(q&0x7fff),0x8000+q)
def last(t,k,n):return t[n-1:n] if k in ('official_router_logits','official_router_weights','official_router_ids','experts','shared','shared_gate') else t[:,n-1:n]
def metric(a,z):
 out={'dtype':str(a.dtype),'shape':list(a.shape),'different_elements':int((a!=z).sum()),'exact_equal':bool(torch.equal(a,z)),'reference_sha256':hashlib.sha256(tb(a)).hexdigest(),'observed_sha256':hashlib.sha256(tb(z)).hexdigest()}
 if a.dtype==torch.bfloat16:out.update(max_bf16_ulp=int((ordered(a)-ordered(z)).abs().max()),max_abs=float((a.float()-z.float()).abs().max()),rel_l2=float(torch.linalg.vector_norm(a.float()-z.float())/torch.linalg.vector_norm(a.float()).clamp_min(1e-30)))
 return out
def recompute_metrics(x):
 rows=[]
 for p in range(4):
  for n in range(1,17):rows.append({'prompt':p,'length':n,'stages':{s:metric(last(x[f'p{p}_whole_{s}'],s,n),last(x[f'p{p}_n{n}_{s}'],s,n)) for s in STAGES}})
 repeats={}
 for a,z in (('p1_whole','p1_whole_repeat'),('p1_n3','p1_n3_repeat')):repeats[z]={'stages':{s:metric(x[f'{a}_{s}'],x[f'{z}_{s}']) for s in STAGES},'cache_conv':metric(x[f'{a}_cache_conv'],x[f'{z}_cache_conv']),'cache_recurrent':metric(x[f'{a}_cache_recurrent'],x[f'{z}_cache_recurrent'])}
 first={str(p):{s:next((q['length'] for q in rows if q['prompt']==p and not q['stages'][s]['exact_equal']),None) for s in STAGES} for p in range(4)};cache16={str(p):{'conv':metric(x[f'p{p}_whole_cache_conv'],x[f'p{p}_n16_cache_conv']),'recurrent':metric(x[f'p{p}_whole_cache_recurrent'],x[f'p{p}_n16_cache_recurrent'])} for p in range(4)};classes={'same_length_nondeterminism_observed':any(not m['exact_equal'] for q in repeats.values() for m in list(q['stages'].values())+[q['cache_conv'],q['cache_recurrent']]),'first_divergent_length_by_prompt_stage':first,'whole_prefix16_cache':cache16,'whole_prefix16_cache_divergence':any(not m['exact_equal'] for q in cache16.values() for m in q.values())};return rows,repeats,classes
def verify():
 r=json.loads((D/'t0r12d2_result.json').read_text());x={}
 with safe_open(D/'t0r12d2_raw.safetensors',framework='pt',device='cpu') as f:
  for k in f.keys():x[k]=f.get_tensor(k)
 sc=schema();manifest={k:{'semantic_key':k,'dtype':str(v.dtype),'shape':list(v.shape),'bytes':v.numel()*v.element_size(),'sha256':hashlib.sha256(tb(v)).hexdigest()} for k,v in sorted(x.items())};tuple_ok=True;ties_ok=True
 for prefix in [f'p{p}_whole' for p in range(4)]+[f'p{p}_n{n}' for p in range(4) for n in range(1,17)]+['p1_whole_repeat','p1_n3_repeat']:
  logits=x[f'{prefix}_official_router_logits'];probs=torch.softmax(logits.float(),-1);v11,i11=torch.topk(probs,11,-1);w=v11[:,:10];w=w/w.sum(-1,keepdim=True);ids=i11[:,:10];boundary=probs==v11[:,9:10];tuple_ok &= torch.equal(logits,x[f'{prefix}_diagnostic_router_logits']) and torch.equal(x[f'{prefix}_official_router_weights'],x[f'{prefix}_diagnostic_router_weights']) and torch.equal(x[f'{prefix}_official_router_ids'],x[f'{prefix}_diagnostic_router_ids']) and torch.equal(logits.float(),x[f'{prefix}_router_logits_fp32']) and torch.equal(probs,x[f'{prefix}_router_probs_fp32']) and torch.equal(ids,x[f'{prefix}_router_top10_ids_recomputed']) and torch.equal(w,x[f'{prefix}_router_weights_precast_fp32']) and torch.equal(w.to(torch.bfloat16),x[f'{prefix}_router_weights_recomputed_bf16']) and torch.equal(x[f'{prefix}_official_router_weights'],w.to(torch.bfloat16)) and torch.equal(x[f'{prefix}_official_router_ids'],ids);ties_ok &= torch.equal(boundary,x[f'{prefix}_router_boundary_tie_mask']) and torch.equal(torch.gather(boundary,1,x[f'{prefix}_official_router_ids']),x[f'{prefix}_router_selected_boundary_mask']) and torch.equal(i11,x[f'{prefix}_router_top11_ids']) and torch.equal(torch.gather(logits,1,i11),x[f'{prefix}_router_top11_native_bf16_logits']) and torch.equal(v11[:,9]-v11[:,10],x[f'{prefix}_router_top10_top11_margin_fp32'])
 rows,repeats,classes=recompute_metrics(x)
 c={'raw_sha':sha(D/'t0r12d2_raw.safetensors')==r['raw_sha256'],'schema_exact':set(x)==set(sc) and all(str(x[k].dtype)==d and list(x[k].shape)==q for k,(d,q) in sc.items()),'manifest_exact':manifest==r['raw_manifest'],'finite':all(bool(torch.isfinite(v.float()).all()) for v in x.values()),'direct_tuple_recomputed':tuple_ok,'ties_recomputed':ties_ok,'metrics_recomputed':rows==r['stage_metrics'] and repeats==r['repeat_metrics'] and classes==r['interpretation_classes'],'provenance':r['runner_sha256']==sha(RUN) and r['verifier_sha256']==sha(__file__) and r['verifier_lock_sha256']==sha(VL) and r['prereg_sha256']==sha(PR) and r['base_sha256']==sha(BASE) and r['failure_sha256']==sha(FAIL),'resources':r['resources']['windows_peak_working_set_bytes']<=12*2**30 and r['resources']['minimum_available_ram_bytes']>=2*2**30,'source_ids':len(r['source_tensor_sha256'])>0,'diagnostic_only':r['kind']=='port80b_t0r12d2_full_stage_diagnostic' and r['status']=='diagnostic_only_not_pass' and r['cuda_initialized'] is False};return {'kind':'d2_independent_verification','pass':False,'valid_diagnostic':all(c.values()),'checks':c}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preflight','verify'),required=True);a=p.parse_args();r=preflight() if a.phase=='preflight' else verify();print(json.dumps(r,indent=2));return 0 if r.get('pass') else (3 if r.get('valid_diagnostic') else 2)
if __name__=='__main__':raise SystemExit(main())
