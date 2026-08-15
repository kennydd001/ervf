#!/usr/bin/env python3
"""Independent S0-R1 verifier; reruns source/Q5 graphs directly from shard+D2."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r1_selected_route_validation';RAW=D/'s0r1_raw.safetensors';RES=D/'s0r1_result.json';COM=D/'s0r1_commit.json';D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';RUN=S/'run_port80b_t0q5s0r1_selected_route_validation.py';SELF=Path(__file__);PR=R/'PORT80B_T0Q5S0R1_SELECTED_ROUTE_VALIDATION_PREREGISTRATION_2026-08-13.md';DEP=R/'port80b_t0r4_dependency_execution_lock.json';VL=R/'port80b_t0q5s0r1_verifier_lock.json';NAMES=('gate','up','down');SHAPES=((512,2048),(512,2048),(2048,512))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tb(t):return t.contiguous().view(torch.uint8).numpy().tobytes()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def key(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{NAMES[j]}_proj.weight'
def quant(v):
 r,c=v.shape;w=v.float().reshape(r,c//128,128);mx=w.abs().amax(-1,keepdim=True);s=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);field=(q.to(torch.int16)+15).numpy();f=field.astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales,q,int((mx==0).sum()),bool(field.min()>=0 and field.max()<=30 and not(field==31).any())
def decode(codes,scales,r,c):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1);q=torch.from_numpy((f.astype(np.int16)-15).reshape(r,c//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(r,c//128,1);return(q*s).reshape(r,c).to(torch.bfloat16)
def graph(x,ids,w,get):
 out=torch.zeros_like(x);mask=F.one_hot(ids,num_classes=512).permute(2,1,0)
 for ei in torch.greater(mask.sum((-1,-2)),0).nonzero():ei=ei[0];pos,tok=torch.where(mask[ei]);gate,up=F.linear(x[tok],torch.cat((get(int(ei),0),get(int(ei),1)),0)).chunk(2,-1);down=F.linear(F.silu(gate)*up,get(int(ei),2))*w[tok,pos,None];out.index_add_(0,tok,down.to(out.dtype))
 shared=F.linear(F.silu(F.linear(x,get(512,0)))*F.linear(x,get(512,1)),get(512,2));return out,shared
def metric(a,z):
 ss=ee=dot=cn=0.;ma=0.
 for x,y in zip(a.reshape(-1).float().double().tolist(),z.reshape(-1).float().double().tolist()):d=y-x;ss+=x*x;ee+=d*d;dot+=x*y;cn+=y*y;ma=max(ma,abs(d))
 rn=math.sqrt(ss);en=math.sqrt(ee);zn=math.sqrt(cn);ua=a.contiguous().view(torch.uint16).to(torch.int32);uz=z.contiguous().view(torch.uint16).to(torch.int32);oa=torch.where((ua&32768)!=0,32768-(ua&32767),32768+ua);oz=torch.where((uz&32768)!=0,32768-(uz&32767),32768+uz);return {'max_abs':ma,'rel_l2':0. if rn==0 and en==0 else(math.inf if rn==0 else en/rn),'cosine':1. if rn==0 and zn==0 else(0. if rn==0 or zn==0 else dot/(rn*zn)),'different_words':int((ua!=uz).sum()),'max_bf16_ulp':int((oa-oz).abs().max())}
def expected_schema():
 s={}
 for p in range(4):
  for k in ('routed','shared_raw','shared_gated'):
   s[f'p{p}_source_{k}']=('torch.bfloat16',[16,2048]);s[f'p{p}_q5_{k}']=('torch.bfloat16',[16,2048])
  for n in (8,15):s[f'p{p}_n{n}_wrong_routed']=('torch.bfloat16',[1,2048]);s[f'p{p}_n{n}_swap_routed']=('torch.bfloat16',[1,2048])
 for n in (8,15):s[f'p0_n{n}_mutation_shared_raw']=('torch.bfloat16',[1,2048]);s[f'p0_n{n}_mutation_shared_gated']=('torch.bfloat16',[1,2048])
 return s
def verify():
 r=json.loads(RES.read_text());commit=json.loads(COM.read_text());checks={'commit':set(commit['files'])=={RAW.name,RES.name} and commit['files'][RAW.name]=={'bytes':RAW.stat().st_size,'sha256':sha(RAW)} and commit['files'][RES.name]=={'bytes':RES.stat().st_size,'sha256':sha(RES)},'bindings':r['runner_sha256']==sha(RUN) and r['verifier_sha256']==sha(SELF) and r['verifier_lock_sha256']==sha(VL) and r['prereg_sha256']==sha(PR) and r['dependency_lock_sha256']==sha(DEP) and r['d2_raw_sha256']==sha(D2),'claim':r['kind']=='port80b_t0q5s0r1_validation' and r['status'] in ('selected_route_validation_positive','selected_route_q5_quality_negative') and 'no heldout/pass/complete/layer' in r['claim_boundary']};stored={}
 with safe_open(RAW,framework='pt',device='cpu') as f:
  for k in f.keys():stored[k]=f.get_tensor(k)
 schema=expected_schema();manifest={k:{'dtype':str(v.dtype),'shape':list(v.shape),'bytes':v.numel()*v.element_size(),'sha256':hashlib.sha256(tb(v)).hexdigest()} for k,v in sorted(stored.items())};checks['schema_manifest_finite']=set(stored)==set(schema) and all(str(stored[k].dtype)==d and list(stored[k].shape)==q for k,(d,q) in schema.items()) and manifest==r['raw_manifest'] and r['raw_sha256']==sha(RAW) and all(torch.isfinite(v.float()).all() for v in stored.values())
 with safe_open(D2,framework='pt',device='cpu') as d2:union=sorted(set(torch.cat([d2.get_tensor(f'p{p}_whole_official_router_ids') for p in range(4)]).reshape(-1).tolist()))
 checks['union']=union==r['selected_union'] and len(union)==252;cache={};ev=r['matrix_evidence'];codec=len(ev)==759
 with safe_open(SHARD,framework='pt',device='cpu') as src:
  for ordinal,(e,j) in enumerate(( (e,j) for e in union+[512] for j in range(3))):
   v=src.get_tensor(key(e,j));codes,scales,q,zero,valid=quant(v);d=decode(codes,scales,*SHAPES[j]);cache[e,j]=d;m=ev[ordinal];wm=metric(v,d);codec &= m=={'ordinal':ordinal,'expert':e,'projection':j,'source_key':key(e,j),'shape':list(v.shape),'source_sha256':hashlib.sha256(tb(v)).hexdigest(),'codes_sha256':hashlib.sha256(codes).hexdigest(),'scales_sha256':hashlib.sha256(scales).hexdigest(),'codes_scales_sha256':hashlib.sha256(codes+scales).hexdigest(),'decoded_sha256':hashlib.sha256(tb(d)).hexdigest(),'group_count':v.numel()//128,'zero_group_count':zero,'q_min':int(q.min()),'q_max':int(q.max()),'field31_absent':valid,'weight_max_abs':wm['max_abs'],'weight_rel_l2':wm['rel_l2']}
 checks['evidence_759_exact']=bool(codec);metrics={};replay=True;controls=[]
 with safe_open(D2,framework='pt',device='cpu') as d2,safe_open(SHARD,framework='pt',device='cpu') as src,torch.inference_mode():
  source=lambda e,j:src.get_tensor(key(e,j));q5=lambda e,j:cache[e,j]
  for p in range(4):
   x=d2.get_tensor(f'p{p}_whole_post_norm').reshape(16,2048);ids=d2.get_tensor(f'p{p}_whole_official_router_ids');w=d2.get_tensor(f'p{p}_whole_official_router_weights');gate=torch.sigmoid(d2.get_tensor(f'p{p}_whole_shared_gate'));sr,ss=graph(x,ids,w,source);qr,qs=graph(x,ids,w,q5);replay &= torch.equal(sr,d2.get_tensor(f'p{p}_whole_experts')) and torch.equal(ss,d2.get_tensor(f'p{p}_whole_shared'));metrics[str(p)]={}
   for k,a,z in (('routed',sr,qr),('shared_raw',ss,qs),('shared_gated',gate*ss,gate*qs)):replay &= torch.equal(a,stored[f'p{p}_source_{k}']) and torch.equal(z,stored[f'p{p}_q5_{k}']);metrics[str(p)][k]=[metric(a[n:n+1],z[n:n+1]) for n in range(8,16)]
   for row in [x for x in r['controls'] if x['prompt']==p]:
    n=row['position']
    if row['control']=='wrong_expert_isolated_route':changed=ids.clone();changed[n,0]=row['presented']['expert'];ur,_=graph(x,changed,w,q5);expected=ur[n:n+1];baseline=qr[n:n+1];safe='expert' in row['rejection_errors']
    elif row['control']=='projection_swap_graph_wide':
     swap=lambda e,j:q5(e,1) if e==row['expert'] and j==0 else q5(e,j);ur,_=graph(x,ids,w,swap);expected=ur[n:n+1];baseline=qr[n:n+1];safe='projection' in row['rejection_errors'] and 'codes_scales_digest' in row['rejection_errors']
    else:
     v=src.get_tensor(key(512,2));codes,scales,q,_,_=quant(v);act=F.silu(F.linear(x,q5(512,0)))*F.linear(x,q5(512,1));qq=q.reshape(2048,512);chosen=next(((a,c) for a in range(2048) for c in range(512) if qq[a,c]!=0 and act[n,c]!=0),None);a,cc=chosen;new=int(qq[a,cc]-(1 if qq[a,cc]>0 else -1));mut=q5(512,2).clone();sb=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16);mut[a,cc]=torch.tensor(float(new)*float(sb[a*4+cc//128]),dtype=torch.bfloat16);mg=lambda e,j:mut if e==512 and j==2 else q5(e,j);_,ms=graph(x,ids,w,mg);expected=ms[n:n+1];baseline=qs[n:n+1];safe=chosen==(row['matrix_row'],row['matrix_column']) and row['source_q']==int(qq[a,cc]) and row['mutated_q']==new and row['expected_digest']!=row['presented_digest'] and 'codes_scales_digest' in row['rejection_errors'];controls.append(safe and torch.equal((gate*ms)[n:n+1],stored[row['gated_raw_key']]))
    controls.append(bool(safe and torch.equal(expected,stored[row['raw_key']]) and (expected!=baseline).any()))
 checks['source_q5_graph_replay']=bool(replay);checks['metrics_exact']=metrics==r['metrics'];quality=all(q['rel_l2']<=.08 for p in metrics.values() for a in p.values() for q in a);checks['status_quality']=r['status']==('selected_route_validation_positive' if quality else 'selected_route_q5_quality_negative');checks['controls']=len(controls)==20 and all(controls);checks['runtime_resources']=r['runtime']=={'affinity':json.loads(DEP.read_text())['runtime']['process_affinity'],'threads':1,'interop':1,'deterministic':True,'mkldnn':True,'matmul_precision':'highest','autocast_cpu':False,'inference_mode':True,'cuda_initialized':False} and r['resources'][0]['available']>=16*2**30 and {x['stage'] for x in r['resources']}>= {'start','computed','cleanup','final'} and all(x['available']>=2*2**30 and x['peak']<=12*2**30 for x in r['resources']) and sum(p.stat().st_size for p in (RAW,RES,COM))<=512*2**20;out={'kind':'t0q5s0r1_independent_verification','pass':all(checks.values()),'checks':checks};return out
def main():
 o=verify();print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
