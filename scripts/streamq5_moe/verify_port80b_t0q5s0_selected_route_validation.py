#!/usr/bin/env python3
"""Standalone independent verifier for S0; imports neither runner nor codec."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0_selected_route_validation';RAW=D/'s0_raw.safetensors';RES=D/'s0_result.json';D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';RUN=S/'run_port80b_t0q5s0_selected_route_validation.py';SELF=Path(__file__);PR=R/'PORT80B_T0Q5S0_SELECTED_ROUTE_VALIDATION_PREREGISTRATION_2026-08-13.md';VL=R/'port80b_t0q5s0_verifier_lock.json';NAMES=('gate','up','down');SHAPES=((512,2048),(512,2048),(2048,512))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tb(t):return t.contiguous().view(torch.uint8).numpy().tobytes()
def key(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{NAMES[j]}_proj.weight'
def quant(v):
 r,c=v.shape;w=v.float().reshape(r,c//128,128);mx=w.abs().amax(-1,keepdim=True);s=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);f=(q.to(torch.int16)+15).numpy().astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales
def decode(codes,scales,r,c):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1);q=torch.from_numpy((f.astype(np.int16)-15).reshape(r,c//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(r,c//128,1);return(q*s).reshape(r,c).to(torch.bfloat16)
def graph(x,ids,w,get):
 out=torch.zeros_like(x);mask=F.one_hot(ids,num_classes=512).permute(2,1,0)
 for ei in torch.greater(mask.sum((-1,-2)),0).nonzero():ei=ei[0];pos,tok=torch.where(mask[ei]);gate,up=F.linear(x[tok],torch.cat((get(int(ei),0),get(int(ei),1)),0)).chunk(2,-1);down=F.linear(F.silu(gate)*up,get(int(ei),2))*w[tok,pos,None];out.index_add_(0,tok,down.to(out.dtype))
 shared=F.linear(F.silu(F.linear(x,get(512,0)))*F.linear(x,get(512,1)),get(512,2));return out,shared
def metric(a,z):
 ss=ee=dot=cn=0.;ma=0.
 for x,y in zip(a.reshape(-1).float().double().tolist(),z.reshape(-1).float().double().tolist()):d=y-x;ss+=x*x;ee+=d*d;dot+=x*y;cn+=y*y;ma=max(ma,abs(d))
 rn=math.sqrt(ss);en=math.sqrt(ee);zn=math.sqrt(cn);return {'max_abs':ma,'rel_l2':0. if rn==0 and en==0 else(math.inf if rn==0 else en/rn),'cosine':1. if rn==0 and zn==0 else(0. if rn==0 or zn==0 else dot/(rn*zn)),'different_words':int((a.view(torch.uint16)!=z.view(torch.uint16)).sum())}
def verify():
 r=json.loads(RES.read_text());union=[]
 with safe_open(D2,framework='pt',device='cpu') as d2:
  union=sorted(set(torch.cat([d2.get_tensor(f'p{p}_whole_official_router_ids') for p in range(4)]).reshape(-1).tolist()))
 checks={'bindings':r['runner_sha256']==sha(RUN) and r['verifier_sha256']==sha(SELF) and r['verifier_lock_sha256']==sha(VL) and r['prereg_sha256']==sha(PR) and r['d2_raw_sha256']==sha(D2),'union_exact':union==r['selected_union'] and len(union)==252,'raw_hash_size':r['raw_sha256']==sha(RAW) and RAW.stat().st_size==r['raw_bytes'] and RAW.stat().st_size<=512*2**20,'claim_narrow':'validation only' in r['claim_boundary'] and 'layer' not in r['metrics'].get('0',{})};store={};ev={(x['expert'],x['projection']):x for x in r['matrix_evidence']};codec=True
 with safe_open(SHARD,framework='pt',device='cpu') as src:
  for e in union+[512]:
   for j in range(3):
    v=src.get_tensor(key(e,j));codes,scales=quant(v);d=decode(codes,scales,*SHAPES[j]);store[e,j]=d;m=ev[e,j];codec &= m['source_key']==key(e,j) and m['source_sha256']==hashlib.sha256(tb(v)).hexdigest() and m['codes_sha256']==hashlib.sha256(codes).hexdigest() and m['scales_sha256']==hashlib.sha256(scales).hexdigest() and m['decoded_sha256']==hashlib.sha256(tb(d)).hexdigest()
 checks['all_selected_source_codec']=bool(codec and len(ev)==(253*3));replay=True;quality=True;metrics={};control_ok=[]
 with safe_open(D2,framework='pt',device='cpu') as d2,safe_open(RAW,framework='pt',device='cpu') as raw:
  for p in range(4):
   x=d2.get_tensor(f'p{p}_whole_post_norm').reshape(16,2048);ids=d2.get_tensor(f'p{p}_whole_official_router_ids');w=d2.get_tensor(f'p{p}_whole_official_router_weights');gate=torch.sigmoid(d2.get_tensor(f'p{p}_whole_shared_gate'));qr,qs=graph(x,ids,w,lambda e,j:store[e,j]);vals={'routed':qr,'shared_raw':qs,'shared_gated':gate*qs};metrics[str(p)]={}
   for k,z in vals.items():stored=raw.get_tensor(f'p{p}_q5_{k}');replay &= torch.equal(z,stored);a=raw.get_tensor(f'p{p}_source_{k}');metrics[str(p)][k]=[metric(a[n:n+1],z[n:n+1]) for n in range(8,16)];quality &= all(q['rel_l2']<=.08 for q in metrics[str(p)][k])
   for row in [x for x in r['controls'] if x['prompt']==p]:
    n=row['position']
    if row['control']=='wrong_expert':changed=ids.clone();changed[n,0]=row['presented'];ur,_=graph(x,changed,w,lambda e,j:store[e,j]);expected=ur[n:n+1];baseline=qr[n:n+1];safe=row['requested']!=row['presented']
    elif row['control']=='projection_swap':
     def sg(e,j):return store[e,1] if e==row['expert'] and j==0 else store[e,j]
     ur,_=graph(x,ids,w,sg);expected=ur[n:n+1];baseline=qr[n:n+1];safe=row['requested_projection']!=row['presented_projection']
    else:
     codes,scales=quant(src.get_tensor(key(512,2))) if False else (None,None);mut=store[512,2].clone();bits=None
     # Selector is recomputed from decoded q via source requantization.
     with safe_open(SHARD,framework='pt',device='cpu') as sr:codes,scales=quant(sr.get_tensor(key(512,2)))
     pp=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);ww=pp[:,0]|pp[:,1]<<8|pp[:,2]<<16|pp[:,3]<<24|pp[:,4]<<32;ff=np.stack([(ww>>(5*i))&31 for i in range(8)],-1).reshape(2048,512);qq=ff.astype(np.int16)-15;act=F.silu(F.linear(x,store[512,0]))*F.linear(x,store[512,1]);chosen=next(( (a,c) for a in range(2048) for c in range(512) if qq[a,c]!=0 and act[n,c]!=0),None);safe=chosen==(row['matrix_row'],row['matrix_column']);a,c=chosen;sb=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16);mut[a,c]=torch.tensor(float(row['mutated_q'])*float(sb[a*4+c//128]),dtype=torch.bfloat16)
     def mg(e,j):return mut if e==512 and j==2 else store[e,j]
     _,ms=graph(x,ids,w,mg);expected=ms[n:n+1];baseline=qs[n:n+1]
    stored=raw.get_tensor(row['raw_key']);control_ok.append(bool(safe and torch.equal(expected,stored) and (stored!=baseline).any()))
 checks['graph_replay']=bool(replay);checks['metrics_exact']=metrics==r['metrics'];checks['quality']=bool(quality and r['status']==('selected_route_validation_positive' if quality else 'selected_route_q5_quality_negative'));checks['controls_independent']=len(control_ok)==18 and all(control_ok);checks['resources']=not r['cuda_initialized'] and r['resources'][0]['available']>=16*2**30 and all(x['available']>=2*2**30 and x['peak']<=12*2**30 for x in r['resources']);return {'kind':'t0q5s0_independent_verification','pass':all(checks.values()),'checks':checks}
def main():print(json.dumps(verify(),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
