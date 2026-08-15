#!/usr/bin/env python3
"""Standalone CPU-only T0Q5-S0 validation runner. No model forward or persistent bank."""
from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,sys,traceback,zlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,psutil,torch,torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0_selected_route_validation';RAW=D/'s0_raw.safetensors';RES=D/'s0_result.json';FAIL=D/'s0_failure.json';D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors';D2R=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_result.json';AUD=R/'PORT80B_T0R12D2R3_INDEPENDENT_ARTIFACT_AUDIT_2026-08-13.json';PR=R/'PORT80B_T0Q5S0_SELECTED_ROUTE_VALIDATION_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5s0_runner_lock.json';VL=R/'port80b_t0q5s0_verifier_lock.json';VER=S/'verify_port80b_t0q5s0_selected_route_validation.py';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';ACK='T0Q5S0_SELECTED_ROUTE_VALIDATION_ONLY';GROUP=128;NAMES=('gate','up','down');SHAPES=((512,2048),(512,2048),(2048,512))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tb(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def key(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{NAMES[j]}_proj.weight'
def lockcheck():
 l=json.loads(LOCK.read_text());a={'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'d2_raw_sha256':sha(D2),'d2_result_sha256':sha(D2R),'d2_audit_sha256':sha(AUD)};return {'pass':all(l.get(k)==v for k,v in a.items()),'bindings':a}
def quant(v):
 r,c=v.shape;w=v.float().reshape(r,c//GROUP,GROUP);mx=w.abs().amax(-1,keepdim=True);scale=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/scale).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);fields=(q.to(torch.int16)+15).numpy()
 if fields.min()<0 or fields.max()>30:raise ValueError('field31')
 f=fields.astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=scale.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales
def decode(codes,scales,r,c):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if (f==31).any():raise ValueError('field31')
 q=torch.from_numpy((f.astype(np.int16)-15).reshape(r,c//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(r,c//128,1);return(q*s).reshape(r,c).to(torch.bfloat16)
def graph(x,ids,w,get):
 final=torch.zeros_like(x);mask=F.one_hot(ids,num_classes=512).permute(2,1,0)
 for ei in torch.greater(mask.sum((-1,-2)),0).nonzero():ei=ei[0];pos,tok=torch.where(mask[ei]);gate,up=F.linear(x[tok],torch.cat((get(int(ei),0),get(int(ei),1)),0)).chunk(2,-1);down=F.linear(F.silu(gate)*up,get(int(ei),2))*w[tok,pos,None];final.index_add_(0,tok,down.to(final.dtype))
 shared=F.linear(F.silu(F.linear(x,get(512,0)))*F.linear(x,get(512,1)),get(512,2));return final,shared
def metric(a,z):
 ss=ee=dot=cn=0.;ma=0.
 for x,y in zip(a.reshape(-1).float().double().tolist(),z.reshape(-1).float().double().tolist()):d=y-x;ss+=x*x;ee+=d*d;dot+=x*y;cn+=y*y;ma=max(ma,abs(d))
 rn=math.sqrt(ss);en=math.sqrt(ee);zn=math.sqrt(cn);return {'max_abs':ma,'rel_l2':0. if rn==0 and en==0 else(math.inf if rn==0 else en/rn),'cosine':1. if rn==0 and zn==0 else(0. if rn==0 or zn==0 else dot/(rn*zn)),'different_words':int((a.view(torch.uint16)!=z.view(torch.uint16)).sum())}
def atom(raw,result):
 D.mkdir(parents=True,exist_ok=False);rt=RAW.with_suffix('.safetensors.inprogress');jt=RES.with_suffix('.json.inprogress');save_file({k:v.clone() for k,v in raw.items()},rt)
 with rt.open('rb') as h:os.fsync(h.fileno())
 result['raw_sha256']=sha(rt);result['raw_bytes']=rt.stat().st_size;jt.write_bytes(canon(result)+b'\n')
 with jt.open('rb') as h:os.fsync(h.fileno())
 os.rename(rt,RAW);os.rename(jt,RES)
def run():
 if not lockcheck()['pass'] or D.exists():raise RuntimeError('lock/output')
 if psutil.virtual_memory().available<16*2**30 or torch.cuda.is_initialized():raise RuntimeError('resources/GPU')
 proc=psutil.Process();samples=[];sample=lambda s:samples.append({'stage':s,'rss':proc.memory_info().rss,'peak':getattr(proc.memory_info(),'peak_wset',proc.memory_info().rss),'available':psutil.virtual_memory().available});sample('start');raw={};evidence=[];metrics={};controls=[]
 with safe_open(D2,framework='pt',device='cpu') as d2,safe_open(SHARD,framework='pt',device='cpu') as src,torch.inference_mode():
  routes=[d2.get_tensor(f'p{p}_whole_official_router_ids') for p in range(4)];union=sorted(set(torch.cat(routes).reshape(-1).tolist()))
  if len(union)!=252:raise RuntimeError('union')
  store={};wires={}
  for e in union+[512]:
   for j in range(3):
    v=src.get_tensor(key(e,j));codes,scales=quant(v);dec=decode(codes,scales,*SHAPES[j]);store[e,j]=dec;wires[e,j]=(codes,scales);evidence.append({'source_key':key(e,j),'expert':e,'projection':j,'source_sha256':hashlib.sha256(tb(v)).hexdigest(),'codes_sha256':hashlib.sha256(codes).hexdigest(),'scales_sha256':hashlib.sha256(scales).hexdigest(),'decoded_sha256':hashlib.sha256(tb(dec)).hexdigest(),'zero_groups':int((torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16)==1).sum()),'field_range':[0,30]})
   if e%32==0:sample(f'quant_{e}')
  source=lambda e,j:src.get_tensor(key(e,j));q5=lambda e,j:store[e,j]
  for p in range(4):
   x=d2.get_tensor(f'p{p}_whole_post_norm').reshape(16,2048);ids=routes[p];w=d2.get_tensor(f'p{p}_whole_official_router_weights');ref_r=d2.get_tensor(f'p{p}_whole_experts');ref_s=d2.get_tensor(f'p{p}_whole_shared');gate=torch.sigmoid(d2.get_tensor(f'p{p}_whole_shared_gate'));sr,ss=graph(x,ids,w,source);qr,qs=graph(x,ids,w,q5)
   if not torch.equal(sr,ref_r) or not torch.equal(ss,ref_s):raise RuntimeError(f'graph control {p}')
   vals={'routed':(sr,qr),'shared_raw':(ss,qs),'shared_gated':(gate*ss,gate*qs)}
   for k,(a,z) in vals.items():raw[f'p{p}_source_{k}']=a.clone();raw[f'p{p}_q5_{k}']=z.clone();metrics.setdefault(str(p),{})[k]=[metric(a[n:n+1],z[n:n+1]) for n in range(8,16)]
   for n in (8,15):
    original=int(ids[n,0]);replacement=next(e for e in union if e not in set(ids[n].tolist()));changed=ids.clone();changed[n,0]=replacement;ur,_=graph(x,changed,w,q5);rk=f'p{p}_n{n}_wrong_unsafe';raw[rk]=ur[n:n+1];controls.append({'prompt':p,'position':n,'control':'wrong_expert','requested':original,'presented':replacement,'metadata_rejected':original!=replacement,'raw_key':rk,'baseline_key':f'p{p}_q5_routed'})
    def swap(e,j):return q5(e,1) if e==original and j==0 else q5(e,j)
    ur,_=graph(x,ids,w,swap);rk=f'p{p}_n{n}_swap_unsafe';raw[rk]=ur[n:n+1];controls.append({'prompt':p,'position':n,'control':'projection_swap','expert':original,'requested_projection':0,'presented_projection':1,'metadata_rejected':True,'raw_key':rk,'baseline_key':f'p{p}_q5_routed'})
    if p==0:
     codes,scales=wires[512,2];pp=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);word=pp[:,0]|pp[:,1]<<8|pp[:,2]<<16|pp[:,3]<<24|pp[:,4]<<32;fields=np.stack([(word>>(5*i))&31 for i in range(8)],-1).reshape(2048,512);qq=fields.astype(np.int16)-15;act=F.silu(F.linear(x,q5(512,0)))*F.linear(x,q5(512,1));chosen=None
     for rr in range(2048):
      for cc in range(512):
       if qq[rr,cc]!=0 and act[n,cc]!=0:chosen=(rr,cc);break
      if chosen:break
     if chosen is None:raise RuntimeError('mutation selector')
     rr,cc=chosen;new=int(qq[rr,cc]-(1 if qq[rr,cc]>0 else -1));mut=q5(512,2).clone();sb=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16);mut[rr,cc]=torch.tensor(float(new)*float(sb[rr*4+cc//128]),dtype=torch.bfloat16)
     def mq(e,j):return mut if e==512 and j==2 else q5(e,j)
     _,ms=graph(x,ids,w,mq);kr=f'p0_n{n}_mutation_unsafe_shared_raw';kg=f'p0_n{n}_mutation_unsafe_shared_gated';raw[kr]=ms[n:n+1];raw[kg]=(gate*ms)[n:n+1];controls.append({'prompt':0,'position':n,'control':'code_mutation','matrix_row':rr,'matrix_column':cc,'source_q':int(qq[rr,cc]),'mutated_q':new,'metadata_rejected':True,'raw_key':kr,'gated_raw_key':kg,'baseline_key':'p0_q5_shared_raw','gated_baseline_key':'p0_q5_shared_gated'})
  sample('computed');quality=all(q['rel_l2']<=.08 for p in metrics.values() for a in p.values() for q in a);result={'kind':'port80b_t0q5s0_selected_route_validation','status':'selected_route_validation_positive' if quality else 'selected_route_q5_quality_negative','claim_boundary':'validation only; no heldout/pass/layer/model/GPU/performance claim','runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'d2_raw_sha256':sha(D2),'shard_sha256':sha(SHARD),'selected_union':union,'matrix_evidence':evidence,'metrics':metrics,'controls':controls,'resources':samples,'cuda_initialized':torch.cuda.is_initialized()};atom(raw,result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lockcheck','run'),required=True);p.add_argument('--ack');a=p.parse_args()
 if a.phase=='lockcheck':print(json.dumps({'kind':'s0_lockcheck',**lockcheck(),'physical_actions':False}));return 0
 if a.ack!=ACK:raise SystemExit('ack')
 try:r=run();print(r['status']);return 3
 except Exception as e:
  D.mkdir(parents=True,exist_ok=True)
  if not FAIL.exists():FAIL.write_bytes(canon({'kind':'t0q5s0_failure','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'runner_sha256':sha(__file__),'cuda_initialized':torch.cuda.is_initialized()})+b'\n')
  raise
if __name__=='__main__':raise SystemExit(main())
