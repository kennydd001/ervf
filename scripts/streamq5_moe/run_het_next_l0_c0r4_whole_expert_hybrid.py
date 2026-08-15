#!/usr/bin/env python3
"""C0-R4 execution-closed runner contract. No action without an audited phase lock."""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, struct, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; S=ROOT/'scripts/streamq5_moe'; R=ROOT/'reports/streamq5_moe'
LOCK=R/'het_next_l0_c0r4_runner_lock.json'; VLOCK=R/'het_next_l0_c0r4_verifier_lock.json'
PR=R/'HET_NEXT_L0_C0R3_WHOLE_EXPERT_HYBRID_PREREGISTRATION_2026-08-13.md'; REV=R/'HET_NEXT_L0_C0R4_WORKER_EPOCH_REVISION_2026-08-13.md'
DESIGN=R/'HET_NEXT_L0_C0R3_CAPABILITY_PREFLIGHT_DESIGN_2026-08-13.md'; ADD=R/'HET_NEXT_L0_C0R4_CAPABILITY_PREFLIGHT_ADDENDUM_2026-08-13.md'
VER=S/'verify_het_next_l0_c0r4_whole_expert_hybrid.py'; PREF=S/'preflight_het_next_l0_c0r4_static.py'; KERNEL=S/'het_next_l0_c0r4_kernel_contract.py'
D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors'
SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors'
OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_c0r4_whole_expert_hybrid'
ROUTES=((50,199,237,474,245,374,239,8,168,12),(42,162,267,299,467,307,326,145,297,182),(474,232,382,80,31,450,103,372,286,206),(26,159,28,176,253,84,431,294,386,356))
SHARD_SHA='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a'; SEED=2026081302
TEMPLATES=(('A','B','S','B','A','S','S','A','B','S','B','A'),('A','S','B','B','S','A','A','S','B','B','S','A'),('S','A','B','S','B','A','A','B','S','B','A','S'))
REVERSE=((3,2,1,0),(1,0,3,2),(3,2,1,0)); Q5_SHAPES=((512,2048),(512,2048),(2048,512)); NAMES=('gate','up','down')
ACK='PENDING_C0R4_IMPLEMENTATION_AUDIT'; MAX_WAIT_MS=30000

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def schedule():
 out=[]
 for b in range(30):
  ti=(SEED+b)%3
  for k,a in enumerate(TEMPLATES[ti]):
   g=k//3; out.append({'observation':len(out),'block':b,'template':ti,'group':g,'arm':a,'pair':(b,min(g,REVERSE[ti][g]),max(g,REVERSE[ti][g]))})
 assert len(out)==360 and all(sum(x['arm']==a for x in out)==120 for a in 'ASB')
 return out
def linear_q(xs,q):
 z=sorted(float(v) for v in xs); h=(len(z)-1)*q; lo=math.floor(h); hi=math.ceil(h); return z[lo]+(h-lo)*(z[hi]-z[lo])
def gates(samples):
 q={a:{p:linear_q(samples[a],p) for p in (.5,.95)} for a in 'ASB'}
 return {'p50_ratio':q['B'][.5]/q['A'][.5],'p95_ratio':q['B'][.95]/q['A'][.95],'p50_b_lt_s':q['B'][.5]<q['S'][.5],'p95_b_lt_s':q['B'][.95]<q['S'][.95]}
def sm64(x):
 m=(1<<64)-1;z=(x+0x9E3779B97F4A7C15)&m;z=((z^(z>>30))*0xBF58476D1CE4E5B9)&m;z=((z^(z>>27))*0x94D049BB133111EB)&m;return (z^(z>>31))&m
def thrash_small(buf,row,phase,j,counter):
 L=len(buf)//64; dig=hashlib.sha256(f'HET-NEXT-L0-C0-R2|{row}|{phase}|{j}'.encode()).digest(); start=int.from_bytes(dig[:8],'little')%L;v=0
 for k in range(L):
  o=64*((start+k)%L);old=buf[o];v^=old;buf[o]=(old+(sm64(SEED^counter)&255))&255;counter+=1
 return start,v,counter

class EpochMachine:
 """Pure model used verbatim by static TEMP simulation; physical code must mirror it."""
 def __init__(self):self.last={'intel':0,'nvidia':0};self.ack={'intel':0,'nvidia':0};self.ready=set();self.done=set();self.start=False;self.epoch=0;self.ledger=[]
 def arm(self,active):
  if self.start:raise RuntimeError('start_not_reset')
  for i in active:
   if self.ack[i]!=self.last[i]:raise RuntimeError('stale_ack')
  self.epoch+=1
  for i in active:self.ready.discard(i);self.done.discard(i);self.last[i]=self.epoch
  self.ledger.append(('command',self.epoch,tuple(active)))
  for i in active:self.ready.add(i)
  return self.epoch
 def release(self,active):
  if set(active)-self.ready:raise RuntimeError('not_ready')
  self.start=True;self.ledger.append(('start',self.epoch,tuple(active)))
 def worker_done(self,i):
  if i not in self.ready or not self.start:raise RuntimeError('inactive_done')
  self.ack[i]=self.last[i];self.done.add(i);self.ledger.append(('ack_done',self.epoch,i))
 def collect(self,active):
  if set(active)-self.done:raise RuntimeError('not_done')
  if any(self.ack[i]!=self.last[i] for i in active):raise RuntimeError('ack_mismatch')
  self.start=False;self.ledger.append(('collect_reset',self.epoch,tuple(active)))

def simulate_sync():
 m=EpochMachine()
 for active in (('nvidia',),('intel','nvidia'),('intel',),('nvidia',),('nvidia',)):
  m.arm(active);m.release(active)
  for i in active:m.worker_done(i)
  m.collect(active)
 if m.last['intel']!=3 or m.last['nvidia']!=5:raise AssertionError(m.last)
 neg=[]
 for name,fn in (
  ('stale',lambda:(setattr(m,'ack',{'intel':0,'nvidia':m.ack['nvidia']}),m.arm(('intel',)))),
  ('inactive',lambda:m.worker_done('intel')),
 ):
  try:fn();neg.append(False)
  except RuntimeError:neg.append(True)
 return {'pass':all(neg),'ledger':m.ledger,'negative':neg}

def parse_header(path):
 with Path(path).open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];raw=f.read(n)
 obj=json.loads(raw);base=8+n
 return {k:{**v,'absolute':[base+v['data_offsets'][0],base+v['data_offsets'][1]]} for k,v in obj.items() if k!='__metadata__'}
class SealedReader:
 def __init__(self,path,header,opened_tests=False):self.path=Path(path);self.header=header;self.opened=opened_tests;self.rows=[]
 def read(self,key,row,phase):
  if row not in (0,1,2,3):raise ValueError(row)
  if row>0 and not self.opened:raise PermissionError('test_payload_sealed')
  if key not in self.header:raise KeyError(key)
  a,b=self.header[key]['absolute']; entry={'phase':phase,'row':row,'key':key,'absolute_offset':a,'byte_count':b-a,'completed':False};self.rows.append(entry)
  with self.path.open('rb') as f:f.seek(a);data=f.read(b-a)
  if len(data)!=b-a:raise EOFError(key)
  entry.update(completed=True,observed_sha256=hashlib.sha256(data).hexdigest());return data

def quantize_biased_q5(weight):
 import numpy as np, torch
 r,c=weight.shape;w=weight.float().reshape(r,c//128,128);mx=w.abs().amax(-1,keepdim=True);scale=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/scale).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);field=(q.short()+15).numpy()
 if field.min()<0 or field.max()>30 or (field==31).any():raise ValueError('field31')
 f=field.astype(np.uint64).reshape(-1,8);words=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),axis=-1);codes=np.stack([(words>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=scale.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales
def decode_q5(codes,scales,r,c):
 import numpy as np,torch
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if (f==31).any() or f.max()>30:raise ValueError('field31')
 q=torch.from_numpy((f.astype(np.int16)-15).reshape(r,c//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).view(torch.uint16).view(torch.bfloat16).float().reshape(r,c//128,1);return(q*s).reshape(r,c).to(torch.bfloat16)
def cpu_expert(x,gate,up,down):
 import torch.nn.functional as F
 g=F.linear(x,gate).to(x.dtype);u=F.linear(x,up).to(x.dtype);s=F.silu(g,inplace=False)
 if s.dtype!=x.dtype:raise RuntimeError('silu_dtype')
 a=(s*u).to(x.dtype);d=F.linear(a,down).to(x.dtype);return {'gate':g,'up':u,'silu':s,'activation':a,'down':d}
def merge_official(downs,ids,weights,shared_raw,shared_gate_linear):
 import torch
 out=torch.zeros_like(shared_raw)
 states=[]
 for e in sorted(int(v) for v in ids):
  rank=list(map(int,ids)).index(e);contrib=(downs[e]*weights[rank]).to(torch.bfloat16);out.index_add_(0,torch.tensor([0]),contrib.reshape(1,-1));states.append(out.clone())
 sig=torch.sigmoid(shared_gate_linear)
 if sig.dtype!=torch.bfloat16:raise RuntimeError('sigmoid_dtype')
 gated=(sig*shared_raw).to(torch.bfloat16);return out,(out+gated).to(torch.bfloat16),states,sig,gated

def checker(requested,presented):
 errors=[k for k in ('expert','slot','projection','shape','source_sha256','codes_sha256','scales_sha256') if requested.get(k)!=presented.get(k)]
 if presented.get('field31'):errors.append('field31')
 if not errors:raise RuntimeError('control_not_rejected')
 return errors
def lockcheck():
 l=json.loads(LOCK.read_text());expected={'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'preflight_sha256':sha(PREF),'kernel_sha256':sha(KERNEL),'prereg_sha256':sha(PR),'revision_sha256':sha(REV),'design_sha256':sha(DESIGN),'addendum_sha256':sha(ADD)}
 return {'pass':all(l.get(k)==v for k,v in expected.items()) and l.get('execution_open') is False and l.get('capability_open') is False and l.get('source_build_open') is False,'expected':expected,'lock':l}

def require_phase_authorization(phase,ack):
 l=json.loads(LOCK.read_text()); token=l.get(f'{phase}_audit_token'); opened=l.get(f'{phase}_open')
 if opened is not True or not isinstance(token,str) or token.startswith('PENDING') or ack!=token:raise PermissionError(f'{phase}_closed')

def phase0_contract():
 """No device imports/calls. Physical Phase-0 lives in separately frozen preflight."""
 return {'kind':'c0r4_runner_contract','lockcheck':lockcheck(),'schedule_sha256':hashlib.sha256(canonical(schedule())).hexdigest(),'sync':simulate_sync(),'no_device_calls':True}
def capability_phase():
 require_phase_authorization('capability',ARGS.ack)
 raise NotImplementedError('capability backend remains closed until post-source-audit revision')
def source_build_phase():
 require_phase_authorization('source_build',ARGS.ack)
 raise NotImplementedError('p0-only source builder remains closed until post-source-audit revision')
def validation_phase():
 require_phase_authorization('execution',ARGS.ack)
 raise NotImplementedError('physical worker/backend binding remains closed until post-source-audit revision')
def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('contract','capability','source_build','validation'),required=True);p.add_argument('--ack');ARGS=p.parse_args()
 if ARGS.phase=='contract':print(json.dumps(phase0_contract(),sort_keys=True));return 0
 return {'capability':capability_phase,'source_build':source_build_phase,'validation':validation_phase}[ARGS.phase]()
if __name__=='__main__':raise SystemExit(main())
