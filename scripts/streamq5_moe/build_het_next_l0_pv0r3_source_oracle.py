#!/usr/bin/env python3
"""PV0-R3 CPU builder. Physical payload access is lock-gated; no device import."""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct, sys, time, uuid
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import numpy as np
import psutil, torch, torch.nn.functional as F
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/streamq5_moe'; RUN=ROOT/'reports/runs/streamq5_moe/het_next_l0_pv0r3_real_weight_process_validation'
D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors'
SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors'
MAN=R/'het_next_l0_pv0r2_selected_source_manifest.json'; LOCK=R/'het_next_l0_pv0r3_runner_lock.json'
ORAW=RUN/'pv0r3_cpu_oracle.safetensors'; ORES=RUN/'pv0r3_cpu_builder.json'; IPKG=RUN/'pv0r3_intel_package.npz'; NPKG=RUN/'pv0r3_nvidia_package.npz'
ACK='PV0R3_BUILDER_AFTER_SOURCE_AUDIT_AND_STATIC_PREFLIGHT'; EXPERTS=(8,12,50,168,199,237,239,245,374,474); INTEL=(50,199,237,474); NVIDIA=(8,12,168,239,245,374); NAMES=('gate','up','down')
D2_FILE_BYTES=171696126; D2_HEAD_BYTES=170664; D2_HEAD_SHA='8eed6e625a1cac3e0cf71e621d95fa901d7f2ff517e7d307c24435b1baa3c2f4'
D2_ALLOW={
'ids':('p0_whole_official_router_ids',216552,217832,'I64',(16,10),'c183be31d947f3a74865eb58f874a0ffd2289adbe455d4689d38239c5a6be2ca'),
'experts':('p0_whole_experts',154798500,154864036,'BF16',(16,2048),'a74a8a9ef47df5a43ff6ca3ecd28a14650c6275586b21ef6c0fc9f1c3559477c'),
'weights':('p0_whole_official_router_weights',155077028,155077348,'BF16',(16,10),'d048f9eddc9f3e358d59383557da8f3fc3b91ab84baddb6b412c82164b2e3be2'),
'x':('p0_whole_post_norm',155077348,155142884,'BF16',(1,16,2048),'d82286fac9616cdf8b03b8eddb8347acd3679afb639c8db696daf3f643084853'),
'shared':('p0_whole_shared',155143556,155209092,'BF16',(16,2048),'3e1f0052460430ca03c19f7a312a80c68034d86b387d3981ae0cce3224e67125'),
'shared_gate':('p0_whole_shared_gate',155209092,155209124,'BF16',(16,1),'3630e2b1cb0ad297f0efd2f029140f5befd810c3520c4dc7eeb0ce746ed49fc0')}
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def atomic(path,data):
 path=Path(path); tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.inprogress')
 if path.exists(): raise FileExistsError(path)
 tmp.write_bytes(data); 
 with tmp.open('r+b') as f: os.fsync(f.fileno())
 os.rename(tmp,path)
def tensor_bytes(t): return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def read_d2():
 if D2.stat().st_size!=D2_FILE_BYTES: raise RuntimeError('D2 size')
 with D2.open('rb') as f: head=f.read(D2_HEAD_BYTES)
 if sha_bytes(head)!=D2_HEAD_SHA: raise RuntimeError('D2 header')
 out={}; ledger=[]
 with D2.open('rb') as f:
  for alias,(key,a,b,dtype,shape,digest) in D2_ALLOW.items():
   ledger.append({'key':key,'absolute':[a,b],'started_ns':time.perf_counter_ns()}); f.seek(a); raw=f.read(b-a)
   if len(raw)!=b-a or sha_bytes(raw)!=digest: raise RuntimeError('D2 payload '+alias)
   if dtype=='I64': t=torch.from_numpy(np.frombuffer(raw,'<i8').copy()).reshape(shape)
   else: t=torch.from_numpy(np.frombuffer(raw,'<u2').copy()).view(torch.bfloat16).reshape(shape)
   out[alias]=t; ledger[-1].update(completed=True,sha256=digest)
 return out,ledger
def read_source(row):
 a,b=row['absolute_offsets'];
 with SHARD.open('rb') as f: f.seek(a); raw=f.read(b-a)
 if len(raw)!=row['source_bytes'] or sha_bytes(raw)!=row['source_sha256']: raise RuntimeError('source '+row['source_key'])
 return torch.from_numpy(np.frombuffer(raw,'<u2').copy()).view(torch.bfloat16).reshape(row['shape'])
def quant(v):
 w=v.float().reshape(v.shape[0],v.shape[1]//128,128); mx=w.abs().amax(-1,keepdim=True); scale=torch.where(mx==0,torch.ones_like(mx),mx/15.0)
 q=torch.where(mx==0,torch.zeros_like(w),torch.round(w/scale).clamp(-15,15)).to(torch.int8); fields=(q.to(torch.int16)+15).numpy().astype(np.uint64).reshape(-1,8)
 if fields.max()>30: raise RuntimeError('field31')
 word=np.bitwise_or.reduce(fields << (np.arange(8,dtype=np.uint64)*5),axis=1); codes=np.stack([(word>>(8*i))&255 for i in range(5)],1).astype(np.uint8).reshape(-1)
 scales=scale.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).view(np.uint8).reshape(-1)
 return codes,scales
def decode(c,s,shape):
 p=c.reshape(-1,5).astype(np.uint64); w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32; fields=np.stack([(w>>(5*i))&31 for i in range(8)],1)
 if (fields==31).any(): raise RuntimeError('field31 decode')
 q=torch.from_numpy((fields.astype(np.int16)-15).reshape(shape[0],shape[1]//128,128)).float(); ss=torch.from_numpy(s.view('<u2').copy()).view(torch.bfloat16).float().reshape(shape[0],shape[1]//128,1)
 return (q*ss).reshape(shape).to(torch.bfloat16)
def source_graph(x,ids,weights,get):
 out=torch.zeros((16,2048),dtype=torch.bfloat16); stages={}; mask=F.one_hot(ids,num_classes=512).permute(2,1,0)
 for e in EXPERTS:
  pos,tok=torch.where(mask[e]); gu=F.linear(x[tok],torch.cat((get(e,0),get(e,1)),0)); gate,up=gu.chunk(2,-1); silu=F.silu(gate,inplace=False); act=silu*up; down=F.linear(act,get(e,2)); weighted=(down*weights[tok,pos,None]).to(torch.bfloat16); out.index_add_(0,tok,weighted)
  for n,t in [('gate',gate),('up',up),('silu',silu),('activation',act),('down',down),('weighted',weighted)]: stages[f'e{e}_{n}']=t.contiguous()
  stages[f'e{e}_token_indices']=tok.to(torch.int64); stages[f'e{e}_topk_positions']=pos.to(torch.int64)
 shared_gu=F.linear(x,torch.cat((get(512,0),get(512,1)),0)); sg,su=shared_gu.chunk(2,-1); ss=F.silu(sg,inplace=False); sa=ss*su; raw=F.linear(sa,get(512,2)); return out,raw,stages,{'gate':sg,'up':su,'silu':ss,'activation':sa}
def ervg_linear(x,w):
 # independent fixed width-8/virtual-32 reduction emulator; BF16 weights/inputs.
 xf=x.float(); wf=w.float(); batch=xf.shape[0]; rows=wf.shape[0]; out=torch.empty((batch,rows),dtype=torch.bfloat16)
 for r in range(rows):
  prod=xf*wf[r]
  lanes=[]
  for lane in range(8):
   partial=[prod[:,lane+8*v::256].sum(-1) if lane+8*v<prod.shape[1] else torch.zeros(batch) for v in range(32)]
   stride=16
   while stride: partial[:stride]=[partial[i]+partial[i+stride] for i in range(stride)]; stride//=2
   lanes.append(partial[0])
  stride=4
  while stride: lanes[:stride]=[lanes[i]+lanes[i+stride] for i in range(stride)]; stride//=2
  out[:,r]=lanes[0].to(torch.bfloat16)
 return out
def q5_graph(x,ids,weights,get):
 out=torch.zeros((16,2048),dtype=torch.bfloat16); stages={}; mask=F.one_hot(ids,num_classes=512).permute(2,1,0); acc=[]
 for e in EXPERTS:
  pos,tok=torch.where(mask[e]); gate=ervg_linear(x[tok],get(e,0)); up=ervg_linear(x[tok],get(e,1)); silu=F.silu(gate); act=(silu*up).to(torch.bfloat16); down=ervg_linear(act,get(e,2)); weighted=(down*weights[tok,pos,None]).to(torch.bfloat16); out.index_add_(0,tok,weighted); acc.append(out[15].clone())
  for n,t in [('gate',gate),('up',up),('silu',silu),('activation',act),('down',down),('weighted',weighted)]: stages[f'e{e}_{n}']=t.contiguous()
 sg=ervg_linear(x,get(512,0)); su=ervg_linear(x,get(512,1)); ss=F.silu(sg); sa=(ss*su).to(torch.bfloat16); raw=ervg_linear(sa,get(512,2)); return out,raw,stages,{'gate':sg,'up':su,'silu':ss,'activation':sa},acc
def metric(a,z):
 a=a.float().double().reshape(-1); z=z.float().double().reshape(-1); err=torch.linalg.vector_norm(z-a); den=torch.linalg.vector_norm(a); return {'max_abs':float((z-a).abs().max()),'rel_l2':float(0 if den==0 and err==0 else math.inf if den==0 else err/den),'different_words':int((a.to(torch.bfloat16).view(torch.uint16)!=z.to(torch.bfloat16).view(torch.uint16)).sum())}
def save_npz(path,arrays):
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.inprogress');
 if path.exists(): raise FileExistsError(path)
 with tmp.open('xb') as f: np.savez(f,**arrays); f.flush(); os.fsync(f.fileno())
 os.rename(tmp,path)
def build():
 lock=json.loads(LOCK.read_text());
 if not(lock.get('execution_open') and lock.get('audit_token')=='PV0R3_SOURCE_AUDIT_GO'): raise RuntimeError('closed lock')
 if psutil.virtual_memory().available<2*2**30: raise RuntimeError('RAM')
 if sha_file(SHARD)!='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a': raise RuntimeError('shard')
 d2,reads=read_d2(); manifest=json.loads(MAN.read_text()); rows={(r['expert'],NAMES.index(r['projection'])):r for r in manifest['records']}; source={}; decoded={}; wire={}; evidence=[]
 for k in sorted(rows):
  v=read_source(rows[k]); c,s=quant(v); d=decode(c,s,tuple(rows[k]['shape'])); row=rows[k]
  observed=(sha_bytes(c.tobytes()),sha_bytes(s.tobytes()),sha_bytes(c.tobytes()+s.tobytes()),sha_bytes(tensor_bytes(d)))
  expected=(row['codes_sha256'],row['scales_sha256'],row['codes_scales_sha256'],row['decoded_sha256'])
  if observed!=expected: raise RuntimeError('codec '+row['source_key'])
  source[k]=v; decoded[k]=d; wire[k]=(c,s); evidence.append({'key':row['source_key'],'digests':observed})
 x=d2['x'].reshape(16,2048); ids=d2['ids']; weights=d2['weights']; src=lambda e,j:source[e,j]; q5=lambda e,j:decoded[e,j]
 sr,ss,sstage,ssh=source_graph(x,ids,weights,src); qr,qs,qstage,qsh,acc=q5_graph(x,ids,weights,q5)
 if not torch.equal(sr[15],d2['experts'][15]) or not torch.equal(ss,d2['shared']): raise RuntimeError('source graph')
 outer=torch.sigmoid(d2['shared_gate']); ssg=(outer*ss).to(torch.bfloat16); qsg=(outer*qs).to(torch.bfloat16)
 metrics={k:metric(a,b) for k,a,b in [('routed',sr[15],qr[15]),('shared_raw',ss[15],qs[15]),('shared_gated',ssg[15],qsg[15])]}
 if any(v['rel_l2']>.08 for v in metrics.values()): raise RuntimeError('quality')
 raw={'source_routed_token15':sr[15],'source_shared':ss,'source_shared_gated':ssg,'cpu_q5_routed_token15':qr[15],'cpu_q5_shared':qs,'cpu_q5_shared_gated':qsg,'shared_gate_linear':d2['shared_gate'],'shared_sigmoid':outer}
 raw.update({'source_'+k:v for k,v in sstage.items()}); raw.update({'cpu_q5_'+k:v for k,v in qstage.items()}); raw.update({'source_shared_'+k:v for k,v in ssh.items()});raw.update({'cpu_q5_shared_'+k:v for k,v in qsh.items()});raw.update({f'cpu_q5_accumulator_{i}':v for i,v in enumerate(acc)})
 tmp=ORAW.with_name(ORAW.name+'.'+uuid.uuid4().hex+'.inprogress'); save_file({k:v.contiguous() for k,v in raw.items()},tmp); os.rename(tmp,ORAW)
 def package(owned,shared):
  a={'x_u16':x.view(torch.uint16).numpy(),'ids_i64':ids.numpy(),'weights_u16':weights.view(torch.uint16).numpy(),'shared_gate_u16':d2['shared_gate'].view(torch.uint16).numpy()}
  for e in tuple(owned)+((512,) if shared else ()):
   for j,n in enumerate(NAMES): a[f'e{e}_{n}_codes']=wire[e,j][0];a[f'e{e}_{n}_scales']=wire[e,j][1]
  return a
 save_npz(IPKG,package(INTEL,False));save_npz(NPKG,package(NVIDIA,True))
 result={'kind':'het_next_l0_pv0r3_cpu_builder','status':'cpu_builder_positive','manifest_sha256':sha_file(MAN),'d2_reads':reads,'source_evidence':evidence,'metrics':metrics,'raw_sha256':sha_file(ORAW),'intel_package':{'bytes':IPKG.stat().st_size,'sha256':sha_file(IPKG)},'nvidia_package':{'bytes':NPKG.stat().st_size,'sha256':sha_file(NPKG)},'resources':{'rss':psutil.Process().memory_info().rss,'peak':getattr(psutil.Process().memory_info(),'peak_wset',0),'available':psutil.virtual_memory().available}}
 atomic(ORES,canon(result)+b'\n'); print(json.dumps({'status':result['status'],'result':str(ORES)})); return 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--ack',required=True);a=p.parse_args();
 if a.ack!=ACK: raise SystemExit('ack')
 return build()
if __name__=='__main__': raise SystemExit(main())
