#!/usr/bin/env python3
"""Standalone S0-R5 CPU validation; no model forward, GPU, or persistent weights."""
from __future__ import annotations
import argparse,gc,hashlib,json,math,os,shutil,sys,traceback,uuid
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,psutil,torch,torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r5_selected_route_validation';RAW=D/'s0r5_raw.safetensors';RES=D/'s0r5_result.json';COM=D/'s0r5_commit.json';FAIL=D/'s0r5_failure.json';BAD=D/'failed_attempts';D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors';D2R=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_result.json';AUD=R/'PORT80B_T0R12D2R3_INDEPENDENT_ARTIFACT_AUDIT_2026-08-13.json';R4FAIL=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r4_selected_route_validation/s0r4_failure.json.dabd00e1014f4395872e03a86de48392.inprogress';R4FAIL_SHA='96b8345f967446b8abc76c3b7d180543a566f41608db15c92eaee658da148647';PR=R/'PORT80B_T0Q5S0R5_SELECTED_ROUTE_VALIDATION_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5s0r5_runner_lock.json';VL=R/'port80b_t0q5s0r5_verifier_lock.json';VER=S/'verify_port80b_t0q5s0r5_selected_route_validation.py';DEP=R/'port80b_t0r4_dependency_execution_lock.json';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';SHARD_SHA='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a';ACK='T0Q5S0R5_VALIDATION_ONLY_AFTER_AUDIT';NAMES=('gate','up','down');SHAPES=((512,2048),(512,2048),(2048,512));MAX=512*2**20
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
AUDIT_TOKEN='S0R5_IMPLEMENTATION_AUDIT_GO'
def tb(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def key(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{NAMES[j]}_proj.weight'
def lockcheck():
 l=json.loads(LOCK.read_text());a={'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'dependency_lock_sha256':sha(DEP),'d2_raw_sha256':sha(D2),'d2_result_sha256':sha(D2R),'d2_audit_sha256':sha(AUD),'r4_failure_sha256':sha(R4FAIL)};auth=l.get('execution_open') is True and l.get('s0_validation_authorization') is True and l.get('implementation_audit_token')==AUDIT_TOKEN and l.get('dependency_use')=='runtime fields/hash only; dependency outputs_opened false is not S0 authorization' and l.get('r4_failure_sha256')==R4FAIL_SHA;return {'pass':all(l.get(k)==v for k,v in a.items()) and auth,'bindings':a}
def fsync(p):
 with Path(p).open('r+b' if os.name=='nt' else 'rb') as h:os.fsync(h.fileno())
def fsyncdir(p):
 if os.name!='nt':
  fd=os.open(str(p),os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
def committed():
 if not (COM.exists() and RAW.exists() and RES.exists()):return False
 try:m=json.loads(COM.read_text());return m['files'][RAW.name]=={'bytes':RAW.stat().st_size,'sha256':sha(RAW)} and m['files'][RES.name]=={'bytes':RES.stat().st_size,'sha256':sha(RES)}
 except Exception:return False
def recover():
 if not D.exists():return {}
 if committed():raise FileExistsError('valid committed result exists')
 BAD.mkdir(exist_ok=True);moved={};nonce=uuid.uuid4().hex
 for p in list(D.glob('*.inprogress'))+[x for x in (RAW,RES,COM,FAIL) if x.exists()]:q=BAD/f'{nonce}_{p.name}';os.rename(p,q);moved[p.name]=str(q)
 fsyncdir(D);return moved
def sample(rows,stage):
 mi=psutil.Process().memory_info();r={'stage':stage,'rss':mi.rss,'peak':getattr(mi,'peak_wset',mi.rss),'available':psutil.virtual_memory().available};rows.append(r)
 if r['available']<2*2**30 or r['peak']>12*2**30:raise RuntimeError('resource gate')
def quant(v):
 r,c=v.shape;w=v.float().reshape(r,c//128,128);mx=w.abs().amax(-1,keepdim=True);s=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);field=(q.to(torch.int16)+15).numpy()
 if field.min()<0 or field.max()>30 or (field==31).any():raise ValueError('field31')
 f=field.astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales,q,int((mx==0).sum())
def decode(codes,scales,r,c):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if (f==31).any() or f.max()>30:raise ValueError('field31')
 q=torch.from_numpy((f.astype(np.int16)-15).reshape(r,c//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(r,c//128,1);return(q*s).reshape(r,c).to(torch.bfloat16)
def graph(x,ids,w,get):
 out=torch.zeros_like(x);mask=F.one_hot(ids,num_classes=512).permute(2,1,0)
 for ei in torch.greater(mask.sum((-1,-2)),0).nonzero():ei=ei[0];pos,tok=torch.where(mask[ei]);gate,up=F.linear(x[tok],torch.cat((get(int(ei),0),get(int(ei),1)),0)).chunk(2,-1);down=F.linear(F.silu(gate)*up,get(int(ei),2))*w[tok,pos,None];out.index_add_(0,tok,down.to(out.dtype))
 shared=F.linear(F.silu(F.linear(x,get(512,0)))*F.linear(x,get(512,1)),get(512,2));return out,shared
def metric(a,z):
 ss=ee=dot=cn=0.;ma=0.
 for x,y in zip(a.reshape(-1).float().double().tolist(),z.reshape(-1).float().double().tolist()):d=y-x;ss+=x*x;ee+=d*d;dot+=x*y;cn+=y*y;ma=max(ma,abs(d))
 rn=math.sqrt(ss);en=math.sqrt(ee);zn=math.sqrt(cn);ua=a.contiguous().view(torch.uint16).to(torch.int32);uz=z.contiguous().view(torch.uint16).to(torch.int32);oa=torch.where((ua&32768)!=0,32768-(ua&32767),32768+ua);oz=torch.where((uz&32768)!=0,32768-(uz&32767),32768+uz);return {'max_abs':ma,'rel_l2':0. if rn==0 and en==0 else(math.inf if rn==0 else en/rn),'cosine':1. if rn==0 and zn==0 else(0. if rn==0 or zn==0 else dot/(rn*zn)),'different_words':int((ua!=uz).sum()),'max_bf16_ulp':int((oa-oz).abs().max())}
def checker(requested,presented,expected,actual):
 errors=[]
 for k in ('expert','projection','shape'):
  if requested[k]!=presented[k]:errors.append(k)
 if expected!=actual:errors.append('codes_scales_digest')
 if not errors:raise RuntimeError('negative control was not rejected')
 return errors
def run():
 if not lockcheck()['pass']:raise RuntimeError('lock')
 moved=recover()
 if D.exists() and any(p!=BAD for p in D.iterdir()):raise FileExistsError('nonclean output')
 if psutil.virtual_memory().available<16*2**30 or torch.cuda.is_initialized():raise RuntimeError('start/GPU')
 D.mkdir(parents=True,exist_ok=True);proc=psutil.Process();aff=json.loads(DEP.read_text())['runtime']['process_affinity'];proc.cpu_affinity(aff);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False);resources=[];sample(resources,'start');raw={};evidence=[];controls=[];metrics={};cache={};wiremeta={}
 with safe_open(D2,framework='pt',device='cpu') as d2,safe_open(SHARD,framework='pt',device='cpu') as src,torch.inference_mode():
  routes=[d2.get_tensor(f'p{p}_whole_official_router_ids') for p in range(4)];union=sorted(set(torch.cat(routes).reshape(-1).tolist()))
  if len(union)!=252:raise RuntimeError('union')
  for e in union+[512]:
   for j in range(3):
    v=src.get_tensor(key(e,j));codes,scales,q,zero=quant(v);dec=decode(codes,scales,*SHAPES[j]);err=metric(v,dec);ds=hashlib.sha256(codes+scales).hexdigest();cache[e,j]=dec;wiremeta[e,j]=(codes,scales,q,ds);evidence.append({'ordinal':len(evidence),'expert':e,'projection':j,'source_key':key(e,j),'shape':list(v.shape),'source_sha256':hashlib.sha256(tb(v)).hexdigest(),'codes_sha256':hashlib.sha256(codes).hexdigest(),'scales_sha256':hashlib.sha256(scales).hexdigest(),'codes_scales_sha256':ds,'decoded_sha256':hashlib.sha256(tb(dec)).hexdigest(),'group_count':v.numel()//128,'zero_group_count':zero,'q_min':int(q.min()),'q_max':int(q.max()),'field31_absent':True,'weight_max_abs':err['max_abs'],'weight_rel_l2':err['rel_l2']})
   if e%32==0:sample(resources,f'quant{e}')
  source=lambda e,j:src.get_tensor(key(e,j));q5=lambda e,j:cache[e,j]
  for p in range(4):
   x=d2.get_tensor(f'p{p}_whole_post_norm').reshape(16,2048);ids=routes[p];w=d2.get_tensor(f'p{p}_whole_official_router_weights');gate=torch.sigmoid(d2.get_tensor(f'p{p}_whole_shared_gate'));ref_r=d2.get_tensor(f'p{p}_whole_experts');ref_s=d2.get_tensor(f'p{p}_whole_shared');sr,ss=graph(x,ids,w,source);qr,qs=graph(x,ids,w,q5)
   if not torch.equal(sr,ref_r) or not torch.equal(ss,ref_s):raise RuntimeError(f'source graph {p}')
   for k,a,z in (('routed',sr,qr),('shared_raw',ss,qs),('shared_gated',gate*ss,gate*qs)):raw[f'p{p}_source_{k}']=a.clone();raw[f'p{p}_q5_{k}']=z.clone();metrics.setdefault(str(p),{})[k]=[metric(a[n:n+1],z[n:n+1]) for n in range(8,16)]
   for n in (8,15):
    original=int(ids[n,0]);replacement=next(e for e in union if e not in set(ids[n].tolist()));requested={'expert':original,'projection':0,'shape':list(SHAPES[0])};presented={'expert':replacement,'projection':0,'shape':list(SHAPES[0])};errors=checker(requested,presented,wiremeta[original,0][3],wiremeta[replacement,0][3]);changed=ids.clone();changed[n,0]=replacement;ur,_=graph(x,changed,w,q5);rk=f'p{p}_n{n}_wrong_routed';raw[rk]=ur[n:n+1];controls.append({'prompt':p,'position':n,'control':'wrong_expert_isolated_route','requested':requested,'presented':presented,'rejection_errors':errors,'raw_key':rk,'baseline_key':f'p{p}_q5_routed'})
    presented={'expert':original,'projection':1,'shape':list(SHAPES[1])};errors=checker(requested,presented,wiremeta[original,0][3],wiremeta[original,1][3]);swap=lambda e,j:q5(e,1) if e==original and j==0 else q5(e,j);ur,_=graph(x,ids,w,swap);rk=f'p{p}_n{n}_swap_routed';raw[rk]=ur[n:n+1];controls.append({'prompt':p,'position':n,'control':'projection_swap_graph_wide','expert':original,'requested':requested,'presented':presented,'rejection_errors':errors,'raw_key':rk,'baseline_key':f'p{p}_q5_routed'})
    if p==0:
     codes,scales,q,expected=wiremeta[512,2];act=F.silu(F.linear(x,q5(512,0)))*F.linear(x,q5(512,1));qq=q.reshape(2048,512);chosen=next(((a,c) for a in range(2048) for c in range(512) if qq[a,c]!=0 and act[n,c]!=0),None)
     if chosen is None:raise RuntimeError('mutation selector')
     a,cc=chosen;new=int(qq[a,cc]-(1 if qq[a,cc]>0 else -1));mutcodes=bytearray(codes);linear=a*512+cc;block=linear//8;slot=linear%8;word=int.from_bytes(mutcodes[block*5:block*5+5],'little');word=(word&~(31<<(5*slot)))|((new+15)<<(5*slot));mutcodes[block*5:block*5+5]=word.to_bytes(5,'little');actual=hashlib.sha256(bytes(mutcodes)+scales).hexdigest();mut=decode(bytes(mutcodes),scales,*SHAPES[2]);errors=checker({'expert':512,'projection':2,'shape':list(SHAPES[2])},{'expert':512,'projection':2,'shape':list(SHAPES[2])},expected,actual);mg=lambda e,j:mut if e==512 and j==2 else q5(e,j);_,ms=graph(x,ids,w,mg);kr=f'p0_n{n}_mutation_shared_raw';kg=f'p0_n{n}_mutation_shared_gated';raw[kr]=ms[n:n+1];raw[kg]=(gate*ms)[n:n+1];controls.append({'prompt':0,'position':n,'control':'shared_down_code_mutation_graph_wide','matrix_row':a,'matrix_column':cc,'source_q':int(qq[a,cc]),'mutated_q':new,'expected_digest':expected,'presented_digest':actual,'rejection_errors':errors,'raw_key':kr,'gated_raw_key':kg,'baseline_key':'p0_q5_shared_raw','gated_baseline_key':'p0_q5_shared_gated'})
  sample(resources,'computed');cache.clear();wiremeta.clear();gc.collect();sample(resources,'cleanup');manifest={k:{'dtype':str(v.dtype),'shape':list(v.shape),'bytes':v.numel()*v.element_size(),'sha256':hashlib.sha256(tb(v)).hexdigest()} for k,v in sorted(raw.items())}
  if not all(torch.isfinite(v.float()).all() for v in raw.values()):raise RuntimeError('finite')
  finite_metrics=all(math.isfinite(float(v)) for p in metrics.values() for a in p.values() for q in a for v in q.values())
  if not finite_metrics:raise RuntimeError('non-finite metric scalar')
  status='selected_route_validation_positive' if all(q['rel_l2']<=.08 for p in metrics.values() for a in p.values() for q in a) else 'selected_route_q5_quality_negative';runtime={'affinity':proc.cpu_affinity(),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),'mkldnn':torch.backends.mkldnn.enabled,'matmul_precision':torch.get_float32_matmul_precision(),'autocast_cpu':torch.is_autocast_enabled('cpu'),'inference_mode':torch.is_inference_mode_enabled(),'cuda_initialized':torch.cuda.is_initialized()};nonce=uuid.uuid4().hex;rt=D/f'{RAW.name}.{nonce}.inprogress';jt=D/f'{RES.name}.{nonce}.inprogress';ct=D/f'{COM.name}.{nonce}.inprogress';save_file({k:v.clone() for k,v in raw.items()},rt);fsync(rt);sample(resources,'post_serialization');result={'kind':'port80b_t0q5s0r5_validation','status':status,'claim_boundary':'validation only; no heldout/pass/complete/layer/model/GPU/performance','runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'runner_lock_sha256':sha(LOCK),'runner_lock_content':json.loads(LOCK.read_text()),'prereg_sha256':sha(PR),'dependency_lock_sha256':sha(DEP),'d2_raw_sha256':sha(D2),'d2_result_sha256':sha(D2R),'d2_audit_sha256':sha(AUD),'shard_expected_sha256':SHARD_SHA,'shard_sha256':sha(SHARD),'selected_union':union,'matrix_evidence':evidence,'raw_manifest':manifest,'raw_sha256':sha(rt),'metrics':metrics,'controls':controls,'runtime':runtime,'resources':resources,'recovered':moved};jt.write_bytes(canon(result)+b'\n');fsync(jt);commit={'kind':'t0q5s0r5_commit','files':{RAW.name:{'bytes':rt.stat().st_size,'sha256':sha(rt)},RES.name:{'bytes':jt.stat().st_size,'sha256':sha(jt)}}};ct.write_bytes(canon(commit)+b'\n');fsync(ct)
  if sum(x.stat().st_size for x in (rt,jt,ct))>MAX:raise RuntimeError('output size')
  for f in (RAW,RES,COM):
   if f.exists():raise FileExistsError(f)
  os.rename(rt,RAW);os.rename(jt,RES);fsyncdir(D);os.rename(ct,COM);fsyncdir(D);return result
def atomic_failure(e):
 gc.collect();dispositions=recover() if D.exists() and not committed() else {};D.mkdir(parents=True,exist_ok=True);nonce=uuid.uuid4().hex;t=D/f'{FAIL.name}.{nonce}.inprogress';payload={'kind':'t0q5s0r5_failure','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'runner_sha256':sha(__file__),'runner_lock_sha256':sha(LOCK),'d2_result_sha256':sha(D2R),'d2_audit_sha256':sha(AUD),'shard_expected_sha256':SHARD_SHA,'resources':{'rss':psutil.Process().memory_info().rss,'peak':getattr(psutil.Process().memory_info(),'peak_wset',0),'available':psutil.virtual_memory().available},'dispositions':dispositions,'cuda_initialized':torch.cuda.is_initialized()};t.write_bytes(canon(payload)+b'\n');fsync(t)
 if t.stat().st_size>MAX:raise RuntimeError('failure size')
 if FAIL.exists():raise FileExistsError(FAIL)
 os.rename(t,FAIL);fsyncdir(D)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lockcheck','run'),required=True);p.add_argument('--ack');a=p.parse_args()
 if a.phase=='lockcheck':print(json.dumps({'kind':'s0r5_lockcheck',**lockcheck(),'physical_actions':False}));return 0
 if a.ack!=ACK:raise SystemExit('ack')
 if committed():print('already_complete');return 0
 try:o=run();print(o['status']);return 0
 except FileExistsError as e:
  if committed():print('already_complete');return 0
  atomic_failure(e);raise
 except Exception as e:atomic_failure(e);raise
if __name__=='__main__':raise SystemExit(main())
