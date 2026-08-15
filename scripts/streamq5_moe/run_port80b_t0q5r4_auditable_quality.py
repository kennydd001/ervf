#!/usr/bin/env python3
"""T0Q5-R4 audited two-phase candidate. CPU only; execution initially lock-blocked."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,struct,sys,traceback,uuid,zlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5r4_auditable_quality';SNAP=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f';SHARD=SNAP/'model-00001-of-00040.safetensors'
R3=S/'run_port80b_t0q5r3_real_layer0_quality.py';C3=S/'port80b_t0q5r3_codec_contract.py';GEN=S/'generate_port80b_t0q5r1_prompts.py';VER=S/'verify_port80b_t0q5r4_auditable_quality.py';PR=R/'PORT80B_T0Q5R4_AUDITABLE_EXECUTION_PREREGISTRATION_2026-08-13.md';SCI=R/'PORT80B_T0Q5R3_REAL_LAYER0_NUMERICAL_QUALITY_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5r4_runner_lock.json';VL=R/'port80b_t0q5r4_verifier_lock.json';PL=R/'port80b_t0q5r4_prompt_lock.json'
REFX=D/'reference_raw.safetensors';REFR=D/'reference_result.json';REFC=D/'reference_commit.json';REFV=D/'reference_verification.json';BANK=D/'layer0_bank.sq5m';MAN=D/'bank_manifest.json';BC=D/'bank_commit.json';QX=D/'q5_raw.safetensors';QR=D/'q5_result.json';QC=D/'q5_commit.json';FAILED=D/'failed_attempts';HF='<4sHHHBBIIH2xIII28s';MB=675840;CB=655360;SB=16384;PAD=4032;SHAPES=((512,2048),(512,2048),(2048,512));NAMES=('gate','up','down');MAX_ART=int(1.10*2**30)
def mod(path,name):q=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(q);sys.modules[name]=m;q.loader.exec_module(m);return m
r3=mod(R3,'q5r3base');b=r3.b;c=r3.c;g=r3.g;torch=b.torch;F=b.F;safe_open=b.safe_open;save_file=b.save_file;DynamicCache=b.DynamicCache;psutil=b.psutil
def sha(p):return b.sha256(Path(p))
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def fsync(p):
 with Path(p).open('rb') as f:os.fsync(f.fileno())
def resources(samples,stage):
 vm=psutil.virtual_memory();mi=psutil.Process().memory_info();row={'stage':stage,'rss':mi.rss,'peak':getattr(mi,'peak_wset',mi.rss),'available':vm.available,'disk_free':shutil.disk_usage(ROOT).free};samples.append(row)
 if row['available']<2*2**30 or row['peak']>12*2**30:raise RuntimeError('resource gate')
 return row
def lockcheck():
 l=json.loads(LOCK.read_text());a={'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'execution_prereg_sha256':sha(PR),'science_prereg_sha256':sha(SCI),'codec_sha256':sha(C3),'generator_sha256':sha(GEN),'r3_source_sha256':sha(R3),'prompt_lock_sha256':sha(PL) if PL.exists() else '__ABSENT_PENDING_IMPLEMENTATION_AUDIT__'};return {'pass':PL.exists() and all(l.get(k)==v for k,v in a.items()),'bindings':a}
def prompts():
 if not PL.exists():raise RuntimeError('prompt lock absent')
 frozen=json.loads(PL.read_text());observed=g.generate();old=json.loads(b.PROMPT_LOCK.read_text());
 if canon(frozen)!=canon(observed) or len(frozen['prompts'])!=4 or any(len(x['token_ids'])!=16 for x in frozen['prompts']):raise RuntimeError('prompt replay/schema')
 for i,x in enumerate(frozen['prompts']):
  for y in old['prompts']+frozen['prompts'][i+1:]:
   if x['utf8_text']==y['utf8_text'] or x['token_ids']==y['token_ids'] or x['token_ids_le_u32_sha256']==y['token_ids_le_u32_sha256']:raise RuntimeError('prompt disjointness')
 return frozen
def guard(phase):
 if not lockcheck()['pass']:raise RuntimeError('execution lock')
 if psutil.virtual_memory().available<16*2**30 or shutil.disk_usage(ROOT).free<4*2**30 or torch.cuda.is_initialized():raise RuntimeError('start resource/GPU')
 targets=(REFX,REFR,REFC,REFV) if phase=='reference' else (BANK,MAN,BC,QX,QR,QC)
 if any(x.exists() for x in targets):raise FileExistsError('target exists')
 if phase=='q5':
  if not REFV.exists():raise RuntimeError('reference verification absent')
  v=json.loads(REFV.read_text());
  if not v.get('pass') or v['reference_raw_sha256']!=sha(REFX) or v['reference_result_sha256']!=sha(REFR) or v['verifier_sha256']!=sha(VER):raise RuntimeError('reference verification binding')
def temp(path,nonce):return path.with_name(path.name+'.'+nonce+'.inprogress')
def commit_bundle(files,marker,kind):
 nonce=uuid.uuid4().hex;tmps={p:temp(p,nonce) for p in files};return nonce,tmps,temp(marker,nonce),{'kind':kind,'nonce':nonce,'files':{}}
def promote(tmps,marker_tmp,marker,journal):
 for final,t in tmps.items():journal['files'][final.name]={'bytes':t.stat().st_size,'sha256':sha(t)}
 marker_tmp.write_bytes(canon(journal)+b'\n');fsync(marker_tmp)
 for final,t in tmps.items():os.replace(t,final)
 os.replace(marker_tmp,marker)
def quarantine(phase):
 if not D.exists():return {}
 FAILED.mkdir(exist_ok=True);out={};nonce=uuid.uuid4().hex
 for p in D.glob('*.inprogress'):
  q=FAILED/f'{phase}_{nonce}_{p.name}';p.replace(q);out[p.name]=str(q)
 for finals,marker in (((REFX,REFR),REFC),((BANK,MAN),BC),((QX,QR),QC)):
  if any(p.exists() for p in finals) and not marker.exists():
   for p in finals:
    if p.exists():q=FAILED/f'{phase}_{nonce}_{p.name}';p.replace(q);out[p.name]=str(q)
 return out
def capture(layer,h,cfg):
 cap={};hooks=[]
 def save(n):
  def hook(_m,_i,o):cap[n]=(o[0] if isinstance(o,tuple) else o).detach().cpu().clone()
  return hook
 def gate(_m,i,o):cap['router_input']=i[0].detach().cpu().clone();cap['router_logits'],cap['router_weights'],cap['router_ids']=[x.detach().cpu().clone() for x in o]
 def residual(_m,i):cap['pre_mlp_residual']=i[0].detach().cpu().clone()
 hooks=[layer.post_attention_layernorm.register_forward_pre_hook(residual),layer.post_attention_layernorm.register_forward_hook(save('mlp_input')),layer.mlp.experts.register_forward_hook(save('routed')),layer.mlp.shared_expert.register_forward_hook(save('shared_raw')),layer.mlp.shared_expert_gate.register_forward_hook(save('shared_gate_linear')),layer.mlp.register_forward_hook(save('complete_mlp')),layer.mlp.gate.register_forward_hook(gate)]
 try:cache=DynamicCache(config=cfg);e=torch.empty(0,dtype=torch.bfloat16);out=layer(h,position_embeddings=(e,e),attention_mask=None,past_key_values=cache);_,ct=b.cache_state(cache,0,16)
 finally:
  for q in hooks:q.remove()
 sl,sw,si=layer.mlp.gate(cap['router_input']);cap.update(router_second_logits=sl.cpu(),router_second_weights=sw.cpu(),router_second_ids=si.cpu());cap['shared_gate']=torch.sigmoid(cap['shared_gate_linear']);cap['shared_gated']=cap['shared_gate']*cap['shared_raw'];cap['layer_output']=out.cpu();cap['cache_conv']=next(v for k,v in ct.items() if k.endswith('cache_conv'));cap['cache_recurrent']=next(v for k,v in ct.items() if k.endswith('cache_recurrent'))
 return cap
def reference():
 guard('reference');p=prompts();D.mkdir(parents=True,exist_ok=True);samples=[];resources(samples,'start');cfg=b.Qwen3NextConfig.from_pretrained(SNAP,local_files_only=True,trust_remote_code=False);layer,embed,src=b.load_official(cfg,{});ids=torch.tensor([x['token_ids'] for x in p]);hidden=F.embedding(ids,embed).to(torch.bfloat16);raw={'token_ids':ids,'embedding':hidden.cpu()}
 with torch.inference_mode():
  for n in range(4):
   for k,v in capture(layer,hidden[n:n+1],cfg).items():raw[f'p{n}_{k}']=v.clone()
 resources(samples,'captured');manifest=b.tensor_manifest(raw);nonce,tmps,mt,j=commit_bundle((REFX,REFR),REFC,'t0q5r4_reference_commit');save_file({k:v.clone() for k,v in raw.items()},tmps[REFX]);fsync(tmps[REFX]);result={'kind':'port80b_t0q5r4_reference','status':'unverified_reference','runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'execution_prereg_sha256':sha(PR),'science_prereg_sha256':sha(SCI),'codec_sha256':sha(C3),'generator_sha256':sha(GEN),'prompt_lock_sha256':sha(PL),'shard_sha256':sha(SHARD),'source_tensor_sha256':src,'raw_manifest':manifest,'raw_sha256':sha(tmps[REFX]),'resources':samples,'cuda_initialized':torch.cuda.is_initialized()};tmps[REFR].write_bytes(canon(result)+b'\n');fsync(tmps[REFR]);promote(tmps,mt,REFC,j);return result
def skey(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{NAMES[j]}_proj.weight'
def iq(v):
 r,k=v.shape;w=v.float().reshape(r,k//128,128);mx=w.abs().amax(-1,keepdim=True);s=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);f=(q.to(torch.int16)+15).numpy().astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales
def idecode(codes,scales,r,k):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if (f==31).any():raise ValueError('field31')
 q=torch.from_numpy((f.astype(np.int16)-15).reshape(r,k//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(r,k//128,1);return(q*s).reshape(r,k).to(torch.bfloat16)
def parse(record,e,j):
 f=struct.unpack(HF,record[:64]);codes=record[64:64+CB];scales=record[64+CB:64+CB+SB]
 if f[:11]!=(b'SQ5M',1,0,e,j,5,SHAPES[j][0],SHAPES[j][1],128,CB,SB) or f[12]!=bytes(28) or record[-PAD:]!=bytes(PAD) or (zlib.crc32(scales,zlib.crc32(codes))&0xffffffff)!=f[11]:raise ValueError('record identity/integrity')
 return idecode(codes,scales,f[6],f[7])
def build(samples,tmpbank):
 rows=[]
 with safe_open(SHARD,framework='pt',device='cpu') as src,tmpbank.open('xb') as h:
  for e in range(513):
   for j in range(3):v=src.get_tensor(skey(e,j));record,m=c.make_record(v,e,j);m.update(source_key=skey(e,j),offset=(e*3+j)*MB);h.write(record);rows.append(m)
   if e%32==0:resources(samples,f'build{e}')
  h.flush();os.fsync(h.fileno())
 with safe_open(SHARD,framework='pt',device='cpu') as src,tmpbank.open('rb') as h:
  for e in range(513):
   for j in range(3):
    record=h.read(MB);v=src.get_tensor(skey(e,j));codes,scales=iq(v);d=parse(record,e,j);m=rows[e*3+j]
    if record[64:64+CB]!=codes or record[64+CB:64+CB+SB]!=scales or m['source_sha256']!=hashlib.sha256(b.tensor_bytes(v)).hexdigest() or m['decoded_weight_sha256']!=hashlib.sha256(b.tensor_bytes(d)).hexdigest():raise RuntimeError('prepromotion independent record verification')
 core={'revision':c.REVISION,'shard_sha256':sha(SHARD),'codec_sha256':sha(C3),'bank_bytes':tmpbank.stat().st_size,'bank_sha256':sha(tmpbank),'records':rows};return core
def getbank(h,e,j):h.seek((e*3+j)*MB);return parse(h.read(MB),e,j)
def graph(x,ids,weights,get):
 final=torch.zeros_like(x);mask=F.one_hot(ids,num_classes=512).permute(2,1,0)
 for ei in torch.greater(mask.sum((-1,-2)),0).nonzero():
  ei=ei[0];top_k_pos,token_idx=torch.where(mask[ei]);state=x[token_idx];gate,up=F.linear(state,torch.cat((get(int(ei),0),get(int(ei),1)),0)).chunk(2,-1);down=F.linear(F.silu(gate)*up,get(int(ei),2))*weights[token_idx,top_k_pos,None];final.index_add_(0,token_idx,down.to(final.dtype))
 shared=F.linear(F.silu(F.linear(x,get(512,0)))*F.linear(x,get(512,1)),get(512,2));return final,shared
def metric(a,z):return r3.metric(a,z)
def safe_reject(record,e,j):
 try:parse(record,e,j);return None
 except Exception as x:return {'error_type':type(x).__name__,'error':str(x)}
def controls(p,x,ids,w,gate,get,h,baseline,raw):
 rows=[]
 for n in (8,15):
  original=int(ids[n,0]);absent=next(e for e in range(512) if e not in set(ids[n].tolist()))
  specs=[('wrong_expert',original,0,absent,0),('fixed_boundary_identity',499,0,498,0),('projection_swap',original,0,original,1)]
  for name,re,jj,pe,pj in specs:
   h.seek((pe*3+pj)*MB);record=h.read(MB);rejection=safe_reject(record,re,jj);changed=ids.clone();changed[n,0]=pe;rr,ss=graph(x,changed,w,get);out=rr+gate*ss;rk=f'p{p}_n{n}_{name}_unsafe_complete';raw[rk]=out[n:n+1].clone();rows.append({'prompt':p,'position':n,'control':name,'requested_expert':re,'requested_projection':jj,'presented_expert':pe,'presented_projection':pj,'safe_rejection':rejection,'raw_key':rk})
  h.seek((512*3+2)*MB);record=bytearray(h.read(MB));d=get(512,2);act=F.silu(F.linear(x,get(512,0)))*F.linear(x,get(512,1));chosen=None
  codes=bytes(record[64:64+CB]);scales=bytes(record[64+CB:64+CB+SB]);pp=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);ww=pp[:,0]|pp[:,1]<<8|pp[:,2]<<16|pp[:,3]<<24|pp[:,4]<<32;ff=np.stack([(ww>>(5*i))&31 for i in range(8)],-1).reshape(2048,512);qq=ff.astype(np.int16)-15
  for a in range(2048):
   for cc in range(512):
    if qq[a,cc]!=0 and act[n,cc]!=0:chosen=(a,cc);break
   if chosen:break
  if chosen is None:raise RuntimeError('mutation selector empty')
  a,cc=chosen;new=int(qq[a,cc]-(1 if qq[a,cc]>0 else -1));slot=a*512+cc;block=slot//8;within=slot%8;word=int.from_bytes(record[64+block*5:64+block*5+5],'little');word=(word&~(31<<(5*within)))|((new+15)<<(5*within));record[64+block*5:64+block*5+5]=word.to_bytes(5,'little');rejection=safe_reject(bytes(record),512,2);mut=idecode(bytes(record[64:64+CB]),bytes(record[64+CB:64+CB+SB]),2048,512)
  def mg(e,j):return mut if e==512 and j==2 else get(e,j)
  rr,ss=graph(x,ids,w,mg);out=rr+gate*ss;rk=f'p{p}_n{n}_code_mutation_unsafe_complete';raw[rk]=out[n:n+1].clone();rows.append({'prompt':p,'position':n,'control':'code_mutation','requested_expert':512,'requested_projection':2,'presented_expert':512,'presented_projection':2,'matrix_row':a,'matrix_column':cc,'mutated_q':new,'safe_rejection':rejection,'raw_key':rk})
 return rows
def q5():
 guard('q5');prompts();D.mkdir(parents=True,exist_ok=True);samples=[];resources(samples,'start');nonce,bt,bc_tmp,bj=commit_bundle((BANK,MAN),BC,'t0q5r4_bank_commit');core=build(samples,bt[BANK]);bt[MAN].write_bytes(c.manifest_file(core));fsync(bt[MAN]);promote(bt,bc_tmp,BC,bj);raw={};metrics={};ctrl=[]
 with safe_open(REFX,framework='pt',device='cpu') as rf,safe_open(SHARD,framework='pt',device='cpu') as src,BANK.open('rb') as h,torch.inference_mode():
  for p in range(4):
   x=rf.get_tensor(f'p{p}_mlp_input').reshape(16,2048);ids=rf.get_tensor(f'p{p}_router_ids');w=rf.get_tensor(f'p{p}_router_weights');gate=rf.get_tensor(f'p{p}_shared_gate').reshape(16,1);res=rf.get_tensor(f'p{p}_pre_mlp_residual');source=lambda e,j:src.get_tensor(skey(e,j));decoded=lambda e,j:getbank(h,e,j);sr,ss=graph(x,ids,w,source);qr,qs=graph(x,ids,w,decoded);sm=sr+gate*ss;qm=qr+gate*qs;sl=torch.add(res,sm.reshape(1,16,2048));ql=torch.add(res,qm.reshape(1,16,2048));official={'routed':rf.get_tensor(f'p{p}_routed'),'shared_raw':rf.get_tensor(f'p{p}_shared_raw'),'shared_gated':rf.get_tensor(f'p{p}_shared_gated'),'complete_mlp':rf.get_tensor(f'p{p}_complete_mlp'),'layer':rf.get_tensor(f'p{p}_layer_output')};sv={'routed':sr,'shared_raw':ss,'shared_gated':gate*ss,'complete_mlp':sm,'layer':sl};qv={'routed':qr,'shared_raw':qs,'shared_gated':gate*qs,'complete_mlp':qm,'layer':ql}
   if any(not torch.equal(sv[k].reshape_as(official[k]),official[k]) for k in sv):raise RuntimeError(f'graph control {p}')
   for k,v in sv.items():raw[f'p{p}_source_{k}']=v.clone()
   for k,v in qv.items():raw[f'p{p}_q5_{k}']=v.clone()
   metrics[str(p)]={k:[metric(sv[k].reshape_as(official[k])[:,n:n+1] if official[k].ndim==3 else sv[k][n:n+1],qv[k].reshape_as(official[k])[:,n:n+1] if official[k].ndim==3 else qv[k][n:n+1]) for n in range(8,16)] for k in sv};ctrl.extend(controls(p,x,ids,w,gate,decoded,h,qm,raw))
 resources(samples,'computed');manifest=b.tensor_manifest(raw);nonce,tmps,mt,j=commit_bundle((QX,QR),QC,'t0q5r4_q5_commit');save_file({k:v.clone() for k,v in raw.items()},tmps[QX]);fsync(tmps[QX]);result={'kind':'port80b_t0q5r4_q5','status':'requires_independent_q5_verification','runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'execution_prereg_sha256':sha(PR),'science_prereg_sha256':sha(SCI),'codec_sha256':sha(C3),'prompt_lock_sha256':sha(PL),'reference_verification_sha256':sha(REFV),'reference_raw_sha256':sha(REFX),'reference_result_sha256':sha(REFR),'bank_sha256':sha(BANK),'bank_manifest_sha256':sha(MAN),'bank_commit_sha256':sha(BC),'raw_manifest':manifest,'raw_sha256':sha(tmps[QX]),'metrics':metrics,'controls':ctrl,'resources':samples,'cuda_initialized':torch.cuda.is_initialized()};tmps[QR].write_bytes(canon(result)+b'\n');fsync(tmps[QR]);added=sum(x.stat().st_size for x in (BANK,MAN,BC,tmps[QX],tmps[QR]));
 if added>MAX_ART:raise RuntimeError('artifact byte gate')
 result['artifact_bytes_before_commit']=added;tmps[QR].write_bytes(canon(result)+b'\n');fsync(tmps[QR]);promote(tmps,mt,QC,j);return result
def fail(phase,e):
 moved=quarantine(phase);FAILED.mkdir(parents=True,exist_ok=True);p=FAILED/f'{phase}_{uuid.uuid4().hex}_failure.json';mi=psutil.Process().memory_info();payload={'kind':'port80b_t0q5r4_failure','phase':phase,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'prereg_sha256':sha(PR),'resources':{'rss':mi.rss,'peak':getattr(mi,'peak_wset',mi.rss),'available':psutil.virtual_memory().available,'disk_free':shutil.disk_usage(ROOT).free},'dispositions':moved,'cuda_initialized':torch.cuda.is_initialized()};p.write_bytes(canon(payload)+b'\n');fsync(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lockcheck','reference','q5'),required=True);p.add_argument('--ack');a=p.parse_args()
 if a.phase=='lockcheck':print(json.dumps({'kind':'t0q5r4_lockcheck',**lockcheck(),'physical_actions':False}));return 0
 expected='T0Q5R4_REFERENCE_AFTER_AUDIT' if a.phase=='reference' else 'T0Q5R4_Q5_AFTER_VERIFIED_REFERENCE_AND_AUDIT'
 if a.ack!=expected:raise SystemExit('ack')
 try:out=reference() if a.phase=='reference' else q5();print(out['status']);return 3
 except Exception as e:fail(a.phase,e);raise
if __name__=='__main__':raise SystemExit(main())
