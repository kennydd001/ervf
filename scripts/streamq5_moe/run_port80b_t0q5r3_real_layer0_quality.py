#!/usr/bin/env python3
"""T0Q5-R3 two-phase runner. Closed until audited; CPU only."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,os,shutil,struct,sys,traceback,zlib
from pathlib import Path
import numpy as np
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HUB_KERNELS='0',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5r3_real_layer0_quality';SNAP=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f';SHARD=SNAP/'model-00001-of-00040.safetensors'
BASE=S/'run_port80b_t0r12_official_cpu_reference_only.py';CODEC=S/'port80b_t0q5r3_codec_contract.py';GEN=S/'generate_port80b_t0q5r1_prompts.py';VER=S/'verify_port80b_t0q5r3_real_layer0_quality.py';PRE=R/'PORT80B_T0Q5R3_REAL_LAYER0_NUMERICAL_QUALITY_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5r3_runner_lock.json';VL=R/'port80b_t0q5r3_verifier_lock.json';PL=R/'port80b_t0q5r3_prompt_lock.json';BANK=D/'t0q5r3_layer0.sq5m';MAN=D/'t0q5r3_bank_manifest.json';REF_RAW=D/'t0q5r3_reference.safetensors';REF_RES=D/'t0q5r3_reference.json';Q5_RAW=D/'t0q5r3_q5.safetensors';Q5_RES=D/'t0q5r3_q5.json'
def load(path,name):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
b=load(BASE,'t0q5base');c=load(CODEC,'t0q5codec');g=load(GEN,'t0q5gen');torch=b.torch;F=b.F;safe_open=b.safe_open;save_file=b.save_file;DynamicCache=b.DynamicCache;psutil=b.psutil
ACK_REF='T0Q5R3_REFERENCE_AFTER_IMPLEMENTATION_AUDIT';ACK_Q5='T0Q5R3_Q5_AFTER_REFERENCE_VERIFICATION_AND_AUDIT'
def sha(p):return b.sha256(Path(p))
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def lockcheck():
 l=json.loads(LOCK.read_text());a={'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PRE),'codec_sha256':sha(CODEC),'generator_sha256':sha(GEN),'base_sha256':sha(BASE),'prompt_lock_sha256':sha(PL) if PL.exists() else '__ABSENT_PENDING_IMPLEMENTATION_AUDIT__'};return {'pass':all(l.get(k)==v for k,v in a.items()) and PL.exists(),'bindings':a}
def promptcheck():
 if not PL.exists():raise RuntimeError('canonical prompt lock absent')
 observed=g.generate();frozen=json.loads(PL.read_text());old=json.loads(b.PROMPT_LOCK.read_text());
 if canon(observed)!=canon(frozen) or len(frozen['prompts'])!=4 or any(len(x['token_ids'])!=16 for x in frozen['prompts']):raise RuntimeError('prompt replay/schema')
 for x in frozen['prompts']:
  for y in old['prompts']:
   if x['utf8_text']==y['utf8_text'] or x['token_ids']==y['token_ids'] or x['token_ids_le_u32_sha256']==y['token_ids_le_u32_sha256']:raise RuntimeError('old/new not disjoint')
 for i,x in enumerate(frozen['prompts']):
  for y in frozen['prompts'][i+1:]:
   if x['utf8_text']==y['utf8_text'] or x['token_ids']==y['token_ids'] or x['token_ids_le_u32_sha256']==y['token_ids_le_u32_sha256']:raise RuntimeError('new/new not disjoint')
 return frozen
def guard(phase):
 if not lockcheck()['pass']:raise RuntimeError('lock mismatch')
 targets=[REF_RAW,REF_RES] if phase=='reference' else [Q5_RAW,Q5_RES,BANK,MAN]
 if any(x.exists() for x in targets) or any(D.glob(f'{phase}_failure*.json')):raise FileExistsError('target/failure exists')
 if psutil.virtual_memory().available<16*2**30 or torch.cuda.is_initialized():raise RuntimeError('resource/GPU guard')
 if phase=='q5' and not (REF_RAW.exists() and REF_RES.exists()):raise RuntimeError('verified reference inputs absent')
def capture(layer,h,config):
 cap={};hooks=[]
 def save(n):
  def hook(_m,_i,o):cap[n]=(o[0] if isinstance(o,tuple) else o).detach().cpu().contiguous().clone()
  return hook
 def gate(_m,i,o):cap['router_input']=i[0].detach().cpu().contiguous().clone();cap['router_logits'],cap['router_weights'],cap['router_ids']=[x.detach().cpu().contiguous().clone() for x in o]
 def residual(_m,i):cap['pre_mlp_residual']=i[0].detach().cpu().contiguous().clone()
 hooks += [layer.post_attention_layernorm.register_forward_pre_hook(residual),layer.post_attention_layernorm.register_forward_hook(save('mlp_input')),layer.mlp.experts.register_forward_hook(save('routed')),layer.mlp.shared_expert.register_forward_hook(save('shared_raw')),layer.mlp.shared_expert_gate.register_forward_hook(save('shared_gate_linear')),layer.mlp.register_forward_hook(save('complete_mlp')),layer.mlp.gate.register_forward_hook(gate)]
 try:
  cache=DynamicCache(config=config);empty=torch.empty(0,dtype=torch.bfloat16);out=layer(h,position_embeddings=(empty,empty),attention_mask=None,past_key_values=cache);_,ct=b.cache_state(cache,0,16)
 finally:
  for q in hooks:q.remove()
 second=layer.mlp.gate(cap['router_input']);cap['router_second_logits'],cap['router_second_weights'],cap['router_second_ids']=[x.detach().cpu().clone() for x in second];cap['shared_gate']=torch.sigmoid(cap['shared_gate_linear']);cap['shared_gated']=cap['shared_gate']*cap['shared_raw'];cap['layer_output']=out.detach().cpu().clone();cap['cache_conv']=next(v for k,v in ct.items() if k.endswith('cache_conv'));cap['cache_recurrent']=next(v for k,v in ct.items() if k.endswith('cache_recurrent'))
 if not all(torch.equal(cap[k],cap['router_second_'+k.split('_',1)[1]]) for k in ('router_logits','router_weights','router_ids')):raise RuntimeError('direct tuple mismatch')
 if not torch.equal(torch.add(cap['pre_mlp_residual'],cap['complete_mlp']),cap['layer_output']):raise RuntimeError('residual identity')
 return cap
def reference():
 guard('reference');prompts=promptcheck();D.mkdir(parents=True,exist_ok=False);peak={};b.rss_guard('start',peak);cfg=b.Qwen3NextConfig.from_pretrained(SNAP,local_files_only=True,trust_remote_code=False);layer,embed,source=b.load_official(cfg,peak);ids=torch.tensor([x['token_ids'] for x in prompts]);hidden=F.embedding(ids,embed).to(torch.bfloat16);raw={'token_ids':ids,'embedding':hidden.cpu()}
 with torch.inference_mode():
  for p in range(4):
   for k,v in capture(layer,hidden[p:p+1],cfg).items():raw[f'p{p}_{k}']=v
 manifest=b.tensor_manifest(raw);save_file({k:v.clone() for k,v in raw.items()},REF_RAW);b.rss_guard('serialized',peak);res={'kind':'port80b_t0q5r3_reference','status':'reference_complete_not_q5_pass','runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PRE),'codec_sha256':sha(CODEC),'generator_sha256':sha(GEN),'base_sha256':sha(BASE),'prompt_lock_sha256':sha(PL),'shard_sha256':sha(SHARD),'source_tensor_sha256':source,'raw_manifest':manifest,'raw_sha256':sha(REF_RAW),'resources':peak,'cuda_initialized':torch.cuda.is_initialized()};REF_RES.write_bytes(canon(res)+b'\n');return res
def key(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{c.NAMES[j]}_proj.weight'
def build_bank(peak):
 tmp=BANK.with_suffix('.sq5m.inprogress');mt=MAN.with_suffix('.json.inprogress');rows=[]
 if tmp.exists() or mt.exists():raise FileExistsError('transaction temp exists')
 try:
  with safe_open(SHARD,framework='pt',device='cpu') as src,tmp.open('xb') as out:
   offset=0
   for e in range(513):
    for j in range(3):
     v=src.get_tensor(key(e,j));record,m=c.make_record(v,e,j);m.update(source_key=key(e,j),offset=offset);out.write(record);rows.append(m);offset+=len(record);del v,record
    if e%32==0:b.rss_guard(f'bank_{e}',peak)
   out.flush();os.fsync(out.fileno())
  core={'revision':c.REVISION,'shard_sha256':sha(SHARD),'codec_sha256':sha(CODEC),'bank_bytes':tmp.stat().st_size,'bank_sha256':sha(tmp),'records':rows};mt.write_bytes(c.manifest_file(core));
  with mt.open('rb') as q:os.fsync(q.fileno())
  os.replace(tmp,BANK);os.replace(mt,MAN);return core
 except Exception:
  for x in (tmp,mt):
   if x.exists():x.replace(x.with_suffix(x.suffix+'.failed'))
  raise
def read_matrix(handle,e,j):handle.seek((e*3+j)*c.MATRIX_BYTES);record=handle.read(c.MATRIX_BYTES);header=struct.unpack(c.HEADER_FORMAT,record[:64]);cb,sb=header[9],header[10];return c.decode(record[64:64+cb],record[64+cb:64+cb+sb],header[6],header[7])
def graph(x,ids,weights,get):
 final=torch.zeros_like(x);mask=F.one_hot(ids,num_classes=512).permute(2,1,0);hits=torch.greater(mask.sum((-1,-2)),0).nonzero()
 for ei in hits:
  ei=ei[0];top_k_pos,token_idx=torch.where(mask[ei]);current=x[token_idx];fused=torch.cat((get(int(ei),0),get(int(ei),1)),0);gate,up=F.linear(current,fused).chunk(2,-1);down=F.linear(F.silu(gate)*up,get(int(ei),2));down=down*weights[token_idx,top_k_pos,None];final.index_add_(0,token_idx,down.to(final.dtype))
 shared=F.linear(F.silu(F.linear(x,get(512,0)))*F.linear(x,get(512,1)),get(512,2));return final,shared
def controls_for_prompt(p,x,ids,weights,decoded,bank,baseline):
 rows=[]
 for n in (8,15):
  original=int(ids[n,0]);absent=next(e for e in range(512) if e not in set(ids[n].tolist()))
  for name,replacement in (('wrong_expert',absent),('fixed_boundary_identity',498)):
   changed=ids.clone();changed[n,0]=replacement;routed,shared=graph(x,changed,weights,decoded);candidate=routed+shared
   rows.append({'prompt':p,'position':n,'control':name,'requested_expert':original if name=='wrong_expert' else 499,'presented_expert':replacement,'safe_rejected':(replacement!=(original if name=='wrong_expert' else 499)),'unsafe_different_words':int((candidate[n]!=baseline[n]).sum())})
  def swapped(e,j):return decoded(e,1) if e==original and j==0 else decoded(e,j)
  routed,shared=graph(x,ids,weights,swapped);candidate=routed+shared;rows.append({'prompt':p,'position':n,'control':'projection_swap','requested_projection':0,'presented_projection':1,'expert':original,'safe_rejected':True,'unsafe_different_words':int((candidate[n]!=baseline[n]).sum())})
  gate=F.linear(x,decoded(512,0));up=F.linear(x,decoded(512,1));activation=F.silu(gate)*up;mut=decoded(512,2).clone();chosen=None
  bank.seek((512*3+2)*c.MATRIX_BYTES);record=bank.read(c.MATRIX_BYTES);head=struct.unpack(c.HEADER_FORMAT,record[:64]);codes=record[64:64+c.CODE_BYTES];scales=record[64+c.CODE_BYTES:64+c.CODE_BYTES+c.SCALE_BYTES];packed=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);word=packed[:,0]|packed[:,1]<<8|packed[:,2]<<16|packed[:,3]<<24|packed[:,4]<<32;fields=np.stack([(word>>(5*i))&31 for i in range(8)],-1).reshape(2048,512);q=fields.astype(np.int16)-15;sbits=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16)
  for rr in range(2048):
   for cc in range(512):
    if q[rr,cc]!=0 and activation[n,cc]!=0:chosen=(rr,cc);break
   if chosen:break
  if chosen is None:raise RuntimeError('code mutation selector empty')
  rr,cc=chosen;old=int(q[rr,cc]);new=old-(1 if old>0 else -1);mut[rr,cc]=torch.tensor(float(new)*float(sbits[rr*4+cc//128]),dtype=torch.bfloat16)
  def mutated(e,j):return mut if e==512 and j==2 else decoded(e,j)
  routed,shared=graph(x,ids,weights,mutated);candidate=routed+shared;mutated_codes=bytearray(codes);bit=rr*512+cc;group8=bit//8;slot=bit%8;w=int.from_bytes(mutated_codes[group8*5:group8*5+5],'little');w=(w&~(31<<(5*slot)))|((new+15)<<(5*slot));mutated_codes[group8*5:group8*5+5]=w.to_bytes(5,'little');safe=(zlib.crc32(scales,zlib.crc32(mutated_codes))&0xffffffff)!=head[11]
  rows.append({'prompt':p,'position':n,'control':'code_mutation','source_q':old,'mutated_q':new,'matrix_row':rr,'matrix_column':cc,'safe_rejected':safe,'unsafe_different_words':int((candidate[n]!=baseline[n]).sum())})
 return rows
def metric(a,z):
 av=a.reshape(-1).float().double();zv=z.reshape(-1).float().double();diff=zv-av;ss=ee=cc=aa=bb=0.0;ma=0.0
 for x,y,d in zip(av.tolist(),zv.tolist(),diff.tolist()):ss+=x*x;ee+=d*d;cc+=x*y;aa+=x*x;bb+=y*y;ma=max(ma,abs(d))
 rn=ss**.5;en=ee**.5;cn=bb**.5;rel=0.0 if rn==0 and en==0 else (float('inf') if rn==0 else en/rn);cos=1.0 if rn==0 and cn==0 else (0.0 if rn==0 or cn==0 else cc/(rn*cn));return {'max_abs':ma,'rel_l2':rel,'cosine':cos,'different_words':int((a.view(torch.uint16)!=z.view(torch.uint16)).sum()),'max_bf16_ulp':b.max_bf16_ulp(a,z)}
def q5():
 guard('q5');prompts=promptcheck();peak={};b.rss_guard('start',peak);ref=json.loads(REF_RES.read_text());
 if ref['raw_sha256']!=sha(REF_RAW):raise RuntimeError('reference hash')
 core=build_bank(peak);raw={};metrics={};controls=[]
 with safe_open(REF_RAW,framework='pt',device='cpu') as rf,safe_open(SHARD,framework='pt',device='cpu') as src,BANK.open('rb') as bank,torch.inference_mode():
  for p in range(4):
   x=rf.get_tensor(f'p{p}_mlp_input').reshape(16,2048);ids=rf.get_tensor(f'p{p}_router_ids');weights=rf.get_tensor(f'p{p}_router_weights');residual=rf.get_tensor(f'p{p}_pre_mlp_residual')
   def source(e,j):return src.get_tensor(key(e,j))
   def decoded(e,j):return read_matrix(bank,e,j)
   sr,ss=graph(x,ids,weights,source);qr,qs=graph(x,ids,weights,decoded);sg=rf.get_tensor(f'p{p}_shared_gate').reshape(16,1);sm=sr+ss*sg;qm=qr+qs*sg;sl=torch.add(residual,sm.reshape(1,16,2048));ql=torch.add(residual,qm.reshape(1,16,2048))
   official={'routed':rf.get_tensor(f'p{p}_routed'),'shared_raw':rf.get_tensor(f'p{p}_shared_raw'),'shared_gated':rf.get_tensor(f'p{p}_shared_gated'),'complete_mlp':rf.get_tensor(f'p{p}_complete_mlp'),'layer':rf.get_tensor(f'p{p}_layer_output')};sourcevals={'routed':sr,'shared_raw':ss,'shared_gated':ss*sg,'complete_mlp':sm,'layer':sl};qvals={'routed':qr,'shared_raw':qs,'shared_gated':qs*sg,'complete_mlp':qm,'layer':ql}
   if any(not torch.equal(sourcevals[k].reshape_as(v),v) for k,v in official.items()):raise RuntimeError(f'graph control p{p}')
   for k,v in sourcevals.items():raw[f'p{p}_source_{k}']=v.clone()
   for k,v in qvals.items():raw[f'p{p}_q5_{k}']=v.clone()
   metrics[str(p)]={k:[metric(sourcevals[k].reshape_as(official[k])[:,n:n+1] if official[k].ndim==3 else sourcevals[k][n:n+1],qvals[k].reshape_as(official[k])[:,n:n+1] if official[k].ndim==3 else qvals[k][n:n+1]) for n in range(8,16)] for k in sourcevals};controls.extend(controls_for_prompt(p,x,ids,weights,decoded,bank,qm))
 manifest=b.tensor_manifest(raw);save_file({k:v.clone() for k,v in raw.items()},Q5_RAW);b.rss_guard('serialized',peak);res={'kind':'port80b_t0q5r3_q5','status':'q5_evidence_requires_independent_verification','runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PRE),'codec_sha256':sha(CODEC),'generator_sha256':sha(GEN),'base_sha256':sha(BASE),'prompt_lock_sha256':sha(PL),'reference_result_sha256':sha(REF_RES),'reference_raw_sha256':sha(REF_RAW),'bank_sha256':sha(BANK),'manifest_file_sha256':sha(MAN),'bank_core':core,'raw_sha256':sha(Q5_RAW),'raw_manifest':manifest,'metrics':metrics,'controls':controls,'resources':peak,'cuda_initialized':torch.cuda.is_initialized()};Q5_RES.write_bytes(canon(res)+b'\n');return res
def failure(phase,e):
 D.mkdir(parents=True,exist_ok=True);p=D/f'{phase}_failure.json'
 if p.exists():return
 vm=psutil.virtual_memory();proc=psutil.Process();resources={'rss_bytes':proc.memory_info().rss,'windows_peak_working_set_bytes':getattr(proc.memory_info(),'peak_wset',proc.memory_info().rss),'available_ram_bytes':vm.available,'disk_free_bytes':shutil.disk_usage(ROOT).free}
 payload=canon({'kind':'port80b_t0q5r3_failure','phase':phase,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PRE),'codec_sha256':sha(CODEC),'generator_sha256':sha(GEN),'base_sha256':sha(BASE),'resources':resources,'cuda_initialized':torch.cuda.is_initialized(),'partial':{x.name:x.exists() for x in (BANK,MAN,REF_RAW,REF_RES,Q5_RAW,Q5_RES)}})+b'\n'
 with p.open('xb') as h:h.write(payload);h.flush();os.fsync(h.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lockcheck','reference','q5'),required=True);p.add_argument('--acknowledge');a=p.parse_args()
 if a.phase=='lockcheck':print(json.dumps({'kind':'t0q5r3_lockcheck',**lockcheck(),'physical_actions':False}));return 0
 if a.acknowledge!=(ACK_REF if a.phase=='reference' else ACK_Q5):raise SystemExit('exact acknowledgement required')
 try:r=reference() if a.phase=='reference' else q5();print(r['status']);return 3
 except Exception as e:failure(a.phase,e);raise
if __name__=='__main__':raise SystemExit(main())
