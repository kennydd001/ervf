#!/usr/bin/env python3
"""S0-C1-R2 standalone real-record control sentinel. No model, GPU, bank, or quality rerun."""
from __future__ import annotations
import argparse,gc,hashlib,json,math,os,shutil,struct,traceback,uuid
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,psutil,torch,torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0c1r2_control_only';RAW=D/'s0c1r2_raw.safetensors';RES=D/'s0c1r2_result.json';COM=D/'s0c1r2_commit.json';FAIL=D/'s0c1r2_failure.json';BAD=D/'failed_attempts';PR=R/'PORT80B_T0Q5S0C1R2_CONTROL_ONLY_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5s0c1r2_runner_lock.json';VL=R/'port80b_t0q5s0c1r2_verifier_lock.json';VER=S/'verify_port80b_t0q5s0c1r2_control_only.py';DEP=R/'port80b_t0r4_dependency_execution_lock.json';DIAG=R/'PORT80B_T0Q5S0R5_CONTROL_DIAGNOSIS_2026-08-13.md';CLOSED=R/'PORT80B_T0Q5S0C1R1_CLOSED_PREFLIGHT_RESULT_2026-08-13.json';R1AD=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0c1r1a_control_only';R1AFAIL=R1AD/'s0c1r1a_failure.json';R1ARAW=R1AD/'failed_attempts/105c1ea0873d48a896624bf44bb1f7fe_s0c1r1a_raw.safetensors.2cf2eb2e5b454a2c8d5aa93d57cc41b1.inprogress';R5D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r5_selected_route_validation';R5RAW=R5D/'s0r5_raw.safetensors';R5RES=R5D/'s0r5_result.json';R5COM=R5D/'s0r5_commit.json';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';KEY='model.layers.0.mlp.shared_expert.down_proj.weight';SHARD_SHA='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a';ACK='T0Q5S0C1R2_CONTROL_ONLY_AFTER_SOURCE_AUDIT';TOKEN='S0C1R2_IMPLEMENTATION_AUDIT_GO';MAX=1024*1024
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def durable(p):
 with Path(p).open('r+b' if os.name=='nt' else 'rb') as h:os.fsync(h.fileno())
def fsyncdir(p):
 try:
  fd=os.open(str(p),os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 except OSError:
  if os.name!='nt':raise
def sample(stage):
 m=psutil.Process().memory_info();return {'stage':stage,'rss':m.rss,'peak':getattr(m,'peak_wset',m.rss),'available':psutil.virtual_memory().available}
def bindings():return {'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'dependency_lock_sha256':sha(DEP),'diagnosis_sha256':sha(DIAG),'closed_preflight_sha256':sha(CLOSED),'r1a_failure_sha256':sha(R1AFAIL),'r1a_quarantined_raw_sha256':sha(R1ARAW),'r5_raw_sha256':sha(R5RAW),'r5_result_sha256':sha(R5RES),'r5_commit_sha256':sha(R5COM),'shard_sha256':sha(SHARD)}
def lockcheck():
 l=json.loads(LOCK.read_text());b=bindings();auth=l.get('execution_open') is True and l.get('control_only_authorization') is True and l.get('implementation_audit_token')==TOKEN and l.get('physical_run_requires_independent_source_go') is True;return {'pass':auth and all(l.get(k)==v for k,v in b.items()),'bindings':b}
def committed():
 try:
  c=json.loads(COM.read_text());return set(c['files'])=={RAW.name,RES.name} and all((D/n).exists() and c['files'][n]=={'bytes':(D/n).stat().st_size,'sha256':sha(D/n)} for n in c['files'])
 except Exception:return False
def quant(v):
 r,c=v.shape;w=v.float().reshape(r,c//128,128);mx=w.abs().amax(-1,keepdim=True);unrounded=mx/15;s=torch.where(mx>0,unrounded,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);field=(q.to(torch.int16)+15).numpy();f=field.astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales,q
def decode(codes,scales):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if (f==31).any():raise ValueError('field31')
 q=torch.from_numpy((f.astype(np.int16)-15).reshape(2048,4,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(2048,4,1);return(q*s).reshape(2048,512).to(torch.bfloat16),f
class Rejected(Exception):pass
def checker(requested,presented,ledger,counters):
 counters['safe_checker_calls']+=1;errors=[]
 for k in ('expert','projection','shape'):
  if requested[k]!=presented[k]:errors.append(k)
 if requested['codes_scales_digest']!=presented['codes_scales_digest']:errors.append('codes_scales_digest')
 ledger.append({'ordinal':len(ledger),'event':'safe_checker_rejected','errors':errors,'unsafe_decode_calls':counters['unsafe_decode_calls'],'unsafe_linear_calls':counters['unsafe_linear_calls']})
 if errors:raise Rejected(errors)
 raise RuntimeError('control not rejected')
def recover():
 moved={}
 if not D.exists():return moved
 if COM.exists():
  try:
   c=json.loads(COM.read_text());ok=set(c['files'])=={RAW.name,RES.name} and all((D/n).exists() and c['files'][n]=={'bytes':(D/n).stat().st_size,'sha256':sha(D/n)} for n in c['files'])
   if ok:return moved
  except Exception:pass
 BAD.mkdir(exist_ok=True)
 for p in list(D.iterdir()):
  if p==BAD:continue
  q=BAD/f'{uuid.uuid4().hex}_{p.name}';os.rename(p,q);moved[p.name]=q.name
 fsyncdir(D);return moved
@torch.inference_mode()
def execute():
 if not lockcheck()['pass']:raise RuntimeError('lock')
 if committed():return {'kind':'port80b_t0q5s0c1r2_already_complete','status':'already_complete'}
 moved=recover()
 if D.exists() and any(p!=BAD for p in D.iterdir()):raise FileExistsError('nonclean')
 if psutil.virtual_memory().available<16*2**30 or torch.cuda.is_initialized():raise RuntimeError('start/GPU')
 D.mkdir(parents=True,exist_ok=True);proc=psutil.Process();aff=json.loads(DEP.read_text())['runtime']['process_affinity'];proc.cpu_affinity(aff);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False);resources=[sample('start')];ledger=[];counters={'safe_checker_calls':0,'unsafe_decode_calls':0,'unsafe_linear_calls':0}
 r5=json.loads(R5RES.read_text());r5c=json.loads(R5COM.read_text())
 if r5['status']!='selected_route_validation_positive' or r5c['files'][R5RAW.name]!={'bytes':R5RAW.stat().st_size,'sha256':sha(R5RAW)} or r5c['files'][R5RES.name]!={'bytes':R5RES.stat().st_size,'sha256':sha(R5RES)}:raise RuntimeError('R5 provenance')
 with safe_open(SHARD,framework='pt',device='cpu') as f:source=f.get_tensor(KEY)
 if str(source.dtype)!='torch.bfloat16' or list(source.shape)!=[2048,512] or hashlib.sha256(source.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()!='83565fde9bab5de0109f102c0f21cebd6533d7776b8f6fb837400534dbc5e1f5':raise RuntimeError('source')
 codes,scales,q=quant(source);decoded,fields=decode(codes,scales)
 if hashlib.sha256(codes).hexdigest()!='7d2311c8c455cb556d7c65b25df833196272c8f195762b9ac3d482afdf68e65d' or hashlib.sha256(scales).hexdigest()!='85d438d73b626b7513356c4792e947162d18c3115ad08bd2de428d08b47a197b' or hashlib.sha256(codes+scales).hexdigest()!='ca74f57285f066334ac9adfdf47ea3cc9e3823859b8d7c3c0a775ffb9168f076':raise RuntimeError('codec')
 qq=q.reshape(2048,512);sel=next(( (a,c) for a in range(2048) for c in range(512) if int(qq[a,c])!=0 and -15<=int(qq[a,c])- (1 if int(qq[a,c])>0 else -1)<=15 and decoded[a,c].view(torch.uint16)!=torch.tensor((int(qq[a,c])-(1 if int(qq[a,c])>0 else -1))*float(torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).reshape(2048,4)[a,c//128]),dtype=torch.bfloat16).view(torch.uint16)),None)
 if sel!=(0,0) or int(qq[0,0])!=6:raise RuntimeError('selection')
 a,c=sel;qp=5;scale=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).reshape(2048,4)[a,c//128];ow=decoded[a,c];mw=torch.tensor(qp*float(scale),dtype=torch.bfloat16);chosen=None
 for k in range(-8,9):
  oy=torch.tensor((2.0**k)*float(ow),dtype=torch.bfloat16);my=torch.tensor((2.0**k)*float(mw),dtype=torch.bfloat16)
  if torch.isfinite(oy) and torch.isfinite(my) and int(oy.view(torch.uint16))!=int(my.view(torch.uint16)):chosen=(k,oy,my);break
 if chosen is None:raise RuntimeError('control_design_negative')
 k,oracle,mut_oracle=chosen;requested={'expert':512,'projection':2,'shape':[2048,512],'codes_scales_digest':hashlib.sha256(codes+scales).hexdigest()};mutcodes=bytearray(codes);linear=a*512+c;block=linear//8;slot=linear%8;before_block=codes[block*5:block*5+5];word=int.from_bytes(before_block,'little');word=(word&~(31<<(5*slot)))|((qp+15)<<(5*slot));after_block=word.to_bytes(5,'little');mutcodes[block*5:block*5+5]=after_block;presented={**requested,'codes_scales_digest':hashlib.sha256(bytes(mutcodes)+scales).hexdigest()}
 try:checker(requested,presented,ledger,counters)
 except Rejected as e:errors=e.args[0]
 counters['unsafe_decode_calls']+=1;ledger.append({'ordinal':len(ledger),'event':'unsafe_decode','unsafe_decode_calls':counters['unsafe_decode_calls'],'unsafe_linear_calls':counters['unsafe_linear_calls']});mutated,mutfields=decode(bytes(mutcodes),scales);x=torch.zeros((1,512),dtype=torch.bfloat16);x[0,c]=2.0**k;counters['unsafe_linear_calls']+=1;ledger.append({'ordinal':len(ledger),'event':'unsafe_linear','unsafe_decode_calls':counters['unsafe_decode_calls'],'unsafe_linear_calls':counters['unsafe_linear_calls']});original_out=F.linear(x,decoded);mutated_out=F.linear(x,mutated)
 if errors!=['codes_scales_digest'] or counters!={'safe_checker_calls':1,'unsafe_decode_calls':1,'unsafe_linear_calls':1} or not torch.equal(original_out[0,a],oracle) or not torch.equal(mutated_out[0,a],mut_oracle) or torch.equal(oracle,mut_oracle):raise RuntimeError('control')
 activation_sha=hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();fp32_original=torch.tensor((2.0**k)*float(ow),dtype=torch.float32);fp32_mutated=torch.tensor((2.0**k)*float(mw),dtype=torch.float32);product_bits=[int(fp32_original.view(torch.uint32)),int(fp32_mutated.view(torch.uint32))];bf16_words=[int(oracle.view(torch.uint16)),int(mut_oracle.view(torch.uint16))];bf16_xor=bf16_words[0]^bf16_words[1];changed_field_count=int((fields!=mutfields).sum())
 raw={'activation':x,'original_output':original_out,'mutated_output':mutated_out,'selected_q':torch.tensor([6,5],dtype=torch.int8),'selected_bf16_words':torch.tensor(bf16_words,dtype=torch.int32),'fp32_product_bits':torch.tensor(product_bits,dtype=torch.int64)};manifest={n:{'dtype':str(t.dtype),'shape':list(t.shape),'bytes':t.numel()*t.element_size(),'sha256':hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()} for n,t in sorted(raw.items())}
 resources.append(sample('computed'));del source,decoded,mutated,q,fields,mutfields,original_out,mutated_out;gc.collect();resources.append(sample('cleanup'));nonce=uuid.uuid4().hex;rt=D/f'{RAW.name}.{nonce}.inprogress';jt=D/f'{RES.name}.{nonce}.inprogress';ct=D/f'{COM.name}.{nonce}.inprogress';save_file(raw,rt);durable(rt);resources.append(sample('post_serialization'));runtime={'affinity':proc.cpu_affinity(),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),'mkldnn':torch.backends.mkldnn.enabled,'matmul_precision':torch.get_float32_matmul_precision(),'autocast_cpu':torch.is_autocast_enabled('cpu'),'inference_mode':torch.is_inference_mode_enabled(),'cuda_initialized':torch.cuda.is_initialized()};result={'kind':'port80b_t0q5s0c1r2_control_only','status':'control_sensitivity_positive','claim_boundary':'synthetic control sensitivity only; R5 remains formal verifier-negative; no quality rerun/heldout/layer/model/GPU/performance','runner_lock_content':json.loads(LOCK.read_text()),**bindings(),'runner_lock_sha256':sha(LOCK),'verifier_lock_sha256':sha(VL),'record':{'source_key':KEY,'row':a,'column':c,'q':6,'q_prime':5,'k':k,'original_codes_scales_sha256':requested['codes_scales_digest'],'mutated_codes_scales_sha256':presented['codes_scales_digest'],'changed_field_count':changed_field_count,'packed_block_index':block,'packed_slot_index':slot,'packed_block_before_hex':before_block.hex(),'packed_block_after_hex':after_block.hex(),'scales_unchanged':True,'activation_sha256':activation_sha,'activation_nonzero_count':int(torch.count_nonzero(x)),'activation_nonzero_index':[0,c],'activation_nonzero_value':float(x[0,c]),'fp32_product_bits':product_bits,'original_bf16_word':bf16_words[0],'mutated_bf16_word':bf16_words[1],'bf16_word_xor':bf16_xor},'requested_metadata':requested,'presented_metadata':presented,'safe_rejection_errors':errors,'ledger':ledger,'counters':counters,'raw_manifest':manifest,'raw_sha256':sha(rt),'runtime':runtime,'resources':resources,'resource_policy':'conservative inherited gates: start 16GiB, peak 12GiB; expected C1 incremental working RAM <64MiB','recovered':moved};jt.write_bytes(canon(result)+b'\n');durable(jt);commit={'kind':'t0q5s0c1r2_commit','files':{RAW.name:{'bytes':rt.stat().st_size,'sha256':sha(rt)},RES.name:{'bytes':jt.stat().st_size,'sha256':sha(jt)}}};ct.write_bytes(canon(commit)+b'\n');durable(ct)
 if sum(p.stat().st_size for p in (rt,jt,ct))>MAX:raise RuntimeError('artifact size')
 for p in (RAW,RES,COM):
  if p.exists():raise FileExistsError(p)
 os.rename(rt,RAW);os.rename(jt,RES);fsyncdir(D);os.rename(ct,COM);fsyncdir(D);return result
def failure(e):
 if committed():return
 gc.collect();moved=recover() if D.exists() else {};D.mkdir(parents=True,exist_ok=True);t=D/f'{FAIL.name}.{uuid.uuid4().hex}.inprogress';p={'kind':'t0q5s0c1r2_failure','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'runner_sha256':sha(__file__),'runner_lock_sha256':sha(LOCK),'resources':sample('failure_cleanup'),'dispositions':moved,'cuda_initialized':torch.cuda.is_initialized()};t.write_bytes(canon(p)+b'\n');durable(t)
 if t.stat().st_size>MAX:raise RuntimeError('failure size')
 if FAIL.exists():raise FileExistsError(FAIL)
 os.rename(t,FAIL);fsyncdir(D)
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=('lockcheck','execute'),required=True);a.add_argument('--ack',default='');z=a.parse_args()
 if z.phase=='lockcheck':print(json.dumps({'kind':'s0c1r2_lockcheck',**lockcheck(),'physical_actions':False},indent=2));return 0
 if z.ack!=ACK:raise SystemExit('ack')
 try:print(json.dumps(execute(),indent=2));return 0
 except Exception as e:failure(e);print(json.dumps({'status':'failure','error':str(e)}));return 3
if __name__=='__main__':raise SystemExit(main())
