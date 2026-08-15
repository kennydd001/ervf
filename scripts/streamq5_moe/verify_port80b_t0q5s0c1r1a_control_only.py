#!/usr/bin/env python3
"""Independent S0-C1-R1A verifier; directly rereads official source and reconstructs the sentinel."""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,psutil,torch,torch.nn.functional as F
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0c1r1a_control_only';RAW=D/'s0c1r1a_raw.safetensors';RES=D/'s0c1r1a_result.json';COM=D/'s0c1r1a_commit.json';RUN=S/'run_port80b_t0q5s0c1r1a_control_only.py';SELF=Path(__file__);PR=R/'PORT80B_T0Q5S0C1R1A_CONTROL_ONLY_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5s0c1r1a_runner_lock.json';VL=R/'port80b_t0q5s0c1r1a_verifier_lock.json';DEP=R/'port80b_t0r4_dependency_execution_lock.json';DIAG=R/'PORT80B_T0Q5S0R5_CONTROL_DIAGNOSIS_2026-08-13.md';CLOSED=R/'PORT80B_T0Q5S0C1R1_CLOSED_PREFLIGHT_RESULT_2026-08-13.json';R5D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5s0r5_selected_route_validation';R5RAW=R5D/'s0r5_raw.safetensors';R5RES=R5D/'s0r5_result.json';R5COM=R5D/'s0r5_commit.json';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';KEY='model.layers.0.mlp.shared_expert.down_proj.weight';SHARD_SHA='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bits(t):return t.contiguous().view(torch.uint8).numpy().tobytes()
def pack_independent(v):
 rows,cols=v.shape;vf=v.float();groups=vf.reshape(rows,cols//128,128);maxima=groups.abs().amax(2,keepdim=True);scale_fp32=maxima/15.0;quant=torch.where(maxima>0,torch.round(groups/scale_fp32).clamp(-15,15),torch.zeros_like(groups)).to(torch.int8);scale_store=torch.where(maxima>0,scale_fp32,torch.ones_like(scale_fp32)).squeeze(2).to(torch.bfloat16);fields=(quant.to(torch.int16)+15).reshape(-1).tolist();out=bytearray()
 for i in range(0,len(fields),8):
  word=0
  for slot,val in enumerate(fields[i:i+8]):word|=int(val)<<(5*slot)
  out.extend(word.to_bytes(5,'little'))
 return bytes(out),scale_store.view(torch.uint16).numpy().astype('<u2',copy=False).tobytes(),quant
def unpack_independent(codes,scales):
 fs=[]
 for i in range(0,len(codes),5):
  word=int.from_bytes(codes[i:i+5],'little');fs.extend((word>>(5*j))&31 for j in range(8))
 if len(fs)!=2048*512 or any(x==31 for x in fs):raise RuntimeError('wire')
 q=torch.tensor(fs,dtype=torch.int16).sub(15).reshape(2048,4,128).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(2048,4,1);return(q*s).reshape(2048,512).to(torch.bfloat16),torch.tensor(fs,dtype=torch.int16).reshape(2048,512)
def expected_manifest(tensors):return {n:{'dtype':str(t.dtype),'shape':list(t.shape),'bytes':t.numel()*t.element_size(),'sha256':hashlib.sha256(bits(t)).hexdigest()} for n,t in sorted(tensors.items())}
def independent_checker(requested,presented):
 errors=[]
 for name in ('expert','projection','shape'):
  if requested[name]!=presented[name]:errors.append(name)
 if requested['codes_scales_digest']!=presented['codes_scales_digest']:errors.append('codes_scales_digest')
 return errors
@torch.inference_mode()
def verify():
 proc=psutil.Process();aff=json.loads(DEP.read_text())['runtime']['process_affinity'];proc.cpu_affinity(aff);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False);sub=torch.nextafter(torch.zeros(1,dtype=torch.float32),torch.ones(1,dtype=torch.float32));flush=int((sub*torch.ones_like(sub)).view(torch.uint32)[0])==1
 r=json.loads(RES.read_text());c=json.loads(COM.read_text());rl=json.loads(LOCK.read_text());vl=json.loads(VL.read_text());checks={}
 checks['commit']=set(c['files'])=={RAW.name,RES.name} and c['files'][RAW.name]=={'bytes':RAW.stat().st_size,'sha256':sha(RAW)} and c['files'][RES.name]=={'bytes':RES.stat().st_size,'sha256':sha(RES)}
 expected_bind={'runner_sha256':sha(RUN),'verifier_sha256':sha(SELF),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'dependency_lock_sha256':sha(DEP),'diagnosis_sha256':sha(DIAG),'closed_preflight_sha256':sha(CLOSED),'r5_raw_sha256':sha(R5RAW),'r5_result_sha256':sha(R5RES),'r5_commit_sha256':sha(R5COM),'shard_sha256':sha(SHARD)};checks['bindings']=all(r[k]==v and rl[k]==v for k,v in expected_bind.items()) and r['runner_lock_sha256']==sha(LOCK) and r['runner_lock_content']==rl and vl['verifier_sha256']==sha(SELF) and rl['execution_open'] is True and rl['implementation_audit_token']==vl['implementation_audit_token']=='S0C1R1A_AUTHORIZED_AFTER_SOURCE_GO'
 r5=json.loads(R5RES.read_text());r5c=json.loads(R5COM.read_text());checks['r5_immutable_provenance']=r5['status']=='selected_route_validation_positive' and r5c['files'][R5RAW.name]=={'bytes':R5RAW.stat().st_size,'sha256':sha(R5RAW)} and r5c['files'][R5RES.name]=={'bytes':R5RES.stat().st_size,'sha256':sha(R5RES)} and sha(SHARD)==SHARD_SHA
 with safe_open(RAW,framework='pt',device='cpu') as f:stored={n:f.get_tensor(n) for n in f.keys()}
 schema={'activation':('torch.bfloat16',[1,512]),'original_output':('torch.bfloat16',[1,2048]),'mutated_output':('torch.bfloat16',[1,2048]),'selected_q':('torch.int8',[2]),'selected_bf16_words':('torch.int32',[2]),'fp32_product_bits':('torch.int64',[2])};checks['raw_schema_manifest']=set(stored)==set(schema) and all((str(stored[n].dtype),list(stored[n].shape))==v for n,v in schema.items()) and expected_manifest(stored)==r['raw_manifest'] and r['raw_sha256']==sha(RAW) and all(torch.isfinite(x.float()).all() for x in stored.values())
 with safe_open(SHARD,framework='pt',device='cpu') as f:source=f.get_tensor(KEY)
 source_ok=str(source.dtype)=='torch.bfloat16' and list(source.shape)==[2048,512] and hashlib.sha256(bits(source)).hexdigest()=='83565fde9bab5de0109f102c0f21cebd6533d7776b8f6fb837400534dbc5e1f5';codes,scales,q=pack_independent(source);decoded,fields=unpack_independent(codes,scales);checks['official_record']=source_ok and hashlib.sha256(codes).hexdigest()=='7d2311c8c455cb556d7c65b25df833196272c8f195762b9ac3d482afdf68e65d' and hashlib.sha256(scales).hexdigest()=='85d438d73b626b7513356c4792e947162d18c3115ad08bd2de428d08b47a197b' and hashlib.sha256(codes+scales).hexdigest()=='ca74f57285f066334ac9adfdf47ea3cc9e3823859b8d7c3c0a775ffb9168f076' and hashlib.sha256(bits(decoded)).hexdigest()=='9b24af43030dde4854c7a76cdfaf92045f22099f2e198a3f4f78f187a026b91d' and not(fields==31).any()
 scale_words=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).reshape(2048,4);sel=None
 for a in range(2048):
  for col in range(512):
   z=int(q.reshape(2048,512)[a,col]);zp=z-(1 if z>0 else -1)
   if z!=0 and -15<=zp<=15 and int(decoded[a,col].view(torch.uint16))!=int(torch.tensor(zp*float(scale_words[a,col//128]),dtype=torch.bfloat16).view(torch.uint16)):sel=(a,col,z,zp);break
  if sel is not None:break
 checks['selection']=sel==(0,0,6,5) and stored['selected_q'].tolist()==[6,5]
 a,col,z,zp=sel;mut=bytearray(codes);linear=a*512+col;block=linear//8;slot=linear%8;before=bytes(mut[block*5:block*5+5]);word=int.from_bytes(before,'little');word=(word&~(31<<(5*slot)))|((zp+15)<<(5*slot));after=word.to_bytes(5,'little');mut[block*5:block*5+5]=after;mutbytes=bytes(mut);mutated,mutfields=unpack_independent(mutbytes,scales);changed=int((fields!=mutfields).sum());original_digest=hashlib.sha256(codes+scales).hexdigest();mutated_digest=hashlib.sha256(mutbytes+scales).hexdigest();outside_preserved=codes[:block*5]==mutbytes[:block*5] and codes[block*5+5:]==mutbytes[block*5+5:];field_mask=31<<(5*slot);within_preserved=(int.from_bytes(before,'little')&~field_mask)==(int.from_bytes(after,'little')&~field_mask);checks['one_field_mutation']=changed==1 and int(fields[a,col])==z+15 and int(mutfields[a,col])==zp+15 and outside_preserved and within_preserved and len(codes)==len(mutbytes) and original_digest!=mutated_digest and r['record']['original_codes_scales_sha256']==original_digest and r['record']['mutated_codes_scales_sha256']==mutated_digest and r['record']['changed_field_count']==1 and r['record']['packed_block_index']==block and r['record']['packed_slot_index']==slot and r['record']['packed_block_before_hex']==before.hex() and r['record']['packed_block_after_hex']==after.hex() and r['record']['scales_unchanged'] is True
 chosen=None
 for k in range(-8,9):
  oy=torch.tensor((2.0**k)*float(decoded[a,col]),dtype=torch.bfloat16);my=torch.tensor((2.0**k)*float(mutated[a,col]),dtype=torch.bfloat16)
  if torch.isfinite(oy) and torch.isfinite(my) and int(oy.view(torch.uint16))!=int(my.view(torch.uint16)):chosen=(k,oy,my);break
 k,oracle,mut_oracle=chosen;x=torch.zeros((1,512),dtype=torch.bfloat16);x[0,col]=2.0**k;fp0=x[0,col].float()*decoded[a,col].float();fp1=x[0,col].float()*mutated[a,col].float();product_bits=[int(fp0.view(torch.uint32)),int(fp1.view(torch.uint32))];bf16_words=[int(oracle.view(torch.uint16)),int(mut_oracle.view(torch.uint16))];oo=F.linear(x,decoded);mo=F.linear(x,mutated);activation_sha=hashlib.sha256(bits(x)).hexdigest();rec=r['record'];record_exact=rec['source_key']==KEY and rec['row']==a==0 and rec['column']==col==0 and rec['q']==z==6 and rec['q_prime']==zp==5 and rec['k']==k==-8 and rec['activation_sha256']==activation_sha and rec['activation_nonzero_count']==1 and rec['activation_nonzero_index']==[0,col] and rec['activation_nonzero_value']==float(x[0,col]) and rec['fp32_product_bits']==product_bits and rec['original_bf16_word']==bf16_words[0] and rec['mutated_bf16_word']==bf16_words[1] and rec['bf16_word_xor']==(bf16_words[0]^bf16_words[1])
 checks['onehot_oracle']=record_exact and int(torch.count_nonzero(x))==1 and torch.equal(x,stored['activation']) and torch.equal(oo,stored['original_output']) and torch.equal(mo,stored['mutated_output']) and torch.equal(oo[0,a],oracle) and torch.equal(mo[0,a],mut_oracle) and torch.equal(fp0.to(torch.bfloat16),oracle) and torch.equal(fp1.to(torch.bfloat16),mut_oracle) and bf16_words[0]!=bf16_words[1] and stored['selected_bf16_words'].tolist()==bf16_words and stored['fp32_product_bits'].tolist()==product_bits
 requested={'expert':512,'projection':2,'shape':[2048,512],'codes_scales_digest':original_digest};presented={'expert':512,'projection':2,'shape':[2048,512],'codes_scales_digest':mutated_digest};errors=independent_checker(requested,presented);expected_ledger=[{'ordinal':0,'event':'safe_checker_rejected','errors':['codes_scales_digest'],'unsafe_decode_calls':0,'unsafe_linear_calls':0},{'ordinal':1,'event':'unsafe_decode','unsafe_decode_calls':1,'unsafe_linear_calls':0},{'ordinal':2,'event':'unsafe_linear','unsafe_decode_calls':1,'unsafe_linear_calls':1}];checks['safe_before_unsafe']=r['requested_metadata']==requested and r['presented_metadata']==presented and errors==r['safe_rejection_errors']==['codes_scales_digest'] and r['ledger']==expected_ledger and r['counters']=={'safe_checker_calls':1,'unsafe_decode_calls':1,'unsafe_linear_calls':1}
 checks['claim_status']=r['kind']=='port80b_t0q5s0c1r1a_control_only' and r['status']=='control_sensitivity_positive' and 'synthetic control sensitivity only' in r['claim_boundary'] and 'R5 remains formal verifier-negative' in r['claim_boundary']
 expected_runtime={'affinity':aff,'threads':1,'interop':1,'deterministic':True,'mkldnn':True,'matmul_precision':'highest','autocast_cpu':False,'inference_mode':True,'cuda_initialized':False};checks['runtime_resources']=flush and r['runtime']==expected_runtime and r['resource_policy']=='conservative inherited gates: start 16GiB, peak 12GiB; expected C1 incremental working RAM <64MiB' and r['resources'][0]['available']>=16*2**30 and {x['stage'] for x in r['resources']}=={'start','computed','cleanup','post_serialization'} and all(x['available']>=2*2**30 and x['peak']<=12*2**30 for x in r['resources']) and sum(p.stat().st_size for p in (RAW,RES,COM))<=1024*1024 and not torch.cuda.is_initialized()
 return {'kind':'t0q5s0c1r1a_independent_verification','pass':all(checks.values()),'checks':checks}
def main():
 o=verify();print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
