#!/usr/bin/env python3
"""Standalone, CPU-only verification of the PH1 CPU-freeze R2 package."""
from __future__ import annotations

import hashlib, json, math, os, platform, struct, zlib
from pathlib import Path
import mpmath as mp
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[2]; REPORTS=ROOT/'reports/streamq5_moe'
PKG=REPORTS/'het_next_l0_ph1_cpu_freeze_r2'; RAW=PKG/'cpu_stage_freeze.safetensors'
OUT=REPORTS/'het_next_l0_ph1_cpu_freeze_r2_independent_verification.json'
SHARD=Path(r'C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors')
D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors'
SPECS=(
 ('gate',0,(512,2048),(3498051416,3500148568),'05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c','20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b','658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8','9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77',1976639022,'e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9'),
 ('up',1,(512,2048),(3500148568,3502245720),'4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d','6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb','c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9','ca239543f7a478e757040a994d001a15b70481c7b87bca3cc8641831305394ea',4920057,'6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb'),
 ('down',2,(2048,512),(3495954264,3498051416),'bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479','3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703','a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e','ef9c19383d9b1ff90a4ba0015942594c4188dd42c407103a06f26a1953d56c34',4066311128,'bd1a8ef9ae689fefebf73408f3985c96a0725670dc0b0f7f46268a5a89d12157'))
EXPECTED_STAGE={'natural_input':'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f','source_gate_up':'94550e9b214edd4713aff00902ee5083f0bf1d9e633bf43a950ecdcf5f8efdf7','source_gate':'2a898f7c33c8df8ed441222cfe3a62672fab0e5ae612905e0bf98cd53ea861cc','source_up':'0018b298d0c0f55fa38a8fd0141fb4684601911cfd88e7da4cf2480083cf580f','source_silu':'184fb8cc8c0a46cd7a6f00c65350d8cd12e3a38defb899e5401caf2d3f2d03be','source_activation':'598a656ed0d56ae51bd503ffcdb93f73fff239ab725000209469835b08dbfa26','source_down':'ed49c260c3b09985dbfec10106a04eaea99b59d97114514f7099d4bdb84c6e09','cpu_q5_gate':'e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867','cpu_q5_up':'f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08','cpu_q5_silu':'a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8','cpu_q5_activation':'762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f','cpu_q5_down':'142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc'}
HEADER=struct.Struct('<4sHHHBBIIH2xIII28s'); GROUP=128

def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def rr(p,o,n):
 with p.open('rb') as f:f.seek(o);b=f.read(n)
 if len(b)!=n:raise EOFError(p)
 return b
def b2f(w):return (np.asarray(w,np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)
def f2b(v):
 b=np.asarray(v,np.float32).view(np.uint32);return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
def quant(src,shape):
 r,c=shape;v=b2f(np.frombuffer(src,'<u2')).reshape(r,c);bl=v.reshape(r,c//128,128);m=np.max(np.abs(bl),axis=-1,keepdims=True);s=np.where(m>0,np.asarray(m/np.float32(15),np.float32),np.float32(1));q=np.where(m>0,np.clip(np.rint(np.asarray(bl/s,np.float32)),-15,15),0).astype(np.int16);fld=(q+15).astype(np.uint64).reshape(-1,8);w=np.bitwise_or.reduce(fld<<(np.arange(8,dtype=np.uint64)*5),axis=1);codes=np.stack([(w>>(8*i))&255 for i in range(5)],axis=1).astype(np.uint8).tobytes();sw=f2b(s.reshape(-1));sc=sw.astype('<u2',copy=False).tobytes();dec=f2b(q.reshape(r,c).astype(np.float32)*b2f(sw).reshape(r,c//128).repeat(128,axis=1)).astype('<u2').tobytes();return codes,sc,dec,q.reshape(r,c)
def rse(n,s):
 if s<=0:return n<<(-s)
 q,r=divmod(n,1<<s);h=1<<(s-1);return q+int(r>h or (r==h and q&1))
def parts(b):
 sg=-1 if b>>31 else 1;e=(b>>23)&255;f=b&0x7fffff
 if e==255:raise ValueError('nonfinite')
 return (sg*f,-149) if e==0 else (sg*((1<<23)|f),e-150)
def pack(n,e):
 if n==0:return 0
 sg=0x80000000 if n<0 else 0;n=abs(n);top=n.bit_length()-1+e
 if top>127:return sg|0x7f800000
 if top>=-126:
  sh=n.bit_length()-24;si=rse(n,sh)
  if si==1<<24:si>>=1;sh+=1
  ue=e+sh+23
  return sg|0x7f800000 if ue>127 else sg|((ue+127)<<23)|(si&0x7fffff)
 fr=rse(n,-149-e)
 return sg if fr==0 else sg|(1<<23) if fr>=1<<23 else sg|fr
def fma(a,b,c):
 an,ae=parts(a);bn,be=parts(b);cn,ce=parts(c);pn,pe=an*bn,ae+be;e=min(pe,ce);return pack((pn<<(pe-e))+(cn<<(ce-e)),e)
def add(a,b):return fma(a,0x3f800000,b)
def rb(b):
 if (b&0x7f800000)==0x7f800000:raise ValueError('nonfinite')
 return ((b+0x7fff+((b>>16)&1))>>16)&0xffff
def mul(a,b):
 if (a&0x7f80)==0x7f80 or (b&0x7f80)==0x7f80:raise ValueError('nonfinite')
 if (a&0x7fff)==0 or (b&0x7fff)==0:return (a^b)&0x8000
 return rb(fma(a<<16,b<<16,0))
def linear(weights,x):
 rows,cols=weights.shape;vc=cols//64;tree=(16,8,4,2,1) if cols==2048 else (4,2,1);out=np.empty(rows,np.uint16)
 for row in range(rows):
  p=[[0]*vc for _ in range(8)]
  for lane in range(8):
   for v in range(vc):
    col=(lane+8*v)*8;acc=0
    for k in range(8):acc=fma(int(weights[row,col+k])<<16,int(x[col+k])<<16,acc)
    p[lane][v]=acc
  for d in tree:
   for lane in range(8):
    old=p[lane].copy()
    for i in range(d):p[lane][i]=add(old[i],old[i+d])
  lanes=[p[i][0] for i in range(8)]
  for off in (4,2,1):
   old=lanes.copy()
   for i in range(off):lanes[i]=add(old[i],old[i+off])
  out[row]=rb(lanes[0])
 return out
def mpval(w):
 sg=-1 if w>>15 else 1;e=(w>>7)&255;f=w&127
 if e==255:return None
 return mp.mpf(sg)*mp.mpf(f)*mp.power(2,-133) if e==0 else mp.mpf(sg)*mp.mpf(128+f)*mp.power(2,e-134)
def mpbf(v,word):
 if v==0:return 0x8000 if word==0x8000 else 0
 sg=0x8000 if v<0 else 0;a=abs(v);e=int(mp.floor(mp.log(a,2)))
 if e< -126:
  ex=a*mp.power(2,133);n=int(mp.floor(ex));r=ex-n;n+=int(r>mp.mpf('.5') or (r==mp.mpf('.5') and n&1));return sg|(0x80 if n>=128 else n)
 bi=e+127;ex=a/mp.power(2,e)*128;n=int(mp.floor(ex));r=ex-n;n+=int(r>mp.mpf('.5') or (r==mp.mpf('.5') and n&1))
 if n==256:n=128;bi+=1
 return (sg|0x7f80) if bi>=255 else sg|(bi<<7)|(n-128)
def thash(t):return sha(t.contiguous().view(torch.uint8).numpy().tobytes())

def main():
 if OUT.exists():raise FileExistsError(OUT)
 manifest=json.loads((PKG/'manifest.json').read_text());commit=json.loads((PKG/'commit.json').read_text());result=json.loads((PKG/'cpu_stage_freeze.json').read_text());handoff=json.loads((PKG/'handoff.json').read_text())
 names=[x['name'] for x in manifest['files']];exact_set=set(names)|{'manifest.json','commit.json'}=={p.name for p in PKG.iterdir()};rows_ok=all((PKG/x['name']).stat().st_size==x['bytes'] and fsha(PKG/x['name'])==x['sha256'] for x in manifest['files']);commit_ok=commit=={'kind':'ph1_cpu_freeze_r2_commit','manifest_sha256':fsha(PKG/'manifest.json'),'handoff_sha256':fsha(PKG/'handoff.json'),'base_result_sha256':fsha(PKG/'cpu_stage_freeze.json')}
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False)
 inp=rr(D2,155138788,4096);iw=np.frombuffer(inp,'<u2').copy();sources={};decoded={};record_checks={}
 for name,ordinal,shape,rng,ss,cs,scs,ds,crc,rs in SPECS:
  src=rr(SHARD,rng[0],rng[1]-rng[0]);codes,scales,dec,q=quant(src,shape);head=HEADER.pack(b'SQ5M',1,0,50,ordinal,5,*shape,128,len(codes),len(scales),zlib.crc32(scales,zlib.crc32(codes))&0xffffffff,bytes(28));record=head+codes+scales+bytes(4032);record_checks[name]=sha(src)==ss and sha(codes)==cs and sha(scales)==scs and sha(dec)==ds and (zlib.crc32(scales,zlib.crc32(codes))&0xffffffff)==crc and sha(record)==rs and len(record)==675840;sources[name]=torch.from_numpy(np.frombuffer(src,'<u2').copy()).view(torch.bfloat16).reshape(shape);decoded[name]=np.frombuffer(dec,'<u2').copy().reshape(shape)
 words=np.arange(65536,dtype=np.uint16);t=torch.from_numpy(words.copy()).view(torch.bfloat16);finite=torch.isfinite(t);lut=np.zeros(65536,np.uint16);lut[finite.numpy()]=F.silu(t[finite],inplace=False).view(torch.uint16).numpy();mp.mp.dps=100;mlut=np.zeros(65536,np.uint16)
 for w in range(65536):
  v=mpval(w)
  if v is not None:mlut[w]=mpbf(v/(1+mp.exp(-v)),w)
 x=torch.from_numpy(iw.copy()).view(torch.bfloat16)
 with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
  fused=torch.cat((sources['gate'],sources['up']),0).contiguous();gu=F.linear(x.contiguous(),fused);sg,su=gu.chunk(2,-1);ssilu=F.silu(sg,inplace=False);sact=ssilu*su;sdown=F.linear(sact.contiguous(),sources['down'])
 qg=linear(decoded['gate'],iw);qu=linear(decoded['up'],iw);qs=lut[qg];qa=np.asarray([mul(int(a),int(b)) for a,b in zip(qs,qu,strict=True)],np.uint16);qd=linear(decoded['down'],qa)
 calc={'natural_input':x,'source_gate_up':gu,'source_gate':sg,'source_up':su,'source_silu':ssilu,'source_activation':sact,'source_down':sdown,'cpu_q5_gate':torch.from_numpy(qg.copy()).view(torch.bfloat16),'cpu_q5_up':torch.from_numpy(qu.copy()).view(torch.bfloat16),'cpu_q5_silu':torch.from_numpy(qs.copy()).view(torch.bfloat16),'cpu_q5_activation':torch.from_numpy(qa.copy()).view(torch.bfloat16),'cpu_q5_down':torch.from_numpy(qd.copy()).view(torch.bfloat16)}
 raw=load_file(RAW,device='cpu');stage_hashes={k:thash(v) for k,v in calc.items()};raw_equal={k:k in raw and raw[k].dtype==v.dtype and tuple(raw[k].shape)==tuple(v.shape) and torch.equal(raw[k],v) for k,v in calc.items()}
 rv=b2f(sdown.view(torch.uint16).numpy().reshape(-1)).astype(np.float64);cv=b2f(qd).astype(np.float64);rn=en=0.0;ma=0.0
 for a,b in zip(rv,cv,strict=True):d=float(b-a);rn+=float(a*a);en+=d*d;ma=max(ma,abs(d))
 metric={'rel_l2':math.sqrt(en)/math.sqrt(rn),'max_abs':ma,'different_words':int(np.count_nonzero(sdown.view(torch.uint16).numpy()!=qd))}
 provenance_ok=all((ROOT/Path(k)).is_file() and fsha(ROOT/Path(k))==v for k,v in handoff['bindings'].items()) and fsha(ROOT/'scripts/streamq5_moe/run_het_next_l0_ph1_cpu_freeze_r2.py')==handoff['runner_sha256'] and fsha(REPORTS/'HET_NEXT_L0_PH1_CPU_FREEZE_R2_LIFECYCLE_PREREGISTRATION_2026-08-13.md')==handoff['r2_prereg_sha256']
 resource=handoff['resource'];resource_ok=resource['start_available']>=16*2**30 and resource['final_available']>=2*2**30 and resource['peak_wset']<=12*2**30
 checks={'package_exact_set':exact_set,'manifest_rows_exact':rows_ok,'commit_exact':commit_ok,'input_exact':sha(inp)==EXPECTED_STAGE['natural_input'],'three_records_exact':all(record_checks.values()),'normative_lut_exact':sha(lut.astype('<u2').tobytes())=='a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed' and (PKG/'bf16_silu_lut.bin').read_bytes()==lut.astype('<u2').tobytes(),'math_lut_exact':sha(mlut.astype('<u2').tobytes())=='f2efcbdc3b94b42a24dfe187321ae2a426e7685ab447e05452be994e843693c2' and (PKG/'high_precision_silu_diagnostic.bin').read_bytes()==mlut.astype('<u2').tobytes(),'lut_diagnostic_145':int(np.count_nonzero(lut!=mlut))==145,'stage_hashes_exact':stage_hashes==EXPECTED_STAGE and stage_hashes==result['stage_hashes'],'raw_all_stages_exact':set(raw)==set(calc) and all(raw_equal.values()),'quality_exact':metric==result['quality'] and math.isfinite(metric['rel_l2']) and metric['rel_l2']<=.08,'all_stages_finite':all(bool(torch.isfinite(v).all()) for v in calc.values()),'positive_schema':result['positive'] is True and result['status']=='cpu_predevice_positive' and result['quality_threshold']['pass'] is True,'provenance_exact':provenance_ok,'authorization_exact':handoff['authorization']['lock_sha256']==fsha(REPORTS/'het_next_l0_ph1_cpu_freeze_r2_authorization_lock.json'),'resources_pass':resource_ok,'no_device_compiler':handoff['device_or_compiler_opened'] is False}
 out={'kind':'het_next_l0_ph1_cpu_freeze_r2_independent_verification','verifier_sha256':fsha(Path(__file__)),'checks':checks,'check_count':len(checks),'pass_count':sum(checks.values()),'pass':all(checks.values()),'record_checks':record_checks,'stage_hashes':stage_hashes,'raw_equal':raw_equal,'lut':{'normative_sha256':sha(lut.astype('<u2').tobytes()),'mathematical_sha256':sha(mlut.astype('<u2').tobytes()),'different_words':int(np.count_nonzero(lut!=mlut))},'quality':metric,'resource':resource,'eligibility':'intel_implementation_design_open' if all(checks.values()) else 'closed','claim_boundary':'CPU eligibility for implementing an Intel validation arm for one known expert/input only; no Intel execution authorization and no full MoE/layer/model/performance/generalization/breakthrough claim.'}
 OUT.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'pass':out['pass'],'checks':f"{out['pass_count']}/{out['check_count']}",'quality':metric,'eligibility':out['eligibility']},indent=2));return 0 if out['pass'] else 3
if __name__=='__main__':raise SystemExit(main())
