#!/usr/bin/env python3
"""Independent T0Q5-R3 verifier; deliberately imports neither runner nor codec."""
from __future__ import annotations
import argparse,hashlib,json,math,struct,zlib
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5r3_real_layer0_quality';SNAP=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f';SHARD=SNAP/'model-00001-of-00040.safetensors';BANK=D/'t0q5r3_layer0.sq5m';MAN=D/'t0q5r3_bank_manifest.json';RR=D/'t0q5r3_reference.json';RX=D/'t0q5r3_reference.safetensors';QR=D/'t0q5r3_q5.json';QX=D/'t0q5r3_q5.safetensors';RUN=ROOT/'scripts/streamq5_moe/run_port80b_t0q5r3_real_layer0_quality.py';CODEC=ROOT/'scripts/streamq5_moe/port80b_t0q5r3_codec_contract.py';GEN=ROOT/'scripts/streamq5_moe/generate_port80b_t0q5r1_prompts.py';PRE=R/'PORT80B_T0Q5R3_REAL_LAYER0_NUMERICAL_QUALITY_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5r3_runner_lock.json';VL=R/'port80b_t0q5r3_verifier_lock.json';PL=R/'port80b_t0q5r3_prompt_lock.json'
HF='<4sHHHBBIIH2xIII28s';MB=675840;CB=655360;SB=16384;PAD=4032;SHAPES=((512,2048),(512,2048),(2048,512));NAMES=('gate','up','down')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tb(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def key(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{NAMES[j]}_proj.weight'
def independent_quant(v):
 r,c=v.shape;w=v.float().reshape(r,c//128,128);mx=w.abs().amax(-1,keepdim=True);s=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);f=(q.to(torch.int16)+15).numpy().astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales
def independent_decode(codes,scales,r,c):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if (f==31).any():raise ValueError('field31')
 q=torch.from_numpy((f.astype(np.int16)-15).reshape(r,c//128,128)).float();bits=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16);s=bits.view(torch.bfloat16).float().reshape(r,c//128,1);return (q*s).reshape(r,c).to(torch.bfloat16)
def bank_verify(result):
 env=json.loads(MAN.read_text());core=env['manifest'];checks=env['manifest_sha256']==hashlib.sha256(canon(core)).hexdigest() and MAN.read_bytes()==canon(env)+b'\n' and BANK.stat().st_size==1040117760 and sha(BANK)==core['bank_sha256'];rows=core['records'];checks &= len(rows)==1539
 with BANK.open('rb') as h,safe_open(SHARD,framework='pt',device='cpu') as src:
  for e in range(513):
   for j in range(3):
    m=rows[e*3+j];record=h.read(MB);head=record[:64];f=struct.unpack(HF,head);codes=record[64:64+CB];scales=record[64+CB:64+CB+SB];pad=record[-PAD:];v=src.get_tensor(key(e,j));ic,is_=independent_quant(v);dec=independent_decode(codes,scales,f[6],f[7]);crc=zlib.crc32(scales,zlib.crc32(codes))&0xffffffff
    checks &= (f[:12]==(b'SQ5M',1,0,e,j,5,SHAPES[j][0],SHAPES[j][1],128,CB,SB,crc) and f[12]==bytes(28) and pad==bytes(PAD) and codes==ic and scales==is_ and m['offset']==(e*3+j)*MB and m['source_key']==key(e,j) and m['source_sha256']==hashlib.sha256(tb(v)).hexdigest() and m['decoded_weight_sha256']==hashlib.sha256(tb(dec)).hexdigest() and m['record_sha256']==hashlib.sha256(record).hexdigest())
 return bool(checks)
def metric(a,z):
 av=a.reshape(-1).float().double().tolist();zv=z.reshape(-1).float().double().tolist();ss=ee=dot=cn=0.;ma=0.
 for x,y in zip(av,zv):d=y-x;ss+=x*x;ee+=d*d;dot+=x*y;cn+=y*y;ma=max(ma,abs(d))
 rn=math.sqrt(ss);en=math.sqrt(ee);zn=math.sqrt(cn);rel=0. if rn==0 and en==0 else (math.inf if rn==0 else en/rn);cos=1. if rn==0 and zn==0 else (0. if rn==0 or zn==0 else dot/(rn*zn));ua=a.contiguous().view(torch.uint16).to(torch.int32);uz=z.contiguous().view(torch.uint16).to(torch.int32);oa=torch.where((ua&0x8000)!=0,0x8000-(ua&0x7fff),0x8000+ua);oz=torch.where((uz&0x8000)!=0,0x8000-(uz&0x7fff),0x8000+uz);return {'max_abs':ma,'rel_l2':rel,'cosine':cos,'different_words':int((ua!=uz).sum()),'max_bf16_ulp':int((oa-oz).abs().max())}
def raw_manifest(path):
 d={}
 with safe_open(path,framework='pt',device='cpu') as f:
  for k in f.keys():
   v=f.get_tensor(k);d[k]={'semantic_key':k,'dtype':str(v.dtype),'shape':list(v.shape),'bytes':v.numel()*v.element_size(),'sha256':hashlib.sha256(tb(v)).hexdigest()}
 return d
def verify_reference():
 r=json.loads(RR.read_text());m=raw_manifest(RX);ok=r['kind']=='port80b_t0q5r3_reference' and r['raw_sha256']==sha(RX) and r['raw_manifest']==m and r['shard_sha256']=='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a' and not r['cuda_initialized'];return {'kind':'t0q5r3_reference_verification','pass':bool(ok),'checks':{'result_and_manifest':bool(ok)}}
def verify_q5():
 r=json.loads(QR.read_text());checks={'bindings':r['runner_sha256']==sha(RUN) and r['verifier_sha256']==sha(__file__) and r['verifier_lock_sha256']==sha(VL) and r['prereg_sha256']==sha(PRE) and r['codec_sha256']==sha(CODEC) and r['generator_sha256']==sha(GEN),'reference_hashes':r['reference_result_sha256']==sha(RR) and r['reference_raw_sha256']==sha(RX),'bank_independent_all1539':bank_verify(r),'q5_raw_manifest':r['raw_sha256']==sha(QX) and r['raw_manifest']==raw_manifest(QX),'no_gpu_resources':not r['cuda_initialized'] and r['resources']['windows_peak_working_set_bytes']<=12*2**30 and r['resources']['minimum_available_ram_bytes']>=2*2**30}
 recomputed={};quality=True;graph_control=True
 with safe_open(QX,framework='pt',device='cpu') as f,safe_open(RX,framework='pt',device='cpu') as ref:
  for p in range(4):
   recomputed[str(p)]={}
   for k in ('routed','shared_raw','shared_gated','complete_mlp','layer'):
    a=f.get_tensor(f'p{p}_source_{k}');z=f.get_tensor(f'p{p}_q5_{k}');official=ref.get_tensor(f'p{p}_{"layer_output" if k=="layer" else k}');graph_control &= torch.equal(a.reshape_as(official),official);recomputed[str(p)][k]=[metric(a[:,n:n+1] if a.ndim==3 else a[n:n+1],z[:,n:n+1] if z.ndim==3 else z[n:n+1]) for n in range(8,16)]
    for q in recomputed[str(p)][k]:
     lim=.02 if k=='layer' else .08;quality &= q['rel_l2']<=lim
     if k=='layer':quality &= q['cosine']>=.999 and q['max_abs']<=.125
 checks['graph_control_bitwise']=bool(graph_control);checks['metrics_exact']=recomputed==r['metrics'];checks['quality_gates']=bool(quality and sum(q['rel_l2'] for p in recomputed.values() for q in p['layer'])/32<=.01);checks['controls']=len(r.get('controls',[]))==32 and all(x['safe_rejected'] and x['unsafe_different_words']>=1 for x in r.get('controls',[]));return {'kind':'t0q5r3_independent_verification','pass':all(checks.values()),'checks':checks}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('reference','q5'),required=True);a=p.parse_args();out=verify_reference() if a.phase=='reference' else verify_q5();print(json.dumps(out,indent=2));return 0 if out['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
