#!/usr/bin/env python3
"""Independent R4 verifier. Imports no runner/codec and rebuilds bank-selected graphs."""
from __future__ import annotations
import argparse,hashlib,json,math,struct,zlib
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5r4_auditable_quality';SNAP=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f';SHARD=SNAP/'model-00001-of-00040.safetensors';RUN=S/'run_port80b_t0q5r4_auditable_quality.py';SELF=Path(__file__);PR=R/'PORT80B_T0Q5R4_AUDITABLE_EXECUTION_PREREGISTRATION_2026-08-13.md';SCI=R/'PORT80B_T0Q5R3_REAL_LAYER0_NUMERICAL_QUALITY_PREREGISTRATION_2026-08-13.md';GEN=S/'generate_port80b_t0q5r1_prompts.py';CODEC=S/'port80b_t0q5r3_codec_contract.py';VL=R/'port80b_t0q5r4_verifier_lock.json';PL=R/'port80b_t0q5r4_prompt_lock.json';RX=D/'reference_raw.safetensors';RR=D/'reference_result.json';RC=D/'reference_commit.json';RV=D/'reference_verification.json';BANK=D/'layer0_bank.sq5m';MAN=D/'bank_manifest.json';BC=D/'bank_commit.json';QX=D/'q5_raw.safetensors';QR=D/'q5_result.json';QC=D/'q5_commit.json';HF='<4sHHHBBIIH2xIII28s';MB=675840;CB=655360;SB=16384;PAD=4032;SHAPES=((512,2048),(512,2048),(2048,512));NAMES=('gate','up','down')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tb(t):return t.contiguous().view(torch.uint8).numpy().tobytes()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def key(e,j):return f'model.layers.0.mlp.{"experts."+str(e) if e<512 else "shared_expert"}.{NAMES[j]}_proj.weight'
def manifest(path):
 out={}
 with safe_open(path,framework='pt',device='cpu') as f:
  for k in f.keys():v=f.get_tensor(k);out[k]={'semantic_key':k,'dtype':str(v.dtype),'shape':list(v.shape),'bytes':v.numel()*v.element_size(),'sha256':hashlib.sha256(tb(v)).hexdigest()}
 return out
def decode(record,e,j):
 f=struct.unpack(HF,record[:64]);codes=record[64:64+CB];scales=record[64+CB:64+CB+SB]
 if f[:11]!=(b'SQ5M',1,0,e,j,5,SHAPES[j][0],SHAPES[j][1],128,CB,SB) or f[12]!=bytes(28) or record[-PAD:]!=bytes(PAD) or zlib.crc32(scales,zlib.crc32(codes))&0xffffffff!=f[11]:raise ValueError('record')
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;ff=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if (ff==31).any():raise ValueError('field31')
 q=torch.from_numpy((ff.astype(np.int16)-15).reshape(f[6],f[7]//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(f[6],f[7]//128,1);return(q*s).reshape(f[6],f[7]).to(torch.bfloat16)
def idecode(codes,scales,r,k):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;ff=np.stack([(w>>(5*i))&31 for i in range(8)],-1);q=torch.from_numpy((ff.astype(np.int16)-15).reshape(r,k//128,128)).float();s=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16).view(torch.bfloat16).float().reshape(r,k//128,1);return(q*s).reshape(r,k).to(torch.bfloat16)
def get(h,e,j):h.seek((e*3+j)*MB);return decode(h.read(MB),e,j)
def graph(x,ids,w,getter):
 final=torch.zeros_like(x);mask=F.one_hot(ids,num_classes=512).permute(2,1,0)
 for ei in torch.greater(mask.sum((-1,-2)),0).nonzero():ei=ei[0];pos,tok=torch.where(mask[ei]);gate,up=F.linear(x[tok],torch.cat((getter(int(ei),0),getter(int(ei),1)),0)).chunk(2,-1);down=F.linear(F.silu(gate)*up,getter(int(ei),2))*w[tok,pos,None];final.index_add_(0,tok,down.to(final.dtype))
 shared=F.linear(F.silu(F.linear(x,getter(512,0)))*F.linear(x,getter(512,1)),getter(512,2));return final,shared
def metric(a,z):
 ss=ee=dot=cn=0.;ma=0.
 for x,y in zip(a.reshape(-1).float().double().tolist(),z.reshape(-1).float().double().tolist()):d=y-x;ss+=x*x;ee+=d*d;dot+=x*y;cn+=y*y;ma=max(ma,abs(d))
 rn=math.sqrt(ss);en=math.sqrt(ee);zn=math.sqrt(cn);rel=0. if rn==0 and en==0 else(math.inf if rn==0 else en/rn);cos=1. if rn==0 and zn==0 else(0. if rn==0 or zn==0 else dot/(rn*zn));ua=a.contiguous().view(torch.uint16).to(torch.int32);uz=z.contiguous().view(torch.uint16).to(torch.int32);oa=torch.where((ua&32768)!=0,32768-(ua&32767),32768+ua);oz=torch.where((uz&32768)!=0,32768-(uz&32767),32768+uz);return {'max_abs':ma,'rel_l2':rel,'cosine':cos,'different_words':int((ua!=uz).sum()),'max_bf16_ulp':int((oa-oz).abs().max())}
def commit(marker,files):
 m=json.loads(marker.read_text());return set(m['files'])=={x.name for x in files} and all(m['files'][x.name]['sha256']==sha(x) and m['files'][x.name]['bytes']==x.stat().st_size for x in files)
def reference():
 r=json.loads(RR.read_text());checks={'commit':commit(RC,(RX,RR)),'bindings':r['runner_sha256']==sha(RUN) and r['verifier_sha256']==sha(SELF) and r['verifier_lock_sha256']==sha(VL) and r['execution_prereg_sha256']==sha(PR) and r['science_prereg_sha256']==sha(SCI) and r['generator_sha256']==sha(GEN) and r['prompt_lock_sha256']==sha(PL),'raw':r['raw_sha256']==sha(RX) and r['raw_manifest']==manifest(RX),'resources':not r['cuda_initialized'] and all(x['available']>=2*2**30 and x['peak']<=12*2**30 for x in r['resources']),'shard':r['shard_sha256']=='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a'}
 with safe_open(RX,framework='pt',device='cpu') as f:
  semantic=True
  for p in range(4):
   semantic &= torch.equal(f.get_tensor(f'p{p}_router_logits'),f.get_tensor(f'p{p}_router_second_logits')) and torch.equal(f.get_tensor(f'p{p}_router_weights'),f.get_tensor(f'p{p}_router_second_weights')) and torch.equal(f.get_tensor(f'p{p}_router_ids'),f.get_tensor(f'p{p}_router_second_ids'));semantic &= torch.equal(f.get_tensor(f'p{p}_shared_gated'),f.get_tensor(f'p{p}_shared_gate')*f.get_tensor(f'p{p}_shared_raw'));semantic &= torch.equal(f.get_tensor(f'p{p}_layer_output'),torch.add(f.get_tensor(f'p{p}_pre_mlp_residual'),f.get_tensor(f'p{p}_complete_mlp')))
 checks['semantics']=bool(semantic);out={'kind':'port80b_t0q5r4_reference_verification','pass':all(checks.values()),'checks':checks,'reference_raw_sha256':sha(RX),'reference_result_sha256':sha(RR),'verifier_sha256':sha(SELF),'verifier_lock_sha256':sha(VL)}
 if out['pass']:
  tmp=RV.with_suffix('.json.inprogress');tmp.write_bytes(canon(out)+b'\n');
  with tmp.open('rb') as h:import os;os.fsync(h.fileno())
  os.replace(tmp,RV)
 return out
def q5():
 r=json.loads(QR.read_text());checks={'commits':commit(BC,(BANK,MAN)) and commit(QC,(QX,QR)),'reference_pass':json.loads(RV.read_text()).get('pass') and r['reference_verification_sha256']==sha(RV) and r['reference_raw_sha256']==sha(RX) and r['reference_result_sha256']==sha(RR),'bindings':r['runner_sha256']==sha(RUN) and r['verifier_sha256']==sha(SELF) and r['verifier_lock_sha256']==sha(VL) and r['execution_prereg_sha256']==sha(PR) and r['science_prereg_sha256']==sha(SCI) and r['codec_sha256']==sha(CODEC) and r['prompt_lock_sha256']==sha(PL),'raw':r['raw_sha256']==sha(QX) and r['raw_manifest']==manifest(QX),'resources':not r['cuda_initialized'] and all(x['available']>=2*2**30 and x['peak']<=12*2**30 for x in r['resources']) and r['artifact_bytes_before_commit']<=int(1.10*2**30)};replay=True;metrics={};quality=True;controls=[]
 with safe_open(RX,framework='pt',device='cpu') as rf,safe_open(QX,framework='pt',device='cpu') as qf,BANK.open('rb') as h,torch.inference_mode():
  for p in range(4):
   x=rf.get_tensor(f'p{p}_mlp_input').reshape(16,2048);ids=rf.get_tensor(f'p{p}_router_ids');w=rf.get_tensor(f'p{p}_router_weights');gate=rf.get_tensor(f'p{p}_shared_gate').reshape(16,1);res=rf.get_tensor(f'p{p}_pre_mlp_residual');getter=lambda e,j:get(h,e,j);qr,qs=graph(x,ids,w,getter);vals={'routed':qr,'shared_raw':qs,'shared_gated':gate*qs,'complete_mlp':qr+gate*qs,'layer':torch.add(res,(qr+gate*qs).reshape(1,16,2048))};metrics[str(p)]={}
   for k,v in vals.items():stored=qf.get_tensor(f'p{p}_q5_{k}');replay &= torch.equal(v,stored);a=qf.get_tensor(f'p{p}_source_{k}');metrics[str(p)][k]=[metric(a[:,n:n+1] if a.ndim==3 else a[n:n+1],stored[:,n:n+1] if stored.ndim==3 else stored[n:n+1]) for n in range(8,16)];quality &= all(q['rel_l2']<=(.02 if k=='layer' else .08) and (k!='layer' or(q['cosine']>=.999 and q['max_abs']<=.125)) for q in metrics[str(p)][k])
   for row in [z for z in r['controls'] if z['prompt']==p]:
    name=row['control'];n=row['position'];changed=ids.clone();mut=None
    if name in ('wrong_expert','fixed_boundary_identity'):changed[n,0]=row['presented_expert']
    if name=='projection_swap':
     def cg(e,j):return getter(e,1) if e==row['presented_expert'] and j==0 else getter(e,j)
    elif name=='code_mutation':
     mut=getter(512,2).clone();rr=row['matrix_row'];cc=row['matrix_column'];h.seek((512*3+2)*MB);rec=bytearray(h.read(MB));block=(rr*512+cc)//8;slot=(rr*512+cc)%8;word=int.from_bytes(rec[64+block*5:64+block*5+5],'little');word=(word&~(31<<(5*slot)))|((row['mutated_q']+15)<<(5*slot));rec[64+block*5:64+block*5+5]=word.to_bytes(5,'little');safe=False
     try:decode(bytes(rec),512,2)
     except Exception:safe=True
     def cg(e,j):return idecode(bytes(rec[64:64+CB]),bytes(rec[64+CB:64+CB+SB]),2048,512) if e==512 and j==2 else getter(e,j)
    else:cg=getter
    if name!='code_mutation':
     h.seek((row['presented_expert']*3+row['presented_projection'])*MB);rec=h.read(MB);safe=False
     try:decode(rec,row['requested_expert'],row['requested_projection'])
     except Exception:safe=True
    cr,cs=graph(x,changed,w,cg);out=cr+gate*cs;stored=qf.get_tensor(row['raw_key']);baseline=qf.get_tensor(f'p{p}_q5_complete_mlp')[n:n+1];controls.append(safe and torch.equal(out[n:n+1],stored) and bool((stored!=baseline).any()))
 checks['bank_graph_replay']=bool(replay);checks['metrics']=metrics==r['metrics'];checks['quality']=bool(quality and sum(x['rel_l2'] for p in metrics.values() for x in p['layer'])/32<=.01);checks['controls_independent']=len(controls)==32 and all(controls);return {'kind':'port80b_t0q5r4_independent_verification','pass':all(checks.values()),'checks':checks}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('reference','q5'),required=True);a=p.parse_args();o=reference() if a.phase=='reference' else q5();print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
