#!/usr/bin/env python3
"""R5 independent verifier: R4 replay plus all-record source/manifest audit and selector checks."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5r5_auditable_quality';V4=S/'verify_port80b_t0q5r4_auditable_quality.py';TX=S/'port80b_t0q5r5_transaction.py';PV=S/'verify_port80b_t0q5r5_prompts.py';RUN=S/'run_port80b_t0q5r5_auditable_quality.py';SELF=Path(__file__);PL=R/'port80b_t0q5r5_prompt_lock.json';VL=R/'port80b_t0q5r5_verifier_lock.json';SHARD=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors';BANK=D/'layer0_bank.sq5m';MAN=D/'bank_manifest.json';RX=D/'reference_raw.safetensors';RR=D/'reference_result.json';RC=D/'reference_commit.json';RV=D/'reference_verification.json';QX=D/'q5_raw.safetensors';QR=D/'q5_result.json';QC=D/'q5_commit.json'
def load(p,n):q=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(q);sys.modules[n]=m;q.loader.exec_module(m);return m
v=load(V4,'v4ind');tx=load(TX,'txind');torch=v.torch;F=v.F;safe_open=v.safe_open
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iq(value):
 import numpy as np
 r,k=value.shape;w=value.float().reshape(r,k//128,128);mx=w.abs().amax(-1,keepdim=True);s=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);f=(q.to(torch.int16)+15).numpy().astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).view(torch.uint16).numpy().astype('<u2',copy=False).tobytes();return codes,scales
def all1539():
 env=json.loads(MAN.read_text());core=env['manifest'];ok=env['manifest_sha256']==hashlib.sha256(v.canon(core)).hexdigest() and MAN.read_bytes()==v.canon(env)+b'\n' and len(core['records'])==1539 and core['bank_bytes']==1040117760 and core['bank_sha256']==sha(BANK)
 with safe_open(SHARD,framework='pt',device='cpu') as src,BANK.open('rb') as h:
  for e in range(513):
   for j in range(3):
    rec=h.read(v.MB);m=core['records'][e*3+j];source=src.get_tensor(v.key(e,j));codes,scales=iq(source);dec=v.decode(rec,e,j);ok &= m['offset']==(e*3+j)*v.MB and m['source_key']==v.key(e,j) and m['source_sha256']==hashlib.sha256(v.tb(source)).hexdigest() and rec[64:64+v.CB]==codes and rec[64+v.CB:64+v.CB+v.SB]==scales and m['decoded_weight_sha256']==hashlib.sha256(v.tb(dec)).hexdigest() and m['record_sha256']==hashlib.sha256(rec).hexdigest()
 return bool(ok)
def prompt_semantics():
 import subprocess
 q=subprocess.run([sys.executable,str(PV)],capture_output=True,text=True);return q.returncode==0 and json.loads(q.stdout)['pass']
def reference():
 v.D=D;v.RX=RX;v.RR=RR;v.RC=RC;v.RV=RV;v.RUN=RUN;v.SELF=SELF;v.PL=PL;v.VL=VL;o=v.reference();o['checks']['prompt_replay_disjoint']=prompt_semantics();o['pass']=all(o['checks'].values());return o
def selector_recompute(result):
 ok=True
 with safe_open(RX,framework='pt',device='cpu') as rf,BANK.open('rb') as h:
  for row in [x for x in result['controls'] if x['control']=='code_mutation']:
   p,n=row['prompt'],row['position'];x=rf.get_tensor(f'p{p}_mlp_input').reshape(16,2048);getter=lambda e,j:v.get(h,e,j);act=F.silu(F.linear(x,getter(512,0)))*F.linear(x,getter(512,1));h.seek((512*3+2)*v.MB);rec=h.read(v.MB);import numpy as np;codes=rec[64:64+v.CB];pp=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=pp[:,0]|pp[:,1]<<8|pp[:,2]<<16|pp[:,3]<<24|pp[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1).reshape(2048,512);q=f.astype(np.int16)-15;chosen=None
   for a in range(2048):
    for cc in range(512):
     if q[a,cc]!=0 and act[n,cc]!=0:chosen=(a,cc);break
    if chosen:break
   ok &= chosen==(row['matrix_row'],row['matrix_column'])
 return bool(ok)
def q5():
 v.D=D;v.RX=RX;v.RR=RR;v.RV=RV;v.BANK=BANK;v.MAN=MAN;v.BC=D/'bank_commit.json';v.QX=QX;v.QR=QR;v.QC=QC;v.RUN=RUN;v.SELF=SELF;v.PL=PL;v.VL=VL;o=v.q5();r=json.loads(QR.read_text());o['checks']['all1539_source_requant_manifest']=all1539();o['checks']['mutation_selector_recomputed']=selector_recompute(r);o['checks']['prompt_replay_disjoint']=prompt_semantics();o['checks']['total_artifacts']=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<=int(1.10*2**30);o['pass']=all(o['checks'].values());return o
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('reference','q5'),required=True);a=p.parse_args();o=reference() if a.phase=='reference' else q5();print(json.dumps(o,indent=2));return 0 if o['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
