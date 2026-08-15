#!/usr/bin/env python3
"""R5 runner scaffold: R4 numerics, R5 strict execution overrides. Initially blocked."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';D=ROOT/'reports/runs/streamq5_moe/port80b_t0q5r5_auditable_quality';R4=S/'run_port80b_t0q5r4_auditable_quality.py';TX=S/'port80b_t0q5r5_transaction.py';PV=S/'verify_port80b_t0q5r5_prompts.py';VER=S/'verify_port80b_t0q5r5_auditable_quality.py';PR=R/'PORT80B_T0Q5R5_EXECUTION_PREREGISTRATION_2026-08-13.md';LOCK=R/'port80b_t0q5r5_runner_lock.json';VL=R/'port80b_t0q5r5_verifier_lock.json';PL=R/'port80b_t0q5r5_prompt_lock.json'
def load(p,n):q=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(q);sys.modules[n]=m;q.loader.exec_module(m);return m
x=load(TX,'r5tx');r4=load(R4,'r5base');torch=r4.torch
REFX=D/'reference_raw.safetensors';REFR=D/'reference_result.json';REFC=D/'reference_commit.json';REFV=D/'reference_verification.json';BANK=D/'layer0_bank.sq5m';MAN=D/'bank_manifest.json';BC=D/'bank_commit.json';QX=D/'q5_raw.safetensors';QR=D/'q5_result.json';QC=D/'q5_commit.json';FAILED=D/'failed_attempts'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lockcheck():
 l=json.loads(LOCK.read_text());a={'runner_sha256':sha(__file__),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(PR),'transaction_sha256':sha(TX),'prompt_verifier_sha256':sha(PV),'r4_source_sha256':sha(R4),'prompt_lock_sha256':sha(PL) if PL.exists() else '__ABSENT_PENDING_IMPLEMENTATION_AUDIT__'};return {'pass':PL.exists() and all(l.get(k)==v for k,v in a.items()),'bindings':a}
def committed_bank():return r4.BANK.exists() and r4.MAN.exists() and r4.BC.exists() and x.verify(r4.BC,(r4.BANK,r4.MAN))
def strict_guard(phase):
 if not lockcheck()['pass']:raise RuntimeError('R5 lock')
 proc=r4.psutil.Process();want=json.loads(r4.b.DEPENDENCY_LOCK.read_text())['runtime']['process_affinity'];proc.cpu_affinity(want);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.mkldnn.enabled=True;torch.set_flush_denormal(False)
 if phase=='q5' and committed_bank():return 'reuse_verified_committed_bank'
 return 'fresh'
def install():
 for n,v in {'D':D,'LOCK':LOCK,'VL':VL,'PL':PL,'VER':VER,'PR':PR,'REFX':REFX,'REFR':REFR,'REFC':REFC,'REFV':REFV,'BANK':BANK,'MAN':MAN,'BC':BC,'QX':QX,'QR':QR,'QC':QC,'FAILED':FAILED}.items():setattr(r4,n,v)
 r4.lockcheck=lockcheck
 original_sha=r4.sha
 r4.sha=lambda p:sha(__file__) if Path(p).resolve()==R4.resolve() else original_sha(p)
 original_canon=r4.canon
 def result_canon(obj):
  if isinstance(obj,dict) and obj.get('kind') in ('port80b_t0q5r4_reference','port80b_t0q5r4_q5'):
   obj=dict(obj);obj['kind']=obj['kind'].replace('t0q5r4','t0q5r5');obj['runner_sha256']=sha(__file__);obj['runtime_dependencies']=r4.b.runtime_dependencies();obj['runtime_contract']={'affinity':r4.psutil.Process().cpu_affinity(),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),'mkldnn':torch.backends.mkldnn.enabled,'matmul_precision':torch.get_float32_matmul_precision(),'flush_denormal':False,'cuda_initialized':torch.cuda.is_initialized()}
  return original_canon(obj)
 r4.canon=result_canon
 original_guard=r4.guard
 def guard(phase):
  mode=strict_guard(phase)
  if phase=='q5' and mode=='reuse_verified_committed_bank':
   if any(z.exists() for z in (QX,QR,QC)):raise FileExistsError('q5 target')
   if not REFV.exists():raise RuntimeError('verified reference absent')
   return
  return original_guard(phase)
 r4.guard=guard
 states={}
 def bundle(files,marker,kind):
  name=marker.stem.replace('_commit','');state,j=x.begin(D,name,files);states[state['nonce']]=(state,j);tmps={Path(k):Path(v) for k,v in state['files'].items()};return state['nonce'],tmps,Path(marker).with_name(Path(marker).name+'.'+state['nonce']+'.inprogress'),state
 def promote(tmps,_marker_tmp,_marker,state):x.commit(state,states[state['nonce']][1])
 r4.commit_bundle=bundle;r4.promote=promote
 def quarantine(phase):
  moved={}
  for name in ('reference','bank','q5'):moved.update(x.recover(D,name,FAILED))
  return moved
 r4.quarantine=quarantine
 original_build=r4.build
 def build(samples,tmpbank):
  if committed_bank():return json.loads(MAN.read_text())['manifest']
  return original_build(samples,tmpbank)
 r4.build=build
install();reference=r4.reference;q5=r4.q5
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lockcheck','reference','q5'),required=True);p.add_argument('--ack');a=p.parse_args()
 if a.phase=='lockcheck':print(json.dumps({'kind':'t0q5r5_lockcheck',**lockcheck(),'physical_actions':False}));return 0
 strict_guard(a.phase);return (reference() if a.phase=='reference' else q5())
if __name__=='__main__':raise SystemExit(main())
