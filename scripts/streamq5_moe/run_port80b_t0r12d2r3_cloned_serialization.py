#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'scripts/streamq5_moe/run_port80b_t0r12d2_full_stage_diagnostic.py'
spec=importlib.util.spec_from_file_location('d2base_r3',SRC)
d=importlib.util.module_from_spec(spec);sys.modules['d2base_r3']=d;spec.loader.exec_module(d)
R=ROOT/'reports/streamq5_moe'
LOCK=R/'port80b_t0r12d2r3_runner_lock.json'
VER=ROOT/'scripts/streamq5_moe/verify_port80b_t0r12d2r3_cloned_serialization.py'
VL=R/'port80b_t0r12d2r3_verifier_lock.json'
PR=R/'PORT80B_T0R12D2R3_CLONED_SERIALIZATION_REPAIR_2026-08-13.md'
FAIL=R/'port80b_t0r12d2r2_shared_storage_serialization_failure.json'
OUT=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization'
ACK='T0R12D2R3_CLONED_SERIALIZATION_ONLY'

def lockcheck():
 l=json.loads(LOCK.read_text())
 a={'runner_sha256':d.b.sha256(Path(__file__)),'verifier_sha256':d.b.sha256(VER),'verifier_lock_sha256':d.b.sha256(VL),'prereg_sha256':d.b.sha256(PR),'d2_source_sha256':d.b.sha256(SRC),'failure_sha256':d.b.sha256(FAIL)}
 return {'pass':all(l.get(k)==v for k,v in a.items()),'bindings':a}

def run():
 if OUT.exists():raise FileExistsError('R3 output exists')
 if not lockcheck()['pass']:raise RuntimeError('R3 lock')
 old_save=d.save_file
 def cloned_save(tensors,path):
  return old_save({k:v.detach().clone().contiguous() for k,v in tensors.items()},path)
 d.D=OUT;d.save_file=cloned_save
 result=d.run()
 rp=OUT/'t0r12d2_raw.safetensors';jp=OUT/'t0r12d2_result.json'
 result['revision']='t0r12d2r3_cloned_serialization'
 result['runner_sha256']=d.b.sha256(Path(__file__))
 result['verifier_sha256']=d.b.sha256(VER)
 result['verifier_lock_sha256']=d.b.sha256(VL)
 result['prereg_sha256']=d.b.sha256(PR)
 result['d2_source_sha256']=d.b.sha256(SRC)
 result['serialization_failure_sha256']=d.b.sha256(FAIL)
 result['raw_sha256']=d.b.sha256(rp)
 result['serialization_repair_only']=True
 jp.write_text(json.dumps(result,indent=2)+'\n')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lockcheck','diagnostic'),required=True);p.add_argument('--acknowledge-diagnostic');a=p.parse_args()
 if a.phase=='lockcheck':print(json.dumps({'kind':'d2r3_lockcheck',**lockcheck(),'physical_actions':{'model':False,'forward':False,'gpu':False}}));return 0
 if a.acknowledge_diagnostic!=ACK:raise SystemExit('exact acknowledgement required')
 print(json.dumps({'status':run()['status']}));return 3
if __name__=='__main__':raise SystemExit(main())
