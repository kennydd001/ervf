#!/usr/bin/env python3
"""Static PV0-R3 preflight: no D2/shard/model/device import or access."""
from __future__ import annotations
import ast, hashlib, json, os, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';LOCK=R/'het_next_l0_pv0r3_runner_lock.json';VL=R/'het_next_l0_pv0r3_verifier_lock.json';PL=R/'het_next_l0_pv0r3_preflight_lock.json';OUT=ROOT/'reports/runs/streamq5_moe/het_next_l0_pv0r3_real_weight_process_validation'
FILES={'prereg':R/'HET_NEXT_L0_PV0R3_REAL_WEIGHT_PROCESS_VALIDATION_PREREGISTRATION_2026-08-13.md','design':R/'HET_NEXT_L0_PV0R3_IMPLEMENTATION_DESIGN_2026-08-13.md','manifest':R/'het_next_l0_pv0r2_selected_source_manifest.json','builder':S/'build_het_next_l0_pv0r3_source_oracle.py','intel':S/'run_het_next_l0_pv0r3_intel_child.py','nvidia':S/'run_het_next_l0_pv0r3_nvidia_child.py','coordinator':S/'run_het_next_l0_pv0r3_coordinator.py','verifier':S/'verify_het_next_l0_pv0r3_independent.py','runner_lock':LOCK,'verifier_lock':VL}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frame(x):
 b=json.dumps(x,sort_keys=True,separators=(',',':')).encode();return len(b).to_bytes(8,'little')+b
def parse(b):
 if len(b)<8:raise EOFError
 n=int.from_bytes(b[:8],'little')
 if n>1<<20 or len(b)!=8+n:raise ValueError
 return json.loads(b[8:])
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());p=json.loads(PL.read_text());checks={}
 checks['closed']=l['execution_open'] is False and v['execution_open'] is False and l['audit_token']==v['audit_token']=='PENDING_INDEPENDENT_SOURCE_AUDIT'
 checks['hashes']=all(l[k+'_sha256']==sha(path) for k,path in FILES.items() if k not in ('runner_lock','verifier_lock')) and l['verifier_lock_sha256']==sha(VL)
 checks['preflight_self']=p['preflight_sha256']==sha(__file__) and p['runner_lock_sha256']==sha(LOCK)
 checks['output_absent']=not OUT.exists()
 forbidden=('transformers','cupy','pyopencl','safetensors')
 tree=ast.parse(Path(__file__).read_text());imports={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names};checks['static_imports']=not imports.intersection(forbidden)
 x={'type':'ready','nonce':'n','role':'intel','seq':0};checks['frame']=parse(frame(x))==x
 negatives=0
 for b in (b'1', (2<<20).to_bytes(8,'little'), (5).to_bytes(8,'little')+b'x'):
  try:parse(b)
  except Exception:negatives+=1
 checks['frame_negatives']=negatives==3
 vals=[14106624,14106624,65536,32,32,11264,11264,11264,11264,45056,65536,32,28672,221184,28704];checks['allocation_sum']=sum(vals)==28713088 and 90112+131072==221184 and 24576+4096+32==28704
 with tempfile.TemporaryDirectory() as td:
  q=Path(td)/'x';t=Path(td)/'x.inprogress';t.write_bytes(b'x');os.rename(t,q);checks['transaction']=q.read_bytes()==b'x' and not t.exists()
 checks['pass']=all(checks.values());print(json.dumps({'kind':'het_next_l0_pv0r3_static_preflight','checks':checks,'physical_actions':False},sort_keys=True));return 0 if checks['pass'] else 3
if __name__=='__main__':raise SystemExit(main())
