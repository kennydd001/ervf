#!/usr/bin/env python3
"""No-forward R10 preflight; source only until independent audit."""
import hashlib,json,py_compile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";X=ROOT/"scripts/streamq5_moe/run_port80b_t0r10_official_cpu_route_repro.py";V=ROOT/"scripts/streamq5_moe/verify_port80b_t0r10_official_cpu_route_repro.py";L=R/"port80b_t0r10_runner_lock.json";VL=R/"port80b_t0r10_verifier_lock.json";PY=ROOT/".venv-next-ref/Scripts/python.exe";OUT=R/"port80b_t0r10_cpu_preflight.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 l=json.loads(L.read_text());checks={'runner':sha(X)==l['runner_sha256'],'verifier':sha(V)==l['verifier_sha256'],'vlock':sha(VL)==l['verifier_lock_sha256']}
 for p in (X,V):py_compile.compile(str(p),doraise=True)
 s=X.read_text();checks['state_machine']=all(x in s for x in ['verify_committed_bank','run_index == 1 and complete.exists','transaction_simulation','manifest_final_sha256','complete.with_suffix(".json.inprogress")'])
 p=subprocess.run([str(PY),str(X),'--phase','lockcheck'],capture_output=True,text=True,timeout=90);checks['lockcheck']=p.returncode==0 and json.loads(p.stdout)['pass']
 p=subprocess.run([str(PY),str(X),'--phase','transaction-sim'],capture_output=True,text=True,timeout=90);d=json.loads(p.stdout) if p.returncode==0 else {};checks['actual_temp_transaction_sim']=p.returncode==0 and d.get('pass') and d['physical_actions']=={'model_loaded':False,'forward':False,'real_bank':False,'gpu':False}
 result={'kind':'port80b_t0r10_cpu_preflight','pass':all(checks.values()),'checks':checks,'checks_passed':sum(checks.values()),'checks_total':len(checks),'claim_boundary':'TEMP transaction simulation and exact lockcheck only; no model/real bank/GPU.'};OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
