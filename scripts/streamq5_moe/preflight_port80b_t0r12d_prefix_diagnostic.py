#!/usr/bin/env python3
import ast,hashlib,json,py_compile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";X=ROOT/"scripts/streamq5_moe/run_port80b_t0r12d_prefix_diagnostic.py";V=ROOT/"scripts/streamq5_moe/verify_port80b_t0r12d_prefix_diagnostic.py";L=R/"port80b_t0r12d_runner_lock.json";VL=R/"port80b_t0r12d_verifier_lock.json";PY=ROOT/".venv-next-ref/Scripts/python.exe"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*2**20),b''):h.update(z)
 return h.hexdigest()
def main():
 l=json.loads(L.read_text());c={'runner':sha(X)==l['runner_sha256'],'verifier':sha(V)==l['verifier_sha256'],'vlock':sha(VL)==l['verifier_lock_sha256'],'base':sha(ROOT/'scripts/streamq5_moe/run_port80b_t0r12_official_cpu_reference_only.py')==l['r12_runner_sha256'],'failure':sha(ROOT/'reports/runs/streamq5_moe/port80b_t0r12_official_cpu_reference_only/t0r12_capture_1_failure.json')==l['r12_failure_sha256'],'output_absent':not (ROOT/'reports/runs/streamq5_moe/port80b_t0r12d_prefix_diagnostic').exists()}
 for p in (X,V):py_compile.compile(str(p),doraise=True)
 s=X.read_text();c['protocol']=all(x in s for x in ['for n in range(1,17)','same_length_repeat','p1_whole_repeat_output','p1_prefix3_repeat_final','whole_cache_conv','prefix16_cache_conv','diagnostic_only_not_pass']) and not any(x in s for x in ['stream_records','quantize_matrix','RECORD_ARTIFACT'])
 p=subprocess.run([str(PY),str(X),'--phase','lockcheck'],capture_output=True,text=True,timeout=60);d=json.loads(p.stdout) if p.returncode==0 else {};c['lockcheck']=p.returncode==0 and d.get('pass') and not any(d['physical_actions'].values())
 r={'kind':'r12d_cpu_preflight','pass':all(c.values()),'checks':c,'claim_boundary':'No model/forward/GPU; source/lock only.'};print(json.dumps(r,indent=2));return 0 if r['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
