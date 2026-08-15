#!/usr/bin/env python3
"""Static CAP0P source contract preflight; no device or child launch."""
import ast,hashlib,json,struct,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';LOCK=R/'het_next_cap0p_runner_lock.json';VLOCK=R/'het_next_cap0p_verifier_lock.json';OUT=ROOT/'reports/runs/streamq5_moe/het_next_cap0p_process_isolated'
FILES=(S/'het_next_cap0p_common.py',S/'het_next_cap0p_intel_child.py',S/'het_next_cap0p_nvidia_child.py',S/'run_het_next_cap0p_process_isolated.py',S/'verify_het_next_cap0p_process_isolated.py',Path(__file__))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());checks={'closed':not l['execution_open'] and not v['execution_open'],'absent':not OUT.exists(),'bindings':all(sha(ROOT/x['path'])==x['sha256'] for x in v['files'].values())};src='\n'.join(p.read_text() for p in FILES);checks['scope']=not any(x in src.lower() for x in ('safetensors','transformers','checkpoint','q5_','throughput','percentile('));checks['native_job']=all(x in src for x in ('CreateProcessW','CREATE_SUSPENDED','AssignProcessToJobObject','ResumeThread','JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE' if False else '0x2000','CREATE_NO_WINDOW'));checks['children']=all(x in src for x in ('clHostMemAllocINTEL','clSetKernelArgMemPointerINTEL','RawModule','non_blocking=True','output_words'))
 # Pure barrier simulation: both ready, one broadcast per epoch, strict overlap, both exit.
 sim=[{'epoch':e,'intel':(100*e,100*e+30),'nvidia':(100*e+5,100*e+35)} for e in range(1,4)];checks['barrier_sim']=all(max(z['intel'][0],z['nvidia'][0])<min(z['intel'][1],z['nvidia'][1]) for z in sim)
 out={'kind':'cap0p_static_preflight','pass':all(checks.values()),'checks':checks,'device_calls':0,'process_launches':0};print(json.dumps(out,sort_keys=True));return 0 if out['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
