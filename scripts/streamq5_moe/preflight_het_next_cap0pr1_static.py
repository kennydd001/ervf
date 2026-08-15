#!/usr/bin/env python3
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe';LOCK=R/'het_next_cap0pr1_runner_lock.json';VLOCK=R/'het_next_cap0pr1_verifier_lock.json';OUT=ROOT/'reports/runs/streamq5_moe/het_next_cap0pr1_process_isolated';PRO=S/'het_next_cap0pr1_protocol.py';RUN=S/'run_het_next_cap0pr1_process_isolated.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());c={'closed':not l['execution_open'] and not v['execution_open'],'absent':not OUT.exists(),'bindings':all(sha(ROOT/x['path'])==x['sha256'] for x in v['files'].values())};ns={};exec(compile(PRO.read_text(),str(PRO),'exec'),ns);s=ns['simulate']();c['state_machine']=s['pass'] and all(s['negative'].values());src=RUN.read_text();c['abi']=all(x in src for x in ('SA(C.sizeof(SA),None,True)','SetHandleInformation','CreateProcessW','AssignProcessToJobObject','ResumeThread','PeekNamedPipe','ReadFile','WriteFile','GetExitCodeProcess','GetProcessTimes','WaitForMultipleObjects','TerminateJobObject','QueryInformationJobObject'));c['scope']=not any(x in src.lower() for x in ('safetensors','transformers','q5_','throughput'));o={'kind':'cap0pr1_static','pass':all(c.values()),'checks':c,'device_calls':0,'process_launches':0};print(json.dumps(o,sort_keys=True));return 0 if o['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
