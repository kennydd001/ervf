#!/usr/bin/env python3
"""CPU-only D2 source/lock preflight; never loads model or runs a forward."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';RUN=ROOT/'scripts/streamq5_moe/run_port80b_t0r12d2_full_stage_diagnostic.py';VER=ROOT/'scripts/streamq5_moe/verify_port80b_t0r12d2_full_stage_diagnostic.py';LOCK=R/'port80b_t0r12d2_runner_lock.json';VL=R/'port80b_t0r12d2_verifier_lock.json';D=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2_full_stage_diagnostic'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VL.read_text());actual={'runner_sha256':sha(RUN),'verifier_sha256':sha(VER),'verifier_lock_sha256':sha(VL),'prereg_sha256':sha(R/'PORT80B_T0R12D2_FULL_STAGE_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md'),'base_sha256':sha(ROOT/'scripts/streamq5_moe/run_port80b_t0r12_official_cpu_reference_only.py'),'failure_sha256':sha(ROOT/'reports/runs/streamq5_moe/port80b_t0r12_official_cpu_reference_only/t0r12_capture_1_failure.json')}
 q=subprocess.run([sys.executable,str(RUN),'--phase','lockcheck'],capture_output=True,text=True);tree=ast.parse(RUN.read_text());names={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};c={'bindings':all(l.get(k)==x for k,x in actual.items()),'verifier_lock':v.get('verifier_sha256')==actual['verifier_sha256'],'exact_lockcheck':q.returncode==0 and json.loads(q.stdout)['pass'],'output_absent':not D.exists(),'diagnostic_functions':{'route_evidence','retain','all_metrics','run'}.issubset(names),'no_bank_q5_gpu':all(x not in RUN.read_text() for x in ('stream_records','quantize_source','cuda('))};out={'kind':'port80b_t0r12d2_preflight','pass':all(c.values()),'checks':c};print(json.dumps(out,indent=2));return 0 if out['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
