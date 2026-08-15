#!/usr/bin/env python3
"""No-forward R8 source/lock preflight. Execution remains closed."""
import hashlib,json,py_compile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";RUNNER=ROOT/"scripts/streamq5_moe/run_port80b_t0r8_official_cpu_route_repro.py";VERIFIER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r8_official_cpu_route_repro.py";LOCK=R/"port80b_t0r8_runner_lock.json";VLOCK=R/"port80b_t0r8_verifier_lock.json";REF=ROOT/".venv-next-ref/Scripts/python.exe";OUT=R/"port80b_t0r8_cpu_preflight.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""):h.update(b)
 return h.hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());checks={"runner":sha(RUNNER)==l["runner_sha256"],"verifier":sha(VERIFIER)==l["verifier_sha256"]==v["verifier_sha256"],"vlock":sha(VLOCK)==l["verifier_lock_sha256"],"prompt_generator":sha(ROOT/"scripts/streamq5_moe/generate_port80b_t0r8_prompts.py")==json.loads((R/"port80b_t0r8_prompt_generation_lock.json").read_text())["generator_sha256"]}
 for p in (RUNNER,VERIFIER):py_compile.compile(str(p),doraise=True)
 s=RUNNER.read_text();checks["direct_tuple"]=all(x in s for x in ["official_router_logits","official_router_weights","official_router_ids","direct official router tuple differs"]);checks["invocation_ledger"]=all(x in s for x in ["process_create_time_unix","start_utc","start_perf_counter_ns","commandline","uuid.uuid4","capture1_scientific_outputs_read_permission"]);checks["whole_prefix16_cache"]=all(x in s for x in ["whole_cache_conv","whole_cache_recurrent","whole/prefix16 cache state mismatch"])
 s=VERIFIER.read_text();checks["independent_verifier"]=all(x in s for x in ["direct_official_router_tuple","whole_prefix16_cache_equal","distinct_invocations","capture2_no_scientific_read_permission"])
 p=subprocess.run([str(REF),str(RUNNER),"--phase","lockcheck"],capture_output=True,text=True,timeout=60);d=json.loads(p.stdout) if p.returncode==0 else {};checks["exact_lockcheck_no_model"]=p.returncode==0 and d.get("pass") and d["physical_actions"]=={"model_loaded":False,"forward":False,"bank_build":False,"gpu":False}
 p=subprocess.run([str(REF),str(VERIFIER),"--phase","preflight"],capture_output=True,text=True,timeout=60);checks["verifier_preflight"]=p.returncode==0 and json.loads(p.stdout)["pass"]
 result={"kind":"port80b_t0r8_cpu_preflight","pass":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"claim_boundary":"No model/forward/bank/GPU. Source and execution provenance only."};OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0 if result["pass"] else 2
if __name__=="__main__":raise SystemExit(main())
