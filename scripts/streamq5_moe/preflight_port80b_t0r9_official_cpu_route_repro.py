#!/usr/bin/env python3
"""No-forward R9 source/lock preflight; do not run before independent source audit."""
import hashlib,json,py_compile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";RUNNER=ROOT/"scripts/streamq5_moe/run_port80b_t0r9_official_cpu_route_repro.py";VERIFIER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r9_official_cpu_route_repro.py";LOCK=R/"port80b_t0r9_runner_lock.json";VLOCK=R/"port80b_t0r9_verifier_lock.json";REF=ROOT/".venv-next-ref/Scripts/python.exe";OUT=R/"port80b_t0r9_cpu_preflight.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""):h.update(b)
 return h.hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());checks={"runner":sha(RUNNER)==l["runner_sha256"],"verifier":sha(VERIFIER)==l["verifier_sha256"]==v["verifier_sha256"],"vlock":sha(VLOCK)==l["verifier_lock_sha256"],"prompt_generator":sha(ROOT/"scripts/streamq5_moe/generate_port80b_t0r9_prompts.py")==l["prompt_generator_sha256"],"run_dir_absent":not (ROOT/"reports/runs/streamq5_moe/port80b_t0r9_official_cpu_route_repro").exists()}
 for p in (RUNNER,VERIFIER):py_compile.compile(str(p),doraise=True)
 s=RUNNER.read_text();checks["runner_contract"]=all(x in s for x in ["official_router_ids","entry[\"pid\"] == prior[\"pid\"]","capture1_scientific_outputs_read_permission","subprocess.run([sys.executable,str(generator)]","premodel_target_guard(run_index)","raw_manifest = tensor_manifest(raw)","bank_commit_journal.json","os.fsync","bank_commit_complete.json"])
 s=VERIFIER.read_text();checks["verifier_contract"]=all(x in s for x in ["official_router_ids\":(\"torch.int64\"","p{p}_whole_official_router_logits","distinct_invocations","ledger_bound_to_results","whole_prefix16_cache_equal"])
 p=subprocess.run([str(REF),str(RUNNER),"--phase","lockcheck"],capture_output=True,text=True,timeout=90);d=json.loads(p.stdout) if p.returncode==0 else {"stdout":p.stdout,"stderr":p.stderr};checks["exact_lockcheck_no_model"]=p.returncode==0 and d.get("pass") and d["physical_actions"]=={"model_loaded":False,"forward":False,"bank_build":False,"gpu":False}
 p=subprocess.run([str(REF),str(VERIFIER),"--phase","preflight"],capture_output=True,text=True,timeout=60);checks["verifier_preflight"]=p.returncode==0 and json.loads(p.stdout)["pass"]
 result={"kind":"port80b_t0r9_cpu_preflight","pass":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"claim_boundary":"No model/forward/bank/GPU; source and exact lockcheck only."};OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0 if result["pass"] else 2
if __name__=="__main__":raise SystemExit(main())
