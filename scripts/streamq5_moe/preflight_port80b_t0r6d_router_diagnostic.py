#!/usr/bin/env python3
"""CPU/source preflight only for R6-D; never invokes diagnostic."""
import hashlib,json,py_compile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";RUNNER=ROOT/"scripts/streamq5_moe/run_port80b_t0r6d_router_diagnostic.py";VERIFIER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r6d_router_diagnostic.py";LOCK=R/"port80b_t0r6d_runner_lock.json";VLOCK=R/"port80b_t0r6d_verifier_lock.json";REF=ROOT/".venv-next-ref/Scripts/python.exe";OUT=R/"port80b_t0r6d_cpu_preflight.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""):h.update(b)
 return h.hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());checks={"runner_lock":sha(RUNNER)==l["runner_sha256"],"verifier_lock":sha(VERIFIER)==l["verifier_sha256"]==v["verifier_sha256"],"vlock_bound":sha(VLOCK)==l["verifier_lock_sha256"],"r5_failure_bound":sha(ROOT/"reports/runs/streamq5_moe/port80b_t0r4r5_official_layer0/t0r4r5_run_1_failure.json")==l["r5_failure_sha256"]}
 for p in (RUNNER,VERIFIER):py_compile.compile(str(p),doraise=True)
 s=RUNNER.read_text();required=["for r in range(16)","boundary_tie_expert_ids","selected_boundary_subset","strict_margin_negative","native_bf16_logit","u16_bits","\"bank_built\":False","output=layer(hidden"]
 checks["source_contract"]=all(x in s for x in required);p=subprocess.run([str(REF),str(VERIFIER),"--phase","preflight"],capture_output=True,text=True,timeout=60);checks["verifier_preflight"]=p.returncode==0 and json.loads(p.stdout)["pass"]
 result={"kind":"port80b_t0r6d_cpu_preflight","pass":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"claim_boundary":"Compile/source/provenance only; no forward, bank, GPU, or pass."};OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0 if result["pass"] else 2
if __name__=="__main__":raise SystemExit(main())
