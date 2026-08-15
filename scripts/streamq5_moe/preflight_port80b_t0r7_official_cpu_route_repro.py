#!/usr/bin/env python3
"""No-forward CPU/source preflight for T0-R7."""
import hashlib,json,py_compile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";RUNNER=ROOT/"scripts/streamq5_moe/run_port80b_t0r7_official_cpu_route_repro.py";VERIFIER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r7_official_cpu_route_repro.py";LOCK=R/"port80b_t0r7_runner_lock.json";VLOCK=R/"port80b_t0r7_verifier_lock.json";REF=ROOT/".venv-next-ref/Scripts/python.exe";OUT=R/"port80b_t0r7_cpu_preflight.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""):h.update(b)
 return h.hexdigest()
def main():
 l=json.loads(LOCK.read_text());v=json.loads(VLOCK.read_text());checks={"runner_bound":sha(RUNNER)==l["runner_sha256"],"verifier_bound":sha(VERIFIER)==l["verifier_sha256"]==v["verifier_sha256"],"vlock_bound":sha(VLOCK)==l["verifier_lock_sha256"],"prompt_bound":sha(R/"port80b_t0r7_prompt_lock.json")==l["prompt_lock_sha256"],"prereg_bound":sha(R/"PORT80B_T0R7_OFFICIAL_CPU_ROUTE_REPRO_PREREGISTRATION_2026-08-13.md")==l["prereg_sha256"]}
 for p in (RUNNER,VERIFIER):py_compile.compile(str(p),doraise=True)
 s=RUNNER.read_text();checks["runner_contract"]=all(x in s for x in ["choices=(\"smoke\", \"capture\")","router_boundary_tie_mask","router_selected_boundary_mask","capture_complete_pending_independent_compare","stream_records(layer, run_index, peak)","MIN_START_RAM = 16","prefix_cache = DynamicCache"])
 s=VERIFIER.read_text();checks["verifier_contract"]=all(x in s for x in ["def compare()","route_ids_weights_bitwise","tie_classification_bitwise","cross_backend_not_claimed","bank_fully_reconstructed","cache_recomputed"])
 p=subprocess.run([str(REF),str(VERIFIER),"--phase","preflight"],capture_output=True,text=True,timeout=60);checks["verifier_preflight"]=p.returncode==0 and json.loads(p.stdout)["pass"]
 result={"kind":"port80b_t0r7_cpu_preflight","pass":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"claim_boundary":"No-forward compile/source/provenance preflight only; no model, bank, GPU or capture execution."};OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0 if result["pass"] else 2
if __name__=="__main__":raise SystemExit(main())
