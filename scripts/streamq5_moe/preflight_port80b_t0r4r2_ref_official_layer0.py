#!/usr/bin/env python3
"""No-forward CPU preflight for immutable PORT80B T0-R4-REF-R2."""
from __future__ import annotations
import hashlib, json, os, py_compile, subprocess
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/"reports/streamq5_moe"
RUNNER=ROOT/"scripts/streamq5_moe/run_port80b_t0r4r2_official_layer0_reference.py"
VERIFIER=ROOT/"scripts/streamq5_moe/verify_port80b_t0r4r2_official_layer0_reference.py"
LOCK=R/"port80b_t0r4r2_runner_lock.json"; VLOCK=R/"port80b_t0r4r2_verifier_lock.json"
OUT=R/"port80b_t0r4r2_ref_cpu_preflight.json"; REF=ROOT/".venv-next-ref/Scripts/python.exe"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""): h.update(b)
 return h.hexdigest()

def main():
 checks=[]
 def ck(name,value,detail=None): checks.append({"name":name,"pass":bool(value),"detail":detail})
 lock=json.loads(LOCK.read_text()); vlock=json.loads(VLOCK.read_text())
 ck("runner_lock",sha(RUNNER)==lock["runner_sha256"] and sha(VERIFIER)==lock["verifier_sha256"] and sha(VLOCK)==lock["verifier_lock_sha256"])
 ck("verifier_lock",sha(VERIFIER)==vlock["verifier_sha256"] and vlock["schema_version"]=="PORT80B_T0R4R2_REF_V1")
 for p in (RUNNER,VERIFIER): py_compile.compile(str(p),doraise=True)
 ck("python_compile",True)
 source=RUNNER.read_text(encoding="utf-8")
 required=["runtime_dependencies()", "torch.is_autocast_enabled(\"cpu\")", "torch.is_inference_mode_enabled()",
  "[1, 8192, 4]", "[1, 32, 128, 128]", "tensor_manifest(raw)", "minimum_top10_top11_margin_fp32",
  "record_manifest_semantic_equal", "total_record_bytes", "native_weights[:, :-1] >= native_weights[:, 1:]",
  "sha256(VERIFIER)", "sha256(VERIFIER_LOCK)", "flush_denormal_nonzero_subnormal_probe"]
 present={x:x in source for x in required}; ck("runner_source_contract",all(present.values()),present)
 vsource=VERIFIER.read_text(encoding="utf-8")
 vrequired=["raw_manifest_exact","cache_schema","routes_positive_finite_nonincreasing","record_manifest_bindings","clean_replay","runtime_contract"]
 vpresent={x:x in vsource for x in vrequired}; ck("independent_verifier_source_contract",all(vpresent.values()),vpresent)
 dep=json.loads((R/"port80b_t0r4_dependency_execution_lock.json").read_text()); base=ROOT/".venv-next-ref/Lib/site-packages/transformers"
 ck("dependency_source_hashes",all(sha(base/name)==digest for name,digest in dep["transformers_sources"].items()))
 projected=3_919_393_152*2+1_040_117_760+288_358_400
 ck("resource_gate",psutil.virtual_memory().available>=16*2**30 and projected<=int(10.5*2**30),{"available":psutil.virtual_memory().available,"projected":projected})
 env=dict(os.environ); env.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","USE_HUB_KERNELS":"0","CUDA_VISIBLE_DEVICES":"-1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
 pv=subprocess.run([str(REF),str(VERIFIER),"--phase","preflight"],capture_output=True,text=True,env=env,timeout=120)
 vd=json.loads(pv.stdout) if pv.returncode==0 else {"stdout":pv.stdout,"stderr":pv.stderr}; ck("independent_verifier_preflight",pv.returncode==0 and vd.get("pass"),vd)
 ps=subprocess.run([str(REF),str(RUNNER),"--phase","smoke"],capture_output=True,text=True,env=env,timeout=240)
 sd=json.loads(ps.stdout) if ps.returncode==0 else {"stdout":ps.stdout,"stderr":ps.stderr}; ck("meta_only_no_forward_smoke",ps.returncode==0 and sd.get("pass") and sd["physical_actions"]["reference_forward"] is False,sd)
 fail=[x for x in checks if not x["pass"]]
 result={"kind":"port80b_t0r4r2_ref_cpu_preflight","pass":not fail,"status":"r4r2_ref_cpu_preflight_pass_execution_still_closed" if not fail else "blocked","checks_passed":len(checks)-len(fail),"checks_total":len(checks),"blocked_reasons":[x["name"] for x in fail],"checks":checks,"preflight_sha256":sha(Path(__file__)),"runner_sha256":sha(RUNNER),"runner_lock_sha256":sha(LOCK),"verifier_sha256":sha(VERIFIER),"verifier_lock_sha256":sha(VLOCK),"claim_boundary":"No-forward CPU provenance/source/meta-only smoke. No official forward, bank build, Q5 execution, GPU, registration or registry edit."}
 OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps({k:result[k] for k in ("status","pass","checks_passed","checks_total","blocked_reasons")},indent=2)); return 0 if not fail else 2
if __name__=="__main__": raise SystemExit(main())
