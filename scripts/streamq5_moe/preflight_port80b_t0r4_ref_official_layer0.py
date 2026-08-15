#!/usr/bin/env python3
"""No-forward CPU preflight for immutable PORT80B T0-R4-REF."""
from __future__ import annotations

import hashlib, json, os, py_compile, struct, subprocess, sys
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
SNAP = Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f"
FILES = {
 R/"PORT80B_T0R4_OFFICIAL_LAYER0_REFERENCE_PREREGISTRATION_2026-08-13.md": "4c5da965e47e11e9ff36594e15387c48a1a2943aa0d9cbdc3b9b271418400a03",
 R/"PORT80B_T0P4_OFFICIAL_LAYER0_PHYSICAL_PREREGISTRATION_2026-08-13.md": "ff544820774fb9655f08b02756b0d36b8c5bb8f86e12f7a2173fe29db1942e2d",
 R/"port80b_t0r1_prompt_lock.json": "f283da7e86adf915431459b08aac967d9c18c3de155699c369f5a55be20e5f34",
 R/"port80b_t0r1_reference_environment_lock.json": "eb31d4e0c1f6a806434ea8a20b6b00200781a89ed9f91e485aad0e3583c0f455",
 R/"port80b_t0r4_dependency_execution_lock.json": "1d08457aded09f139d25af84ba778d8e275ab5ff71967a3dc8b9a7452e6d2fae",
 R/"port80b_t0r4_runner_lock.json": "LOCK_HASH_SELF_EXCLUDED",
 ROOT/"scripts/streamq5_moe/run_port80b_t0r4_official_layer0_reference.py": "a47a40eafc30c988c7c918e67896bdfb42c689739db30626d211f601864f66f6",
 SNAP/"model.safetensors.index.json": "e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b",
 SNAP/"model-00001-of-00040.safetensors": "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a",
}
RUNNER = ROOT/"scripts/streamq5_moe/run_port80b_t0r4_official_layer0_reference.py"
OUT = R/"port80b_t0r4_ref_cpu_preflight.json"
REF = ROOT/".venv-next-ref/Scripts/python.exe"

def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*2**20),b""): d.update(b)
 return d.hexdigest()

def main():
 checks=[]
 def ck(name,value,detail=None): checks.append({"name":name,"pass":bool(value),"detail":detail})
 exact={str(p):sha(p) if p.is_file() else None for p in FILES}
 ck("immutable_inputs_exact",all(p.is_file() and (e=="LOCK_HASH_SELF_EXCLUDED" or sha(p)==e) for p,e in FILES.items()),exact)
 ck("runner_lock_binds_runner",json.loads((R/"port80b_t0r4_runner_lock.json").read_text())["runner_sha256"]==sha(RUNNER))
 py_compile.compile(str(RUNNER),doraise=True); ck("runner_py_compile",True)
 source=RUNNER.read_text(encoding="utf-8")
 required=["USE_HUB_KERNELS\"] = \"0", "torch.set_num_interop_threads(1)", "materialize=False",
           "for length in range(1, 17)", "fp32_logits = native_logits.float()", "prefix_cache = DynamicCache(config=config)",
           "max_bf16_ulp", "projected > MAX_PROJECTED_STEADY", "field_31_rejected", "all-zero codes 0"]
 # zero-group behavior is implemented by where(maximum > 0, maximum/15, ones), then zero/scale smoke is separately required before reference.
 present={item:item in source for item in required[:-1]}; present[required[-1]]="torch.where(maximum > 0" in source and "torch.ones_like(maximum)" in source
 ck("runner_source_contract",all(present.values()),present)
 shard=SNAP/"model-00001-of-00040.safetensors"
 with shard.open("rb") as f: n=struct.unpack("<Q",f.read(8))[0]; header=json.loads(f.read(n))
 entries={k:v for k,v in header.items() if k!="__metadata__"}; spans=sorted((*v["data_offsets"],k) for k,v in entries.items())
 header_ok=len(entries)==1567 and all(v["dtype"]=="BF16" for v in entries.values()) and all(spans[i-1][1]==spans[i][0] for i in range(1,len(spans))) and spans[0][0]==0 and spans[-1][1]==shard.stat().st_size-8-n
 ck("independent_shard_header",header_ok,{"header_bytes":n,"entries":len(entries),"payload_bytes":spans[-1][1]})
 dep=json.loads((R/"port80b_t0r4_dependency_execution_lock.json").read_text())
 t=ROOT/".venv-next-ref/Lib/site-packages/transformers"
 dep_ok=all(sha(t/name)==digest for name,digest in dep["transformers_sources"].items())
 ck("dependency_source_hashes",dep_ok)
 projected=3_919_393_152*2+1_040_117_760+288_358_400
 ck("resource_gate",psutil.virtual_memory().available>=16*2**30 and projected<=int(10.5*2**30),{"available":psutil.virtual_memory().available,"projected":projected})
 env=dict(os.environ); env.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","USE_HUB_KERNELS":"0","CUDA_VISIBLE_DEVICES":"-1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
 p=subprocess.run([str(REF),str(RUNNER),"--phase","smoke"],capture_output=True,text=True,env=env,timeout=180)
 smoke=json.loads(p.stdout) if p.returncode==0 else {"stderr":p.stderr,"stdout":p.stdout}
 ck("meta_only_no_forward_smoke",p.returncode==0 and smoke.get("pass") and smoke.get("physical_actions",{}).get("reference_forward") is False,smoke)
 fail=[x for x in checks if not x["pass"]]
 result={"kind":"port80b_t0r4_ref_cpu_preflight","pass":not fail,"status":"r4_ref_cpu_preflight_pass_execution_still_closed" if not fail else "blocked","checks_passed":len(checks)-len(fail),"checks_total":len(checks),"blocked_reasons":[x["name"] for x in fail],"checks":checks,"preflight_sha256":sha(Path(__file__)),"runner_lock_sha256":sha(R/"port80b_t0r4_runner_lock.json"),"claim_boundary":"No-forward CPU provenance/source/meta-only smoke. No official tensor payload loaded into a model, no model forward, bank build, GPU, registration or registry edit. R4-Q5 and independent verifier remain unwritten/closed."}
 OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps({k:result[k] for k in ("status","pass","checks_passed","checks_total","blocked_reasons")},indent=2)); return 0 if not fail else 2
if __name__=="__main__": raise SystemExit(main())
