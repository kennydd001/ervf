#!/usr/bin/env python3
"""Closed, CPU-only static PH1 NVIDIA N1 source preflight."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import het_next_l0_ph1_nvidia_n1_transaction as tx

ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";S=ROOT/"scripts/streamq5_moe";LOCK=R/"het_next_l0_ph1_nvidia_n1_preflight_lock.json";OUT=R/"het_next_l0_ph1_nvidia_n1_static_preflight.json"
FILES=(S/"het_next_l0_ph1_nvidia_n1_common.py",S/"het_next_l0_ph1_nvidia_n1_kernels.cu",S/"het_next_l0_ph1_nvidia_n1_backend.py",S/"het_next_l0_ph1_nvidia_n1_transaction.py",S/"run_het_next_l0_ph1_nvidia_n1.py",S/"verify_het_next_l0_ph1_nvidia_n1.py",Path(__file__))
ACK="ACK-PH1-NVIDIA-N1-STATIC-PREFLIGHT"

def fsha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {p.name:p.read_text(encoding="utf-8") for p in FILES}
def calls(tree):return {n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else "" for n in ast.walk(tree) if isinstance(n,ast.Call)}
def imports(tree):
 out=set()
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):out|={x.name for x in n.names}
  if isinstance(n,ast.ImportFrom):out.add(n.module or "")
 return out
def hash_lock():
 lock=json.loads(LOCK.read_text());ok=lock.get("kind")=="het_next_l0_ph1_nvidia_n1_preflight_lock"
 for item in lock.get("bindings",{}).values():
  p=Path(item["path"]);p=p if p.is_absolute() else ROOT/p;ok=ok and p.is_file() and fsha(p)==item["sha256"]
 return lock,ok
def kernel_contract(text):
 required=("tiled_partition<8>","extern \"C\" __global__ void q5_linear","extern \"C\" __global__ void bf16_lut_activation","fmaf(","__fadd_rn","tile.shfl_down","atomicAdd(&counters[row], 1U)","multiply_bf16_exact","CODE_BYTES 655360ULL")
 forbidden=("--use_fast_math",".ftz","__exp","expf(","atomicAdd(output","<<<")
 baseline=all(x in text for x in required) and not any(x in text for x in forbidden) and text.count("extern \"C\" __global__ void")==2
 mutations=[text.replace(x,x+"_MUT",1) for x in required]
 return baseline and all(not(all(x in m for x in required) and not any(x in m for x in forbidden)) for m in mutations)
def source_contract(src):
 backend=src["het_next_l0_ph1_nvidia_n1_backend.py"];runner=src["run_het_next_l0_ph1_nvidia_n1.py"];verifier=src["verify_het_next_l0_ph1_nvidia_n1.py"]
 driver=("C.WinDLL(str(NVCUDA_DLL), use_last_error=True, winmode=0x00000800)","cuMemHostAlloc(C.byref(out), size, 0)","cuStreamCreate(C.byref(out), CU_STREAM_NON_BLOCKING)","cuModuleLoadDataEx(C.byref(out), C.cast(self.cubin_buffer, C.c_void_p), 0, None, None)","*grid, *block, 0, self.stream, params, None","cuMemGetInfo_v2","post_ordinary_releases_pre_pop")
 compiler=("nvrtcCreateProgram","nvrtcCompileProgram","nvrtcGetProgramLogSize","nvrtcGetProgramLog","nvrtcGetPTXSize","nvrtcGetPTX","nvrtcGetCUBINSize","nvrtcGetCUBIN","nvrtcDestroyProgram","--gpu-architecture=sm_120")
 independent=not ({"het_next_l0_ph1_nvidia_n1_common","het_next_l0_ph1_nvidia_n1_backend","run_het_next_l0_ph1_nvidia_n1","het_next_l0_ph1_nvidia_n1_transaction"}&imports(ast.parse(verifier)))
 main=next(n for n in ast.parse(runner).body if isinstance(n,ast.FunctionDef) and n.name=="main");names=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else "" for n in ast.walk(main) if isinstance(n,ast.Call)]
 return all(x in backend for x in driver+compiler) and independent and "validate_authorization" in names and names.index("validate_authorization")<min(i for i,x in enumerate(names) if x in {"compile_phase","physical_phase"})
def forbidden_surface(src):
 tree=ast.parse(src[Path(__file__).name]);bad_import={"torch","transformers","safetensors","cupy","pycuda"};bad_call={"WinDLL","CDLL","LoadLibrary","Popen","run","system","cuInit","nvrtcCompileProgram","from_pretrained"}
 return not(bad_import&imports(tree)) and not(bad_call&calls(tree))
def transaction_simulation():
 with tempfile.TemporaryDirectory(prefix="ph1_nvidia_n1_static_") as td:
  root=Path(td);out=root/"bundle";fail=root/"fail";q=root/"quarantine";kind="fixture";result={"kind":"fixture","status":"positive"}
  tx.publish_bundle(out,result,kind,lambda p:tx.verify_bundle(p,kind));positive=tx.verify_bundle(out,kind) and tx.clean_or_quarantine(out,fail,q,kind)=="already_complete"
  stale=root/"stale";stale.mkdir();(stale/"orphan").write_bytes(b"x")
  try:tx.clean_or_quarantine(stale,fail,q,kind);stale_rejected=False
  except RuntimeError:stale_rejected=not stale.exists() and q.exists()
  f1=tx.atomic_failure(fail,{"kind":"fixture_failure","stage":"x","error":"primary","error_type":"RuntimeError","device_opened":False,"disposition":"bounded"});f2=tx.atomic_failure(fail,{"kind":"fixture_failure","stage":"y","error":"secondary","error_type":"RuntimeError","device_opened":True,"disposition":"bounded"})
  create_new=f1!=f2 and f1.exists() and f2.exists() and len(list(fail.glob("attempt_*/failure.json")))==2
  corrupt=root/"corrupt";shutil.copytree(out,corrupt);(corrupt/"result.json").write_bytes(b"{}\n")
  return {"pass":positive and stale_rejected and create_new and not tx.verify_bundle(corrupt,kind),"positive":positive,"stale_rejected":stale_rejected,"create_new":create_new,"corrupt_rejected":not tx.verify_bundle(corrupt,kind)}
def topology():
 targets=(R/"het_next_l0_ph1_nvidia_n1_compile",R/"het_next_l0_ph1_nvidia_n1_physical",R/"het_next_l0_ph1_nvidia_n1_compile_failures",R/"het_next_l0_ph1_nvidia_n1_physical_failures",R/"het_next_l0_ph1_nvidia_n1_quarantine",OUT)
 return all(not p.exists() for p in targets) and not any(R.glob("het_next_l0_ph1_nvidia_n1*.inprogress*"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--ack",required=True);a=p.parse_args();lock,hashes=hash_lock()
 if lock.get("preflight_open") is not True or lock.get("preflight_token")!=ACK or a.ack!=ACK:raise PermissionError("preflight_closed")
 src=sources();simulation=transaction_simulation();checks={"hash_bindings":hashes,"output_absent":topology(),"kernel_contract":kernel_contract(src["het_next_l0_ph1_nvidia_n1_kernels.cu"]),"source_contract":source_contract(src),"forbidden_surface":forbidden_surface(src),"transaction_simulation":simulation["pass"],"transaction_details":all(simulation.values()),"lock_closed_physical":lock.get("compiler_open") is False and lock.get("physical_open") is False,"design_bindings":all(x in lock.get("bindings",{}) for x in ("n1_prereg","n1_design","n1_audit","r1_contract","r1_audit","r2_contract","r2_audit","intel_final")),"cardinality_literals":all(x in "\n".join(src.values()) for x in ("2_185_216","675840","post_ordinary_releases_pre_pop","range(30)","len(rows) != 22"))}
 result={"kind":"het_next_l0_ph1_nvidia_n1_static_preflight","checks":checks,"pass":all(checks.values()),"passed":sum(checks.values()),"total":len(checks),"no_payload":True,"no_compiler":True,"no_device":True}
 if OUT.exists():raise FileExistsError(OUT)
 OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0 if result["pass"] else 3
if __name__=="__main__":raise SystemExit(main())
