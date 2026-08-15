#!/usr/bin/env python3
"""Closed, CPU-only static PH1 NVIDIA N4 source preflight."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];R=ROOT/"reports/streamq5_moe";S=ROOT/"scripts/streamq5_moe";LOCK=R/"het_next_l0_ph1_nvidia_n4_preflight_lock.json";OUT=R/"het_next_l0_ph1_nvidia_n4_static_preflight.json"
tx=None;BIG_BINDINGS={"d2_raw","official_shard1"};READ_COUNTER={"payload_bytes":0,"metadata_bytes":0}
FILES=(S/"het_next_l0_ph1_nvidia_n4_common.py",S/"het_next_l0_ph1_nvidia_n4_kernels.cu",S/"het_next_l0_ph1_nvidia_n4_backend.py",S/"het_next_l0_ph1_nvidia_n4_transaction.py",S/"run_het_next_l0_ph1_nvidia_n4.py",S/"verify_het_next_l0_ph1_nvidia_n4.py",Path(__file__))
ACK="ACK-PH1-NVIDIA-N4-STATIC-PREFLIGHT"
ABI_SOURCE={
"cuInit":("u",),"cuDriverGetVersion":("POINTER(i)",),"cuDeviceGetCount":("POINTER(i)",),"cuDeviceGet":("POINTER(i)","i"),
"cuDeviceGetName":("c_char_p","i","i"),"cuDeviceGetUuid_v2":("POINTER(CUuuid)","i"),"cuDeviceGetPCIBusId":("c_char_p","i","i"),
"cuDeviceGetAttribute":("POINTER(i)","i","i"),"cuDeviceTotalMem_v2":("POINTER(z)","i"),"cuMemGetInfo_v2":("POINTER(z)","POINTER(z)"),
"cuCtxGetCurrent":("POINTER(p)",),"cuDevicePrimaryCtxGetState":("i","POINTER(u)","POINTER(i)"),"cuDevicePrimaryCtxRetain":("POINTER(p)","i"),
"cuCtxPushCurrent_v2":("p",),"cuCtxPopCurrent_v2":("POINTER(p)",),"cuDevicePrimaryCtxRelease_v2":("i",),
"cuStreamCreate":("POINTER(p)","u"),"cuStreamSynchronize":("p",),"cuStreamDestroy_v2":("p",),
"cuModuleLoadDataEx":("POINTER(p)","p","u","POINTER(i)","POINTER(p)"),"cuModuleGetFunction":("POINTER(p)","p","c_char_p"),"cuModuleUnload":("p",),
"cuMemHostAlloc":("POINTER(p)","z","u"),"cuMemFreeHost":("p",),"cuMemAlloc_v2":("POINTER(d)","z"),"cuMemFree_v2":("d",),
"cuMemcpyHtoDAsync_v2":("d","p","z","p"),"cuMemcpyDtoHAsync_v2":("p","d","z","p"),"cuMemsetD8Async":("d","c_ubyte","z","p"),
"cuLaunchKernel":("p","u","u","u","u","u","u","u","p","POINTER(p)","POINTER(p)")}

def fsha(p):
 data=Path(p).read_bytes();READ_COUNTER["metadata_bytes"]+=len(data);return hashlib.sha256(data).hexdigest()
def load_absolute(path,name):
 spec=importlib.util.spec_from_file_location(name,Path(path).resolve());module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def bootstrap_contract_modules():
 global tx
 tx=load_absolute(S/"het_next_l0_ph1_nvidia_n4_transaction.py","het_next_l0_ph1_nvidia_n4_transaction")
 return load_absolute(S/"het_next_l0_ph1_nvidia_n4_common.py","het_next_l0_ph1_nvidia_n4_common")
def sources():return {p.name:p.read_text(encoding="utf-8") for p in FILES}
def calls(tree):return {n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else "" for n in ast.walk(tree) if isinstance(n,ast.Call)}
def imports(tree):
 out=set()
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):out|={x.name for x in n.names}
  if isinstance(n,ast.ImportFrom):out.add(n.module or "")
 return out
def hash_lock():
 lock=json.loads(LOCK.read_text());ok=lock.get("kind")=="het_next_l0_ph1_nvidia_n4_preflight_lock"
 for name,item in lock.get("bindings",{}).items():
  p=Path(item["path"]);p=p if p.is_absolute() else ROOT/p
  if name in BIG_BINDINGS:ok=ok and p.is_absolute() and p.is_file() and p.stat().st_size==item["bytes"] and len(item["sha256"])==64
  else:ok=ok and p.is_file() and fsha(p)==item["sha256"]
 return lock,ok and READ_COUNTER["payload_bytes"]==0
def kernel_contract(text):
 required=("tiled_partition<8>","extern \"C\" __global__ void q5_linear","extern \"C\" __global__ void bf16_lut_activation","fmaf(","__fadd_rn","tile.shfl_down","atomicAdd(&counters[row], 1U)","multiply_bf16_exact","CODE_BYTES 655360ULL")
 forbidden=("--use_fast_math",".ftz","__exp","expf(","atomicAdd(output","<<<")
 baseline=all(x in text for x in required) and not any(x in text for x in forbidden) and text.count("extern \"C\" __global__ void")==2
 mutations=[text.replace(x,x+"_MUT",1) for x in required]
 return baseline and all(not(all(x in m for x in required) and not any(x in m for x in forbidden)) for m in mutations)
def function(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def method(tree,cls,name):return next(n for n in function(tree,cls).body if isinstance(n,ast.FunctionDef) and n.name==name)
def abi_contract(text):
 tree=ast.parse(text);node=method(tree,"DriverBackend","_bind");table=None
 for n in ast.walk(node):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="table" for t in n.targets):table=n.value
 if not isinstance(table,ast.Dict):return False
 names=[k.value for k in table.keys if isinstance(k,ast.Constant)];observed={}
 for key,value in zip(table.keys,table.values):
  if not isinstance(key,ast.Constant) or not isinstance(value,ast.Tuple) or len(value.elts)!=2 or not isinstance(value.elts[0],ast.List):return False
  def norm(node):return ast.unparse(node).replace("C.","")
  observed[key.value]=tuple(norm(x) for x in value.elts[0].elts)
 attrs=[n.attr for n in ast.walk(node) if isinstance(n,ast.Attribute)]
 return observed==ABI_SOURCE and len(names)==30 and attrs.count("argtypes")==1 and attrs.count("restype")==1 and "c_uint64" in attrs and "c_int" in attrs and "c_size_t" in attrs and all(isinstance(v,ast.Tuple) and ast.unparse(v.elts[1])=="i" for v in table.values)
def abi_mutations(text):
 if not abi_contract(text):return False
 mutations=[]
 for name in ABI_SOURCE:
  mutations.append(text.replace('"'+name+'":', '"MUT_'+name+'":',1))
 mutations.append(text.replace("function.restype = result","function.restype = C.c_void_p",1))
 mutations.append(text.replace("p, u, i, z, d = C.c_void_p, C.c_uint, C.c_int, C.c_size_t, C.c_uint64","p, u, i, z, d = C.c_void_p, C.c_uint, C.c_int, C.c_size_t, C.c_uint32",1))
 return all(not abi_contract(m) for m in mutations)
def schedule_contract(text):
 tree=ast.parse(text);run=method(tree,"DriverBackend","run");launch=method(tree,"DriverBackend","_launch")
 constants=[n.value for n in ast.walk(run) if isinstance(n,ast.Constant)]
 required={"gate_record","up_record","down_record","natural_input","silu_lut","gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters","post_ordinary_releases_pre_pop"}
 calls=[n.func.attr for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 launch_attrs=[n.attr for n in ast.walk(launch) if isinstance(n,ast.Attribute)]
 return required<=set(constants) and all(x in calls for x in ("cuMemsetD8Async","cuMemcpyHtoDAsync_v2","cuMemcpyDtoHAsync_v2","cuStreamSynchronize")) and "cuLaunchKernel" in launch_attrs and "c_uint64" in launch_attrs and "byref" in launch_attrs
def loader_callgraph(text):
 tree=ast.parse(text);im=imports(tree);all_calls=calls(tree);bad={"cupy","torch","pycuda","cudart"}
 return not any(any(x in name.casefold() for x in bad) for name in im) and "WinDLL" in all_calls and "CDLL" in all_calls and "_cuda_related_modules" in {n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
def source_contract(src):
 backend=src["het_next_l0_ph1_nvidia_n4_backend.py"];runner=src["run_het_next_l0_ph1_nvidia_n4.py"];verifier=src["verify_het_next_l0_ph1_nvidia_n4.py"]
 driver=("C.WinDLL(str(NVCUDA_DLL), use_last_error=True, winmode=0x00000800)","cuMemHostAlloc(C.byref(out), size, 0)","cuStreamCreate(C.byref(out), CU_STREAM_NON_BLOCKING)","cuModuleLoadDataEx(C.byref(out), C.cast(self.cubin_buffer, C.c_void_p), 0, None, None)","*grid, *block, 0, self.stream, params, None","cuMemGetInfo_v2","post_ordinary_releases_pre_pop")
 compiler=("nvrtcCreateProgram","nvrtcCompileProgram","nvrtcGetProgramLogSize","nvrtcGetProgramLog","nvrtcGetPTXSize","nvrtcGetPTX","nvrtcGetCUBINSize","nvrtcGetCUBIN","nvrtcDestroyProgram","--gpu-architecture=sm_120")
 independent=not ({"het_next_l0_ph1_nvidia_n4_common","het_next_l0_ph1_nvidia_n4_backend","run_het_next_l0_ph1_nvidia_n4","het_next_l0_ph1_nvidia_n4_transaction"}&imports(ast.parse(verifier)))
 main=next(n for n in ast.parse(runner).body if isinstance(n,ast.FunctionDef) and n.name=="main");names=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else "" for n in ast.walk(main) if isinstance(n,ast.Call)]
 return all(x in backend for x in driver+compiler) and abi_contract(backend) and schedule_contract(backend) and loader_callgraph(backend) and independent and "validate_authorization" in names and names.index("validate_authorization")<min(i for i,x in enumerate(names) if x in {"compile_phase","physical_phase"})
def forbidden_surface(src):
 tree=ast.parse(src[Path(__file__).name]);bad_import={"torch","transformers","safetensors","cupy","pycuda"};bad_call={"WinDLL","CDLL","LoadLibrary","Popen","run","system","cuInit","nvrtcCompileProgram","from_pretrained"}
 return not(bad_import&imports(tree)) and not(bad_call&calls(tree))
def transaction_simulation():
 with tempfile.TemporaryDirectory(prefix="ph1_nvidia_n4_static_") as td:
  root=Path(td);out=root/"bundle";fail=root/"fail";q=root/"quarantine";kind="fixture";result={"kind":"fixture","status":"positive"}
  tx.publish_bundle(out,result,kind,lambda p:tx.verify_bundle(p,kind));positive=tx.verify_bundle(out,kind) and tx.clean_or_quarantine(out,fail,q,kind)=="already_complete"
  stale=root/"stale";stale.mkdir();(stale/"orphan").write_bytes(b"x")
  try:tx.clean_or_quarantine(stale,fail,q,kind);stale_rejected=False
  except RuntimeError:stale_rejected=not stale.exists() and q.exists()
  f1=tx.atomic_failure(fail,{"kind":"fixture_failure","stage":"x","error":"primary","error_type":"RuntimeError","device_opened":False,"disposition":"bounded"});f2=tx.atomic_failure(fail,{"kind":"fixture_failure","stage":"y","error":"secondary","error_type":"RuntimeError","device_opened":True,"disposition":"bounded"})
  create_new=f1!=f2 and f1.exists() and f2.exists() and len(list(fail.glob("attempt_*/failure.json")))==2
  try:tx.publish_bundle(out,result,kind,lambda p:True);repeat_rejected=False
  except FileExistsError:repeat_rejected=tx.verify_bundle(out,kind)
  rejected=root/"rejected"
  try:tx.publish_bundle(rejected,result,kind,lambda p:False);verifier_rejected=False
  except RuntimeError:verifier_rejected=not rejected.exists() and not rejected.with_name(rejected.name+".inprogress").exists()
  corrupt=root/"corrupt";shutil.copytree(out,corrupt);(corrupt/"result.json").write_bytes(b"{}\n");corrupt_rejected=not tx.verify_bundle(corrupt,kind)
  cq=root/"cq"
  try:tx.clean_or_quarantine(corrupt,root/"no_fail",cq,kind);corrupt_quarantined=False
  except RuntimeError:corrupt_quarantined=not corrupt.exists() and len(list(cq.iterdir()))==1
  huge=tx.atomic_failure(root/"huge_fail",{"kind":"fixture_failure","stage":"oversize","error":"x"*(17*2**20),"error_type":"RuntimeError","device_opened":True,"disposition":"bounded"});huge_row=json.loads(huge.read_text());oversize_bounded=huge.stat().st_size<4096 and huge_row.get("disposition")=="bounded_summary"
  values={"positive":positive,"stale_rejected":stale_rejected,"create_new":create_new,"repeat_rejected":repeat_rejected,"verifier_rejected":verifier_rejected,"corrupt_rejected":corrupt_rejected,"corrupt_quarantined":corrupt_quarantined,"oversize_bounded":oversize_bounded}
  return {"pass":all(values.values()),**values}
def load_inert(path,name):
 bootstrap_contract_modules()
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def verifier_mutations():
 verify=load_inert(S/"verify_het_next_l0_ph1_nvidia_n4.py","ph1_n4_verify_static")
 baseline={"protocol":{n:True for n in ("finite","controls","resources","cleanup","operation_codes","schedule","abi","runtime_surface")},"numerical_false":[],"status":"nvidia_physical_positive","positive":True,"abi_functions":30,"all_codes_zero":True,"allocations":[14,14],"schedule":[9,5,4,9,1,7],"release_count":30,"pointer_crosslinks":True,"stream_crosslinks":True,"runtime_forbidden":[]}
 cases=[]
 for name in baseline["protocol"]:
  row=copy.deepcopy(baseline);row["protocol"][name]=False;cases.append(("protocol:"+name,row))
 for key,value in (("abi_functions",29),("all_codes_zero",False),("allocations",[14,13]),("schedule",[9,5,4,9,1,6]),("release_count",29),("pointer_crosslinks",False),("stream_crosslinks",False),("runtime_forbidden",["cudart64_130.dll"])):
  row=copy.deepcopy(baseline);row[key]=value;cases.append((key,row))
 good_negative=copy.deepcopy(baseline);good_negative.update(numerical_false=["stages_exact"],status="nvidia_device_numerical_negative",positive=False)
 bad_negative=copy.deepcopy(good_negative);bad_negative["numerical_false"]=["resources"]
 pure=verify.contract_snapshot_valid(baseline) and verify.contract_snapshot_valid(good_negative) and not verify.contract_snapshot_valid(bad_negative) and all(not verify.contract_snapshot_valid(row) for _,row in cases)
 with tempfile.TemporaryDirectory(prefix="ph1_n4_verifier_paths_") as td:
  root=Path(td);kind="fixture";candidate=root/"bundle";tx.publish_bundle(candidate,{"kind":"fixture","status":"positive"},kind,lambda p:tx.verify_bundle(p,kind))
  bundle_ok=verify.bundle(candidate,kind);(candidate/"result.json").write_bytes(b"{}\n");bundle_mutated=not verify.bundle(candidate,kind)
  injected=lambda:True;missing_compile=verify.verify_compile(root/"missing_compile",injected);missing_physical=verify.verify_physical(root/"missing_physical",injected,lambda:None)
  production_paths=(missing_compile.get("bundle") is False and missing_compile.get("parse") is False and missing_physical.get("bundle") is False and missing_physical.get("parse") is False)
 return pure and bundle_ok and bundle_mutated and production_paths
def cleanup_faults():
 backend=load_inert(S/"het_next_l0_ph1_nvidia_n4_backend.py","ph1_n4_backend_static")
 class Fake:
  def __init__(self,fail):self.fail=fail;self.index=0
  def __getattr__(self,name):
   def call(*args):self.index+=1;return 91 if self.index==self.fail else 0
   return call
 for fail in range(1,31):
  obj=backend.DriverBackend();obj.lib=Fake(fail);obj.device=[(n,1000+i,z) for i,(n,z) in enumerate(backend.BUFFER_TABLE)];obj.pinned=[(n,3000+i,z) for i,(n,z) in enumerate(backend.BUFFER_TABLE)];obj.module=backend.C.c_void_p(5001);obj.stream=backend.C.c_void_p(5002);rows=obj._release_ordinary()
  if len(rows)!=30 or sum(r["code"]!=0 for r in rows)!=1:return False
 return True
def topology():
 targets=(R/"het_next_l0_ph1_nvidia_n4_compile",R/"het_next_l0_ph1_nvidia_n4_physical",R/"het_next_l0_ph1_nvidia_n4_compile_failures",R/"het_next_l0_ph1_nvidia_n4_physical_failures",R/"het_next_l0_ph1_nvidia_n4_quarantine",OUT)
 return all(not p.exists() for p in targets) and not any(R.glob("het_next_l0_ph1_nvidia_n4*.inprogress*"))
def isolated_suite():
 bootstrap_contract_modules();return {"cleanup_faults":cleanup_faults(),"verifier_mutations":verifier_mutations(),"payload_bytes_read":READ_COUNTER["payload_bytes"]}
def main():
 p=argparse.ArgumentParser();p.add_argument("--ack");p.add_argument("--isolated-fixtures",action="store_true");a=p.parse_args()
 if a.isolated_fixtures:
  result=isolated_suite();print(json.dumps(result,separators=(",",":")));return 0 if result=={"cleanup_faults":True,"verifier_mutations":True,"payload_bytes_read":0} else 3
 bootstrap_contract_modules()
 lock,hashes=hash_lock()
 if lock.get("preflight_open") is not True or lock.get("preflight_token")!=ACK or a.ack!=ACK:raise PermissionError("preflight_closed")
 child=S/"het_next_l0_ph1_nvidia_n4_fixture_child.py";isolated=subprocess.run([sys.executable,"-I","-B",str(child.resolve())],cwd=str(ROOT),capture_output=True,timeout=120,check=False);isolated_row=json.loads(isolated.stdout) if isolated.returncode==0 else {}
 src=sources();simulation=transaction_simulation();checks={"hash_bindings":hashes,"output_absent":topology(),"kernel_contract":kernel_contract(src["het_next_l0_ph1_nvidia_n4_kernels.cu"]),"source_contract":source_contract(src),"abi_ast_exact":abi_contract(src["het_next_l0_ph1_nvidia_n4_backend.py"]),"abi_mutations":abi_mutations(src["het_next_l0_ph1_nvidia_n4_backend.py"]),"schedule_ast":schedule_contract(src["het_next_l0_ph1_nvidia_n4_backend.py"]),"forbidden_surface":forbidden_surface(src),"transaction_simulation":simulation["pass"],"transaction_details":all(simulation.values()),"isolated_production_fixtures":isolated_row=={"cleanup_faults":True,"verifier_mutations":True,"payload_bytes_read":0},"lock_closed_physical":lock.get("compiler_open") is False and lock.get("physical_open") is False,"audit_binding":lock.get("bindings",{}).get("n3_source_audit",{}).get("sha256")=="9b326722f4d61a083fb7bbd991f3ea6d00b717962d2ea6748afdbc6ecf2560bb","cardinality_literals":all(x in "\n".join(src.values()) for x in ("2_185_216","675840","post_ordinary_releases_pre_pop","range(30)","len(rows) != 22"))}
 result={"kind":"het_next_l0_ph1_nvidia_n4_static_preflight","checks":checks,"pass":all(checks.values()),"passed":sum(checks.values()),"total":len(checks),"no_payload":True,"no_compiler":True,"no_device":True}
 if OUT.exists():raise FileExistsError(OUT)
 OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0 if result["pass"] else 3
if __name__=="__main__":raise SystemExit(main())
