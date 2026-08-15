#!/usr/bin/env python3
"""R8A1 terminal-state authorization wrapper; physical R7A delegate unchanged."""
from __future__ import annotations
import ctypes as C, hashlib, json, os, sys, traceback, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; S=ROOT/"scripts/streamq5_moe"; R=ROOT/"reports/streamq5_moe"; sys.path.insert(0,str(S))
import run_het_next_l0_ph1_intel_execution_r8a as frozen
SCRIPT=Path(__file__).resolve(); VERIFIER=S/"verify_het_next_l0_ph1_intel_execution_r8a1.py"; PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A1_PREREGISTRATION_2026-08-14.md"; LOCK=R/"het_next_l0_ph1_intel_execution_r8a1_lock.json"; AUDIT=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
OUT=R/"het_next_l0_ph1_intel_execution_r8a1"; INHERITED_FAILED=R/"het_next_l0_ph1_intel_execution_r8a1_backend_failed_attempts"; INHERITED_QUAR=R/"het_next_l0_ph1_intel_execution_r8a1_backend_quarantine"; FAILED=R/"het_next_l0_ph1_intel_execution_r8a1_failed_attempts"; QUAR=R/"het_next_l0_ph1_intel_execution_r8a1_quarantine"; VERIFY=R/"het_next_l0_ph1_intel_execution_r8a1_independent_verification.json"
ACK="PH1_INTEL_EXECUTION_R8A1_AFTER_R8P8_PASS_AND_TERMINAL_AUDIT_GO"; MAX_FAILURE=16*2**20; VENV=ROOT/".venv"; VENV_PY=VENV/"Scripts/python.exe"; PYVENV=VENV/"pyvenv.cfg"; ALIAS=Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe"); BASE_PREFIX=Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
EXPECTED_NATIVE=[str(ALIAS),"-I","-B",str(SCRIPT),"--ack",ACK]; EXPECTED_ARGV=[str(SCRIPT),"--ack",ACK]
OLD_OUTPUTS=(frozen.OUT,frozen.BACKEND_FAILED,frozen.BACKEND_QUARANTINE,frozen.FAILED,frozen.QUARANTINE,frozen.VERIFY_RESULT)
CHAIN={"runner_sha256":SCRIPT,"verifier_sha256":VERIFIER,"prereg_sha256":PREREG,"r8a_audit_sha256":AUDIT,"r8a_runner_sha256":Path(frozen.__file__),"r8a_verifier_sha256":frozen.VERIFIER,"r8a_prereg_sha256":frozen.PREREG,"r8a_lock_sha256":frozen.LOCK,**{"base_"+k:v for k,v in frozen.CHAIN.items() if k not in {"runner_sha256","verifier_sha256","prereg_sha256"}}}

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x:object)->bytes:return (json.dumps(x,sort_keys=True,separators=(",",":"))+"\n").encode()
def same(a:object,b:object)->bool:return isinstance(a,str) and isinstance(b,str) and a.casefold()==b.casefold()
def parse(raw:str)->list[str]:
 k=C.WinDLL("kernel32",use_last_error=True);s=C.WinDLL("shell32",use_last_error=True);f=s.CommandLineToArgvW;f.argtypes=(C.c_wchar_p,C.POINTER(C.c_int));f.restype=C.POINTER(C.c_wchar_p);free=k.LocalFree;free.argtypes=(C.c_void_p,);free.restype=C.c_void_p;n=C.c_int();ptr=f(raw,C.byref(n))
 if not ptr:raise C.WinError(C.get_last_error())
 try:return [ptr[i] for i in range(n.value)]
 finally:
  if free(C.cast(ptr,C.c_void_p)):raise C.WinError(C.get_last_error())
def identity()->dict:
 k=C.WinDLL("kernel32",use_last_error=True);g=k.GetCommandLineW;g.argtypes=();g.restype=C.c_wchar_p;raw=g()
 return {"native_raw":raw,"native_argv":parse(raw),"orig_argv":list(sys.orig_argv),"argv":list(sys.argv),"sys_executable":sys.executable,"sys_prefix":sys.prefix,"base_executable":getattr(sys,"_base_executable",None),"base_prefix":sys.base_prefix,"isolated":sys.flags.isolated,"dont_write_bytecode":sys.dont_write_bytecode,"entry_name":__name__,"entry_spec_is_none":__spec__ is None,"entry_package":__package__,"entry_file":str(Path(__file__).resolve()),"direct_entry":__name__=="__main__" and __spec__ is None and __package__ in (None,"") and Path(__file__).resolve()==SCRIPT,"python_sha256":sha(VENV_PY),"pyvenv_sha256":sha(PYVENV)}
def identity_valid(x:dict)->bool:
 keys={"native_raw","native_argv","orig_argv","argv","sys_executable","sys_prefix","base_executable","base_prefix","isolated","dont_write_bytecode","entry_name","entry_spec_is_none","entry_package","entry_file","direct_entry","python_sha256","pyvenv_sha256"}
 derived=x.get("entry_name")=="__main__" and x.get("entry_spec_is_none") is True and x.get("entry_package") in (None,"") and same(x.get("entry_file"),str(SCRIPT))
 return set(x)==keys and parse(x["native_raw"])==EXPECTED_NATIVE and x["native_argv"]==x["orig_argv"]==EXPECTED_NATIVE and x["argv"]==EXPECTED_ARGV and same(x["sys_executable"],str(VENV_PY.resolve())) and same(x["sys_prefix"],str(VENV.resolve())) and same(x["base_executable"],str(ALIAS)) and same(x["base_prefix"],str(BASE_PREFIX)) and x["isolated"]==1 and x["dont_write_bytecode"] is True and x["python_sha256"]=="0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and x["pyvenv_sha256"]=="9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and x["direct_entry"] is True and derived
def clean()->bool:
 current=(OUT,INHERITED_FAILED,INHERITED_QUAR,FAILED,QUAR,VERIFY);temps=list(R.glob("het_next_l0_ph1_intel_execution_r8a1*.inprogress*"));return all(not p.exists() for p in (*current,*OLD_OUTPUTS)) and not temps
def authorize()->dict:
 ident=identity()
 if not identity_valid(ident) or not clean():raise RuntimeError("identity_or_topology")
 observed={k:sha(v) for k,v in CHAIN.items()};lock=json.loads(LOCK.read_text())
 exact=set(lock)=={"kind","execution_open","audit_token","one_attempt",*observed} and lock.get("kind")=="ph1_intel_execution_r8a1_lock" and lock.get("execution_open") is True and lock.get("audit_token")==ACK and lock.get("one_attempt") is True and all(lock.get(k)==v for k,v in observed.items())
 historical=frozen.r8p8_pass() and frozen.r7d_contract() and frozen.exact_failure(frozen.R7D1_FAILURE_ROOT,frozen.R7D1_FAILURE,"88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0","ph1_intel_execution_r7c2_failure") and frozen.exact_failure(frozen.R8P6_FAILURE_ROOT,frozen.R8P6_FAILURE,"03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372","ph1_intel_execution_r8p6_failure") and sha(AUDIT)=="cae07159709049ef452ac8c49aeedfc9e930964940958ff9bf284936d565dd7a"
 if not exact or not historical:raise RuntimeError("authorization")
 inherited=frozen.physical.authorize();inherited["r8a1_authorization"]={"kind":"ph1_intel_execution_r8a1_authorization","lock_sha256":sha(LOCK),"observed":observed,"audit_token":ACK,"invocation":ident,"r8p8_pass":True,"r7d_contract":True,"r7a_verification_absent":True,"historical_failures_exact":True};return inherited
def configure()->None:
 frozen.physical.OUT=OUT;frozen.physical.FAILED=INHERITED_FAILED;frozen.physical.QUAR=INHERITED_QUAR
def committed()->dict|None:
 try:return frozen.physical.verify_bundle(OUT) if OUT.exists() else None
 except Exception:return None
def inherited_files()->set[Path]:return {p.resolve() for p in INHERITED_FAILED.rglob("failure.json")} if INHERITED_FAILED.exists() else set()
def inherited_row(p:Path)->dict:
 files=sorted(x for x in p.parent.rglob("*") if x.is_file());total=sum(x.stat().st_size for x in files);row=json.loads(p.read_text());normal={"kind","status","error","traceback","device_opened","backend_evidence","secondary_resource_sample","disposition"};oversize={"kind","status","error","device_opened","oversized_temp_bytes","oversized_temp_digest","disposition"};valid=set(row) in (normal,oversize) and row.get("kind")=="ph1_intel_execution_r7a_failure" and row.get("status")=="valid_negative_failure" and isinstance(row.get("error"),str) and isinstance(row.get("device_opened"),bool) and row.get("disposition") in ("attempt_archived_create_new","oversized_temp_quarantined_not_retained_failure_bundle") and 1<=len(files)<=4 and total<=MAX_FAILURE
 return {"relative_path":str(p.relative_to(R)),"failure_sha256":sha(p),"failure_bytes":p.stat().st_size,"bundle_files":[{"name":str(x.relative_to(p.parent)),"bytes":x.stat().st_size,"sha256":sha(x)} for x in files],"bundle_bytes":total,"bundle_file_count":len(files),"kind":row.get("kind"),"status":row.get("status"),"disposition":row.get("disposition"),"device_opened":row.get("device_opened"),"valid":valid}
def write_summary(row:dict)->Path:
 data=canon(row)
 if len(data)>MAX_FAILURE:data=canon({"kind":"ph1_intel_execution_r8a1_failure","status":"invalid_protocol","terminal_type":"oversize_summary","error":"wrapper_summary_oversize","original_bytes":len(data),"original_sha256":hashlib.sha256(data).hexdigest(),"device_opened":bool(row.get("device_opened",False)),"disposition":"bounded_summary_only"})
 FAILED.mkdir(parents=True,exist_ok=True);nonce=uuid.uuid4().hex;temp=R/(FAILED.name+"."+nonce+".inprogress");dest=FAILED/("attempt_"+nonce);temp.mkdir()
 try:frozen.physical.base.write(temp/"failure.json",data);frozen.physical.base.move(temp,dest)
 except Exception:
  if temp.exists():QUAR.mkdir(parents=True,exist_ok=True);frozen.physical.base.move(temp,QUAR/("failed_commit_"+nonce))
  raise
 return dest
def execute(auth:dict)->int:
 before=inherited_files()
 try:code=frozen.physical.execute_authorized(auth)
 except Exception as exc:
  if committed() is not None:return 3
  write_summary({"kind":"ph1_intel_execution_r8a1_failure","status":"infrastructure_negative","terminal_type":"early_outer_failure","stage":"outer_boundary","error":f"{type(exc).__name__}:{exc}"[:2048],"traceback":traceback.format_exc()[-32768:],"device_opened":False,"delegated_return":None,"inherited_failure_count":0,"inherited":None,"disposition":"atomic_bounded_outer_failure"});return 3
 result=committed()
 if result is not None:return 0 if result.get("positive") is True else 3
 new=sorted(inherited_files()-before);evidence=[]
 for p in new:
  try:evidence.append(inherited_row(p))
  except Exception as exc:evidence.append({"valid":False,"error":type(exc).__name__})
 correlated=code!=0 and len(evidence)==1 and evidence[0].get("valid") is True
 write_summary({"kind":"ph1_intel_execution_r8a1_failure","status":"correlated_delegated_negative" if correlated else "invalid_protocol","terminal_type":"delegated_failure" if correlated else "invalid_delegation","stage":"delegated_return","error":"delegated_execution_nonzero" if code else "success_without_commit","device_opened":evidence[0].get("device_opened",False) if len(evidence)==1 else False,"delegated_return":code,"inherited_failure_count":len(evidence),"inherited":evidence[0] if len(evidence)==1 else evidence,"correlation_valid":correlated,"disposition":"atomic_bounded_correlated_summary" if correlated else "atomic_bounded_invalid_protocol_summary"});return 3
def main()->int:
 if sys.argv!=EXPECTED_ARGV:return 3
 try:auth=authorize()
 except Exception:return 3
 configure();return execute(auth)
if __name__=="__main__":raise SystemExit(main())
