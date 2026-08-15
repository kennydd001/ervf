#!/usr/bin/env python3
"""R8A2 namespace/auth wrapper around unchanged frozen R8A1/R7A execution."""
from __future__ import annotations
import ctypes as C, hashlib, json, sys, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];S=ROOT/"scripts/streamq5_moe";R=ROOT/"reports/streamq5_moe";sys.path.insert(0,str(S))
import run_het_next_l0_ph1_intel_execution_r8a1 as prior
SCRIPT=Path(__file__).resolve();VERIFIER=S/"verify_het_next_l0_ph1_intel_execution_r8a2.py";PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A2_PREREGISTRATION_2026-08-14.md";LOCK=R/"het_next_l0_ph1_intel_execution_r8a2_lock.json";AUDIT=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
OUT=R/"het_next_l0_ph1_intel_execution_r8a2";INHERITED_FAILED=R/"het_next_l0_ph1_intel_execution_r8a2_backend_failed_attempts";INHERITED_QUAR=R/"het_next_l0_ph1_intel_execution_r8a2_backend_quarantine";FAILED=R/"het_next_l0_ph1_intel_execution_r8a2_failed_attempts";QUAR=R/"het_next_l0_ph1_intel_execution_r8a2_quarantine";VERIFY=R/"het_next_l0_ph1_intel_execution_r8a2_independent_verification.json"
ACK="PH1_INTEL_EXECUTION_R8A2_AFTER_R8P8_PASS_AND_TERMINAL_AUDIT_GO";VENV=ROOT/".venv";VENV_PY=VENV/"Scripts/python.exe";PYVENV=VENV/"pyvenv.cfg";ALIAS=Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe");BASE_PREFIX=Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
EXPECTED_NATIVE=[str(ALIAS),"-I","-B",str(SCRIPT),"--ack",ACK];EXPECTED_ARGV=[str(SCRIPT),"--ack",ACK];MAX_FAILURE=16*2**20
OLD_PATHS=(prior.OUT,prior.INHERITED_FAILED,prior.INHERITED_QUAR,prior.FAILED,prior.QUAR,prior.VERIFY,*prior.OLD_OUTPUTS)
CHAIN={"runner_sha256":SCRIPT,"verifier_sha256":VERIFIER,"prereg_sha256":PREREG,"r8a1_audit_sha256":AUDIT,"r8a1_runner_sha256":Path(prior.__file__),"r8a1_verifier_sha256":prior.VERIFIER,"r8a1_prereg_sha256":prior.PREREG,"r8a1_lock_sha256":prior.LOCK,**{"prior_"+k:v for k,v in prior.CHAIN.items() if k not in {"runner_sha256","verifier_sha256","prereg_sha256"}}}
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a:object,b:object)->bool:return isinstance(a,str) and isinstance(b,str) and a.casefold()==b.casefold()
def parse(raw:str)->list[str]:
 k=C.WinDLL("kernel32",use_last_error=True);s=C.WinDLL("shell32",use_last_error=True);f=s.CommandLineToArgvW;f.argtypes=(C.c_wchar_p,C.POINTER(C.c_int));f.restype=C.POINTER(C.c_wchar_p);free=k.LocalFree;free.argtypes=(C.c_void_p,);free.restype=C.c_void_p;n=C.c_int();p=f(raw,C.byref(n))
 if not p:raise C.WinError(C.get_last_error())
 try:return [p[i] for i in range(n.value)]
 finally:
  if free(C.cast(p,C.c_void_p)):raise C.WinError(C.get_last_error())
def identity()->dict:
 k=C.WinDLL("kernel32",use_last_error=True);g=k.GetCommandLineW;g.argtypes=();g.restype=C.c_wchar_p;raw=g()
 return {"native_raw":raw,"native_argv":parse(raw),"orig_argv":list(sys.orig_argv),"argv":list(sys.argv),"sys_executable":sys.executable,"sys_prefix":sys.prefix,"base_executable":getattr(sys,"_base_executable",None),"base_prefix":sys.base_prefix,"isolated":sys.flags.isolated,"dont_write_bytecode":sys.dont_write_bytecode,"entry_name":__name__,"entry_spec_is_none":__spec__ is None,"entry_package":__package__,"entry_file":str(SCRIPT),"direct_entry":__name__=="__main__" and __spec__ is None and __package__ in (None,"") and Path(__file__).resolve()==SCRIPT,"python_sha256":sha(VENV_PY),"pyvenv_sha256":sha(PYVENV)}
def identity_valid(x:dict)->bool:
 keys={"native_raw","native_argv","orig_argv","argv","sys_executable","sys_prefix","base_executable","base_prefix","isolated","dont_write_bytecode","entry_name","entry_spec_is_none","entry_package","entry_file","direct_entry","python_sha256","pyvenv_sha256"};derived=x.get("entry_name")=="__main__" and x.get("entry_spec_is_none") is True and x.get("entry_package") in (None,"") and same(x.get("entry_file"),str(SCRIPT))
 return set(x)==keys and parse(x["native_raw"])==EXPECTED_NATIVE and x["native_argv"]==x["orig_argv"]==EXPECTED_NATIVE and x["argv"]==EXPECTED_ARGV and same(x["sys_executable"],str(VENV_PY.resolve())) and same(x["sys_prefix"],str(VENV.resolve())) and same(x["base_executable"],str(ALIAS)) and same(x["base_prefix"],str(BASE_PREFIX)) and x["isolated"]==1 and x["dont_write_bytecode"] is True and x["python_sha256"]=="0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and x["pyvenv_sha256"]=="9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and x["direct_entry"] is True and derived
def clean()->bool:
 fresh=(OUT,INHERITED_FAILED,INHERITED_QUAR,FAILED,QUAR,VERIFY);return all(not p.exists() for p in (*fresh,*OLD_PATHS)) and not list(R.glob("het_next_l0_ph1_intel_execution_r8a2*.inprogress*"))
def authorize()->dict:
 ident=identity()
 if not identity_valid(ident) or not clean():raise RuntimeError("identity_or_topology")
 observed={k:sha(v) for k,v in CHAIN.items()};lock=json.loads(LOCK.read_text());exact=set(lock)=={"kind","execution_open","audit_token","one_attempt",*observed} and lock.get("kind")=="ph1_intel_execution_r8a2_lock" and lock.get("execution_open") is True and lock.get("audit_token")==ACK and lock.get("one_attempt") is True and all(lock.get(k)==v for k,v in observed.items())
 historical=prior.frozen.r8p8_pass() and prior.frozen.r7d_contract() and prior.frozen.exact_failure(prior.frozen.R7D1_FAILURE_ROOT,prior.frozen.R7D1_FAILURE,"88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0","ph1_intel_execution_r7c2_failure") and prior.frozen.exact_failure(prior.frozen.R8P6_FAILURE_ROOT,prior.frozen.R8P6_FAILURE,"03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372","ph1_intel_execution_r8p6_failure") and sha(AUDIT)=="7aec53fa62ddb8412e1364499f8c39326cd6fc2dd49451395399618304635287"
 if not exact or not historical:raise RuntimeError("authorization")
 auth=prior.frozen.physical.authorize();auth["r8a2_authorization"]={"kind":"ph1_intel_execution_r8a2_authorization","lock_sha256":sha(LOCK),"observed":observed,"audit_token":ACK,"invocation":ident,"r8p8_pass":True,"r7d_contract":True,"r7a_verification_absent":True,"historical_failures_exact":True};return auth
def write_summary(row:dict)->Path:
 row=dict(row);row["kind"]="ph1_intel_execution_r8a2_failure";data=prior.canon(row)
 if len(data)>MAX_FAILURE:data=prior.canon({"kind":"ph1_intel_execution_r8a2_failure","status":"invalid_protocol","terminal_type":"oversize_summary","error":"wrapper_summary_oversize","original_bytes":len(data),"original_sha256":hashlib.sha256(data).hexdigest(),"device_opened":bool(row.get("device_opened",False)),"disposition":"bounded_summary_only"})
 FAILED.mkdir(parents=True,exist_ok=True);nonce=uuid.uuid4().hex;temp=R/(FAILED.name+"."+nonce+".inprogress");dest=FAILED/("attempt_"+nonce);temp.mkdir()
 try:prior.frozen.physical.base.write(temp/"failure.json",data);prior.frozen.physical.base.move(temp,dest)
 except Exception:
  if temp.exists():QUAR.mkdir(parents=True,exist_ok=True);prior.frozen.physical.base.move(temp,QUAR/("failed_commit_"+nonce))
  raise
 return dest
def configure()->None:
 prior.OUT=OUT;prior.INHERITED_FAILED=INHERITED_FAILED;prior.INHERITED_QUAR=INHERITED_QUAR;prior.FAILED=FAILED;prior.QUAR=QUAR;prior.VERIFY=VERIFY;prior.write_summary=write_summary;prior.frozen.physical.OUT=OUT;prior.frozen.physical.FAILED=INHERITED_FAILED;prior.frozen.physical.QUAR=INHERITED_QUAR
def main()->int:
 if sys.argv!=EXPECTED_ARGV:return 3
 try:auth=authorize()
 except Exception:return 3
 configure();return prior.execute(auth)
if __name__=="__main__":raise SystemExit(main())
