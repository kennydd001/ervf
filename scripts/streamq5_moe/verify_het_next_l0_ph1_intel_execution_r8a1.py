#!/usr/bin/env python3
"""Standalone R8A1 verifier with exact mutually-exclusive terminal adjudication."""
from __future__ import annotations
import copy, ctypes as C, hashlib, json, os, sys, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];S=ROOT/"scripts/streamq5_moe";R=ROOT/"reports/streamq5_moe"
RUNNER=S/"run_het_next_l0_ph1_intel_execution_r8a1.py";PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A1_PREREGISTRATION_2026-08-14.md";LOCK=R/"het_next_l0_ph1_intel_execution_r8a1_lock.json";AUDIT=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md";OUT=R/"het_next_l0_ph1_intel_execution_r8a1";FAILED=R/"het_next_l0_ph1_intel_execution_r8a1_failed_attempts";BACKEND_FAILED=R/"het_next_l0_ph1_intel_execution_r8a1_backend_failed_attempts";QUAR=R/"het_next_l0_ph1_intel_execution_r8a1_quarantine";BACKEND_QUAR=R/"het_next_l0_ph1_intel_execution_r8a1_backend_quarantine";VERIFY=R/"het_next_l0_ph1_intel_execution_r8a1_independent_verification.json"
ACK="PH1_INTEL_EXECUTION_R8A1_AFTER_R8P8_PASS_AND_TERMINAL_AUDIT_GO";VENV=ROOT/".venv";VENV_PY=VENV/"Scripts/python.exe";PYVENV=VENV/"pyvenv.cfg";ALIAS=Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe");BASE_PREFIX=Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
RUN_NATIVE=[str(ALIAS),"-I","-B",str(RUNNER),"--ack",ACK];RUN_ARGV=[str(RUNNER),"--ack",ACK];VERIFY_NATIVE=[str(ALIAS),"-I","-B",str(Path(__file__).resolve())];VERIFY_ARGV=[str(Path(__file__).resolve())]
OLD_RUNNER=S/"run_het_next_l0_ph1_intel_execution_r8a.py";OLD_VERIFIER=S/"verify_het_next_l0_ph1_intel_execution_r8a.py";OLD_PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A_PREREGISTRATION_2026-08-14.md";OLD_LOCK=R/"het_next_l0_ph1_intel_execution_r8a_lock.json"
BASE_CHAIN={"r8p8_result_sha256":R/"het_next_l0_ph1_intel_execution_r8p8_static_preflight.json","r8p8_manifest_sha256":R/"het_next_l0_ph1_intel_execution_r8p8_static_preflight.manifest.json","r8p8_commit_sha256":R/"het_next_l0_ph1_intel_execution_r8p8_static_preflight.commit.json","r8p8_verification_sha256":R/"het_next_l0_ph1_intel_execution_r8p8_independent_verification.json","r8p8_audit_sha256":R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P8_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md","r8p8_lock_sha256":R/"het_next_l0_ph1_intel_execution_r8p8_lock.json","r7d_runner_sha256":S/"run_het_next_l0_ph1_intel_execution_r7d.py","r7d_verifier_sha256":S/"verify_het_next_l0_ph1_intel_execution_r7d.py","r7d_lock_sha256":R/"het_next_l0_ph1_intel_execution_r7d_lock.json","r7c2_result_sha256":R/"het_next_l0_ph1_intel_execution_r7c2_static_preflight.json","r7a_result_sha256":R/"het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json","r7p_result_sha256":R/"het_next_l0_ph1_intel_execution_r7p_static_preflight.json","r7d1_failure_sha256":R/"het_next_l0_ph1_intel_execution_r7d1_failed_attempts/attempt_7c45ba0bda09470eba7145ef75281ea3/failure.json","r7d1_diagnosis_sha256":R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md","r8p6_failure_sha256":R/"het_next_l0_ph1_intel_execution_r8p6_failed_attempts/attempt_71e198678f004a56a6912d07a4187dfd/failure.json","r8p6_diagnosis_sha256":R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_DIRECT_ENTRY_FAILURE_DIAGNOSIS_2026-08-14.md","physical_runner_sha256":S/"run_het_next_l0_ph1_intel_execution_r7a.py","physical_verifier_sha256":S/"verify_het_next_l0_ph1_intel_execution_r7a.py","backend_sha256":S/"het_next_l0_ph1_intel_execution_r6_backend.py","common_sha256":S/"het_next_l0_ph1_intel_execution_r6_common.py"}
CHAIN={"runner_sha256":RUNNER,"verifier_sha256":Path(__file__),"prereg_sha256":PREREG,"r8a_audit_sha256":AUDIT,"r8a_runner_sha256":OLD_RUNNER,"r8a_verifier_sha256":OLD_VERIFIER,"r8a_prereg_sha256":OLD_PREREG,"r8a_lock_sha256":OLD_LOCK,**{"base_"+k:v for k,v in BASE_CHAIN.items()}}
IDENT_KEYS={"native_raw","native_argv","orig_argv","argv","sys_executable","sys_prefix","base_executable","base_prefix","isolated","dont_write_bytecode","entry_name","entry_spec_is_none","entry_package","entry_file","direct_entry","python_sha256","pyvenv_sha256"}
GATE_KEYS={"controls","stages","counters","identity","compile_identity","ledger_order","ownership","resource_samples","allocations","writes","initialization","args","launch","finish_reads","release","extensions","forbidden_static_and_runtime","resources"}

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x:object)->bytes:return (json.dumps(x,sort_keys=True,separators=(",",":"))+"\n").encode()
def same(a:object,b:object)->bool:return isinstance(a,str) and isinstance(b,str) and a.casefold()==b.casefold()
def parse(raw:str)->list[str]:
 k=C.WinDLL("kernel32",use_last_error=True);s=C.WinDLL("shell32",use_last_error=True);f=s.CommandLineToArgvW;f.argtypes=(C.c_wchar_p,C.POINTER(C.c_int));f.restype=C.POINTER(C.c_wchar_p);free=k.LocalFree;free.argtypes=(C.c_void_p,);free.restype=C.c_void_p;n=C.c_int();p=f(raw,C.byref(n))
 if not p:raise C.WinError(C.get_last_error())
 try:return [p[i] for i in range(n.value)]
 finally:
  if free(C.cast(p,C.c_void_p)):raise C.WinError(C.get_last_error())
def current_identity()->dict:
 k=C.WinDLL("kernel32",use_last_error=True);g=k.GetCommandLineW;g.argtypes=();g.restype=C.c_wchar_p;raw=g();script=Path(__file__).resolve()
 return {"native_raw":raw,"native_argv":parse(raw),"orig_argv":list(sys.orig_argv),"argv":list(sys.argv),"sys_executable":sys.executable,"sys_prefix":sys.prefix,"base_executable":getattr(sys,"_base_executable",None),"base_prefix":sys.base_prefix,"isolated":sys.flags.isolated,"dont_write_bytecode":sys.dont_write_bytecode,"entry_name":__name__,"entry_spec_is_none":__spec__ is None,"entry_package":__package__,"entry_file":str(script),"direct_entry":__name__=="__main__" and __spec__ is None and __package__ in (None,"") and Path(__file__).resolve()==script,"python_sha256":sha(VENV_PY),"pyvenv_sha256":sha(PYVENV)}
def identity_valid(x:dict,native:list[str],argv:list[str],script:Path)->bool:
 derived=x.get("entry_name")=="__main__" and x.get("entry_spec_is_none") is True and x.get("entry_package") in (None,"") and same(x.get("entry_file"),str(script))
 try:raw_ok=parse(x["native_raw"])==native
 except Exception:return False
 return set(x)==IDENT_KEYS and raw_ok and x["native_argv"]==x["orig_argv"]==native and x["argv"]==argv and same(x["sys_executable"],str(VENV_PY.resolve())) and same(x["sys_prefix"],str(VENV.resolve())) and same(x["base_executable"],str(ALIAS)) and same(x["base_prefix"],str(BASE_PREFIX)) and x["isolated"]==1 and x["dont_write_bytecode"] is True and x["python_sha256"]=="0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and x["pyvenv_sha256"]=="9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and x["direct_entry"] is True and derived
def identity_mutations(x:dict,native:list[str],argv:list[str],script:Path)->bool:
 cases=[]
 for key,value in (("direct_entry",False),("entry_name","imported"),("entry_spec_is_none",False),("entry_package","pkg"),("entry_file",str(script)+".wrong"),("sys_executable",str(ALIAS)),("isolated",0),("dont_write_bytecode",False)):
  y=copy.deepcopy(x);y[key]=value;cases.append(y)
 y=copy.deepcopy(x);y["argv"]=[*argv,"extra"];cases.append(y);y=copy.deepcopy(x);y["native_argv"]=[*native,"extra"];cases.append(y);y=copy.deepcopy(x);y["orig_argv"]=[*native,"extra"];cases.append(y);y=copy.deepcopy(x);y["native_raw"]='python -c "import x"';cases.append(y)
 return len(cases)==12 and all(not identity_valid(y,native,argv,script) for y in cases)
def lock_valid()->tuple[bool,dict]:
 observed={k:sha(v) for k,v in CHAIN.items()};lock=json.loads(LOCK.read_text());ok=set(lock)=={"kind","execution_open","audit_token","one_attempt",*observed} and lock.get("kind")=="ph1_intel_execution_r8a1_lock" and lock.get("execution_open") is True and lock.get("audit_token")==ACK and lock.get("one_attempt") is True and all(lock.get(k)==v for k,v in observed.items()) and observed.get("r8a_audit_sha256")=="cae07159709049ef452ac8c49aeedfc9e930964940958ff9bf284936d565dd7a"
 return ok,observed
def historical_valid()->bool:
 if sha(OLD_VERIFIER)!="c1125ef9ff47600f608f2b163311381bdc05097a35b2313b429f61cc5271c4c1":return False
 sys.path.insert(0,str(S));import verify_het_next_l0_ph1_intel_execution_r8a as old
 return old.r8p8() and old.r7d() and old.exact_failure(old.R7D1_ROOT,old.R7D1_FAILURE,"88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0","ph1_intel_execution_r7c2_failure",931) and old.exact_failure(old.R8P6_ROOT,old.R8P6_FAILURE,"03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372","ph1_intel_execution_r8p6_failure",2986)
def extension_valid(result:dict,observed:dict)->tuple[bool,dict|None]:
 ext=result.get("authorization",{}).get("r8a1_authorization");lock_sha=sha(LOCK)
 ok=isinstance(ext,dict) and set(ext)=={"kind","lock_sha256","observed","audit_token","invocation","r8p8_pass","r7d_contract","r7a_verification_absent","historical_failures_exact"} and ext.get("kind")=="ph1_intel_execution_r8a1_authorization" and ext.get("lock_sha256")==lock_sha and ext.get("observed")==observed and ext.get("audit_token")==ACK and all(ext.get(k) is True for k in ("r8p8_pass","r7d_contract","r7a_verification_absent","historical_failures_exact")) and identity_valid(ext.get("invocation",{}),RUN_NATIVE,RUN_ARGV,RUNNER) and identity_mutations(ext["invocation"],RUN_NATIVE,RUN_ARGV,RUNNER)
 return ok,ext
def bundle()->tuple[bool,dict|None]:
 try:
  rp,mp,cp=(OUT/n for n in ("result.json","manifest.json","commit.json"));rb=rp.read_bytes();rr=json.loads(rb);mm=json.loads(mp.read_text());cc=json.loads(cp.read_text());row={"name":"result.json","bytes":len(rb),"sha256":hashlib.sha256(rb).hexdigest()};mb=canon(mm)
  ok=rr.get("kind")=="ph1_intel_execution_r7a" and mm=={"kind":"ph1_intel_execution_r7a_manifest","files":[row]} and cc=={"kind":"ph1_intel_execution_r7a_commit","manifest_sha256":hashlib.sha256(mb).hexdigest(),"result_sha256":row["sha256"]} and {p.name for p in OUT.iterdir()}=={"result.json","manifest.json","commit.json"} and sum(p.stat().st_size for p in OUT.iterdir())<=16*2**20
  return ok,rr
 except Exception:return False,None
def reconstruct_inherited(p:Path)->dict:
 files=sorted(x for x in p.parent.rglob("*") if x.is_file());row=json.loads(p.read_text());total=sum(x.stat().st_size for x in files);normal={"kind","status","error","traceback","device_opened","backend_evidence","secondary_resource_sample","disposition"};oversize={"kind","status","error","device_opened","oversized_temp_bytes","oversized_temp_digest","disposition"};valid=set(row) in (normal,oversize) and row.get("kind")=="ph1_intel_execution_r7a_failure" and row.get("status")=="valid_negative_failure" and isinstance(row.get("error"),str) and isinstance(row.get("device_opened"),bool) and row.get("disposition") in ("attempt_archived_create_new","oversized_temp_quarantined_not_retained_failure_bundle") and 1<=len(files)<=4 and total<=16*2**20
 return {"relative_path":str(p.relative_to(R)),"failure_sha256":sha(p),"failure_bytes":p.stat().st_size,"bundle_files":[{"name":str(x.relative_to(p.parent)),"bytes":x.stat().st_size,"sha256":sha(x)} for x in files],"bundle_bytes":total,"bundle_file_count":len(files),"kind":row.get("kind"),"status":row.get("status"),"disposition":row.get("disposition"),"device_opened":row.get("device_opened"),"valid":valid}
def wrapper_file()->tuple[Path|None,dict|None]:
 files=sorted(p for p in FAILED.rglob("*") if p.is_file()) if FAILED.exists() else []
 if len(files)!=1 or files[0].name!="failure.json":return None,None
 try:return files[0],json.loads(files[0].read_text())
 except Exception:return files[0],None
def correlated_shape(w:dict)->bool:
 keys={"kind","status","terminal_type","stage","error","device_opened","delegated_return","inherited_failure_count","inherited","correlation_valid","disposition"}
 return set(w)==keys and w.get("kind")=="ph1_intel_execution_r8a1_failure" and w.get("status")=="correlated_delegated_negative" and w.get("terminal_type")=="delegated_failure" and w.get("stage")=="delegated_return" and w.get("error")=="delegated_execution_nonzero" and isinstance(w.get("delegated_return"),int) and w["delegated_return"]!=0 and w.get("inherited_failure_count")==1 and w.get("correlation_valid") is True and w.get("disposition")=="atomic_bounded_correlated_summary" and isinstance(w.get("device_opened"),bool) and isinstance(w.get("inherited"),dict)
def correlation_matches(w:dict,actual:dict,count:int)->bool:
 return correlated_shape(w) and count==1 and actual.get("valid") is True and w.get("inherited")==actual and w.get("device_opened") is actual.get("device_opened")
def failure_state()->tuple[str,bool,dict]:
 wp,w=wrapper_file();backend=sorted(BACKEND_FAILED.rglob("failure.json")) if BACKEND_FAILED.exists() else []
 if wp is None or not isinstance(w,dict) or OUT.exists() or QUAR.exists() or BACKEND_QUAR.exists():return "invalid",False,{"reason":"topology"}
 if w.get("terminal_type")=="early_outer_failure":
  keys={"kind","status","terminal_type","stage","error","traceback","device_opened","delegated_return","inherited_failure_count","inherited","disposition"};ok=set(w)==keys and w.get("kind")=="ph1_intel_execution_r8a1_failure" and w.get("status")=="infrastructure_negative" and w.get("stage")=="outer_boundary" and isinstance(w.get("error"),str) and w.get("device_opened") is False and w.get("delegated_return") is None and w.get("inherited_failure_count")==0 and w.get("inherited") is None and w.get("disposition")=="atomic_bounded_outer_failure" and not backend and wp.stat().st_size<=16*2**20
  return "early_failure",ok,{"wrapper_sha256":sha(wp)}
 if w.get("terminal_type")!="delegated_failure":return "invalid",False,{"reason":"terminal_type"}
 actual=reconstruct_inherited(backend[0]) if len(backend)==1 else {};ok=correlation_matches(w,actual,len(backend)) and wp.stat().st_size<=16*2**20
 return "delegated_failure",ok,{"wrapper_sha256":sha(wp),"inherited":actual}
def terminal_mutations()->bool:
 actual={"relative_path":"x/failure.json","failure_sha256":"a"*64,"failure_bytes":123,"bundle_files":[{"name":"failure.json","bytes":123,"sha256":"a"*64}],"bundle_bytes":123,"bundle_file_count":1,"kind":"ph1_intel_execution_r7a_failure","status":"valid_negative_failure","disposition":"attempt_archived_create_new","device_opened":True,"valid":True};base={"kind":"ph1_intel_execution_r8a1_failure","status":"correlated_delegated_negative","terminal_type":"delegated_failure","stage":"delegated_return","error":"delegated_execution_nonzero","device_opened":True,"delegated_return":3,"inherited_failure_count":1,"inherited":actual,"correlation_valid":True,"disposition":"atomic_bounded_correlated_summary"}
 cases=[]
 for k,v in (("status","invalid_protocol"),("terminal_type","invalid_delegation"),("delegated_return",0),("inherited_failure_count",2),("correlation_valid",False),("disposition","wrong"),("error","success_without_commit")):
  y=copy.deepcopy(base);y[k]=v;cases.append(y)
 y=copy.deepcopy(base);y["extra"]=1;cases.append(y);y=copy.deepcopy(base);del y["inherited"];cases.append(y);y=copy.deepcopy(base);y["inherited"]["failure_sha256"]="b"*64;cases.append(y);y=copy.deepcopy(base);y["device_opened"]=False;cases.append(y)
 return correlation_matches(base,actual,1) and len(cases)==11 and all(not correlation_matches(x,actual,1) for x in cases) and not correlation_matches(base,actual,0) and not correlation_matches(base,{**actual,"valid":False},1)
def write(row:dict)->None:
 if VERIFY.exists():raise FileExistsError(VERIFY)
 temp=R/(VERIFY.name+"."+uuid.uuid4().hex+".inprogress");data=canon(row)
 try:
  with temp.open("xb") as h:h.write(data);h.flush();os.fsync(h.fileno())
  os.link(temp,VERIFY);temp.unlink()
 finally:
  if temp.exists():temp.unlink()
def main()->int:
 live=current_identity();live_ok=identity_valid(live,VERIFY_NATIVE,VERIFY_ARGV,Path(__file__).resolve());lock_ok,observed=lock_valid();b_ok,result=bundle();state="invalid";terminal_valid=False;checks={"live_invocation":live_ok,"live_invocation_mutations":identity_mutations(live,VERIFY_NATIVE,VERIFY_ARGV,Path(__file__).resolve()),"lock":lock_ok,"historical_contract":historical_valid(),"terminal_mutations":terminal_mutations()}
 if b_ok and result is not None and not any(p.exists() for p in (FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR)):
  state="committed";auth_ok,ext=extension_valid(result,observed);checks.update({"bundle":True,"authorization":auth_ok})
  numerical={}
  if auth_ok:
   if sha(S/"verify_het_next_l0_ph1_intel_execution_r7a.py")!="18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88":raise RuntimeError("numerical_hash")
   sys.path.insert(0,str(S));import verify_het_next_l0_ph1_intel_execution_r7a as nv
   try:numerical=nv.verify_dict(result)
   except Exception:numerical={"verifier_exception":False}
  checks.update({"numerical:"+k:v for k,v in numerical.items()});positive=auth_ok and result.get("positive") is True and result.get("status")=="intel_execution_positive" and numerical and all(numerical.values());gates=result.get("gates",{});provenance=all(numerical.get(k) is True for k in ("authorization","compile_package","records_input_lut"));negative=auth_ok and result.get("positive") is False and result.get("status")=="intel_execution_negative" and set(gates)==GATE_KEYS and all(isinstance(v,bool) for v in gates.values()) and any(v is False for v in gates.values()) and provenance
  false_gates={k for k,v in gates.items() if v is False} if isinstance(gates,dict) else set();negative=negative and bool(false_gates) and false_gates<=GATE_KEYS
  state="positive" if positive else "committed_negative" if negative else "invalid_committed";terminal_valid=positive or negative;checks["terminal_contract"]=terminal_valid;checks["committed_negative_stage_allowlist"]=positive or negative
 else:
  state,terminal_valid,evidence=failure_state();checks["bundle_absent"]=not OUT.exists();checks["failure_contract"]=terminal_valid
 passed=state=="positive" and all(checks.values());row={"kind":"ph1_intel_execution_r8a1_independent_verification","terminal_state":state,"terminal_valid":terminal_valid,"checks":checks,"pass":passed,"passed":sum(v is True for v in checks.values()),"total":len(checks),"claim":"one real expert/input Intel correctness component only"};write(row);print(json.dumps(row,indent=2));return 0 if passed else 3
if __name__=="__main__":raise SystemExit(main())
