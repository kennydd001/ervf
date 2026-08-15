#!/usr/bin/env python3
"""Standalone R8A4 verifier with injected production topology/failure matrix."""
from __future__ import annotations
import copy,ctypes as C,hashlib,json,os,shutil,sys,tempfile,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/"scripts/streamq5_moe";R=ROOT/"reports/streamq5_moe";sys.path.insert(0,str(S));import verify_het_next_l0_ph1_intel_execution_r8a3 as frozen
SELF=Path(__file__).resolve();RUNNER=S/"run_het_next_l0_ph1_intel_execution_r8a4.py";PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A4_PREREGISTRATION_2026-08-14.md";LOCK=R/"het_next_l0_ph1_intel_execution_r8a4_lock.json";AUDIT=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A3_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md";OUT=R/"het_next_l0_ph1_intel_execution_r8a4";FAILED=R/"het_next_l0_ph1_intel_execution_r8a4_failed_attempts";BACKEND_FAILED=R/"het_next_l0_ph1_intel_execution_r8a4_backend_failed_attempts";QUAR=R/"het_next_l0_ph1_intel_execution_r8a4_quarantine";BACKEND_QUAR=R/"het_next_l0_ph1_intel_execution_r8a4_backend_quarantine";VERIFY=R/"het_next_l0_ph1_intel_execution_r8a4_independent_verification.json";FAMILY_PARENT=R;FAMILY_PREFIX="het_next_l0_ph1_intel_execution_r8a4"
ACK="PH1_INTEL_EXECUTION_R8A4_AFTER_R8P8_PASS_AND_TOPOLOGY_AUDIT_GO";VENV=ROOT/".venv";VENV_PY=VENV/"Scripts/python.exe";PYVENV=VENV/"pyvenv.cfg";ALIAS=Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe");BASE_PREFIX=Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0");RUN_NATIVE=[str(ALIAS),"-I","-B",str(RUNNER),"--ack",ACK];RUN_ARGV=[str(RUNNER),"--ack",ACK];VERIFY_NATIVE=[str(ALIAS),"-I","-B",str(SELF)];VERIFY_ARGV=[str(SELF)]
OLD_RUNNER=S/"run_het_next_l0_ph1_intel_execution_r8a3.py";OLD_VERIFIER=S/"verify_het_next_l0_ph1_intel_execution_r8a3.py";OLD_PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A3_PREREGISTRATION_2026-08-14.md";OLD_LOCK=R/"het_next_l0_ph1_intel_execution_r8a3_lock.json";CHAIN={"runner_sha256":RUNNER,"verifier_sha256":SELF,"prereg_sha256":PREREG,"r8a3_audit_sha256":AUDIT,"r8a3_runner_sha256":OLD_RUNNER,"r8a3_verifier_sha256":OLD_VERIFIER,"r8a3_prereg_sha256":OLD_PREREG,"r8a3_lock_sha256":OLD_LOCK,**{"prior_"+k:v for k,v in frozen.CHAIN.items() if k not in {"runner_sha256","verifier_sha256","prereg_sha256"}}}
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x:object)->bytes:return (json.dumps(x,sort_keys=True,separators=(",",":"))+"\n").encode()
def same(a:object,b:object)->bool:return isinstance(a,str) and isinstance(b,str) and a.casefold()==b.casefold()
def parse(raw:str)->list[str]:
 k=C.WinDLL("kernel32",use_last_error=True);s=C.WinDLL("shell32",use_last_error=True);f=s.CommandLineToArgvW;f.argtypes=(C.c_wchar_p,C.POINTER(C.c_int));f.restype=C.POINTER(C.c_wchar_p);free=k.LocalFree;free.argtypes=(C.c_void_p,);free.restype=C.c_void_p;n=C.c_int();p=f(raw,C.byref(n))
 if not p:raise C.WinError(C.get_last_error())
 try:return [p[i] for i in range(n.value)]
 finally:
  if free(C.cast(p,C.c_void_p)):raise C.WinError(C.get_last_error())
def live()->dict:
 k=C.WinDLL("kernel32",use_last_error=True);g=k.GetCommandLineW;g.argtypes=();g.restype=C.c_wchar_p;raw=g();return {"native_raw":raw,"native_argv":parse(raw),"orig_argv":list(sys.orig_argv),"argv":list(sys.argv),"sys_executable":sys.executable,"sys_prefix":sys.prefix,"base_executable":getattr(sys,"_base_executable",None),"base_prefix":sys.base_prefix,"isolated":sys.flags.isolated,"dont_write_bytecode":sys.dont_write_bytecode,"entry_name":__name__,"entry_spec_is_none":__spec__ is None,"entry_package":__package__,"entry_file":str(SELF),"direct_entry":__name__=="__main__" and __spec__ is None and __package__ in (None,"") and Path(__file__).resolve()==SELF,"python_sha256":sha(VENV_PY),"pyvenv_sha256":sha(PYVENV)}
def identity_valid(x:dict,native:list[str],argv:list[str],script:Path)->bool:
 try:raw=parse(x["native_raw"])==native
 except Exception:return False
 direct=x.get("entry_name")=="__main__" and x.get("entry_spec_is_none") is True and x.get("entry_package") in (None,"") and same(x.get("entry_file"),str(script));return set(x)==frozen.frozen.IDENT_KEYS and raw and x["native_argv"]==x["orig_argv"]==native and x["argv"]==argv and same(x["sys_executable"],str(VENV_PY.resolve())) and same(x["sys_prefix"],str(VENV.resolve())) and same(x["base_executable"],str(ALIAS)) and same(x["base_prefix"],str(BASE_PREFIX)) and x["isolated"]==1 and x["dont_write_bytecode"] is True and x["python_sha256"]=="0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and x["pyvenv_sha256"]=="9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and x["direct_entry"] is True and direct
def identity_mutations(x:dict,native:list[str],argv:list[str],script:Path)->bool:
 cases=[]
 for k,v in (("direct_entry",False),("entry_name","x"),("entry_spec_is_none",False),("entry_package","x"),("entry_file",str(script)+"x"),("sys_executable",str(ALIAS)),("isolated",0),("dont_write_bytecode",False)):
  y=copy.deepcopy(x);y[k]=v;cases.append(y)
 for k in ("argv","native_argv","orig_argv"):
  y=copy.deepcopy(x);y[k]=[*y[k],"extra"];cases.append(y)
 y=copy.deepcopy(x);y["native_raw"]='python -c "x"';cases.append(y);return len(cases)==12 and all(not identity_valid(y,native,argv,script) for y in cases)
def lock_contract()->tuple[bool,dict]:
 obs={k:sha(v) for k,v in CHAIN.items()};l=json.loads(LOCK.read_text());ok=set(l)=={"kind","execution_open","audit_token","one_attempt",*obs} and l.get("kind")=="ph1_intel_execution_r8a4_lock" and l.get("execution_open") is True and l.get("audit_token")==ACK and l.get("one_attempt") is True and all(l.get(k)==v for k,v in obs.items()) and obs.get("r8a3_audit_sha256")=="49131e25d27e2272a19f5890d11221b7d0e06ae90294c148d34e9d58ebff9131";return ok,obs
def extension(result:dict,obs:dict)->bool:
 x=result.get("authorization",{}).get("r8a4_authorization",{});return set(x)=={"kind","lock_sha256","observed","audit_token","invocation","r8p8_pass","r7d_contract","r7a_verification_absent","historical_failures_exact"} and x.get("kind")=="ph1_intel_execution_r8a4_authorization" and x.get("lock_sha256")==sha(LOCK) and x.get("observed")==obs and x.get("audit_token")==ACK and all(x.get(k) is True for k in ("r8p8_pass","r7d_contract","r7a_verification_absent","historical_failures_exact")) and identity_valid(x.get("invocation",{}),RUN_NATIVE,RUN_ARGV,RUNNER) and identity_mutations(x["invocation"],RUN_NATIVE,RUN_ARGV,RUNNER)
def bundle()->tuple[bool,dict|None]:
 try:
  rp,mp,cp=(OUT/n for n in ("result.json","manifest.json","commit.json"));rb=rp.read_bytes();rr=json.loads(rb);mm=json.loads(mp.read_text());cc=json.loads(cp.read_text());row={"name":"result.json","bytes":len(rb),"sha256":hashlib.sha256(rb).hexdigest()};ok=rr.get("kind")=="ph1_intel_execution_r7a" and mm=={"kind":"ph1_intel_execution_r7a_manifest","files":[row]} and cc=={"kind":"ph1_intel_execution_r7a_commit","manifest_sha256":hashlib.sha256(canon(mm)).hexdigest(),"result_sha256":row["sha256"]} and {p.name for p in OUT.iterdir()}=={"result.json","manifest.json","commit.json"} and sum(p.stat().st_size for p in OUT.iterdir())<=16*2**20;return ok,rr
 except Exception:return False,None
def exact_tree(root:Path)->tuple[bool,Path|None]:
 if not root.is_dir():return False,None
 e=sorted(root.rglob("*"));d=[p for p in e if p.is_dir()];f=[p for p in e if p.is_file()];dd=[p for p in root.iterdir() if p.is_dir()];rf=[p for p in root.iterdir() if p.is_file()];ok=len(e)==2 and len(d)==len(dd)==1 and not rf and len(f)==1 and f[0].parent==d[0] and f[0].name=="failure.json" and not any("inprogress" in p.name.casefold() for p in e);return ok,f[0] if ok else None
def reconstruct(p:Path)->dict:
 row=json.loads(p.read_text());normal={"kind","status","error","traceback","device_opened","backend_evidence","secondary_resource_sample","disposition"};over={"kind","status","error","device_opened","oversized_temp_bytes","oversized_temp_digest","disposition"};valid=set(row) in (normal,over) and row.get("kind")=="ph1_intel_execution_r7a_failure" and row.get("status")=="valid_negative_failure" and isinstance(row.get("error"),str) and row.get("device_opened") is True and row.get("disposition") in ("attempt_archived_create_new","oversized_temp_quarantined_not_retained_failure_bundle") and p.stat().st_size<=16*2**20;rel=str(p.relative_to(R)) if p.is_relative_to(R) else str(p);return {"relative_path":rel,"failure_sha256":sha(p),"failure_bytes":p.stat().st_size,"bundle_files":[{"name":"failure.json","bytes":p.stat().st_size,"sha256":sha(p)}],"bundle_bytes":p.stat().st_size,"bundle_file_count":1,"kind":row.get("kind"),"status":row.get("status"),"disposition":row.get("disposition"),"device_opened":row.get("device_opened"),"valid":valid}
def correlated(w:dict,a:dict)->bool:
 keys={"kind","status","terminal_type","stage","error","device_opened","delegated_return","inherited_failure_count","inherited","correlation_valid","disposition"};return set(w)==keys and w.get("kind")=="ph1_intel_execution_r8a4_failure" and w.get("status")=="correlated_delegated_negative" and w.get("terminal_type")=="delegated_failure" and w.get("stage")=="delegated_return" and w.get("error")=="delegated_execution_nonzero" and isinstance(w.get("delegated_return"),int) and not isinstance(w.get("delegated_return"),bool) and w["delegated_return"]!=0 and w.get("device_opened") is True and w.get("inherited_failure_count")==1 and w.get("correlation_valid") is True and w.get("disposition")=="atomic_bounded_correlated_summary" and a.get("valid") is True and a.get("device_opened") is True and w.get("inherited")==a
def failure_terminal()->str:
 wok,wp=exact_tree(FAILED);bok,bp=exact_tree(BACKEND_FAILED)
 if not wok or OUT.exists() or QUAR.exists() or BACKEND_QUAR.exists():return "invalid"
 try:w=json.loads(wp.read_text())
 except Exception:return "invalid"
 if w.get("terminal_type")=="early_outer_failure":return "early_invalid" if not BACKEND_FAILED.exists() else "invalid"
 if not bok:return "invalid"
 try:a=reconstruct(bp)
 except Exception:return "invalid"
 return "correlated_device_negative" if correlated(w,a) else "invalid"
def topology()->bool:
 observed={p.resolve() for p in FAMILY_PARENT.glob(FAMILY_PREFIX+"*")};allowed={LOCK.resolve()}|{p.resolve() for p in (OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR,VERIFY) if p.exists()};return observed==allowed and not list(FAMILY_PARENT.glob(FAMILY_PREFIX+"*.inprogress*"))
def production_terminal(committed:dict|None=None,num:dict|None=None,auth:bool=True,bundle_ok:bool=True)->str:
 if not topology():return "invalid"
 if OUT.exists():
  if any(p.exists() for p in (FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR)) or committed is None:return "invalid"
  return frozen.frozen.adjudicate_committed(committed,num or {},auth,bundle_ok)
 return failure_terminal()
def mutation_harness()->dict:
 global FAMILY_PARENT,FAMILY_PREFIX,LOCK,OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR,VERIFY
 saved=(FAMILY_PARENT,FAMILY_PREFIX,LOCK,OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR,VERIFY);got={}
 with tempfile.TemporaryDirectory(prefix="r8a4_matrix_") as td:
  root=Path(td);FAMILY_PARENT=root;FAMILY_PREFIX="case";LOCK=root/"case_lock.json";OUT=root/"case_out";FAILED=root/"case_failed";BACKEND_FAILED=root/"case_backend";QUAR=root/"case_quar";BACKEND_QUAR=root/"case_bquar";VERIFY=root/"case_verify.json";LOCK.write_text("{}")
  def clear():
   for p in (OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR,VERIFY):
    if p.exists():shutil.rmtree(p) if p.is_dir() else p.unlink()
   for p in root.glob("case*.inprogress*"):
    shutil.rmtree(p) if p.is_dir() else p.unlink()
  def failure()->tuple[Path,Path,dict,dict]:
   b=BACKEND_FAILED/"attempt_b";b.mkdir(parents=True);br={"kind":"ph1_intel_execution_r7a_failure","status":"valid_negative_failure","error":"device","traceback":"x","device_opened":True,"backend_evidence":None,"secondary_resource_sample":{"available":1,"telemetry_error":None},"disposition":"attempt_archived_create_new"};bp=b/"failure.json";bp.write_bytes(canon(br));a=reconstruct(bp);w=FAILED/"attempt_w";w.mkdir(parents=True);wr={"kind":"ph1_intel_execution_r8a4_failure","status":"correlated_delegated_negative","terminal_type":"delegated_failure","stage":"delegated_return","error":"delegated_execution_nonzero","device_opened":True,"delegated_return":3,"inherited_failure_count":1,"inherited":a,"correlation_valid":True,"disposition":"atomic_bounded_correlated_summary"};wp=w/"failure.json";wp.write_bytes(canon(wr));return bp,wp,br,wr
  def record(name:str,expected:str,committed=None,num=None,auth=True,bundle_ok=True):got[name]=production_terminal(committed,num,auth,bundle_ok)==expected
  try:
   bp,wp,br,wr=failure();record("baseline","correlated_device_negative");clear();bp,wp,br,wr=failure();shutil.rmtree(FAILED);record("missing_wrapper","invalid");clear();bp,wp,br,wr=failure();x=FAILED/"attempt_x";x.mkdir();(x/"failure.json").write_bytes(wp.read_bytes());record("multiple_wrapper","invalid");clear();bp,wp,br,wr=failure();(wp.parent/"extra.bin").write_bytes(b"x");record("wrapper_extra","invalid");clear();bp,wp,br,wr=failure();(FAILED/"orphan.bin").write_bytes(b"x");record("wrapper_orphan","invalid");clear();bp,wp,br,wr=failure();(wp.parent/"extra").mkdir();record("wrapper_dir","invalid");clear();bp,wp,br,wr=failure();(BACKEND_FAILED/"orphan.bin").write_bytes(b"x");record("backend_root_extra","invalid");clear();bp,wp,br,wr=failure();(bp.parent/"extra.bin").write_bytes(b"x");record("backend_attempt_extra","invalid");clear();bp,wp,br,wr=failure();(root/"case_x.inprogress").write_bytes(b"x");record("inprogress","invalid");clear();bp,wp,br,wr=failure();QUAR.mkdir();record("quarantine","invalid");clear();bp,wp,br,wr=failure();shutil.rmtree(BACKEND_FAILED);record("missing_backend","invalid");clear();bp,wp,br,wr=failure();x=BACKEND_FAILED/"attempt_x";x.mkdir();(x/"failure.json").write_bytes(bp.read_bytes());record("multiple_backend","invalid")
   for name,key,value in (("wrong_kind","kind","wrong"),("wrong_status","status","wrong"),("wrong_disposition","disposition","wrong"),("device_false","device_opened",False),("device_type","device_opened","true")):
    clear();bp,wp,br,wr=failure();br[key]=value;bp.write_bytes(canon(br));record(name,"invalid")
   for name,key,value in (("wrong_stage","stage","wrong"),("wrong_correlation","correlation_valid",False),("wrong_hash","inherited",{**wr["inherited"],"failure_sha256":"0"*64})):
    clear();bp,wp,br,wr=failure();wr[key]=value;wp.write_bytes(canon(wr));record(name,"invalid")
   clear();bp,wp,br,wr=failure();shutil.rmtree(FAILED);record("bare_nonzero","invalid");clear();record("success_no_commit","invalid");clear();bp,wp,br,wr=failure();OUT.mkdir();record("mixed_commit_failure","invalid");clear();FAILED.mkdir();w=FAILED/"attempt_w";w.mkdir();early={"kind":"ph1_intel_execution_r8a4_failure","status":"infrastructure_negative","terminal_type":"early_outer_failure","stage":"outer_boundary","error":"x","traceback":"x","device_opened":False,"delegated_return":None,"inherited_failure_count":0,"inherited":None,"disposition":"atomic_bounded_outer_failure"};(w/"failure.json").write_bytes(canon(early));record("early","early_invalid")
   gates={k:True for k in frozen.frozen.GATES};num={k:True for k in frozen.frozen.NUMERICAL};pos={"positive":True,"status":"intel_execution_positive","gates":gates};clear();OUT.mkdir();record("committed_positive","positive",pos,num);clear();OUT.mkdir();st=copy.deepcopy(pos);st["positive"]=False;st["status"]="intel_execution_negative";st["gates"]["stages"]=False;ns=dict(num);ns["positive_schema"]=ns["runner_gates"]=ns["oracle_outputs"]=False;record("stages_negative","allowed_device_negative",st,ns);clear();OUT.mkdir();ct=copy.deepcopy(pos);ct["positive"]=False;ct["status"]="intel_execution_negative";ct["gates"]["counters"]=False;nc=dict(num);nc["positive_schema"]=nc["runner_gates"]=nc["counters"]=False;record("counters_negative","allowed_device_negative",ct,nc)
   for name,gate in (("precheck_negative","controls"),("protocol_negative","ledger_order"),("lifecycle_negative","release"),("resource_negative","resources")):
    clear();OUT.mkdir();rr=copy.deepcopy(pos);rr["positive"]=False;rr["status"]="intel_execution_negative";rr["gates"][gate]=False;nn=dict(num);nn["positive_schema"]=nn["runner_gates"]=False;record(name,"invalid",rr,nn)
  finally:FAMILY_PARENT,FAMILY_PREFIX,LOCK,OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR,VERIFY=saved
 return got
def write(row:dict)->None:
 if VERIFY.exists():raise FileExistsError(VERIFY)
 t=R/(VERIFY.name+"."+uuid.uuid4().hex+".inprogress");data=canon(row)
 try:
  with t.open("xb") as h:h.write(data);h.flush();os.fsync(h.fileno())
  os.link(t,VERIFY);t.unlink()
 finally:
  if t.exists():t.unlink()
def main()->int:
 ident=live();lock_ok,obs=lock_contract();matrix=mutation_harness();checks={"live_invocation":identity_valid(ident,VERIFY_NATIVE,VERIFY_ARGV,SELF),"live_invocation_mutations":identity_mutations(ident,VERIFY_NATIVE,VERIFY_ARGV,SELF),"lock":lock_ok,"historical":frozen.frozen.historical(),"topology":topology(),"committed_adjudicator_mutations":frozen.frozen.adjudicator_mutations(),"production_matrix":len(matrix)==31 and all(matrix.values())};b_ok,result=bundle();state="invalid"
 if b_ok and result is not None and not any(p.exists() for p in (FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR)):
  auth=extension(result,obs);checks["authorization"]=auth;num={}
  if auth:
   if sha(S/"verify_het_next_l0_ph1_intel_execution_r7a.py")!="18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88":raise RuntimeError("numerical_hash")
   import verify_het_next_l0_ph1_intel_execution_r7a as nv
   try:num=nv.verify_dict(result)
   except Exception:num={}
  state=production_terminal(result,num,auth,b_ok);checks["terminal_contract"]=state in ("positive","allowed_device_negative");checks.update({"numerical:"+k:v for k,v in num.items()})
 else:state=production_terminal();checks["terminal_contract"]=state=="correlated_device_negative";checks["bundle_absent"]=not OUT.exists()
 valid=state in ("positive","allowed_device_negative","correlated_device_negative");passed=state=="positive" and all(checks.values());row={"kind":"ph1_intel_execution_r8a4_independent_verification","terminal_state":state,"terminal_valid":valid,"checks":checks,"mutation_matrix":matrix,"pass":passed,"passed":sum(v is True for v in checks.values()),"total":len(checks),"claim":"one real expert/input Intel correctness component only"};write(row);print(json.dumps(row,indent=2));return 0 if passed else 3
if __name__=="__main__":raise SystemExit(main())
