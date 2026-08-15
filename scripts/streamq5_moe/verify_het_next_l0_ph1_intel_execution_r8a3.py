#!/usr/bin/env python3
"""Standalone R8A3 verifier with recursive exact-tree failure adjudication."""
from __future__ import annotations
import copy,ctypes as C,hashlib,json,os,shutil,sys,tempfile,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT/"scripts/streamq5_moe";R=ROOT/"reports/streamq5_moe";sys.path.insert(0,str(S));import verify_het_next_l0_ph1_intel_execution_r8a2 as frozen
SELF=Path(__file__).resolve();RUNNER=S/"run_het_next_l0_ph1_intel_execution_r8a3.py";PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A3_PREREGISTRATION_2026-08-14.md";LOCK=R/"het_next_l0_ph1_intel_execution_r8a3_lock.json";AUDIT=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md";OUT=R/"het_next_l0_ph1_intel_execution_r8a3";FAILED=R/"het_next_l0_ph1_intel_execution_r8a3_failed_attempts";BACKEND_FAILED=R/"het_next_l0_ph1_intel_execution_r8a3_backend_failed_attempts";QUAR=R/"het_next_l0_ph1_intel_execution_r8a3_quarantine";BACKEND_QUAR=R/"het_next_l0_ph1_intel_execution_r8a3_backend_quarantine";VERIFY=R/"het_next_l0_ph1_intel_execution_r8a3_independent_verification.json"
ACK="PH1_INTEL_EXECUTION_R8A3_AFTER_R8P8_PASS_AND_TREE_AUDIT_GO";VENV=ROOT/".venv";VENV_PY=VENV/"Scripts/python.exe";PYVENV=VENV/"pyvenv.cfg";ALIAS=Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe");BASE_PREFIX=Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0");RUN_NATIVE=[str(ALIAS),"-I","-B",str(RUNNER),"--ack",ACK];RUN_ARGV=[str(RUNNER),"--ack",ACK];VERIFY_NATIVE=[str(ALIAS),"-I","-B",str(SELF)];VERIFY_ARGV=[str(SELF)]
OLD_RUNNER=S/"run_het_next_l0_ph1_intel_execution_r8a2.py";OLD_VERIFIER=S/"verify_het_next_l0_ph1_intel_execution_r8a2.py";OLD_PREREG=R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A2_PREREGISTRATION_2026-08-14.md";OLD_LOCK=R/"het_next_l0_ph1_intel_execution_r8a2_lock.json"
CHAIN={"runner_sha256":RUNNER,"verifier_sha256":SELF,"prereg_sha256":PREREG,"r8a2_audit_sha256":AUDIT,"r8a2_runner_sha256":OLD_RUNNER,"r8a2_verifier_sha256":OLD_VERIFIER,"r8a2_prereg_sha256":OLD_PREREG,"r8a2_lock_sha256":OLD_LOCK,**{"prior_"+k:v for k,v in frozen.CHAIN.items() if k not in {"runner_sha256","verifier_sha256","prereg_sha256"}}}
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
 direct=x.get("entry_name")=="__main__" and x.get("entry_spec_is_none") is True and x.get("entry_package") in (None,"") and same(x.get("entry_file"),str(script));return set(x)==frozen.IDENT_KEYS and raw and x["native_argv"]==x["orig_argv"]==native and x["argv"]==argv and same(x["sys_executable"],str(VENV_PY.resolve())) and same(x["sys_prefix"],str(VENV.resolve())) and same(x["base_executable"],str(ALIAS)) and same(x["base_prefix"],str(BASE_PREFIX)) and x["isolated"]==1 and x["dont_write_bytecode"] is True and x["python_sha256"]=="0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14" and x["pyvenv_sha256"]=="9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe" and x["direct_entry"] is True and direct
def identity_mutations(x:dict,native:list[str],argv:list[str],script:Path)->bool:
 cases=[]
 for k,v in (("direct_entry",False),("entry_name","x"),("entry_spec_is_none",False),("entry_package","x"),("entry_file",str(script)+"x"),("sys_executable",str(ALIAS)),("isolated",0),("dont_write_bytecode",False)):
  y=copy.deepcopy(x);y[k]=v;cases.append(y)
 for k in ("argv","native_argv","orig_argv"):
  y=copy.deepcopy(x);y[k]=[*y[k],"extra"];cases.append(y)
 y=copy.deepcopy(x);y["native_raw"]='python -c "x"';cases.append(y);return len(cases)==12 and all(not identity_valid(y,native,argv,script) for y in cases)
def lock_contract()->tuple[bool,dict]:
 obs={k:sha(v) for k,v in CHAIN.items()};l=json.loads(LOCK.read_text());ok=set(l)=={"kind","execution_open","audit_token","one_attempt",*obs} and l.get("kind")=="ph1_intel_execution_r8a3_lock" and l.get("execution_open") is True and l.get("audit_token")==ACK and l.get("one_attempt") is True and all(l.get(k)==v for k,v in obs.items()) and obs.get("r8a2_audit_sha256")=="148ef9cd0cfcb345f43ebe3449e3801776037bbc8d174e4cb130976576590296";return ok,obs
def extension(result:dict,obs:dict)->bool:
 x=result.get("authorization",{}).get("r8a3_authorization",{});return set(x)=={"kind","lock_sha256","observed","audit_token","invocation","r8p8_pass","r7d_contract","r7a_verification_absent","historical_failures_exact"} and x.get("kind")=="ph1_intel_execution_r8a3_authorization" and x.get("lock_sha256")==sha(LOCK) and x.get("observed")==obs and x.get("audit_token")==ACK and all(x.get(k) is True for k in ("r8p8_pass","r7d_contract","r7a_verification_absent","historical_failures_exact")) and identity_valid(x.get("invocation",{}),RUN_NATIVE,RUN_ARGV,RUNNER) and identity_mutations(x["invocation"],RUN_NATIVE,RUN_ARGV,RUNNER)
def bundle()->tuple[bool,dict|None]:
 try:
  rp,mp,cp=(OUT/n for n in ("result.json","manifest.json","commit.json"));rb=rp.read_bytes();rr=json.loads(rb);mm=json.loads(mp.read_text());cc=json.loads(cp.read_text());row={"name":"result.json","bytes":len(rb),"sha256":hashlib.sha256(rb).hexdigest()};ok=rr.get("kind")=="ph1_intel_execution_r7a" and mm=={"kind":"ph1_intel_execution_r7a_manifest","files":[row]} and cc=={"kind":"ph1_intel_execution_r7a_commit","manifest_sha256":hashlib.sha256(canon(mm)).hexdigest(),"result_sha256":row["sha256"]} and {p.name for p in OUT.iterdir()}=={"result.json","manifest.json","commit.json"} and sum(p.stat().st_size for p in OUT.iterdir())<=16*2**20;return ok,rr
 except Exception:return False,None
def exact_tree(root:Path)->tuple[bool,Path|None]:
 if not root.is_dir():return False,None
 entries=sorted(root.rglob("*"));dirs=[p for p in entries if p.is_dir()];files=[p for p in entries if p.is_file()];direct_dirs=[p for p in root.iterdir() if p.is_dir()];root_files=[p for p in root.iterdir() if p.is_file()]
 ok=len(entries)==2 and len(dirs)==len(direct_dirs)==1 and not root_files and len(files)==1 and files[0].parent==dirs[0] and files[0].name=="failure.json" and not any("inprogress" in p.name.casefold() for p in entries)
 return ok,files[0] if ok else None
def reconstruct(p:Path)->dict:
 row=json.loads(p.read_text());normal={"kind","status","error","traceback","device_opened","backend_evidence","secondary_resource_sample","disposition"};over={"kind","status","error","device_opened","oversized_temp_bytes","oversized_temp_digest","disposition"};valid=set(row) in (normal,over) and row.get("kind")=="ph1_intel_execution_r7a_failure" and row.get("status")=="valid_negative_failure" and isinstance(row.get("error"),str) and row.get("device_opened") is True and row.get("disposition") in ("attempt_archived_create_new","oversized_temp_quarantined_not_retained_failure_bundle") and p.stat().st_size<=16*2**20
 return {"relative_path":str(p.relative_to(R)) if p.is_relative_to(R) else str(p),"failure_sha256":sha(p),"failure_bytes":p.stat().st_size,"bundle_files":[{"name":"failure.json","bytes":p.stat().st_size,"sha256":sha(p)}],"bundle_bytes":p.stat().st_size,"bundle_file_count":1,"kind":row.get("kind"),"status":row.get("status"),"disposition":row.get("disposition"),"device_opened":row.get("device_opened"),"valid":valid}
def correlated(w:dict,actual:dict)->bool:
 keys={"kind","status","terminal_type","stage","error","device_opened","delegated_return","inherited_failure_count","inherited","correlation_valid","disposition"};return set(w)==keys and w.get("kind")=="ph1_intel_execution_r8a3_failure" and w.get("status")=="correlated_delegated_negative" and w.get("terminal_type")=="delegated_failure" and w.get("stage")=="delegated_return" and w.get("error")=="delegated_execution_nonzero" and isinstance(w.get("delegated_return"),int) and w["delegated_return"]!=0 and w.get("device_opened") is True and w.get("inherited_failure_count")==1 and w.get("correlation_valid") is True and w.get("disposition")=="atomic_bounded_correlated_summary" and actual.get("valid") is True and actual.get("device_opened") is True and w.get("inherited")==actual
def failure_terminal()->str:
 wok,wp=exact_tree(FAILED);bok,bp=exact_tree(BACKEND_FAILED)
 if not wok or OUT.exists() or QUAR.exists() or BACKEND_QUAR.exists():return "invalid"
 try:w=json.loads(wp.read_text())
 except Exception:return "invalid"
 if w.get("terminal_type")=="early_outer_failure":return "early_invalid" if not BACKEND_FAILED.exists() else "invalid"
 if not bok:return "invalid"
 try:actual=reconstruct(bp)
 except Exception:return "invalid"
 return "correlated_device_negative" if correlated(w,actual) else "invalid"
def topology()->bool:
 observed={p.resolve() for p in R.glob("het_next_l0_ph1_intel_execution_r8a3*")};allowed={LOCK.resolve()}|{p.resolve() for p in (OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR,VERIFY) if p.exists()};return observed==allowed and not list(R.glob("het_next_l0_ph1_intel_execution_r8a3*.inprogress*"))
def failure_fs_mutations()->dict:
 global OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR
 saved=(OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR);results={}
 with tempfile.TemporaryDirectory(prefix="r8a3_terminal_") as td:
  root=Path(td);OUT=root/"out";FAILED=root/"failed";BACKEND_FAILED=root/"backend";QUAR=root/"quar";BACKEND_QUAR=root/"bquar"
  def clear():
   for p in (OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR):
    if p.exists():shutil.rmtree(p)
  def build()->tuple[Path,Path,dict]:
   b=BACKEND_FAILED/"attempt_b";b.mkdir(parents=True);br={"kind":"ph1_intel_execution_r7a_failure","status":"valid_negative_failure","error":"device","traceback":"x","device_opened":True,"backend_evidence":None,"secondary_resource_sample":{"available":1,"telemetry_error":None},"disposition":"attempt_archived_create_new"};bp=b/"failure.json";bp.write_bytes(canon(br));actual=reconstruct(bp);w=FAILED/"attempt_w";w.mkdir(parents=True);wr={"kind":"ph1_intel_execution_r8a3_failure","status":"correlated_delegated_negative","terminal_type":"delegated_failure","stage":"delegated_return","error":"delegated_execution_nonzero","device_opened":True,"delegated_return":3,"inherited_failure_count":1,"inherited":actual,"correlation_valid":True,"disposition":"atomic_bounded_correlated_summary"};wp=w/"failure.json";wp.write_bytes(canon(wr));return bp,wp,wr
  try:
   bp,wp,wr=build();results["baseline"]=failure_terminal()=="correlated_device_negative";clear();bp,wp,wr=build();shutil.rmtree(BACKEND_FAILED);results["missing"]=failure_terminal()=="invalid";clear();bp,wp,wr=build();x=BACKEND_FAILED/"attempt_x";x.mkdir();(x/"failure.json").write_bytes(bp.read_bytes());results["multiple_attempts"]=failure_terminal()=="invalid";clear();bp,wp,wr=build();(bp.parent/"extra.bin").write_bytes(b"x");results["extra_file"]=failure_terminal()=="invalid";clear();bp,wp,wr=build();(BACKEND_FAILED/"orphan.bin").write_bytes(b"x");results["root_orphan"]=failure_terminal()=="invalid";clear();bp,wp,wr=build();(bp.parent/"extra").mkdir();results["extra_dir"]=failure_terminal()=="invalid";clear();bp,wp,wr=build();br=json.loads(bp.read_text());br["kind"]="wrong";bp.write_bytes(canon(br));results["wrong_kind"]=failure_terminal()=="invalid";clear();bp,wp,wr=build();wr["inherited"]["failure_sha256"]="0"*64;wp.write_bytes(canon(wr));results["wrong_correlation"]=failure_terminal()=="invalid";clear();bp,wp,wr=build();br=json.loads(bp.read_text());br["disposition"]="wrong";bp.write_bytes(canon(br));results["wrong_disposition"]=failure_terminal()=="invalid";clear();FAILED.mkdir();w=FAILED/"attempt_w";w.mkdir();early={"kind":"ph1_intel_execution_r8a3_failure","status":"infrastructure_negative","terminal_type":"early_outer_failure","stage":"outer_boundary","error":"x","traceback":"x","device_opened":False,"delegated_return":None,"inherited_failure_count":0,"inherited":None,"disposition":"atomic_bounded_outer_failure"};(w/"failure.json").write_bytes(canon(early));results["early"]=failure_terminal()=="early_invalid";clear();bp,wp,wr=build();wr["delegated_return"]=0;wr["error"]="success_without_commit";wp.write_bytes(canon(wr));results["bare_or_success_without"]=failure_terminal()=="invalid"
  finally:OUT,FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR=saved
 return results
def write(row:dict)->None:
 if VERIFY.exists():raise FileExistsError(VERIFY)
 t=R/(VERIFY.name+"."+uuid.uuid4().hex+".inprogress");data=canon(row)
 try:
  with t.open("xb") as h:h.write(data);h.flush();os.fsync(h.fileno())
  os.link(t,VERIFY);t.unlink()
 finally:
  if t.exists():t.unlink()
def main()->int:
 ident=live();lock_ok,obs=lock_contract();mut=failure_fs_mutations();checks={"live_invocation":identity_valid(ident,VERIFY_NATIVE,VERIFY_ARGV,SELF),"live_invocation_mutations":identity_mutations(ident,VERIFY_NATIVE,VERIFY_ARGV,SELF),"lock":lock_ok,"historical":frozen.historical(),"topology":topology(),"committed_adjudicator_mutations":frozen.adjudicator_mutations(),"failure_filesystem_mutations":set(mut)=={"baseline","missing","multiple_attempts","extra_file","root_orphan","extra_dir","wrong_kind","wrong_correlation","wrong_disposition","early","bare_or_success_without"} and all(mut.values())};b_ok,result=bundle();state="invalid"
 if b_ok and result is not None and not any(p.exists() for p in (FAILED,BACKEND_FAILED,QUAR,BACKEND_QUAR)):
  auth=extension(result,obs);checks["authorization"]=auth;num={}
  if auth:
   if sha(S/"verify_het_next_l0_ph1_intel_execution_r7a.py")!="18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88":raise RuntimeError("numerical_hash")
   import verify_het_next_l0_ph1_intel_execution_r7a as nv
   try:num=nv.verify_dict(result)
   except Exception:num={}
  state=frozen.adjudicate_committed(result,num,auth,b_ok);checks["terminal_contract"]=state in ("positive","allowed_device_negative");checks.update({"numerical:"+k:v for k,v in num.items()})
 else:state=failure_terminal();checks["terminal_contract"]=state=="correlated_device_negative";checks["bundle_absent"]=not OUT.exists()
 valid=state in ("positive","allowed_device_negative","correlated_device_negative");passed=state=="positive" and all(checks.values());row={"kind":"ph1_intel_execution_r8a3_independent_verification","terminal_state":state,"terminal_valid":valid,"checks":checks,"failure_mutations":mut,"pass":passed,"passed":sum(v is True for v in checks.values()),"total":len(checks),"claim":"one real expert/input Intel correctness component only"};write(row);print(json.dumps(row,indent=2));return 0 if passed else 3
if __name__=="__main__":raise SystemExit(main())
