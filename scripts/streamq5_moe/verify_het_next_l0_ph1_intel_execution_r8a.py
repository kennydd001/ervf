#!/usr/bin/env python3
"""Independent R8A authorization, physical bundle, and negative-evidence verifier."""
from __future__ import annotations

import hashlib, json, os, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT/"scripts/streamq5_moe"; R = ROOT/"reports/streamq5_moe"
OUT = R/"het_next_l0_ph1_intel_execution_r8a"; FAILED = R/"het_next_l0_ph1_intel_execution_r8a_failed_attempts"; BACKEND_FAILED = R/"het_next_l0_ph1_intel_execution_r8a_backend_failed_attempts"; QUAR = R/"het_next_l0_ph1_intel_execution_r8a_quarantine"; BACKEND_QUAR = R/"het_next_l0_ph1_intel_execution_r8a_backend_quarantine"; VERIFY = R/"het_next_l0_ph1_intel_execution_r8a_independent_verification.json"
LOCK = R/"het_next_l0_ph1_intel_execution_r8a_lock.json"; PREREG = R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A_PREREGISTRATION_2026-08-14.md"; RUNNER = S/"run_het_next_l0_ph1_intel_execution_r8a.py"
R8P8_RESULT = R/"het_next_l0_ph1_intel_execution_r8p8_static_preflight.json"; R8P8_MANIFEST = R/"het_next_l0_ph1_intel_execution_r8p8_static_preflight.manifest.json"; R8P8_COMMIT = R/"het_next_l0_ph1_intel_execution_r8p8_static_preflight.commit.json"; R8P8_VERIFY = R/"het_next_l0_ph1_intel_execution_r8p8_independent_verification.json"; R8P8_AUDIT = R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P8_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"; R8P8_LOCK = R/"het_next_l0_ph1_intel_execution_r8p8_lock.json"
R7D_LOCK = R/"het_next_l0_ph1_intel_execution_r7d_lock.json"; R7C2_RESULT = R/"het_next_l0_ph1_intel_execution_r7c2_static_preflight.json"; R7A_RESULT = R/"het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json"; R7P_RESULT = R/"het_next_l0_ph1_intel_execution_r7p_static_preflight.json"; R7A_VERIFY = R/"het_next_l0_ph1_intel_execution_r7a_independent_verification.json"
R7D1_ROOT = R/"het_next_l0_ph1_intel_execution_r7d1_failed_attempts"; R7D1_FAILURE = R7D1_ROOT/"attempt_7c45ba0bda09470eba7145ef75281ea3/failure.json"; R7D1_DIAG = R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md"; R8P6_ROOT = R/"het_next_l0_ph1_intel_execution_r8p6_failed_attempts"; R8P6_FAILURE = R8P6_ROOT/"attempt_71e198678f004a56a6912d07a4187dfd/failure.json"; R8P6_DIAG = R/"HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_DIRECT_ENTRY_FAILURE_DIAGNOSIS_2026-08-14.md"
ACK = "PH1_INTEL_EXECUTION_R8A_AFTER_R8P8_PASS_AND_SOURCE_AUDIT_GO"; R8P8_SHA = "5e77ef9fd4d5a374bc56fd51707ba6bb1a5353e9276a96d3190af589d1d636b0"; R8P8_MANIFEST_SHA = "b6b702846753ffc32dd94802752d847284a16ddf9a6a530047544febbe025a34"; R8P8_COMMIT_SHA = "4431a49104e46384ca8927b5a38e6c85e3d729ca2565c7474cbfc623cd9b3a89"; R8P8_VERIFY_SHA = "577881b17a47b2d1208687192ff0582d2bfcfa36c84e7abd85ffe38a769148c8"; PREP_SHA = "f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49"
R8P8_CHECKS = {"base_clean", "closed_pending", "cpu_preparation", "current_transactions", "entry_mutations", "explicit_topology", "failure_simulation", "hash_bindings", "local_entry_identity", "r7d1_failure", "r8p6_failure", "runtime", "runtime_lock", "runtime_mutations", "start_ram", "static_boundary", "topology", "wheel_records"}; R8P8_VERIFY_CHECKS = {"bundle", "explicit_topology", "independent_output_transactions", "live_identity", "live_identity_mutations", "lock", "preparation", "r8p6_failure", "result", "result_mutations", "runtime", "static_boundary", "topology", "wheel_records"}
CHAIN = {"runner_sha256":RUNNER, "verifier_sha256":Path(__file__), "prereg_sha256":PREREG, "r8p8_result_sha256":R8P8_RESULT, "r8p8_manifest_sha256":R8P8_MANIFEST, "r8p8_commit_sha256":R8P8_COMMIT, "r8p8_verification_sha256":R8P8_VERIFY, "r8p8_audit_sha256":R8P8_AUDIT, "r8p8_lock_sha256":R8P8_LOCK, "r7d_runner_sha256":S/"run_het_next_l0_ph1_intel_execution_r7d.py", "r7d_verifier_sha256":S/"verify_het_next_l0_ph1_intel_execution_r7d.py", "r7d_lock_sha256":R7D_LOCK, "r7c2_result_sha256":R7C2_RESULT, "r7a_result_sha256":R7A_RESULT, "r7p_result_sha256":R7P_RESULT, "r7d1_failure_sha256":R7D1_FAILURE, "r7d1_diagnosis_sha256":R7D1_DIAG, "r8p6_failure_sha256":R8P6_FAILURE, "r8p6_diagnosis_sha256":R8P6_DIAG, "physical_runner_sha256":S/"run_het_next_l0_ph1_intel_execution_r7a.py", "physical_verifier_sha256":S/"verify_het_next_l0_ph1_intel_execution_r7a.py", "backend_sha256":S/"het_next_l0_ph1_intel_execution_r6_backend.py", "common_sha256":S/"het_next_l0_ph1_intel_execution_r6_common.py"}

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canon(x: object) -> bytes: return (json.dumps(x,sort_keys=True,separators=(",",":"))+"\n").encode()
def r8p8() -> bool:
    if [sha(x) for x in (R8P8_RESULT,R8P8_MANIFEST,R8P8_COMMIT,R8P8_VERIFY)] != [R8P8_SHA,R8P8_MANIFEST_SHA,R8P8_COMMIT_SHA,R8P8_VERIFY_SHA]: return False
    rb=R8P8_RESULT.read_bytes(); row=json.loads(rb); manifest=json.loads(R8P8_MANIFEST.read_text()); commit=json.loads(R8P8_COMMIT.read_text()); vr=json.loads(R8P8_VERIFY.read_text())
    return manifest=={"files":[{"bytes":len(rb),"name":R8P8_RESULT.name,"sha256":R8P8_SHA}],"kind":"ph1_intel_execution_r8p8_static_preflight_manifest"} and commit=={"kind":"ph1_intel_execution_r8p8_static_preflight_commit","manifest_sha256":R8P8_MANIFEST_SHA,"result_sha256":R8P8_SHA} and row.get("kind")=="ph1_intel_execution_r8p8_static_preflight" and row.get("pass") is True and row.get("passed")==row.get("total")==18 and set(row.get("checks",{}))==R8P8_CHECKS and all(v is True for v in row["checks"].values()) and row.get("preparation_digest")==PREP_SHA and all(row.get(k) is False for k in ("model_forward","compiler_opened","opencl_opened","device_opened")) and vr.get("kind")=="ph1_intel_execution_r8p8_independent_verification" and vr.get("pass") is True and vr.get("passed")==vr.get("total")==14 and set(vr.get("checks",{}))==R8P8_VERIFY_CHECKS and all(v is True for v in vr["checks"].values()) and vr.get("result_sha256")==R8P8_SHA and vr.get("manifest_sha256")==R8P8_MANIFEST_SHA and vr.get("commit_sha256")==R8P8_COMMIT_SHA and all(vr.get(k) is False for k in ("model_forward","compiler_opened","opencl_opened","device_opened"))
def exact_failure(root: Path,path: Path,digest: str,kind: str,size: int) -> bool:
    files=sorted(p.resolve() for p in root.rglob("*") if p.is_file()); dirs=sorted(p.resolve() for p in root.iterdir() if p.is_dir()) if root.exists() else []; row=json.loads(path.read_text()) if files==[path.resolve()] and sha(path)==digest else {}
    return len(dirs)==1 and files==[path.resolve()] and path.stat().st_size==size and row.get("kind")==kind and row.get("device_opened") is False and isinstance(row.get("error"),str)
def r7d() -> bool:
    if sha(R7D_LOCK)!="fa2a514af78ac75cd94376f8d04d801fd1dfa27592b595bf42a662bab3e15658" or sha(S/"verify_het_next_l0_ph1_intel_execution_r7d.py")!="8fa44558412eed80891d013fda8a08881e65ca30caf35062c9b7428a02d10fb4": return False
    sys.path.insert(0,str(S)); import verify_het_next_l0_ph1_intel_execution_r7d as frozen
    lock=json.loads(R7D_LOCK.read_text()); observed={k:sha(v) for k,v in frozen.CHAIN.items()}; r7c2=json.loads(R7C2_RESULT.read_text()); r7a=json.loads(R7A_RESULT.read_text()); r7p=json.loads(R7P_RESULT.read_text())
    return set(lock)=={"kind","execution_open","audit_token","physical_output","physical_verifier",*observed} and lock.get("kind")=="ph1_intel_execution_r7d_lock" and lock.get("execution_open") is True and lock.get("audit_token")==frozen.ACK and lock.get("physical_output")=="het_next_l0_ph1_intel_execution_r7a" and lock.get("physical_verifier")=="verify_het_next_l0_ph1_intel_execution_r7d.py" and all(lock.get(k)==v for k,v in observed.items()) and observed.get("r7c2_result_sha256")==frozen.R7C2_SHA and observed.get("authorization_result_sha256")==frozen.R7A_SHA and observed.get("r7p_result_sha256")==frozen.R7P_SHA and frozen.r7c2_pass9(r7c2) and frozen.r7a_pass7(r7a) and frozen.r7p_pass18(r7p) and not R7A_VERIFY.exists()
def authorization(result: dict) -> bool:
    observed={k:sha(v) for k,v in CHAIN.items()}; lock=json.loads(LOCK.read_text()); ext=result.get("authorization",{}).get("r8a_authorization",{})
    exact=set(lock)=={"kind","execution_open","audit_token","one_attempt",*observed} and lock.get("kind")=="ph1_intel_execution_r8a_lock" and lock.get("execution_open") is True and lock.get("audit_token")==ACK and lock.get("one_attempt") is True and all(lock.get(k)==v for k,v in observed.items())
    return exact and set(ext)=={"lock_sha256","observed","audit_token","invocation","r8p8_pass","r7d_contract","r7a_verification_absent","r7d1_failure_sha256","r8p6_failure_sha256"} and ext.get("lock_sha256")==sha(LOCK) and ext.get("observed")==observed and ext.get("audit_token")==ACK and ext.get("r8p8_pass") is ext.get("r7d_contract") is ext.get("r7a_verification_absent") is True and ext.get("r7d1_failure_sha256")=="88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0" and ext.get("r8p6_failure_sha256")=="03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372" and isinstance(ext.get("invocation"),dict)
def bundle() -> tuple[bool,dict|None]:
    try:
        rp,mp,cp=(OUT/n for n in ("result.json","manifest.json","commit.json")); rb=rp.read_bytes(); result=json.loads(rb); manifest=json.loads(mp.read_text()); commit=json.loads(cp.read_text()); row={"name":"result.json","bytes":len(rb),"sha256":hashlib.sha256(rb).hexdigest()}; mb=canon(manifest)
        okay=result.get("kind")=="ph1_intel_execution_r7a" and manifest=={"kind":"ph1_intel_execution_r7a_manifest","files":[row]} and commit=={"kind":"ph1_intel_execution_r7a_commit","manifest_sha256":hashlib.sha256(mb).hexdigest(),"result_sha256":row["sha256"]} and {p.name for p in OUT.iterdir()}=={"result.json","manifest.json","commit.json"} and sum(p.stat().st_size for p in OUT.iterdir())<=16*2**20
        return okay,result
    except Exception:return False,None
def atomic_output(row: dict) -> None:
    if VERIFY.exists(): raise FileExistsError(VERIFY)
    temp=R/(VERIFY.name+"."+uuid.uuid4().hex+".inprogress"); data=canon(row)
    try:
        with temp.open("xb") as h: h.write(data); h.flush(); os.fsync(h.fileno())
        os.link(temp,VERIFY); temp.unlink()
    finally:
        if temp.exists(): temp.unlink()
def failure_adjudication() -> tuple[dict,int]:
    rows=[]
    for root,label in ((FAILED,"outer"),(BACKEND_FAILED,"backend")):
        for p in sorted(root.rglob("failure.json")) if root.exists() else []:
            try:
                x=json.loads(p.read_text()); rows.append({"label":label,"path":str(p.relative_to(R)),"sha256":sha(p),"bytes":p.stat().st_size,"kind":x.get("kind"),"status":x.get("status"),"device_opened":x.get("device_opened"),"valid":p.stat().st_size<=16*2**20 and isinstance(x.get("error"),str) and x.get("status")=="valid_negative_failure" and isinstance(x.get("device_opened"),bool)})
            except Exception: rows.append({"label":label,"valid":False})
    valid=len(rows) in (1,2) and all(x.get("valid") is True for x in rows) and not OUT.exists() and not QUAR.exists() and not BACKEND_QUAR.exists()
    out={"kind":"ph1_intel_execution_r8a_independent_verification","adjudication":"valid_committed_negative_failure" if valid else "invalid_or_incomplete_failure","checks":{"r8p8":r8p8(),"r7d":r7d(),"r7d1_failure":exact_failure(R7D1_ROOT,R7D1_FAILURE,"88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0","ph1_intel_execution_r7c2_failure",931),"r8p6_failure":exact_failure(R8P6_ROOT,R8P6_FAILURE,"03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372","ph1_intel_execution_r8p6_failure",2986),"failure_bundle":valid},"failure_rows":rows,"pass":False,"valid_negative":valid,"claim":"one real expert/input Intel correctness component only"}; out["passed"]=sum(v is True for v in out["checks"].values()); out["total"]=len(out["checks"]); return out,3
def main() -> int:
    if VERIFY.exists(): return 3
    ok,result=bundle()
    if not ok or result is None:
        row,code=failure_adjudication(); atomic_output(row); print(json.dumps(row,indent=2)); return code
    checks={"bundle":ok,"authorization":authorization(result),"r8p8":r8p8(),"r7d":r7d(),"r7d1_failure":exact_failure(R7D1_ROOT,R7D1_FAILURE,"88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0","ph1_intel_execution_r7c2_failure",931),"r8p6_failure":exact_failure(R8P6_ROOT,R8P6_FAILURE,"03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372","ph1_intel_execution_r8p6_failure",2986),"fresh_failure_paths":not FAILED.exists() and not BACKEND_FAILED.exists() and not QUAR.exists() and not BACKEND_QUAR.exists()}
    if all(checks.values()):
        if sha(S/"verify_het_next_l0_ph1_intel_execution_r7a.py")!="18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88": raise RuntimeError("numerical_verifier_hash")
        sys.path.insert(0,str(S)); import verify_het_next_l0_ph1_intel_execution_r7a as numerical
        checks.update({"numerical:"+k:v for k,v in numerical.verify_dict(result).items()})
    passed=all(checks.values()); row={"kind":"ph1_intel_execution_r8a_independent_verification","adjudication":"positive" if passed else "committed_negative","checks":checks,"pass":passed,"valid_negative":not passed,"passed":sum(v is True for v in checks.values()),"total":len(checks),"result_sha256":sha(OUT/"result.json"),"claim":"one real expert/input Intel correctness component only"}; atomic_output(row); print(json.dumps(row,indent=2)); return 0 if passed else 3
if __name__=="__main__": raise SystemExit(main())
