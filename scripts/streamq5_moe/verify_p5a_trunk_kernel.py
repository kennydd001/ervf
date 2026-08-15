from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p5a_trunk_kernel import sha256


R=ROOT/"reports/streamq5_moe"
PREREG=R/"P5A_PHYSICAL_TRUNK_PREREGISTRATION.md";BANK=R/"p5a_trunk_bank_result.json";BANK_VERIFY=R/"p5a_trunk_bank_verification.json";LOCK=R/"p5a_trunk_kernel_input_lock.json";EVAL_LOCK=R/"p5a_trunk_kernel_evaluator_lock.json";EVALUATOR=ROOT/"scripts/streamq5_moe/run_p5a_trunk_kernel.py";VALIDATION=R/"p5a_trunk_kernel_validation.json";TEST=R/"p5a_trunk_kernel_test.json";OUTPUT=R/"p5a_trunk_kernel_verification.json";REPORT=R/"P5A_TRUNK_KERNEL_VERIFICATION.md"


def close(a,b,tol=1e-9):return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)


def main():
    if OUTPUT.exists() or REPORT.exists():raise FileExistsError("refusing to overwrite P5A kernel verification")
    lock=json.loads(LOCK.read_text());el=json.loads(EVAL_LOCK.read_text());bank=json.loads(BANK.read_text());bv=json.loads(BANK_VERIFY.read_text());v=json.loads(VALIDATION.read_text());t=json.loads(TEST.read_text());checks=[]
    def add(name,value):checks.append({"name":name,"pass":bool(value)})
    add("preregistration hash",sha256(PREREG)==lock["preregistration_sha256"])
    add("bank result hash",sha256(BANK)==lock["bank_result_sha256"])
    add("bank verification hash",sha256(BANK_VERIFY)==lock["bank_verification_sha256"] and bv["status"]=="p5a_trunk_bank_verification_pass")
    add("input lock hash",sha256(LOCK)==el["input_lock_sha256"])
    add("evaluator hash",sha256(EVALUATOR)==el["evaluator_sha256"])
    add("split decisions",v["status"]=="p5a_validation_pass_test_authorized" and t["status"]=="p5a_physical_trunk_pass")
    for label,row in (("validation",v),("test",t)):
        host=np.asarray(row["timing"]["host_ms"],float);event=np.asarray(row["timing"]["event_ms"],float)
        add(label+" iteration count",len(host)==lock["iterations"][label] and len(event)==len(host))
        add(label+" host arithmetic",close(host.mean(),row["timing"]["host_stats"]["mean"]) and close(np.percentile(host,95),row["timing"]["host_stats"]["p95"]))
        add(label+" event arithmetic",close(event.mean(),row["timing"]["event_stats"]["mean"]) and close(np.percentile(event,50),row["timing"]["event_stats"]["p50"]))
        recomputed={"verified_bank":bv["status"]=="p5a_trunk_bank_verification_pass","physical_bytes_exact":bank["aggregate"]["bytes"]==1248931840 and bank["aggregate"]["weights"]==1229717504,"co_resident_and_scratch":row["physical"]["free_after_bytes"]>=lock["co_resident_bytes"]["minimum_scratch"],"correctness_15_records":len(row["correctness"])==15 and all(c["finite"] and c["max_abs"]<=lock["gates"]["max_abs"] and c["relative_l2"]<=lock["gates"]["relative_l2"] for c in row["correctness"]),"finite":bool(np.isfinite(host).all()) and bool(np.isfinite(event).all()),"host_mean_le_30":host.mean()<=lock["gates"]["host_mean_ms_max"],"host_p95_le_35":np.percentile(host,95)<=lock["gates"]["host_p95_ms_max"],"event_host_ratio_ge_0_90":np.percentile(event,50)/np.percentile(host,50)>=lock["gates"]["event_host_p50_ratio_min"],"event_host_ratio_le_1_05":np.percentile(event,50)/np.percentile(host,50)<=lock["gates"]["event_host_p50_ratio_max"]}
        add(label+" gates recomputed",recomputed==row["gates"] and all(recomputed.values()))
        add(label+" physical accounting",row["physical"]["trunk_bank_bytes"]==bank["aggregate"]["bytes"] and row["physical"]["expert_cache_bytes"]==lock["co_resident_bytes"]["expert_cache"] and row["physical"]["kv_bytes"]==lock["co_resident_bytes"]["kv"])
    passed=sum(c["pass"] for c in checks);payload={"kind":"streamq5_moe_p5a_independent_trunk_kernel_verification","status":"p5a_trunk_kernel_verification_pass" if passed==len(checks) else "p5a_trunk_kernel_verification_fail","checks_passed":passed,"checks_total":len(checks),"checks":checks,"validation_mean_ms":v["timing"]["host_stats"]["mean"],"validation_p95_ms":v["timing"]["host_stats"]["p95"],"test_mean_ms":t["timing"]["host_stats"]["mean"],"test_p95_ms":t["timing"]["host_stats"]["p95"],"claim_boundary":"Physical INT8 projection-plane verification only; missing full decoder operations remain unproven."}
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");REPORT.write_text(f"# P5A trunkkernel-verificatie\n\nStatus: **{payload['status']}**; {passed}/{len(checks)} checks.\n",encoding="utf-8");print(json.dumps(payload,indent=2));
    if passed!=len(checks):raise SystemExit(1)


if __name__=="__main__":main()
