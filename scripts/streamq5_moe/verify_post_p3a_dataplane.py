from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from moe_lab.reporting import ROOT


R=ROOT/"reports/streamq5_moe";OUTPUT=R/"post_p3a_dataplane_verification.json";REPORT=R/"POST_P3A_DATAPLANE_VERIFICATION.md"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*2**20),b""):h.update(chunk)
    return h.hexdigest()


def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9)


def main():
    if OUTPUT.exists() or REPORT.exists():raise FileExistsError("refusing to overwrite post-P3A verification")
    checks=[]
    def add(name,value):checks.append({"name":name,"pass":bool(value)})
    p3b=json.loads((R/"p3b_dispatch_gap_validation.json").read_text());p3b_lock=json.loads((R/"p3b_dispatch_input_lock.json").read_text());p3b_el=json.loads((R/"p3b_dispatch_evaluator_lock.json").read_text())
    add("P3B provenance",sha(R/"P3B_DISPATCH_GAP_PREREGISTRATION.md")==p3b_lock["preregistration_sha256"] and sha(R/"p3b_dispatch_input_lock.json")==p3b_el["input_lock_sha256"] and sha(ROOT/"scripts/streamq5_moe/run_p3b_dispatch_gap.py")==p3b_el["evaluator_sha256"])
    add("P3B decision",p3b["status"]=="p3b_validation_closed_test_unopened" and not (R/"p3b_dispatch_gap_test.json").exists())
    add("P3B exact output",p3b["correctness"]["exact_output_match"] and p3b["gates"]["exact_output_match"])
    add("P3B ratio arithmetic",close(p3b["ratios"]["graph_to_eager_host_p50"],p3b["graph"]["host_stats"]["p50"]/p3b["eager"]["host_stats"]["p50"]) and not p3b["gates"]["graph_host_p50_ratio_le_0_90"])
    summaries={"P3B":{"validation_status":p3b["status"],"eager_p50_ms":p3b["eager"]["host_stats"]["p50"],"graph_p50_ms":p3b["graph"]["host_stats"]["p50"]}}
    for phase in ("validation","test"):
        row=json.loads((R/f"p4a_causal_async_{phase}.json").read_text());serial=np.concatenate([np.asarray(x["misses"],dtype=np.int64) for x in row["serial"].values()]);async_m=np.concatenate([np.asarray(x["misses"],dtype=np.int64) for x in row["causal_async"].values()]);times=np.concatenate([np.asarray(x["wall_ms"],float) for x in row["causal_async"].values()])
        add(f"P4A {phase} miss equality",np.array_equal(serial,async_m) and row["gates"]["async_misses_match_serial"])
        add(f"P4A {phase} timing arithmetic",close(times.mean(),row["aggregate"]["causal_async"]["wall_ms"]["mean"]) and close(np.percentile(times,95),row["aggregate"]["causal_async"]["wall_ms"]["p95"]))
        add(f"P4A {phase} exact outputs",all(x["exact"] for x in row["correctness"].values()) and row["gates"]["outputs_exact"])
        add(f"P4A {phase} byte arithmetic",row["physical"]["async_copied_records"]==int(async_m.sum()) and row["physical"]["async_copied_bytes"]==int(async_m.sum())*3035136)
    p4av=json.loads((R/"p4a_causal_async_validation.json").read_text());p4at=json.loads((R/"p4a_causal_async_test.json").read_text())
    add("P4A split decision",p4av["status"]=="p4a_validation_pass_test_authorized" and p4at["status"]=="p4a_causal_async_closed" and not p4at["gates"]["aggregate_async_mean_le_20"] and all(v for k,v in p4at["gates"].items() if k!="aggregate_async_mean_le_20"))
    summaries["P4A"]={"validation_mean_ms":p4av["aggregate"]["causal_async"]["wall_ms"]["mean"],"validation_p95_ms":p4av["aggregate"]["causal_async"]["wall_ms"]["p95"],"test_mean_ms":p4at["aggregate"]["causal_async"]["wall_ms"]["mean"],"test_p95_ms":p4at["aggregate"]["causal_async"]["wall_ms"]["p95"],"test_speedup":p4at["aggregate"]["speedup"],"status":p4at["status"]}
    for label,stem,status in (("P4B","p4b_rendezvous_async","p4b_validation_closed_test_unopened"),("P4C","p4c_fused_overlap","p4c_validation_closed_test_unopened"),("P4D","p4d_packed_overlap","p4d_validation_closed_test_unopened")):
        row=json.loads((R/f"{stem}_validation.json").read_text());async_key={"P4B":"rendezvous_async","P4C":"fused_async","P4D":"packed_async"}[label]
        add(label+" validation closed",row["status"]==status and not (R/f"{stem}_test.json").exists())
        add(label+" exact outputs",row["gates"]["outputs_exact"] and all(x["exact"] for x in row["correctness"].values()))
        add(label+" physical accounting",row["gates"]["copy_records_exact"] and row["gates"]["copy_bytes_exact"] and row["gates"]["device_co_resident_and_scratch"])
        summaries[label]={"status":row["status"],"mean_ms":row["aggregate"][async_key]["wall_ms"]["mean"],"p95_ms":row["aggregate"][async_key]["wall_ms"]["p95"],"speedup":row["aggregate"]["speedup"]}
    capture=json.loads((R/"p4d_route_capture_result.json").read_text());lock=json.loads((R/"p4d_route_input_lock.json").read_text())
    add("P4D fresh route controls",capture["status"]=="route_capture_complete" and all(capture["controls"].values()) and lock["exact_128_context_disjoint_from_prior_decisions"])
    add("P4D 48 artifact hashes",len(capture["manifests"])==48 and all(sha(ROOT/row["artifact"])==row["artifact_sha256"] for row in capture["manifests"].values()))
    passed=sum(x["pass"] for x in checks);payload={"kind":"streamq5_moe_post_p3a_independent_verification","status":"post_p3a_verification_pass" if passed==len(checks) else "post_p3a_verification_fail","checks_passed":passed,"checks_total":len(checks),"checks":checks,"summaries":summaries,"claim_boundary":"Verifies P3B/P4A-D physical expert-dataplane experiments; does not expand their claim boundaries."}
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");REPORT.write_text(f"# Post-P3A dataplane-verificatie\n\nStatus: **{payload['status']}**; {passed}/{len(checks)} checks.\n",encoding="utf-8");print(json.dumps(payload,indent=2));
    if passed!=len(checks):raise SystemExit(1)


if __name__=="__main__":main()
