"""CPU-only independent consistency verifier for phase-3 fidelity results."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent; PROMPTS=REPO/"pro_research"/"S100_PHASE3_PROMPTS.json"
TH={"top1_agreement_min":.95,"target_in_top5_min":.995,"mean_ce_delta_max":.05,
    "bootstrap95_mean_ce_delta_max":.075,"p95_ce_delta_max":.25,"mean_coarse_kl_max":.02,
    "p95_coarse_kl_max":.08,"per_domain_top1_min":.90,"per_domain_mean_ce_delta_max":.10}
PROFILES=("qfast","mamba","fast","k5","k4","fast_k5","fast_k4","k1_control")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def close(a,b,tol=2e-6): return math.isclose(float(a),float(b),rel_tol=2e-5,abs_tol=tol)
def bootstrap(ce,top1,rounds=1000):
    rng=np.random.default_rng(20260817); n=ce.size; cm=[]; am=[]
    for _ in range(rounds):
        j=rng.integers(0,n,size=n); cm.append(float(ce[j].mean())); am.append(float(top1[j].mean()))
    return float(np.percentile(cm,95)),float(np.percentile(am,5))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("smoke","full"),required=True); ap.add_argument("--profile",choices=PROFILES,required=True); a=ap.parse_args()
    base=REPO/"pro_research"/"results"/f"S100_PHASE3_FIDELITY_{a.profile.upper()}_{a.mode.upper()}"; result=base.with_suffix(".json")
    if not result.exists(): print(f"FAIL: missing {result}"); return 2
    d=json.loads(result.read_text()); fail=[]
    for k,v in TH.items():
        if k not in d.get("thresholds",{}) or not close(d["thresholds"][k],v,1e-12): fail.append(f"threshold drift {k}")
    mr=d.get("metrics") or {}; metrics=REPO/str(mr.get("path","")); tr=d.get("trace") or {}; trace=REPO/str(tr.get("path",""))
    if not metrics.exists() or sha(metrics)!=mr.get("sha256"): fail.append("metrics SHA/path mismatch")
    if not trace.exists() or sha(trace)!=tr.get("sha256"): fail.append("trace SHA/path mismatch")
    if fail: print(json.dumps({"status":"FAIL","failures":fail},indent=2)); return 1
    with np.load(metrics,allow_pickle=False) as z:
        top1=z["top1_agree"].reshape(-1).astype(bool); in5=z["target_in_top5"].reshape(-1).astype(bool)
        rank=z["target_rank"].reshape(-1); ce=z["ce_delta"].reshape(-1).astype(float); kl=z["coarse_kl"].reshape(-1).astype(float)
        overlap=z["top5_overlap"].reshape(-1).astype(float)
    if not all(np.isfinite(x).all() for x in (ce,kl,overlap)): fail.append("non-finite metrics")
    prompts=json.loads(PROMPTS.read_text())["prompts"]; prompts=prompts[:8] if a.mode=="smoke" else prompts
    per=int(top1.size//len(prompts)); domains=np.asarray([p["domain"] for p in prompts for _ in range(per)])
    ce95,acc05=bootstrap(ce,top1)
    fresh={"tokens":int(top1.size),"top1_agreement":float(top1.mean()),"target_in_top5":float(in5.mean()),"mean_top5_overlap":float(overlap.mean()),
           "mean_target_rank":float(rank.mean()),"p99_target_rank":float(np.percentile(rank,99)),"max_target_rank":int(rank.max()),
           "mean_ce_delta":float(ce.mean()),"p95_ce_delta":float(np.percentile(ce,95)),"mean_coarse_kl":float(kl.mean()),"p95_coarse_kl":float(np.percentile(kl,95))}
    rep=d.get("summary") or {}
    for k,v in fresh.items():
        if not close(v,rep.get(k,math.nan)): fail.append(f"summary mismatch {k}")
    if not close(ce95,(rep.get("bootstrap") or {}).get("mean_ce_delta_p95",math.nan)): fail.append("bootstrap CE mismatch")
    dg_top=True; dg_ce=True
    for name in sorted(set(domains.tolist())):
        m=domains==name; t=float(top1[m].mean()); c=float(ce[m].mean()); rr=(d.get("per_domain") or {}).get(name) or {}
        if not close(t,rr.get("top1_agreement",math.nan)): fail.append(f"domain top1 mismatch {name}")
        if not close(c,rr.get("mean_ce_delta",math.nan)): fail.append(f"domain CE mismatch {name}")
        dg_top = bool(dg_top and bool(t >= TH["per_domain_top1_min"]))
        dg_ce = bool(dg_ce and bool(c <= TH["per_domain_mean_ce_delta_max"]))
    deterministic=bool(rep.get("deterministic_anchor_repeat")); finite=bool(rep.get("all_finite")) and np.isfinite(ce).all() and np.isfinite(kl).all()
    gates={"F1_top1_agreement":bool(fresh["top1_agreement"]>=TH["top1_agreement_min"]),
           "F2_target_in_top5":bool(fresh["target_in_top5"]>=TH["target_in_top5_min"]),
           "F3_mean_ce_delta":bool(fresh["mean_ce_delta"]<=TH["mean_ce_delta_max"]),
           "F4_bootstrap95_mean_ce_delta":bool(ce95<=TH["bootstrap95_mean_ce_delta_max"]),
           "F5_p95_ce_delta":bool(fresh["p95_ce_delta"]<=TH["p95_ce_delta_max"]),
           "F6_mean_coarse_kl":bool(fresh["mean_coarse_kl"]<=TH["mean_coarse_kl_max"]),
           "F7_p95_coarse_kl":bool(fresh["p95_coarse_kl"]<=TH["p95_coarse_kl_max"]),
           "F8_every_domain_top1":bool(dg_top),"F9_every_domain_mean_ce_delta":bool(dg_ce),
           "F10_deterministic_anchor_repeat":bool(deterministic),"F11_all_finite":bool(finite)}
    rg=d.get("gates") or {}
    for k,v in gates.items():
        if bool(rg.get(k)) != bool(v): fail.append(f"gate mismatch {k}")
    normal=all(bool(v) for v in gates.values())
    expected=("control_failed_as_expected" if not normal else "control_failed_to_fail") if a.profile=="k1_control" else ("v18_fidelity_candidate" if normal and rg.get("H1_k1_control_failed_as_expected") is True else "v18_fidelity_failed")
    if d.get("status")!=expected: fail.append(f"status mismatch {d.get('status')} != {expected}")
    print(json.dumps({"status":"PASS" if not fail else "FAIL","result_status":d.get("status"),"fresh_summary":fresh,"fresh_gates":gates,"failures":fail},indent=2,allow_nan=False)); return 0 if not fail else 1
if __name__=="__main__": sys.exit(main())
