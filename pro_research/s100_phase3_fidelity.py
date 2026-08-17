"""Teacher-forced and greedy V18-fidelity gate for phase-3 profiles."""
from __future__ import annotations
import argparse,hashlib,json,math,os,tempfile,traceback
from pathlib import Path
from typing import Any
import numpy as np
from common import (REPO,archive_existing,environment_snapshot,first_divergence,require_model_dir,
                    sha256_file,utc_now,write_json_atomic,write_text_atomic)
from diag_component_marginals_graph import _prefill,_reset_exact_state,_run
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from s100_phase3_profiles import FIDELITY_PROFILES
from s100_phase3_runtime import build_v18_runtime,public_bundle_record
PROMPTS=REPO/"pro_research"/"S100_PHASE3_PROMPTS.json"
PREREG=REPO/"pro_research"/"S100_PHASE3_FIDELITY_PREREGISTRATION.md"
TH={"top1_agreement_min":.95,"target_in_top5_min":.995,"mean_ce_delta_max":.05,
    "bootstrap95_mean_ce_delta_max":.075,"p95_ce_delta_max":.25,"mean_coarse_kl_max":.02,
    "p95_coarse_kl_max":.08,"per_domain_top1_min":.90,"per_domain_mean_ce_delta_max":.10}

def _paths(mode,profile):
    b=REPO/"pro_research"/"results"/f"S100_PHASE3_FIDELITY_{profile.upper()}_{mode.upper()}"
    return b.with_suffix(".json"),b.with_suffix(".npz"),b.with_suffix(".md")
def _trace_paths(mode):
    p=REPO/"pro_research"/"results"/f"S100_PHASE3_V18_TRACE_{mode.upper()}.npz"; return p,p.with_suffix(".json")
def _prompts(mode):
    p=list(json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]); return p[:8] if mode=="smoke" else p
def _lse(cp,x):
    m=cp.max(x); return float((m+cp.log(cp.exp(x-m).sum())).item())
def _snap(cp,logits,target,base_ids):
    if not bool(cp.isfinite(logits).all().item()): raise RuntimeError("candidate non-finite logits")
    lse=_lse(cp,logits); tv=float(logits[int(target)].item()); rank=int(cp.count_nonzero(logits>tv).item())
    idx=cp.argpartition(logits,-5)[-5:]; idx=idx[cp.argsort(-logits[idx])]
    top5=cp.asnumpy(idx).astype(np.int32,copy=False)
    vals=cp.asnumpy(logits[cp.asarray(base_ids.astype(np.int64,copy=False))]).astype(np.float64,copy=False)
    return tv-lse,rank,top5,vals-lse
def _advance(rt,token):
    slot=int(rt._ring_i); rt.step_graph(int(token)); return int(rt.ring_harvest(slot,1)[0])
def _write_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True); archive_existing(path)
    with tempfile.NamedTemporaryFile("wb",dir=path.parent,delete=False,suffix=".tmp") as f:
        np.savez_compressed(f,**arrays); tmp=Path(f.name)
    os.replace(tmp,path)
def _bootstrap(ce,top1,rounds=1000):
    rng=np.random.default_rng(20260817); n=ce.size; cm=np.empty(rounds); am=np.empty(rounds)
    for i in range(rounds):
        j=rng.integers(0,n,size=n); cm[i]=ce[j].mean(); am[i]=top1[j].mean()
    return {"rounds":rounds,"mean_ce_delta_p95":float(np.percentile(cm,95)),"top1_agreement_p05":float(np.percentile(am,5))}
def _teacher_hash(rt,cp,pids,targets,count):
    _reset_exact_state(rt); _prefill(rt,pids); h=hashlib.sha256(); count=min(count,len(targets))
    for i in range(count):
        t=int(targets[i]); l=_lse(cp,rt.logits); a=int(cp.argmax(rt.logits).item()); lp=np.float32(float(rt.logits[t].item())-l)
        h.update(np.asarray([a],dtype="<i4").tobytes()); h.update(np.asarray([lp],dtype="<f4").tobytes())
        if i+1<count: _advance(rt,t)
    return h.hexdigest()
def _domain_summary(dom,top1,in5,ce,kl,rank):
    out={}
    for name in sorted(set(dom.tolist())):
        m=dom==name; out[name]={"tokens":int(m.sum()),"top1_agreement":float(top1[m].mean()),
            "target_in_top5":float(in5[m].mean()),"mean_ce_delta":float(ce[m].mean()),
            "p95_ce_delta":float(np.percentile(ce[m],95)),"mean_coarse_kl":float(kl[m].mean()),
            "p95_coarse_kl":float(np.percentile(kl[m],95)),"mean_target_rank":float(rank[m].mean()),
            "max_target_rank":int(rank[m].max())}
    return out
def _report(profile,mode,rollouts,summary,gates):
    lines=[f"# S100 phase-3 fidelity review — {profile} / {mode}","",
           "This is V18-fidelity evidence, not external task quality.","","## Summary","","```json",
           json.dumps(summary,indent=2,ensure_ascii=False),"```","","## Gates","","```json",json.dumps(gates,indent=2),"```","","## Greedy rollouts",""]
    for r in rollouts:
        lines += [f"### {r['id']} — {r['domain']}","",f"First divergence: `{r['first_divergence']}`; position-wise agreement: `{r['position_agreement']:.4f}`","",
                  "**Prompt**","",r["prompt"],"","**Exact V18**","",r["baseline_text"],"","**Candidate**","",r["candidate_text"],""]
    return "\n".join(lines)+"\n"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("smoke","full"),default="smoke"); ap.add_argument("--profile",choices=FIDELITY_PROFILES,required=True); a=ap.parse_args()
    out,metrics_path,report_path=_paths(a.mode,a.profile); trace,trace_meta_path=_trace_paths(a.mode)
    payload={"kind":"s100_phase3_fidelity","status":"started","profile":a.profile,"mode":a.mode,"started_utc":utc_now(),
             "preregistration":str(PREREG.relative_to(REPO)),"thresholds":TH,
             "claim_boundary":"fidelity to frozen exact V18 trajectories; not external corpus perplexity or task accuracy"}
    try:
        if not trace.exists() or not trace_meta_path.exists(): raise FileNotFoundError(f"missing frozen trace {a.mode}")
        tm=json.loads(trace_meta_path.read_text(encoding="utf-8"))
        if tm.get("status")!="trace_ready" or tm.get("trace_sha256")!=sha256_file(trace) or tm.get("prompt_manifest_sha256")!=sha256_file(PROMPTS):
            raise RuntimeError("trace metadata/hash/prompt mismatch")
        payload["gpu_idle_preflight"]=_require_gpu_idle_wddm()
        import cupy as cp
        from transformers import AutoTokenizer
        prompts=_prompts(a.mode)
        with np.load(trace,allow_pickle=False) as z:
            targets=z["target_ids"].astype(np.int32,copy=True); base_tlp=z["target_logprob"].astype(np.float32,copy=True)
            base_ids=z["top_ids"].astype(np.int32,copy=True); base_lp=z["top_logprob"].astype(np.float32,copy=True); base_rest=z["rest_prob"].astype(np.float32,copy=True)
        pc,n=targets.shape
        if pc!=len(prompts): raise RuntimeError("trace prompt count mismatch")
        tok=AutoTokenizer.from_pretrained(str(require_model_dir()),local_files_only=True,trust_remote_code=True,use_fast=True)
        bundle=build_v18_runtime(72,a.profile); rt=bundle.rt
        top1=np.zeros((pc,n),bool); in5=np.zeros((pc,n),bool); overlap=np.zeros((pc,n),np.float32); rank=np.empty((pc,n),np.int32)
        cand_tlp=np.empty((pc,n),np.float32); ce=np.empty((pc,n),np.float32); kl=np.empty((pc,n),np.float32); dom=np.empty((pc,n),dtype="<U32")
        pids_all=[]; finite=True
        for pi,p in enumerate(prompts):
            pids=tok.encode(p["prompt"],add_special_tokens=False); pids_all.append(pids)
            ah=hashlib.sha256(np.asarray(pids,dtype="<i4").tobytes()).hexdigest()
            if ah!=tm["prompt_records"][pi]["prompt_ids_sha256"]: raise RuntimeError(f"tokenizer drift {p['id']}")
            dom[pi,:]=p["domain"]; _reset_exact_state(rt); _prefill(rt,pids)
            for ti in range(n):
                t=int(targets[pi,ti]); clp,rr,ctop5,qlog=_snap(cp,rt.logits,t,base_ids[pi,ti])
                top1[pi,ti]=int(ctop5[0])==t; in5[pi,ti]=t in set(int(x) for x in ctop5)
                overlap[pi,ti]=len(set(map(int,ctop5)).intersection(set(map(int,base_ids[pi,ti,:5]))))/5
                rank[pi,ti]=rr; cand_tlp[pi,ti]=clp; ce[pi,ti]=float(base_tlp[pi,ti])-clp
                plog=base_lp[pi,ti].astype(np.float64); pp=np.exp(plog); qq=np.exp(qlog)
                pr=max(float(base_rest[pi,ti]),1e-30); qr=max(1-float(qq.sum()),1e-30)
                kl[pi,ti]=max(float(np.sum(pp*(plog-qlog))+pr*(math.log(pr)-math.log(qr))),0.0)
                if ti+1<n: _advance(rt,t)
            print(f"fidelity {pi+1:02d}/{pc}: {p['id']} ({n} targets)",flush=True)
        ft=top1.reshape(-1); fi=in5.reshape(-1); fo=overlap.reshape(-1); fr=rank.reshape(-1); fc=ce.reshape(-1); fk=kl.reshape(-1); fd=dom.reshape(-1)
        boot=_bootstrap(fc,ft); domains=_domain_summary(fd,ft,fi,fc,fk,fr)
        ha=hashlib.sha256(); hb=hashlib.sha256(); ac=min(4,pc); rc=min(64,n)
        for pi in range(ac): ha.update(bytes.fromhex(_teacher_hash(rt,cp,pids_all[pi],targets[pi],rc)))
        for pi in range(ac): hb.update(bytes.fromhex(_teacher_hash(rt,cp,pids_all[pi],targets[pi],rc)))
        deterministic=ha.hexdigest()==hb.hexdigest()
        rn=64 if a.mode=="smoke" else 128; rollouts=[]
        for pi,p in enumerate(prompts):
            ids,_=_run(rt,pids_all[pi],rn); base=targets[pi,:rn].tolist(); div=first_divergence(base,ids)
            agree=float(np.mean(np.asarray(base,np.int32)==np.asarray(ids,np.int32)))
            rollouts.append({**p,"tokens":rn,"first_divergence":div,"position_agreement":agree,"baseline_ids":base,"candidate_ids":ids,
                "baseline_text":tok.decode(base,skip_special_tokens=False),"candidate_text":tok.decode(ids,skip_special_tokens=False)})
        summary={"tokens":int(ft.size),"top1_agreement":float(ft.mean()),"target_in_top5":float(fi.mean()),"mean_top5_overlap":float(fo.mean()),
            "mean_target_rank":float(fr.mean()),"p99_target_rank":float(np.percentile(fr,99)),"max_target_rank":int(fr.max()),
            "mean_baseline_self_ce":float((-base_tlp.reshape(-1)).mean()),"mean_candidate_ce_on_v18_target":float((-cand_tlp.reshape(-1)).mean()),
            "mean_ce_delta":float(fc.mean()),"p95_ce_delta":float(np.percentile(fc,95)),"mean_coarse_kl":float(fk.mean()),"p95_coarse_kl":float(np.percentile(fk,95)),
            "bootstrap":boot,"deterministic_anchor_repeat":deterministic,"anchor_hash":ha.hexdigest(),"all_finite":finite,
            "mean_greedy_position_agreement":float(np.mean([r["position_agreement"] for r in rollouts])),
            "median_greedy_first_divergence":float(np.median([rn if r["first_divergence"] is None else r["first_divergence"] for r in rollouts]))}
        normal={
            "F1_top1_agreement":summary["top1_agreement"]>=TH["top1_agreement_min"],
            "F2_target_in_top5":summary["target_in_top5"]>=TH["target_in_top5_min"],
            "F3_mean_ce_delta":summary["mean_ce_delta"]<=TH["mean_ce_delta_max"],
            "F4_bootstrap95_mean_ce_delta":boot["mean_ce_delta_p95"]<=TH["bootstrap95_mean_ce_delta_max"],
            "F5_p95_ce_delta":summary["p95_ce_delta"]<=TH["p95_ce_delta_max"],
            "F6_mean_coarse_kl":summary["mean_coarse_kl"]<=TH["mean_coarse_kl_max"],
            "F7_p95_coarse_kl":summary["p95_coarse_kl"]<=TH["p95_coarse_kl_max"],
            "F8_every_domain_top1":all(d["top1_agreement"]>=TH["per_domain_top1_min"] for d in domains.values()),
            "F9_every_domain_mean_ce_delta":all(d["mean_ce_delta"]<=TH["per_domain_mean_ce_delta_max"] for d in domains.values()),
            "F10_deterministic_anchor_repeat":deterministic,"F11_all_finite":finite}
        normal_pass=all(normal.values()); control_status=None; control_ok=False; gates=dict(normal)
        control_path,_,_=_paths("smoke","k1_control")
        if a.profile!="k1_control" and control_path.exists():
            control_status=json.loads(control_path.read_text()).get("status"); control_ok=control_status=="control_failed_as_expected"
            gates["H1_k1_control_failed_as_expected"]=control_ok
        status=("control_failed_as_expected" if not normal_pass else "control_failed_to_fail") if a.profile=="k1_control" else ("v18_fidelity_candidate" if normal_pass and control_ok else "v18_fidelity_failed")
        arrays={"top1_agree":top1,"target_in_top5":in5,"top5_overlap":overlap,"target_rank":rank,"candidate_target_logprob":cand_tlp,"ce_delta":ce,"coarse_kl":kl}
        _write_npz(metrics_path,arrays); write_text_atomic(report_path,_report(a.profile,a.mode,rollouts,summary,gates),archive=True)
        payload.update({"status":status,"completed_utc":utc_now(),
            "environment":environment_snapshot((Path(__file__),PROMPTS,PREREG,REPO/"pro_research"/"s100_phase3_profiles.py",REPO/"pro_research"/"s100_phase3_runtime.py")),
            "trace":{"path":str(trace.relative_to(REPO)),"sha256":sha256_file(trace),"metadata":str(trace_meta_path.relative_to(REPO))},
            "runtime":public_bundle_record(bundle),"summary":summary,"per_domain":domains,"gates":gates,
            "normal_fidelity_pass_without_control":normal_pass,"control_status":control_status,
            "metrics":{"path":str(metrics_path.relative_to(REPO)),"sha256":sha256_file(metrics_path),"array_shapes":{k:list(v.shape) for k,v in arrays.items()}},
            "manual_report":str(report_path.relative_to(REPO)),"greedy_rollouts":rollouts})
        bundle.restore_combined(); bundle.restore_selective(); del rt,bundle; cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        payload.update({"status":"technical_failure","completed_utc":utc_now(),"error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}})
    write_json_atomic(out,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"profile":a.profile,"mode":a.mode,"summary":payload.get("summary"),"per_domain":payload.get("per_domain"),
                      "gates":payload.get("gates"),"metrics":payload.get("metrics"),"manual_report":payload.get("manual_report"),
                      "error":(payload.get("error") or {}).get("message"),"output":str(out)},indent=2,allow_nan=False,ensure_ascii=False))
    return 2 if payload.get("status") in {"technical_failure","control_failed_to_fail"} else 0
if __name__=="__main__": raise SystemExit(main())
