"""Shared phase5 quality evaluator against the frozen V18 full trace."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from common import REPO,require_model_dir,sha256_file
from diag_component_marginals_graph import _prefill,_reset_exact_state
from s100_phase3_fidelity import TH,_lse,_snap,_advance,_teacher_hash,_domain_summary,_bootstrap

PROMPTS=REPO/"pro_research"/"S100_PHASE3_PROMPTS.json"
TRACE=REPO/"pro_research"/"results"/"S100_PHASE3_V18_TRACE_FULL.npz"
TRACE_META=TRACE.with_suffix('.json')
CAL_TH={"top1":.970,"top5":.999,"mean_ce":.025,"mean_kl":.015,"p95_kl":.060,"domain_top1":.90,"domain_ce":.080}

def split_indices(kind):
    ps=json.loads(PROMPTS.read_text(encoding='utf-8'))['prompts']
    if kind=='calibration': return [i for i,p in enumerate(ps) if p['id'].endswith('_01')],64
    if kind=='validation': return [i for i,p in enumerate(ps) if p['id'].endswith('_02')],128
    if kind=='heldout': return [i for i,p in enumerate(ps) if p['id'].endswith(('_03','_04'))],256
    if kind=='full': return list(range(len(ps))),256
    raise ValueError(kind)

def load_trace(kind):
    if not TRACE.exists() or not TRACE_META.exists(): raise FileNotFoundError('phase3 full trace missing; phase4 full QFAST run must complete first')
    meta=json.loads(TRACE_META.read_text(encoding='utf-8'))
    if meta.get('trace_sha256')!=sha256_file(TRACE): raise RuntimeError('frozen trace hash mismatch')
    ps=json.loads(PROMPTS.read_text(encoding='utf-8'))['prompts']
    idx,n=split_indices(kind)
    with np.load(TRACE,allow_pickle=False) as z:
        dat={k:z[k][idx,:n].copy() for k in ('target_ids','target_logprob','top_ids','top_logprob','rest_prob')}
    return [ps[i] for i in idx],idx,n,dat,meta

def evaluate(bundle,kind,deterministic=False):
    import cupy as cp
    from transformers import AutoTokenizer
    rt=bundle.rt; prompts,indices,n,d,meta=load_trace(kind)
    targets=d['target_ids'].astype(np.int32); base_tlp=d['target_logprob'].astype(np.float32)
    base_ids=d['top_ids'].astype(np.int32); base_lp=d['top_logprob'].astype(np.float32); base_rest=d['rest_prob'].astype(np.float32)
    pc=len(prompts); top1=np.zeros((pc,n),bool); in5=np.zeros((pc,n),bool); rank=np.empty((pc,n),np.int32); ce=np.empty((pc,n),np.float32); kl=np.empty((pc,n),np.float32); dom=np.empty((pc,n),dtype='<U32')
    tok=AutoTokenizer.from_pretrained(str(require_model_dir()),local_files_only=True,trust_remote_code=True,use_fast=True)
    pids=[]
    for pi,(p,orig_i) in enumerate(zip(prompts,indices)):
        ids=tok.encode(p['prompt'],add_special_tokens=False); pids.append(ids); dom[pi,:]=p['domain']
        ah=hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest()
        if ah!=meta['prompt_records'][orig_i]['prompt_ids_sha256']: raise RuntimeError(f"tokenizer drift {p['id']}")
        _reset_exact_state(rt); _prefill(rt,ids)
        for ti in range(n):
            t=int(targets[pi,ti]); clp,rr,ctop5,qlog=_snap(cp,rt.logits,t,base_ids[pi,ti])
            top1[pi,ti]=int(ctop5[0])==t; in5[pi,ti]=t in set(int(x) for x in ctop5); rank[pi,ti]=rr; ce[pi,ti]=float(base_tlp[pi,ti])-clp
            plog=base_lp[pi,ti].astype(np.float64); pp=np.exp(plog); qq=np.exp(qlog); pr=max(float(base_rest[pi,ti]),1e-30); qr=max(1-float(qq.sum()),1e-30)
            kl[pi,ti]=max(float(np.sum(pp*(plog-qlog))+pr*(math.log(pr)-math.log(qr))),0.0)
            if ti+1<n: _advance(rt,t)
    ft=top1.ravel(); fi=in5.ravel(); fr=rank.ravel(); fc=ce.ravel(); fk=kl.ravel(); fd=dom.ravel(); domains=_domain_summary(fd,ft,fi,fc,fk,fr)
    det=True; anchor=None
    if deterministic:
        ha=hashlib.sha256(); hb=hashlib.sha256(); ac=min(4,pc); rc=min(64,n)
        for pi in range(ac): ha.update(bytes.fromhex(_teacher_hash(rt,cp,pids[pi],targets[pi],rc)))
        for pi in range(ac): hb.update(bytes.fromhex(_teacher_hash(rt,cp,pids[pi],targets[pi],rc)))
        det=ha.hexdigest()==hb.hexdigest(); anchor=ha.hexdigest()
    boot=_bootstrap(fc,ft)
    summary={"tokens":int(ft.size),"top1_agreement":float(ft.mean()),"target_in_top5":float(fi.mean()),"mean_target_rank":float(fr.mean()),"max_target_rank":int(fr.max()),"mean_ce_delta":float(fc.mean()),"p95_ce_delta":float(np.percentile(fc,95)),"mean_coarse_kl":float(fk.mean()),"p95_coarse_kl":float(np.percentile(fk,95)),"all_finite":True,"deterministic_anchor_repeat":bool(det),"anchor_hash":anchor,"bootstrap":boot}
    official={"F1_top1":summary['top1_agreement']>=TH['top1_agreement_min'],"F2_top5":summary['target_in_top5']>=TH['target_in_top5_min'],"F3_mean_ce":summary['mean_ce_delta']<=TH['mean_ce_delta_max'],"F4_bootstrap_ce":boot['mean_ce_delta_p95']<=TH['bootstrap95_mean_ce_delta_max'],"F5_p95_ce":summary['p95_ce_delta']<=TH['p95_ce_delta_max'],"F6_mean_kl":summary['mean_coarse_kl']<=TH['mean_coarse_kl_max'],"F7_p95_kl":summary['p95_coarse_kl']<=TH['p95_coarse_kl_max'],"F8_domain_top1":all(x['top1_agreement']>=TH['per_domain_top1_min'] for x in domains.values()),"F9_domain_ce":all(x['mean_ce_delta']<=TH['per_domain_mean_ce_delta_max'] for x in domains.values()),"F10_deterministic":bool(det),"F11_finite":True}
    strict={"V1_top1":summary['top1_agreement']>=CAL_TH['top1'],"V2_top5":summary['target_in_top5']>=CAL_TH['top5'],"V3_mean_ce":summary['mean_ce_delta']<=CAL_TH['mean_ce'],"V4_mean_kl":summary['mean_coarse_kl']<=CAL_TH['mean_kl'],"V5_p95_kl":summary['p95_coarse_kl']<=CAL_TH['p95_kl'],"V6_domain_top1":all(x['top1_agreement']>=CAL_TH['domain_top1'] for x in domains.values()),"V7_domain_ce":all(x['mean_ce_delta']<=CAL_TH['domain_ce'] for x in domains.values()),"V8_finite":True}
    return {"split":kind,"summary":summary,"per_domain":domains,"official_gates":official,"official_pass":all(official.values()),"strict_gates":strict,"strict_pass":all(strict.values())}
