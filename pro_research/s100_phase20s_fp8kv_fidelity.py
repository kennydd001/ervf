from __future__ import annotations
import gc,json,math,traceback
import numpy as np
from common import REPO, require_model_dir, utc_now, write_json_atomic

OUT=REPO/"pro_research"/"results"/"s100_phase20s"/"S100_PHASE20S_FP8KV_FIDELITY.json"
TOKENS=64
PROMPT_TEXTS=[
 "Explain why the sky is blue in three concise sentences.",
 "Write a Python function for binary search and state its invariant.",
 "A shop discounts a 120 euro item by 15 percent. Explain the calculation.",
]

def make_rt(fp8):
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    rt=LightningRuntime(require_model_dir(),contexts_max=4096,embed_on_host=True,
                        fp8_kv=fp8,verbose=False)
    rt.enable_cache(48)
    rt.load_routed_bank()
    rt.device_cache=True
    rt.deterministic_accum=True
    return rt

def release(rt):
    import cupy as cp
    try:
        rt.bank={};rt.cache={};rt._dev_cache={}
    except Exception: pass
    del rt;gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def lse(cp,x):
    m=cp.max(x)
    return float((m+cp.log(cp.exp(x-m).sum())).item())

def make_reference(prompt_ids):
    import cupy as cp
    rt=make_rt(False); allp=[]
    try:
        for ids in prompt_ids:
            rt.reset()
            for t in ids: rt.step(int(t))
            rows=[]
            for j in range(TOKENS):
                logits=rt.logits
                le=lse(cp,logits)
                top=cp.argpartition(logits,-64)[-64:]
                top=top[cp.argsort(-logits[top])]
                target=int(top[0].item())
                ids64=cp.asnumpy(top).astype(np.int32)
                lp=cp.asnumpy(logits[top]).astype(np.float64)-le
                rows.append({"target":target,"ids64":ids64.tolist(),
                             "lp64":lp.tolist(),
                             "rest":max(1.0-float(np.exp(lp).sum()),1e-30)})
                if j+1<TOKENS:rt.step(target)
            allp.append(rows)
    finally: release(rt)
    return allp

def candidate(prompt_ids,ref):
    import cupy as cp
    rt=make_rt(True); rec=[]
    try:
        for pi,ids in enumerate(prompt_ids):
            rt.reset()
            for t in ids:rt.step(int(t))
            for j,e in enumerate(ref[pi]):
                logits=rt.logits;lc=lse(cp,logits);target=int(e["target"])
                top=cp.argpartition(logits,-5)[-5:];top=top[cp.argsort(-logits[top])]
                ids64=np.asarray(e["ids64"],np.int32)
                qlog=cp.asnumpy(logits[cp.asarray(ids64)]).astype(np.float64)-lc
                plog=np.asarray(e["lp64"],np.float64)
                pp,qq=np.exp(plog),np.exp(qlog)
                pr=max(float(e["rest"]),1e-30);qr=max(1.0-float(qq.sum()),1e-30)
                kl=max(float(np.sum(pp*(plog-qlog))+pr*(math.log(pr)-math.log(qr))),0.0)
                rec.append({
                    "prompt":pi,"step":j,
                    "top1":int(top[0].item())==target,
                    "top5":bool(cp.any(top==target).item()),
                    "ce":float(plog[0]-(float(logits[target].item())-lc)),
                    "kl":kl,
                })
                if j+1<TOKENS:rt.step(target)
    finally:release(rt)
    ce=np.asarray([r["ce"] for r in rec],np.float64)
    kl=np.asarray([r["kl"] for r in rec],np.float64)
    return rec,{
        "tokens":len(rec),
        "top1":float(np.mean([r["top1"] for r in rec])),
        "top5":float(np.mean([r["top5"] for r in rec])),
        "mean_ce":float(ce.mean()),"p95_ce":float(np.percentile(ce,95)),
        "mean_kl":float(kl.mean()),"p95_kl":float(np.percentile(kl,95)),
        "finite":bool(np.isfinite(ce).all() and np.isfinite(kl).all()),
    }

def main():
    payload={"kind":"s100_phase20s_fp8kv_fidelity","status":"started",
             "tokens_per_prompt":TOKENS,"started_utc":utc_now(),
             "reference":"same target runtime with FP32 KV cache",
             "candidate":"unchanged unit-scale FP8 KV cache"}
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(require_model_dir()),local_files_only=True,
                                           trust_remote_code=True,use_fast=True)
        prompt_ids=[tok.encode(x,add_special_tokens=False) for x in PROMPT_TEXTS]
        ref=make_reference(prompt_ids)
        rec,s=candidate(prompt_ids,ref)
        gates={
            "top1_ge_099":s["top1"]>=.99,
            "top5_eq_1":s["top5"]==1.0,
            "mean_ce_le_001":s["mean_ce"]<=.01,
            "mean_kl_le_0008":s["mean_kl"]<=.008,
            "p95_kl_le_003":s["p95_kl"]<=.03,
            "finite":s["finite"],
        }
        payload.update({"status":"measured","summary":s,"gates":gates,
                        "FP8_KV_SERVING_OPEN":all(gates.values()),
                        "phase20b_kv_policy":(
                            "fp8_kv=True" if all(gates.values()) else "fp8_kv=False"
                        ),"completed_utc":utc_now()})
    except Exception as exc:
        payload.update({"status":"technical_failure",
            "error":{"type":type(exc).__name__,"message":str(exc),
                     "traceback":traceback.format_exc()},
            "completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({k:payload.get(k) for k in
        ("status","summary","gates","FP8_KV_SERVING_OPEN","phase20b_kv_policy","error")},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
