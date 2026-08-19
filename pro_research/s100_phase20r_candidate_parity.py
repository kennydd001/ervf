from __future__ import annotations

import json
import math
import traceback

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now

REF = REPO / "pro_research" / "results" / "s100_phase20r" / "S100_PHASE20R_REFERENCE.json"
OUT = REPO / "pro_research" / "results" / "s100_phase20r" / "S100_PHASE20R_CANDIDATE_PARITY.json"


def lse(cp,x):
    m=cp.max(x)
    return float((m+cp.log(cp.exp(x-m).sum())).item())


def main():
    payload={"kind":"s100_phase20r_candidate_parity","status":"started","started_utc":utc_now()}
    try:
        ref=json.loads(REF.read_text(encoding="utf-8"))
        if ref.get("full_reference_available") is not True:
            raise RuntimeError("independent full reference did not execute")
        import cupy as cp
        from transformers import AutoTokenizer
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        model=require_model_dir()
        rt=LightningRuntime(model,contexts_max=512,embed_on_host=True,fp8_kv=True,verbose=False)
        rt.load_routed_bank();rt.deterministic_accum=True;rt.graph_mode=False
        tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True,use_fast=True)
        # Assert the production runtime actually loaded all scales.
        scale_rows=[]
        for li in rt.attn_layers:
            d=rt.layer[int(li)]
            if "k_scale" not in d or "v_scale" not in d:
                raise RuntimeError(f"attention layer {li} does not consume k_scale/v_scale")
            scale_rows.append({"layer":int(li),"k_scale":float(d["k_scale"]),"v_scale":float(d["v_scale"])})

        records=[]
        for trace in ref["traces"]:
            ids=tok.encode(trace["prompt"],add_special_tokens=False)
            if ids != trace["prompt_ids"]:
                raise RuntimeError("tokenizer mismatch against independent reference")
            rt.reset()
            for token in ids:
                rt.step(int(token))
            for rec in trace["records"]:
                target=int(rec["target"])
                logits=rt.logits
                ll=lse(cp,logits)
                top5=cp.argpartition(logits,-5)[-5:]
                top5=top5[cp.argsort(-logits[top5])]
                cand_top1=int(top5[0].item())
                cand_lp=float(logits[target].item())-ll
                ref_ids=np.asarray(rec["top64_ids"],np.int32)
                ref_logp=np.asarray(rec["top64_logprob"],np.float64)
                qlog=cp.asnumpy(logits[cp.asarray(ref_ids)]).astype(np.float64)-ll
                pp=np.exp(ref_logp);qq=np.exp(qlog)
                pr=max(float(rec["rest_prob"]),1e-30)
                qr=max(1.0-float(qq.sum()),1e-30)
                kl=max(float(np.sum(pp*(ref_logp-qlog))+pr*(math.log(pr)-math.log(qr))),0.0)
                records.append({
                    "prompt":trace["prompt"],"step":int(rec["step"]),"target":target,
                    "candidate_top1":cand_top1,"top1":cand_top1==target,
                    "top5":bool(cp.any(top5==target).item()),
                    "ce_delta":float(ref_logp[0]-cand_lp),
                    "coarse_kl":kl,
                })
                if int(rec["step"])+1 < len(trace["records"]):
                    rt.step(target)
            print(f"candidate parity complete: {trace['prompt'][:40]}...",flush=True)

        ce=np.asarray([x["ce_delta"] for x in records],np.float64)
        kl=np.asarray([x["coarse_kl"] for x in records],np.float64)
        summary={
            "tokens":len(records),
            "top1_agreement":float(np.mean([x["top1"] for x in records])),
            "target_in_top5":float(np.mean([x["top5"] for x in records])),
            "mean_ce_delta":float(ce.mean()),
            "p95_ce_delta":float(np.percentile(ce,95)),
            "mean_coarse_kl":float(kl.mean()),
            "p95_coarse_kl":float(np.percentile(kl,95)),
            "finite":bool(np.isfinite(ce).all() and np.isfinite(kl).all()),
        }

        # Sabotage one definitely-consumed Mamba target tensor.
        first=ref["traces"][0]
        ids=first["prompt_ids"]
        def first_logits():
            rt.reset()
            for t in ids: rt.step(int(t))
            cp.cuda.get_current_stream().synchronize()
            return cp.asnumpy(rt.logits).astype(np.float32,copy=True),int(cp.argmax(rt.logits).item())
        base_logits,base_token=first_logits()
        w=rt.layer[0]["in_w8"]; backup=cp.asnumpy(w).copy();w.fill(0)
        bad_logits,bad_token=first_logits();w.set(backup);cp.cuda.get_current_stream().synchronize()
        delta=float(np.max(np.abs(base_logits-bad_logits)))
        sabotage={"tensor":"backbone.layers.0.mixer.in_proj.weight","max_abs_logit_delta":delta,
                  "base_token":base_token,"sabotaged_token":bad_token,
                  "observable":bool(delta>0 or base_token!=bad_token)}

        gates={
            "top1":summary["top1_agreement"]==1.0,
            "top5":summary["target_in_top5"]==1.0,
            "mean_ce":summary["mean_ce_delta"]<=0.015,
            "mean_kl":summary["mean_coarse_kl"]<=0.010,
            "p95_kl":summary["p95_coarse_kl"]<=0.040,
            "finite":summary["finite"],
            "sabotage":sabotage["observable"],
        }
        payload.update({"status":"measured","scale_rows":scale_rows,"summary":summary,"gates":gates,
                        "PARITY_GREEN":all(gates.values()),"sabotage":sabotage,"records":records,"completed_utc":utc_now()})
        rt.bank={};cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({"status":"technical_block","error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"summary":payload.get("summary"),"gates":payload.get("gates"),
                      "PARITY_GREEN":payload.get("PARITY_GREEN"),"sabotage":payload.get("sabotage"),
                      "error":(payload.get("error") or {}).get("message"),"output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__": raise SystemExit(main())
