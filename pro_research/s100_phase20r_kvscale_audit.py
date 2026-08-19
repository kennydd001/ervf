from __future__ import annotations

import json
import math
import types
import traceback

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_phase20r_scaled_runtime import discover_attention_scales

OUT = REPO / "pro_research" / "results" / "s100_phase20r" / "S100_PHASE20R_KVSCALE_AUDIT.json"
HARD_SNAPSHOT = "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"

PROMPTS = (
    "The history of computing and artificial intelligence",
    "Explain why the sky appears blue during the day.",
)
EXTRA = 12


def e4_decode(code: int) -> float:
    s = code >> 7
    E = (code >> 3) & 0xF
    m = code & 7
    v = (m * 2.0**-9) if E == 0 else ((8 + m) * 2.0 ** (E - 10))
    return -v if s else v


_POS_CODES = np.array([e4_decode(i) for i in range(128)], np.float32)


def e4_quant_dequant(x: np.ndarray, scale: float) -> np.ndarray:
    """Mirror the runtime E4M3 finite encoding, with a scalar dequant scale."""
    z = np.asarray(x, np.float32) / np.float32(scale)
    flat = z.ravel()
    out = np.empty_like(flat)
    for j, v in enumerate(flat):
        if not np.isfinite(v) or v == 0.0:
            out[j] = 0.0
            continue
        a = min(abs(float(v)), 448.0)
        # 128 positive codes is tiny; nearest representable code is robust and
        # avoids relying on host rounding mode details.
        idx = int(np.argmin(np.abs(_POS_CODES - a)))
        q = float(_POS_CODES[idx])
        out[j] = (-q if v < 0 else q) * float(scale)
    return out.reshape(z.shape)


def causal_ctx(qseq, kseq, vseq, n_heads, n_kv, head_dim, k_scale=None, v_scale=None):
    groups = n_heads // n_kv
    T = len(qseq)
    ctx = np.empty((T, n_heads * head_dim), np.float32)
    K = np.asarray(kseq, np.float32).reshape(T, n_kv, head_dim)
    V = np.asarray(vseq, np.float32).reshape(T, n_kv, head_dim)
    if k_scale is not None:
        K = e4_quant_dequant(K, float(k_scale))
    if v_scale is not None:
        V = e4_quant_dequant(V, float(v_scale))
    Q = np.asarray(qseq, np.float32).reshape(T, n_heads, head_dim)
    inv = 1.0 / math.sqrt(head_dim)
    for t in range(T):
        out = np.empty((n_heads, head_dim), np.float32)
        for h in range(n_heads):
            g = h // groups
            scores = (K[:t+1, g] @ Q[t, h]) * inv
            scores = scores.astype(np.float64)
            scores -= scores.max()
            p = np.exp(scores)
            p /= p.sum()
            out[h] = (p[:, None] * V[:t+1, g].astype(np.float64)).sum(axis=0)
        ctx[t] = out.reshape(-1)
    return ctx


def nrmse(a, b):
    aa=np.asarray(a,np.float64); bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-30))


def main():
    payload={"kind":"s100_phase20r_kvscale_audit","status":"started","started_utc":utc_now()}
    try:
        import cupy as cp
        from transformers import AutoTokenizer
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        model_dir=require_model_dir()
        if model_dir.name != HARD_SNAPSHOT:
            raise RuntimeError(f"wrong snapshot {model_dir.name}")
        rt=LightningRuntime(model_dir,contexts_max=512,embed_on_host=True,fp8_kv=True,verbose=False)
        rt.load_routed_bank()
        rt.deterministic_accum=True
        scales=discover_attention_scales(rt)
        if len(scales)!=6:
            raise RuntimeError(f"expected six attention scale pairs, got {len(scales)}")
        for layer,r in scales.items():
            if not (np.isfinite(r["k_scale"]) and np.isfinite(r["v_scale"]) and r["k_scale"]>0 and r["v_scale"]>0):
                raise RuntimeError(f"invalid scale layer {layer}: {r}")

        tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=True,use_fast=True)
        captured={int(i):[] for i in rt.attn_layers}
        original=rt._attention
        enabled={"value":False,"prompt":None}

        def wrapped(self,i,out):
            result=original(i,out)
            if enabled["value"]:
                captured[int(i)].append({
                    "prompt":enabled["prompt"],
                    "q":cp.asnumpy(self.qv).astype(np.float32,copy=True),
                    "k":cp.asnumpy(self.kv_).astype(np.float32,copy=True),
                    "v":cp.asnumpy(self.vv).astype(np.float32,copy=True),
                    "ctx_kernel":cp.asnumpy(self.ctx).astype(np.float32,copy=True),
                })
            return result
        rt._attention=types.MethodType(wrapped,rt)

        # Capture full prompt+continuation trajectories so the independent
        # causal attention oracle sees the complete history.
        for pi,prompt in enumerate(PROMPTS):
            ids=tok.encode(prompt,add_special_tokens=False)
            rt.reset(); enabled["value"]=True; enabled["prompt"]=pi
            nxt=None
            for token in ids:
                nxt=int(rt.step(int(token)))
            for _ in range(EXTRA):
                nxt=int(rt.step(int(nxt)))
            enabled["value"]=False

        per_layer={}
        green=True
        for layer in sorted(captured):
            recs=captured[layer]
            layer_rows=[]
            for pi in range(len(PROMPTS)):
                r=[x for x in recs if x["prompt"]==pi]
                q=np.stack([x["q"] for x in r]); k=np.stack([x["k"] for x in r]); v=np.stack([x["v"] for x in r])
                kernel=np.stack([x["ctx_kernel"] for x in r])
                exact=causal_ctx(q,k,v,rt.n_heads,rt.n_kv,rt.head_dim,None,None)
                unit=causal_ctx(q,k,v,rt.n_heads,rt.n_kv,rt.head_dim,1.0,1.0)
                sr=scales[layer]
                scaled=causal_ctx(q,k,v,rt.n_heads,rt.n_kv,rt.head_dim,sr["k_scale"],sr["v_scale"])
                layer_rows.append({
                    "prompt":pi,
                    "tokens":len(r),
                    "unit_vs_fp32_nrmse":nrmse(exact,unit),
                    "scaled_vs_fp32_nrmse":nrmse(exact,scaled),
                    "runtime_kernel_vs_unit_oracle_nrmse":nrmse(unit,kernel),
                })
            unit=float(np.mean([x["unit_vs_fp32_nrmse"] for x in layer_rows]))
            scaled=float(np.mean([x["scaled_vs_fp32_nrmse"] for x in layer_rows]))
            kernel=float(np.mean([x["runtime_kernel_vs_unit_oracle_nrmse"] for x in layer_rows]))
            passed=bool(scaled<=0.01 and scaled<unit and kernel<=0.02)
            green &= passed
            per_layer[str(layer)]={**scales[layer],"prompts":layer_rows,
                "mean_unit_nrmse":unit,"mean_scaled_nrmse":scaled,
                "mean_runtime_kernel_vs_unit_oracle_nrmse":kernel,"pass":passed}

        payload.update({"status":"measured","model_dir":str(model_dir),"scale_pair_count":len(scales),
            "per_layer":per_layer,"KVSCALE_SEMANTICS_GREEN":bool(green),"completed_utc":utc_now()})
        rt.bank={}; cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({"status":"technical_failure","error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"green":payload.get("KVSCALE_SEMANTICS_GREEN"),
        "per_layer":payload.get("per_layer"),"error":(payload.get("error") or {}).get("message"),"output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":
    raise SystemExit(main())
