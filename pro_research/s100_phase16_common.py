from __future__ import annotations
import gc,json,math,os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from common import REPO

RESULTS=REPO/"pro_research"/"results"/"s100_phase16"

@dataclass
class BF16Case:
    name:str; family:str; layer:int|None; side:str
    W:object; rows:int; cols:int; weight_bytes:int

def collect_cases(rt):
    out=[]
    for li in rt.mamba_layers:
        i=int(li); d=rt.layer[i]
        if d.get("in_k")=="bf16":
            out.append(BF16Case(f"mamba_{i}_in","mamba",i,"in",d["in_w"],
                                int(rt.proj.size),int(rt.hidden),int(d["in_w"].nbytes)))
        if d.get("out_k")=="bf16":
            out.append(BF16Case(f"mamba_{i}_out","mamba",i,"out",d["out_w"],
                                int(rt.hidden),int(rt.d_inner),int(d["out_w"].nbytes)))
    for li in rt.attn_layers:
        i=int(li); d=rt.layer[i]; hq=int(rt.n_heads*rt.head_dim)
        if d.get("q_kind","bf16")=="bf16" and "q_proj" in d:
            out.append(BF16Case(f"attention_{i}_q","attention",i,"q",d["q_proj"],
                                hq,int(rt.hidden),int(d["q_proj"].nbytes)))
        for side,rows,cols in (
            ("k",int(rt.kv_dim),int(rt.hidden)),
            ("v",int(rt.kv_dim),int(rt.hidden)),
            ("o",int(rt.hidden),hq),
        ):
            key=f"{side}_proj"
            if key in d:
                out.append(BF16Case(f"attention_{i}_{side}","attention",i,side,d[key],
                                    rows,cols,int(d[key].nbytes)))
    if getattr(rt,"lm_head_kind",None)!="nvfp4" and hasattr(rt,"lm_head"):
        out.append(BF16Case("lm_head","lm_head",None,"lm_head",rt.lm_head,
                            int(rt.vocab),int(rt.hidden),int(rt.lm_head.nbytes)))
    return out

class SelectiveNativeBF16:
    def __init__(self,rt):
        import torch
        self.rt=rt; self.torch=torch; self.cp=rt.cp
        self.original=rt.k.mv_bf16
        self.cases={int(c.W.data.ptr):c for c in collect_cases(rt)}
        self.selected=set(); self.weights_t={}
        self.native_calls=0; self.original_calls=0
    def set_selected(self,names):
        self.selected=set(names)
    def _weight_t(self,W,rows,cols):
        key=(int(W.data.ptr),int(rows),int(cols))
        wt=self.weights_t.get(key)
        if wt is None:
            # Match the exact Phase-14 D2 primitive that was measured green:
            # first materialize a Torch-owned BF16 copy, then transpose+pack.
            # The prior Phase16 skipped clone(); q_proj then hit
            # CUBLAS_STATUS_INVALID_VALUE on the runtime path.
            raw=(self.torch.utils.dlpack.from_dlpack(W)
                 .view(self.torch.bfloat16).reshape(int(rows),int(cols)).clone())
            wt=raw.t().contiguous()
            self.weights_t[key]=wt
        return wt
    def __call__(self,out,W,x,rows,cols):
        case=self.cases.get(int(W.data.ptr))
        if case is None or case.name not in self.selected:
            self.original_calls+=1
            return self.original(out,W,x,rows,cols)
        torch=self.torch; cp=self.cp
        stream=torch.cuda.ExternalStream(cp.cuda.get_current_stream().ptr)
        with torch.cuda.stream(stream):
            xt=torch.utils.dlpack.from_dlpack(x)
            xb=xt.to(torch.bfloat16).reshape(1,-1)
            y=torch.mm(xb,self._weight_t(W,rows,cols)).float().reshape(-1)
            torch.utils.dlpack.from_dlpack(out).copy_(y)
        self.native_calls+=1
        return None

def make_runtime(contexts_max=2048):
    # Match the working Phase-14 D2 initialization order.
    import torch
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    rt=LightningRuntime(Path(os.environ["LS_MODEL_DIR"]),contexts_max=contexts_max,
                        embed_on_host=True,fp8_kv=True,verbose=False)
    rt.load_routed_bank()
    rt.deterministic_accum=True
    dispatch=SelectiveNativeBF16(rt)
    rt.k.mv_bf16=dispatch
    return rt,dispatch

def release_runtime(rt):
    import cupy as cp, torch
    try:
        rt.bank={}; rt.cache={}; rt._dev_cache={}
    except Exception: pass
    del rt; gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

def prompt_rows(split):
    from transformers import AutoTokenizer
    rows=json.loads((REPO/"pro_research"/"S100_PHASE3_PROMPTS.json").read_text(encoding="utf-8"))["prompts"]
    tok=AutoTokenizer.from_pretrained(os.environ["LS_MODEL_DIR"],local_files_only=True,
                                     trust_remote_code=True,use_fast=True)
    suff={"calibration":("_01",),"validation":("_02",),"heldout":("_03","_04")}[split]
    return [{"id":r["id"],"domain":r["domain"],
             "prompt_ids":[int(x) for x in tok.encode(r["prompt"],add_special_tokens=False)]}
            for r in rows if r["id"].endswith(suff)]

def compact_snapshot(rt):
    pos=int(rt.pos)
    snap={"pos":pos,
          "ssm":{int(k):v.copy() for k,v in rt.ssm.items()},
          "conv":{int(k):v.copy() for k,v in rt.conv.items()},
          "kc":{},"vc":{}}
    if pos>0:
        nk,mc,hd=int(rt.n_kv),int(rt.max_ctx),int(rt.head_dim)
        for li in rt.attn_layers:
            i=int(li)
            snap["kc"][i]=rt.kc[i].reshape(nk,mc,hd)[:,:pos,:].copy()
            snap["vc"][i]=rt.vc[i].reshape(nk,mc,hd)[:,:pos,:].copy()
    return snap

def restore_snapshot(rt,snap):
    pos=int(snap["pos"]); rt.pos=pos
    for k,v in snap["ssm"].items(): rt.ssm[int(k)][...]=v
    for k,v in snap["conv"].items(): rt.conv[int(k)][...]=v
    if pos>0:
        nk,mc,hd=int(rt.n_kv),int(rt.max_ctx),int(rt.head_dim)
        for k,v in snap["kc"].items():
            rt.kc[int(k)].reshape(nk,mc,hd)[:,:pos,:]=v
        for k,v in snap["vc"].items():
            rt.vc[int(k)].reshape(nk,mc,hd)[:,:pos,:]=v

def logsumexp_gpu(cp,x):
    m=cp.max(x)
    return float((m+cp.log(cp.exp(x-m).sum())).item())

def compare_logits(cp,candidate,exact):
    e=cp.argpartition(exact,-64)[-64:]; e=e[cp.argsort(-exact[e])]
    c=cp.argpartition(candidate,-64)[-64:]; c=c[cp.argsort(-candidate[c])]
    target=int(e[0].item()); le=logsumexp_gpu(cp,exact); lc=logsumexp_gpu(cp,candidate)
    ids=e
    plog=cp.asnumpy(exact[ids]).astype(np.float64)-le
    qlog=cp.asnumpy(candidate[ids]).astype(np.float64)-lc
    pp,qq=np.exp(plog),np.exp(qlog)
    pr=max(1.0-float(pp.sum()),1e-30); qr=max(1.0-float(qq.sum()),1e-30)
    kl=max(float(np.sum(pp*(plog-qlog))+pr*(math.log(pr)-math.log(qr))),0.0)
    return {"target":target,"candidate_top1":int(c[0].item()),
            "top1":int(c[0].item())==target,
            "top5":bool(cp.any(c[:5]==target).item()),
            "K16":bool(cp.any(c[:16]==target).item()),
            "ce":float((float(exact[target].item())-le)-(float(candidate[target].item())-lc)),
            "kl":kl}

def summarize(records):
    if not records:return None
    ce=np.asarray([r["ce"] for r in records],np.float64)
    kl=np.asarray([r["kl"] for r in records],np.float64)
    return {"tokens":len(records),
            "top1":float(np.mean([r["top1"] for r in records])),
            "top5":float(np.mean([r["top5"] for r in records])),
            "K16":float(np.mean([r["K16"] for r in records])),
            "mean_ce":float(ce.mean()),"p95_ce":float(np.percentile(ce,95)),
            "mean_kl":float(kl.mean()),"p95_kl":float(np.percentile(kl,95)),
            "finite":bool(np.isfinite(ce).all() and np.isfinite(kl).all())}

def phase14_savings():
    p=REPO/"pro_research"/"results"/"s100_phase14d2"/"S100_PHASE14D2_COMPONENT.json"
    if not p.exists(): return {}
    try:
        d=json.loads(p.read_text(encoding="utf-8")); out={}
        for B in ("1","4"):
            for row in d.get("per_B",{}).get(B,{}).get("cases",[]):
                rec=out.setdefault(row["case"],{})
                rec[f"B{B}_baseline_ms"]=float(row["baseline_independent_ervf"]["median_ms"])
                rec[f"B{B}_native_ms"]=float(row["native_bf16"]["median_ms"])
                rec[f"B{B}_saving_ms"]=rec[f"B{B}_baseline_ms"]-rec[f"B{B}_native_ms"]
                rec[f"B{B}_speedup"]=float(row["speedup"])
        return out
    except Exception:return {}
