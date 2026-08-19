from __future__ import annotations

import gc
import json
import math
import time

import numpy as np

from common import REPO
from s100_phase21_common import (
    identity_gate,load_trace,make_rt,verifier_for,prefill_to,
    expected_for_block,release,
)
from s100_phase22_graph_kernels import Phase22GraphKernels

RESULTS=REPO/"pro_research"/"results"/"s100_phase22"
HEADSEL=RESULTS/"S100_PHASE22_HEAD_SELECTION.json"

class SequentialHeadAdapter:
    def __init__(self,rt):
        self.rt=rt
    def nvfp4(self,codes,scales,e2,e4,x,out,gscale,rows,cols,M,apply_relu2=False):
        if apply_relu2:
            raise RuntimeError("sequential head adapter is lm_head only")
        if int(M)!=4:
            raise RuntimeError(f"expected M4 head, got {M}")
        rt=self.rt
        for t in range(4):
            rt.fused.gemv_into(
                out[t],codes,scales,x[t],float(gscale),int(rows),int(cols)
            )

def selected_head_mode():
    d=json.loads(HEADSEL.read_text(encoding="utf-8"))
    if d.get("status")!="measured" or not d.get("selected_mode"):
        raise RuntimeError("Phase22 head selection missing/incomplete")
    return str(d["selected_mode"])

def apply_head_mode(v,mode):
    if mode=="generic_m4":
        return
    if mode=="production_x4":
        v.bk=SequentialHeadAdapter(v.rt)
        return
    raise ValueError(mode)

def make_v6(context):
    identity_gate()
    rt,keep=make_rt(int(context),"v6_device_rows")
    return rt,keep

def eager_verifier(rt,head_mode):
    v=verifier_for(rt,"v6_device_rows")
    apply_head_mode(v,head_mode)
    return v

class GraphH4Verifier:
    def __init__(self,rt,head_mode):
        import cupy as cp
        self.cp=cp;self.rt=rt;self.head_mode=head_mode
        self.v=eager_verifier(rt,head_mode)
        self.gk=Phase22GraphKernels(cp,int(rt.max_ctx))

        self.pos_dev=cp.zeros(1,cp.int32)
        self.tok_dev=cp.zeros(4,cp.int32)
        self.ids_dev=cp.zeros(4,cp.int32)
        self.nparts=256
        self.am_max=cp.zeros(4*self.nparts,cp.float32)
        self.am_idx=cp.zeros(4*self.nparts,cp.int32)

        # Fixed-stride partials for the graph-safe FP32 split attention.
        ns=int(self.gk.max_splits)
        self.part_acc=cp.zeros(
            int(rt.n_heads)*ns*4*int(rt.head_dim),cp.float32
        )
        self.part_ml=cp.zeros(int(rt.n_heads)*ns*4*2,cp.float32)

        # Same mapped pinned embedding strategy as the already-proven runtime
        # setup_graph() path; no 0.65-GiB VRAM embedding copy is introduced.
        nbytes=int(rt.embed_host.nbytes)
        self.embed_pm=cp.cuda.alloc_pinned_memory(nbytes)
        np.frombuffer(self.embed_pm,dtype=np.uint8,count=nbytes)[:] = \
            rt.embed_host.view(np.uint8)
        self.embed_ptr=int(self.embed_pm.ptr)

        self.stage_pm=cp.cuda.alloc_pinned_memory(4*4)
        self.stage_np=np.frombuffer(self.stage_pm,dtype=np.int32,count=4)
        self.out_pm=cp.cuda.alloc_pinned_memory(4*4)
        self.out_np=np.frombuffer(self.out_pm,dtype=np.int32,count=4)

        self.stream=cp.cuda.Stream(non_blocking=True)
        self.graph=None
        self.capture_info={}

    def set_pos_from_host(self):
        self.pos_dev.fill(np.int32(self.rt.pos))
        self.cp.cuda.Device(0).synchronize()

    def _attention_row(self,i,row,out,offset):
        rt=self.rt;k=rt.k;d=rt.layer[int(i)]
        k.mv_bf16(rt.qv,d["q_proj"],row,rt.n_heads*rt.head_dim,rt.hidden)
        k.mv_bf16(rt.kv_,d["k_proj"],row,rt.kv_dim,rt.hidden)
        k.mv_bf16(rt.vv,d["v_proj"],row,rt.kv_dim,rt.hidden)
        self.gk.kv_write(
            rt.kc[int(i)],rt.kv_,self.pos_dev,offset,
            rt.n_kv,rt.head_dim,rt.max_ctx
        )
        self.gk.kv_write(
            rt.vc[int(i)],rt.vv,self.pos_dev,offset,
            rt.n_kv,rt.head_dim,rt.max_ctx
        )
        self.gk.attention(
            rt.ctx,rt.qv,rt.kc[int(i)],rt.vc[int(i)],
            self.pos_dev,offset,rt.n_heads,rt.head_dim,rt.groups,
            rt.max_ctx,1.0/math.sqrt(float(rt.head_dim)),
            self.part_acc,self.part_ml
        )
        k.mv_bf16(
            out,d["o_proj"],rt.ctx,rt.hidden,rt.n_heads*rt.head_dim
        )

    def _head(self):
        rt=self.rt;v=self.v
        if self.head_mode=="generic_m4":
            v.bk.nvfp4(
                rt.lm_head_codes,rt.lm_head_scales,
                rt.fused.e2m1,rt.fused.e4m3,
                v.final_normed,v.logits,rt.lm_head_g,
                rt.vocab,rt.hidden,4,False
            )
        else:
            for t in range(4):
                rt.fused.gemv_into(
                    v.logits[t],rt.lm_head_codes,rt.lm_head_scales,
                    v.final_normed[t],rt.lm_head_g,rt.vocab,rt.hidden
                )

    def body(self):
        cp=self.cp;rt=self.rt;v=self.v
        self.gk.embed4(self.embed_ptr,self.tok_dev,v.h,v.hidden)
        for i,ch in enumerate(rt.pattern):
            d=rt.layer[i]
            v._norm_rows(d["norm"],v.h,v.normed)
            if ch=="M":
                v._mamba(i,v.normed,v.acc)
            elif ch=="*":
                for t in range(4):
                    self._attention_row(i,v.normed[t],v.acc[t],t)
            else:
                for t in range(4):
                    cp.copyto(rt.normed,v.normed[t])
                    rt._moe(int(i),v.acc[t])
            for t in range(4):
                rt.k.add_(v.h[t],v.acc[t],v.hidden)

        v._norm_rows(rt.norm_f,v.h,v.final_normed)
        self._head()
        self.gk.argmax4(
            v.logits,rt.vocab,self.am_max,self.am_idx,self.ids_dev,self.nparts
        )
        self.gk.add4pos(self.pos_dev)

    def setup_graph(self):
        cp=self.cp;rt=self.rt;s=self.stream
        if self.graph is not None:return self.capture_info

        # Compile kernels and allocate all V6 lazy per-layer device-cache state
        # before capture. The body intentionally mutates model/cache state.
        self.tok_dev.fill(0);self.pos_dev.fill(0)
        with s:
            self.body()
        s.synchronize()
        rt.copy_stream.synchronize()

        rt.reset()
        self.pos_dev.fill(0)
        cp.cuda.Device(0).synchronize()

        free0=int(cp.cuda.Device(0).mem_info[0])
        s.begin_capture()
        with s:
            self.body()
        self.graph=s.end_capture()
        s.synchronize()
        rt.copy_stream.synchronize()

        rt.reset()
        self.pos_dev.fill(0)
        cp.cuda.Device(0).synchronize()
        free1=int(cp.cuda.Device(0).mem_info[0])

        dot={}
        try:
            txt=self.graph.debug_dot_str()
            if isinstance(txt,bytes):txt=txt.decode("utf-8","replace")
            lo=txt.lower()
            dot={
              "available":True,"length":len(txt),
              "contains_fp8_residual2":"batched_fp8_residual2_t4" in lo,
              "contains_device_route":"route_topk" in lo,
              "contains_moe_batched_up":"gemv_nvfp4_ervf_ind_batched" in lo,
              "contains_fp32_kv":"kv_append_f32_dp" in lo,
              "contains_fp32_attention":(
                  "attn_decode_f32_dp" in lo or "attn_decode_warp_f32_dp" in lo
              ),
              "contains_argmax4":"argmax4_part" in lo,
              "contains_pos_add4":"pos_add4" in lo,
            }
        except Exception as exc:
            dot={"available":False,"error":f"{type(exc).__name__}: {exc}"}

        self.capture_info={
          "graph_extra_vram_bytes":max(0,free0-free1),
          "max_splits":int(self.gk.max_splits),
          "dot_probe":dot,
        }
        return self.capture_info

    def launch(self,drafts):
        if self.graph is None:
            raise RuntimeError("call setup_graph first")
        if len(drafts)!=4:raise ValueError(drafts)
        rtapi=self.cp.cuda.runtime;s=self.stream
        self.stage_np[:]=np.asarray(drafts,dtype=np.int32)

        # These copies are part of the target-verifier wall clock.
        rtapi.memcpyAsync(
            self.tok_dev.data.ptr,self.stage_pm.ptr,16,
            rtapi.memcpyHostToDevice,s.ptr
        )
        self.graph.launch(s)
        rtapi.memcpyAsync(
            self.out_pm.ptr,self.ids_dev.data.ptr,16,
            rtapi.memcpyDeviceToHost,s.ptr
        )
        s.synchronize()
        self.rt.pos += 4
        return self.out_np.copy()

def capture_state(rt,v,pos):
    import cupy as cp
    out={
      "pos":int(pos),
      "ssm":{int(k):cp.asnumpy(x).copy() for k,x in rt.ssm.items()},
      "conv":{int(k):cp.asnumpy(x).copy() for k,x in rt.conv.items()},
      "kv":{},
      "logits":cp.asnumpy(v.logits).copy(),
    }
    nk,mc,hd=int(rt.n_kv),int(rt.max_ctx),int(rt.head_dim)
    for li in rt.attn_layers:
        i=int(li)
        out["kv"][i]={
          "k":cp.asnumpy(rt.kc[i].reshape(nk,mc,hd)[:,:pos,:]).copy(),
          "v":cp.asnumpy(rt.vc[i].reshape(nk,mc,hd)[:,:pos,:]).copy(),
        }
    return out

def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30))

def compare_states(a,b):
    ssm=max(nrmse(a["ssm"][k],b["ssm"][k]) for k in a["ssm"])
    conv=max(nrmse(a["conv"][k],b["conv"][k]) for k in a["conv"])
    kv=0.0
    for k in a["kv"]:
        kv=max(kv,nrmse(a["kv"][k]["k"],b["kv"][k]["k"]),
               nrmse(a["kv"][k]["v"],b["kv"][k]["v"]))
    return {
      "max_ssm_nrmse":ssm,
      "max_conv_nrmse":conv,
      "max_kv_nrmse":kv,
      "logits_nrmse":nrmse(a["logits"],b["logits"]),
    }

def timed_graph_blocks(rt,g,trace_tokens,context,blocks,warmup):
    import time
    prefill_to(rt,trace_tokens,context)
    g.set_pos_from_host()
    rows=[]
    for bi in range(warmup+blocks):
        pos=int(rt.pos)
        draft,expected=expected_for_block(trace_tokens,pos)
        t0=time.perf_counter_ns()
        got=g.launch(draft.tolist())
        ms=(time.perf_counter_ns()-t0)/1e6
        if not np.array_equal(got,expected):
            raise RuntimeError(
              f"graph token mismatch pos={pos} got={got.tolist()} "
              f"expected={expected.tolist()}"
            )
        if bi>=warmup:
            rows.append({"block":bi-warmup,"pos":pos,"ms":ms,
                         "got":got.tolist(),"expected":expected.tolist()})
    vals=np.asarray([r["ms"] for r in rows],np.float64)
    return rows,{
      "count":len(rows),"median_ms":float(np.median(vals)),
      "p10_ms":float(np.percentile(vals,10)),
      "p90_ms":float(np.percentile(vals,90)),
      "mean_ms":float(vals.mean()),
      "ms_per_useful_token":float(np.median(vals)/4.0),
      "target_only_tok_s":float(4000.0/np.median(vals)),
      "all_token_exact":True,
    }
