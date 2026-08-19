from __future__ import annotations

import math
from collections import defaultdict
import numpy as np

from s100_phase17_kernels import Phase17Kernels
from s100_phase19_residual_projection import FP8ProjectionBlock
from s100_phase20b_kernels import Phase20BKernels, H


def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30))


class GroupedMoEH4:
    def __init__(self, rt):
        import cupy as cp
        self.cp=cp; self.rt=rt; self.k=Phase20BKernels()
        self.hidden=int(rt.hidden); self.inter=int(rt.moe_inter)
        self.shared=int(rt.shared_inter); self.topk=int(rt.top_k)
        self.nexp=int(rt.n_experts)
        if H*self.topk > 48:
            raise RuntimeError("H*topk exceeds preregistered cache safety bound")
        self.rlog=cp.empty((H,self.nexp),cp.float32)
        self.ids=cp.empty((H,self.topk),cp.int32)
        self.w=cp.empty((H,self.topk),cp.float32)
        self.shared_act=cp.empty((H,self.shared),cp.float32)
        self.group_x=cp.empty((H,self.hidden),cp.float32)
        self.group_act=cp.empty((H,self.inter),cp.float32)
        self.group_down=cp.empty((H,self.hidden),cp.float32)
        self.contrib=cp.empty((H,self.topk,self.hidden),cp.float32)
        self.down_state=self.k.alloc_down_state(rt,self.hidden,self.inter)
        self.copy_done=cp.cuda.Event()

    def _routes(self,d,normed):
        cp=self.cp;rt=self.rt
        for t in range(H):
            rt.k.mv_f32(self.rlog[t],d["gate_w"],normed[t],self.nexp,self.hidden)
            scores=1.0/(1.0+cp.exp(-self.rlog[t]))
            choice=scores+d["gate_b"]
            ids=cp.argsort(-choice)[:self.topk]
            ww=scores[ids]
            ww=ww/(ww.sum()+1e-20)*rt.scaling
            self.ids[t]=ids.astype(cp.int32)
            self.w[t]=ww.astype(cp.float32)
        return cp.asnumpy(self.ids),cp.asnumpy(self.w)

    def __call__(self,layer,normed,out,collect_stats=False):
        cp=self.cp;rt=self.rt;d=rt.layer[int(layer)]
        ids,w=self._routes(d,normed)  # one synchronization for all 4 routes
        ids=ids.astype(np.int32); w=w.astype(np.float32)
        bank,c=rt.bank[int(layer)],rt.cache[int(layer)]
        cmap,cap=c["map"],int(c["cap"])
        slots=np.empty_like(ids)
        pending=[]; hits=misses=0

        # Emulate sequential token-major LRU accesses exactly. Since cap=48 and
        # one H4 block has <=24 accesses, a newly accessed expert cannot be
        # evicted again inside this block.
        for t in range(H):
            for s in range(self.topk):
                e=int(ids[t,s])
                if e in cmap:
                    slot=int(cmap[e]);cmap.move_to_end(e);hits+=1
                else:
                    misses+=1
                    if len(cmap)<cap: slot=len(cmap)
                    else:
                        _,slot=cmap.popitem(last=False)
                    cmap[e]=slot
                    pending.append((e,slot))
                slots[t,s]=slot
        rt.cache_stats["hits"]+=hits;rt.cache_stats["misses"]+=misses

        # Stage unique routed-up misses on the copy stream.
        if pending:
            with rt.copy_stream:
                seen=set()
                for e,slot in pending:
                    if (e,slot) in seen: continue
                    seen.add((e,slot))
                    c["slot_codes"][slot].set(bank["up_code_view"][e],stream=rt.copy_stream)
                    c["slot_scales"][slot].set(bank["up_scale_view"][e],stream=rt.copy_stream)
                self.copy_done.record(rt.copy_stream)

        # Shared expert is guaranteed weight reuse across all 4 rows.
        self.k.nvfp4(d["sh_up_c"],d["sh_up_s"],rt.fused.e2m1,rt.fused.e4m3,
                     normed,self.shared_act,d["sh_up_g"],self.shared,self.hidden,H,True)
        self.k.nvfp4(d["sh_dn_c"],d["sh_dn_s"],rt.fused.e2m1,rt.fused.e4m3,
                     self.shared_act,out,d["sh_dn_g"],self.hidden,self.shared,H,False)
        if pending:
            cp.cuda.get_current_stream().wait_event(self.copy_done)

        self.contrib.fill(0)
        groups=defaultdict(list)
        for t in range(H):
            for s in range(self.topk): groups[int(ids[t,s])].append((t,s))

        down_bytes=0; rows_hist=[]
        for e,refs in groups.items():
            M=len(refs); rows_hist.append(M)
            slot=int(slots[refs[0][0],refs[0][1]])
            # Gather activation rows for this expert.
            for r,(t,s) in enumerate(refs): self.group_x[r]=normed[t]
            self.k.nvfp4(c["slot_codes"][slot],c["slot_scales"][slot],
                         rt.fused.e2m1,rt.fused.e4m3,self.group_x,self.group_act,
                         float(bank["g_up"][e]),self.inter,self.hidden,M,True)
            self.k.down_group(rt,int(bank["down_ptr"][e]),self.group_act,self.group_down,
                              float(bank["g_dn"][e]),M,self.down_state,self.hidden,self.inter)
            for r,(t,s) in enumerate(refs): self.contrib[t,s]=self.group_down[r]
            if collect_stats:
                pc=int(cp.asnumpy(self.down_state["pcount"])[0])
                nz=int(cp.asnumpy(self.down_state["nzc"])[0])
                down_bytes += pc*self.hidden + nz*(self.hidden//2)

        # Preserve per-token route-slot accumulation order exactly.
        for t in range(H):
            for s in range(self.topk):
                rt.fused.accumulate_into(out[t],self.contrib[t,s],float(w[t,s]),self.hidden)

        stats=None
        if collect_stats:
            total=H*self.topk;unique=len(groups)
            stats={
                "layer":int(layer),"route_slots":total,"unique_experts":unique,
                "repeat_rate":1.0-unique/float(total),"rows_per_expert":rows_hist,
                "cache_hits":hits,"cache_misses":misses,
                "up_bytes_loaded":misses*(int(c["slot_codes"][0].nbytes)+int(c["slot_scales"][0].nbytes)),
                "down_sparse_bytes_loaded":int(down_bytes),
                "ids":ids.tolist(),"weights":w.tolist(),
            }
        return ids,w,stats


class FullH4Verifier:
    def __init__(self,rt):
        import cupy as cp
        self.cp=cp;self.rt=rt;self.H=H
        self.hidden=int(rt.hidden);self.vocab=int(rt.vocab)
        self.phase17=Phase17Kernels();self.fp8=FP8ProjectionBlock(cp)
        self.moeb=GroupedMoEH4(rt);self.bk=self.moeb.k
        self.h=cp.empty((H,self.hidden),cp.float32)
        self.normed=cp.empty_like(self.h);self.acc=cp.empty_like(self.h)
        self.final_normed=cp.empty_like(self.h)
        self.logits=cp.empty((H,self.vocab),cp.float32)

        # Mamba buffers reused by all M layers; dimensions are invariant.
        self.Hh=int(rt.m_heads);self.P=int(rt.m_hdim);self.N=int(rt.n_state)
        self.hpg=int(rt.hpg);self.G=self.Hh//self.hpg
        self.di=int(rt.d_inner);self.conv_dim=int(rt.conv_dim)
        self.proj_size=int(rt.proj.size);self.gs=self.di//int(rt.n_groups)
        self.m_proj=cp.empty((H,self.proj_size),cp.float32)
        self.m_convo=cp.empty((H,self.conv_dim),cp.float32)
        self.m_conv_final=cp.empty(self.conv_dim*int(rt.conv_k),cp.float32)
        self.m_dt=cp.empty((H,self.Hh),cp.float32)
        self.m_dx=cp.empty((H,self.di),cp.float32)
        self.m_decay=cp.empty((H,self.Hh),cp.float32)
        self.m_states=cp.empty((H,self.Hh*self.P*self.N),cp.float32)
        self.m_y=cp.empty((H,self.di),cp.float32)
        self.m_gn=cp.empty((H,self.di),cp.float32)

    def _embed(self,tokens):
        cp=self.cp;rt=self.rt
        if not rt.embed_on_host: raise RuntimeError("20B expects host embeddings")
        for t,token in enumerate(tokens):
            row=cp.asarray(rt.embed_host[int(token)*self.hidden:(int(token)+1)*self.hidden])
            self.h[t]=(row.astype(cp.uint32)<<cp.uint32(16)).view(cp.float32)

    def _norm_rows(self,w,src,dst):
        rt=self.rt
        for t in range(H): rt.k.norm(dst[t],src[t],w,self.hidden,rt.eps)

    def _mamba(self,i,normed,out):
        cp=self.cp;rt=self.rt;d=rt.layer[int(i)]
        if d["in_k"]!="fp8_tensor" or d["out_k"]!="fp8_tensor":
            raise RuntimeError(f"Mamba {i} not FP8/FP8: {d['in_k']}/{d['out_k']}")
        self.fp8.apply_residual2(d["in_w8"],normed,self.m_proj,d["in_s"])
        xbc_off=self.di;dtr_off=self.di+self.conv_dim
        Boff=self.di;Coff=self.di+int(rt.n_groups)*self.N
        self.phase17.block_conv(H,rt.conv[int(i)],self.m_proj,d["conv_w"],d["conv_b"],
                                self.m_convo,self.m_conv_final,self.conv_dim,int(rt.conv_k),
                                x_stride=self.proj_size,x_offset=xbc_off)
        self.phase17.block_dt(H,self.m_proj,d["dt_bias"],self.m_dt,self.Hh,
                              dtr_stride=self.proj_size,dtr_offset=dtr_off)
        self.phase17.prepare(H,self.m_convo,self.m_dt,d["A_log"],self.m_dx,self.m_decay,
                             self.Hh,self.P,x_stride=self.conv_dim,x_offset=0)
        self.phase17.scan("serial",H,rt.ssm[int(i)],self.m_dx,self.m_convo,self.m_decay,
                          self.m_states,self.Hh,self.P,self.N,self.hpg,
                          b_stride=self.conv_dim,b_offset=Boff)
        self.phase17.y(H,self.m_states,self.m_convo,self.m_convo,d["D"],self.m_y,
                       self.Hh,self.P,self.N,self.hpg,c_stride=self.conv_dim,c_offset=Coff,
                       x_stride=self.conv_dim,x_offset=0)
        self.phase17.gated(H,self.m_y,self.m_proj,d["m_norm"],self.m_gn,self.di,self.gs,
                           float(rt.eps),z_stride=self.proj_size,z_offset=0)
        self.fp8.apply_residual2(d["out_w8"],self.m_gn,out,d["out_s"])
        cp.copyto(rt.conv[int(i)],self.m_conv_final)
        cp.copyto(rt.ssm[int(i)],self.m_states[H-1])

    def _attention(self,i,normed,out,base_pos):
        rt=self.rt;k=rt.k;d=rt.layer[int(i)]
        if rt.fp8_kv: raise RuntimeError("Phase20B must use fp8_kv=False")
        scale=1.0/math.sqrt(float(rt.head_dim))
        for t in range(H):
            k.mv_bf16(rt.qv,d["q_proj"],normed[t],rt.n_heads*rt.head_dim,self.hidden)
            k.mv_bf16(rt.kv_,d["k_proj"],normed[t],rt.kv_dim,self.hidden)
            k.mv_bf16(rt.vv,d["v_proj"],normed[t],rt.kv_dim,self.hidden)
            pos=int(base_pos)+t
            k.kv_write(rt.kc[int(i)],rt.kv_,pos,rt.n_kv,rt.head_dim,rt.max_ctx)
            k.kv_write(rt.vc[int(i)],rt.vv,pos,rt.n_kv,rt.head_dim,rt.max_ctx)
            k.attention(rt.ctx,rt.qv,rt.kc[int(i)],rt.vc[int(i)],pos+1,
                        rt.n_heads,rt.head_dim,rt.groups,rt.max_ctx,scale,rt.part_acc,rt.part_ml)
            k.mv_bf16(out[t],d["o_proj"],rt.ctx,self.hidden,rt.n_heads*rt.head_dim)

    def block(self,tokens,collect_census=False):
        cp=self.cp;rt=self.rt
        if len(tokens)!=H: raise ValueError(tokens)
        base_pos=int(rt.pos)
        self._embed(tokens)
        census=[]
        for i,ch in enumerate(rt.pattern):
            d=rt.layer[i]
            self._norm_rows(d["norm"],self.h,self.normed)
            if ch=="M": self._mamba(i,self.normed,self.acc)
            elif ch=="*": self._attention(i,self.normed,self.acc,base_pos)
            else:
                _,_,st=self.moeb(i,self.normed,self.acc,collect_stats=collect_census)
                if st is not None:census.append(st)
            for t in range(H): rt.k.add_(self.h[t],self.acc[t],self.hidden)

        self._norm_rows(rt.norm_f,self.h,self.final_normed)
        if rt.lm_head_kind!="nvfp4": raise RuntimeError("20B expects NVFP4 lm_head")
        self.bk.nvfp4(rt.lm_head_codes,rt.lm_head_scales,rt.fused.e2m1,rt.fused.e4m3,
                      self.final_normed,self.logits,rt.lm_head_g,self.vocab,self.hidden,H,False)
        rt.pos=base_pos+H
        ids=cp.asnumpy(cp.argmax(self.logits,axis=1)).astype(np.int32)
        return ids,census
