from __future__ import annotations

from collections import Counter
import json
import math
import time

import numpy as np

from common import REPO
from moe_dev_batched import UP_CODE,UP_SCALE,DOWN_PANEL_BYTES
from scale_resident_kernels import ScaleResidentKernels,PLANE_BYTES
from s100_phase17_kernels import Phase17Kernels
from s100_phase19_residual_projection import FP8ProjectionBlock
from s100_phase21_common import identity_gate,make_rt,load_trace,prefill_to,release
from s100_phase22_common import selected_head_mode
from s100_phase22_graph_kernels import Phase22GraphKernels
from s100_phase24_common import selected_config,make_synth
from s100_phase25_h8_kernels import H8GroupedKernels,H,TOPK,ROUTES,GROUPS,MAXM
from s100_phase25_sres_h8 import H8ScaleResidentGather

RESULTS=REPO/"pro_research"/"results"/"s100_phase25"
OFFICIAL_PARENT_H8_MS=148.3404
ADOPTION_ABS_MS=140.91
STRONG_MS=120.0
BREAKTHROUGH_MS=100.0
S100_MS=80.0
VARIANTS={
    "split4_route":("split4","route"),
    "direct8_route":("direct8","route"),
    "direct8_groupdown":("direct8","grouped"),
}


def load_json(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}


def phase24_gate():
    identity_gate()
    p24=load_json(REPO/"pro_research"/"results"/"s100_phase24"/"S100_PHASE24_SUMMARY.json")
    th=load_json(REPO/"pro_research"/"results"/"s100_phase24"/"S100_PHASE24_THERMAL_ADJUDICATION.json")
    st=load_json(REPO/"pro_research"/"results"/"s100_phase24"/"S100_PHASE24_STATE_CHECK.json")
    if not p24.get("instrumentation_complete"):
        raise RuntimeError("Phase24 summary incomplete")
    if not th.get("BEST_OF_ALL_ADOPTED"):
        raise RuntimeError("Phase24 thermal adoption is not green")
    if not st.get("BEST_OF_ALL_STATE_GREEN"):
        raise RuntimeError("Phase24 state gate is not green")
    cfg=selected_config()
    if cfg is None:
        raise RuntimeError("Phase24 selected config unexpectedly baseline")
    if cfg.attention_m4 or cfg.router_m4 or cfg.shared_m4:
        raise RuntimeError("Phase24 closed dense M4 components must remain off")
    if len(cfg.sres_layers)!=23:
        raise RuntimeError(f"Phase24 expected 23 resident MoE layers, got {len(cfg.sres_layers)}")
    return cfg,p24,th,st


def expected_for_h8(trace_tokens,pos):
    drafts=np.asarray(trace_tokens[pos:pos+H],np.int32)
    expected=np.asarray(trace_tokens[pos+1:pos+H+1],np.int32)
    if len(drafts)!=H or len(expected)!=H:
        raise RuntimeError("canonical trace too short for H8")
    return drafts,expected


def summarize_h8(rows):
    vals=np.asarray([r["ms"] for r in rows],np.float64)
    med=float(np.median(vals))
    return {
      "count":len(rows),"median_ms":med,
      "p10_ms":float(np.percentile(vals,10)),
      "p90_ms":float(np.percentile(vals,90)),
      "mean_ms":float(vals.mean()),
      "mad_ms":float(np.median(np.abs(vals-med))),
      "ms_per_useful_token":med/8.0,
      "target_only_tok_s":8000.0/med,
      "all_token_exact":True,
    }


def event_ms(cp,fn):
    cp.cuda.get_current_stream().synchronize()
    a=cp.cuda.Event();b=cp.cuda.Event();a.record();res=fn();b.record();b.synchronize()
    return res,float(cp.cuda.get_elapsed_time(a,b))


class H8Core:
    def __init__(self,rt):
        import cupy as cp
        self.cp=cp;self.rt=rt;self.hidden=int(rt.hidden);self.vocab=int(rt.vocab)
        self.phase17=Phase17Kernels();self.fp8=FP8ProjectionBlock(cp)
        self.h=cp.empty((H,self.hidden),cp.float32)
        self.normed=cp.empty_like(self.h);self.acc=cp.empty_like(self.h)
        self.final_normed=cp.empty_like(self.h)
        self.logits=cp.empty((H,self.vocab),cp.float32)

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

    def norm_rows(self,w,src,dst):
        rt=self.rt
        for t in range(H):rt.k.norm(dst[t],src[t],w,self.hidden,rt.eps)

    def mamba(self,i,normed,out):
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


class GPUGroupedMoEH8:
    def __init__(self,rt,sres_layers,up_mode:str,down_mode:str,diagnostic=False):
        import cupy as cp
        if up_mode not in ("split4","direct8"):raise ValueError(up_mode)
        if down_mode not in ("route","grouped"):raise ValueError(down_mode)
        self.cp=cp;self.rt=rt;self.k=H8GroupedKernels();self.sg=H8ScaleResidentGather()
        self.sres=ScaleResidentKernels();self.up_mode=up_mode;self.down_mode=down_mode
        self.diagnostic=bool(diagnostic);self.profile_rows=[]
        self.hidden=int(rt.hidden);self.inter=int(rt.moe_inter);self.shared=int(rt.shared_inter)
        self.nexp=int(rt.n_experts);self.npanel=self.inter//16;self.nc=int(rt.fused.nchunks)
        self.sres_layers=tuple(int(x) for x in sres_layers)
        if set(self.sres_layers)!=set(int(x) for x in rt.moe_layers):
            raise RuntimeError("Phase25 requires Phase24 H-SCALE resident on every MoE layer")
        # H8 assigns all 48 route accesses before any routed compute.  To keep
        # every recorded slot valid until compute, a cache entry touched early
        # in the block must not be evictable by the <=47 later route accesses.
        # The frozen Phase24 nonuniform capacities satisfy this (minimum 52),
        # but fail closed if the parent/runtime ever changes.
        small_caps={int(li):int(rt.cache[int(li)]["cap"]) for li in rt.moe_layers
                    if int(rt.cache[int(li)]["cap"]) < ROUTES}
        if small_caps:
            raise RuntimeError(
                f"H8 cache safety requires cap >= {ROUTES} on every MoE layer; "
                f"found {small_caps}"
            )

        self.rlog=cp.empty((H,self.nexp),cp.float32)
        self.ids=cp.empty(ROUTES,cp.int32);self.w=cp.empty(ROUTES,cp.float32)
        self.slots=cp.empty(ROUTES,cp.int32);self.need=cp.empty(ROUTES,cp.int32)
        self.route_group=cp.empty(ROUTES,cp.int32);self.group_ids=cp.empty(GROUPS,cp.int32)
        self.group_count=cp.empty(GROUPS,cp.int32);self.group_refs=cp.empty(GROUPS*MAXM,cp.int32)
        self.ngroups=cp.zeros(1,cp.int32)
        self.shared_act=cp.empty((H,self.shared),cp.float32)
        self.route_act=cp.empty((ROUTES,self.inter),cp.float32)
        self.route_masks=cp.zeros((ROUTES,self.npanel),cp.uint32)
        self.route_plist=cp.empty((ROUTES,self.npanel),cp.int32)
        self.route_pcount=cp.zeros(ROUTES,cp.int32)
        self.union_masks=cp.zeros((GROUPS,self.npanel),cp.uint32)
        self.union_plist=cp.empty((GROUPS,self.npanel),cp.int32)
        self.union_pcount=cp.zeros(GROUPS,cp.int32)
        self.union_nz=cp.empty((GROUPS,self.inter),cp.int32)
        self.union_nzc=cp.zeros(GROUPS,cp.int32)
        self.mirrors=cp.zeros(GROUPS*DOWN_PANEL_BYTES,cp.uint8)
        self.route_down=cp.empty((ROUTES,self.hidden),cp.float32)
        self.partials=(cp.empty((ROUTES,self.nc,self.hidden),cp.float32)
                       if down_mode=="route" else None)

        self.actual_plane_bytes=0;self.plane_population={}
        for layer in self.sres_layers:
            cap=int(rt.cache[int(layer)]["cap"])
            plane=self.sres.alloc_planes(int(layer),cap)
            self.actual_plane_bytes+=int(plane.nbytes)
            slots=cp.arange(cap,dtype=cp.int32);need=cp.empty(cap,cp.int32)
            self.plane_population[int(layer)]=(slots,need)

    def _stage(self,name,fn,times):
        if not self.diagnostic:return fn()
        res,ms=event_ms(self.cp,fn);times[name]=times.get(name,0.0)+ms;return res

    def _dev(self,layer):
        rt=self.rt;bank=rt.bank[int(layer)];c=rt.cache[int(layer)]
        if not hasattr(rt,"_dev_cache"):rt._dev_cache={}
        if int(layer) not in rt._dev_cache:
            rt._dev_cache[int(layer)]=rt.fused.alloc_device_cache(
                rt.n_experts,int(c["cap"]),rt.top_k,bank["globals"])
        return rt._dev_cache[int(layer)]

    def populate_planes_from_cache(self):
        cp=self.cp;rt=self.rt
        for layer in self.sres_layers:
            layer=int(layer);dev=self._dev(layer);cap=int(rt.cache[layer]["cap"])
            slots,need=self.plane_population[layer]
            need[...] = (dev["expert_of"]>=0).astype(cp.int32)
            bank=rt.bank[layer]
            self.sres.fetch_plane_k((cap,64),(256,),
              (np.uint64(bank["down_base_ptr"]),self.sres.planes[layer],dev["expert_of"],
               slots,need,np.uint64(DOWN_PANEL_BYTES),np.uint64(PLANE_BYTES),
               np.int32(self.hidden),np.int32(self.npanel)))
        cp.cuda.get_current_stream().synchronize()

    def __call__(self,layer,normed,out,collect_stats=False):
        cp=self.cp;rt=self.rt;i=int(layer);d=rt.layer[i];bank,c=rt.bank[i],rt.cache[i]
        dev=self._dev(i);times={}

        def route_stage():
            for t in range(H):
                a=t*TOPK;b=a+TOPK
                rt.k.mv_f32(self.rlog[t],d["gate_w"],normed[t],self.nexp,self.hidden)
                rt.fused.route_topk(self.rlog[t],d["gate_b"],self.ids[a:b],self.w[a:b],
                    self.nexp,TOPK,rt.scaling,bad_pick=rt._bad_pick)
        self._stage("router",route_stage,times)

        def cache_group():
            self.k.cache_assign(dev,self.ids,self.slots,self.need,int(c["cap"]))
            self.k.group(self.ids,self.route_group,self.group_ids,self.group_count,
                         self.group_refs,self.ngroups)
        self._stage("cache_group",cache_group,times)

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k((ROUTES,64),(256,),
              (np.uint64(bank["up_codes"].ctypes.data),np.uint64(bank["up_scales"].ctypes.data),
               c["codes"],c["scales"],self.ids,self.slots,self.need,
               np.uint64(UP_CODE),np.uint64(UP_SCALE)))
            self.sres.fetch_plane_k((ROUTES,64),(256,),
              (np.uint64(bank["down_base_ptr"]),self.sres.planes[i],self.ids,self.slots,self.need,
               np.uint64(DOWN_PANEL_BYTES),np.uint64(PLANE_BYTES),np.int32(self.hidden),
               np.int32(self.npanel)))
            rt.evt[1].record(rt.copy_stream)

        def shared_stage():
            out.fill(0)
            for t in range(H):
                rt.fused.gemv_into(self.shared_act[t],d["sh_up_c"],d["sh_up_s"],normed[t],
                    d["sh_up_g"],self.shared,self.hidden,apply_relu2=True)
                rt.fused.gemv_into(out[t],d["sh_dn_c"],d["sh_dn_s"],self.shared_act[t],
                    d["sh_dn_g"],self.hidden,self.shared)
        self._stage("shared",shared_stage,times)
        cp.cuda.get_current_stream().wait_event(rt.evt[1])

        def up_stage():
            if self.up_mode=="split4":
                self.k.up_split4(c["codes"],c["scales"],self.slots,self.ids,dev["globals"],
                    rt.fused.e2m1,rt.fused.e4m3,normed,self.route_act,self.inter,self.hidden,
                    UP_CODE,UP_SCALE,self.group_count,self.group_refs)
            else:
                for m in range(1,9):
                    self.k.up_direct(m,c["codes"],c["scales"],self.slots,self.ids,dev["globals"],
                        rt.fused.e2m1,rt.fused.e4m3,normed,self.route_act,self.inter,self.hidden,
                        UP_CODE,UP_SCALE,self.group_count,self.group_refs)
        self._stage("routed_up",up_stage,times)

        self._stage("mask_union",lambda:self.k.scan(
            self.route_act,self.group_count,self.group_refs,self.route_masks,self.route_plist,
            self.route_pcount,self.union_masks,self.union_plist,self.union_pcount,
            self.union_nz,self.union_nzc,self.inter),times)

        self._stage("down_gather",lambda:self.sg.gather(
            int(bank["down_base_ptr"]),self.group_ids,self.group_count,self.union_nz,
            self.union_nzc,self.mirrors,self.hidden,self.inter),times)

        def down_stage():
            if self.down_mode=="route":
                self.k.down_route(self.mirrors,self.sres.planes[i],self.slots,self.ids,
                    self.route_group,dev["globals"],self.route_act,self.route_plist,
                    self.route_masks,self.route_pcount,rt.fused.e2m1,rt.fused.e4m3,
                    self.partials,DOWN_PANEL_BYTES,PLANE_BYTES,self.hidden,self.inter,self.nc)
                self.k.reduce(self.partials,self.route_down,self.hidden,self.nc)
            else:
                self.k.down_grouped(self.mirrors,self.sres.planes[i],self.slots,self.ids,
                    self.group_count,self.group_refs,dev["globals"],self.route_act,
                    self.route_masks,self.union_nz,self.union_nzc,rt.fused.e2m1,rt.fused.e4m3,
                    self.route_down,DOWN_PANEL_BYTES,PLANE_BYTES,self.hidden,self.inter)
            self.k.accumulate(out,self.route_down,self.w,self.hidden)
        self._stage("down_compute_reduce",down_stage,times)

        stats=None
        if collect_stats or self.diagnostic:
            cp.cuda.get_current_stream().synchronize()
            ng=int(cp.asnumpy(self.ngroups)[0]);gcnt=cp.asnumpy(self.group_count)[:ng]
            active=[int(x) for x in gcnt]
            if not active or any(x<1 or x>MAXM for x in active):
                raise RuntimeError(f"invalid H8 group counts {active}")
            ideal_streams=ng
            split4_streams=sum((x+3)//4 for x in active)
            stats={
              "layer":i,"ngroups":ng,"repeat_rate":1.0-ng/float(ROUTES),
              "m_histogram":{str(k):int(v) for k,v in sorted(Counter(active).items())},
              "max_m":max(active),"ideal_weight_streams":ideal_streams,
              "split4_weight_streams":split4_streams,
              "selected_weight_streams":split4_streams if self.up_mode=="split4" else ideal_streams,
              "cache_miss_routes":int(cp.asnumpy(self.need).sum()),"stage_ms":times,
            }
            if self.diagnostic:self.profile_rows.append(stats)
        return None,None,stats


class GraphH8Verifier:
    def __init__(self,rt,variant:str,diagnostic=False):
        import cupy as cp
        if variant not in VARIANTS:raise ValueError(variant)
        cfg,_,_,_=phase24_gate()
        self.cp=cp;self.rt=rt;self.variant=variant;self.config=cfg;self.head_mode=selected_head_mode()
        self.up_mode,self.down_mode=VARIANTS[variant]
        self.core=H8Core(rt);self.gk=Phase22GraphKernels(cp,int(rt.max_ctx))
        self.pos_dev=cp.zeros(1,cp.int32);self.tok_dev=cp.zeros(H,cp.int32);self.ids_dev=cp.zeros(H,cp.int32)
        self.nparts=256;self.am_max=cp.zeros(H*self.nparts,cp.float32);self.am_idx=cp.zeros(H*self.nparts,cp.int32)
        ns=int(self.gk.max_splits)
        self.part_acc=cp.zeros(int(rt.n_heads)*ns*4*int(rt.head_dim),cp.float32)
        self.part_ml=cp.zeros(int(rt.n_heads)*ns*4*2,cp.float32)
        nbytes=int(rt.embed_host.nbytes)
        self.embed_pm=cp.cuda.alloc_pinned_memory(nbytes)
        np.frombuffer(self.embed_pm,dtype=np.uint8,count=nbytes)[:] = rt.embed_host.view(np.uint8)
        self.embed_ptr=int(self.embed_pm.ptr)
        self.stage_pm=cp.cuda.alloc_pinned_memory(H*4);self.stage_np=np.frombuffer(self.stage_pm,dtype=np.int32,count=H)
        self.out_pm=cp.cuda.alloc_pinned_memory(H*4);self.out_np=np.frombuffer(self.out_pm,dtype=np.int32,count=H)
        self.stream=cp.cuda.Stream(non_blocking=True);self.graph=None;self.capture_info={}
        # Allocate H8 ordinary buffers before resident scale planes.
        self.gmoe=GPUGroupedMoEH8(rt,cfg.sres_layers,self.up_mode,self.down_mode,diagnostic=diagnostic)

    @property
    def v(self):
        # Compatibility with Phase24 state helpers.
        return self.core

    def set_pos_from_host(self):
        self.pos_dev.fill(np.int32(self.rt.pos));self.cp.cuda.Device(0).synchronize()

    def prepare_after_prefill(self):
        self.gmoe.populate_planes_from_cache();self.set_pos_from_host()

    def _attention_row(self,i,row,out,offset):
        rt=self.rt;k=rt.k;d=rt.layer[int(i)]
        k.mv_bf16(rt.qv,d["q_proj"],row,rt.n_heads*rt.head_dim,rt.hidden)
        k.mv_bf16(rt.kv_,d["k_proj"],row,rt.kv_dim,rt.hidden)
        k.mv_bf16(rt.vv,d["v_proj"],row,rt.kv_dim,rt.hidden)
        self.gk.kv_write(rt.kc[int(i)],rt.kv_,self.pos_dev,offset,rt.n_kv,rt.head_dim,rt.max_ctx)
        self.gk.kv_write(rt.vc[int(i)],rt.vv,self.pos_dev,offset,rt.n_kv,rt.head_dim,rt.max_ctx)
        self.gk.attention(rt.ctx,rt.qv,rt.kc[int(i)],rt.vc[int(i)],self.pos_dev,offset,
            rt.n_heads,rt.head_dim,rt.groups,rt.max_ctx,1.0/math.sqrt(float(rt.head_dim)),
            self.part_acc,self.part_ml)
        k.mv_bf16(out,d["o_proj"],rt.ctx,rt.hidden,rt.n_heads*rt.head_dim)

    def _head(self):
        rt=self.rt;c=self.core
        # Phase22 selected production_x4; keep exact production GEMV arithmetic x8.
        if self.head_mode!="production_x4":
            raise RuntimeError("Phase25 preregisters the Phase22 production_x4 head")
        for t in range(H):
            rt.fused.gemv_into(c.logits[t],rt.lm_head_codes,rt.lm_head_scales,
                c.final_normed[t],rt.lm_head_g,rt.vocab,rt.hidden)

    def body(self):
        rt=self.rt;c=self.core
        self.gk.embed4(self.embed_ptr,self.tok_dev[:4],c.h[:4],c.hidden)
        self.gk.embed4(self.embed_ptr,self.tok_dev[4:],c.h[4:],c.hidden)
        for i,ch in enumerate(rt.pattern):
            d=rt.layer[i];c.norm_rows(d["norm"],c.h,c.normed)
            if ch=="M":c.mamba(i,c.normed,c.acc)
            elif ch=="*":
                for t in range(H):self._attention_row(i,c.normed[t],c.acc[t],t)
            else:self.gmoe(i,c.normed,c.acc,False)
            for t in range(H):rt.k.add_(c.h[t],c.acc[t],c.hidden)
        c.norm_rows(rt.norm_f,c.h,c.final_normed);self._head()
        self.gk.argmax4(c.logits[:4],rt.vocab,self.am_max[:4*self.nparts],
                        self.am_idx[:4*self.nparts],self.ids_dev[:4],self.nparts)
        self.gk.argmax4(c.logits[4:],rt.vocab,self.am_max[4*self.nparts:],
                        self.am_idx[4*self.nparts:],self.ids_dev[4:],self.nparts)
        self.gk.add4pos(self.pos_dev);self.gk.add4pos(self.pos_dev)

    def setup_graph(self):
        cp=self.cp;rt=self.rt;s=self.stream
        if self.graph is not None:return self.capture_info
        self.tok_dev.fill(0);self.pos_dev.fill(0)
        with s:self.body()
        s.synchronize();rt.copy_stream.synchronize();rt.reset();self.pos_dev.fill(0);cp.cuda.Device(0).synchronize()
        free0=int(cp.cuda.Device(0).mem_info[0])
        s.begin_capture()
        with s:self.body()
        self.graph=s.end_capture();s.synchronize();rt.copy_stream.synchronize()
        rt.reset();self.pos_dev.fill(0);cp.cuda.Device(0).synchronize()
        free1=int(cp.cuda.Device(0).mem_info[0])
        dot={}
        try:
            txt=self.graph.debug_dot_str();txt=txt.decode("utf-8","replace") if isinstance(txt,bytes) else txt
            lo=txt.lower();dot={
              "available":True,"length":len(txt),"group_routes48":"group_routes48" in lo,
              "cache_assign_h8":"cache_assign_h8" in lo,"split4":"grouped_up_h8_split4" in lo,
              "direct_m8":"grouped_up_h8_m8" in lo,"group_down":"down_grouped_sres_h8" in lo,
              "route_down":"down_routes_partial_sres_h8" in lo,"argmax4":"argmax4_part" in lo,
              "pos_add4":"pos_add4" in lo,
            }
        except Exception as exc:dot={"available":False,"error":f"{type(exc).__name__}: {exc}"}
        self.capture_info={
          "variant":self.variant,"up_mode":self.up_mode,"down_mode":self.down_mode,
          "graph_extra_vram_bytes":max(0,free0-free1),"max_splits":int(self.gk.max_splits),
          "actual_plane_bytes":int(self.gmoe.actual_plane_bytes),"dot_probe":dot,
        }
        return self.capture_info

    def launch(self,drafts):
        if self.graph is None:raise RuntimeError("call setup_graph first")
        if len(drafts)!=H:raise ValueError(drafts)
        rtapi=self.cp.cuda.runtime;s=self.stream
        self.stage_np[:]=np.asarray(drafts,dtype=np.int32)
        rtapi.memcpyAsync(self.tok_dev.data.ptr,self.stage_pm.ptr,H*4,rtapi.memcpyHostToDevice,s.ptr)
        self.graph.launch(s)
        rtapi.memcpyAsync(self.out_pm.ptr,self.ids_dev.data.ptr,H*4,rtapi.memcpyDeviceToHost,s.ptr)
        s.synchronize();self.rt.pos += H
        return self.out_np.copy()


def make_h8(context:int,variant:str,diagnostic=False):
    cfg,_,_,_=phase24_gate()
    # Same V6 device-row runtime parent as Phase24, without allocating an H4 verifier.
    rt,keep=make_rt(int(context),"v6_device_rows")
    g=GraphH8Verifier(rt,variant,diagnostic=diagnostic)
    return rt,g,keep


def timed_h8_blocks(rt,g,trace_tokens,context,blocks,warmup):
    prefill_to(rt,trace_tokens,context);g.prepare_after_prefill();rows=[]
    for bi in range(warmup+blocks):
        pos=int(rt.pos);draft,expected=expected_for_h8(trace_tokens,pos)
        t0=time.perf_counter_ns();got=g.launch(draft.tolist());ms=(time.perf_counter_ns()-t0)/1e6
        if not np.array_equal(got,expected):
            raise RuntimeError(f"H8 mismatch pos={pos} got={got.tolist()} expected={expected.tolist()}")
        if bi>=warmup:rows.append({"block":bi-warmup,"pos":pos,"ms":ms,"got":got.tolist(),"expected":expected.tolist()})
    return rows,summarize_h8(rows)


def timed_parent_h8_windows(rt,g,trace_tokens,context,blocks,warmup):
    prefill_to(rt,trace_tokens,context);g.prepare_after_prefill();rows=[]
    for bi in range(warmup+blocks):
        pos=int(rt.pos);draft,expected=expected_for_h8(trace_tokens,pos)
        t0=time.perf_counter_ns();a=g.launch(draft[:4].tolist());b=g.launch(draft[4:].tolist());ms=(time.perf_counter_ns()-t0)/1e6
        got=np.concatenate([a,b])
        if not np.array_equal(got,expected):
            raise RuntimeError(f"parent H4+H4 mismatch pos={pos} got={got.tolist()} expected={expected.tolist()}")
        if bi>=warmup:rows.append({"block":bi-warmup,"pos":pos,"ms":ms,"got":got.tolist(),"expected":expected.tolist()})
    return rows,summarize_h8(rows)
