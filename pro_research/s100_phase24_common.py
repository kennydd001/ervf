from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import time

import numpy as np

from common import REPO
from moe_dev_batched import UP_CODE,UP_SCALE,DOWN_PANEL_BYTES
from scale_resident_kernels import ScaleResidentKernels,PLANE_BYTES
from s100_phase20b_kernels import Phase20BKernels
from s100_phase21_common import (
    identity_gate,load_trace,prefill_to,expected_for_block,release,
)
from s100_phase22_common import (
    GraphH4Verifier,make_v6,selected_head_mode,capture_state,compare_states,
)
from s100_phase23_common import GPUGroupedMoEH4
from s100_phase23_group_kernels import ROUTES,GROUPS,TOPK,MAXM
from s100_phase24_dense_kernels import DenseM4Kernels
from s100_phase24_sres_grouped import GroupedScaleResidentKernels

RESULTS=REPO/"pro_research"/"results"/"s100_phase24"
PROFILE=RESULTS/"S100_PHASE24_PROFILE.json"
COMPONENTS=RESULTS/"S100_PHASE24_COMPONENTS.json"

@dataclass(frozen=True)
class SynthesisConfig:
    attention_m4: bool=False
    router_m4: bool=False
    shared_m4: bool=False
    sres_layers: tuple[int,...]=()

    def as_dict(self):
        return {
          "attention_m4":self.attention_m4,
          "router_m4":self.router_m4,
          "shared_m4":self.shared_m4,
          "sres_layers":list(self.sres_layers),
        }

def load_json(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return {}

def component_config():
    d=load_json(COMPONENTS)
    return SynthesisConfig(
      attention_m4=bool(d.get("ATTENTION_BF16_M4_OPEN")),
      router_m4=bool(d.get("ROUTER_F32_M4_OPEN")),
      shared_m4=bool(d.get("SHARED_NVFP4_M4_OPEN")),
      sres_layers=(),
    )

def ranked_sres_layers():
    d=load_json(PROFILE)
    return [int(x["layer"]) for x in d.get("sres_layer_ranking",[])]

def config_for_k(k:int):
    base=component_config()
    layers=tuple(ranked_sres_layers()[:int(k)])
    return SynthesisConfig(
      attention_m4=base.attention_m4,
      router_m4=base.router_m4,
      shared_m4=base.shared_m4,
      sres_layers=layers,
    )

def selected_record():
    d=load_json(RESULTS/"S100_PHASE24_SELECTION.json")
    rec=d.get("selected")
    if not rec:
        raise RuntimeError("Phase24 selection missing")
    return rec

def selected_config():
    rec=selected_record()
    if rec.get("label")=="baseline":
        return None
    c=rec.get("config") or {}
    return SynthesisConfig(
      attention_m4=bool(c.get("attention_m4")),
      router_m4=bool(c.get("router_m4")),
      shared_m4=bool(c.get("shared_m4")),
      sres_layers=tuple(int(x) for x in c.get("sres_layers",[])),
    )

def event_ms(cp,fn):
    cp.cuda.get_current_stream().synchronize()
    a=cp.cuda.Event();b=cp.cuda.Event()
    a.record();result=fn();b.record();b.synchronize()
    return result,float(cp.cuda.get_elapsed_time(a,b))

class GPUGroupedMoEH4Synth(GPUGroupedMoEH4):
    def __init__(self,rt,config:SynthesisConfig,diagnostic=False):
        super().__init__(rt)
        import cupy as cp
        self.config=config
        self.diagnostic=bool(diagnostic)
        self.dense=DenseM4Kernels()
        self.bk=Phase20BKernels()
        self.sres=ScaleResidentKernels()
        self.sgres=GroupedScaleResidentKernels()
        self.profile_rows=[]
        self.actual_plane_bytes=0
        self.plane_population={}

        # Actual allocation after all ordinary H4 verifier buffers exist.
        # Never reject from mem_info.
        for layer in config.sres_layers:
            cap=int(rt.cache[int(layer)]["cap"])
            plane=self.sres.alloc_planes(int(layer),cap)
            self.actual_plane_bytes+=int(plane.nbytes)
            slots=cp.arange(cap,dtype=cp.int32)
            need=cp.empty(cap,cp.int32)
            self.plane_population[int(layer)]=(slots,need)

    def _stage(self,name,fn,times):
        if not self.diagnostic:
            return fn()
        result,ms=event_ms(self.cp,fn)
        times[name]=times.get(name,0.0)+ms
        return result

    def populate_planes_from_cache(self):
        """Make resident planes coherent with the device cache after prefill.

        Canonical prefill uses the mature V6 single-token path and may replace
        cache slots without touching Phase24 planes. This one-time operation is
        outside scored timing.
        """
        cp=self.cp;rt=self.rt
        for layer in self.config.sres_layers:
            layer=int(layer);dev=self._dev(layer)
            cap=int(rt.cache[layer]["cap"])
            slots,need=self.plane_population[layer]
            # Filled slots have expert_of >= 0.
            need[...] = (dev["expert_of"] >= 0).astype(cp.int32)
            bank=rt.bank[layer]
            self.sres.fetch_plane_k(
              (cap,64),(256,),
              (np.uint64(bank["down_base_ptr"]),self.sres.planes[layer],
               dev["expert_of"],slots,need,np.uint64(DOWN_PANEL_BYTES),
               np.uint64(PLANE_BYTES),np.int32(self.hidden),
               np.int32(self.npanel))
            )
        cp.cuda.get_current_stream().synchronize()

    def __call__(self,layer,normed,out,collect_stats=False):
        cp=self.cp;rt=self.rt;i=int(layer);d=rt.layer[i]
        bank,c=rt.bank[i],rt.cache[i];dev=self._dev(i)
        times={}

        def route_stage():
            if self.config.router_m4:
                self.dense.f32(
                  d["gate_w"],normed,self.rlog,self.nexp,self.hidden
                )
            else:
                for t in range(4):
                    rt.k.mv_f32(
                      self.rlog[t],d["gate_w"],normed[t],
                      self.nexp,self.hidden
                    )
            for t in range(4):
                a=t*TOPK;b=a+TOPK
                rt.fused.route_topk(
                  self.rlog[t],d["gate_b"],self.ids[a:b],self.w[a:b],
                  self.nexp,TOPK,rt.scaling,bad_pick=rt._bad_pick
                )
        self._stage("router",route_stage,times)

        def cache_group_stage():
            self.k.cache_assign(
              dev,self.ids,self.slots,self.need,int(c["cap"])
            )
            self.k.group(
              self.ids,self.route_group,self.group_ids,self.group_count,
              self.group_refs,self.ngroups
            )
        self._stage("cache_group",cache_group_stage,times)

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
              (ROUTES,64),(256,),
              (np.uint64(bank["up_codes"].ctypes.data),
               np.uint64(bank["up_scales"].ctypes.data),
               c["codes"],c["scales"],self.ids,self.slots,self.need,
               np.uint64(UP_CODE),np.uint64(UP_SCALE))
            )
            if i in self.sres.planes:
                self.sres.fetch_plane_k(
                  (ROUTES,64),(256,),
                  (np.uint64(bank["down_base_ptr"]),self.sres.planes[i],
                   self.ids,self.slots,self.need,
                   np.uint64(DOWN_PANEL_BYTES),np.uint64(PLANE_BYTES),
                   np.int32(self.hidden),np.int32(self.npanel))
                )
            rt.evt[1].record(rt.copy_stream)

        def shared_stage():
            out.fill(0)
            if self.config.shared_m4:
                self.bk.nvfp4(
                  d["sh_up_c"],d["sh_up_s"],
                  rt.fused.e2m1,rt.fused.e4m3,
                  normed,self.shared_act,d["sh_up_g"],
                  self.shared,self.hidden,4,True
                )
                self.bk.nvfp4(
                  d["sh_dn_c"],d["sh_dn_s"],
                  rt.fused.e2m1,rt.fused.e4m3,
                  self.shared_act,out,d["sh_dn_g"],
                  self.hidden,self.shared,4,False
                )
            else:
                for t in range(4):
                    rt.fused.gemv_into(
                      self.shared_act[t],d["sh_up_c"],d["sh_up_s"],
                      normed[t],d["sh_up_g"],self.shared,self.hidden,
                      apply_relu2=True
                    )
                    rt.fused.gemv_into(
                      out[t],d["sh_dn_c"],d["sh_dn_s"],
                      self.shared_act[t],d["sh_dn_g"],
                      self.hidden,self.shared
                    )
        self._stage("shared",shared_stage,times)

        cp.cuda.get_current_stream().wait_event(rt.evt[1])

        def up_stage():
            for m in (1,2,3,4):
                self.k.up(
                  m,c["codes"],c["scales"],self.slots,self.ids,
                  dev["globals"],rt.fused.e2m1,rt.fused.e4m3,
                  normed,self.route_act,self.inter,self.hidden,
                  UP_CODE,UP_SCALE
                )
        self._stage("routed_up",up_stage,times)

        self._stage(
          "mask_union",
          lambda:self.k.scan(
            self.route_act,self.group_count,self.group_refs,
            self.route_masks,self.route_plist,self.route_pcount,
            self.union_masks,self.union_plist,self.union_pcount,
            self.union_nz,self.union_nzc,self.inter
          ),times
        )

        def gather_stage():
            if i in self.sres.planes:
                self.sgres.gather(
                  int(bank["down_base_ptr"]),self.group_ids,self.group_count,
                  self.union_nz,self.union_nzc,self.mirrors,
                  self.hidden,self.inter
                )
            else:
                self.k.gather(
                  int(bank["down_base_ptr"]),self.group_ids,self.group_count,
                  self.union_plist,self.union_pcount,
                  self.union_nz,self.union_nzc,self.mirrors,
                  DOWN_PANEL_BYTES,self.hidden,self.inter
                )
        self._stage("down_gather",gather_stage,times)

        def down_stage():
            if i in self.sres.planes:
                self.sgres.down(
                  self.mirrors,self.sres.planes[i],self.slots,self.ids,
                  self.route_group,dev["globals"],self.route_act,
                  self.route_plist,self.route_masks,self.route_pcount,
                  rt.fused.e2m1,rt.fused.e4m3,self.partials,
                  self.hidden,self.inter,self.nc
                )
            else:
                self.k.down(
                  self.mirrors,self.ids,self.route_group,dev["globals"],
                  self.route_act,self.route_plist,self.route_masks,
                  self.route_pcount,rt.fused.e2m1,rt.fused.e4m3,
                  self.partials,DOWN_PANEL_BYTES,self.hidden,
                  self.inter,self.nc
                )
            self.k.reduce(
              self.partials,self.route_down,self.hidden,self.nc
            )
            self.k.accumulate(
              out,self.route_down,self.w,self.hidden
            )
        self._stage("down_compute_reduce",down_stage,times)

        stats=None
        if collect_stats or self.diagnostic:
            cp.cuda.get_current_stream().synchronize()
            gcnt=cp.asnumpy(self.group_count)
            ng=int(cp.asnumpy(self.ngroups)[0])
            active=[int(x) for x in gcnt[:ng]]
            if not active or any(x<1 or x>4 for x in active):
                raise RuntimeError(f"invalid group counts {active}")

            upc=cp.asnumpy(self.union_pcount)[:ng]
            unz=cp.asnumpy(self.union_nzc)[:ng]
            rpc=cp.asnumpy(self.route_pcount)
            rms=cp.asnumpy(self.route_masks)
            route_nz=[
              sum((int(x)&0xFFFF).bit_count() for x in rms[r])
              for r in range(ROUTES)
            ]
            rowhalf=self.hidden//2
            parent_scale=sum(int(rpc[r])*self.hidden for r in range(ROUTES))
            parent_code=sum(int(route_nz[r])*rowhalf for r in range(ROUTES))
            group_scale=sum(int(upc[g])*self.hidden for g in range(ng))
            group_code=sum(int(unz[g])*rowhalf for g in range(ng))
            actual_down=group_code if i in self.sres.planes else group_code+group_scale
            plane_cost=int(c["cap"])*PLANE_BYTES
            ids_h=cp.asnumpy(self.ids).astype(np.int32)

            stats={
              "layer":i,"ngroups":ng,"repeat_rate":1.0-ng/24.0,
              "group_counts":active,"m_histogram":dict(Counter(active)),
              "ids":ids_h.reshape(4,TOPK).tolist(),
              "up_weight_streams_parent":24,
              "up_weight_streams_grouped":ng,
              "parent_scale_bytes":int(parent_scale),
              "parent_code_bytes":int(parent_code),
              "group_scale_bytes":int(group_scale),
              "group_code_bytes":int(group_code),
              "group_total_down_bytes":int(group_scale+group_code),
              "actual_down_gather_bytes":int(actual_down),
              "resident_plane_bytes":int(plane_cost),
              "sres_enabled":bool(i in self.sres.planes),
              "sres_score_scale_bytes_per_vram_byte":(
                 float(group_scale/plane_cost) if plane_cost else 0.0
              ),
              "cache_miss_routes":int(cp.asnumpy(self.need).sum()),
              "plane_miss_fetch_bytes":(
                 int(cp.asnumpy(self.need).sum())*PLANE_BYTES
                 if i in self.sres.planes else 0
              ),
              "stage_ms":times,
            }
            if self.diagnostic:
                self.profile_rows.append(stats)
        return None,None,stats

class GraphH4VerifierSynth(GraphH4Verifier):
    def __init__(self,rt,config:SynthesisConfig,diagnostic=False):
        import cupy as cp
        super().__init__(rt,selected_head_mode())
        self.config=config
        self.dense=DenseM4Kernels()
        self.gmoe=GPUGroupedMoEH4Synth(rt,config,diagnostic=diagnostic)
        self.v.moeb=self.gmoe
        self.q4=cp.empty((4,rt.n_heads*rt.head_dim),cp.float32)
        self.k4=cp.empty((4,rt.kv_dim),cp.float32)
        self.v4=cp.empty((4,rt.kv_dim),cp.float32)
        self.ctx4=cp.empty((4,rt.n_heads*rt.head_dim),cp.float32)

    def prepare_after_prefill(self):
        if self.config.sres_layers:
            self.gmoe.populate_planes_from_cache()
        self.set_pos_from_host()

    def _attention_block(self,i,normed,out):
        rt=self.rt;d=rt.layer[int(i)]
        self.dense.bf16(
          d["q_proj"],normed,self.q4,
          rt.n_heads*rt.head_dim,rt.hidden
        )
        self.dense.bf16(
          d["k_proj"],normed,self.k4,rt.kv_dim,rt.hidden
        )
        self.dense.bf16(
          d["v_proj"],normed,self.v4,rt.kv_dim,rt.hidden
        )
        for t in range(4):
            self.gk.kv_write(
              rt.kc[int(i)],self.k4[t],self.pos_dev,t,
              rt.n_kv,rt.head_dim,rt.max_ctx
            )
            self.gk.kv_write(
              rt.vc[int(i)],self.v4[t],self.pos_dev,t,
              rt.n_kv,rt.head_dim,rt.max_ctx
            )
            self.gk.attention(
              self.ctx4[t],self.q4[t],rt.kc[int(i)],rt.vc[int(i)],
              self.pos_dev,t,rt.n_heads,rt.head_dim,rt.groups,
              rt.max_ctx,1.0/math.sqrt(float(rt.head_dim)),
              self.part_acc,self.part_ml
            )
        self.dense.bf16(
          d["o_proj"],self.ctx4,out,
          rt.hidden,rt.n_heads*rt.head_dim
        )

    def body(self):
        rt=self.rt;v=self.v
        self.gk.embed4(self.embed_ptr,self.tok_dev,v.h,v.hidden)
        for i,ch in enumerate(rt.pattern):
            d=rt.layer[i]
            v._norm_rows(d["norm"],v.h,v.normed)
            if ch=="M":
                v._mamba(i,v.normed,v.acc)
            elif ch=="*":
                if self.config.attention_m4:
                    self._attention_block(i,v.normed,v.acc)
                else:
                    for t in range(4):
                        self._attention_row(i,v.normed[t],v.acc[t],t)
            else:
                self.gmoe(i,v.normed,v.acc,False)
            for t in range(4):
                rt.k.add_(v.h[t],v.acc[t],v.hidden)
        v._norm_rows(rt.norm_f,v.h,v.final_normed)
        self._head()
        self.gk.argmax4(
          v.logits,rt.vocab,self.am_max,self.am_idx,self.ids_dev,self.nparts
        )
        self.gk.add4pos(self.pos_dev)

    def setup_graph(self):
        info=super().setup_graph()
        info["phase24_config"]=self.config.as_dict()
        info["actual_plane_bytes"]=int(self.gmoe.actual_plane_bytes)
        try:
            txt=self.graph.debug_dot_str()
            if isinstance(txt,bytes):txt=txt.decode("utf-8","replace")
            lo=txt.lower()
            info["phase24_dot"]={
              "bf16_m4":"gemv_bf16_m4" in lo,
              "f32_m4":"gemv_f32_m4" in lo,
              "shared_m4":"batched_nvfp4_m4" in lo,
              "sres_gather":"gather_group_union_cols" in lo,
              "sres_down":"down_routes_partial_sres" in lo,
            }
        except Exception as exc:
            info["phase24_dot"]={"error":f"{type(exc).__name__}: {exc}"}
        return info

def make_synth(context:int,config:SynthesisConfig,diagnostic=False):
    identity_gate()
    rt,keep=make_v6(int(context))
    g=GraphH4VerifierSynth(rt,config,diagnostic=diagnostic)
    return rt,g,keep

def timed_synth_blocks(rt,g,trace_tokens,context,blocks,warmup):
    import time
    prefill_to(rt,trace_tokens,context)
    g.prepare_after_prefill()
    rows=[]
    for bi in range(warmup+blocks):
        pos=int(rt.pos)
        drafts,expected=expected_for_block(trace_tokens,pos)
        t0=time.perf_counter_ns()
        got=g.launch(drafts.tolist())
        ms=(time.perf_counter_ns()-t0)/1e6
        if not np.array_equal(got,expected):
            raise RuntimeError(
              f"synth mismatch pos={pos} got={got.tolist()} "
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
