from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from common import REPO
from moe_dev_batched import UP_CODE,UP_SCALE,DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES
from s100_phase21_common import release
from s100_phase24_common import selected_config,make_synth,timed_synth_blocks
from s100_phase27_kernels import (
    Phase27DownPipelineKernels,GROUPS,ROUTES,TOPK,
)

RESULTS=REPO/"pro_research"/"results"/"s100_phase27"
SNAPSHOT="e8f3c7c4de75ad84fe1bcef95d38eca76214480b"


def load_json(path):
    try:return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:return {}


def phase27_gate():
    p24=load_json(REPO/"pro_research"/"results"/"s100_phase24"/"S100_PHASE24_SUMMARY.json")
    t24=load_json(REPO/"pro_research"/"results"/"s100_phase24"/"S100_PHASE24_THERMAL_ADJUDICATION.json")
    s24=load_json(REPO/"pro_research"/"results"/"s100_phase24"/"S100_PHASE24_STATE_CHECK.json")
    p26=load_json(REPO/"pro_research"/"results"/"s100_phase26"/"S100_PHASE26_SUMMARY.json")
    if not p24.get("instrumentation_complete"):
        raise RuntimeError("Phase24 summary incomplete")
    if not t24.get("BEST_OF_ALL_ADOPTED"):
        raise RuntimeError("Phase24 active parent is not thermally adopted")
    if not s24.get("BEST_OF_ALL_STATE_GREEN"):
        raise RuntimeError("Phase24 active parent state gate is not green")
    if not p26.get("instrumentation_complete"):
        raise RuntimeError("Phase26 summary incomplete")
    if p26.get("PHASE26_ACTIVE_PARENT_ADOPTED"):
        raise RuntimeError("Phase26 unexpectedly changed active parent")
    if p26.get("NEXT_ROUTE")!="BUILD_DOWN_GATHER_TRANSFER_COMPUTE_PIPELINE":
        raise RuntimeError(f"unexpected Phase26 route: {p26.get('NEXT_ROUTE')}")
    cfg=selected_config()
    if cfg is None:
        raise RuntimeError("Phase24 selected config missing")
    if cfg.attention_m4 or cfg.router_m4 or cfg.shared_m4:
        raise RuntimeError("closed Phase24 dense M4 arms must remain disabled")
    if len(cfg.sres_layers)!=23:
        raise RuntimeError(f"expected 23 resident scale layers, got {len(cfg.sres_layers)}")
    return cfg,p24,p26


@dataclass(frozen=True)
class Variant:
    gather_y:int=32
    batches:int=1
    shared_overlap:bool=False

    def __post_init__(self):
        if self.gather_y not in (4,8,16,32):
            raise ValueError(self.gather_y)
        if self.batches not in (1,2,3,4):
            raise ValueError(self.batches)

    @property
    def label(self):
        return f"g{self.gather_y}_b{self.batches}" + ("_ovl" if self.shared_overlap else "")

    def as_dict(self):
        return {
          "gather_y":int(self.gather_y),
          "batches":int(self.batches),
          "shared_overlap":bool(self.shared_overlap),
          "label":self.label,
        }


class PipelinedMoEH4:
    H=4
    TOPK=6

    def __init__(self,base,variant:Variant,diagnostic:bool=False):
        import cupy as cp
        self.cp=cp
        self.base=base
        self.rt=base.rt
        self.variant=variant
        self.diagnostic=bool(diagnostic)
        self.pk=Phase27DownPipelineKernels()
        self.ranges=self.pk.ranges(variant.batches)

        self.gather_stream=cp.cuda.Stream(non_blocking=True)
        self.mask_ready={int(i):cp.cuda.Event() for i in self.rt.moe_layers}
        self.gather_ready={
          int(i):tuple(cp.cuda.Event() for _ in self.ranges)
          for i in self.rt.moe_layers
        }

        self.shared_stream=None
        self.shared_out=None
        self.shared_fork={}
        self.shared_done={}
        self.pipe_start={}
        self.pipe_end={}
        self.gather_start={}
        self.gather_end={}
        if self.diagnostic:
            self.pipe_start={int(i):cp.cuda.Event() for i in self.rt.moe_layers}
            self.pipe_end={int(i):cp.cuda.Event() for i in self.rt.moe_layers}
            self.gather_start={int(i):cp.cuda.Event() for i in self.rt.moe_layers}
            self.gather_end={int(i):cp.cuda.Event() for i in self.rt.moe_layers}

        if variant.shared_overlap:
            self.shared_stream=cp.cuda.Stream(non_blocking=True)
            self.shared_out=cp.empty((self.H,base.hidden),cp.float32)
            self.shared_fork={
              int(i):cp.cuda.Event() for i in self.rt.moe_layers
            }
            self.shared_done={
              int(i):cp.cuda.Event() for i in self.rt.moe_layers
            }

    def __getattr__(self,name):
        return getattr(self.base,name)

    def _shared_parent(self,d,normed,out):
        b=self.base;rt=self.rt
        out.fill(0)
        for t in range(self.H):
            rt.fused.gemv_into(
              b.shared_act[t],d["sh_up_c"],d["sh_up_s"],normed[t],
              d["sh_up_g"],b.shared,b.hidden,apply_relu2=True
            )
            rt.fused.gemv_into(
              out[t],d["sh_dn_c"],d["sh_dn_s"],b.shared_act[t],
              d["sh_dn_g"],b.hidden,b.shared
            )

    def _shared_fork(self,i,d,normed,main):
        b=self.base;rt=self.rt
        self.shared_fork[i].record(main)
        with self.shared_stream:
            self.shared_stream.wait_event(self.shared_fork[i])
            for t in range(self.H):
                rt.fused.gemv_into(
                  b.shared_act[t],d["sh_up_c"],d["sh_up_s"],normed[t],
                  d["sh_up_g"],b.shared,b.hidden,apply_relu2=True
                )
                rt.fused.gemv_into(
                  self.shared_out[t],d["sh_dn_c"],d["sh_dn_s"],b.shared_act[t],
                  d["sh_dn_g"],b.hidden,b.shared
                )
            self.shared_done[i].record(self.shared_stream)

    def __call__(self,layer,normed,out,collect_stats=False):
        cp=self.cp;b=self.base;rt=self.rt;i=int(layer);d=rt.layer[i]
        bank,c=rt.bank[i],rt.cache[i];dev=b._dev(i)
        main=cp.cuda.get_current_stream()

        # Phase26 exact shared branch is optional and measured fresh in Phase27.
        if self.variant.shared_overlap:
            self._shared_fork(i,d,normed,main)

        # Frozen Phase24 router/cache/group.
        for t in range(self.H):
            a=t*self.TOPK;z=a+self.TOPK
            rt.k.mv_f32(b.rlog[t],d["gate_w"],normed[t],b.nexp,b.hidden)
            rt.fused.route_topk(
              b.rlog[t],d["gate_b"],b.ids[a:z],b.w[a:z],
              b.nexp,self.TOPK,rt.scaling,bad_pick=rt._bad_pick
            )

        b.k.cache_assign(dev,b.ids,b.slots,b.need,int(c["cap"]))
        b.k.group(
          b.ids,b.route_group,b.group_ids,b.group_count,b.group_refs,b.ngroups
        )

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
              (ROUTES,64),(256,),
              (
                np.uint64(bank["up_codes"].ctypes.data),
                np.uint64(bank["up_scales"].ctypes.data),
                c["codes"],c["scales"],b.ids,b.slots,b.need,
                np.uint64(UP_CODE),np.uint64(UP_SCALE),
              )
            )
            if i not in b.sres.planes:
                raise RuntimeError(f"Phase27 requires H-SCALE plane on layer {i}")
            b.sres.fetch_plane_k(
              (ROUTES,64),(256,),
              (
                np.uint64(bank["down_base_ptr"]),b.sres.planes[i],
                b.ids,b.slots,b.need,
                np.uint64(DOWN_PANEL_BYTES),np.uint64(PLANE_BYTES),
                np.int32(b.hidden),np.int32(b.npanel),
              )
            )
            rt.evt[1].record(rt.copy_stream)

        # Exact Phase24 shared execution for pipeline-only arms.
        if not self.variant.shared_overlap:
            self._shared_parent(d,normed,out)

        main.wait_event(rt.evt[1])

        # Frozen Phase24 grouped routed-up.
        for m in (1,2,3,4):
            b.k.up(
              m,c["codes"],c["scales"],b.slots,b.ids,dev["globals"],
              rt.fused.e2m1,rt.fused.e4m3,normed,b.route_act,
              b.inter,b.hidden,UP_CODE,UP_SCALE
            )

        b.k.scan(
          b.route_act,b.group_count,b.group_refs,
          b.route_masks,b.route_plist,b.route_pcount,
          b.union_masks,b.union_plist,b.union_pcount,
          b.union_nz,b.union_nzc,b.inter
        )

        if self.diagnostic:
            self.pipe_start[i].record(main)

        if self.variant.batches==1:
            # Geometry-only control: change only gather grid.y. Keep the
            # exact original Phase24 down kernel and reduction unchanged.
            g0,g1=self.ranges[0]
            if self.diagnostic:
                self.gather_start[i].record(main)
            self.pk.gather_range(
              int(bank["down_base_ptr"]),b.group_ids,b.group_count,
              b.union_nz,b.union_nzc,b.mirrors,b.hidden,b.inter,
              g0,g1,self.variant.gather_y
            )
            if self.diagnostic:
                self.gather_end[i].record(main)
            b.sgres.down(
              b.mirrors,b.sres.planes[i],b.slots,b.ids,
              b.route_group,dev["globals"],b.route_act,
              b.route_plist,b.route_masks,b.route_pcount,
              rt.fused.e2m1,rt.fused.e4m3,b.partials,
              b.hidden,b.inter,b.nc
            )
        else:
            # Fork the transfer side after masks/union lists exist.
            self.mask_ready[i].record(main)
            with self.gather_stream:
                self.gather_stream.wait_event(self.mask_ready[i])
                if self.diagnostic:
                    self.gather_start[i].record(self.gather_stream)
                for bi,(g0,g1) in enumerate(self.ranges):
                    self.pk.gather_range(
                      int(bank["down_base_ptr"]),b.group_ids,b.group_count,
                      b.union_nz,b.union_nzc,b.mirrors,b.hidden,b.inter,
                      g0,g1,self.variant.gather_y
                    )
                    self.gather_ready[i][bi].record(self.gather_stream)
                if self.diagnostic:
                    self.gather_end[i].record(self.gather_stream)

            # Each route/chunk is executed once by exactly one range. The
            # kernel body is copied from Phase24 down_routes_partial_sres.
            for bi,(g0,g1) in enumerate(self.ranges):
                main.wait_event(self.gather_ready[i][bi])
                self.pk.down_range(
                  b.mirrors,b.sres.planes[i],b.slots,b.ids,b.route_group,
                  dev["globals"],b.route_act,b.route_plist,b.route_masks,
                  b.route_pcount,rt.fused.e2m1,rt.fused.e4m3,b.partials,
                  b.hidden,b.inter,b.nc,g0,g1
                )

        # Frozen exact Phase23 reduction and route-slot accumulation.
        b.k.reduce(b.partials,b.route_down,b.hidden,b.nc)

        if self.diagnostic:
            self.pipe_end[i].record(main)

        if self.variant.shared_overlap:
            main.wait_event(self.shared_done[i])
            cp.copyto(out,self.shared_out)

        b.k.accumulate(out,b.route_down,b.w,b.hidden)

        stats=None
        if collect_stats:
            stats={
              "layer":i,
              "variant":self.variant.as_dict(),
              "ranges":[list(x) for x in self.ranges],
              "arithmetic_order":"parent_down_chunks_then_parent_reduce_then_slot0_to_slot5_fmaf",
            }
        return None,None,stats

    def profile_snapshot(self):
        if not self.diagnostic:
            raise RuntimeError("diagnostic events are disabled")
        cp=self.cp
        layers=[]
        for li in self.rt.moe_layers:
            i=int(li)
            layers.append({
              "layer":i,
              "pipeline_span_ms":float(cp.cuda.get_elapsed_time(
                self.pipe_start[i],self.pipe_end[i]
              )),
              "gather_stream_span_ms":float(cp.cuda.get_elapsed_time(
                self.gather_start[i],self.gather_end[i]
              )),
            })
        return {
          "variant":self.variant.as_dict(),
          "layers":layers,
          "pipeline_span_ms_per_h4":sum(x["pipeline_span_ms"] for x in layers),
          "gather_stream_span_ms_per_h4":sum(x["gather_stream_span_ms"] for x in layers),
        }


def make_candidate(context:int,variant:Variant,diagnostic:bool=False):
    cfg,_,_=phase27_gate()
    rt,g,keep=make_synth(int(context),cfg)
    wrapped=PipelinedMoEH4(g.gmoe,variant,diagnostic=diagnostic)
    g.gmoe=wrapped
    g.v.moeb=wrapped
    keep=list(keep)+[wrapped]
    return rt,g,keep


def timed_candidate(rt,g,tokens,context,blocks,warmup):
    return timed_synth_blocks(rt,g,tokens,context,blocks,warmup)


def capture_arrays(rt,logits,pos,ids,ids_repeat=None):
    import cupy as cp
    arrays={
      "ids":np.asarray(ids,np.int32),
      "ids_repeat":np.asarray(ids if ids_repeat is None else ids_repeat,np.int32),
      "logits":cp.asnumpy(logits).astype(np.float32,copy=True),
    }
    for k,x in rt.ssm.items():
        arrays[f"ssm_{int(k)}"]=cp.asnumpy(x).astype(np.float32,copy=True)
    for k,x in rt.conv.items():
        arrays[f"conv_{int(k)}"]=cp.asnumpy(x).astype(np.float32,copy=True)
    nk,mc,hd=int(rt.n_kv),int(rt.max_ctx),int(rt.head_dim)
    for li in rt.attn_layers:
        i=int(li)
        arrays[f"k_{i}"]=cp.asnumpy(
          rt.kc[i].reshape(nk,mc,hd)[:,:pos,:]
        ).astype(np.float32,copy=True)
        arrays[f"v_{i}"]=cp.asnumpy(
          rt.vc[i].reshape(nk,mc,hd)[:,:pos,:]
        ).astype(np.float32,copy=True)
    return arrays


def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-30))


def compare_npz(parent,candidate):
    with np.load(parent) as p,np.load(candidate) as c:
        keys=sorted(set(p.files)&set(c.files))
        ssm=max(nrmse(p[k],c[k]) for k in keys if k.startswith("ssm_"))
        conv=max(nrmse(p[k],c[k]) for k in keys if k.startswith("conv_"))
        kv=max(
          [nrmse(p[k],c[k]) for k in keys
           if k.startswith("k_") or k.startswith("v_")] or [0.0]
        )
        logits=nrmse(p["logits"],c["logits"])
        ids=bool(np.array_equal(p["ids"],c["ids"]))
        det=bool(np.array_equal(c["ids"],c["ids_repeat"]))
        finite=all(np.isfinite(c[k]).all() for k in keys)
    state={
      "max_ssm_nrmse":ssm,"max_conv_nrmse":conv,
      "max_kv_nrmse":kv,"logits_nrmse":logits,
    }
    gates={
      "ids_exact":ids,"candidate_deterministic_ids":det,
      "ssm":ssm<=5e-5,"conv":conv<=1e-5,"kv":kv<=5e-6,
      "logits":logits<=5e-4,"finite":bool(finite),
    }
    return state,gates


def robust_cv(vals):
    a=np.asarray(vals,np.float64);m=float(np.median(a))
    return float(1.4826*np.median(np.abs(a-m))/max(abs(m),1e-30))
