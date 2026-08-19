from __future__ import annotations

from collections import Counter
import json
import numpy as np

from common import REPO
from moe_dev_batched import UP_CODE,UP_SCALE,DOWN_PANEL_BYTES
from s100_phase23_group_kernels import (
    Phase23Kernels,ROUTES,GROUPS,TOPK,MAXM,
)
from s100_phase22_common import (
    GraphH4Verifier,eager_verifier,selected_head_mode,
)

RESULTS=REPO/"pro_research"/"results"/"s100_phase23"

class GPUGroupedMoEH4:
    def __init__(self,rt):
        import cupy as cp
        self.cp=cp;self.rt=rt;self.k=Phase23Kernels()
        self.hidden=int(rt.hidden);self.inter=int(rt.moe_inter)
        self.shared=int(rt.shared_inter);self.nexp=int(rt.n_experts)
        self.npanel=self.inter//16;self.nc=int(rt.fused.nchunks)

        self.rlog=cp.empty((4,self.nexp),cp.float32)
        self.ids=cp.empty(ROUTES,cp.int32)
        self.w=cp.empty(ROUTES,cp.float32)
        self.slots=cp.empty(ROUTES,cp.int32)
        self.need=cp.empty(ROUTES,cp.int32)

        self.route_group=cp.empty(ROUTES,cp.int32)
        self.group_ids=cp.empty(GROUPS,cp.int32)
        self.group_count=cp.empty(GROUPS,cp.int32)
        self.group_refs=cp.empty(GROUPS*MAXM,cp.int32)
        self.ngroups=cp.zeros(1,cp.int32)
        self.k.bind_group_arrays(self.group_count,self.group_refs)

        self.shared_act=cp.empty((4,self.shared),cp.float32)
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
        self.partials=cp.empty((ROUTES,self.nc,self.hidden),cp.float32)
        self.route_down=cp.empty((ROUTES,self.hidden),cp.float32)

    def _dev(self,layer):
        rt=self.rt;bank=rt.bank[int(layer)];c=rt.cache[int(layer)]
        if not hasattr(rt,"_dev_cache"):rt._dev_cache={}
        if int(layer) not in rt._dev_cache:
            rt._dev_cache[int(layer)]=rt.fused.alloc_device_cache(
                rt.n_experts,int(c["cap"]),rt.top_k,bank["globals"]
            )
        return rt._dev_cache[int(layer)]

    def __call__(self,layer,normed,out,collect_stats=False):
        cp=self.cp;rt=self.rt;i=int(layer);d=rt.layer[i]
        bank,c=rt.bank[i],rt.cache[i];dev=self._dev(i)

        # Four independent production routers, no host readback.
        for t in range(4):
            a=t*TOPK;b=a+TOPK
            rt.k.mv_f32(
                self.rlog[t],d["gate_w"],normed[t],self.nexp,self.hidden
            )
            rt.fused.route_topk(
                self.rlog[t],d["gate_b"],self.ids[a:b],self.w[a:b],
                self.nexp,TOPK,rt.scaling,bad_pick=rt._bad_pick
            )

        # Exact token-major LRU semantics over all 24 routes.
        self.k.cache_assign(dev,self.ids,self.slots,self.need,int(c["cap"]))
        self.k.group(
            self.ids,self.route_group,self.group_ids,self.group_count,
            self.group_refs,self.ngroups
        )
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
            rt.evt[1].record(rt.copy_stream)

        # Shared expert uses the already-proven production M1 path x4.
        out.fill(0)
        for t in range(4):
            rt.fused.gemv_into(
                self.shared_act[t],d["sh_up_c"],d["sh_up_s"],normed[t],
                d["sh_up_g"],self.shared,self.hidden,apply_relu2=True
            )
            rt.fused.gemv_into(
                out[t],d["sh_dn_c"],d["sh_dn_s"],self.shared_act[t],
                d["sh_dn_g"],self.hidden,self.shared
            )

        cp.cuda.get_current_stream().wait_event(rt.evt[1])

        # ERVF-M dispatch. Each active group is executed by exactly one M kernel.
        for m in (1,2,3,4):
            self.k.up(
                m,c["codes"],c["scales"],self.slots,self.ids,dev["globals"],
                rt.fused.e2m1,rt.fused.e4m3,normed,self.route_act,
                self.inter,self.hidden,UP_CODE,UP_SCALE
            )

        # Row-specific arithmetic metadata + group-union gather metadata.
        self.k.scan(
            self.route_act,self.group_count,self.group_refs,
            self.route_masks,self.route_plist,self.route_pcount,
            self.union_masks,self.union_plist,self.union_pcount,
            self.union_nz,self.union_nzc,self.inter
        )
        self.k.gather(
            int(bank["down_base_ptr"]),self.group_ids,self.group_count,
            self.union_plist,self.union_pcount,self.union_nz,self.union_nzc,
            self.mirrors,DOWN_PANEL_BYTES,self.hidden,self.inter
        )
        self.k.down(
            self.mirrors,self.ids,self.route_group,dev["globals"],
            self.route_act,self.route_plist,self.route_masks,self.route_pcount,
            rt.fused.e2m1,rt.fused.e4m3,self.partials,
            DOWN_PANEL_BYTES,self.hidden,self.inter,self.nc
        )
        self.k.reduce(self.partials,self.route_down,self.hidden,self.nc)
        self.k.accumulate(out,self.route_down,self.w,self.hidden)

        stats=None
        if collect_stats:
            cp.cuda.get_current_stream().synchronize()
            gcnt=cp.asnumpy(self.group_count)
            ng=int(cp.asnumpy(self.ngroups)[0])
            active=[int(x) for x in gcnt[:ng]]
            if not active or any(x<1 or x>4 for x in active):
                raise RuntimeError(f"invalid device group counts {active}")

            upc=cp.asnumpy(self.union_pcount)[:ng]
            unz=cp.asnumpy(self.union_nzc)[:ng]
            rpc=cp.asnumpy(self.route_pcount)
            rms=cp.asnumpy(self.route_masks)
            route_nz=[]
            for r in range(ROUTES):
                route_nz.append(sum((int(x)&0xFFFF).bit_count() for x in rms[r]))
            rowhalf=self.hidden//2
            base_down=sum(
                int(route_nz[r])*rowhalf+int(rpc[r])*self.hidden
                for r in range(ROUTES)
            )
            group_down=sum(
                int(unz[g])*rowhalf+int(upc[g])*self.hidden
                for g in range(ng)
            )
            stats={
                "layer":i,"ngroups":ng,
                "repeat_rate":1.0-ng/float(ROUTES),
                "group_counts":active,
                "m_histogram":dict(Counter(active)),
                "up_weight_streams_parent":ROUTES,
                "up_weight_streams_grouped":ng,
                "up_stream_reduction_fraction":1.0-ng/float(ROUTES),
                "down_sparse_bytes_parent_est":int(base_down),
                "down_sparse_bytes_grouped":int(group_down),
                "down_byte_reduction_fraction":(
                    1.0-group_down/float(base_down) if base_down else 0.0
                ),
                "cache_miss_routes":int(cp.asnumpy(self.need).sum()),
            }
        return None,None,stats

def grouped_eager_verifier(rt):
    mode=selected_head_mode()
    v=eager_verifier(rt,mode)
    v.moeb=GPUGroupedMoEH4(rt)
    return v

class GraphH4VerifierGrouped(GraphH4Verifier):
    def __init__(self,rt,head_mode=None):
        super().__init__(rt,head_mode or selected_head_mode())
        self.gmoe=GPUGroupedMoEH4(rt)
        self.v.moeb=self.gmoe

    def body(self):
        rt=self.rt;v=self.v
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
        try:
            txt=self.graph.debug_dot_str()
            if isinstance(txt,bytes):txt=txt.decode("utf-8","replace")
            lo=txt.lower()
            info["grouped_dot"]={
                "group_routes24":"group_routes24" in lo,
                "cache_assign_h4":"cache_assign_h4" in lo,
                "up_m1":"grouped_up_m1" in lo,
                "up_m2":"grouped_up_m2" in lo,
                "up_m3":"grouped_up_m3" in lo,
                "up_m4":"grouped_up_m4" in lo,
                "group_union_gather":"gather_group_union" in lo,
                "row_specific_down":"down_routes_partial" in lo,
                "accumulate_h4":"accumulate_h4" in lo,
            }
        except Exception as exc:
            info["grouped_dot"]={"error":f"{type(exc).__name__}: {exc}"}
        return info
