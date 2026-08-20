from __future__ import annotations

import json
import traceback
import numpy as np

from common import utc_now,write_json_atomic
from moe_dev_batched import DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES
from s100_phase24_sres_grouped import GroupedScaleResidentKernels
from s100_phase27_common import RESULTS,phase27_gate
from s100_phase27_kernels import (
    Phase27DownPipelineKernels,GROUPS,ROUTES,NCHUNKS,
)

OUT=RESULTS/"S100_PHASE27_PREFLIGHT.json"
ROWS=2688
INTER=1856
NPANEL=INTER//16


def prop(props,name,default=None):
    if name in props:return props[name]
    b=name.encode()
    if b in props:return props[b]
    return default


def main():
    payload={
      "kind":"s100_phase27_preflight","status":"started",
      "started_utc":utc_now(),
      "claim_boundary":"synthetic exact kernel/capture preflight; not model throughput",
    }
    try:
        phase27_gate()
        import cupy as cp

        pk=Phase27DownPipelineKernels()
        parent=GroupedScaleResidentKernels()

        # ---------------- gather exactness ----------------
        nexp=GROUPS
        bank_bytes=nexp*DOWN_PANEL_BYTES
        down_base=(cp.arange(bank_bytes,dtype=cp.uint32)&255).astype(cp.uint8)

        group_ids=cp.arange(GROUPS,dtype=cp.int32)
        group_count=cp.ones(GROUPS,dtype=cp.int32)
        union_nz_h=np.zeros(GROUPS*INTER,dtype=np.int32)
        union_nzc_h=np.zeros(GROUPS,dtype=np.int32)
        for g in range(GROUPS):
            cols=sorted({int((g*17+q*113)%INTER) for q in range(19)})
            union_nzc_h[g]=len(cols)
            union_nz_h[g*INTER:g*INTER+len(cols)]=cols
        union_nz=cp.asarray(union_nz_h)
        union_nzc=cp.asarray(union_nzc_h)

        mirror_parent=cp.zeros(GROUPS*DOWN_PANEL_BYTES,dtype=cp.uint8)
        mirror_range=cp.zeros_like(mirror_parent)

        parent.gather(
          down_base.data.ptr,group_ids,group_count,
          union_nz,union_nzc,mirror_parent,ROWS,INTER
        )
        for g0,g1 in pk.ranges(3):
            pk.gather_range(
              down_base.data.ptr,group_ids,group_count,
              union_nz,union_nzc,mirror_range,ROWS,INTER,
              g0,g1,8
            )
        cp.cuda.get_current_stream().synchronize()
        gather_exact=bool(cp.asnumpy(cp.array_equal(mirror_parent,mirror_range)))

        # ---------------- down partial exactness ----------------
        # Use a fully populated deterministic mirror here, independent of the
        # sparse gather test above, so every active down byte is meaningful.
        # Because group_ids[g] == g, copying the full synthetic bank produces
        # exactly one complete expert record per group mirror region.
        mirror_down=down_base.copy()

        slots=cp.arange(ROUTES,dtype=cp.int32)
        ids=cp.arange(ROUTES,dtype=cp.int32)
        route_group=cp.arange(ROUTES,dtype=cp.int32)
        globals_dev=cp.ones(ROUTES*2,dtype=cp.float32)

        route_plist_h=np.zeros(ROUTES*NPANEL,dtype=np.int32)
        route_masks_h=np.zeros(ROUTES*NPANEL,dtype=np.uint32)
        route_pcount_h=np.full(ROUTES,8,dtype=np.int32)
        act_h=np.zeros((ROUTES,INTER),dtype=np.float32)
        active_bits=(0,3,7,12)
        for r in range(ROUTES):
            panels=[int((r*5+k*11)%NPANEL) for k in range(8)]
            panels=sorted(set(panels))
            while len(panels)<8:
                panels.append((panels[-1]+1)%NPANEL)
                panels=sorted(set(panels))
            panels=panels[:8]
            route_pcount_h[r]=len(panels)
            route_plist_h[r*NPANEL:r*NPANEL+len(panels)]=panels
            for p in panels:
                mk=0
                for c in active_bits:
                    mk|=1<<c
                    act_h[r,p*16+c]=np.float32(
                      0.01*(r+1)+0.001*(p+1)+0.0001*(c+1)
                    )
                route_masks_h[r*NPANEL+p]=mk

        route_plist=cp.asarray(route_plist_h)
        route_masks=cp.asarray(route_masks_h)
        route_pcount=cp.asarray(route_pcount_h)
        act=cp.asarray(act_h.reshape(-1))

        # The complete Lightning routed down-scale bank was audited before this
        # repair: all 2,944 tensors (917,962,752 bytes) use FP8 scale bytes
        # 62..124, with zero bytes >= 128.  Keep the synthetic contract inside
        # that proven checkpoint domain; values 128..255 are unreachable here.
        planes=(62+(cp.arange(ROUTES*PLANE_BYTES,dtype=cp.uint32)%63)).astype(cp.uint8)
        e2=cp.linspace(-1.0,1.0,16,dtype=cp.float32)
        e4=cp.linspace(0.25,1.75,256,dtype=cp.float32)

        partial_parent=cp.zeros(ROUTES*NCHUNKS*ROWS,dtype=cp.float32)
        partial_range=cp.zeros_like(partial_parent)

        parent.down(
          mirror_down,planes,slots,ids,route_group,globals_dev,
          act,route_plist,route_masks,route_pcount,e2,e4,
          partial_parent,ROWS,INTER,NCHUNKS
        )
        for g0,g1 in pk.ranges(3):
            pk.down_range(
              mirror_down,planes,slots,ids,route_group,globals_dev,
              act,route_plist,route_masks,route_pcount,e2,e4,
              partial_range,ROWS,INTER,NCHUNKS,g0,g1
            )
        cp.cuda.get_current_stream().synchronize()
        partial_exact=bool(cp.asnumpy(cp.array_equal(partial_parent,partial_range)))
        finite=bool(cp.asnumpy(cp.isfinite(partial_range).all()))

        # ---------------- actual cross-stream capture topology ----------------
        mirror_graph=mirror_down.copy()
        mirror_graph_ref=mirror_down.copy()
        parent.gather(
          down_base.data.ptr,group_ids,group_count,
          union_nz,union_nzc,mirror_graph_ref,ROWS,INTER
        )
        partial_graph=cp.zeros_like(partial_parent)
        main_stream=cp.cuda.Stream(non_blocking=True)
        gather_stream=cp.cuda.Stream(non_blocking=True)
        fork=cp.cuda.Event()
        ready=tuple(cp.cuda.Event() for _ in pk.ranges(3))

        # Warm every kernel and stream outside capture.
        with gather_stream:
            for g0,g1 in pk.ranges(3):
                pk.gather_range(
                  down_base.data.ptr,group_ids,group_count,
                  union_nz,union_nzc,mirror_graph,ROWS,INTER,
                  g0,g1,8
                )
        gather_stream.synchronize()
        with main_stream:
            for g0,g1 in pk.ranges(3):
                pk.down_range(
                  mirror_graph,planes,slots,ids,route_group,globals_dev,
                  act,route_plist,route_masks,route_pcount,e2,e4,
                  partial_graph,ROWS,INTER,NCHUNKS,g0,g1
                )
        main_stream.synchronize()
        cp.copyto(mirror_graph,mirror_down);partial_graph.fill(0)
        cp.cuda.Device(0).synchronize()

        main_stream.begin_capture()
        with main_stream:
            fork.record(main_stream)
            with gather_stream:
                gather_stream.wait_event(fork)
                for bi,(g0,g1) in enumerate(pk.ranges(3)):
                    pk.gather_range(
                      down_base.data.ptr,group_ids,group_count,
                      union_nz,union_nzc,mirror_graph,ROWS,INTER,
                      g0,g1,8
                    )
                    ready[bi].record(gather_stream)
            for bi,(g0,g1) in enumerate(pk.ranges(3)):
                main_stream.wait_event(ready[bi])
                pk.down_range(
                  mirror_graph,planes,slots,ids,route_group,globals_dev,
                  act,route_plist,route_masks,route_pcount,e2,e4,
                  partial_graph,ROWS,INTER,NCHUNKS,g0,g1
                )
        graph=main_stream.end_capture()
        graph.launch(main_stream)
        main_stream.synchronize()

        graph_gather_exact=bool(cp.asnumpy(cp.array_equal(mirror_graph_ref,mirror_graph)))
        graph_partial_exact=bool(cp.asnumpy(cp.array_equal(partial_parent,partial_graph)))

        props=cp.cuda.runtime.getDeviceProperties(0)
        concurrent=int(prop(props,"concurrentKernels",0) or 0)
        async_engines=int(prop(props,"asyncEngineCount",0) or 0)

        green=bool(
          gather_exact and partial_exact and finite
          and graph_gather_exact and graph_partial_exact
        )
        payload.update({
          "status":"measured",
          "gather_range_byte_exact":gather_exact,
          "down_range_partial_bit_exact":partial_exact,
          "down_range_finite":finite,
          "graph_gather_byte_exact":graph_gather_exact,
          "graph_down_partial_bit_exact":graph_partial_exact,
          "concurrentKernels":concurrent,
          "asyncEngineCount":async_engines,
          "PREFLIGHT_GREEN":green,
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({
          "status":"technical_failure",
          "PREFLIGHT_GREEN":False,
          "error":{
            "type":type(exc).__name__,
            "message":str(exc),
            "traceback":traceback.format_exc(),
          },
          "completed_utc":utc_now(),
        })

    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2))
    return 0 if payload.get("PREFLIGHT_GREEN") else 2

if __name__=="__main__":
    raise SystemExit(main())
