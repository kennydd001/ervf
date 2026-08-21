"""Phase40 exact H8 transfer/compute pipeline on the Phase32 parent."""
from __future__ import annotations

import numpy as np

from moe_dev_batched import UP_CODE, UP_SCALE
from scale_resident_kernels import PLANE_BYTES
from moe_dev_batched import DOWN_PANEL_BYTES
from s100_phase25_h8_kernels import H, ROUTES, TOPK
from s100_phase32_common import make_candidate as make_phase32
from s100_phase40_h8_pipeline_kernels import Phase40H8PipelineKernels


class Phase40PipelinedMoEH8:
    def __init__(self, phase32_moe, *, batches: int = 3, gather_y: int = 4):
        import cupy as cp

        self.cp = cp
        self.phase32 = phase32_moe
        self.base = phase32_moe.base
        self.rt = phase32_moe.rt
        self.pipeline = Phase40H8PipelineKernels()
        self.ranges = self.pipeline.ranges(int(batches))
        self.gather_y = int(gather_y)
        self.gather_stream = cp.cuda.Stream(non_blocking=True)
        self.mask_ready = {int(i): cp.cuda.Event() for i in self.rt.moe_layers}
        self.gather_ready = {
            int(i): tuple(cp.cuda.Event() for _ in self.ranges)
            for i in self.rt.moe_layers
        }

    def __getattr__(self, name):
        return getattr(self.phase32, name)

    def __call__(self, layer, normed, out, collect_stats=False):
        cp, p32, b, rt = self.cp, self.phase32, self.base, self.rt
        i = int(layer)
        d, bank, cache = rt.layer[i], rt.bank[i], rt.cache[i]
        dev = b._dev(i)
        main = cp.cuda.get_current_stream()

        p32._shared_fork(i, d, normed, main)
        p32._dense_f32(d["gate_w"], normed, b.rlog, b.nexp, b.hidden)
        for token in range(H):
            begin = token * TOPK
            rt.fused.route_topk(
                b.rlog[token], d["gate_b"],
                b.ids[begin : begin + TOPK], b.w[begin : begin + TOPK],
                b.nexp, TOPK, rt.scaling, bad_pick=rt._bad_pick,
            )

        b.k.cache_assign(dev, b.ids, b.slots, b.need, int(cache["cap"]))
        b.k.group(
            b.ids, b.route_group, b.group_ids, b.group_count,
            b.group_refs, b.ngroups,
        )

        rt.evt[0].record(main)
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
                (ROUTES, 64), (256,),
                (
                    np.uint64(bank["up_codes"].ctypes.data),
                    np.uint64(bank["up_scales"].ctypes.data),
                    cache["codes"], cache["scales"], b.ids, b.slots, b.need,
                    np.uint64(UP_CODE), np.uint64(UP_SCALE),
                ),
            )
            b.sres.fetch_plane_k(
                (ROUTES, 64), (256,),
                (
                    np.uint64(bank["down_base_ptr"]), b.sres.planes[i],
                    b.ids, b.slots, b.need,
                    np.uint64(DOWN_PANEL_BYTES), np.uint64(PLANE_BYTES),
                    np.int32(b.hidden), np.int32(b.npanel),
                ),
            )
            rt.evt[1].record(rt.copy_stream)

        main.wait_event(rt.evt[1])
        for multiplicity in range(1, 9):
            b.k.up_direct(
                multiplicity,
                cache["codes"], cache["scales"], b.slots, b.ids,
                dev["globals"], rt.fused.e2m1, rt.fused.e4m3,
                normed, b.route_act, b.inter, b.hidden,
                UP_CODE, UP_SCALE, b.group_count, b.group_refs,
            )

        b.k.scan(
            b.route_act, b.group_count, b.group_refs,
            b.route_masks, b.route_plist, b.route_pcount,
            b.union_masks, b.union_plist, b.union_pcount,
            b.union_nz, b.union_nzc, b.inter,
        )

        self.mask_ready[i].record(main)
        with self.gather_stream:
            self.gather_stream.wait_event(self.mask_ready[i])
            for batch_index, (begin, end) in enumerate(self.ranges):
                self.pipeline.gather_range(
                    int(bank["down_base_ptr"]), b.group_ids, b.group_count,
                    b.union_nz, b.union_nzc, b.mirrors, b.hidden, b.inter,
                    begin, end, self.gather_y,
                )
                self.gather_ready[i][batch_index].record(self.gather_stream)

        for batch_index, (begin, end) in enumerate(self.ranges):
            main.wait_event(self.gather_ready[i][batch_index])
            self.pipeline.down_range(
                b.mirrors, b.sres.planes[i], b.slots, b.ids,
                b.route_group, dev["globals"], b.route_act,
                b.route_plist, b.route_masks, b.route_pcount,
                rt.fused.e2m1, rt.fused.e4m3, b.partials,
                b.hidden, b.inter, b.nc, begin, end,
            )

        b.k.reduce(b.partials, b.route_down, b.hidden, b.nc)
        main.wait_event(p32.done_events[i])
        cp.copyto(out, p32.shared_out)
        b.k.accumulate(out, b.route_down, b.w, b.hidden)

        stats = None
        if collect_stats:
            stats = {
                "layer": i,
                "phase40_ranges": [list(pair) for pair in self.ranges],
                "gather_y": self.gather_y,
                "arithmetic_order": "unchanged_route_chunk_reduce_then_slot0_to5_fmaf",
            }
        return None, None, stats


def make_candidate(context: int):
    rt, graph, keep = make_phase32(int(context), "dense_m8")
    wrapped = Phase40PipelinedMoEH8(graph.gmoe, batches=3, gather_y=4)
    graph.gmoe = wrapped
    return rt, graph, list(keep) + [wrapped, wrapped.pipeline]


def make_parent(context: int):
    return make_phase32(int(context), "dense_m8")

