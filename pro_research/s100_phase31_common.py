from __future__ import annotations

import math
import types

import numpy as np

from common import REPO
from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE
from scale_resident_kernels import PLANE_BYTES
from s100_phase24_common import make_synth
from s100_phase24_common import SynthesisConfig
from s100_phase27_common import ROUTES, phase27_gate
from s100_phase30e_common import Phase30EMoEH4
from s100_phase30e_shared_kernels import SharedOccupancyKernels
from s100_phase31_group_down_kernels import Phase31GroupDownKernels
from s100_phase31_dense_direct_kernels import Phase31DenseDirectKernels
from s100_phase31_residual_kernels import Phase31ResidualKernels
from s100_phase31_staged_kernels import Phase31StagedKernels


MODES = ("sink", "reduce_sink")
RESULTS = REPO / "pro_research" / "results" / "s100_phase31"


class Phase31MoEH4(Phase30EMoEH4):
    """Phase30E with exact MoE terminal fusion into the residual stream."""

    def __init__(self, base, *, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(base, shared_direct=True, group_dispatch=True)
        self.mode = mode
        self.residual_kernels = Phase31ResidualKernels()

    def forward_residual(self, layer, normed, residual, collect_stats=False):
        cp = self.cp
        b = self.base
        rt = self.rt
        i = int(layer)
        d = rt.layer[i]
        bank, c = rt.bank[i], rt.cache[i]
        dev = b._dev(i)
        main = cp.cuda.get_current_stream()

        # Exact Phase30E shared branch: direct-L2 M4 on its own stream.
        self._shared_fork(i, d, normed, main)

        # Frozen router/cache/group path.
        for token in range(self.H):
            start = token * self.TOPK
            stop = start + self.TOPK
            rt.k.mv_f32(
                b.rlog[token], d["gate_w"], normed[token], b.nexp, b.hidden
            )
            rt.fused.route_topk(
                b.rlog[token],
                d["gate_b"],
                b.ids[start:stop],
                b.w[start:stop],
                b.nexp,
                self.TOPK,
                rt.scaling,
                bad_pick=rt._bad_pick,
            )

        b.k.cache_assign(dev, b.ids, b.slots, b.need, int(c["cap"]))
        b.k.group(
            b.ids,
            b.route_group,
            b.group_ids,
            b.group_count,
            b.group_refs,
            b.ngroups,
        )

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["up_codes"].ctypes.data),
                    np.uint64(bank["up_scales"].ctypes.data),
                    c["codes"],
                    c["scales"],
                    b.ids,
                    b.slots,
                    b.need,
                    np.uint64(UP_CODE),
                    np.uint64(UP_SCALE),
                ),
            )
            if i not in b.sres.planes:
                raise RuntimeError(f"Phase31 requires H-SCALE plane on layer {i}")
            b.sres.fetch_plane_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["down_base_ptr"]),
                    b.sres.planes[i],
                    b.ids,
                    b.slots,
                    b.need,
                    np.uint64(DOWN_PANEL_BYTES),
                    np.uint64(PLANE_BYTES),
                    np.int32(b.hidden),
                    np.int32(b.npanel),
                ),
            )
            rt.evt[1].record(rt.copy_stream)

        main.wait_event(rt.evt[1])

        # Phase30E two-launch M1-2/M3-4 routed-UP dispatch.
        for multiplicity in (1, 2, 3, 4):
            b.k.up(
                multiplicity,
                c["codes"],
                c["scales"],
                b.slots,
                b.ids,
                dev["globals"],
                rt.fused.e2m1,
                rt.fused.e4m3,
                normed,
                b.route_act,
                b.inter,
                b.hidden,
                UP_CODE,
                UP_SCALE,
            )

        b.k.scan(
            b.route_act,
            b.group_count,
            b.group_refs,
            b.route_masks,
            b.route_plist,
            b.route_pcount,
            b.union_masks,
            b.union_plist,
            b.union_pcount,
            b.union_nz,
            b.union_nzc,
            b.inter,
        )

        # Frozen Phase27R three-batch gather/down pipeline.
        self.mask_ready[i].record(main)
        with self.gather_stream:
            self.gather_stream.wait_event(self.mask_ready[i])
            for batch_index, (begin, end) in enumerate(self.ranges):
                self.pk.gather_range(
                    int(bank["down_base_ptr"]),
                    b.group_ids,
                    b.group_count,
                    b.union_nz,
                    b.union_nzc,
                    b.mirrors,
                    b.hidden,
                    b.inter,
                    begin,
                    end,
                    self.variant.gather_y,
                )
                self.gather_ready[i][batch_index].record(self.gather_stream)

        for batch_index, (begin, end) in enumerate(self.ranges):
            main.wait_event(self.gather_ready[i][batch_index])
            self.pk.down_range(
                b.mirrors,
                b.sres.planes[i],
                b.slots,
                b.ids,
                b.route_group,
                dev["globals"],
                b.route_act,
                b.route_plist,
                b.route_masks,
                b.route_pcount,
                rt.fused.e2m1,
                rt.fused.e4m3,
                b.partials,
                b.hidden,
                b.inter,
                b.nc,
                begin,
                end,
            )

        # The sink arm preserves Phase30E's reduction-before-shared-wait
        # overlap.  The reduce_sink arm removes route_down as well.
        if self.mode == "sink":
            b.k.reduce(b.partials, b.route_down, b.hidden, b.nc)

        main.wait_event(self.shared_done[i])
        if self.mode == "sink":
            self.residual_kernels.sink(
                b.route_down, self.shared_out, b.w, residual, b.hidden
            )
        else:
            self.residual_kernels.reduce_sink(
                b.partials,
                self.shared_out,
                b.w,
                residual,
                b.hidden,
                b.nc,
            )

        stats = None
        if collect_stats:
            stats = {
                "layer": i,
                "phase31_mode": self.mode,
                "ranges": [list(x) for x in self.ranges],
                "arithmetic_order": (
                    "chunk0_to_n_then_slot0_to5_fmaf_then_fp32_residual_add"
                ),
            }
        return None, None, stats


class Phase31StagedMoEH4(Phase30EMoEH4):
    """Phase30E with independent M1-2 and M3-4 UP/scan/gather/down lanes."""

    LANES = (
        ("m12", "split_m12", 1, 2),
        ("m34", "split_m34", 3, 4),
    )

    def __init__(self, base):
        import cupy as cp

        super().__init__(base, shared_direct=True, group_dispatch=True)
        self.staged_kernels = Phase31StagedKernels()
        self.lane_gather_stream = {
            name: cp.cuda.Stream(non_blocking=True)
            for name, _, _, _ in self.LANES
        }
        self.lane_down_stream = {
            name: cp.cuda.Stream(non_blocking=True)
            for name, _, _, _ in self.LANES
        }
        self.up_ready = {
            int(layer): {
                name: cp.cuda.Event() for name, _, _, _ in self.LANES
            }
            for layer in self.rt.moe_layers
        }
        self.lane_gather_ready = {
            int(layer): {
                name: tuple(cp.cuda.Event() for _ in self.ranges)
                for name, _, _, _ in self.LANES
            }
            for layer in self.rt.moe_layers
        }
        self.lane_done = {
            int(layer): {
                name: cp.cuda.Event() for name, _, _, _ in self.LANES
            }
            for layer in self.rt.moe_layers
        }

    def __call__(self, layer, normed, out, collect_stats=False):
        cp = self.cp
        b = self.base
        rt = self.rt
        i = int(layer)
        d = rt.layer[i]
        bank, c = rt.bank[i], rt.cache[i]
        dev = b._dev(i)
        main = cp.cuda.get_current_stream()

        self._shared_fork(i, d, normed, main)

        for token in range(self.H):
            start = token * self.TOPK
            stop = start + self.TOPK
            rt.k.mv_f32(
                b.rlog[token], d["gate_w"], normed[token], b.nexp, b.hidden
            )
            rt.fused.route_topk(
                b.rlog[token],
                d["gate_b"],
                b.ids[start:stop],
                b.w[start:stop],
                b.nexp,
                self.TOPK,
                rt.scaling,
                bad_pick=rt._bad_pick,
            )

        b.k.cache_assign(dev, b.ids, b.slots, b.need, int(c["cap"]))
        b.k.group(
            b.ids,
            b.route_group,
            b.group_ids,
            b.group_count,
            b.group_refs,
            b.ngroups,
        )

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["up_codes"].ctypes.data),
                    np.uint64(bank["up_scales"].ctypes.data),
                    c["codes"],
                    c["scales"],
                    b.ids,
                    b.slots,
                    b.need,
                    np.uint64(UP_CODE),
                    np.uint64(UP_SCALE),
                ),
            )
            if i not in b.sres.planes:
                raise RuntimeError(f"Phase31B requires H-SCALE plane on layer {i}")
            b.sres.fetch_plane_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["down_base_ptr"]),
                    b.sres.planes[i],
                    b.ids,
                    b.slots,
                    b.need,
                    np.uint64(DOWN_PANEL_BYTES),
                    np.uint64(PLANE_BYTES),
                    np.int32(b.hidden),
                    np.int32(b.npanel),
                ),
            )
            rt.evt[1].record(rt.copy_stream)

        main.wait_event(rt.evt[1])
        up_args = (
            c["codes"],
            c["scales"],
            b.slots,
            b.ids,
            dev["globals"],
            b.group_count,
            b.group_refs,
            rt.fused.e2m1,
            rt.fused.e4m3,
            normed,
            b.route_act,
            b.inter,
            b.hidden,
            UP_CODE,
            UP_SCALE,
        )

        # Launch the first producer and immediately release its consumer lane.
        # While lane M1-2 scans/gathers/down-projects, main computes M3-4.
        for name, kernel, count_lo, count_hi in self.LANES:
            self.group_dispatch._launch(kernel, *up_args)
            self.up_ready[i][name].record(main)
            gather_stream = self.lane_gather_stream[name]
            down_stream = self.lane_down_stream[name]
            with gather_stream:
                gather_stream.wait_event(self.up_ready[i][name])
                self.staged_kernels.scan(
                    b.route_act,
                    b.group_count,
                    b.group_refs,
                    b.route_masks,
                    b.route_plist,
                    b.route_pcount,
                    b.union_masks,
                    b.union_plist,
                    b.union_pcount,
                    b.union_nz,
                    b.union_nzc,
                    b.inter,
                    count_lo,
                    count_hi,
                )
                for batch_index, (begin, end) in enumerate(self.ranges):
                    self.staged_kernels.gather(
                        int(bank["down_base_ptr"]),
                        b.group_ids,
                        b.group_count,
                        b.union_nz,
                        b.union_nzc,
                        b.mirrors,
                        b.hidden,
                        b.inter,
                        begin,
                        end,
                        self.variant.gather_y,
                        count_lo,
                        count_hi,
                    )
                    self.lane_gather_ready[i][name][batch_index].record(
                        gather_stream
                    )

            with down_stream:
                for batch_index, (begin, end) in enumerate(self.ranges):
                    down_stream.wait_event(
                        self.lane_gather_ready[i][name][batch_index]
                    )
                    self.staged_kernels.down(
                        b.mirrors,
                        b.sres.planes[i],
                        b.slots,
                        b.ids,
                        b.route_group,
                        b.group_count,
                        dev["globals"],
                        b.route_act,
                        b.route_plist,
                        b.route_masks,
                        b.route_pcount,
                        rt.fused.e2m1,
                        rt.fused.e4m3,
                        b.partials,
                        b.hidden,
                        b.inter,
                        b.nc,
                        begin,
                        end,
                        count_lo,
                        count_hi,
                    )
                self.lane_done[i][name].record(down_stream)

        for name, _, _, _ in self.LANES:
            main.wait_event(self.lane_done[i][name])

        b.k.reduce(b.partials, b.route_down, b.hidden, b.nc)
        main.wait_event(self.shared_done[i])
        cp.copyto(out, self.shared_out)
        b.k.accumulate(out, b.route_down, b.w, b.hidden)

        stats = None
        if collect_stats:
            stats = {
                "layer": i,
                "phase31_mode": "staged_multiplicity_lanes",
                "lanes": [
                    {"name": name, "count_range": [lo, hi]}
                    for name, _, lo, hi in self.LANES
                ],
                "ranges": [list(x) for x in self.ranges],
                "arithmetic_order": (
                    "parent_down_chunks_then_parent_reduce_then_slot0_to5_fmaf"
                ),
            }
        return None, None, stats


class Phase31GroupDownMoEH4(Phase30EMoEH4):
    """Phase30E with exact cross-route code reuse in sparse-down."""

    def __init__(self, base):
        super().__init__(base, shared_direct=True, group_dispatch=True)
        self.group_down = Phase31GroupDownKernels()

    def __call__(self, layer, normed, out, collect_stats=False):
        cp = self.cp
        b = self.base
        rt = self.rt
        i = int(layer)
        d = rt.layer[i]
        bank, c = rt.bank[i], rt.cache[i]
        dev = b._dev(i)
        main = cp.cuda.get_current_stream()

        self._shared_fork(i, d, normed, main)

        for token in range(self.H):
            start = token * self.TOPK
            stop = start + self.TOPK
            rt.k.mv_f32(
                b.rlog[token], d["gate_w"], normed[token], b.nexp, b.hidden
            )
            rt.fused.route_topk(
                b.rlog[token],
                d["gate_b"],
                b.ids[start:stop],
                b.w[start:stop],
                b.nexp,
                self.TOPK,
                rt.scaling,
                bad_pick=rt._bad_pick,
            )

        b.k.cache_assign(dev, b.ids, b.slots, b.need, int(c["cap"]))
        b.k.group(
            b.ids,
            b.route_group,
            b.group_ids,
            b.group_count,
            b.group_refs,
            b.ngroups,
        )

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["up_codes"].ctypes.data),
                    np.uint64(bank["up_scales"].ctypes.data),
                    c["codes"],
                    c["scales"],
                    b.ids,
                    b.slots,
                    b.need,
                    np.uint64(UP_CODE),
                    np.uint64(UP_SCALE),
                ),
            )
            if i not in b.sres.planes:
                raise RuntimeError(f"Phase31C requires H-SCALE plane on layer {i}")
            b.sres.fetch_plane_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["down_base_ptr"]),
                    b.sres.planes[i],
                    b.ids,
                    b.slots,
                    b.need,
                    np.uint64(DOWN_PANEL_BYTES),
                    np.uint64(PLANE_BYTES),
                    np.int32(b.hidden),
                    np.int32(b.npanel),
                ),
            )
            rt.evt[1].record(rt.copy_stream)

        main.wait_event(rt.evt[1])

        for multiplicity in (1, 2, 3, 4):
            b.k.up(
                multiplicity,
                c["codes"],
                c["scales"],
                b.slots,
                b.ids,
                dev["globals"],
                rt.fused.e2m1,
                rt.fused.e4m3,
                normed,
                b.route_act,
                b.inter,
                b.hidden,
                UP_CODE,
                UP_SCALE,
            )

        b.k.scan(
            b.route_act,
            b.group_count,
            b.group_refs,
            b.route_masks,
            b.route_plist,
            b.route_pcount,
            b.union_masks,
            b.union_plist,
            b.union_pcount,
            b.union_nz,
            b.union_nzc,
            b.inter,
        )

        self.mask_ready[i].record(main)
        with self.gather_stream:
            self.gather_stream.wait_event(self.mask_ready[i])
            for batch_index, (begin, end) in enumerate(self.ranges):
                self.pk.gather_range(
                    int(bank["down_base_ptr"]),
                    b.group_ids,
                    b.group_count,
                    b.union_nz,
                    b.union_nzc,
                    b.mirrors,
                    b.hidden,
                    b.inter,
                    begin,
                    end,
                    self.variant.gather_y,
                )
                self.gather_ready[i][batch_index].record(self.gather_stream)

        for batch_index, (begin, end) in enumerate(self.ranges):
            main.wait_event(self.gather_ready[i][batch_index])
            self.group_down.launch_range(
                b.mirrors,
                b.sres.planes[i],
                b.slots,
                b.ids,
                b.group_count,
                b.group_refs,
                dev["globals"],
                b.route_act,
                b.route_plist,
                b.route_masks,
                b.route_pcount,
                rt.fused.e2m1,
                rt.fused.e4m3,
                b.partials,
                b.hidden,
                b.inter,
                b.nc,
                begin,
                end,
            )

        b.k.reduce(b.partials, b.route_down, b.hidden, b.nc)
        main.wait_event(self.shared_done[i])
        cp.copyto(out, self.shared_out)
        b.k.accumulate(out, b.route_down, b.w, b.hidden)

        stats = None
        if collect_stats:
            stats = {
                "layer": i,
                "phase31_mode": "device_mirror_group_down",
                "ranges": [list(x) for x in self.ranges],
                "arithmetic_order": (
                    "parent_panel_order_and_column_fmas_per_route_then_parent_reduce"
                ),
            }
        return None, None, stats


class Phase31RouterDirectMoEH4(Phase30EMoEH4):
    """Phase30E whose four sequential router GEMVs are one direct-L2 M4."""

    def __init__(self, base, dense_direct):
        super().__init__(base, shared_direct=True, group_dispatch=True)
        self.dense_direct = dense_direct
        self.parent_mv_f32 = self.rt.k.mv_f32
        self.gate_weight_ptrs = {
            int(self.rt.layer[int(layer)]["gate_w"].data.ptr)
            for layer in self.rt.moe_layers
        }
        self._router_normed = None
        self._router_calls = 0
        self.rt.k.mv_f32 = self._mv_f32

    def _mv_f32(self, out, weights, x, rows, cols):
        if int(weights.data.ptr) not in self.gate_weight_ptrs:
            return self.parent_mv_f32(out, weights, x, rows, cols)
        if self._router_normed is None:
            # Canonical prefill uses the single-token runtime after this
            # wrapper is installed; it must retain the production GEMV.
            return self.parent_mv_f32(out, weights, x, rows, cols)
        token = self._router_calls
        if token >= self.H:
            raise RuntimeError(f"unexpected router call {token}")
        expected = int(self.base.rlog[token].data.ptr)
        if int(out.data.ptr) != expected:
            raise RuntimeError(
                f"router output order drift token={token} "
                f"got={int(out.data.ptr)} expected={expected}"
            )
        if token == 0:
            self.dense_direct.f32(
                weights,
                self._router_normed,
                self.base.rlog,
                int(rows),
                int(cols),
            )
        self._router_calls += 1
        return None

    def __call__(self, layer, normed, out, collect_stats=False):
        self._router_normed = normed
        self._router_calls = 0
        result = super().__call__(layer, normed, out, collect_stats)
        if self._router_calls != self.H:
            raise RuntimeError(
                f"expected {self.H} router calls, got {self._router_calls}"
            )
        self._router_normed = None
        return result


def _phase31_body(graph):
    rt = graph.rt
    v = graph.v
    graph.gk.embed4(graph.embed_ptr, graph.tok_dev, v.h, v.hidden)
    for i, kind in enumerate(rt.pattern):
        d = rt.layer[i]
        v._norm_rows(d["norm"], v.h, v.normed)
        if kind == "M":
            v._mamba(i, v.normed, v.acc)
            for token in range(4):
                rt.k.add_(v.h[token], v.acc[token], v.hidden)
        elif kind == "*":
            if graph.config.attention_m4:
                graph._attention_block(i, v.normed, v.acc)
            else:
                for token in range(4):
                    graph._attention_row(i, v.normed[token], v.acc[token], token)
            for token in range(4):
                rt.k.add_(v.h[token], v.acc[token], v.hidden)
        else:
            # MoE writes its complete shared+routed contribution directly into
            # the residual; no v.acc materialization or four add_ launches.
            graph.gmoe.forward_residual(i, v.normed, v.h, False)

    v._norm_rows(rt.norm_f, v.h, v.final_normed)
    graph._head()
    graph.gk.argmax4(
        v.logits, rt.vocab, graph.am_max, graph.am_idx, graph.ids_dev, graph.nparts
    )
    graph.gk.add4pos(graph.pos_dev)


def _phase31_attention_direct(graph, layer, normed, out):
    rt = graph.rt
    d = rt.layer[int(layer)]
    direct = graph.phase31_dense_direct
    direct.bf16(
        d["q_proj"],
        normed,
        graph.q4,
        rt.n_heads * rt.head_dim,
        rt.hidden,
    )
    direct.bf16(d["k_proj"], normed, graph.k4, rt.kv_dim, rt.hidden)
    direct.bf16(d["v_proj"], normed, graph.v4, rt.kv_dim, rt.hidden)
    for token in range(4):
        graph.gk.kv_write(
            rt.kc[int(layer)],
            graph.k4[token],
            graph.pos_dev,
            token,
            rt.n_kv,
            rt.head_dim,
            rt.max_ctx,
        )
        graph.gk.kv_write(
            rt.vc[int(layer)],
            graph.v4[token],
            graph.pos_dev,
            token,
            rt.n_kv,
            rt.head_dim,
            rt.max_ctx,
        )
        graph.gk.attention(
            graph.ctx4[token],
            graph.q4[token],
            rt.kc[int(layer)],
            rt.vc[int(layer)],
            graph.pos_dev,
            token,
            rt.n_heads,
            rt.head_dim,
            rt.groups,
            rt.max_ctx,
            1.0 / math.sqrt(float(rt.head_dim)),
            graph.part_acc,
            graph.part_ml,
        )
    direct.bf16(
        d["o_proj"],
        graph.ctx4,
        out,
        rt.hidden,
        rt.n_heads * rt.head_dim,
    )


def _phase31_head_direct(graph):
    rt = graph.rt
    v = graph.v
    kernels = graph.gmoe.shared_m4
    mode = graph.phase31_head_direct_mode
    if mode == "m4":
        kernels.nvfp4(
            rt.lm_head_codes,
            rt.lm_head_scales,
            rt.fused.e2m1,
            rt.fused.e4m3,
            v.final_normed,
            v.logits,
            rt.lm_head_g,
            rt.vocab,
            rt.hidden,
            4,
            False,
            False,
        )
        return
    if mode == "m2":
        for begin in (0, 2):
            kernels.nvfp4(
                rt.lm_head_codes,
                rt.lm_head_scales,
                rt.fused.e2m1,
                rt.fused.e4m3,
                v.final_normed[begin : begin + 2],
                v.logits[begin : begin + 2],
                rt.lm_head_g,
                rt.vocab,
                rt.hidden,
                2,
                False,
                False,
            )
        return
    raise ValueError(mode)


def make_candidate(context: int, *, mode: str = "sink"):
    cfg, _, _ = phase27_gate()
    rt, graph, keep = make_synth(int(context), cfg)
    wrapped = Phase31MoEH4(graph.gmoe, mode=mode)
    graph.gmoe = wrapped
    graph.v.moeb = wrapped
    graph.body = types.MethodType(_phase31_body, graph)
    return rt, graph, list(keep) + [wrapped, wrapped.residual_kernels]


def make_staged_candidate(context: int):
    cfg, _, _ = phase27_gate()
    rt, graph, keep = make_synth(int(context), cfg)
    wrapped = Phase31StagedMoEH4(graph.gmoe)
    graph.gmoe = wrapped
    graph.v.moeb = wrapped
    return rt, graph, list(keep) + [
        wrapped,
        wrapped.shared_m4,
        wrapped.group_dispatch,
        wrapped.staged_kernels,
    ]


def make_group_down_candidate(context: int):
    cfg, _, _ = phase27_gate()
    rt, graph, keep = make_synth(int(context), cfg)
    wrapped = Phase31GroupDownMoEH4(graph.gmoe)
    graph.gmoe = wrapped
    graph.v.moeb = wrapped
    return rt, graph, list(keep) + [
        wrapped,
        wrapped.shared_m4,
        wrapped.group_dispatch,
        wrapped.group_down,
    ]


def make_attention_direct_candidate(context: int):
    parent_cfg, _, _ = phase27_gate()
    cfg = SynthesisConfig(
        attention_m4=True,
        router_m4=parent_cfg.router_m4,
        shared_m4=parent_cfg.shared_m4,
        sres_layers=parent_cfg.sres_layers,
    )
    rt, graph, keep = make_synth(int(context), cfg)
    wrapped = Phase30EMoEH4(
        graph.gmoe,
        shared_direct=True,
        group_dispatch=True,
    )
    direct = Phase31DenseDirectKernels()
    graph.gmoe = wrapped
    graph.v.moeb = wrapped
    graph.phase31_dense_direct = direct
    graph._attention_block = types.MethodType(_phase31_attention_direct, graph)
    return rt, graph, list(keep) + [
        wrapped,
        wrapped.shared_m4,
        wrapped.group_dispatch,
        direct,
    ]


def make_dense_direct_candidate(context: int):
    parent_cfg, _, _ = phase27_gate()
    cfg = SynthesisConfig(
        attention_m4=True,
        router_m4=False,
        shared_m4=parent_cfg.shared_m4,
        sres_layers=parent_cfg.sres_layers,
    )
    rt, graph, keep = make_synth(int(context), cfg)
    direct = Phase31DenseDirectKernels()
    wrapped = Phase31RouterDirectMoEH4(graph.gmoe, direct)
    graph.gmoe = wrapped
    graph.v.moeb = wrapped
    graph.phase31_dense_direct = direct
    graph._attention_block = types.MethodType(_phase31_attention_direct, graph)
    return rt, graph, list(keep) + [
        wrapped,
        wrapped.shared_m4,
        wrapped.group_dispatch,
        direct,
    ]


def make_attention_head_direct_candidate(context: int, *, head_mode: str):
    if head_mode not in ("m4", "m2"):
        raise ValueError(head_mode)
    rt, graph, keep = make_attention_direct_candidate(int(context))
    graph.phase31_head_direct_mode = head_mode
    graph._head = types.MethodType(_phase31_head_direct, graph)
    return rt, graph, keep


def compile_audit() -> dict:
    head = SharedOccupancyKernels()
    for fn in head.f.values():
        fn.compile()
    return {
        "residual": Phase31ResidualKernels().resource_audit(),
        "staged": Phase31StagedKernels().resource_audit(),
        "group_down": Phase31GroupDownKernels().resource_audit(),
        "dense_direct": Phase31DenseDirectKernels().resource_audit(),
        "head_direct": head.resource_audit(),
    }
