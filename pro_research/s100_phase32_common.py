from __future__ import annotations

import math

import numpy as np

from common import REPO
from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE
from scale_resident_kernels import PLANE_BYTES
from s100_phase21_common import make_rt
from s100_phase25_common import GraphH8Verifier
from s100_phase25_h8_kernels import H, ROUTES, TOPK
from s100_phase30e_shared_kernels import SharedOccupancyKernels
from s100_phase31_common import make_attention_head_direct_candidate
from s100_phase31_dense_direct_kernels import Phase31DenseDirectKernels
from s100_phase32_dense_h8 import Phase32DenseM8Kernels


RESULTS = REPO / "pro_research" / "results" / "s100_phase32"
ARMS = ("dense_split4", "dense_m8")


class Phase32MoEH8:
    """Phase25 direct8_route with Phase31-style dense/shared weight reuse."""

    def __init__(self, base, dense_mode: str):
        import cupy as cp

        if base.up_mode != "direct8" or base.down_mode != "route":
            raise RuntimeError("Phase32 requires Phase25 direct8_route")
        if dense_mode not in ("split4", "m8"):
            raise ValueError(dense_mode)
        self.cp = cp
        self.base = base
        self.rt = base.rt
        self.dense_mode = dense_mode
        self.dense_m4 = Phase31DenseDirectKernels()
        self.dense_m8 = Phase32DenseM8Kernels()
        self.shared_m4 = SharedOccupancyKernels()
        self.shared_stream = cp.cuda.Stream(non_blocking=True)
        self.shared_out = cp.empty((H, base.hidden), cp.float32)
        self.fork_events = {int(i): cp.cuda.Event() for i in self.rt.moe_layers}
        self.done_events = {int(i): cp.cuda.Event() for i in self.rt.moe_layers}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _dense_f32(self, weights, x8, out8, rows: int, cols: int) -> None:
        if self.dense_mode == "m8":
            self.dense_m8.f32(weights, x8, out8, rows, cols)
            return
        for begin in (0, 4):
            self.dense_m4.f32(
                weights,
                x8[begin : begin + 4],
                out8[begin : begin + 4],
                rows,
                cols,
            )

    def _shared_fork(self, i, d, normed, main) -> None:
        b, rt = self.base, self.rt
        self.fork_events[i].record(main)
        with self.shared_stream:
            self.shared_stream.wait_event(self.fork_events[i])
            for begin in (0, 4):
                sl = slice(begin, begin + 4)
                self.shared_m4.nvfp4(
                    d["sh_up_c"], d["sh_up_s"],
                    rt.fused.e2m1, rt.fused.e4m3,
                    normed[sl], b.shared_act[sl], d["sh_up_g"],
                    b.shared, b.hidden, 4, False, True,
                )
                self.shared_m4.nvfp4(
                    d["sh_dn_c"], d["sh_dn_s"],
                    rt.fused.e2m1, rt.fused.e4m3,
                    b.shared_act[sl], self.shared_out[sl], d["sh_dn_g"],
                    b.hidden, b.shared, 4, False, False,
                )
            self.done_events[i].record(self.shared_stream)

    def __call__(self, layer, normed, out, collect_stats=False):
        cp, b, rt = self.cp, self.base, self.rt
        i = int(layer)
        d = rt.layer[i]
        bank, cache = rt.bank[i], rt.cache[i]
        dev = b._dev(i)
        main = cp.cuda.get_current_stream()

        self._shared_fork(i, d, normed, main)

        self._dense_f32(d["gate_w"], normed, b.rlog, b.nexp, b.hidden)
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
        b.sg.gather(
            int(bank["down_base_ptr"]), b.group_ids, b.group_count,
            b.union_nz, b.union_nzc, b.mirrors, b.hidden, b.inter,
        )
        b.k.down_route(
            b.mirrors, b.sres.planes[i], b.slots, b.ids,
            b.route_group, dev["globals"], b.route_act,
            b.route_plist, b.route_masks, b.route_pcount,
            rt.fused.e2m1, rt.fused.e4m3, b.partials,
            DOWN_PANEL_BYTES, PLANE_BYTES, b.hidden, b.inter, b.nc,
        )
        b.k.reduce(b.partials, b.route_down, b.hidden, b.nc)

        main.wait_event(self.done_events[i])
        cp.copyto(out, self.shared_out)
        b.k.accumulate(out, b.route_down, b.w, b.hidden)

        stats = None
        if collect_stats:
            stats = {
                "layer": i,
                "dense_mode": self.dense_mode,
                "shared_two_m4_waves": True,
                "arithmetic_order": "shared_then_slot0_to_slot5_fmaf",
            }
        return None, None, stats


class Phase32GraphH8(GraphH8Verifier):
    def __init__(self, rt, arm: str):
        import cupy as cp

        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(rt, "direct8_route", diagnostic=False)
        self.arm = arm
        self.dense_mode = "m8" if arm == "dense_m8" else "split4"
        self.phase32_dense_m4 = Phase31DenseDirectKernels()
        self.phase32_dense_m8 = Phase32DenseM8Kernels()
        self.phase32_head_m4 = SharedOccupancyKernels()
        self.gmoe = Phase32MoEH8(self.gmoe, self.dense_mode)
        self.q8 = cp.empty((H, rt.n_heads * rt.head_dim), cp.float32)
        self.k8 = cp.empty((H, rt.kv_dim), cp.float32)
        self.v8 = cp.empty((H, rt.kv_dim), cp.float32)
        self.ctx8 = cp.empty((H, rt.n_heads * rt.head_dim), cp.float32)

    def _dense_bf16(self, weights, x8, out8, rows: int, cols: int) -> None:
        if self.dense_mode == "m8":
            self.phase32_dense_m8.bf16(weights, x8, out8, rows, cols)
            return
        for begin in (0, 4):
            self.phase32_dense_m4.bf16(
                weights,
                x8[begin : begin + 4],
                out8[begin : begin + 4],
                rows,
                cols,
            )

    def _attention_block(self, layer, normed, out) -> None:
        rt = self.rt
        d = rt.layer[int(layer)]
        self._dense_bf16(
            d["q_proj"], normed, self.q8,
            rt.n_heads * rt.head_dim, rt.hidden,
        )
        self._dense_bf16(d["k_proj"], normed, self.k8, rt.kv_dim, rt.hidden)
        self._dense_bf16(d["v_proj"], normed, self.v8, rt.kv_dim, rt.hidden)
        for token in range(H):
            self.gk.kv_write(
                rt.kc[int(layer)], self.k8[token], self.pos_dev, token,
                rt.n_kv, rt.head_dim, rt.max_ctx,
            )
            self.gk.kv_write(
                rt.vc[int(layer)], self.v8[token], self.pos_dev, token,
                rt.n_kv, rt.head_dim, rt.max_ctx,
            )
            self.gk.attention(
                self.ctx8[token], self.q8[token],
                rt.kc[int(layer)], rt.vc[int(layer)], self.pos_dev, token,
                rt.n_heads, rt.head_dim, rt.groups, rt.max_ctx,
                1.0 / math.sqrt(float(rt.head_dim)),
                self.part_acc, self.part_ml,
            )
        self._dense_bf16(
            d["o_proj"], self.ctx8, out,
            rt.hidden, rt.n_heads * rt.head_dim,
        )

    def _head(self) -> None:
        rt, core = self.rt, self.core
        for begin in (0, 4):
            sl = slice(begin, begin + 4)
            self.phase32_head_m4.nvfp4(
                rt.lm_head_codes, rt.lm_head_scales,
                rt.fused.e2m1, rt.fused.e4m3,
                core.final_normed[sl], core.logits[sl], rt.lm_head_g,
                rt.vocab, rt.hidden, 4, False, False,
            )

    def body(self):
        rt, core = self.rt, self.core
        self.gk.embed4(self.embed_ptr, self.tok_dev[:4], core.h[:4], core.hidden)
        self.gk.embed4(self.embed_ptr, self.tok_dev[4:], core.h[4:], core.hidden)
        for i, kind in enumerate(rt.pattern):
            d = rt.layer[i]
            core.norm_rows(d["norm"], core.h, core.normed)
            if kind == "M":
                core.mamba(i, core.normed, core.acc)
            elif kind == "*":
                self._attention_block(i, core.normed, core.acc)
            else:
                self.gmoe(i, core.normed, core.acc, False)
            for token in range(H):
                rt.k.add_(core.h[token], core.acc[token], core.hidden)
        core.norm_rows(rt.norm_f, core.h, core.final_normed)
        self._head()
        self.gk.argmax4(
            core.logits[:4], rt.vocab,
            self.am_max[: 4 * self.nparts], self.am_idx[: 4 * self.nparts],
            self.ids_dev[:4], self.nparts,
        )
        self.gk.argmax4(
            core.logits[4:], rt.vocab,
            self.am_max[4 * self.nparts :], self.am_idx[4 * self.nparts :],
            self.ids_dev[4:], self.nparts,
        )
        self.gk.add4pos(self.pos_dev)
        self.gk.add4pos(self.pos_dev)

    def setup_graph(self):
        info = super().setup_graph()
        info["phase32_arm"] = self.arm
        info["dense_mode"] = self.dense_mode
        info["shared_two_m4_waves"] = True
        info["head_two_m4_waves"] = True
        return info


def make_candidate(context: int, arm: str):
    if arm not in ARMS:
        raise ValueError(arm)
    rt, keep = make_rt(int(context), "v6_device_rows")
    graph = Phase32GraphH8(rt, arm)
    return rt, graph, list(keep) + [graph.gmoe]


def make_parent(context: int):
    return make_attention_head_direct_candidate(int(context), head_mode="m4")


def compile_audit() -> dict:
    m4 = Phase31DenseDirectKernels()
    m8 = Phase32DenseM8Kernels()
    shared = SharedOccupancyKernels()
    for fn in tuple(m4.f.values()) + tuple(m8.f.values()) + tuple(shared.f.values()):
        fn.compile()
    return {
        "dense_m4": m4.resource_audit(),
        "dense_m8": m8.resource_audit(),
        "shared_head_m4": shared.resource_audit(),
    }
