from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import REPO
from moe_dev_batched import UP_CODE, UP_SCALE, DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES
from s100_phase21_common import (
    identity_gate, load_trace, prefill_to, expected_for_block, release,
)
from s100_phase24_common import (
    selected_config, make_synth, timed_synth_blocks,
)
from s100_phase25_common import (
    make_h8, timed_h8_blocks, timed_parent_h8_windows,
    expected_for_h8, summarize_h8, phase24_gate,
)

RESULTS = REPO / "pro_research" / "results" / "s100_phase26"
SNAPSHOT = "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def phase26_gate():
    cfg, p24, th24, st24 = phase24_gate()
    p25 = load_json(
        REPO/"pro_research"/"results"/"s100_phase25"/"S100_PHASE25_SUMMARY.json"
    )
    if not p25.get("instrumentation_complete"):
        raise RuntimeError("Phase25 summary incomplete")
    h8_adopted=bool(
        (p25.get("gates") or {}).get("H8_ADOPTED")
        or p25.get("H8_ACTIVE_PARENT")
    )
    if h8_adopted:
        raise RuntimeError("Phase26 expects Phase25 H8 to remain unadopted")
    if p25.get("NEXT_ROUTE") != "PROFILE_H8_ECONOMICS_AND_REDUCE_DOMINANT_STAGE":
        raise RuntimeError(f"unexpected Phase25 route: {p25.get('NEXT_ROUTE')}")
    if cfg.attention_m4 or cfg.router_m4 or cfg.shared_m4:
        raise RuntimeError("Phase24 closed dense M4 components must remain disabled")
    if len(cfg.sres_layers) != 23:
        raise RuntimeError("Phase26 requires all 23 Phase24 H-SCALE planes")
    return cfg, p24, p25


class SharedRoutedOverlapH4:
    """Scheduling-only wrapper around the frozen Phase24 H4 MoE object.

    The shared branch is launched on a side stream. Routed vectors are fully
    computed on the main stream. At the join, shared_out is copied to the
    output and the original route-slot FMA accumulation kernel is invoked.
    This preserves the parent floating-point association exactly.
    """
    H = 4
    TOPK = 6

    def __init__(self, base):
        import cupy as cp
        self.cp = cp
        self.base = base
        self.rt = base.rt
        self.shared_stream = cp.cuda.Stream(non_blocking=True)
        self.shared_out = cp.empty((self.H, base.hidden), cp.float32)
        self.fork_events = {
            int(i): cp.cuda.Event() for i in self.rt.moe_layers
        }
        self.done_events = {
            int(i): cp.cuda.Event() for i in self.rt.moe_layers
        }

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, layer, normed, out, collect_stats=False):
        cp = self.cp
        b = self.base
        rt = self.rt
        i = int(layer)
        d = rt.layer[i]
        bank, c = rt.bank[i], rt.cache[i]
        dev = b._dev(i)
        main = cp.cuda.get_current_stream()

        # Fork after the layer norm has produced `normed` on the main stream.
        self.fork_events[i].record(main)
        with self.shared_stream:
            self.shared_stream.wait_event(self.fork_events[i])
            for t in range(self.H):
                rt.fused.gemv_into(
                    b.shared_act[t], d["sh_up_c"], d["sh_up_s"],
                    normed[t], d["sh_up_g"], b.shared, b.hidden,
                    apply_relu2=True,
                )
                rt.fused.gemv_into(
                    self.shared_out[t], d["sh_dn_c"], d["sh_dn_s"],
                    b.shared_act[t], d["sh_dn_g"], b.hidden, b.shared,
                )
            self.done_events[i].record(self.shared_stream)

        # Routed branch: identical Phase24 order, except final accumulate is
        # delayed until the shared branch joins.
        for t in range(self.H):
            a=t*self.TOPK
            z=a+self.TOPK
            rt.k.mv_f32(
                b.rlog[t], d["gate_w"], normed[t], b.nexp, b.hidden
            )
            rt.fused.route_topk(
                b.rlog[t], d["gate_b"], b.ids[a:z], b.w[a:z],
                b.nexp, self.TOPK, rt.scaling, bad_pick=rt._bad_pick,
            )

        b.k.cache_assign(dev, b.ids, b.slots, b.need, int(c["cap"]))
        b.k.group(
            b.ids, b.route_group, b.group_ids, b.group_count,
            b.group_refs, b.ngroups,
        )

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
                (24,64),(256,),
                (
                    np.uint64(bank["up_codes"].ctypes.data),
                    np.uint64(bank["up_scales"].ctypes.data),
                    c["codes"], c["scales"], b.ids, b.slots, b.need,
                    np.uint64(UP_CODE), np.uint64(UP_SCALE),
                ),
            )
            if i not in b.sres.planes:
                raise RuntimeError(f"Phase26 H4 expected resident scale plane on layer {i}")
            b.sres.fetch_plane_k(
                (24,64),(256,),
                (
                    np.uint64(bank["down_base_ptr"]), b.sres.planes[i],
                    b.ids, b.slots, b.need,
                    np.uint64(DOWN_PANEL_BYTES), np.uint64(PLANE_BYTES),
                    np.int32(b.hidden), np.int32(b.npanel),
                ),
            )
            rt.evt[1].record(rt.copy_stream)

        main.wait_event(rt.evt[1])

        for m in (1,2,3,4):
            b.k.up(
                m, c["codes"], c["scales"], b.slots, b.ids,
                dev["globals"], rt.fused.e2m1, rt.fused.e4m3,
                normed, b.route_act, b.inter, b.hidden,
                UP_CODE, UP_SCALE,
            )

        b.k.scan(
            b.route_act, b.group_count, b.group_refs,
            b.route_masks, b.route_plist, b.route_pcount,
            b.union_masks, b.union_plist, b.union_pcount,
            b.union_nz, b.union_nzc, b.inter,
        )

        b.sgres.gather(
            int(bank["down_base_ptr"]), b.group_ids, b.group_count,
            b.union_nz, b.union_nzc, b.mirrors, b.hidden, b.inter,
        )

        b.sgres.down(
            b.mirrors, b.sres.planes[i], b.slots, b.ids,
            b.route_group, dev["globals"], b.route_act,
            b.route_plist, b.route_masks, b.route_pcount,
            rt.fused.e2m1, rt.fused.e4m3, b.partials,
            b.hidden, b.inter, b.nc,
        )
        b.k.reduce(b.partials, b.route_down, b.hidden, b.nc)

        # Join and preserve parent arithmetic order.
        main.wait_event(self.done_events[i])
        cp.copyto(out, self.shared_out)
        b.k.accumulate(out, b.route_down, b.w, b.hidden)

        stats = None
        if collect_stats:
            stats = {
                "layer": i,
                "overlap": True,
                "arithmetic_order": "shared_then_slot0_to_slot5_fmaf",
            }
        return None, None, stats


class SharedRoutedOverlapH8:
    """Scheduling-only wrapper around Phase25 direct8_route."""
    H = 8
    TOPK = 6
    ROUTES = 48

    def __init__(self, base):
        import cupy as cp
        if base.up_mode != "direct8" or base.down_mode != "route":
            raise RuntimeError(
                f"Phase26 H8 requires direct8_route, got "
                f"{base.up_mode}/{base.down_mode}"
            )
        self.cp = cp
        self.base = base
        self.rt = base.rt
        self.shared_stream = cp.cuda.Stream(non_blocking=True)
        self.shared_out = cp.empty((self.H, base.hidden), cp.float32)
        self.fork_events = {
            int(i): cp.cuda.Event() for i in self.rt.moe_layers
        }
        self.done_events = {
            int(i): cp.cuda.Event() for i in self.rt.moe_layers
        }

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, layer, normed, out, collect_stats=False):
        cp = self.cp
        b = self.base
        rt = self.rt
        i = int(layer)
        d = rt.layer[i]
        bank, c = rt.bank[i], rt.cache[i]
        dev = b._dev(i)
        main = cp.cuda.get_current_stream()

        self.fork_events[i].record(main)
        with self.shared_stream:
            self.shared_stream.wait_event(self.fork_events[i])
            for t in range(self.H):
                rt.fused.gemv_into(
                    b.shared_act[t], d["sh_up_c"], d["sh_up_s"],
                    normed[t], d["sh_up_g"], b.shared, b.hidden,
                    apply_relu2=True,
                )
                rt.fused.gemv_into(
                    self.shared_out[t], d["sh_dn_c"], d["sh_dn_s"],
                    b.shared_act[t], d["sh_dn_g"], b.hidden, b.shared,
                )
            self.done_events[i].record(self.shared_stream)

        for t in range(self.H):
            a=t*self.TOPK
            z=a+self.TOPK
            rt.k.mv_f32(
                b.rlog[t], d["gate_w"], normed[t], b.nexp, b.hidden
            )
            rt.fused.route_topk(
                b.rlog[t], d["gate_b"], b.ids[a:z], b.w[a:z],
                b.nexp, self.TOPK, rt.scaling, bad_pick=rt._bad_pick,
            )

        b.k.cache_assign(dev, b.ids, b.slots, b.need, int(c["cap"]))
        b.k.group(
            b.ids, b.route_group, b.group_ids, b.group_count,
            b.group_refs, b.ngroups,
        )

        rt.evt[0].record()
        with rt.copy_stream:
            rt.copy_stream.wait_event(rt.evt[0])
            rt.fused.cache_fetch_k(
                (self.ROUTES,64),(256,),
                (
                    np.uint64(bank["up_codes"].ctypes.data),
                    np.uint64(bank["up_scales"].ctypes.data),
                    c["codes"], c["scales"], b.ids, b.slots, b.need,
                    np.uint64(UP_CODE), np.uint64(UP_SCALE),
                ),
            )
            b.sres.fetch_plane_k(
                (self.ROUTES,64),(256,),
                (
                    np.uint64(bank["down_base_ptr"]), b.sres.planes[i],
                    b.ids, b.slots, b.need,
                    np.uint64(DOWN_PANEL_BYTES), np.uint64(PLANE_BYTES),
                    np.int32(b.hidden), np.int32(b.npanel),
                ),
            )
            rt.evt[1].record(rt.copy_stream)

        main.wait_event(rt.evt[1])

        for m in range(1,9):
            b.k.up_direct(
                m, c["codes"], c["scales"], b.slots, b.ids,
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
                "overlap": True,
                "arithmetic_order": "shared_then_slot0_to_slot5_fmaf",
            }
        return None, None, stats


def make_h4_overlap(context: int):
    cfg, _, _ = phase26_gate()
    rt, g, keep = make_synth(int(context), cfg)
    wrapped = SharedRoutedOverlapH4(g.gmoe)
    g.gmoe = wrapped
    g.v.moeb = wrapped
    keep = list(keep) + [wrapped]
    return rt, g, keep


def make_h8_overlap(context: int):
    phase26_gate()
    rt, g, keep = make_h8(int(context), "direct8_route")
    wrapped = SharedRoutedOverlapH8(g.gmoe)
    g.gmoe = wrapped
    keep = list(keep) + [wrapped]
    return rt, g, keep


def timed_h4_overlap(rt, g, tokens, context, blocks, warmup):
    return timed_synth_blocks(rt, g, tokens, context, blocks, warmup)


def timed_h8_overlap(rt, g, tokens, context, blocks, warmup):
    return timed_h8_blocks(rt, g, tokens, context, blocks, warmup)


def capture_arrays(rt, logits_tail, pos, ids, ids_repeat=None):
    import cupy as cp
    arrays = {
        "ids": np.asarray(ids, np.int32),
        "ids_repeat": np.asarray(
            ids if ids_repeat is None else ids_repeat, np.int32
        ),
        "logits_tail": cp.asnumpy(logits_tail).astype(np.float32, copy=True),
    }
    for k, x in rt.ssm.items():
        arrays[f"ssm_{int(k)}"] = cp.asnumpy(x).astype(np.float32, copy=True)
    for k, x in rt.conv.items():
        arrays[f"conv_{int(k)}"] = cp.asnumpy(x).astype(np.float32, copy=True)
    nk, mc, hd = int(rt.n_kv), int(rt.max_ctx), int(rt.head_dim)
    for li in rt.attn_layers:
        i = int(li)
        arrays[f"k_{i}"] = cp.asnumpy(
            rt.kc[i].reshape(nk,mc,hd)[:,:pos,:]
        ).astype(np.float32, copy=True)
        arrays[f"v_{i}"] = cp.asnumpy(
            rt.vc[i].reshape(nk,mc,hd)[:,:pos,:]
        ).astype(np.float32, copy=True)
    return arrays


def nrmse(a, b):
    aa=np.asarray(a,np.float64)
    bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(aa),1e-30))


def compare_npz(parent_path, cand_path):
    with np.load(parent_path) as p, np.load(cand_path) as c:
        keys=sorted(set(p.files)&set(c.files))
        ssm=max(nrmse(p[k],c[k]) for k in keys if k.startswith("ssm_"))
        conv=max(nrmse(p[k],c[k]) for k in keys if k.startswith("conv_"))
        kv=max(
            [nrmse(p[k],c[k]) for k in keys
             if k.startswith("k_") or k.startswith("v_")] or [0.0]
        )
        logits=nrmse(p["logits_tail"],c["logits_tail"])
        ids=bool(np.array_equal(p["ids"],c["ids"]))
        det=bool(np.array_equal(c["ids"],c["ids_repeat"]))
        finite=all(np.isfinite(c[k]).all() for k in keys)
    state={
        "max_ssm_nrmse":ssm,
        "max_conv_nrmse":conv,
        "max_kv_nrmse":kv,
        "logits_nrmse":logits,
    }
    gates={
        "ids_exact":ids,
        "candidate_deterministic_ids":det,
        "ssm":ssm<=5e-5,
        "conv":conv<=1e-5,
        "kv":kv<=5e-6,
        "logits":logits<=5e-4,
        "finite":bool(finite),
    }
    return state, gates


def robust_cv(vals):
    a=np.asarray(vals,np.float64)
    m=float(np.median(a))
    return float(1.4826*np.median(np.abs(a-m))/max(abs(m),1e-30))
