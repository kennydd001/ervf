"""Composes the two batch>1 MoE mechanisms that were so far only proven
SEPARATELY -- up_proj shared-fetch (proto_batch_moe_layer.py) and down_proj
union-of-masks shared-fetch (proto_batch_down_proj.py) -- into ONE combined
pipeline for N sequences on one real MoE layer, mirroring how V4/V6 combined
separately-proven batch=1 mechanisms into one integrated stack.

Why this isn't just "the two numbers multiplied together": proto_batch_down_
proj.py computed its per-sequence activations by naively RE-FETCHING each
expert's up_proj weights per (sequence, expert) pair -- it deliberately held
up_proj sharing out of scope to isolate the down_proj mechanism. Here the
up_proj stage ALSO shares its fetch (one cache_fetch per union expert, same
as proto_batch_moe_layer.py), and the resulting per-sequence activations feed
directly into the down_proj union-mask stage. This is the first end-to-end
"what would one shared-cache MoE layer forward pass actually look like"
measurement, not two isolated component numbers.

Not a gated PRO experiment -- a scoped feasibility test, explicitly not a
production integration (no batch dimension in the runtime itself).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic
from down_gather_batch_kernels import DownGatherBatchKernels
from down_proj_batch_kernels import DownProjBatchKernels

PROMPTS = [
    "The history of computing began when",
    "Write a correct Python function that computes the longest increasing subsequence length in O(n log n), then explain its invariant.\n",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "In the quiet village, the old fisherman noticed something strange about the tide",
    "The quarterly earnings report showed a significant increase in revenue driven by",
    "Photosynthesis is the process by which plants convert light energy into",
    "The defendant's attorney argued that the evidence presented by the prosecution was",
    "To configure the network firewall, first navigate to the settings panel and",
]

UP_CODE = 2_494_464
UP_SCALE = 311_808
DOWN_PANEL_BYTES = UP_CODE + UP_SCALE
N = 8


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    gk = DownGatherBatchKernels()
    scan_k = DownProjBatchKernels()

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    fused = rt.fused

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    target_layer = moe_layers[10]

    captured = []
    orig_route = rt._route_device

    def capture_route(i):
        packed = orig_route(i)
        if i == target_layer:
            captured.append({"normed": cp.asarray(rt.normed).copy(), "packed": cp.asarray(packed).copy()})
        return packed

    rt._route_device = types.MethodType(lambda self, i: capture_route(i), rt)
    for prompt in PROMPTS[:N]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)
    rt._route_device = orig_route
    cp.cuda.Device(0).synchronize()

    captured = captured[-N:]
    top_k = rt.top_k
    hidden = rt.hidden
    moe_inter = rt.moe_inter
    npanel = moe_inter // 16
    bank = rt.bank[target_layer]

    seq_ids = [[int(x) for x in cp.asnumpy(c["packed"])[:top_k]] for c in captured]
    seq_normed = [c["normed"] for c in captured]
    union_experts = sorted(set(e for ids_s in seq_ids for e in ids_s))
    pairs = [(s, e) for s in range(N) for e in seq_ids[s]]

    blocks = ((moe_inter + npanel) * 32 + 255) // 256

    def panel_scan(act):
        masks = cp.zeros(npanel, dtype=cp.uint32)
        plist = cp.zeros(npanel, dtype=cp.int32)
        pcount = cp.zeros(1, dtype=cp.int32)
        nz = cp.zeros(moe_inter, dtype=cp.int32)
        nzc = cp.zeros(1, dtype=cp.int32)
        scan_k.panel_scan_ref((1,), (256,), (act, np.int32(moe_inter), masks, plist, pcount, nz, nzc))
        return {"masks": masks, "plist": plist, "pcount": pcount, "nz": nz, "nzc": nzc}

    def down_pipeline(mirror, e, act, p):
        partials = cp.zeros(fused.nchunks * hidden, dtype=cp.float32)
        id_dev = cp.asarray([e], dtype=cp.int32)
        gk.run_down_masked_ref(mirror, id_dev, cp.asarray(bank["globals"]),
                               act, p["plist"], p["masks"], p["pcount"], fused.e2m1, fused.e4m3,
                               partials, hidden, moe_inter, fused.nchunks)
        return scan_k.run_reduce_partials_ref(partials, hidden, fused.nchunks)

    # ================= NAIVE: fully independent per (sequence, expert) pair =================
    # up_proj fetch uses the SAME production fetch kernel (fused.cache_fetch,
    # a pinned-host -> device PCIe transfer) as the batched arm below, into a
    # freshly re-fetched per-pair scratch buffer -- so naive vs batched differ
    # only in whether that fetch is shared, not in which code path is timed.
    naive_scratch_c = cp.zeros(UP_CODE, dtype=cp.uint8)
    naive_scratch_s = cp.zeros(UP_SCALE, dtype=cp.uint8)
    ids_dev1 = cp.zeros(1, dtype=cp.int32)
    slots_dev1 = cp.zeros(1, dtype=cp.int32)
    need_dev1 = cp.ones(1, dtype=cp.int32)

    naive_up_ms = 0.0
    naive_down_ms = 0.0
    naive_outputs = {}
    for (s, e) in pairs:
        ids_dev1[0] = e
        f0, f1 = cp.cuda.Event(), cp.cuda.Event()
        f0.record()
        fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                          naive_scratch_c, naive_scratch_s,
                          {"ids": ids_dev1, "slots": slots_dev1, "need": need_dev1},
                          UP_CODE, UP_SCALE, 1)
        act = cp.zeros(moe_inter, dtype=cp.float32)
        fused.gemv_into(act, naive_scratch_c, naive_scratch_s, seq_normed[s], float(bank["globals"][e, 1]), moe_inter, hidden, apply_relu2=True)
        f1.record(); f1.synchronize()
        naive_up_ms += cp.cuda.get_elapsed_time(f0, f1)

        p = panel_scan(act)
        mirror = cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)
        id_dev = cp.asarray([e], dtype=cp.int32)
        d0, d1 = cp.cuda.Event(), cp.cuda.Event()
        d0.record()
        gk.run_gather_ref(np.uint64(bank["down_base_ptr"]), id_dev, DOWN_PANEL_BYTES, mirror,
                          p["plist"], p["pcount"], p["nz"], p["nzc"], hidden, blocks)
        out = down_pipeline(mirror, e, act, p)
        d1.record(); d1.synchronize()
        naive_down_ms += cp.cuda.get_elapsed_time(d0, d1)
        naive_outputs[(s, e)] = cp.asnumpy(out)

    # ================= BATCHED: shared up_proj fetch, shared down_proj union-mask fetch =================
    u = len(union_experts)
    expert_to_slot = {e: i for i, e in enumerate(union_experts)}
    batched_c = cp.zeros(u * UP_CODE, dtype=cp.uint8)
    batched_s = cp.zeros(u * UP_SCALE, dtype=cp.uint8)
    ids_dev_b = cp.asarray(union_experts, dtype=cp.int32)
    slots_dev_b = cp.arange(u, dtype=cp.int32)
    need_dev_b = cp.ones(u, dtype=cp.int32)

    up0, up1 = cp.cuda.Event(), cp.cuda.Event()
    up0.record()
    fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                      batched_c, batched_s,
                      {"ids": ids_dev_b, "slots": slots_dev_b, "need": need_dev_b},
                      UP_CODE, UP_SCALE, u)
    up1.record(); up1.synchronize()
    batched_up_fetch_ms = cp.cuda.get_elapsed_time(up0, up1)

    act_by_pair = {}
    panel_by_pair = {}
    batched_up_gemv_ms = 0.0
    for (s, e) in pairs:
        slot = expert_to_slot[e]
        c_slice = batched_c[slot * UP_CODE:(slot + 1) * UP_CODE]
        s_slice = batched_s[slot * UP_SCALE:(slot + 1) * UP_SCALE]
        g0, g1 = cp.cuda.Event(), cp.cuda.Event()
        g0.record()
        act = cp.zeros(moe_inter, dtype=cp.float32)
        fused.gemv_into(act, c_slice, s_slice, seq_normed[s], float(bank["globals"][e, 1]), moe_inter, hidden, apply_relu2=True)
        g1.record(); g1.synchronize()
        batched_up_gemv_ms += cp.cuda.get_elapsed_time(g0, g1)
        act_by_pair[(s, e)] = act
        panel_by_pair[(s, e)] = panel_scan(act)

    # union mask per expert (OR of masks across sequences selecting it).
    union_mask_by_expert = {}
    union_plist_by_expert = {}
    for e in union_experts:
        acc_mask = np.zeros(npanel, dtype=np.uint32)
        for s in range(N):
            if e in seq_ids[s]:
                acc_mask |= cp.asnumpy(panel_by_pair[(s, e)]["masks"])
        plist_np = np.array([p for p in range(npanel) if acc_mask[p]], dtype=np.int32)
        union_mask_by_expert[e] = cp.asarray(acc_mask)
        union_plist_by_expert[e] = cp.asarray(plist_np)

    batched_down_fetch_ms = 0.0
    batched_down_gemv_ms = 0.0
    batched_outputs = {}
    for e in union_experts:
        union_mask = union_mask_by_expert[e]
        union_plist = union_plist_by_expert[e]
        pcount_u = cp.asarray([int((union_mask.get() != 0).sum())], dtype=cp.int32)
        nz_list = []
        for p in range(npanel):
            m = int(cp.asnumpy(union_mask)[p])
            for c in range(16):
                if m & (1 << c):
                    nz_list.append((p << 4) + c)
        nz_u = cp.asarray(np.array(nz_list, dtype=np.int32))
        nzc_u = cp.asarray([len(nz_list)], dtype=cp.int32)
        nz_pad = cp.zeros(moe_inter, dtype=cp.int32)
        nz_pad[:len(nz_list)] = nz_u

        mirror = cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)
        id_dev = cp.asarray([e], dtype=cp.int32)
        df0, df1 = cp.cuda.Event(), cp.cuda.Event()
        df0.record()
        gk.run_gather_ref(np.uint64(bank["down_base_ptr"]), id_dev, DOWN_PANEL_BYTES, mirror,
                          union_plist, pcount_u, nz_pad, nzc_u, hidden, blocks)
        df1.record(); df1.synchronize()
        batched_down_fetch_ms += cp.cuda.get_elapsed_time(df0, df1)

        for s in range(N):
            if e not in seq_ids[s]:
                continue
            key = (s, e)
            p = panel_by_pair[key]
            dg0, dg1 = cp.cuda.Event(), cp.cuda.Event()
            dg0.record()
            out = down_pipeline(mirror, e, act_by_pair[key], p)
            dg1.record(); dg1.synchronize()
            batched_down_gemv_ms += cp.cuda.get_elapsed_time(dg0, dg1)
            batched_outputs[key] = cp.asnumpy(out)

    mismatches = sum(1 for k in naive_outputs if not (naive_outputs[k] == batched_outputs[k]).all())

    naive_total_ms = naive_up_ms + naive_down_ms
    batched_total_ms = batched_up_fetch_ms + batched_up_gemv_ms + batched_down_fetch_ms + batched_down_gemv_ms

    payload = {
        "kind": "proto_batch_moe_layer_combined",
        "created_utc": utc_now(),
        "note": "composes up_proj shared-fetch and down_proj union-mask shared-fetch into ONE pipeline for N sequences on one real MoE layer, previously only measured separately; not a production integration",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "target_layer": target_layer,
        "n_sequences": N,
        "top_k": top_k,
        "pair_count": len(pairs),
        "union_expert_count": u,
        "correctness_mismatches": mismatches,
        "correctness_pass": mismatches == 0,
        "naive_up_ms": naive_up_ms,
        "naive_down_ms": naive_down_ms,
        "naive_total_ms": naive_total_ms,
        "batched_up_fetch_ms": batched_up_fetch_ms,
        "batched_up_gemv_ms": batched_up_gemv_ms,
        "batched_down_fetch_ms": batched_down_fetch_ms,
        "batched_down_gemv_ms": batched_down_gemv_ms,
        "batched_total_ms": batched_total_ms,
        "combined_speedup": (naive_total_ms / batched_total_ms) if batched_total_ms else None,
        "ms_saved": naive_total_ms - batched_total_ms,
    }
    out = REPO / "pro_research" / "proto_batch_moe_layer_combined.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
