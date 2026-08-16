"""Extends the batch>1 MoE prototype (proto_batch_moe_layer.py,
proto_batch_moe_multilayer.py) to down_proj -- architecturally different
from up_proj and not a trivial copy of that design.

up_proj sharing is simple: the whole weight matrix is fetched regardless of
activation, so N sequences selecting the same expert can literally share one
fetch. down_proj is masked/sparse -- gather_down_sparse_ind only pulls the
nonzero-activation columns (per fused_nvfp4.py's own S5 design, ~30-70%
sparsity from ReLU2), so two sequences selecting the SAME expert can still
need DIFFERENT columns if their activations differ. Naively "sharing" the
down_proj fetch the same way as up_proj would be wrong (a sequence might read
columns nobody gathered for it) or would silently force full dense fetches
(losing the sparsity benefit entirely).

The correct generalization: for an expert selected by multiple sequences,
gather the UNION of nonzero columns across those sequences (superset, so
every individual sequence's own needed columns are guaranteed present), then
run each sequence's masked-GEMV using ITS OWN panel_masks/panel_list (not the
union) against that shared mirror -- the per-sequence sum only touches that
sequence's own columns, which are a subset of what got gathered, so this
changes only which bytes crossed PCIe, never the computed values.

Real captured activations are used throughout (this session already learned
the hard way that synthetic random test data can trigger unrepresentative
edge cases) -- real up_proj GEMV is run first to get the real ReLU2-sparse
post-activation for each (sequence, expert) pair, using the already-verified
production ERVF kernel.

Not a gated PRO experiment -- a scoped feasibility test.
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

PROMPTS = [
    "The history of computing began when",
    "Write a correct Python function that computes the longest increasing subsequence length in O(n log n), then explain its invariant.\n",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "In the quiet village, the old fisherman noticed something strange about the tide",
    "The quarterly earnings report showed a significant increase in revenue driven by",
    "Photosynthesis is the process by which plants convert light energy into",
    "The defendant's attorney argued that the evidence presented by the prosecution was",
    "To configure the network firewall, first navigate to the settings panel and",
    "The ancient Roman aqueducts were engineering marvels that transported water using",
    "She picked up the violin, tucked it under her chin, and began to play a melody that",
    "The stock market experienced significant volatility today as investors reacted to",
    "According to the latest climate research, rising ocean temperatures are causing",
    "The chess grandmaster studied the board carefully before deciding to sacrifice his",
    "In object-oriented programming, inheritance allows a class to acquire properties from",
    "The archaeologists uncovered pottery fragments dating back to",
    "Machine learning models require large amounts of training data to",
]

UP_CODE = 2_494_464
UP_SCALE = 311_808
DOWN_PANEL_BYTES = UP_CODE + UP_SCALE
N = 16


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    gk = DownGatherBatchKernels()

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

    # ---- real post-ReLU2 activations: run the actual up_proj GEMV for every
    # (sequence, expert) pair using the production kernel, real weights.
    act_by_pair = {}
    for s in range(N):
        for e in seq_ids[s]:
            key = (s, e)
            if key in act_by_pair:
                continue
            code_slice_stride, scale_slice_stride = UP_CODE, UP_SCALE
            codes = cp.asarray(np.frombuffer(bank["up_codes"], dtype=np.uint8, count=UP_CODE, offset=e * UP_CODE))
            scales = cp.asarray(np.frombuffer(bank["up_scales"], dtype=np.uint8, count=UP_SCALE, offset=e * UP_SCALE))
            out = cp.zeros(moe_inter, dtype=cp.float32)
            # gsel=1 for up_proj, matching _moe_dev's real gemv_ervf_indirect(..., gsel=1, ...)
            # call and runtime.py's own bank["globals"][e, 1] pattern (lines 570/785) --
            # NOT index 0, which is down_proj's scale (used by gemv_down_masked_partial_ind's
            # hardcoded globals[id*2+0] below).
            fused.gemv_into(out, codes, scales, seq_normed[s], float(bank["globals"][e, 1]), moe_inter, hidden, apply_relu2=True)
            act_by_pair[key] = out
    cp.cuda.Device(0).synchronize()

    # ---- per-(sequence, expert) panel_scan (real, on real ReLU2 activations).
    from down_proj_batch_kernels import DownProjBatchKernels
    scan_k = DownProjBatchKernels()
    panel_by_pair = {}
    for key, act in act_by_pair.items():
        masks = cp.zeros(npanel, dtype=cp.uint32)
        plist = cp.zeros(npanel, dtype=cp.int32)
        pcount = cp.zeros(1, dtype=cp.int32)
        nz = cp.zeros(moe_inter, dtype=cp.int32)
        nzc = cp.zeros(1, dtype=cp.int32)
        scan_k.panel_scan_ref((1,), (256,), (act, np.int32(moe_inter), masks, plist, pcount, nz, nzc))
        panel_by_pair[key] = {"masks": masks, "plist": plist, "pcount": pcount, "nz": nz, "nzc": nzc}
    cp.cuda.Device(0).synchronize()

    union_experts = sorted(set(e for ids_s in seq_ids for e in ids_s))

    # ---- NAIVE: independent gather + down_masked per (sequence, expert) pair.
    naive_fetch_ms = 0.0
    naive_bytes = 0
    naive_outputs = {}
    blocks = ((moe_inter + npanel) * 32 + 255) // 256
    grid_dm = ((hidden + 127) // 128, fused.nchunks)
    for s in range(N):
        for e in seq_ids[s]:
            key = (s, e)
            p = panel_by_pair[key]
            mirror = cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)
            id_dev = cp.asarray([e], dtype=cp.int32)
            ef0, ef1 = cp.cuda.Event(), cp.cuda.Event()
            ef0.record()
            gk.run_gather_ref(np.uint64(bank["down_base_ptr"]), id_dev, DOWN_PANEL_BYTES, mirror,
                              p["plist"], p["pcount"], p["nz"], p["nzc"], hidden, blocks)
            ef1.record()
            ef1.synchronize()
            naive_fetch_ms += cp.cuda.get_elapsed_time(ef0, ef1)
            naive_bytes += int(cp.asnumpy(p["nzc"])[0]) * (hidden // 2) + int(cp.asnumpy(p["pcount"])[0]) * hidden

            partials = cp.zeros(fused.nchunks * hidden, dtype=cp.float32)
            gk.run_down_masked_ref(mirror, id_dev, cp.asarray(bank["globals"]), act_by_pair[key],
                                   p["plist"], p["masks"], p["pcount"], fused.e2m1, fused.e4m3,
                                   partials, hidden, moe_inter, fused.nchunks)
            out = scan_k.run_reduce_partials_ref(partials, hidden, fused.nchunks)
            naive_outputs[key] = cp.asnumpy(out)

    # ---- BATCHED: union nz/panel mask per expert (OR across sequences that
    # selected it), gathered ONCE; each sequence's masked-GEMV still uses its
    # OWN panel_masks/panel_list (not the union) against the shared mirror.
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

    batched_fetch_ms = 0.0
    batched_bytes = 0
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
        ef0, ef1 = cp.cuda.Event(), cp.cuda.Event()
        ef0.record()
        gk.run_gather_ref(np.uint64(bank["down_base_ptr"]), id_dev, DOWN_PANEL_BYTES, mirror,
                          union_plist, pcount_u, nz_pad, nzc_u, hidden, blocks)
        ef1.record()
        ef1.synchronize()
        batched_fetch_ms += cp.cuda.get_elapsed_time(ef0, ef1)
        batched_bytes += len(nz_list) * (hidden // 2) + int(cp.asnumpy(pcount_u)[0]) * hidden

        for s in range(N):
            if e not in seq_ids[s]:
                continue
            key = (s, e)
            p = panel_by_pair[key]
            partials = cp.zeros(fused.nchunks * hidden, dtype=cp.float32)
            gk.run_down_masked_ref(mirror, id_dev, cp.asarray(bank["globals"]), act_by_pair[key],
                                   p["plist"], p["masks"], p["pcount"], fused.e2m1, fused.e4m3,
                                   partials, hidden, moe_inter, fused.nchunks)
            out = scan_k.run_reduce_partials_ref(partials, hidden, fused.nchunks)
            batched_outputs[key] = cp.asnumpy(out)

    mismatches = sum(1 for k in naive_outputs if not (naive_outputs[k] == batched_outputs[k]).all())

    payload = {
        "kind": "proto_batch_down_proj",
        "created_utc": utc_now(),
        "note": "scoped feasibility prototype for down_proj fetch sharing via union-of-nonzero-columns gather; not a production integration",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "target_layer": target_layer,
        "n_sequences": N,
        "top_k": top_k,
        "union_expert_count": len(union_experts),
        "naive_pair_count": sum(len(x) for x in seq_ids),
        "correctness_mismatches": mismatches,
        "correctness_pass": mismatches == 0,
        "naive_fetch_ms": naive_fetch_ms,
        "batched_fetch_ms": batched_fetch_ms,
        "fetch_speedup": (naive_fetch_ms / batched_fetch_ms) if batched_fetch_ms else None,
        "naive_bytes_gathered": naive_bytes,
        "batched_bytes_gathered": batched_bytes,
        "byte_reduction_fraction": 1.0 - (batched_bytes / naive_bytes) if naive_bytes else None,
    }
    out = REPO / "pro_research" / "proto_batch_down_proj.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
