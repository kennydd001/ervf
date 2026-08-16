"""Extends proto_multi_seq_full_model.py (verified bit-exact state-swap
mechanism, first real end-to-end N=2 measurement, +5.4% from incidental
warm-cache reuse alone) with the missing piece: the EXPLICIT union-fed MoE
sharing already proven correct and faster in isolation
(proto_batch_moe_layer_combined.py, one layer, +20.9%) is here integrated
into the REAL per-layer step loop for the first time, across all MoE layers,
across real multi-step decoding.

Two correctness subtleties that were NOT obvious from the earlier isolated
prototypes and had to be worked out before writing any kernel-call code:

1. _moe_dev (the production device-cache MoE path) computes routing via
   fused.route_topk -- a CUDA kernel -- not via _route_device (a different,
   cupy-argsort-based computation used elsewhere in runtime.py for
   diagnostics/capture_routes). These are numerically related but NOT
   guaranteed bit-identical. For a bit-exact comparison against _moe_dev's
   own reference behaviour, routing here uses fused.route_topk, matching
   production exactly -- not the _route_device shortcut earlier prototypes
   used (which was fine there because those never compared against _moe_dev
   itself, only against their own hand-rolled naive arm).

2. _moe_dev accumulates each expert's down_proj contribution via
   fused.accumulate_indirect, which reads its weight from a DEVICE buffer
   slice (dev["w"][s:]) -- not fused.accumulate_into, which takes a host
   float and is used only by the older non-cached _moe path. Different
   kernels are not guaranteed bit-identical even if algebraically
   equivalent (this session's own D1 lesson: FP accumulation order/path
   matters). So the weights obtained from route_topk are kept on-device and
   fed straight into accumulate_indirect, matching _moe_dev's exact call.

Everything else (fetch sharing, down_proj union-of-masks) reuses the
already bit-exact-verified pattern from proto_batch_moe_layer_combined.py
directly.

Phase A: build the shared-MoE per-layer function.
Phase B: correctness gate -- N=2, full interleaved decode, MoE layers use
         the shared path, compared token-for-token against independent
         rt.reset()-based ground truth (device_cache=True, i.e. _moe_dev)
         runs. Must pass before any timing claim.
Phase C: if phase B passes, measure real aggregate tok/s and compare against
         proto_multi_seq_full_model.py's own naive-baseline number (+5.4%)
         and its N=1 solo control, same configuration.

Not a gated PRO experiment -- a scoped integration prototype.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic
from down_gather_batch_kernels import DownGatherBatchKernels
from down_proj_batch_kernels import DownProjBatchKernels

PROMPTS = [
    "The history of computing began when",
    "Write a correct Python function that computes the longest increasing subsequence length in O(n log n), then explain its invariant.\n",
]

STATE_ATTRS = [
    "ssm", "conv", "kc", "vc", "kv_dim", "pos",
    "h", "tmp", "acc", "normed", "act", "_act_moe", "_act_shared",
    "proj", "convo", "dt", "y", "gn", "qv", "kv_", "vv", "ctx",
    "logits", "rlog", "route_pack",
    "stage_c", "stage_s", "mstate", "contrib",
    "copy_stream", "evt", "part_acc", "part_ml",
]

UP_CODE = 2_494_464
UP_SCALE = 311_808
DOWN_PANEL_BYTES = UP_CODE + UP_SCALE

N = 2
DECODE_STEPS = 40

# Section-level profiling, off by default -- when enabled, adds
# cp.cuda.Device(0).synchronize() calls at section boundaries purely to
# attribute wall-clock time correctly (async GPU work would otherwise bleed
# across sections). This changes the pipeline's async overlap slightly, so
# absolute numbers under PROFILE=True are not directly comparable to a
# PROFILE=False timing run -- only the RELATIVE section breakdown is used.
# Does not change any computed value, so it cannot affect correctness.
PROFILE = False
_profile_totals = {}


def _prof_mark(cp, label, t0):
    if not PROFILE:
        return None
    cp.cuda.Device(0).synchronize()
    import time
    now = time.perf_counter()
    _profile_totals[label] = _profile_totals.get(label, 0.0) + (now - t0)
    return now


def snapshot_state(rt):
    rt._alloc_state()
    return {name: getattr(rt, name) for name in STATE_ATTRS}


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
    state["pos"] = rt.pos


def shared_moe_layer(rt, states, i, d, gk, scan_k):
    """One MoE layer, N sequences, union-fed shared up_proj+down_proj fetch.

    Writes each sequence's total MoE-layer output directly into
    states[s]["acc"]; caller does k.add_(states[s]["h"], states[s]["acc"], hidden).
    """
    N_ = len(states)
    cp, k, fused = rt.cp, rt.k, rt.fused
    hidden, moe_inter, shared_inter, top_k = rt.hidden, rt.moe_inter, rt.shared_inter, rt.top_k
    n_experts, scaling = rt.n_experts, rt.scaling
    npanel = moe_inter // 16
    bank = rt.bank[i]

    import time
    t0 = time.perf_counter() if PROFILE else None

    seq_ids = []
    seq_w_dev = []
    for s in range(N_):
        use_state(rt, states[s])
        k.norm(rt.normed, rt.h, d["norm"], hidden, rt.eps)
        # shared expert: unconditional, not expert-routed, confirmed to scale
        # linearly (diag_shared_expert_n_scaling.py) -- no sharing needed,
        # writes straight into this sequence's own acc.
        rt.acc.fill(0)
        fused.gemv_into(rt._act_shared, d["sh_up_c"], d["sh_up_s"], rt.normed,
                        d["sh_up_g"], shared_inter, hidden, apply_relu2=True)
        fused.gemv_into(rt.acc, d["sh_dn_c"], d["sh_dn_s"], rt._act_shared,
                        d["sh_dn_g"], hidden, shared_inter)
        # routing: fused.route_topk, matching _moe_dev exactly (not _route_device).
        k.mv_f32(rt.rlog, d["gate_w"], rt.normed, n_experts, hidden)
        ids_dev = cp.zeros(top_k, dtype=cp.int32)
        w_dev = cp.zeros(top_k, dtype=cp.float32)
        fused.route_topk(rt.rlog, d["gate_b"], ids_dev, w_dev, n_experts, top_k,
                         scaling, bad_pick=rt._bad_pick)
        ids_host = [int(x) for x in cp.asnumpy(ids_dev)]
        seq_ids.append(ids_host)
        seq_w_dev.append(w_dev)

    t0 = _prof_mark(cp, "1_routing_and_shared_expert", t0)

    union_experts = sorted(set(e for ids_s in seq_ids for e in ids_s))
    u = len(union_experts)
    expert_to_slot = {e: idx for idx, e in enumerate(union_experts)}

    # shared up_proj fetch: ONE cache_fetch over the union.
    batched_c = cp.zeros(u * UP_CODE, dtype=cp.uint8)
    batched_s = cp.zeros(u * UP_SCALE, dtype=cp.uint8)
    ids_dev_b = cp.asarray(union_experts, dtype=cp.int32)
    slots_dev_b = cp.arange(u, dtype=cp.int32)
    need_dev_b = cp.ones(u, dtype=cp.int32)
    fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                      batched_c, batched_s,
                      {"ids": ids_dev_b, "slots": slots_dev_b, "need": need_dev_b},
                      UP_CODE, UP_SCALE, u)

    t0 = _prof_mark(cp, "2_up_proj_shared_fetch", t0)

    # per (sequence, expert): up_proj GEMV from the shared buffer + panel_scan.
    act_by_pair = {}
    panel_by_pair = {}
    for s in range(N_):
        normed_s = states[s]["normed"]
        for e in seq_ids[s]:
            slot = expert_to_slot[e]
            c_slice = batched_c[slot * UP_CODE:(slot + 1) * UP_CODE]
            s_slice = batched_s[slot * UP_SCALE:(slot + 1) * UP_SCALE]
            act = cp.zeros(moe_inter, dtype=cp.float32)
            fused.gemv_into(act, c_slice, s_slice, normed_s, float(bank["globals"][e, 1]),
                            moe_inter, hidden, apply_relu2=True)
            masks = cp.zeros(npanel, dtype=cp.uint32)
            plist = cp.zeros(npanel, dtype=cp.int32)
            pcount = cp.zeros(1, dtype=cp.int32)
            nz = cp.zeros(moe_inter, dtype=cp.int32)
            nzc = cp.zeros(1, dtype=cp.int32)
            scan_k.panel_scan_ref((1,), (256,), (act, np.int32(moe_inter), masks, plist, pcount, nz, nzc))
            act_by_pair[(s, e)] = act
            panel_by_pair[(s, e)] = {"masks": masks, "plist": plist, "pcount": pcount, "nz": nz, "nzc": nzc}

    t0 = _prof_mark(cp, "3_up_proj_gemv_and_panel_scan", t0)

    # union mask per expert (OR across sequences that selected it). Stay in
    # numpy end-to-end here -- the original version round-tripped this numpy
    # data through a cupy array and then read it back via cp.asnumpy() INSIDE
    # a per-panel loop (up to npanel separate host syncs per union expert,
    # each re-fetching the WHOLE array) purely to re-derive what was already
    # sitting in acc_mask as plain numpy. That redundant GPU round-trip was
    # identified as a likely large contributor to the 12x slowdown found in
    # the first version of this script and is removed here; see
    # agents/RESEARCH_NOTEBOOK.md 2026-08-16 for the before/after comparison.
    # nz_list construction vectorized with numpy bit tricks instead of a
    # nested Python for-p-for-c loop (pure CPU-side overhead, no GPU
    # semantics -- correctness re-verified by Phase B below exactly as
    # before, since this only changes HOW the same nz indices are computed,
    # not what routing/masks/fetches happen).
    bit_shifts = np.arange(16, dtype=np.uint32)
    union_plist_by_expert = {}
    union_nz_by_expert = {}
    for e in union_experts:
        acc_mask = np.zeros(npanel, dtype=np.uint32)
        for s in range(N_):
            if e in seq_ids[s]:
                acc_mask |= cp.asnumpy(panel_by_pair[(s, e)]["masks"])
        plist_np = np.flatnonzero(acc_mask).astype(np.int32)
        bits = ((acc_mask[plist_np][:, None] >> bit_shifts) & 1).astype(bool)
        idx_matrix = (plist_np[:, None].astype(np.int64) << 4) + bit_shifts[None, :]
        union_plist_by_expert[e] = plist_np
        union_nz_by_expert[e] = idx_matrix[bits].astype(np.int32)

    t0 = _prof_mark(cp, "4_union_mask_build", t0)

    # down_proj gather: batched over the u UNIQUE union experts in ONE launch
    # (gather_down_sparse_ind_batched's slot dimension IS the union-expert
    # dimension already -- no per-pair duplication needed here, unlike
    # down_masked below). A first version of this script left gather as u
    # separate small launches; re-profiling after batching down_masked+
    # reduce showed gather alone had become the new dominant cost (37.4% of
    # total), which this closes -- see agents/RESEARCH_NOTEBOOK.md 2026-08-16.
    blocks = ((moe_inter + npanel) * 32 + 255) // 256
    u = len(union_experts)
    union_ids_dev = cp.asarray(union_experts, dtype=cp.int32)
    union_plist_pad = cp.zeros(u * npanel, dtype=cp.int32)
    union_pcount_dev = cp.zeros(u, dtype=cp.int32)
    union_nz_pad = cp.zeros(u * moe_inter, dtype=cp.int32)
    union_nzc_dev = cp.zeros(u, dtype=cp.int32)
    for ui, e in enumerate(union_experts):
        plist_e = union_plist_by_expert[e]
        nz_e = union_nz_by_expert[e]
        union_plist_pad[ui * npanel:ui * npanel + len(plist_e)] = cp.asarray(plist_e)
        union_pcount_dev[ui] = len(plist_e)
        union_nz_pad[ui * moe_inter:ui * moe_inter + len(nz_e)] = cp.asarray(nz_e)
        union_nzc_dev[ui] = len(nz_e)

    mirror_batched_by_union = cp.zeros(u * DOWN_PANEL_BYTES, dtype=cp.uint8)
    gk.run_gather_batched(np.uint64(bank["down_base_ptr"]), union_ids_dev, DOWN_PANEL_BYTES,
                          mirror_batched_by_union, union_plist_pad, union_pcount_dev,
                          union_nz_pad, union_nzc_dev, hidden, npanel, moe_inter,
                          DOWN_PANEL_BYTES, u, blocks)
    mirror_by_expert = {
        e: mirror_batched_by_union[ui * DOWN_PANEL_BYTES:(ui + 1) * DOWN_PANEL_BYTES]
        for ui, e in enumerate(union_experts)
    }

    t0 = _prof_mark(cp, "5a_down_proj_gather_batched", t0)

    # Batched down_masked + reduce_partials over ALL (sequence, expert)
    # pairs in ONE launch each, using the already-verified V5/V6 batched
    # kernels. gemv_down_masked_partial_ind_batched expects a CONTIGUOUS
    # [pairs, mirror_bytes] bank with no indirection (slot s reads
    # bank + s*mirror_bytes directly) -- so a pair that shares a union
    # expert with another pair gets the SAME already-on-device mirror data
    # copied (VRAM-to-VRAM, cheap) into its own slot rather than re-fetched
    # from host (the expensive part, which stays deduplicated above).
    # `pairs` is built in (sequence, route-slot) order, which also means
    # each sequence's own top_k pairs land in a CONTIGUOUS run of slots --
    # used below to batch that sequence's own accumulate too.
    pairs = [(s, e) for s in range(N_) for e in seq_ids[s]]
    P = len(pairs)
    globals_dev = cp.asarray(bank["globals"])

    bank_batched = cp.zeros(P * DOWN_PANEL_BYTES, dtype=cp.uint8)
    ids_batched = cp.zeros(P, dtype=cp.int32)
    act_batched = cp.zeros(P * moe_inter, dtype=cp.float32)
    masks_batched = cp.zeros(P * npanel, dtype=cp.uint32)
    plist_batched = cp.zeros(P * npanel, dtype=cp.int32)
    pcount_batched = cp.zeros(P, dtype=cp.int32)
    for pi, (s, e) in enumerate(pairs):
        bank_batched[pi * DOWN_PANEL_BYTES:(pi + 1) * DOWN_PANEL_BYTES] = mirror_by_expert[e]
        ids_batched[pi] = e
        p = panel_by_pair[(s, e)]
        act_batched[pi * moe_inter:(pi + 1) * moe_inter] = act_by_pair[(s, e)]
        masks_batched[pi * npanel:(pi + 1) * npanel] = p["masks"]
        plist_batched[pi * npanel:(pi + 1) * npanel] = p["plist"]
        pcount_batched[pi] = p["pcount"]

    partials_batched = cp.zeros(P * fused.nchunks * hidden, dtype=cp.float32)
    gk.run_down_masked_batched(bank_batched, ids_batched, globals_dev, act_batched,
                               plist_batched, masks_batched, pcount_batched,
                               fused.e2m1, fused.e4m3, partials_batched,
                               hidden, moe_inter, npanel, DOWN_PANEL_BYTES, P, fused.nchunks)
    contrib_batched = scan_k.run_reduce_partials_batched(partials_batched, hidden, fused.nchunks, P)

    t0 = _prof_mark(cp, "5b_down_proj_masked_reduce_batched", t0)

    # accumulate: batched per sequence (its own top_k contributions are a
    # CONTIGUOUS slice of contrib_batched thanks to the (sequence, slot)
    # pair order above), using the same weighted_accumulate_ind_batched
    # kernel V6 already uses in production for exactly this per-sequence
    # top_k reduction -- same fixed s=0..top_k-1 fmaf order as sequential
    # accumulate_indirect calls, not a parallel/atomic reduction (D1 lesson).
    for s in range(N_):
        start = s * top_k
        contrib_s = contrib_batched[start * hidden:(start + top_k) * hidden]
        scan_k.run_accumulate_batched(states[s]["acc"], contrib_s, seq_w_dev[s], hidden, top_k)
    _prof_mark(cp, "6_accumulate", t0)


def multi_step(rt, states, token_ids, gk, scan_k):
    cp, k = rt.cp, rt.k
    N_ = len(states)
    for s in range(N_):
        use_state(rt, states[s])
        tid = token_ids[s]
        if rt.embed_on_host:
            row = cp.asarray(rt.embed_host[tid * rt.hidden:(tid + 1) * rt.hidden])
        else:
            row = rt.embed[tid * rt.hidden:(tid + 1) * rt.hidden]
        rt.h[:] = (row.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)

    for i, ch in enumerate(rt.pattern):
        d = rt.layer[i]
        if ch == "M":
            for s in range(N_):
                use_state(rt, states[s])
                k.norm(rt.normed, rt.h, d["norm"], rt.hidden, rt.eps)
                rt._mamba(i, rt.acc)
                k.add_(rt.h, rt.acc, rt.hidden)
        elif ch == "*":
            for s in range(N_):
                use_state(rt, states[s])
                k.norm(rt.normed, rt.h, d["norm"], rt.hidden, rt.eps)
                rt._attention(i, rt.acc)
                k.add_(rt.h, rt.acc, rt.hidden)
        else:
            shared_moe_layer(rt, states, i, d, gk, scan_k)
            for s in range(N_):
                use_state(rt, states[s])
                k.add_(rt.h, rt.acc, rt.hidden)

    next_tokens = []
    for s in range(N_):
        use_state(rt, states[s])
        k.norm(rt.normed, rt.h, rt.norm_f, rt.hidden, rt.eps)
        if rt.lm_head_kind == "nvfp4":
            rt.fused.gemv_into(rt.logits, rt.lm_head_codes, rt.lm_head_scales,
                               rt.normed, rt.lm_head_g, rt.vocab, rt.hidden)
        else:
            k.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden)
        rt.pos += 1
        save_state(rt, states[s])
        next_tokens.append(int(cp.argmax(rt.logits)))
    return next_tokens


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    gk = DownGatherBatchKernels()
    scan_k = DownProjBatchKernels()

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    rt.device_cache = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]

    # ================= Phase B: correctness gate =================
    # Ground truth: N independent, unswapped, sequential rt.step() runs
    # (device_cache=True -> _moe_dev, the real reference this shared path
    # must match).
    ground_truth_tokens = [[] for _ in range(N)]
    for s in range(N):
        rt.reset()
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        ground_truth_tokens[s].append(nxt)
        for _ in range(DECODE_STEPS - 1):
            nxt = int(rt.step(nxt))
            ground_truth_tokens[s].append(nxt)
    cp.cuda.Device(0).synchronize()

    # Shared path: prefill each sequence independently through the swap
    # mechanism (no sharing needed for prefill correctness check -- MoE
    # sharing is exercised starting from the first real decode step), then
    # run the shared multi_step loop for DECODE_STEPS real steps.
    rt.enable_cache(72)
    state = [snapshot_state(rt) for _ in range(N)]
    cur = [None] * N
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        cur[s] = nxt
        save_state(rt, state[s])

    shared_tokens = [[] for _ in range(N)]
    for s in range(N):
        shared_tokens[s].append(cur[s])
    for _ in range(DECODE_STEPS - 1):
        cur = multi_step(rt, state, cur, gk, scan_k)
        for s in range(N):
            shared_tokens[s].append(cur[s])
    cp.cuda.Device(0).synchronize()

    equivalence_pass = (ground_truth_tokens == shared_tokens)
    if not equivalence_pass:
        payload = {
            "kind": "proto_multi_seq_moe_shared",
            "created_utc": utc_now(),
            "phase_reached": "phaseB_correctness_gate_FAILED",
            "ground_truth_tokens": ground_truth_tokens,
            "shared_tokens": shared_tokens,
            "note": "shared MoE integration did NOT reproduce independent _moe_dev-based runs bit-exact. Stopping before any timing claim.",
        }
        out = REPO / "pro_research" / "proto_multi_seq_moe_shared.json"
        write_json_atomic(out, payload, archive=False)
        print(payload)
        return 1

    print(f"Phase B PASS: shared-MoE multi-step run bit-exact matches independent _moe_dev ground truth ({DECODE_STEPS} tokens x {N} sequences)")

    # ================= Phase C: timing =================
    rt.enable_cache(72)
    state = [snapshot_state(rt) for _ in range(N)]
    cur = [None] * N
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        cur[s] = nxt
        save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()

    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    tokens_by_seq = [[] for _ in range(N)]
    for _ in range(DECODE_STEPS):
        cur = multi_step(rt, state, cur, gk, scan_k)
        for s in range(N):
            tokens_by_seq[s].append(cur[s])
    e1.record()
    e1.synchronize()
    total_ms = cp.cuda.get_elapsed_time(e0, e1)

    total_real_tokens = N * DECODE_STEPS
    ms_per_token_aggregate = total_ms / total_real_tokens
    aggregate_tok_s = 1000.0 / ms_per_token_aggregate

    payload = {
        "kind": "proto_multi_seq_moe_shared",
        "created_utc": utc_now(),
        "phase_reached": "phaseC_timing_measured",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps_per_sequence": DECODE_STEPS,
        "phaseB_equivalence_pass": equivalence_pass,
        "note": "MoE layers use the union-fed shared up_proj+down_proj fetch (proto_batch_moe_layer_combined.py's mechanism, now integrated into the real per-layer step loop across all 23 MoE layers and multiple real decode steps); non-MoE layers (attention/Mamba) run per-sequence unchanged via the verified state-swap mechanism from proto_multi_seq_full_model.py. Compare aggregate_tok_s here against that script's naive baseline (31.411 tok/s aggregate, solo 29.798 tok/s) -- same configuration, only the MoE sharing differs.",
        "total_wall_ms_for_all_real_tokens": total_ms,
        "total_real_tokens_across_all_sequences": total_real_tokens,
        "ms_per_real_token_aggregate": ms_per_token_aggregate,
        "aggregate_tok_s": aggregate_tok_s,
        "tokens_by_sequence": tokens_by_seq,
    }
    if PROFILE and _profile_totals:
        total_profiled = sum(_profile_totals.values())
        payload["profile_section_seconds"] = dict(_profile_totals)
        payload["profile_section_fraction"] = {
            k: v / total_profiled for k, v in _profile_totals.items()
        }
        payload["profile_note"] = "PROFILE=True adds sync points at section boundaries, changing async overlap -- these absolute numbers are NOT the same run as the timing above; only relative fractions are meaningful."
    out = REPO / "pro_research" / "proto_multi_seq_moe_shared.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
