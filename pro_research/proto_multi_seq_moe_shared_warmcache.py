"""Combines two findings that were proven SEPARATELY this session but never
together: (1) the explicit union-fed MoE sharing mechanism
(proto_multi_seq_moe_shared.py, bit-exact, 11.23 tok/s at N=2) always
allocated a FRESH device-cache structure and FRESH fetch buffers every
single layer, every single step -- a cold-cache-every-time design. (2) a
PERSISTENT, evolving device-LRU cache reused across real consecutive steps
was separately shown to cut misses by 27.6% even under cold start
(diag_batch_warm_cache.py). Neither prior script combined them: the
diagnostic used a standalone cache with no real GEMV/down_proj pipeline
attached, and the integrated pipeline never kept its cache warm across
steps.

This does: the SAME shared_moe_layer pipeline (routing, union-fed up_proj
fetch, union-of-masks down_proj fetch, batched down_masked/reduce/
accumulate -- all already bit-exact-verified), but with ONE persistent
per-layer device-cache (cap=72, matching production's own default) built
BEFORE the decode loop and reused, evolving, across all real decode steps
-- instead of fused.alloc_device_cache() called fresh inside
shared_moe_layer on every single call. Two independent attempts at fixing
the fresh-allocation overhead already failed (see
agents/RESEARCH_NOTEBOOK.md 2026-08-16, "Routekaartstap 1 begonnen" and its
two follow-ups) by shrinking the buffer size within a still-fresh-per-call
allocation; this instead removes the per-call allocation itself, which is
the more likely actual cost per the same notebook entry's own conclusion.

Same correctness discipline: Phase B compares bit-exact against independent
rt.step()-based ground truth (device_cache=True, i.e. real _moe_dev) before
any timing claim is trusted.

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
CACHE_CAP = 72  # matches production's own default (A1 adoption precedent)

# Root-cause profiling for the 6.5x regression found in the first version of
# this script (1.725 vs 11.234 tok/s) -- off by default, adds sync points at
# section boundaries (changes async overlap, so only relative fractions are
# meaningful), does not change any computed value.
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


def build_layer_caches(rt, moe_layers, cap, max_p):
    fused = rt.fused
    caches = {}
    for i in moe_layers:
        bank = rt.bank[i]
        dev = fused.alloc_device_cache(rt.n_experts, cap, max_p, bank["globals"])
        caches[i] = {
            "dev": dev,
            "codes": rt.cp.zeros(cap * UP_CODE, dtype=rt.cp.uint8),
            "scales": rt.cp.zeros(cap * UP_SCALE, dtype=rt.cp.uint8),
        }
    return caches


def shared_moe_layer(rt, states, i, d, gk, scan_k, layer_cache):
    """Same pipeline as proto_multi_seq_moe_shared.py's shared_moe_layer,
    except the up_proj fetch uses a PERSISTENT, evolving per-layer cache
    (layer_cache, built once before the decode loop) instead of allocating
    a fresh device-cache structure and fresh fetch buffers on every call.
    """
    N_ = len(states)
    cp, k, fused = rt.cp, rt.k, rt.fused
    hidden, moe_inter, shared_inter, top_k = rt.hidden, rt.moe_inter, rt.shared_inter, rt.top_k
    n_experts, scaling = rt.n_experts, rt.scaling
    npanel = moe_inter // 16
    bank = rt.bank[i]

    import time
    t0 = time.perf_counter() if PROFILE else None

    P = N_ * top_k
    all_ids_dev = cp.zeros(P, dtype=cp.int32)
    all_w_dev = cp.zeros(P, dtype=cp.float32)
    for s in range(N_):
        use_state(rt, states[s])
        t0 = _prof_mark(cp, "1a_use_state", t0)
        k.norm(rt.normed, rt.h, d["norm"], hidden, rt.eps)
        t0 = _prof_mark(cp, "1b_norm", t0)
        rt.acc.fill(0)
        t0 = _prof_mark(cp, "1c_acc_fill", t0)
        fused.gemv_into(rt._act_shared, d["sh_up_c"], d["sh_up_s"], rt.normed,
                        d["sh_up_g"], shared_inter, hidden, apply_relu2=True)
        t0 = _prof_mark(cp, "1d_shared_up_gemv", t0)
        fused.gemv_into(rt.acc, d["sh_dn_c"], d["sh_dn_s"], rt._act_shared,
                        d["sh_dn_g"], hidden, shared_inter)
        t0 = _prof_mark(cp, "1e_shared_down_gemv", t0)
        k.mv_f32(rt.rlog, d["gate_w"], rt.normed, n_experts, hidden)
        t0 = _prof_mark(cp, "1f_mv_f32_gate", t0)
        fused.route_topk(rt.rlog, d["gate_b"], all_ids_dev[s * top_k:(s + 1) * top_k],
                         all_w_dev[s * top_k:(s + 1) * top_k], n_experts, top_k,
                         scaling, bad_pick=rt._bad_pick)
        t0 = _prof_mark(cp, "1g_route_topk", t0)

    # persistent, evolving cache: cache_assign carries slot_of/expert_of/
    # last_used/state2 across calls (that IS its production-intended
    # semantics -- a real LRU, not a one-shot union), so experts still
    # resident from a PREVIOUS step's cache_assign correctly get need=0
    # here too, on top of within-this-call deduplication.
    dev = layer_cache["dev"]
    codes = layer_cache["codes"]
    scales = layer_cache["scales"]
    dev["ids"][:P] = all_ids_dev

    t0 = _prof_mark(cp, "2a_ids_copy", t0)

    fused.cache_assign(dev, dev["ids"][:P], CACHE_CAP, P)

    t0 = _prof_mark(cp, "2b_cache_assign", t0)

    fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                      codes, scales, dev, UP_CODE, UP_SCALE, P)

    t0 = _prof_mark(cp, "2c_cache_fetch", t0)

    all_ids_host = cp.asnumpy(all_ids_dev).tolist()
    slots_host = cp.asnumpy(dev["slots"][:P]).tolist()
    seq_ids = [all_ids_host[s * top_k:(s + 1) * top_k] for s in range(N_)]
    seq_w_dev = [all_w_dev[s * top_k:(s + 1) * top_k] for s in range(N_)]

    t0 = _prof_mark(cp, "2d_host_sync", t0)

    act_by_pair = {}
    panel_by_pair = {}
    pair_idx = 0
    for s in range(N_):
        normed_s = states[s]["normed"]
        for e in seq_ids[s]:
            slot = slots_host[pair_idx]
            pair_idx += 1
            c_slice = codes[slot * UP_CODE:(slot + 1) * UP_CODE]
            s_slice = scales[slot * UP_SCALE:(slot + 1) * UP_SCALE]
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

    union_experts = sorted(set(e for ids_s in seq_ids for e in ids_s))

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

    pairs = [(s, e) for s in range(N_) for e in seq_ids[s]]
    Ppairs = len(pairs)
    globals_dev = cp.asarray(bank["globals"])

    bank_batched = cp.zeros(Ppairs * DOWN_PANEL_BYTES, dtype=cp.uint8)
    ids_batched = cp.zeros(Ppairs, dtype=cp.int32)
    act_batched = cp.zeros(Ppairs * moe_inter, dtype=cp.float32)
    masks_batched = cp.zeros(Ppairs * npanel, dtype=cp.uint32)
    plist_batched = cp.zeros(Ppairs * npanel, dtype=cp.int32)
    pcount_batched = cp.zeros(Ppairs, dtype=cp.int32)
    for pi, (s, e) in enumerate(pairs):
        bank_batched[pi * DOWN_PANEL_BYTES:(pi + 1) * DOWN_PANEL_BYTES] = mirror_by_expert[e]
        ids_batched[pi] = e
        p = panel_by_pair[(s, e)]
        act_batched[pi * moe_inter:(pi + 1) * moe_inter] = act_by_pair[(s, e)]
        masks_batched[pi * npanel:(pi + 1) * npanel] = p["masks"]
        plist_batched[pi * npanel:(pi + 1) * npanel] = p["plist"]
        pcount_batched[pi] = p["pcount"]

    partials_batched = cp.zeros(Ppairs * fused.nchunks * hidden, dtype=cp.float32)
    gk.run_down_masked_batched(bank_batched, ids_batched, globals_dev, act_batched,
                               plist_batched, masks_batched, pcount_batched,
                               fused.e2m1, fused.e4m3, partials_batched,
                               hidden, moe_inter, npanel, DOWN_PANEL_BYTES, Ppairs, fused.nchunks)
    contrib_batched = scan_k.run_reduce_partials_batched(partials_batched, hidden, fused.nchunks, Ppairs)

    t0 = _prof_mark(cp, "5b_down_proj_masked_reduce_batched", t0)

    for s in range(N_):
        start = s * top_k
        contrib_s = contrib_batched[start * hidden:(start + top_k) * hidden]
        scan_k.run_accumulate_batched(states[s]["acc"], contrib_s, seq_w_dev[s], hidden, top_k)
    _prof_mark(cp, "6_accumulate", t0)


def multi_step(rt, states, token_ids, gk, scan_k, layer_caches):
    cp, k = rt.cp, rt.k
    N_ = len(states)
    import time
    t0 = time.perf_counter() if PROFILE else None
    for s in range(N_):
        use_state(rt, states[s])
        tid = token_ids[s]
        if rt.embed_on_host:
            row = cp.asarray(rt.embed_host[tid * rt.hidden:(tid + 1) * rt.hidden])
        else:
            row = rt.embed[tid * rt.hidden:(tid + 1) * rt.hidden]
        rt.h[:] = (row.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
    t0 = _prof_mark(cp, "0_embed", t0)

    for i, ch in enumerate(rt.pattern):
        d = rt.layer[i]
        if ch == "M":
            for s in range(N_):
                use_state(rt, states[s])
                k.norm(rt.normed, rt.h, d["norm"], rt.hidden, rt.eps)
                rt._mamba(i, rt.acc)
                k.add_(rt.h, rt.acc, rt.hidden)
            t0 = _prof_mark(cp, "M_mamba_layer", t0)
        elif ch == "*":
            for s in range(N_):
                use_state(rt, states[s])
                k.norm(rt.normed, rt.h, d["norm"], rt.hidden, rt.eps)
                rt._attention(i, rt.acc)
                k.add_(rt.h, rt.acc, rt.hidden)
            t0 = _prof_mark(cp, "S_attention_layer", t0)
        else:
            shared_moe_layer(rt, states, i, d, gk, scan_k, layer_caches[i])
            for s in range(N_):
                use_state(rt, states[s])
                k.add_(rt.h, rt.acc, rt.hidden)
            t0 = _prof_mark(cp, "E_moe_layer_addback", t0)

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

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    max_p = N * rt.top_k

    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]

    # ================= Phase B: correctness gate =================
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

    layer_caches = build_layer_caches(rt, moe_layers, CACHE_CAP, max_p)
    shared_tokens = [[] for _ in range(N)]
    for s in range(N):
        shared_tokens[s].append(cur[s])
    for _ in range(DECODE_STEPS - 1):
        cur = multi_step(rt, state, cur, gk, scan_k, layer_caches)
        for s in range(N):
            shared_tokens[s].append(cur[s])
    cp.cuda.Device(0).synchronize()

    equivalence_pass = (ground_truth_tokens == shared_tokens)
    if not equivalence_pass:
        payload = {
            "kind": "proto_multi_seq_moe_shared_warmcache",
            "created_utc": utc_now(),
            "phase_reached": "phaseB_correctness_gate_FAILED",
            "ground_truth_tokens": ground_truth_tokens,
            "shared_tokens": shared_tokens,
            "note": "warm-cache shared MoE integration did NOT reproduce independent _moe_dev-based runs bit-exact. Stopping before any timing claim.",
        }
        out = REPO / "pro_research" / "proto_multi_seq_moe_shared_warmcache.json"
        write_json_atomic(out, payload, archive=False)
        print(payload)
        return 1

    print(f"Phase B PASS: warm-cache shared-MoE multi-step run bit-exact matches independent _moe_dev ground truth ({DECODE_STEPS} tokens x {N} sequences)")

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

    layer_caches = build_layer_caches(rt, moe_layers, CACHE_CAP, max_p)
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    tokens_by_seq = [[] for _ in range(N)]
    for _ in range(DECODE_STEPS):
        cur = multi_step(rt, state, cur, gk, scan_k, layer_caches)
        for s in range(N):
            tokens_by_seq[s].append(cur[s])
    e1.record()
    e1.synchronize()
    total_ms = cp.cuda.get_elapsed_time(e0, e1)

    total_real_tokens = N * DECODE_STEPS
    ms_per_token_aggregate = total_ms / total_real_tokens
    aggregate_tok_s = 1000.0 / ms_per_token_aggregate

    payload = {
        "kind": "proto_multi_seq_moe_shared_warmcache",
        "created_utc": utc_now(),
        "phase_reached": "phaseC_timing_measured",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps_per_sequence": DECODE_STEPS,
        "cache_cap": CACHE_CAP,
        "phaseB_equivalence_pass": equivalence_pass,
        "note": "same shared_moe_layer pipeline as proto_multi_seq_moe_shared.py (11.234 tok/s, fresh per-call cache) but with ONE persistent per-layer device-cache (cap=72) built once and reused/evolving across all real decode steps, combining within-step union-sharing with cross-step warm-cache reuse for the first time",
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
        payload["profile_note"] = "PROFILE=True adds sync points at section boundaries, changing async overlap -- absolute numbers are NOT the same run as a PROFILE=False timing; only relative fractions are meaningful."
    out = REPO / "pro_research" / "proto_multi_seq_moe_shared_warmcache.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
