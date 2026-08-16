"""Scoped prototype: does batching the MoE expert FETCH across N independent
sequences actually pay off, measured directly (not projected from the union
count alone)? This tests the single highest-impact piece the batch>1
hypothesis identified (agents/RESEARCH_NOTEBOOK.md, TODO.md 2026-08-16) in
isolation -- it does NOT touch attention, Mamba, KV-cache, or graph capture,
which is why it's achievable in one session instead of the multi-week
rearchitecture full batch>1 support would need.

Method, worst case / cold cache (isolates the fetch-amortization effect from
LRU hit-rate dynamics, which are a separate, already-studied axis this
session):

1. Capture N real (normed activation, top-6 route ids) pairs for one real
   MoE layer from N diverse prompts (reusing diag_cross_sequence_union.py's
   prompt set and capture_routes, extended to also snapshot self.normed).
2. NAIVE: for each of the N sequences independently, bulk-fetch each of its
   6 experts' up_proj codes+scales fresh (N*6 total fetches, no
   deduplication -- what N separate batch=1 runtime instances would do).
3. BATCHED: bulk-fetch only the UNIQUE experts in the union once
   (|union| total fetches), then read from that shared buffer for every
   sequence that selected each expert.
4. CORRECTNESS: run the existing production ERVF GEMV kernel
   (gemv_nvfp4_ervf_ind, unmodified, same as the rest of this session) for
   every (sequence, expert) pair in BOTH the naive and batched arms, and
   assert bit-exact per-sequence match. Same standard as every other
   integration this session -- no timing claim survives without this.
5. Time both fetch phases with cp.cuda.Event, same GPU, same session.

Not a gated PRO experiment -- a scoped feasibility test, explicitly not a
production integration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, require_model_dir, utc_now, write_json_atomic

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
N = 16


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

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
    target_layer = moe_layers[10]  # an interior layer

    # ---- step 1: capture N real (normed, route_ids) pairs at target_layer.
    captured = []
    orig_route = rt._route_device

    def capture_route(i):
        packed = orig_route(i)
        if i == target_layer:
            captured.append({
                "normed": cp.asarray(rt.normed).copy(),
                "packed": cp.asarray(packed).copy(),
            })
        return packed

    import types
    rt._route_device = types.MethodType(lambda self, i: capture_route(i), rt)

    for prompt in PROMPTS[:N]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        # one more step to land on a generated (not prompt) position
        rt.step(nxt)

    rt._route_device = orig_route
    cp.cuda.Device(0).synchronize()

    if len(captured) < N:
        print(f"only captured {len(captured)} sequences at layer {target_layer}, expected {N}")
        return 1
    captured = captured[-N:]  # last N calls at target_layer (one per prompt's final step)

    top_k = rt.top_k
    seq_ids = []
    seq_normed = []
    for c in captured:
        packed = cp.asnumpy(c["packed"])
        ids_s = [int(x) for x in packed[:top_k]]
        seq_ids.append(ids_s)
        seq_normed.append(c["normed"])

    union_experts = sorted(set(e for ids_s in seq_ids for e in ids_s))
    bank = rt.bank[target_layer]
    hidden = rt.hidden
    moe_inter = rt.moe_inter

    # ---- NAIVE: N*top_k independent fetches (no dedup), bit-exact GEMV per pair.
    naive_scratch_c = cp.zeros(UP_CODE, dtype=cp.uint8)
    naive_scratch_s = cp.zeros(UP_SCALE, dtype=cp.uint8)
    ids_dev = cp.zeros(1, dtype=cp.int32)
    slots_dev = cp.zeros(1, dtype=cp.int32)
    need_dev = cp.ones(1, dtype=cp.int32)

    naive_fetch_events = []
    naive_outputs = {}
    # bank["globals"] is a host numpy array shaped [n_experts, 2] (matches
    # runtime.py's own bank["globals"][e, 1] indexing, e.g. lines 570/785) --
    # not a flat [n_experts*2] buffer.
    for s in range(N):
        for e in seq_ids[s]:
            ids_dev[0] = e
            e0 = cp.cuda.Event()
            e1 = cp.cuda.Event()
            e0.record()
            fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                              naive_scratch_c, naive_scratch_s,
                              {"ids": ids_dev, "slots": slots_dev, "need": need_dev},
                              UP_CODE, UP_SCALE, 1)
            e1.record()
            e1.synchronize()
            naive_fetch_events.append(cp.cuda.get_elapsed_time(e0, e1))

            out = cp.zeros(moe_inter, dtype=cp.float32)
            fused.gemv_into(out, naive_scratch_c, naive_scratch_s, seq_normed[s],
                            float(bank["globals"][e, 1]), moe_inter, hidden, apply_relu2=True)
            naive_outputs[(s, e)] = cp.asnumpy(out)

    # ---- BATCHED: |union| fetches (dedup), shared buffer, same GEMV kernel per pair.
    u = len(union_experts)
    expert_to_slot = {e: i for i, e in enumerate(union_experts)}
    batched_c = cp.zeros(u * UP_CODE, dtype=cp.uint8)
    batched_s = cp.zeros(u * UP_SCALE, dtype=cp.uint8)
    ids_dev_b = cp.asarray(union_experts, dtype=cp.int32)
    slots_dev_b = cp.arange(u, dtype=cp.int32)
    need_dev_b = cp.ones(u, dtype=cp.int32)

    e0 = cp.cuda.Event()
    e1 = cp.cuda.Event()
    e0.record()
    fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                      batched_c, batched_s,
                      {"ids": ids_dev_b, "slots": slots_dev_b, "need": need_dev_b},
                      UP_CODE, UP_SCALE, u)
    e1.record()
    e1.synchronize()
    batched_fetch_ms = cp.cuda.get_elapsed_time(e0, e1)

    batched_outputs = {}
    for s in range(N):
        for e in seq_ids[s]:
            slot = expert_to_slot[e]
            c_slice = batched_c[slot * UP_CODE:(slot + 1) * UP_CODE]
            s_slice = batched_s[slot * UP_SCALE:(slot + 1) * UP_SCALE]
            out = cp.zeros(moe_inter, dtype=cp.float32)
            fused.gemv_into(out, c_slice, s_slice, seq_normed[s],
                            float(bank["globals"][e, 1]), moe_inter, hidden, apply_relu2=True)
            batched_outputs[(s, e)] = cp.asnumpy(out)

    mismatches = 0
    for key in naive_outputs:
        if not (naive_outputs[key] == batched_outputs[key]).all():
            mismatches += 1
    correctness_pass = mismatches == 0

    naive_total_fetch_ms = sum(naive_fetch_events)
    naive_fetch_count = N * top_k
    batched_fetch_count = u

    payload = {
        "kind": "proto_batch_moe_layer",
        "created_utc": utc_now(),
        "note": "scoped feasibility prototype, not a production integration; cold-cache worst case, isolates fetch-amortization from LRU hit-rate effects",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "target_layer": target_layer,
        "n_sequences": N,
        "top_k": top_k,
        "union_expert_count": u,
        "naive_fetch_count": naive_fetch_count,
        "batched_fetch_count": batched_fetch_count,
        "dedup_fraction": 1.0 - (batched_fetch_count / naive_fetch_count),
        "correctness_mismatches": mismatches,
        "correctness_pass": correctness_pass,
        "naive_total_fetch_ms": naive_total_fetch_ms,
        "naive_fetch_ms_per_call_stats": percentiles(naive_fetch_events),
        "batched_total_fetch_ms": batched_fetch_ms,
        "fetch_speedup": (naive_total_fetch_ms / batched_fetch_ms) if batched_fetch_ms else None,
        "fetch_ms_saved": naive_total_fetch_ms - batched_fetch_ms,
    }
    out = REPO / "pro_research" / "proto_batch_moe_layer.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if correctness_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
