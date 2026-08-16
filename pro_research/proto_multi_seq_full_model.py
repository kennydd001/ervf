"""First real, full-model, multi-step, multi-sequence measurement -- not a
single MoE layer, not an isolated kernel, the actual production step() loop
run for N=2 independent sequences with a real, physically measured aggregate
tok/s number. Every component-level batch>1 risk was already de-risked this
session (staggered positions, warm cache, VRAM, combined MoE mechanism,
N-scaling of every non-routed component); this is the first attempt to
compose them into something that actually produces an end-to-end number,
while staying honest that this is NOT the multi-week full architectural
rewrite agents/BATCH_ARCHITECTURE_DESIGN.md scopes out (no CUDA graph, no
continuous batching / staggered positions, no MoE cache sharing in this
first phase -- see phase 3 below for what IS and is NOT covered).

Mechanism: LightningRuntime.step() operates entirely through self.X instance
attributes (h, normed, kc, vc, ssm, conv, qv, ctx, rlog, ...) with NO
dependency on any other object state except the shared, read-only weights
(self.layer, self.bank, self.fused, self.k) and the shared MoE device-cache
(self.cache / self._dev_cache). That means: capture the ~30 per-sequence
DYNAMIC buffers _alloc_state() allocates into a per-sequence snapshot dict
(by calling the real _alloc_state() N times, not hand-copying each buffer --
avoids transcription risk on ~30 buffer shapes), then swap them onto `rt`
with plain setattr before calling the UNMODIFIED, real rt.step(). This reuses
production code path-for-path; the only new code is the snapshot/swap
mechanism itself, which phase 2 verifies bit-exact before any timing claim.

Phase 1: build the swap mechanism.
Phase 2: verify sequence 0 running through the swapped path reproduces
         bit-exact the same token ids as a normal, unswapped single-sequence
         run on the identical prompt -- the correctness gate for everything
         below. If this fails, STOP -- a buffer was missed.
Phase 3: measure a REAL N=2 naive aggregate baseline: two independent
         sequences, each stepped through the full 52-layer model via the
         verified swap mechanism, MoE device-cache SHARED between them as an
         ordinary side effect of not swapping self.cache/self._dev_cache
         (warm-cache reuse, same class of effect as diag_batch_warm_cache.py,
         but NOT the explicit union-fed sharing from
         proto_batch_moe_layer_combined.py -- that is a separate, larger
         follow-up, not attempted here).

Not a gated PRO experiment -- a scoped integration prototype.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic

PROMPTS = [
    "The history of computing began when",
    "Write a correct Python function that computes the longest increasing subsequence length in O(n log n), then explain its invariant.\n",
]

# Every attribute _alloc_state() sets, read directly from runtime.py's own
# source (lines 270-322) -- not re-derived by hand, to avoid missing one.
STATE_ATTRS = [
    "ssm", "conv", "kc", "vc", "kv_dim", "pos",
    "h", "tmp", "acc", "normed", "act", "_act_moe", "_act_shared",
    "proj", "convo", "dt", "y", "gn", "qv", "kv_", "vv", "ctx",
    "logits", "rlog", "route_pack",
    "stage_c", "stage_s", "mstate", "contrib",
    "copy_stream", "evt", "part_acc", "part_ml",
]

N = 2
DECODE_STEPS = 40


def snapshot_state(rt):
    rt._alloc_state()
    return {name: getattr(rt, name) for name in STATE_ATTRS}


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
    # pos is a plain Python int: step() does `self.pos += 1`, which REBINDS
    # rt.pos to a new int object rather than mutating one in place (unlike
    # every cp.ndarray buffer in STATE_ATTRS, which is written into in place
    # by kernel calls and therefore stays correctly aliased across swaps with
    # no action needed here). Without this explicit write-back, swapping away
    # from a sequence and back would silently reset its position to whatever
    # was last snapshotted -- corrupting its KV-cache read offset.
    state["pos"] = rt.pos


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    rt.device_cache = True  # dispatch _moe -> _moe_dev, same as V6

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    ids_by_seq_gt = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]

    # ================= Phase 2: correctness gate =================
    # Ground truth: N independent, unswapped, sequential single-sequence runs
    # (rt.reset() between them, one at a time, no interleaving).
    ground_truth_tokens = [[] for _ in range(N)]
    for s in range(N):
        rt.reset()
        nxt = None
        for t in ids_by_seq_gt[s]:
            nxt = int(rt.step(int(t)))
        ground_truth_tokens[s].append(nxt)
        for _ in range(DECODE_STEPS - 1):
            nxt = int(rt.step(nxt))
            ground_truth_tokens[s].append(nxt)
    cp.cuda.Device(0).synchronize()

    # Swapped path: the SAME interleaving pattern phase 3 will use for real --
    # swap to sequence s, step once, swap away, repeat -- specifically because
    # a bug like "pos silently resets on swap-back" only manifests under
    # interleaving, not a single uninterrupted run. Fresh cache (invalidates
    # nothing shared) so this does not depend on residue from ground truth.
    rt.enable_cache(72)
    state = [snapshot_state(rt) for _ in range(N)]
    cur = [None] * N
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq_gt[s]:
            nxt = int(rt.step(int(t)))
        cur[s] = nxt
        save_state(rt, state[s])

    swapped_tokens = [[] for _ in range(N)]
    for s in range(N):
        swapped_tokens[s].append(cur[s])
    for _ in range(DECODE_STEPS - 1):
        for s in range(N):
            use_state(rt, state[s])
            cur[s] = int(rt.step(cur[s]))
            save_state(rt, state[s])
            swapped_tokens[s].append(cur[s])
    cp.cuda.Device(0).synchronize()

    equivalence_pass = (ground_truth_tokens == swapped_tokens)
    if not equivalence_pass:
        payload = {
            "kind": "proto_multi_seq_full_model",
            "created_utc": utc_now(),
            "phase_reached": "phase2_correctness_gate_FAILED",
            "ground_truth_tokens": ground_truth_tokens,
            "swapped_tokens": swapped_tokens,
            "note": "state-swap mechanism did NOT reproduce independent single-sequence runs bit-exact under interleaved swapping -- a buffer was likely missed in STATE_ATTRS, or reassigned (not mutated in place) the same way pos was. Stopping before any timing claim.",
        }
        out = REPO / "pro_research" / "proto_multi_seq_full_model.json"
        write_json_atomic(out, payload, archive=False)
        print(payload)
        return 1

    print(f"Phase 2 PASS: swapped-state single-sequence run bit-exact matches ground truth ({DECODE_STEPS} tokens)")

    # ================= Phase 3a: N=1 control, SAME code path, SAME config =================
    # One variable at a time: measure a single sequence through the exact same
    # swap mechanism, timing loop, and cache-reset pattern as phase 3b below,
    # so the N=2 comparison isolates N as the only difference (not a different
    # script, different warmup, or a different config like V6's own 47.41
    # tok/s number, which additionally has graph residency + selective ERVF +
    # batched kernels this proto does not use).
    rt.enable_cache(72)
    ids_solo = tok.encode(PROMPTS[0], add_special_tokens=False)
    state_solo = snapshot_state(rt)
    use_state(rt, state_solo)
    rt.pos = 0
    nxt = None
    for t in ids_solo:
        nxt = int(rt.step(int(t)))
    save_state(rt, state_solo)
    cp.cuda.Device(0).synchronize()

    e0s, e1s = cp.cuda.Event(), cp.cuda.Event()
    e0s.record()
    for _ in range(DECODE_STEPS):
        use_state(rt, state_solo)
        nxt = int(rt.step(nxt))
        save_state(rt, state_solo)
    e1s.record()
    e1s.synchronize()
    solo_total_ms = cp.cuda.get_elapsed_time(e0s, e1s)
    solo_ms_per_token = solo_total_ms / DECODE_STEPS
    solo_tok_s = 1000.0 / solo_ms_per_token

    # ================= Phase 3b: real N=2 naive aggregate baseline =================
    # Fresh cache again so phase 3b's timing does not inherit phase 3a's warm state.
    rt.enable_cache(72)
    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]
    state = [snapshot_state(rt) for _ in range(N)]
    cur_token = [None] * N

    # prefill each sequence's prompt through its own state (sequential, not timed).
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        cur_token[s] = nxt
        save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()

    # timed decode: N sequences, DECODE_STEPS real steps each, real production
    # step() per sequence via the verified swap mechanism.
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    tokens_by_seq = [[] for _ in range(N)]
    for _ in range(DECODE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            cur_token[s] = int(rt.step(cur_token[s]))
            save_state(rt, state[s])
            tokens_by_seq[s].append(cur_token[s])
    e1.record()
    e1.synchronize()
    total_ms = cp.cuda.get_elapsed_time(e0, e1)

    total_real_tokens = N * DECODE_STEPS
    ms_per_token_aggregate = total_ms / total_real_tokens
    aggregate_tok_s = 1000.0 / ms_per_token_aggregate
    ms_per_token_per_sequence_equiv = total_ms / DECODE_STEPS  # wall time per decode "tick" across all N

    payload = {
        "kind": "proto_multi_seq_full_model",
        "created_utc": utc_now(),
        "phase_reached": "phase3_naive_baseline_measured",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps_per_sequence": DECODE_STEPS,
        "phase2_equivalence_pass": equivalence_pass,
        "phase2_ground_truth_tokens": ground_truth_tokens,
        "note": "naive baseline: N real sequences via state-swap, real production step() per sequence, MoE device-cache SHARED as an ordinary side effect of not swapping self.cache/self._dev_cache (warm-cache reuse only, NOT the explicit union-fed sharing mechanism proven separately in proto_batch_moe_layer_combined.py -- that integration is a separate follow-up)",
        "solo_n1_same_config_ms_per_token": solo_ms_per_token,
        "solo_n1_same_config_tok_s": solo_tok_s,
        "total_wall_ms_for_all_real_tokens": total_ms,
        "total_real_tokens_across_all_sequences": total_real_tokens,
        "ms_per_real_token_aggregate": ms_per_token_aggregate,
        "aggregate_tok_s": aggregate_tok_s,
        "aggregate_speedup_vs_solo": aggregate_tok_s / solo_tok_s if solo_tok_s else None,
        "wall_ms_per_decode_tick_across_n_sequences": ms_per_token_per_sequence_equiv,
        "tokens_by_sequence": tokens_by_seq,
    }
    out = REPO / "pro_research" / "proto_multi_seq_full_model.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
