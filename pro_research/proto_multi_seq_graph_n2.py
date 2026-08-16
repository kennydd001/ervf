"""Directly tests today's own root-cause finding (agents/RESEARCH_NOTEBOOK.md
2026-08-16, "N=8 naive baseline" + follow-ups): the N=8 naive-sharing
collapse (0.243-0.256x, confirmed physically real via independent dual-
method timing, not a measurement artifact) traces to Python-orchestration/
kernel-launch overhead scaling with N -- NOT caching (N=8's hit rate was
HIGHER than solo, ruling that out). The established, ALREADY-PROVEN remedy
for launch overhead in this exact codebase is CUDA-graph residency
(setup_graph()/step_graph()/_step_body_graph(), used successfully by V4-V6
this session, reaching the 47.41 tok/s single-stream record -- the stale
"NOT YET RUN" comment on that code in runtime.py predates V4's own
successful adoption of it and should not be trusted over the actual V4-V6
measurement history).

This captures ONE separate CUDA graph per sequence (each bound to that
sequence's own buffer addresses -- setup_graph() captures using whatever is
currently active on `rt`, so use_state(rt, state[s]) before setup_graph()
binds graph s to state[s]'s own memory), replayed via step_graph() in an
interleaved N=2 decode loop. No cross-sequence MoE cache sharing in this
first version (each sequence's graph has its own frozen device-cache,
exactly like solo V4/V6) -- this isolates whether graph residency ALONE
(no explicit sharing) recovers the naive approach's N-scaling loss.

setup_graph() allocates its own per-instance state (_tok_dev, _pos_dev,
_am_max, _am_idx, _graph, _graph_stream, pinned staging/ring buffers) not
in the STATE_ATTRS list used by every other multi-sequence prototype this
session -- STATE_ATTRS_GRAPH extends it. enable_cache() invalidates any
existing graph (rebinds cache pointers), so per sequence the order must be:
_alloc_state() -> enable_cache() -> setup_graph() -> snapshot everything,
exactly once per sequence, never enable_cache() again afterward.

Phase 1: build the graph-aware snapshot/swap mechanism.
Phase 2: correctness gate -- N=2, full interleaved decode via step_graph(),
         compared bit-exact against independent single-sequence step_graph()
         ground truth. Must pass before any timing claim.
Phase 3: if phase 2 passes, measure real aggregate tok/s and compare against
         the naive (non-graph) N=2 baseline (31.66 tok/s robust) and the
         explicit-sharing baseline (11.23 tok/s).

Not a gated PRO experiment -- a scoped integration prototype, real CUDA
engineering territory (per agents/PATH_TO_100_TOKS.md's own roadmap item 2)
attempted carefully with this session's established correctness discipline.
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

STATE_ATTRS = [
    "ssm", "conv", "kc", "vc", "kv_dim", "pos",
    "h", "tmp", "acc", "normed", "act", "_act_moe", "_act_shared",
    "proj", "convo", "dt", "y", "gn", "qv", "kv_", "vv", "ctx",
    "logits", "rlog", "route_pack",
    "stage_c", "stage_s", "mstate", "contrib",
    "copy_stream", "evt", "part_acc", "part_ml",
]

# Extra per-instance state setup_graph() allocates, read directly from
# runtime.py's own source (lines 841-918) -- not guessed.
GRAPH_ATTRS = [
    "_tok_dev", "_pos_dev", "_am_max", "_am_idx",
    "_graph", "_graph_stream",
    "_stage_mem", "_stage_np", "_stage_i", "_ring_mem", "_ring_np", "_ring_i",
    "_embed_pinned", "_embed_tbl_ptr",
    "graph_extra_vram_bytes",
]

# The cache is per-sequence too, and holding the reference is LOAD-BEARING,
# not bookkeeping: each captured graph binds its cache and _dev_cache buffers
# by raw pointer, but replay never executes Python, so the only thing keeping
# those device buffers alive (and NOT reused by the CuPy pool for the NEXT
# sequence's cache -- same sizes, same allocation order, so the pool hands
# back the exact same addresses, silently making both graphs share memory)
# is a live Python reference. Without this, sequence 1's build frees sequence
# 0's cache buffers and sequence 1's cache is allocated over them; replaying
# the two graphs then corrupts both sequences' routing/LRU state. This was
# the phase-2 failure of 2026-08-16 (seq0 diverged at decode token 2, i.e.
# immediately after seq1's first replay).
CACHE_ATTRS = ["cache", "_dev_cache", "cache_mode", "cache_stats"]

N = 2
DECODE_STEPS = 20

# VRAM reality check (found before first run, by arithmetic from runtime.py's
# own formulas): one cap-72 cache is 72 x 2,806,272 B x 23 layers = 4.33 GiB,
# so the docstring's "each sequence its own cap-72 cache exactly like solo"
# needs 8.66 GiB of cache alone on an 8151 MiB GPU -- impossible. Capacity
# changes are numerically invariant (E1F21-INV proved hit/miss cannot change
# tokens), so each sequence gets cap 24 (2 x 1.45 GiB; fits with headroom for
# weights + two graphs). Consequence for the phase-3 comparison: the graph
# arm runs with LESS total cache than the naive N=2 baseline (one shared
# cap-72) and much less than solo V6 -- it faces more PCIe misses per token,
# which makes the overhead-isolation question conservative, not optimistic.
CACHE_CAP_PER_SEQ = 24


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
    state["pos"] = rt.pos
    state["_ring_i"] = rt._ring_i


def build_graph_state(rt):
    """_alloc_state() -> enable_cache() -> setup_graph(), then snapshot
    everything (dynamic + graph-specific) for this one sequence. Must be
    the LAST thing done before this sequence's state is considered ready --
    enable_cache() after this point would invalidate the captured graph."""
    rt._alloc_state()
    rt.enable_cache(CACHE_CAP_PER_SEQ)
    rt.setup_graph()
    return {name: getattr(rt, name) for name in STATE_ATTRS + GRAPH_ATTRS + CACHE_ATTRS}


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    rt.device_cache = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]

    # ================= Phase 2: correctness gate =================
    # Ground truth: N independent, unswapped, sequential graph runs -- each
    # sequence gets a completely fresh rt (own graph, own cache, own state),
    # matching what solo V4/V6 already proved correct, run one at a time.
    ground_truth_tokens = [[] for _ in range(N)]
    for s in range(N):
        rt._alloc_state()
        rt.enable_cache(CACHE_CAP_PER_SEQ)
        rt.setup_graph()
        launches = 0
        for t in ids_by_seq[s]:
            rt.step_graph(int(t))
            rt._graph_stream.synchronize()  # V4 staging discipline: the
            launches += 1                   # staged H2D must land first
        for _ in range(DECODE_STEPS):
            rt.step_graph(None)
            launches += 1
        all_ids = rt.ring_harvest(0, launches)
        # ring slot j holds the id PRODUCED by launch j; the first REAL
        # generated token (from the last prompt-staging launch) is at
        # index len(prompt)-1, matching ring_harvest's own docstring.
        ground_truth_tokens[s] = all_ids[len(ids_by_seq[s]) - 1: len(ids_by_seq[s]) - 1 + DECODE_STEPS]

    # Graph-swapped path: build N per-sequence graphs, interleave step_graph()
    # calls, harvest at the end -- the actual multi-sequence mechanism.
    state = [build_graph_state(rt) for _ in range(N)]
    launches_per_seq = [0] * N
    for s in range(N):
        use_state(rt, state[s])
        for t in ids_by_seq[s]:
            rt.step_graph(int(t))
            rt._graph_stream.synchronize()  # V4 staging discipline
            launches_per_seq[s] += 1
        save_state(rt, state[s])

    for _ in range(DECODE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            rt.step_graph(None)
            launches_per_seq[s] += 1
            save_state(rt, state[s])

    swapped_tokens = [[] for _ in range(N)]
    for s in range(N):
        use_state(rt, state[s])
        all_ids_s = rt.ring_harvest(0, launches_per_seq[s])
        prompt_len = len(ids_by_seq[s])
        swapped_tokens[s] = all_ids_s[prompt_len - 1: prompt_len - 1 + DECODE_STEPS]
        save_state(rt, state[s])

    equivalence_pass = (ground_truth_tokens == swapped_tokens)
    if not equivalence_pass:
        payload = {
            "kind": "proto_multi_seq_graph_n2",
            "created_utc": utc_now(),
            "phase_reached": "phase2_correctness_gate_FAILED",
            "ground_truth_tokens": ground_truth_tokens,
            "swapped_tokens": swapped_tokens,
            "note": "per-sequence CUDA-graph capture/replay did NOT reproduce independent graph runs bit-exact under interleaving. Stopping before any timing claim.",
        }
        out = REPO / "pro_research" / "proto_multi_seq_graph_n2.json"
        write_json_atomic(out, payload, archive=False)
        print(payload)
        return 1

    print(f"Phase 2 PASS: per-sequence graph capture/replay bit-exact matches independent ground truth ({DECODE_STEPS} tokens x {N} sequences)")

    # ================= Phase 3: timing =================
    # Drop phase 2's states (and their caches/graphs) BEFORE building fresh
    # ones: CACHE_ATTRS now keeps those buffers alive, so building the new
    # set first would transiently double the cache VRAM and OOM.
    del state
    cp.get_default_memory_pool().free_all_blocks()
    state = [build_graph_state(rt) for _ in range(N)]
    launches_per_seq = [0] * N
    for s in range(N):
        use_state(rt, state[s])
        for t in ids_by_seq[s]:
            rt.step_graph(int(t))
            rt._graph_stream.synchronize()  # V4 staging discipline
            launches_per_seq[s] += 1
        save_state(rt, state[s])

    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(DECODE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            rt.step_graph(None)
            launches_per_seq[s] += 1
            save_state(rt, state[s])
    # harvest requires a sync per sequence's own ring/stream -- included in
    # the timed region since it's part of getting real results out, same as
    # every other script's e1.synchronize() being part of the measurement.
    tokens_by_seq = [[] for _ in range(N)]
    for s in range(N):
        use_state(rt, state[s])
        all_ids_s = rt.ring_harvest(0, launches_per_seq[s])
        prompt_len = len(ids_by_seq[s])
        tokens_by_seq[s] = all_ids_s[prompt_len - 1: prompt_len - 1 + DECODE_STEPS]
    e1.record()
    e1.synchronize()
    total_ms = cp.cuda.get_elapsed_time(e0, e1)

    total_real_tokens = N * DECODE_STEPS
    ms_per_token_aggregate = total_ms / total_real_tokens
    aggregate_tok_s = 1000.0 / ms_per_token_aggregate

    payload = {
        "kind": "proto_multi_seq_graph_n2",
        "created_utc": utc_now(),
        "phase_reached": "phase3_timing_measured",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps": DECODE_STEPS,
        "phase2_equivalence_pass": equivalence_pass,
        "note": "per-sequence CUDA-graph capture/replay (no cross-sequence MoE sharing; each sequence has its own device-cache at cap %d -- NOT the solo cap-72, because 2x4.33 GiB cannot fit in 8151 MiB; capacity is numerically invariant per E1F21-INV so tokens are unaffected, but the graph arm runs with less cache than the naive N=2 baseline's shared cap-72, making this timing comparison conservative) -- tests whether graph residency alone recovers the Python/launch-overhead loss the naive (non-graph) approach showed at larger N. Compare against naive N=2 (31.66 tok/s robust, 40 steps) and explicit-sharing (11.23 tok/s)." % CACHE_CAP_PER_SEQ,
        "total_wall_ms_for_all_real_tokens": total_ms,
        "total_real_tokens_across_all_sequences": total_real_tokens,
        "ms_per_real_token_aggregate": ms_per_token_aggregate,
        "aggregate_tok_s": aggregate_tok_s,
        "tokens_by_sequence": tokens_by_seq,
    }
    out = REPO / "pro_research" / "proto_multi_seq_graph_n2.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
