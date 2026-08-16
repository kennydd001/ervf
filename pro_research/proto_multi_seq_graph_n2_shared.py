"""Shared-cache multi-sequence graph N=2 -- the follow-up the cap-24 result
pointed to.

proto_multi_seq_graph_n2.py (same day) showed: per-sequence graphs with
PRIVATE per-sequence caches are bit-exact but slow (23.59 tok/s aggregate) --
private caches are VRAM-impossible at solo size (2 x 4.33 GiB > 8151 MiB) and
the forced cap-24 split costs more PCIe misses than the removed launch
overhead pays back. Conclusion drawn there: the multi-sequence graph must
SHARE the device cache (which is also what unlocks the cross-sequence union
benefit later).

This variant: ONE shared cache (cap 64 -- cap 72 leaves no room for the second
sequence's dynamic state; cap 64 frees ~0.48 GiB, numerically invariant per
E1F21-INV) bound into BOTH captured graphs, with both graphs replayed on ONE
shared stream. The single stream is not optional: both graphs read AND write
the shared _dev_cache LRU tables (ids/w/slots/need) and cache slots, so
concurrent replay on two streams would be a data race. Same-stream launches
serialize the graphs completely -- no race, and the comparison becomes: does
a shared-cache graph pair beat the naive eager N=2 baseline (31.66 tok/s
robust, 40 steps, one shared cap-72 cache)?

Phase 2: correctness gate -- interleaved graph decode vs independent
         solo-graph ground truth (capacity-invariant, so solo runs with their
         own cap-64 caches produce the same tokens). Must pass before timing.
Phase 3: real aggregate tok/s.
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

GRAPH_ATTRS = [
    "_tok_dev", "_pos_dev", "_am_max", "_am_idx",
    "_graph", "_graph_stream",
    "_stage_mem", "_stage_np", "_stage_i", "_ring_mem", "_ring_np", "_ring_i",
    "_embed_pinned", "_embed_tbl_ptr",
    "graph_extra_vram_bytes",
]

N = 2
DECODE_STEPS = 20
# cap 72 fits exactly once on this GPU (V6: 0 MiB free at capture). The second
# sequence's dynamic state (~150-200 MiB) plus a second graph instance need
# room, so the shared cache is cap 64 (~0.48 GiB lighter). Capacity is
# numerically invariant (E1F21-INV): tokens are unaffected.
CACHE_CAP_SHARED = 64


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
    state["pos"] = rt.pos
    state["_ring_i"] = rt._ring_i


def build_graph_state_shared(rt):
    """Per-sequence dynamic state + graph, over the ONE shared cache that is
    already active on rt. Never calls enable_cache() -- that would rebind the
    cache buffers and invalidate the other sequence's graph."""
    rt._alloc_state()
    # setup_graph() early-returns when self._graph is not None (runtime.py:
    # "if getattr(self, '_graph', None) is not None: return"). The previous
    # sequence's graph stays alive through its state snapshot, so clearing
    # the attribute here is safe and is what makes setup_graph() actually
    # capture a SECOND graph over the shared cache instead of silently
    # snapshotting sequence 0's graph/token/ring buffers for sequence 1.
    rt._graph = None
    rt.setup_graph()
    return {name: getattr(rt, name) for name in STATE_ATTRS + GRAPH_ATTRS}


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
    # Ground truth: independent solo graph runs, each with its own cache
    # (capacity-invariance makes the tokens identical to the shared case).
    ground_truth_tokens = [[] for _ in range(N)]
    for s in range(N):
        rt._alloc_state()
        rt.enable_cache(CACHE_CAP_SHARED)
        rt.setup_graph()
        launches = 0
        for t in ids_by_seq[s]:
            rt.step_graph(int(t))
            rt._graph_stream.synchronize()  # V4 staging discipline
            launches += 1
        for _ in range(DECODE_STEPS):
            rt.step_graph(None)
            launches += 1
        all_ids = rt.ring_harvest(0, launches)
        ground_truth_tokens[s] = all_ids[len(ids_by_seq[s]) - 1: len(ids_by_seq[s]) - 1 + DECODE_STEPS]

    # Swapped path: ONE shared cache, two graphs, ONE shared launch stream.
    # Sequence 0 IS the state the cache-warming setup_graph() just captured --
    # building it again would double-capture for nothing.
    rt._alloc_state()
    rt.enable_cache(CACHE_CAP_SHARED)
    rt.setup_graph()  # creates the shared stream and the _dev_cache tables
    shared_stream = rt._graph_stream
    state = [{name: getattr(rt, name) for name in STATE_ATTRS + GRAPH_ATTRS}]
    state[0]["_graph_stream"] = shared_stream
    for s in range(1, N):
        st = build_graph_state_shared(rt)
        st["_graph_stream"] = shared_stream  # serialize both graphs: the
        state.append(st)                     # shared _dev_cache races otherwise

    def run_swapped(collect):
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
        if not collect:
            return None
        out_tokens = [[] for _ in range(N)]
        for s in range(N):
            use_state(rt, state[s])
            all_ids_s = rt.ring_harvest(0, launches_per_seq[s])
            prompt_len = len(ids_by_seq[s])
            out_tokens[s] = all_ids_s[prompt_len - 1: prompt_len - 1 + DECODE_STEPS]
            save_state(rt, state[s])
        return out_tokens

    swapped_tokens = run_swapped(collect=True)
    equivalence_pass = (ground_truth_tokens == swapped_tokens)
    if not equivalence_pass:
        payload = {
            "kind": "proto_multi_seq_graph_n2_shared",
            "created_utc": utc_now(),
            "phase_reached": "phase2_correctness_gate_FAILED",
            "cache_cap_shared": CACHE_CAP_SHARED,
            "ground_truth_tokens": ground_truth_tokens,
            "swapped_tokens": swapped_tokens,
            "note": "shared-cache graph pair did NOT reproduce independent graph runs bit-exact under interleaving. Stopping before any timing claim.",
        }
        write_json_atomic(REPO / "pro_research" / "proto_multi_seq_graph_n2_shared.json",
                          payload, archive=False)
        print(payload)
        return 1

    print(f"Phase 2 PASS: shared-cache graph pair bit-exact vs independent ground truth ({DECODE_STEPS} tokens x {N} sequences)")

    # ================= Phase 3: timing =================
    # Fresh states for a clean timed run (same construction as above). Drop
    # phase 2's states first so their graphs/dynamic state don't stack with
    # the new set's on an 8151 MiB GPU.
    del state
    cp.get_default_memory_pool().free_all_blocks()
    rt._alloc_state()
    rt.enable_cache(CACHE_CAP_SHARED)
    rt.setup_graph()
    shared_stream = rt._graph_stream
    state = [{name: getattr(rt, name) for name in STATE_ATTRS + GRAPH_ATTRS}]
    state[0]["_graph_stream"] = shared_stream
    for s in range(1, N):
        st = build_graph_state_shared(rt)
        st["_graph_stream"] = shared_stream
        state.append(st)

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
        "kind": "proto_multi_seq_graph_n2_shared",
        "created_utc": utc_now(),
        "phase_reached": "phase3_timing_measured",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "decode_steps": DECODE_STEPS,
        "cache_cap_shared": CACHE_CAP_SHARED,
        "phase2_equivalence_pass": equivalence_pass,
        "note": "ONE shared device-cache (cap 64, VRAM-forced below 72) bound into BOTH per-sequence graphs; both graphs replayed on ONE shared stream because concurrent replay would race on the shared _dev_cache LRU tables. Tests whether graph residency WITH a realistic cache budget beats naive eager N=2 (31.66 tok/s robust, 40 steps, shared cap-72) and private-cache graph N=2 (23.59 tok/s, cap-24 each, same day).",
        "total_wall_ms_for_all_real_tokens": total_ms,
        "total_real_tokens_across_all_sequences": total_real_tokens,
        "ms_per_real_token_aggregate": ms_per_token_aggregate,
        "aggregate_tok_s": aggregate_tok_s,
        "tokens_by_sequence": tokens_by_seq,
    }
    write_json_atomic(REPO / "pro_research" / "proto_multi_seq_graph_n2_shared.json",
                      payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
