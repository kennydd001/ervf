"""Follow-up to proto_multi_seq_graph_n2{,_shared}.py (2026-08-16): the
shared-cache graph pair (cap 64, one stream, 33.52 tok/s) was SLOWER than
the private-cache pair (cap 24 each, 36.86 tok/s) despite the bigger cache.
Both passed the bit-exact gate, so the difference is cost, not correctness.
Two hypotheses were on the table: (a) two alternating working sets thrash
the shared LRU (effective ~cap/2 per sequence, worse than a dedicated 24),
(b) the single shared stream serializes what two private streams could
overlap. Hypothesis (a) is directly measurable: cache_assign maintains
device-side per-layer counters (stats2[0]=hits, stats2[1]=misses), so this
script runs each configuration and dumps the counters.

Configurations (same two prompts, 20 decode steps, interleaved where N=2):
  solo_cap72 / solo_cap64 / solo_cap24 -- capacity-only references
  private_n2 (cap 24 x 2, own streams) -- the 36.86 tok/s arm
  shared_n2  (cap 64, one stream)      -- the 33.52 tok/s arm

Integrity: the N=2 arms must reproduce the ground-truth tokens recorded in
proto_multi_seq_graph_n2.json / _shared.json (they are deterministic); a
mismatch means this diagnostic measured something else and is void.

Not a gated experiment -- a diagnostic. No tok/s claims here; component
counters only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, require_model_dir, utc_now, write_json_atomic

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
CACHE_ATTRS = ["cache", "_dev_cache", "cache_mode", "cache_stats"]

DECODE_STEPS = 20


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
    state["pos"] = rt.pos
    state["_ring_i"] = rt._ring_i


def build_state(rt, cap, enable):
    rt._alloc_state()
    if enable:
        rt.enable_cache(cap)
    else:
        rt._graph = None  # defeat setup_graph's early-return (shared cache)
    rt.setup_graph()
    return {name: getattr(rt, name) for name in STATE_ATTRS + GRAPH_ATTRS + CACHE_ATTRS}


def run_decode(rt, states, ids_by_seq, shared_stream=None):
    """Prompt staging (sync per token, V4 discipline) + interleaved decode.
    Returns tokens per sequence (for the integrity check)."""
    n = len(states)
    launches = [0] * n
    for s in range(n):
        use_state(rt, states[s])
        for t in ids_by_seq[s]:
            rt.step_graph(int(t))
            rt._graph_stream.synchronize()
            launches[s] += 1
        save_state(rt, states[s])
    for _ in range(DECODE_STEPS):
        for s in range(n):
            use_state(rt, states[s])
            rt.step_graph(None)
            launches[s] += 1
            save_state(rt, states[s])
    tokens = []
    for s in range(n):
        use_state(rt, states[s])
        all_ids = rt.ring_harvest(0, launches[s])
        pl = len(ids_by_seq[s])
        tokens.append(all_ids[pl - 1: pl - 1 + DECODE_STEPS])
        save_state(rt, states[s])
    return tokens


def cache_counters(state):
    """Sum stats2 (hits, misses) over all MoE layers of one state's
    _dev_cache. One sync at the very end -- fine for a diagnostic."""
    hits = misses = 0
    for layer, dev in state["_dev_cache"].items():
        v = dev["stats2"].get()  # device->host, syncs
        hits += int(v[0])
        misses += int(v[1])
    return {"hits": hits, "misses": misses,
            "hit_rate": hits / max(1, hits + misses)}


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
    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]

    # Integrity references: tokens recorded by the gated prototypes.
    ref_private = json.load(open(REPO / "pro_research" / "proto_multi_seq_graph_n2.json",
                                 encoding="utf-8"))["tokens_by_sequence"]
    ref_shared = json.load(open(REPO / "pro_research" / "proto_multi_seq_graph_n2_shared.json",
                                encoding="utf-8"))["tokens_by_sequence"]

    results = {}

    def clear_rt():
        """Drop every big reference rt holds, then free the pool. Without
        this the NEXT config's allocations overlap the previous config's
        still-referenced buffers (cap-72 solo + leftovers > 8151 MiB)."""
        for name in STATE_ATTRS + GRAPH_ATTRS + CACHE_ATTRS:
            if hasattr(rt, name):
                setattr(rt, name, None)
        cp.get_default_memory_pool().free_all_blocks()

    # ---- solo capacity references -------------------------------------
    for cap in (72, 64, 24):
        st = build_state(rt, cap, enable=True)
        toks = run_decode(rt, [st], [ids_by_seq[0]])
        assert toks[0] == ref_private[0], \
            f"solo cap{cap} diverged from gated tokens -- diagnostic void"
        results[f"solo_cap{cap}"] = cache_counters(st)
        del st
        clear_rt()

    # ---- private N=2 (cap 24 each, own streams) ------------------------
    states = [build_state(rt, 24, enable=True) for _ in range(2)]
    toks = run_decode(rt, states, ids_by_seq)
    assert toks == ref_private, "private arm diverged from gated prototype -- diagnostic void"
    results["private_n2"] = {f"seq{s}": cache_counters(states[s]) for s in range(2)}
    del states
    clear_rt()

    # ---- shared N=2 (cap 64, one stream) -------------------------------
    rt._alloc_state()
    rt.enable_cache(64)
    rt.setup_graph()
    shared_stream = rt._graph_stream
    st0 = {name: getattr(rt, name) for name in STATE_ATTRS + GRAPH_ATTRS + CACHE_ATTRS}
    st0["_graph_stream"] = shared_stream
    st1 = build_state(rt, 64, enable=False)  # shares rt.cache / rt._dev_cache
    st1["_graph_stream"] = shared_stream
    states = [st0, st1]
    toks = run_decode(rt, states, ids_by_seq, shared_stream=shared_stream)
    assert toks == ref_shared, "shared arm diverged from gated prototype -- diagnostic void"
    results["shared_n2"] = cache_counters(states[0])  # one shared _dev_cache

    payload = {
        "kind": "diag_n2_graph_cache_hitrates",
        "created_utc": utc_now(),
        "decode_steps": DECODE_STEPS,
        "note": "device-side cache_assign counters (stats2: hits/misses) per "
                "configuration; explains why shared cap-64 one-stream N=2 "
                "(33.52 tok/s) was slower than private cap-24x2 (36.86 tok/s) "
                "despite the bigger cache. Integrity: both N=2 arms "
                "reproduced the gated prototypes' tokens exactly.",
        "results": results,
    }
    write_json_atomic(REPO / "pro_research" / "diag_n2_graph_cache_hitrates.json",
                      payload, archive=False)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
