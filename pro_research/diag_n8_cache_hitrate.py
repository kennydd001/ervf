"""Tests the concrete hypothesis for proto_multi_seq_full_model_n8.py's
severe collapse (0.253x, 4x slower than solo) rather than continuing to
speculate: is it cache THRASHING? The naive mechanism doesn't swap
rt.cache/rt._dev_cache when interleaving sequences (they're deliberately
NOT in STATE_ATTRS, so they stay shared as a side effect) -- with N=8
sequences interleaved round-robin, each sequence's own routing decisions
get pushed into the SAME shared per-layer LRU, evicted by recency. If 8
sequences with substantially different prompt content route to
substantially different experts, by the time sequence 0 gets its next turn
(7 OTHER sequences' cache_assign calls later), the LRU may have evicted
everything it needs -- meaning near-zero hit rate despite (or because of)
sharing, unlike the genuine benefit diag_batch_warm_cache.py found for a
SMALLER N=4 in isolation (not through this same interleaved-real-step
mechanism).

_moe_dev's own device cache already tracks hits/misses on-device
(dev["stats2"]: [0]=hits, [1]=misses, accumulated by cache_assign's kernel
across calls) -- this reads that back directly after a real N=8 run instead
of guessing, extending the SAME verified state-swap mechanism (no new
correctness risk: this only ADDS a read of an existing counter, changes
nothing about what gets computed).

Not a gated PRO experiment -- a root-cause diagnostic, read-only.
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
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "In the quiet village, the old fisherman noticed something strange about the tide",
    "The quarterly earnings report showed a significant increase in revenue driven by",
    "Photosynthesis is the process by which plants convert light energy into",
    "The defendant's attorney argued that the evidence presented by the prosecution was",
    "To configure the network firewall, first navigate to the settings panel and",
]

STATE_ATTRS = [
    "ssm", "conv", "kc", "vc", "kv_dim", "pos",
    "h", "tmp", "acc", "normed", "act", "_act_moe", "_act_shared",
    "proj", "convo", "dt", "y", "gn", "qv", "kv_", "vv", "ctx",
    "logits", "rlog", "route_pack",
    "stage_c", "stage_s", "mstate", "contrib",
    "copy_stream", "evt", "part_acc", "part_ml",
]

N = 8
DECODE_STEPS = 30


def snapshot_state(rt):
    rt._alloc_state()
    return {name: getattr(rt, name) for name in STATE_ATTRS}


def use_state(rt, state):
    for name, value in state.items():
        setattr(rt, name, value)


def save_state(rt, state):
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
    rt.device_cache = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]

    # ---- N=1 solo control: same cache=72, same steps, own stats2 baseline.
    rt.enable_cache(72)
    ids_solo = tok.encode(PROMPTS[0], add_special_tokens=False)
    state_solo = snapshot_state(rt)
    use_state(rt, state_solo)
    rt.pos = 0
    nxt = None
    for t in ids_solo:
        nxt = int(rt.step(int(t)))
    save_state(rt, state_solo)
    for _ in range(DECODE_STEPS):
        use_state(rt, state_solo)
        nxt = int(rt.step(nxt))
        save_state(rt, state_solo)
    cp.cuda.Device(0).synchronize()

    solo_hits = solo_misses = 0
    for i in moe_layers:
        if i in rt._dev_cache:
            stats = cp.asnumpy(rt._dev_cache[i]["stats2"])
            solo_hits += int(stats[0])
            solo_misses += int(stats[1])

    # ---- N=8 naive: identical mechanism to proto_multi_seq_full_model_n8.py,
    # same cache=72, real interleaved decode, then read stats2 back.
    rt.enable_cache(72)
    ids_by_seq = [tok.encode(p, add_special_tokens=False) for p in PROMPTS[:N]]
    state = [snapshot_state(rt) for _ in range(N)]
    cur_token = [None] * N
    for s in range(N):
        use_state(rt, state[s])
        rt.pos = 0
        nxt = None
        for t in ids_by_seq[s]:
            nxt = int(rt.step(int(t)))
        cur_token[s] = nxt
        save_state(rt, state[s])

    for _ in range(DECODE_STEPS):
        for s in range(N):
            use_state(rt, state[s])
            cur_token[s] = int(rt.step(cur_token[s]))
            save_state(rt, state[s])
    cp.cuda.Device(0).synchronize()

    n8_hits = n8_misses = 0
    per_layer = {}
    for i in moe_layers:
        if i in rt._dev_cache:
            stats = cp.asnumpy(rt._dev_cache[i]["stats2"])
            h, m = int(stats[0]), int(stats[1])
            n8_hits += h
            n8_misses += m
            per_layer[str(i)] = {"hits": h, "misses": m}

    solo_total = solo_hits + solo_misses
    n8_total = n8_hits + n8_misses
    solo_hit_rate = solo_hits / solo_total if solo_total else None
    n8_hit_rate = n8_hits / n8_total if n8_total else None

    payload = {
        "kind": "diag_n8_cache_hitrate",
        "created_utc": utc_now(),
        "note": "reads back _dev_cache's own on-device hit/miss counters (dev['stats2'], accumulated by cache_assign's kernel) after a real N=1 solo run and a real N=8 naive interleaved run, same mechanism as proto_multi_seq_full_model_n8.py, to test the cache-thrashing hypothesis for that script's 0.253x collapse directly instead of guessing",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences_n8": N,
        "decode_steps": DECODE_STEPS,
        "solo_hits": solo_hits,
        "solo_misses": solo_misses,
        "solo_hit_rate": solo_hit_rate,
        "n8_hits": n8_hits,
        "n8_misses": n8_misses,
        "n8_hit_rate": n8_hit_rate,
        "n8_hit_rate_per_layer": per_layer,
        "thrashing_hypothesis_supported": (n8_hit_rate is not None and solo_hit_rate is not None
                                           and n8_hit_rate < solo_hit_rate - 0.15),
    }
    out = REPO / "pro_research" / "diag_n8_cache_hitrate.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
