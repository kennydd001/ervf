"""Closes risk #3 in agents/BATCH_ARCHITECTURE_DESIGN.md: every batch>1
measurement so far (diag_cross_sequence_union.py, diag_batch_warm_cache.py,
both proto_batch_*.py) compared N sequences at the SAME step index -- but a
real continuous-batching serving runtime has sequences at independent,
staggered positions (sequence A generating token 40, sequence B token 5, at
the same wall-clock batch tick). This measures whether the expert-union
sharing benefit survives that more realistic condition, holding N/T/prompts
fixed and varying only lockstep-vs-staggered (one variable).

Not a gated PRO experiment.
"""

from __future__ import annotations

import sys
import types
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

N = 4
T = 30            # wall-clock batch ticks compared
TOP_K = 6
# staggered offsets: sequence i is this many steps "further along" than
# sequence 0 at wall-tick 0 -- deterministic, not random, so the run is
# reproducible; chosen to spread across a realistic decode-depth range.
OFFSETS = [0, 7, 15, 23]
MAX_OFFSET = max(OFFSETS)


def route_capture(rt, cp, target_layer, prompt, steps):
    captured = []
    orig_route = rt._route_device

    def capture_route(i, _captured=captured):
        packed = orig_route(i)
        if i == target_layer:
            _captured.append([int(x) for x in cp.asnumpy(packed)[:TOP_K]])
        return packed

    rt._route_device = types.MethodType(lambda self, i, cr=capture_route: cr(i), rt)
    from transformers import AutoTokenizer
    tok = route_capture.tok
    ids = tok.encode(prompt, add_special_tokens=False)
    rt.reset()
    nxt = None
    for t in ids:
        nxt = int(rt.step(int(t)))
    cur = nxt
    for _ in range(steps):
        cur = int(rt.step(cur, capture_routes=None))
    rt._route_device = orig_route
    return captured[-steps:]


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True

    from transformers import AutoTokenizer
    route_capture.tok = AutoTokenizer.from_pretrained(
        str(require_model_dir()), local_files_only=True, trust_remote_code=True, use_fast=True)

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    target_layer = moe_layers[10]

    # capture T + MAX_OFFSET real steps per sequence -- enough to build both
    # the lockstep view (all use steps[0:T]) and the staggered view (sequence
    # i uses steps[offset_i : offset_i+T]) from the SAME underlying route
    # trajectories, so lockstep vs staggered differ only in which slice of
    # each sequence's own real trajectory is compared at a given wall-tick.
    steps_needed = T + MAX_OFFSET
    routes_by_seq = []
    for prompt in PROMPTS[:N]:
        routes_by_seq.append(route_capture(rt, cp, target_layer, prompt, steps_needed))
    cp.cuda.Device(0).synchronize()

    if any(len(r) < steps_needed for r in routes_by_seq):
        print("insufficient captured steps:", [len(r) for r in routes_by_seq])
        return 1

    def union_size(ids_lists):
        s = set()
        for ids in ids_lists:
            s.update(ids)
        return len(s)

    lockstep_unions = []
    staggered_unions = []
    for t in range(T):
        lockstep_view = [routes_by_seq[s][t] for s in range(N)]
        staggered_view = [routes_by_seq[s][OFFSETS[s] + t] for s in range(N)]
        lockstep_unions.append(union_size(lockstep_view))
        staggered_unions.append(union_size(staggered_view))

    max_union = N * TOP_K
    lockstep_mean = sum(lockstep_unions) / T
    staggered_mean = sum(staggered_unions) / T

    payload = {
        "kind": "diag_staggered_position_union",
        "created_utc": utc_now(),
        "note": "read-only diagnostic; closes design-doc risk #3 -- tests whether cross-sequence expert-union sharing survives sequences at independent, staggered decode positions (vs. the lockstep-same-step-index assumption every prior measurement used)",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "target_layer": target_layer,
        "n_sequences": N,
        "wall_ticks_compared": T,
        "top_k": TOP_K,
        "offsets": OFFSETS,
        "max_possible_union": max_union,
        "lockstep_unions_per_tick": lockstep_unions,
        "staggered_unions_per_tick": staggered_unions,
        "lockstep_mean_union": lockstep_mean,
        "staggered_mean_union": staggered_mean,
        "lockstep_pct_of_max": lockstep_mean / max_union,
        "staggered_pct_of_max": staggered_mean / max_union,
        "delta_staggered_minus_lockstep_pct_points": (staggered_mean - lockstep_mean) / max_union * 100.0,
    }
    out = REPO / "pro_research" / "diag_staggered_position_union.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
