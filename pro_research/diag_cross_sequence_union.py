"""Read-only diagnostic: the first, no-build measurement for the batch>1
hypothesis logged in agents/TODO.md and RESEARCH_NOTEBOOK.md (2026-08-16).

Every optimization this session stayed within batch=1, and the 165 tok/s
roofline ceiling is itself computed under that assumption -- nothing within
batch=1 can exceed it. If N independently-running sequences route to
substantially overlapping sets of experts at a given step, a batched runtime
could load each unique expert once and serve all N sequences from it,
amortizing the dominant PCIe cost across N tokens instead of 1 -- a
different, potentially higher ceiling. If sequences route almost entirely
disjointly, there is no such ceiling to find and building batch>1 support
(a multi-week rearchitecture, not attempted here) would not pay for the
PCIe amortization it promises.

Method: N diverse prompts (different domains, so routing diversity is not
an artifact of similar content), each stepped through the backbone with
capture_routes (LightningRuntime.step(token_id, capture_routes=...), the
same already-existing API the MTP route-union measurement used earlier
today). For each of several matched step indices (e.g. the 5th generated
token of every prompt), compute the union of per-layer top-6 expert
selections across subsets of size N in {2,4,8,16}, versus the no-overlap
baseline of N*6.

Not a gated PRO experiment -- no runtime change, no kernel, no claim beyond
the measured union sizes.
"""

from __future__ import annotations

import sys
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

STEPS_PER_PROMPT = 20
UNION_SIZES = (2, 4, 8, 16)


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    # device_cache stays False: _moe_dev returns (None, None) and would not
    # populate capture_routes (same constraint as the MTP route-union script).

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]

    # routes_by_prompt[p][layer] = list of top-6 id-lists, one per generated step
    routes_by_prompt = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        cur = nxt
        capture: dict[str, list] = {}
        for _ in range(STEPS_PER_PROMPT):
            cur = int(rt.step(cur, capture_routes=capture))
        routes_by_prompt.append(capture)
    cp.cuda.Device(0).synchronize()

    import random
    rng = random.Random(20260816)

    results = {}
    for N in UNION_SIZES:
        union_sizes = []
        trials = 30
        for _ in range(trials):
            subset = rng.sample(range(len(PROMPTS)), N)
            for step in range(STEPS_PER_PROMPT):
                for layer in moe_layers:
                    s = set()
                    for p in subset:
                        route_list = routes_by_prompt[p].get(str(layer))
                        if route_list is None or step >= len(route_list):
                            continue
                        s.update(route_list[step])
                    if s:
                        union_sizes.append(len(s))
        stats = percentiles([float(x) for x in union_sizes])
        results[str(N)] = {
            "union_size_stats": stats,
            "no_overlap_baseline": N * 6,
            "amplification_vs_single": (stats["mean"] / 6) if stats["mean"] else None,
            "amplification_vs_no_overlap": (stats["mean"] / (N * 6)) if stats["mean"] else None,
        }
        print(f"N={N}: mean union {stats['mean']:.2f} of max {N*6} experts/layer "
              f"({100*stats['mean']/(N*6):.1f}% of no-overlap baseline)", flush=True)

    payload = {
        "kind": "diag_cross_sequence_union",
        "created_utc": utc_now(),
        "note": "read-only diagnostic; first no-build measurement for the batch>1 hypothesis (agents/TODO.md, RESEARCH_NOTEBOOK.md 2026-08-16)",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_prompts": len(PROMPTS),
        "steps_per_prompt": STEPS_PER_PROMPT,
        "top_k": int(rt.top_k),
        "n_experts": int(rt.n_experts),
        "moe_layer_count": len(moe_layers),
        "results": results,
    }
    out = REPO / "pro_research" / "diag_cross_sequence_union.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
