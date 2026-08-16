"""Closes a gap agents/BATCH_ARCHITECTURE_DESIGN.md never explicitly
addressed: lm_head (like the shared expert and attention) is not expert-
routed -- same weight matrix for every sequence every step -- so it should
scale ~linearly with N under the same reasoning. Unlike the shared expert
(just confirmed, diag_shared_expert_n_scaling.py) and attention (confirmed
earlier), lm_head was never even named as a risk item, let alone measured.
It's worth checking anyway: it's the single largest GEMV in the whole model
(output size = vocab, the biggest weight matrix walked every step), so if
ANY unbatched path were going to show a surprising N-scaling penalty from
sheer memory-bandwidth pressure, this is the most likely candidate.

Runs the existing (unbatched) lm_head GEMV N times against N real captured
post-norm_f activations, same production kernel, no new kernel written.

Not a gated PRO experiment.
"""

from __future__ import annotations

import sys
import time
import types
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

N_VALUES = (1, 2, 4, 8, 16)
ROUNDS = 30


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
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    if rt.lm_head_kind != "nvfp4":
        print(f"lm_head_kind is {rt.lm_head_kind}, expected nvfp4 -- script assumes fused.gemv_into path")
        return 1

    # capture the real post-norm_f activation feeding lm_head by wrapping
    # fused.gemv_into and snapshotting the call whose output width == vocab
    # (lm_head's own signature, distinct from every other gemv_into call in
    # the model -- shared expert/up/down all use hidden or shared_inter).
    captured = []
    fused = rt.fused
    orig_gemv_into = fused.gemv_into

    def capture_gemv_into(out, codes, scales, x, g, rows, cols, **kwargs):
        if rows == rt.vocab:
            captured.append(cp.asarray(x).copy())
        return orig_gemv_into(out, codes, scales, x, g, rows, cols, **kwargs)

    fused.gemv_into = capture_gemv_into
    for prompt in PROMPTS[:max(N_VALUES)]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)
    fused.gemv_into = orig_gemv_into
    cp.cuda.Device(0).synchronize()

    max_n = max(N_VALUES)
    if len(captured) < max_n:
        print(f"only captured {len(captured)} activations, expected {max_n}")
        return 1
    normed_vecs = captured[-max_n:]

    hidden = rt.hidden
    vocab = rt.vocab

    results = {}
    for N in N_VALUES:
        vecs = normed_vecs[:N]
        out_bufs = [cp.zeros(vocab, dtype=cp.float32) for _ in range(N)]
        round_ms = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter_ns()
            for x, out in zip(vecs, out_bufs):
                orig_gemv_into(out, rt.lm_head_codes, rt.lm_head_scales, x, rt.lm_head_g,
                               vocab, hidden)
            cp.cuda.Device(0).synchronize()
            round_ms.append((time.perf_counter_ns() - t0) / 1e6)
        stats = percentiles(round_ms)
        results[str(N)] = stats
        print(f"N={N}: p50={stats['p50']:.4f} ms total, {stats['p50']/N:.4f} ms/sequence", flush=True)

    n1_p50 = results["1"]["p50"]
    scaling = {}
    for N in N_VALUES:
        p50 = results[str(N)]["p50"]
        ideal_linear = n1_p50 * N
        scaling[str(N)] = {
            "measured_p50_ms": p50,
            "ideal_linear_ms": ideal_linear,
            "ratio_measured_over_ideal": p50 / ideal_linear if ideal_linear else None,
            "ms_per_sequence": p50 / N,
        }

    payload = {
        "kind": "diag_lmhead_n_scaling",
        "created_utc": utc_now(),
        "note": "read-only diagnostic testing whether lm_head's GEMV (unbatched, N sequential calls to the existing production kernel) scales linearly with N -- the biggest single GEMV in the model (output=vocab), never named as a risk in agents/BATCH_ARCHITECTURE_DESIGN.md, checked anyway",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "vocab": vocab, "hidden": hidden,
        "rounds": ROUNDS,
        "n_values": N_VALUES,
        "results_by_n": results,
        "scaling_vs_ideal_linear": scaling,
    }
    out = REPO / "pro_research" / "diag_lmhead_n_scaling.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
