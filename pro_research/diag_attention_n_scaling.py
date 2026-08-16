"""Tests the key open assumption in agents/BATCH_ARCHITECTURE_DESIGN.md
(2026-08-16): does attention GEMV cost scale ~linearly with N (no sharing
mechanism available, unlike MoE's expert-fetch), or is there hidden
launch-overhead slack worth exploiting the way MoE's panel_scan/
reduce_partials turned out to have?

Method: time the existing, unmodified production Q-projection GEMV
(rt.k.mv_bf16, already ERVF-dispatched via V4's _install_selective pattern)
called N times in a row against N different real captured normed
activations, for N in {1,2,4,8,16}. If cost scales linearly, N=16 should
cost ~16x N=1. Any sub-linear ratio indicates recoverable launch overhead;
a supra-linear ratio (thermal throttling, contention) would be a different
concern.

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
from ervf_dense import DenseERVF
from selective_ervf_v3 import _install_selective

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
    dense = DenseERVF()
    restore, _ = _install_selective(rt, dense)  # same V4/V6 Q/O ERVF dispatch

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    attn_layers = [i for i, ch in enumerate(rt.pattern) if ch == "*"]
    target_layer = attn_layers[2]

    captured = []
    orig_attn = rt._attention

    def capture_attn(self, i, out):
        if i == target_layer:
            captured.append(cp.asarray(self.normed).copy())
        return orig_attn(i, out)

    rt._attention = types.MethodType(capture_attn, rt)
    for prompt in PROMPTS[:max(N_VALUES)]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)
    rt._attention = orig_attn
    restore()
    cp.cuda.Device(0).synchronize()

    max_n = max(N_VALUES)
    if len(captured) < max_n:
        print(f"only captured {len(captured)} activations, expected {max_n}")
        return 1
    normed_vecs = captured[-max_n:]

    d = rt.layer[target_layer]
    q_proj = d["q_proj"]
    rows = rt.n_heads * rt.head_dim
    cols = rt.hidden

    results = {}
    for N in N_VALUES:
        vecs = normed_vecs[:N]
        qv_out = cp.zeros(rows, dtype=cp.float32)
        round_ms = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter_ns()
            for x in vecs:
                rt.k.mv_bf16(qv_out, q_proj, x, rows, cols)
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
        "kind": "diag_attention_n_scaling",
        "created_utc": utc_now(),
        "note": "read-only diagnostic testing whether attention GEMV cost scales linearly with N, per agents/BATCH_ARCHITECTURE_DESIGN.md's open risk #1",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "target_layer": target_layer,
        "rows": rows, "cols": cols,
        "rounds": ROUNDS,
        "n_values": N_VALUES,
        "results_by_n": results,
        "scaling_vs_ideal_linear": scaling,
    }
    out = REPO / "pro_research" / "diag_attention_n_scaling.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
