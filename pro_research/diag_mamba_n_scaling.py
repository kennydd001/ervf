"""Closes a gap in agents/BATCH_ARCHITECTURE_DESIGN.md's own claims: the
design doc asserts Mamba has no sharing opportunity and would scale
linearly with N, same as attention (diag_attention_n_scaling.py, confirmed
94-97% of ideal linear) -- but that was stated by analogy, never actually
measured for Mamba specifically. This measures it.

Mamba's in_proj uses the FP8-per-tensor kernel (mv_fp8_tensor), a different
kernel entirely from attention's BF16 ERVF path -- worth checking
independently rather than assuming the same conclusion transfers.

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

    mamba_layers = [i for i, ch in enumerate(rt.pattern) if ch == "M"]
    target_layer = mamba_layers[10]
    d = rt.layer[target_layer]
    if d["in_k"] != "fp8_tensor":
        print(f"layer {target_layer} in_proj is {d['in_k']}, expected fp8_tensor -- adjust target_layer")
        return 1

    captured = []
    orig_mamba = rt._mamba

    def capture_mamba(self, i, out):
        if i == target_layer:
            captured.append(cp.asarray(self.normed).copy())
        return orig_mamba(i, out)

    rt._mamba = types.MethodType(capture_mamba, rt)
    for prompt in PROMPTS[:max(N_VALUES)]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)
    rt._mamba = orig_mamba
    cp.cuda.Device(0).synchronize()

    max_n = max(N_VALUES)
    if len(captured) < max_n:
        print(f"only captured {len(captured)} activations, expected {max_n}")
        return 1
    normed_vecs = captured[-max_n:]

    in_w8 = d["in_w8"]
    in_s = d["in_s"]
    rows = rt.proj.size
    cols = rt.hidden

    results = {}
    for N in N_VALUES:
        vecs = normed_vecs[:N]
        out_buf = cp.zeros(rows, dtype=cp.float32)
        round_ms = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter_ns()
            for x in vecs:
                rt.k.mv_fp8_tensor(out_buf, in_w8, x, in_s, rows, cols)
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
        "kind": "diag_mamba_n_scaling",
        "created_utc": utc_now(),
        "note": "read-only diagnostic testing whether Mamba in_proj (FP8-tensor kernel) cost scales linearly with N, closing a gap in agents/BATCH_ARCHITECTURE_DESIGN.md's by-analogy claim",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "target_layer": target_layer,
        "rows": rows, "cols": cols,
        "rounds": ROUNDS,
        "n_values": N_VALUES,
        "results_by_n": results,
        "scaling_vs_ideal_linear": scaling,
    }
    out = REPO / "pro_research" / "diag_mamba_n_scaling.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
