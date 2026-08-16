"""Closes another by-analogy gap in agents/BATCH_ARCHITECTURE_DESIGN.md:
step 6 asserts the shared expert is "trivial" for batch>1 ("draait toch al
voor elke stap ongeacht routing... geen nieuwe deel-logica nodig") because
its weight is not expert-selected -- but that claim was never physically
measured, only reasoned by analogy to attention (also not expert-selected).
The Mamba scaling test already found ONE such by-analogy claim wrong
(diag_mamba_n_scaling.py: mild supra-linear, not flat like attention) --
same discipline applies here: verify, don't assume the analogy transfers.

Runs the existing (unbatched) shared-expert up_proj GEMV N times against N
real captured activations, same production kernel, no new kernel written --
this measures whether N sequential calls to the CURRENT unbatched shared-
expert path already scale ~linearly (cheap, as assumed) or supra-linearly
(a real cost the design doc's "trivial" claim would need to be corrected for,
same as Mamba was).

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

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    target_layer = moe_layers[10]
    d = rt.layer[target_layer]

    captured = []
    orig_route = rt._route_device

    def capture_route(i):
        packed = orig_route(i)
        if i == target_layer:
            captured.append(cp.asarray(rt.normed).copy())
        return packed

    rt._route_device = types.MethodType(lambda self, i: capture_route(i), rt)
    for prompt in PROMPTS[:max(N_VALUES)]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)
    rt._route_device = orig_route
    cp.cuda.Device(0).synchronize()

    max_n = max(N_VALUES)
    if len(captured) < max_n:
        print(f"only captured {len(captured)} activations, expected {max_n}")
        return 1
    normed_vecs = captured[-max_n:]

    fused = rt.fused
    hidden = rt.hidden
    shared_inter = rt.shared_inter

    results = {}
    for N in N_VALUES:
        vecs = normed_vecs[:N]
        out_bufs = [cp.zeros(shared_inter, dtype=cp.float32) for _ in range(N)]
        round_ms = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter_ns()
            for x, out in zip(vecs, out_bufs):
                fused.gemv_into(out, d["sh_up_c"], d["sh_up_s"], x, d["sh_up_g"],
                                shared_inter, hidden, apply_relu2=True)
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
        "kind": "diag_shared_expert_n_scaling",
        "created_utc": utc_now(),
        "note": "read-only diagnostic testing whether the shared-expert up_proj GEMV (unbatched, N sequential calls to the existing production kernel) scales linearly with N, closing a by-analogy gap in agents/BATCH_ARCHITECTURE_DESIGN.md step 6 ('trivial, no sharing logic needed') the same way diag_mamba_n_scaling.py closed the Mamba gap",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "target_layer": target_layer,
        "shared_inter": shared_inter, "hidden": hidden,
        "rounds": ROUNDS,
        "n_values": N_VALUES,
        "results_by_n": results,
        "scaling_vs_ideal_linear": scaling,
    }
    out = REPO / "pro_research" / "diag_shared_expert_n_scaling.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
