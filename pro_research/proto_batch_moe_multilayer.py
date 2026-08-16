"""Extends proto_batch_moe_layer.py (2026-08-16, 2.89x fetch speedup,
bit-exact, one layer) in the two directions its own claim boundary flagged
as unmeasured:

1. Does the fetch-sharing benefit hold across MULTIPLE layers, or was layer
   24 favorable by chance? Tests all 23 MoE layers, not one.
2. GEMV COMPUTE time scales with N (computing against N different activation
   vectors costs more than computing against one), unlike the fetch, which
   is genuinely N-independent once shared. This measures compute time
   separately so the two effects aren't conflated -- summing fetch savings
   without accounting for compute growth would overstate the benefit.

Still a scoped, isolated prototype -- not attention, Mamba, KV-cache, or
graph capture, and still cold-cache (separates fetch-amortization from LRU
hit-rate effects). Correctness (bit-exact naive vs batched) is re-checked
per layer, not assumed from the single-layer result.

Not a gated PRO experiment.
"""

from __future__ import annotations

import sys
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

UP_CODE = 2_494_464
UP_SCALE = 311_808
N = 16


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    fused = rt.fused

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]

    # ---- capture N real (normed, route_ids) pairs at EVERY MoE layer in one pass.
    captured_by_layer = {i: [] for i in moe_layers}
    orig_route = rt._route_device

    def capture_route(i):
        packed = orig_route(i)
        captured_by_layer[i].append({
            "normed": cp.asarray(rt.normed).copy(),
            "packed": cp.asarray(packed).copy(),
        })
        return packed

    rt._route_device = types.MethodType(lambda self, i: capture_route(i), rt)

    for prompt in PROMPTS[:N]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)

    rt._route_device = orig_route
    cp.cuda.Device(0).synchronize()

    top_k = rt.top_k
    hidden = rt.hidden
    moe_inter = rt.moe_inter

    per_layer_results = []
    for layer in moe_layers:
        entries = captured_by_layer[layer][-N:]
        if len(entries) < N:
            continue
        seq_ids = [[int(x) for x in cp.asnumpy(e["packed"])[:top_k]] for e in entries]
        seq_normed = [e["normed"] for e in entries]
        union_experts = sorted(set(e for ids_s in seq_ids for e in ids_s))
        bank = rt.bank[layer]
        u = len(union_experts)

        # ---- NAIVE: N*top_k independent fetches + per-pair GEMV, timed separately.
        naive_scratch_c = cp.zeros(UP_CODE, dtype=cp.uint8)
        naive_scratch_s = cp.zeros(UP_SCALE, dtype=cp.uint8)
        ids_dev = cp.zeros(1, dtype=cp.int32)
        slots_dev = cp.zeros(1, dtype=cp.int32)
        need_dev = cp.ones(1, dtype=cp.int32)

        naive_fetch_ms = 0.0
        naive_compute_ms = 0.0
        naive_outputs = {}
        for s in range(N):
            for e in seq_ids[s]:
                ids_dev[0] = e
                ef0, ef1 = cp.cuda.Event(), cp.cuda.Event()
                ef0.record()
                fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                                  naive_scratch_c, naive_scratch_s,
                                  {"ids": ids_dev, "slots": slots_dev, "need": need_dev},
                                  UP_CODE, UP_SCALE, 1)
                ef1.record()
                ef1.synchronize()
                naive_fetch_ms += cp.cuda.get_elapsed_time(ef0, ef1)

                out = cp.zeros(moe_inter, dtype=cp.float32)
                ec0, ec1 = cp.cuda.Event(), cp.cuda.Event()
                ec0.record()
                fused.gemv_into(out, naive_scratch_c, naive_scratch_s, seq_normed[s],
                                float(bank["globals"][e, 1]), moe_inter, hidden, apply_relu2=True)
                ec1.record()
                ec1.synchronize()
                naive_compute_ms += cp.cuda.get_elapsed_time(ec0, ec1)
                naive_outputs[(s, e)] = cp.asnumpy(out)

        # ---- BATCHED: |union| fetches + per-pair GEMV against shared buffer, timed separately.
        expert_to_slot = {e: i for i, e in enumerate(union_experts)}
        batched_c = cp.zeros(u * UP_CODE, dtype=cp.uint8)
        batched_s = cp.zeros(u * UP_SCALE, dtype=cp.uint8)
        ids_dev_b = cp.asarray(union_experts, dtype=cp.int32)
        slots_dev_b = cp.arange(u, dtype=cp.int32)
        need_dev_b = cp.ones(u, dtype=cp.int32)

        bf0, bf1 = cp.cuda.Event(), cp.cuda.Event()
        bf0.record()
        fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                          batched_c, batched_s,
                          {"ids": ids_dev_b, "slots": slots_dev_b, "need": need_dev_b},
                          UP_CODE, UP_SCALE, u)
        bf1.record()
        bf1.synchronize()
        batched_fetch_ms = cp.cuda.get_elapsed_time(bf0, bf1)

        batched_compute_ms = 0.0
        batched_outputs = {}
        for s in range(N):
            for e in seq_ids[s]:
                slot = expert_to_slot[e]
                c_slice = batched_c[slot * UP_CODE:(slot + 1) * UP_CODE]
                s_slice = batched_s[slot * UP_SCALE:(slot + 1) * UP_SCALE]
                out = cp.zeros(moe_inter, dtype=cp.float32)
                ec0, ec1 = cp.cuda.Event(), cp.cuda.Event()
                ec0.record()
                fused.gemv_into(out, c_slice, s_slice, seq_normed[s],
                                float(bank["globals"][e, 1]), moe_inter, hidden, apply_relu2=True)
                ec1.record()
                ec1.synchronize()
                batched_compute_ms += cp.cuda.get_elapsed_time(ec0, ec1)
                batched_outputs[(s, e)] = cp.asnumpy(out)

        mismatches = sum(1 for k in naive_outputs if not (naive_outputs[k] == batched_outputs[k]).all())

        per_layer_results.append({
            "layer": layer,
            "union_expert_count": u,
            "naive_fetch_count": N * top_k,
            "dedup_fraction": 1.0 - (u / (N * top_k)),
            "correctness_mismatches": mismatches,
            "naive_fetch_ms": naive_fetch_ms,
            "batched_fetch_ms": batched_fetch_ms,
            "fetch_speedup": (naive_fetch_ms / batched_fetch_ms) if batched_fetch_ms else None,
            "naive_compute_ms": naive_compute_ms,
            "batched_compute_ms": batched_compute_ms,
            "compute_delta_ms": batched_compute_ms - naive_compute_ms,
            "naive_total_ms": naive_fetch_ms + naive_compute_ms,
            "batched_total_ms": batched_fetch_ms + batched_compute_ms,
            "total_speedup": ((naive_fetch_ms + naive_compute_ms) / (batched_fetch_ms + batched_compute_ms))
                             if (batched_fetch_ms + batched_compute_ms) else None,
        })
        r = per_layer_results[-1]
        print(f"layer {layer}: union={u}/{N*top_k} mismatches={mismatches} "
              f"fetch {r['naive_fetch_ms']:.2f}->{r['batched_fetch_ms']:.2f}ms "
              f"({r['fetch_speedup']:.2f}x) compute {r['naive_compute_ms']:.2f}->{r['batched_compute_ms']:.2f}ms "
              f"total_speedup={r['total_speedup']:.2f}x", flush=True)

    all_mismatches = sum(r["correctness_mismatches"] for r in per_layer_results)
    total_naive = sum(r["naive_total_ms"] for r in per_layer_results)
    total_batched = sum(r["batched_total_ms"] for r in per_layer_results)

    payload = {
        "kind": "proto_batch_moe_multilayer",
        "created_utc": utc_now(),
        "note": "scoped feasibility prototype across all MoE layers, not a production integration; cold-cache worst case",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_sequences": N,
        "top_k": top_k,
        "moe_layer_count": len(per_layer_results),
        "total_correctness_mismatches": all_mismatches,
        "correctness_pass": all_mismatches == 0,
        "sum_naive_total_ms_across_layers": total_naive,
        "sum_batched_total_ms_across_layers": total_batched,
        "sum_speedup_up_proj_only": (total_naive / total_batched) if total_batched else None,
        "claim_boundary": "up_proj GEMV+fetch only, summed across the 23 MoE layers actually measured (not extrapolated) -- still excludes down_proj, shared expert, attention, Mamba, KV-cache, graph capture, and routing/argmax/norm overhead. Not a full-token or full-throughput claim.",
        "per_layer": per_layer_results,
    }
    out = REPO / "pro_research" / "proto_batch_moe_multilayer.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if all_mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
