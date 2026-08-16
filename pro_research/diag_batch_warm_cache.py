"""Every batch>1 prototype so far (proto_batch_moe_layer.py,
proto_batch_moe_multilayer.py, proto_batch_down_proj.py) measured a single
COLD-cache snapshot -- deliberately, to isolate the fetch-amortization
effect from LRU hit-rate dynamics. This tests the dimension those
deliberately excluded: does the shared-fetch benefit hold up over multiple
CONSECUTIVE steps with a WARM, evolving LRU cache (closer to how a real
batch>1 serving runtime would actually behave), or does cache warmup change
the picture?

Uses the real production cache_assign/alloc_device_cache kernels (not a
reimplementation) fed the UNION of N sequences' routing ids per step, versus
N independent per-sequence caches evolving separately over the same T real
steps -- exactly what N separate batch=1 runtime instances have today.

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
T = 40          # consecutive decode steps per sequence
TOP_K = 6
CAP_SHARED = 72   # one shared cache, same total budget as production default
CAP_NAIVE = 72    # N independent caches, each at the SAME cap (what N separate batch=1 instances have today -- not a reduced per-sequence share)


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

    # ---- capture T real per-step routes for each of N sequences at target_layer.
    routes_by_seq = []
    for prompt in PROMPTS[:N]:
        captured = []
        orig_route = rt._route_device

        def capture_route(i, _captured=captured):
            packed = orig_route(i)
            if i == target_layer:
                _captured.append([int(x) for x in cp.asnumpy(packed)[:TOP_K]])
            return packed

        rt._route_device = types.MethodType(lambda self, i, cr=capture_route: cr(i), rt)
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        cur = nxt
        for _ in range(T):
            cur = int(rt.step(cur, capture_routes=None))
        rt._route_device = orig_route
        routes_by_seq.append(captured[-T:])
    cp.cuda.Device(0).synchronize()

    if any(len(r) < T for r in routes_by_seq):
        print("insufficient captured steps:", [len(r) for r in routes_by_seq])
        return 1

    fused = rt.fused
    n_experts = rt.n_experts
    globals_host = rt.bank[target_layer]["globals"]

    # ---- SHARED: one cache, fed the union of all N sequences' ids each step.
    dev_shared = fused.alloc_device_cache(n_experts, CAP_SHARED, N * TOP_K, globals_host)
    shared_misses_per_step = []
    for t in range(T):
        union_ids = []
        for s in range(N):
            union_ids.extend(routes_by_seq[s][t])
        # pad/truncate isn't needed: cache_assign takes exactly len(ids) slots via top_k param
        ids_dev = cp.asarray(union_ids, dtype=cp.int32)
        # dev_shared's "ids"/"slots"/"need" were sized for N*TOP_K; reuse buffer, write into it
        dev_shared["ids"][:len(union_ids)] = ids_dev
        fused.cache_assign(dev_shared, dev_shared["ids"], CAP_SHARED, len(union_ids))
        cp.cuda.Device(0).synchronize()
        need = cp.asnumpy(dev_shared["need"])[:len(union_ids)]
        shared_misses_per_step.append(int(need.sum()))

    # ---- NAIVE: N independent caches, each evolving over the same T steps.
    naive_misses_per_step = [0] * T
    for s in range(N):
        dev_s = fused.alloc_device_cache(n_experts, CAP_NAIVE, TOP_K, globals_host)
        for t in range(T):
            ids_dev = cp.asarray(routes_by_seq[s][t], dtype=cp.int32)
            dev_s["ids"][:TOP_K] = ids_dev
            fused.cache_assign(dev_s, dev_s["ids"], CAP_NAIVE, TOP_K)
            cp.cuda.Device(0).synchronize()
            need = cp.asnumpy(dev_s["need"])[:TOP_K]
            naive_misses_per_step[t] += int(need.sum())

    total_shared_misses = sum(shared_misses_per_step)
    total_naive_misses = sum(naive_misses_per_step)
    total_shared_calls = T * N * TOP_K
    total_naive_calls = T * N * TOP_K

    payload = {
        "kind": "diag_batch_warm_cache",
        "created_utc": utc_now(),
        "note": "read-only diagnostic; tests fetch-sharing under a warm, evolving LRU cache across consecutive real steps, using the real cache_assign kernel -- complements the cold-cache-only prototypes",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",)),
        "target_layer": target_layer,
        "n_sequences": N,
        "steps": T,
        "top_k": TOP_K,
        "cap_shared": CAP_SHARED,
        "cap_naive_per_sequence": CAP_NAIVE,
        "shared_misses_per_step": shared_misses_per_step,
        "naive_misses_per_step": naive_misses_per_step,
        "total_shared_misses": total_shared_misses,
        "total_naive_misses": total_naive_misses,
        "total_calls": total_shared_calls,
        "shared_hit_rate": 1.0 - total_shared_misses / total_shared_calls,
        "naive_hit_rate": 1.0 - total_naive_misses / total_naive_calls,
        "miss_reduction_fraction": 1.0 - (total_shared_misses / total_naive_misses) if total_naive_misses else None,
        "steady_state_shared_misses_last_quarter": sum(shared_misses_per_step[-T // 4:]),
        "steady_state_naive_misses_last_quarter": sum(naive_misses_per_step[-T // 4:]),
    }
    out = REPO / "pro_research" / "diag_batch_warm_cache.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
