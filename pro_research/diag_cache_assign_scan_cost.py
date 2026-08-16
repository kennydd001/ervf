"""Verifies rather than speculates: proto_multi_seq_full_model_n4_bigcache.py
found that DOUBLING cache capacity (72->144) caused a real regression
(0.706x, not an improvement) and proposed an explanation from reading
fused_nvfp4.py's cache_assign kernel source directly -- it does a LINEAR
scan over `cap` slots on every eviction (single-threaded: `if
(threadIdx.x != 0) return;` then `for (int cix = 1; cix < cap; cix++)`) to
find the least-recently-used slot, so a bigger cache should cost more per
miss. That explanation was never isolated and measured on its own -- this
does that directly: call fused.cache_assign in a tight, isolated loop with a
FULL cache (every call is a guaranteed eviction, worst case) at increasing
`cap` values, holding everything else fixed, and check whether wall time per
call actually scales with cap the way the kernel source implies it should.

Not a gated PRO experiment -- a root-cause micro-benchmark, read-only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, require_model_dir, utc_now, write_json_atomic

CAPS = (72, 144, 288, 576)
TOP_K = 6
ROUNDS = 200


def main() -> int:
    require_gpu_free()
    import cupy as cp
    import numpy as np
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.load_routed_bank()
    fused = rt.fused
    n_experts = rt.n_experts

    globals_host = rt.bank[rt.moe_layers[0]]["globals"]

    results = {}
    for cap in CAPS:
        dev = fused.alloc_device_cache(n_experts, cap, TOP_K, globals_host)
        # fill the cache completely first (cap/TOP_K calls with disjoint ids),
        # so every subsequent call is guaranteed to be a full-cache eviction
        # (worst case for the linear LRU scan), not a cold-fill hit.
        filler_ids = np.arange(cap, dtype=np.int32) % n_experts
        for start in range(0, cap, TOP_K):
            chunk = filler_ids[start:start + TOP_K]
            if len(chunk) < TOP_K:
                chunk = np.pad(chunk, (0, TOP_K - len(chunk)), constant_values=chunk[-1])
            ids_dev = cp.asarray(chunk, dtype=cp.int32)
            fused.cache_assign(dev, ids_dev, cap, TOP_K)
        cp.cuda.Device(0).synchronize()

        # now measure ROUNDS calls, each with a FRESH set of TOP_K ids not
        # currently resident (guaranteed miss+eviction every single call,
        # cycling through a pool larger than the cache so nothing gets
        # reused/hits by chance).
        pool = (np.arange(cap * 4, dtype=np.int32) % n_experts)
        round_ms = []
        for r in range(ROUNDS):
            start = (r * TOP_K) % (len(pool) - TOP_K)
            chunk = pool[start:start + TOP_K]
            ids_dev = cp.asarray(chunk, dtype=cp.int32)
            t0 = time.perf_counter_ns()
            fused.cache_assign(dev, ids_dev, cap, TOP_K)
            cp.cuda.Device(0).synchronize()
            round_ms.append((time.perf_counter_ns() - t0) / 1e6)
        stats = percentiles(round_ms)
        results[str(cap)] = stats
        print(f"cap={cap}: p50={stats['p50']:.5f} ms/call", flush=True)

    cap1 = CAPS[0]
    p50_at_cap1 = results[str(cap1)]["p50"]
    scaling = {}
    for cap in CAPS:
        p50 = results[str(cap)]["p50"]
        ideal_linear = p50_at_cap1 * (cap / cap1)
        scaling[str(cap)] = {
            "measured_p50_ms": p50,
            "ideal_linear_from_cap1_ms": ideal_linear,
            "ratio_measured_over_ideal_linear": p50 / ideal_linear if ideal_linear else None,
        }

    payload = {
        "kind": "diag_cache_assign_scan_cost",
        "created_utc": utc_now(),
        "note": "isolated micro-benchmark of fused.cache_assign's per-call cost as a function of cap, every call a guaranteed full-cache eviction (worst case for the kernel's linear LRU scan) -- verifies the explanation proposed for proto_multi_seq_full_model_n4_bigcache.py's 0.706x regression from doubling cache capacity",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",)),
        "caps_tested": CAPS,
        "top_k": TOP_K,
        "rounds_per_cap": ROUNDS,
        "results_by_cap": results,
        "scaling_vs_linear_from_smallest_cap": scaling,
    }
    out = REPO / "pro_research" / "diag_cache_assign_scan_cost.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
