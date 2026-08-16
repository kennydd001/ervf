"""Tests, rather than leaves speculative, the candidate explanation proposed
for proto_multi_seq_moe_shared_warmcache.py's unexplained 54% "routing +
shared expert" cost (agents/RESEARCH_NOTEBOOK.md 2026-08-16, "Derde poging"
follow-up): does a large permanent VRAM reservation (matching the ~190
MiB/layer x 23 layers the warm-cache script holds for its persistent
codes/scales buffers) slow down CuPy's memory pool for the many small,
short-lived allocations the routing/shared-expert section still does every
call (cp.zeros(top_k), cp.zeros(hidden)-scale buffers)?

Measures small allocation+free cost in a tight loop, first with NOTHING else
reserved, then with a ~4.4 GiB persistent block held (matching the real
script's total reservation across 23 layers), same process, same pool.

Not a gated PRO experiment -- an isolated root-cause micro-benchmark,
read-only, no runtime modification.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, utc_now, write_json_atomic

ROUNDS = 2000
SMALL_SIZES = [6, 2688, 3712]  # top_k, hidden-ish, shared_inter-ish -- representative of section 1's real allocations
UP_CODE = 2_494_464
UP_SCALE = 311_808
CAP = 72
N_LAYERS = 23


def time_small_allocs(cp):
    round_ms = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter_ns()
        bufs = [cp.zeros(sz, dtype=cp.float32) for sz in SMALL_SIZES]
        cp.cuda.Device(0).synchronize()
        round_ms.append((time.perf_counter_ns() - t0) / 1e6)
        del bufs
    return round_ms


def main() -> int:
    require_gpu_free()
    import cupy as cp

    # ---- baseline: small allocations with nothing else reserved.
    baseline_ms = time_small_allocs(cp)
    baseline_stats = percentiles(baseline_ms)
    print(f"baseline (no large reservation): p50={baseline_stats['p50']:.5f} ms/round", flush=True)

    # ---- now hold a large persistent reservation matching the real script:
    # ~(72*UP_CODE + 72*UP_SCALE) bytes per layer x 23 layers.
    per_layer_bytes = CAP * UP_CODE + CAP * UP_SCALE
    total_bytes = per_layer_bytes * N_LAYERS
    large_blocks = [cp.zeros(per_layer_bytes, dtype=cp.uint8) for _ in range(N_LAYERS)]
    cp.cuda.Device(0).synchronize()

    pressured_ms = time_small_allocs(cp)
    pressured_stats = percentiles(pressured_ms)
    print(f"with ~{total_bytes/1e9:.2f} GiB reserved: p50={pressured_stats['p50']:.5f} ms/round", flush=True)

    del large_blocks
    cp.cuda.Device(0).synchronize()

    ratio = pressured_stats["p50"] / baseline_stats["p50"] if baseline_stats["p50"] else None
    hypothesis_supported = bool(ratio and ratio > 2.0)  # a real, large effect, not noise

    payload = {
        "kind": "diag_alloc_pressure",
        "created_utc": utc_now(),
        "note": "tests whether a large persistent VRAM reservation (matching proto_multi_seq_moe_shared_warmcache.py's ~4.4 GiB of persistent codes/scales buffers) slows down small, short-lived allocations via CuPy memory-pool pressure -- the candidate (unverified) explanation proposed for that script's unexplained 54% routing/shared-expert section cost",
        "environment": environment_snapshot(()),
        "rounds": ROUNDS,
        "small_alloc_sizes": SMALL_SIZES,
        "per_layer_reserved_bytes": per_layer_bytes,
        "n_layers_simulated": N_LAYERS,
        "total_reserved_bytes": total_bytes,
        "total_reserved_gib": total_bytes / (1024 ** 3),
        "baseline_small_alloc_stats_ms": baseline_stats,
        "pressured_small_alloc_stats_ms": pressured_stats,
        "ratio_pressured_over_baseline": ratio,
        "hypothesis_supported_gt_2x": hypothesis_supported,
    }
    out = REPO / "pro_research" / "diag_alloc_pressure.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
