"""Fuse `add_` into the following `norm`: 52 launches removed per token.

The completed token map made this the cheapest remaining win. Measured:
**3.53 us per kernel launch inside a captured graph**, from 105 norm/add
launches costing 0.370 ms while `rmsnorm_bf16w` touches only 10.75 KB (0.03 us
of real traffic at 345.9 GB/s). Almost all of it is fixed per-kernel cost.

In `_step_body_graph` the sequence per layer boundary is:

    ... layer i writes self.acc ...
    k.add_(self.h, self.acc, self.hidden)          # h += acc
    k.norm(self.normed, self.h, d["norm"], ...)    # next layer's input norm

The add is immediately followed by a norm over the very buffer it just wrote, so
the pair is one kernel's worth of work split across two launches. Fusing them
turns 105 launches into 53 -- 52 removed, ~0.18 ms at the measured rate -- and
also drops one full re-read of `h` (5 passes over the buffer become 4).

## Why it is bit-exact

`add_inplace` is `dst[i] += src[i]`, elementwise and independent, so moving it
into a different thread mapping cannot change any value. The RMSNorm reduction
is reproduced exactly: same per-thread `fmaf(v, v, acc)` stride loop, same warp
shuffle, same in-order sum of the warp sums by thread 0, same
`rsqrtf(s/n + eps)`, same final scaling loop. Nothing is re-associated -- which
is the trap that killed the two-phase ssm_step variant's alternative and PV2-11.

## Arms

  split   52 x add_inplace (11 blocks each) + 52 x rmsnorm_bf16w (1 block each)
  fused   52 x add_rmsnorm (1 block each)

Gate: `h` and `out` bit-identical between the arms after the full 52-boundary
sequence. Timing read only if that holds.

The harness runs 52 boundaries per "token" on distinct buffers so nothing is
kept hot that the real loop would not keep hot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_gpu_free, utc_now, write_json_atomic

HIDDEN = 2688
BOUNDARIES = 52
EPS = 1e-5
ROUNDS = 200
BLOCK = 256

SRC = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// PRODUCTION, verbatim.
extern "C" __global__ void add_inplace(
    float* __restrict__ dst, const float* __restrict__ src, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// PRODUCTION, verbatim.
extern "C" __global__ void rmsnorm_bf16w(
    const float* __restrict__ x, const unsigned short* __restrict__ w,
    float* __restrict__ out, const int n, const float eps)
{
    extern __shared__ float red[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = x[i]; acc = fmaf(v, v, acc); }
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) s += red[i];
        red[31] = rsqrtf(s / (float)n + eps);
    }
    __syncthreads();
    const float scale = red[31];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = x[i] * scale * bf16_to_f32(w[i]);
}

// FUSED: h += acc, then the identical RMSNorm of the updated h.
// The add is elementwise and independent, so a different thread mapping cannot
// change a value; the reduction below is the production one line for line.
extern "C" __global__ void add_rmsnorm(
    float* __restrict__ h, const float* __restrict__ addend,
    const unsigned short* __restrict__ w, float* __restrict__ out,
    const int n, const float eps)
{
    extern __shared__ float red[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = h[i] + addend[i];
        h[i] = v;
        acc = fmaf(v, v, acc);
    }
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) s += red[i];
        red[31] = rsqrtf(s / (float)n + eps);
    }
    __syncthreads();
    const float scale = red[31];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = h[i] * scale * bf16_to_f32(w[i]);
}
"""


def main() -> int:
    require_gpu_free()
    import cupy as cp

    mod = cp.RawModule(code=SRC, options=("-std=c++14", "--use_fast_math"))
    k_add = mod.get_function("add_inplace")
    k_norm = mod.get_function("rmsnorm_bf16w")
    k_fused = mod.get_function("add_rmsnorm")

    rng = np.random.default_rng(20260816)
    h0 = (rng.standard_normal(HIDDEN) * 0.5).astype(np.float32)
    accs = [cp.asarray((rng.standard_normal(HIDDEN) * 0.1).astype(np.float32))
            for _ in range(BOUNDARIES)]
    ws = [cp.asarray(((rng.standard_normal(HIDDEN) * 0.3).astype(np.float32)
                      .view(np.uint32) >> 16).astype(np.uint16))
          for _ in range(BOUNDARIES)]
    h_s = cp.asarray(h0.copy())
    h_f = cp.asarray(h0.copy())
    out_s = cp.zeros(HIDDEN, dtype=cp.float32)
    out_f = cp.zeros(HIDDEN, dtype=cp.float32)
    smem = 32 * 4
    add_blocks = (HIDDEN + BLOCK - 1) // BLOCK
    n32, eps32 = np.int32(HIDDEN), np.float32(EPS)

    def run_split(h, out):
        for i in range(BOUNDARIES):
            k_add((add_blocks,), (BLOCK,), (h, accs[i], n32))
            k_norm((1,), (BLOCK,), (h, ws[i], out, n32, eps32), shared_mem=smem)

    def run_fused(h, out):
        for i in range(BOUNDARIES):
            k_fused((1,), (BLOCK,), (h, accs[i], ws[i], out, n32, eps32),
                    shared_mem=smem)

    run_split(h_s, out_s)
    run_fused(h_f, out_f)
    cp.cuda.Device(0).synchronize()
    h_exact = bool(np.array_equal(cp.asnumpy(h_s).view(np.uint32),
                                  cp.asnumpy(h_f).view(np.uint32)))
    out_exact = bool(np.array_equal(cp.asnumpy(out_s).view(np.uint32),
                                    cp.asnumpy(out_f).view(np.uint32)))
    finite = bool(np.isfinite(cp.asnumpy(out_s)).all())

    def timed(fn, h, out):
        fn(h, out)
        cp.cuda.Device(0).synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        for _ in range(ROUNDS):
            fn(h, out)
        e1.record()
        e1.synchronize()
        return cp.cuda.get_elapsed_time(e0, e1) / ROUNDS

    ms_s = timed(run_split, h_s, out_s)
    ms_f = timed(run_fused, h_f, out_f)

    payload = {
        "kind": "diag_add_norm_fusion",
        "created_utc": utc_now(),
        "note": "Fuses add_inplace into the following rmsnorm_bf16w. The completed token map measured 3.53 us per in-graph kernel launch and 0.370 ms for the 105 norm/add launches, almost all of it fixed cost rather than work (rmsnorm touches 10.75 KB = 0.03 us of traffic). The add is elementwise so its thread mapping cannot change a value, and the reduction is reproduced line for line, so the fusion is bit-exact rather than approximately equal.",
        "geometry": {"hidden": HIDDEN, "boundaries_per_token": BOUNDARIES,
                     "launches_split": 2 * BOUNDARIES,
                     "launches_fused": BOUNDARIES,
                     "launches_removed": BOUNDARIES},
        "gates": {"h_bit_exact": h_exact, "out_bit_exact": out_exact,
                  "finite": finite},
        "split_ms_per_token": ms_s,
        "fused_ms_per_token": ms_f,
        "saving_ms_per_token": ms_s - ms_f,
        "speedup": ms_s / ms_f if ms_f else None,
        "us_per_launch_implied": (ms_s - ms_f) * 1000.0 / BOUNDARIES,
        "in_graph_reference": {"norms_adds_marginal_ms": 0.370,
                               "us_per_launch_measured": 3.53},
        "status": "measured" if (h_exact and out_exact and finite) else "correctness_failed",
    }
    write_json_atomic(REPO / "pro_research" / "diag_add_norm_fusion.json", payload,
                      archive=False)
    print(json.dumps({k: payload[k] for k in
                      ("gates", "split_ms_per_token", "fused_ms_per_token",
                       "saving_ms_per_token", "speedup", "us_per_launch_implied",
                       "status")}, indent=2))
    return 0 if payload["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
