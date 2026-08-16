"""Exact paired K/V split-warp GEMV for Lightning attention.

Each 32-thread warp is one original reference warp. The partial kernel preserves
the production virtual-thread assignment and per-thread FMA stream exactly.
A second 32-thread finalize kernel reconstructs the production reduction over
the eight warp sums. K and V are processed in the same two launches.
"""
from __future__ import annotations

import numpy as np

ROWS = 256
COLS = 2688
REF_WARPS = 8
WARPS_PER_BLOCK = 4

_SRC = r"""
__device__ __forceinline__ float kvs_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

extern "C" __global__ void kv_pair_partial(
    const unsigned short* __restrict__ Wk,
    const unsigned short* __restrict__ Wv,
    const float* __restrict__ x,
    float* __restrict__ partials,
    const int rows, const int cols)
{
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int which = blockIdx.z;  // 0=K, 1=V
    const int lane = threadIdx.x & 31;
    const int local_warp = threadIdx.x >> 5;
    const int ref_warp = group * 4 + local_warp;
    if (row >= rows || ref_warp >= 8) return;

    const unsigned short* __restrict__ W = which ? Wv : Wk;
    const unsigned short* __restrict__ w = W + (size_t)row * cols;
    const int tid = ref_warp * 32 + lane;

    float acc = 0.0f;
    for (int k = tid; k < cols; k += 256)
        acc = fmaf(kvs_bf16_to_f32(w[k]), x[k], acc);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0)
        partials[((size_t)which * rows + row) * 8 + ref_warp] = acc;
}

extern "C" __global__ void kv_pair_finalize(
    const float* __restrict__ partials,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    const int rows)
{
    const int row = blockIdx.x;
    const int which = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    float v = (lane < 8)
        ? partials[((size_t)which * rows + row) * 8 + lane]
        : 0.0f;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    if (lane == 0) {
        if (which) out_v[row] = v;
        else       out_k[row] = v;
    }
}
"""


class ExactKVSplitK:
    def __init__(self, rows: int = ROWS, cols: int = COLS):
        import cupy as cp

        self.cp = cp
        self.rows = int(rows)
        self.cols = int(cols)
        if self.rows != ROWS or self.cols != COLS:
            raise ValueError(f"frozen target is ({ROWS},{COLS}), got ({rows},{cols})")
        self.mod = cp.RawModule(code=_SRC, options=("-std=c++14",))
        self.partial_kernel = self.mod.get_function("kv_pair_partial")
        self.final_kernel = self.mod.get_function("kv_pair_finalize")
        self.partials = cp.empty((2, self.rows, REF_WARPS), dtype=cp.float32)

    def run_pair(self, out_k, out_v, Wk, Wv, x) -> None:
        groups = (REF_WARPS + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK
        self.partial_kernel(
            (self.rows, groups, 2),
            (WARPS_PER_BLOCK * 32,),
            (Wk, Wv, x, self.partials, np.int32(self.rows), np.int32(self.cols)),
        )
        self.final_kernel(
            (self.rows, 2),
            (32,),
            (self.partials, out_k, out_v, np.int32(self.rows)),
        )


def install_kv_pair_dispatch(rt, candidate: ExactKVSplitK):
    """Replace only the consecutive K/V `(256,2688)` calls during capture."""
    orig = rt.k.mv_bf16
    pending = {"W": None, "x": None}

    def dispatch(out, W, x, rows, cols):
        shape = (int(rows), int(cols))
        if shape != (ROWS, COLS):
            return orig(out, W, x, rows, cols)

        if out is rt.kv_:
            if pending["W"] is not None:
                raise RuntimeError("K/V dispatch saw a second pending K")
            pending["W"] = W
            pending["x"] = x
            return None

        if out is rt.vv:
            if pending["W"] is None:
                raise RuntimeError("V projection arrived without pending K")
            Wk, xk = pending["W"], pending["x"]
            pending["W"] = None
            pending["x"] = None
            if x is not xk:
                raise RuntimeError("K and V activation pointers differ")
            return candidate.run_pair(rt.kv_, rt.vv, Wk, W, x)

        raise RuntimeError("unexpected (256,2688) BF16 output target")

    rt.k.mv_bf16 = dispatch

    def restore():
        if pending["W"] is not None:
            raise RuntimeError("restoring K/V dispatch with unmatched pending K")
        rt.k.mv_bf16 = orig

    return restore
