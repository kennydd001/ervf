"""Exact NVFP4 multi-RHS kernel with one shared-memory decode per output row.

New E100 geometry: N RHS groups do not share accumulator registers.  Each
16-lane subgroup owns one (row, rhs), reproduces the adopted width-16 ERVF
virtual-thread mapping, and consumes a row that the CTA decoded once into
shared memory.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_RHS = (4, 8, 16)
ROW_TILE = {4: 4, 8: 2, 16: 1}

_TEMPLATE = r"""
#define NRHS @NRHS@
#define ROW_TILE @ROW_TILE@
#define WIDTH 16
#define VIRTUAL 16

extern "C" __global__ void nvfp4_smem_mrhs(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut,
    const float* __restrict__ X,
    float* __restrict__ out,
    const float global_scale,
    const int rows,
    const int cols,
    const int apply_relu2,
    const float out_scale)
{
    // ROW_TILE * cols decoded FP32 weights.  No decoded matrix ever reaches
    // global memory; this is a transient CTA-local reuse surface.
    extern __shared__ float sw[];
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    // Decode each physical output row exactly once.  The floating expressions
    // intentionally match the production NVFP4 path: first scale, then E2M1.
    const int total = ROW_TILE * cols;
    const int n_bytes = cols >> 1;
    const int n_scales = cols >> 4;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int rl = idx / cols;
        const int k = idx - rl * cols;
        const int row = blockIdx.x * ROW_TILE + rl;
        float ww = 0.0f;
        if (row < rows) {
            const unsigned char byte = codes[(size_t)row * n_bytes + (k >> 1)];
            const unsigned char nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            const float sc = e4m3_lut[scales[(size_t)row * n_scales + (k >> 4)]] * global_scale;
            ww = s_e2m1[nib] * sc;
        }
        sw[idx] = ww;
    }
    __syncthreads();

    // 16 lanes per (row,rhs).  ROW_TILE*NRHS == 16, so 256 threads are used.
    const int lane = threadIdx.x & 15;
    const int group = threadIdx.x >> 4;
    const int rl = group / NRHS;
    const int rhs = group - rl * NRHS;
    const int row = blockIdx.x * ROW_TILE + rl;
    if (row >= rows) return;

    const float* __restrict__ w = sw + (size_t)rl * cols;
    const float* __restrict__ x = X + (size_t)rhs * cols;
    const int n_vec = (cols >> 1) >> 2;  // uchar4 packed-code vectors, 8 weights each

    // One accumulator per production virtual tid owned by this physical lane.
    float part[VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) part[vi] = 0.0f;

    // Preserve the production assignment and MAC stream exactly: virtual tid
    // walks packed uchar4 vector v = tid, tid+256, ... and performs eight FMAs
    // in nibble order for each vector.
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) {
        const int vtid = lane + WIDTH * vi;
        float a = 0.0f;
        for (int v = vtid; v < n_vec; v += 256) {
            const int k = v << 3;
            a = fmaf(w[k],     x[k],     a);
            a = fmaf(w[k + 1], x[k + 1], a);
            a = fmaf(w[k + 2], x[k + 2], a);
            a = fmaf(w[k + 3], x[k + 3], a);
            a = fmaf(w[k + 4], x[k + 4], a);
            a = fmaf(w[k + 5], x[k + 5], a);
            a = fmaf(w[k + 6], x[k + 6], a);
            a = fmaf(w[k + 7], x[k + 7], a);
        }
        // Same byte-tail ownership as the production kernel.  Relevant
        // Lightning shapes are vector-aligned, but keep this exact path.
        const int n_vec_bytes = n_vec << 2;
        const int nbytes = cols >> 1;
        for (int b = n_vec_bytes + vtid; b < nbytes; b += 256) {
            const int k = b << 1;
            a = fmaf(w[k],     x[k],     a);
            a = fmaf(w[k + 1], x[k + 1], a);
        }
        part[vi] = a;
    }

    // Rebuild the exact 256-thread reference reduction tree for WIDTH=16.
    // Every reference warp contains two virtual accumulators from this lane;
    // the first offset-16 step is therefore the local pair add below.
    float s8[8];
    #pragma unroll
    for (int rw = 0; rw < 8; ++rw) {
        float v = part[rw * 2] + part[rw * 2 + 1];
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, 16);
        s8[rw] = v;
    }

    if (lane == 0) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        const float u0 = t0 + t2;
        const float u1 = t1 + t3;
        const float v = u0 + u1;
        if (apply_relu2) {
            const float z = fmaxf(v, 0.0f);
            out[(size_t)rhs * rows + row] = z * z;
        } else {
            out[(size_t)rhs * rows + row] = v * out_scale;
        }
    }
}
"""


class ExactNVFP4SmemMRHS:
    def __init__(self, rhs_values=SUPPORTED_RHS):
        import cupy as cp

        self.cp = cp
        self.mods = {}
        self.kernels = {}
        for n in rhs_values:
            n = int(n)
            if n not in SUPPORTED_RHS:
                raise ValueError(f"unsupported NRHS={n}; supported={SUPPORTED_RHS}")
            src = _TEMPLATE.replace("@NRHS@", str(n)).replace("@ROW_TILE@", str(ROW_TILE[n]))
            mod = cp.RawModule(code=src, options=("-std=c++14",))
            self.mods[n] = mod
            self.kernels[n] = mod.get_function("nvfp4_smem_mrhs")

    def run(self, n, out, codes, scales, e2m1, e4m3, X, global_scale,
            rows, cols, apply_relu2=False, out_scale=1.0):
        n = int(n)
        rtile = ROW_TILE[n]
        grid = ((int(rows) + rtile - 1) // rtile,)
        shared = int(rtile * int(cols) * 4)
        self.kernels[n](
            grid, (256,),
            (codes, scales, e2m1, e4m3, X, out,
             np.float32(global_scale), np.int32(rows), np.int32(cols),
             np.int32(1 if apply_relu2 else 0), np.float32(out_scale)),
            shared_mem=shared,
        )
