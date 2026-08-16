"""Exact tiled shared-decode NVFP4 MRHS kernels.

V5 fixes the whole-row shared-memory limit of the first SMEM-MRHS prototype.
Each CTA stages exactly one 256-packed-vector epoch (2048 FP32 weights) per
output row, reuses it across N RHS subgroups, then advances to the next epoch.
Each 16-lane subgroup preserves the adopted width-16 ERVF virtual-tid mapping.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_RHS = (4, 8, 16)
ROW_TILE = {4: 4, 8: 2, 16: 1}
TILE_VECS = 256
WEIGHTS_PER_VEC = 8

_TEMPLATE = r"""
#define NRHS @NRHS@
#define ROW_TILE @ROW_TILE@
#define WIDTH 16
#define VIRTUAL 16
#define TILE_VECS 256
#define WPV 8

extern "C" __global__ void nvfp4_tiled_mrhs(
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
    // ROW_TILE * TILE_VECS * 8 decoded FP32 weights.  With ROW_TILE<=4 this
    // is <=32 KiB, independent of the full matrix input width.
    extern __shared__ float sw[];
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x & 15;
    const int group = threadIdx.x >> 4;
    const int rl = group / NRHS;
    const int rhs = group - rl * NRHS;
    const int row = blockIdx.x * ROW_TILE + rl;

    const int n_bytes = cols >> 1;
    const int n_vec = n_bytes >> 2;  // uchar4 packed-code vectors, 8 weights each
    const int n_scales = cols >> 4;

    // One untouched accumulator for every production virtual tid represented by
    // this physical width-16 lane.  The accumulators live across all K tiles.
    float part[VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) part[vi] = 0.0f;

    for (int base_v = 0; base_v < n_vec; base_v += TILE_VECS) {
        // Decode each packed vector once per physical output row.  The exact
        // scalar expression matches production: e4m3_scale*global first, then
        // e2m1*scale.  Shared memory stores the already-rounded float weight.
        const int tasks = ROW_TILE * TILE_VECS;
        for (int task = threadIdx.x; task < tasks; task += blockDim.x) {
            const int drl = task / TILE_VECS;
            const int lv = task - drl * TILE_VECS;
            const int drow = blockIdx.x * ROW_TILE + drl;
            const int v = base_v + lv;
            float* dst = sw + ((size_t)drl * TILE_VECS + lv) * WPV;
            if (drow < rows && v < n_vec) {
                const int b = v << 2;
                const uchar4 q = reinterpret_cast<const uchar4*>(
                    codes + (size_t)drow * n_bytes)[v];
                const unsigned char scode = scales[(size_t)drow * n_scales + (b >> 3)];
                const float sc = e4m3_lut[scode] * global_scale;
                dst[0] = s_e2m1[q.x & 0x0F] * sc;
                dst[1] = s_e2m1[q.x >> 4]   * sc;
                dst[2] = s_e2m1[q.y & 0x0F] * sc;
                dst[3] = s_e2m1[q.y >> 4]   * sc;
                dst[4] = s_e2m1[q.z & 0x0F] * sc;
                dst[5] = s_e2m1[q.z >> 4]   * sc;
                dst[6] = s_e2m1[q.w & 0x0F] * sc;
                dst[7] = s_e2m1[q.w >> 4]   * sc;
            } else {
                #pragma unroll
                for (int j = 0; j < WPV; ++j) dst[j] = 0.0f;
            }
        }
        __syncthreads();

        if (row < rows) {
            const float* __restrict__ xr = X + (size_t)rhs * cols;
            #pragma unroll
            for (int vi = 0; vi < VIRTUAL; ++vi) {
                const int vtid = lane + WIDTH * vi;
                const int v = base_v + vtid;
                if (v < n_vec) {
                    const float* w = sw + ((size_t)rl * TILE_VECS + vtid) * WPV;
                    const int k = v << 3;
                    float a = part[vi];
                    a = fmaf(w[0], xr[k],     a);
                    a = fmaf(w[1], xr[k + 1], a);
                    a = fmaf(w[2], xr[k + 2], a);
                    a = fmaf(w[3], xr[k + 3], a);
                    a = fmaf(w[4], xr[k + 4], a);
                    a = fmaf(w[5], xr[k + 5], a);
                    a = fmaf(w[6], xr[k + 6], a);
                    a = fmaf(w[7], xr[k + 7], a);
                    part[vi] = a;
                }
            }
        }
        // No subgroup may read the previous tile after any thread starts
        // overwriting it for the next epoch.
        __syncthreads();
    }

    if (row >= rows) return;

    // Reference tail: after all full uchar4 vectors, virtual tid owns byte
    // b=(n_vec<<2)+vtid, then b+=256.  There are at most three bytes here;
    // direct decode is negligible and preserves the exact per-tid MAC order.
    const float* __restrict__ xr = X + (size_t)rhs * cols;
    const int vec_bytes = n_vec << 2;
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) {
        const int vtid = lane + WIDTH * vi;
        float a = part[vi];
        for (int b = vec_bytes + vtid; b < n_bytes; b += 256) {
            const unsigned char q = codes[(size_t)row * n_bytes + b];
            const float sc = e4m3_lut[scales[(size_t)row * n_scales + (b >> 3)]] * global_scale;
            const int k = b << 1;
            const float w0 = s_e2m1[q & 0x0F] * sc;
            const float w1 = s_e2m1[q >> 4] * sc;
            a = fmaf(w0, xr[k],     a);
            a = fmaf(w1, xr[k + 1], a);
        }
        part[vi] = a;
    }

    // Rebuild the adopted width-16 ERVF reduction exactly.
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


class ExactNVFP4TiledMRHS:
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
            self.kernels[n] = mod.get_function("nvfp4_tiled_mrhs")

    def run(self, n, out, codes, scales, e2m1, e4m3, X, global_scale,
            rows, cols, apply_relu2=False, out_scale=1.0):
        n = int(n)
        rtile = ROW_TILE[n]
        grid = ((int(rows) + rtile - 1) // rtile,)
        shared = int(rtile * TILE_VECS * WEIGHTS_PER_VEC * 4)
        self.kernels[n](
            grid, (256,),
            (codes, scales, e2m1, e4m3, X, out,
             np.float32(global_scale), np.int32(rows), np.int32(cols),
             np.int32(1 if apply_relu2 else 0), np.float32(out_scale)),
            shared_mem=shared,
        )
