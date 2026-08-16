"""Exact multi-RHS GEMV kernels for the E100 aggregate-throughput line.

One physical warp owns one output row.  Its 32 lanes emulate the reference
kernel's 256 virtual threads as eight independent accumulators per lane.  A
weight loaded for a virtual thread is reused across NRHS activation vectors,
but every RHS keeps the reference virtual-thread MAC order and reduction tree.

This module is additive research code.  It does not patch the runtime.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_RHS = (2, 4, 8)

_CUDA_TEMPLATE = r"""
#define NRHS @NRHS@
#define EVIRTUAL 8
#define EROWS_PER_BLOCK 8

__device__ __forceinline__ float e_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float e_e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

// Lane 0 returns the exact result of the reference's second reduction stage.
// part[g] is the untouched accumulator for virtual reference warp g.
__device__ __forceinline__ float e_reduce_256_exact(float part[EVIRTUAL]) {
    const int lane = threadIdx.x & 31;
    float s8[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = part[g];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        s8[g] = v;
    }
    if (lane == 0) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        const float u0 = t0 + t2;
        const float u1 = t1 + t3;
        return u0 + u1;
    }
    return 0.0f;
}

extern "C" __global__ void e_mrhs_bf16(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,       // [NRHS, cols]
    float* __restrict__ out,           // [NRHS, rows]
    const int rows, const int cols)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * EROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const unsigned short* __restrict__ w = W + (size_t)row * cols;

    float part[NRHS][EVIRTUAL];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r)
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) part[r][vi] = 0.0f;

    #pragma unroll
    for (int vi = 0; vi < EVIRTUAL; ++vi) {
        const int tid = lane + 32 * vi;
        for (int k = tid; k < cols; k += 256) {
            const float ww = e_bf16_to_f32(w[k]);
            #pragma unroll
            for (int r = 0; r < NRHS; ++r)
                part[r][vi] = fmaf(ww, X[(size_t)r * cols + k], part[r][vi]);
        }
    }

    #pragma unroll
    for (int r = 0; r < NRHS; ++r) {
        float p[EVIRTUAL];
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) p[vi] = part[r][vi];
        const float v = e_reduce_256_exact(p);
        if (lane == 0) out[(size_t)r * rows + row] = v;
    }
}

extern "C" __global__ void e_mrhs_f32(
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const int rows, const int cols)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * EROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const float* __restrict__ w = W + (size_t)row * cols;

    float part[NRHS][EVIRTUAL];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r)
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) part[r][vi] = 0.0f;

    #pragma unroll
    for (int vi = 0; vi < EVIRTUAL; ++vi) {
        const int tid = lane + 32 * vi;
        for (int k = tid; k < cols; k += 256) {
            const float ww = w[k];
            #pragma unroll
            for (int r = 0; r < NRHS; ++r)
                part[r][vi] = fmaf(ww, X[(size_t)r * cols + k], part[r][vi]);
        }
    }

    #pragma unroll
    for (int r = 0; r < NRHS; ++r) {
        float p[EVIRTUAL];
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) p[vi] = part[r][vi];
        const float v = e_reduce_256_exact(p);
        if (lane == 0) out[(size_t)r * rows + row] = v;
    }
}

extern "C" __global__ void e_mrhs_fp8_tensor(
    const unsigned char* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const float wscale,
    const int rows, const int cols)
{
    __shared__ float lut[256];
    if (threadIdx.x < 256) lut[threadIdx.x] = e_e4m3_decode((unsigned char)threadIdx.x);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * EROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const unsigned char* __restrict__ w = W + (size_t)row * cols;
    const uchar4* __restrict__ w4 = reinterpret_cast<const uchar4*>(w);
    const int nvec = cols >> 2;

    float part[NRHS][EVIRTUAL];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r)
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) part[r][vi] = 0.0f;

    #pragma unroll
    for (int vi = 0; vi < EVIRTUAL; ++vi) {
        const int tid = lane + 32 * vi;
        for (int qidx = tid; qidx < nvec; qidx += 256) {
            const uchar4 q = w4[qidx];
            const int k = qidx << 2;
            const float w0 = lut[q.x], w1 = lut[q.y], w2 = lut[q.z], w3 = lut[q.w];
            #pragma unroll
            for (int r = 0; r < NRHS; ++r) {
                const float* xr = X + (size_t)r * cols;
                float a = part[r][vi];
                a = fmaf(w0, xr[k],     a);
                a = fmaf(w1, xr[k + 1], a);
                a = fmaf(w2, xr[k + 2], a);
                a = fmaf(w3, xr[k + 3], a);
                part[r][vi] = a;
            }
        }
        for (int b = (nvec << 2) + tid; b < cols; b += 256) {
            const float ww = lut[w[b]];
            #pragma unroll
            for (int r = 0; r < NRHS; ++r)
                part[r][vi] = fmaf(ww, X[(size_t)r * cols + b], part[r][vi]);
        }
    }

    #pragma unroll
    for (int r = 0; r < NRHS; ++r) {
        float p[EVIRTUAL];
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) p[vi] = part[r][vi];
        const float v = e_reduce_256_exact(p);
        if (lane == 0) out[(size_t)r * rows + row] = v * wscale;
    }
}

extern "C" __global__ void e_mrhs_nvfp4(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut,
    const float* __restrict__ X,
    float* __restrict__ out,
    const float global_scale,
    const int rows, const int cols,
    const int apply_relu2, const float out_scale)
{
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * EROWS_PER_BLOCK + warp;
    if (row >= rows) return;

    const int n_bytes = cols >> 1;
    const int n_vec = n_bytes >> 2;
    const unsigned char* __restrict__ crow = codes + (size_t)row * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)row * (cols >> 4);
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);

    float part[NRHS][EVIRTUAL];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r)
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) part[r][vi] = 0.0f;

    #pragma unroll
    for (int vi = 0; vi < EVIRTUAL; ++vi) {
        const int tid = lane + 32 * vi;
        for (int v = tid; v < n_vec; v += 256) {
            const uchar4 q = crow4[v];
            const int b = v << 2;
            const float sc = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            const float w0 = s_e2m1[q.x & 0x0F] * sc;
            const float w1 = s_e2m1[q.x >> 4]   * sc;
            const float w2 = s_e2m1[q.y & 0x0F] * sc;
            const float w3 = s_e2m1[q.y >> 4]   * sc;
            const float w4v = s_e2m1[q.z & 0x0F] * sc;
            const float w5 = s_e2m1[q.z >> 4]   * sc;
            const float w6 = s_e2m1[q.w & 0x0F] * sc;
            const float w7 = s_e2m1[q.w >> 4]   * sc;
            #pragma unroll
            for (int r = 0; r < NRHS; ++r) {
                const float* xr = X + (size_t)r * cols;
                float a = part[r][vi];
                a = fmaf(w0,  xr[k],     a);
                a = fmaf(w1,  xr[k + 1], a);
                a = fmaf(w2,  xr[k + 2], a);
                a = fmaf(w3,  xr[k + 3], a);
                a = fmaf(w4v, xr[k + 4], a);
                a = fmaf(w5,  xr[k + 5], a);
                a = fmaf(w6,  xr[k + 6], a);
                a = fmaf(w7,  xr[k + 7], a);
                part[r][vi] = a;
            }
        }
        for (int b = (n_vec << 2) + tid; b < n_bytes; b += 256) {
            const unsigned char byte = crow[b];
            const float sc = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            const float w0 = s_e2m1[byte & 0x0F] * sc;
            const float w1 = s_e2m1[byte >> 4]   * sc;
            #pragma unroll
            for (int r = 0; r < NRHS; ++r) {
                const float* xr = X + (size_t)r * cols;
                float a = part[r][vi];
                a = fmaf(w0, xr[k],     a);
                a = fmaf(w1, xr[k + 1], a);
                part[r][vi] = a;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < NRHS; ++r) {
        float p[EVIRTUAL];
        #pragma unroll
        for (int vi = 0; vi < EVIRTUAL; ++vi) p[vi] = part[r][vi];
        const float v = e_reduce_256_exact(p);
        if (lane == 0) {
            if (apply_relu2) {
                const float z = fmaxf(v, 0.0f);
                out[(size_t)r * rows + row] = z * z;
            } else {
                out[(size_t)r * rows + row] = v * out_scale;
            }
        }
    }
}
"""


class ExactMRHS:
    """Compile one fixed-N module per RHS count to keep register pressure honest."""

    def __init__(self, rhs_values=SUPPORTED_RHS):
        import cupy as cp

        self.cp = cp
        self.mods = {}
        self.kernels = {}
        for n in rhs_values:
            n = int(n)
            if n not in SUPPORTED_RHS:
                raise ValueError(f"unsupported NRHS={n}; supported={SUPPORTED_RHS}")
            src = _CUDA_TEMPLATE.replace("@NRHS@", str(n))
            mod = cp.RawModule(code=src, options=("-std=c++14",))
            self.mods[n] = mod
            self.kernels[n] = {
                "bf16": mod.get_function("e_mrhs_bf16"),
                "f32": mod.get_function("e_mrhs_f32"),
                "fp8": mod.get_function("e_mrhs_fp8_tensor"),
                "nvfp4": mod.get_function("e_mrhs_nvfp4"),
            }
        self.block = 256
        self.rows_per_block = 8

    def _grid(self, rows: int):
        return ((int(rows) + self.rows_per_block - 1) // self.rows_per_block,)

    def bf16(self, n: int, out, W, X, rows: int, cols: int) -> None:
        self.kernels[int(n)]["bf16"](
            self._grid(rows), (self.block,),
            (W, X, out, np.int32(rows), np.int32(cols)))

    def f32(self, n: int, out, W, X, rows: int, cols: int) -> None:
        self.kernels[int(n)]["f32"](
            self._grid(rows), (self.block,),
            (W, X, out, np.int32(rows), np.int32(cols)))

    def fp8(self, n: int, out, W, X, scale: float, rows: int, cols: int) -> None:
        self.kernels[int(n)]["fp8"](
            self._grid(rows), (self.block,),
            (W, X, out, np.float32(scale), np.int32(rows), np.int32(cols)),
            shared_mem=256 * 4)

    def nvfp4(self, n: int, out, codes, scales, e2m1, e4m3, X,
              global_scale: float, rows: int, cols: int,
              apply_relu2: bool = False, out_scale: float = 1.0) -> None:
        self.kernels[int(n)]["nvfp4"](
            self._grid(rows), (self.block,),
            (codes, scales, e2m1, e4m3, X, out, np.float32(global_scale),
             np.int32(rows), np.int32(cols),
             np.int32(1 if apply_relu2 else 0), np.float32(out_scale)))


def cpu_mapping_selftest(trials: int = 200) -> dict:
    """Check the width-32 virtual reduction mapping against the 256-thread tree."""
    from ervf_dense import _reference_reduce

    rng = np.random.default_rng(20260816)
    mismatches = 0
    examples = []
    for t in range(trials):
        a = (rng.standard_normal(256) * (10.0 ** rng.uniform(-4.0, 4.0))).astype(np.float32)
        ref = _reference_reduce(a)
        # Emulate what lane 0 observes after each physical 32-lane reduction.
        s = np.zeros(8, dtype=np.float32)
        for g in range(8):
            w = a[g * 32:(g + 1) * 32].copy()
            for off in (16, 8, 4, 2, 1):
                old = w.copy()
                for lane in range(32 - off):
                    w[lane] = np.float32(np.float64(old[lane]) + np.float64(old[lane + off]))
            s[g] = w[0]
        t0 = np.float32(np.float64(s[0]) + np.float64(s[4]))
        t1 = np.float32(np.float64(s[1]) + np.float64(s[5]))
        t2 = np.float32(np.float64(s[2]) + np.float64(s[6]))
        t3 = np.float32(np.float64(s[3]) + np.float64(s[7]))
        u0 = np.float32(np.float64(t0) + np.float64(t2))
        u1 = np.float32(np.float64(t1) + np.float64(t3))
        got = np.float32(np.float64(u0) + np.float64(u1))
        if ref.view(np.uint32) != got.view(np.uint32):
            mismatches += 1
            if len(examples) < 3:
                examples.append({"trial": t, "ref_bits": int(ref.view(np.uint32)), "got_bits": int(got.view(np.uint32))})
    return {"trials": trials, "mismatches": mismatches, "examples": examples, "passed": mismatches == 0}
