"""Full-warp exact multi-RHS kernels for E100-MRHS256.

One 256-thread block owns one output row.  Physical thread tid is exactly the
reference kernel's virtual tid, so each thread needs only NRHS accumulators.
A weight/dequantised scalar is loaded once and used by all RHS accumulators;
each RHS preserves the production per-thread MAC stream and two-stage tree.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_RHS = (4, 8, 16)

_TEMPLATE = r"""
#define NRHS @NRHS@
#define NWARPS 8

__device__ __forceinline__ float m256_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float m256_e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

// acc[r] enters as this physical/reference tid's complete accumulator.
// The body reproduces the production 32-lane warp reduction followed by the
// warp-0 reduction over eight warp sums, independently for every RHS.
__device__ __forceinline__ void m256_reduce_write(
    float acc[NRHS], float* __restrict__ out, const int row, const int rows,
    const float final_scale, const int apply_relu2)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int r = 0; r < NRHS; ++r) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    }

    __shared__ float ws[NRHS * NWARPS];
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NRHS; ++r)
            ws[r * NWARPS + warp] = acc[r];
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < NRHS; ++r) {
            float v = (lane < NWARPS) ? ws[r * NWARPS + lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) {
                if (apply_relu2) {
                    const float z = fmaxf(v, 0.0f);
                    out[(size_t)r * rows + row] = z * z;
                } else {
                    out[(size_t)r * rows + row] = v * final_scale;
                }
            }
        }
    }
}

extern "C" __global__ void m256_bf16(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const int rows, const int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const unsigned short* __restrict__ w = W + (size_t)row * cols;
    float acc[NRHS];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r) acc[r] = 0.0f;

    for (int k = tid; k < cols; k += 256) {
        const float ww = m256_bf16_to_f32(w[k]);
        #pragma unroll
        for (int r = 0; r < NRHS; ++r)
            acc[r] = fmaf(ww, X[(size_t)r * cols + k], acc[r]);
    }
    m256_reduce_write(acc, out, row, rows, 1.0f, 0);
}

extern "C" __global__ void m256_f32(
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const int rows, const int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const float* __restrict__ w = W + (size_t)row * cols;
    float acc[NRHS];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r) acc[r] = 0.0f;

    for (int k = tid; k < cols; k += 256) {
        const float ww = w[k];
        #pragma unroll
        for (int r = 0; r < NRHS; ++r)
            acc[r] = fmaf(ww, X[(size_t)r * cols + k], acc[r]);
    }
    m256_reduce_write(acc, out, row, rows, 1.0f, 0);
}

extern "C" __global__ void m256_fp8_tensor(
    const unsigned char* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const float wscale,
    const int rows, const int cols)
{
    __shared__ float lut[256];
    if (threadIdx.x < 256) lut[threadIdx.x] = m256_e4m3_decode((unsigned char)threadIdx.x);
    __syncthreads();

    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const unsigned char* __restrict__ w = W + (size_t)row * cols;
    const uchar4* __restrict__ w4 = reinterpret_cast<const uchar4*>(w);
    const int nvec = cols >> 2;
    float acc[NRHS];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r) acc[r] = 0.0f;

    for (int qidx = tid; qidx < nvec; qidx += 256) {
        const uchar4 q = w4[qidx];
        const int k = qidx << 2;
        const float w0 = lut[q.x], w1 = lut[q.y], w2 = lut[q.z], w3 = lut[q.w];
        #pragma unroll
        for (int r = 0; r < NRHS; ++r) {
            const float* xr = X + (size_t)r * cols;
            float a = acc[r];
            a = fmaf(w0, xr[k],     a);
            a = fmaf(w1, xr[k + 1], a);
            a = fmaf(w2, xr[k + 2], a);
            a = fmaf(w3, xr[k + 3], a);
            acc[r] = a;
        }
    }
    for (int b = (nvec << 2) + tid; b < cols; b += 256) {
        const float ww = lut[w[b]];
        #pragma unroll
        for (int r = 0; r < NRHS; ++r)
            acc[r] = fmaf(ww, X[(size_t)r * cols + b], acc[r]);
    }
    m256_reduce_write(acc, out, row, rows, wscale, 0);
}

extern "C" __global__ void m256_nvfp4(
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

    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int n_bytes = cols >> 1;
    const int n_vec = n_bytes >> 2;
    const unsigned char* __restrict__ crow = codes + (size_t)row * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)row * (cols >> 4);
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);
    float acc[NRHS];
    #pragma unroll
    for (int r = 0; r < NRHS; ++r) acc[r] = 0.0f;

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
            float a = acc[r];
            a = fmaf(w0,  xr[k],     a);
            a = fmaf(w1,  xr[k + 1], a);
            a = fmaf(w2,  xr[k + 2], a);
            a = fmaf(w3,  xr[k + 3], a);
            a = fmaf(w4v, xr[k + 4], a);
            a = fmaf(w5,  xr[k + 5], a);
            a = fmaf(w6,  xr[k + 6], a);
            a = fmaf(w7,  xr[k + 7], a);
            acc[r] = a;
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
            float a = acc[r];
            a = fmaf(w0, xr[k],     a);
            a = fmaf(w1, xr[k + 1], a);
            acc[r] = a;
        }
    }

    // NVFP4's reference writes either relu2(v) or v*out_scale.  The global
    // dequant scale was already applied to every weight above.
    m256_reduce_write(acc, out, row, rows, out_scale, apply_relu2);
}
"""


class ExactMRHS256:
    def __init__(self, rhs_values=SUPPORTED_RHS):
        import cupy as cp

        self.cp = cp
        self.mods = {}
        self.kernels = {}
        for n in rhs_values:
            n = int(n)
            if n not in SUPPORTED_RHS:
                raise ValueError(f"unsupported NRHS={n}; supported={SUPPORTED_RHS}")
            mod = cp.RawModule(code=_TEMPLATE.replace("@NRHS@", str(n)), options=("-std=c++14",))
            self.mods[n] = mod
            self.kernels[n] = {
                "bf16": mod.get_function("m256_bf16"),
                "f32": mod.get_function("m256_f32"),
                "fp8": mod.get_function("m256_fp8_tensor"),
                "nvfp4": mod.get_function("m256_nvfp4"),
            }

    def bf16(self, n, out, W, X, rows, cols):
        self.kernels[int(n)]["bf16"]((int(rows),), (256,),
            (W, X, out, np.int32(rows), np.int32(cols)))

    def f32(self, n, out, W, X, rows, cols):
        self.kernels[int(n)]["f32"]((int(rows),), (256,),
            (W, X, out, np.int32(rows), np.int32(cols)))

    def fp8(self, n, out, W, X, scale, rows, cols):
        self.kernels[int(n)]["fp8"]((int(rows),), (256,),
            (W, X, out, np.float32(scale), np.int32(rows), np.int32(cols)),
            shared_mem=256 * 4)

    def nvfp4(self, n, out, codes, scales, e2m1, e4m3, X, global_scale,
              rows, cols, apply_relu2=False, out_scale=1.0):
        self.kernels[int(n)]["nvfp4"]((int(rows),), (256,),
            (codes, scales, e2m1, e4m3, X, out, np.float32(global_scale),
             np.int32(rows), np.int32(cols), np.int32(1 if apply_relu2 else 0),
             np.float32(out_scale)))
