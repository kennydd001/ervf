"""Exact two-RHS ERVF kernels with streamed virtual accumulators.

Unlike the earlier generic MRHS kernels, the main MAC loop keeps only two live
accumulators at once (one per RHS).  Completed reference-virtual-thread leaves
are written to shared memory and the original 256-thread reduction tree is
reconstructed afterwards.  Weight/dequant scalars are loaded once and reused by
both RHS streams; each RHS preserves its own reference FMA order.

This module is additive research code.  It does not patch the runtime.
See S100_DUALRHS_ERVF_PREREGISTRATION.md.
"""
from __future__ import annotations

import numpy as np

WIDTH = 16
ROWS_PER_BLOCK = 6
BLOCK = WIDTH * ROWS_PER_BLOCK  # 96 = 3 complete warps / 6 half-warp rows
VIRTUAL = 256 // WIDTH          # 16 virtual reference tids per physical lane

CUDA_SOURCE = r"""
#define D_WIDTH 16
#define D_ROWS 6
#define D_VIRTUAL 16
#define D_REF_THREADS 256

__device__ __forceinline__ float d_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float d_e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

// leaves contains the COMPLETE accumulator of every one of the reference
// kernel's 256 logical threads for one row/RHS.  Rebuild the exact production
// two-level reduction tree using the already-proven ERVF width-16 mapping.
__device__ __forceinline__ float d_reduce_exact(const float* __restrict__ leaves) {
    const int lane = (int)threadIdx.x & (D_WIDTH - 1);
    float part[D_VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < D_VIRTUAL; ++vi)
        part[vi] = leaves[lane + D_WIDTH * vi];

    float s8[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        // Reference warp offset-16 pair.  The remaining 8/4/2/1 offsets are
        // the 16-wide physical half-warp shuffle tree.
        float v = part[w * 2] + part[w * 2 + 1];
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, D_WIDTH);
        s8[w] = v;
    }
    if (lane == 0) {
        // Exact second-stage reference warp tree over the eight warp sums.
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

// Dynamic shared layout for every kernel:
//   X0[cols], X1[cols], leaves[D_ROWS][2][256]
// Maximum registered cols is 4096 => 45,056 dynamic bytes.

extern "C" __global__ void dual_bf16(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,      // [2, cols]
    float* __restrict__ out,          // [2, rows]
    const int rows, const int cols)
{
    extern __shared__ float sm[];
    float* x0 = sm;
    float* x1 = sm + cols;
    float* leaves = sm + 2 * cols;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        x0[k] = X[k];
        x1[k] = X[(size_t)cols + k];
    }
    __syncthreads();

    const int sub = (int)threadIdx.x / D_WIDTH;
    const int lane = (int)threadIdx.x & (D_WIDTH - 1);
    const int row = (int)blockIdx.x * D_ROWS + sub;
    const bool valid = row < rows;
    const unsigned short* __restrict__ w = W + (size_t)(valid ? row : 0) * cols;
    float* l0 = leaves + ((size_t)sub * 2 + 0) * D_REF_THREADS;
    float* l1 = leaves + ((size_t)sub * 2 + 1) * D_REF_THREADS;

    #pragma unroll
    for (int vi = 0; vi < D_VIRTUAL; ++vi) {
        const int tid = lane + D_WIDTH * vi;
        float a0 = 0.0f, a1 = 0.0f;
        if (valid) {
            for (int k = tid; k < cols; k += D_REF_THREADS) {
                const float ww = d_bf16_to_f32(w[k]);
                a0 = fmaf(ww, x0[k], a0);
                a1 = fmaf(ww, x1[k], a1);
            }
        }
        l0[tid] = a0;
        l1[tid] = a1;
    }
    __syncthreads();

    if (valid) {
        const float v0 = d_reduce_exact(l0);
        const float v1 = d_reduce_exact(l1);
        if (lane == 0) {
            out[row] = v0;
            out[(size_t)rows + row] = v1;
        }
    }
}

extern "C" __global__ void dual_f32(
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const int rows, const int cols)
{
    extern __shared__ float sm[];
    float* x0 = sm;
    float* x1 = sm + cols;
    float* leaves = sm + 2 * cols;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        x0[k] = X[k];
        x1[k] = X[(size_t)cols + k];
    }
    __syncthreads();

    const int sub = (int)threadIdx.x / D_WIDTH;
    const int lane = (int)threadIdx.x & (D_WIDTH - 1);
    const int row = (int)blockIdx.x * D_ROWS + sub;
    const bool valid = row < rows;
    const float* __restrict__ w = W + (size_t)(valid ? row : 0) * cols;
    float* l0 = leaves + ((size_t)sub * 2 + 0) * D_REF_THREADS;
    float* l1 = leaves + ((size_t)sub * 2 + 1) * D_REF_THREADS;

    #pragma unroll
    for (int vi = 0; vi < D_VIRTUAL; ++vi) {
        const int tid = lane + D_WIDTH * vi;
        float a0 = 0.0f, a1 = 0.0f;
        if (valid) {
            for (int k = tid; k < cols; k += D_REF_THREADS) {
                const float ww = w[k];
                a0 = fmaf(ww, x0[k], a0);
                a1 = fmaf(ww, x1[k], a1);
            }
        }
        l0[tid] = a0;
        l1[tid] = a1;
    }
    __syncthreads();

    if (valid) {
        const float v0 = d_reduce_exact(l0);
        const float v1 = d_reduce_exact(l1);
        if (lane == 0) {
            out[row] = v0;
            out[(size_t)rows + row] = v1;
        }
    }
}

extern "C" __global__ void dual_fp8_tensor(
    const unsigned char* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const float wscale,
    const int rows, const int cols)
{
    extern __shared__ float sm[];
    float* x0 = sm;
    float* x1 = sm + cols;
    float* leaves = sm + 2 * cols;
    __shared__ float lut[256];
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        x0[k] = X[k];
        x1[k] = X[(size_t)cols + k];
    }
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = d_e4m3_decode((unsigned char)i);
    __syncthreads();

    const int sub = (int)threadIdx.x / D_WIDTH;
    const int lane = (int)threadIdx.x & (D_WIDTH - 1);
    const int row = (int)blockIdx.x * D_ROWS + sub;
    const bool valid = row < rows;
    const unsigned char* __restrict__ w = W + (size_t)(valid ? row : 0) * cols;
    const uchar4* __restrict__ w4 = reinterpret_cast<const uchar4*>(w);
    const int nvec = cols >> 2;
    float* l0 = leaves + ((size_t)sub * 2 + 0) * D_REF_THREADS;
    float* l1 = leaves + ((size_t)sub * 2 + 1) * D_REF_THREADS;

    #pragma unroll
    for (int vi = 0; vi < D_VIRTUAL; ++vi) {
        const int tid = lane + D_WIDTH * vi;
        float a0 = 0.0f, a1 = 0.0f;
        if (valid) {
            for (int qidx = tid; qidx < nvec; qidx += D_REF_THREADS) {
                const uchar4 q = w4[qidx];
                const int k = qidx << 2;
                const float w0 = lut[q.x], w1 = lut[q.y], w2 = lut[q.z], w3 = lut[q.w];
                a0 = fmaf(w0, x0[k],     a0);
                a0 = fmaf(w1, x0[k + 1], a0);
                a0 = fmaf(w2, x0[k + 2], a0);
                a0 = fmaf(w3, x0[k + 3], a0);
                a1 = fmaf(w0, x1[k],     a1);
                a1 = fmaf(w1, x1[k + 1], a1);
                a1 = fmaf(w2, x1[k + 2], a1);
                a1 = fmaf(w3, x1[k + 3], a1);
            }
            for (int b = (nvec << 2) + tid; b < cols; b += D_REF_THREADS) {
                const float ww = lut[w[b]];
                a0 = fmaf(ww, x0[b], a0);
                a1 = fmaf(ww, x1[b], a1);
            }
        }
        l0[tid] = a0;
        l1[tid] = a1;
    }
    __syncthreads();

    if (valid) {
        const float v0 = d_reduce_exact(l0) * wscale;
        const float v1 = d_reduce_exact(l1) * wscale;
        if (lane == 0) {
            out[row] = v0;
            out[(size_t)rows + row] = v1;
        }
    }
}

extern "C" __global__ void dual_nvfp4(
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
    extern __shared__ float sm[];
    float* x0 = sm;
    float* x1 = sm + cols;
    float* leaves = sm + 2 * cols;
    __shared__ float s_e2m1[16];
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        x0[k] = X[k];
        x1[k] = X[(size_t)cols + k];
    }
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int sub = (int)threadIdx.x / D_WIDTH;
    const int lane = (int)threadIdx.x & (D_WIDTH - 1);
    const int row = (int)blockIdx.x * D_ROWS + sub;
    const bool valid = row < rows;
    const int n_bytes = cols >> 1;
    const int n_vec = n_bytes >> 2;
    const unsigned char* __restrict__ crow = codes + (size_t)(valid ? row : 0) * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)(valid ? row : 0) * (cols >> 4);
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);
    float* l0 = leaves + ((size_t)sub * 2 + 0) * D_REF_THREADS;
    float* l1 = leaves + ((size_t)sub * 2 + 1) * D_REF_THREADS;

    #pragma unroll
    for (int vi = 0; vi < D_VIRTUAL; ++vi) {
        const int tid = lane + D_WIDTH * vi;
        float a0 = 0.0f, a1 = 0.0f;
        if (valid) {
            for (int v = tid; v < n_vec; v += D_REF_THREADS) {
                const uchar4 q = crow4[v];
                const int b = v << 2;
                const float s = e4m3_lut[srow[b >> 3]] * global_scale;
                const int k = b << 1;
                const float w0 = s_e2m1[q.x & 0x0F] * s;
                const float w1 = s_e2m1[q.x >> 4]   * s;
                const float w2 = s_e2m1[q.y & 0x0F] * s;
                const float w3 = s_e2m1[q.y >> 4]   * s;
                const float w4 = s_e2m1[q.z & 0x0F] * s;
                const float w5 = s_e2m1[q.z >> 4]   * s;
                const float w6 = s_e2m1[q.w & 0x0F] * s;
                const float w7 = s_e2m1[q.w >> 4]   * s;
                a0 = fmaf(w0, x0[k],     a0); a0 = fmaf(w1, x0[k + 1], a0);
                a0 = fmaf(w2, x0[k + 2], a0); a0 = fmaf(w3, x0[k + 3], a0);
                a0 = fmaf(w4, x0[k + 4], a0); a0 = fmaf(w5, x0[k + 5], a0);
                a0 = fmaf(w6, x0[k + 6], a0); a0 = fmaf(w7, x0[k + 7], a0);
                a1 = fmaf(w0, x1[k],     a1); a1 = fmaf(w1, x1[k + 1], a1);
                a1 = fmaf(w2, x1[k + 2], a1); a1 = fmaf(w3, x1[k + 3], a1);
                a1 = fmaf(w4, x1[k + 4], a1); a1 = fmaf(w5, x1[k + 5], a1);
                a1 = fmaf(w6, x1[k + 6], a1); a1 = fmaf(w7, x1[k + 7], a1);
            }
            for (int b = (n_vec << 2) + tid; b < n_bytes; b += D_REF_THREADS) {
                const unsigned char byte = crow[b];
                const float s = e4m3_lut[srow[b >> 3]] * global_scale;
                const int k = b << 1;
                const float w0 = s_e2m1[byte & 0x0F] * s;
                const float w1 = s_e2m1[byte >> 4] * s;
                a0 = fmaf(w0, x0[k], a0); a0 = fmaf(w1, x0[k + 1], a0);
                a1 = fmaf(w0, x1[k], a1); a1 = fmaf(w1, x1[k + 1], a1);
            }
        }
        l0[tid] = a0;
        l1[tid] = a1;
    }
    __syncthreads();

    if (valid) {
        const float v0 = d_reduce_exact(l0);
        const float v1 = d_reduce_exact(l1);
        if (lane == 0) {
            if (apply_relu2) {
                const float z0 = fmaxf(v0, 0.0f), z1 = fmaxf(v1, 0.0f);
                out[row] = z0 * z0;
                out[(size_t)rows + row] = z1 * z1;
            } else {
                out[row] = v0 * out_scale;
                out[(size_t)rows + row] = v1 * out_scale;
            }
        }
    }
}
"""


class DualRHSSVAE:
    """Two-RHS streamed-virtual-accumulator ERVF primitive."""

    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14", "--use_fast_math"))
        self.k_bf16 = self.mod.get_function("dual_bf16")
        self.k_f32 = self.mod.get_function("dual_f32")
        self.k_fp8 = self.mod.get_function("dual_fp8_tensor")
        self.k_nvfp4 = self.mod.get_function("dual_nvfp4")

    @staticmethod
    def shared_bytes(cols: int) -> int:
        return int((2 * int(cols) + ROWS_PER_BLOCK * 2 * 256) * 4)

    @staticmethod
    def grid(rows: int):
        return ((int(rows) + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK,)

    def bf16(self, out, W, X, rows: int, cols: int) -> None:
        self.k_bf16(self.grid(rows), (BLOCK,),
                    (W, X, out, np.int32(rows), np.int32(cols)),
                    shared_mem=self.shared_bytes(cols))

    def f32(self, out, W, X, rows: int, cols: int) -> None:
        self.k_f32(self.grid(rows), (BLOCK,),
                   (W, X, out, np.int32(rows), np.int32(cols)),
                   shared_mem=self.shared_bytes(cols))

    def fp8(self, out, W, X, wscale: float, rows: int, cols: int) -> None:
        self.k_fp8(self.grid(rows), (BLOCK,),
                   (W, X, out, np.float32(wscale), np.int32(rows), np.int32(cols)),
                   shared_mem=self.shared_bytes(cols))

    def nvfp4(self, out, codes, scales, e2m1, e4m3, X, global_scale: float,
              rows: int, cols: int, *, apply_relu2: bool = False,
              out_scale: float = 1.0) -> None:
        self.k_nvfp4(self.grid(rows), (BLOCK,),
                     (codes, scales, e2m1, e4m3, X, out,
                      np.float32(global_scale), np.int32(rows), np.int32(cols),
                      np.int32(1 if apply_relu2 else 0), np.float32(out_scale)),
                     shared_mem=self.shared_bytes(cols))

    def attributes(self) -> dict:
        result = {}
        for name, k in (("bf16", self.k_bf16), ("f32", self.k_f32),
                        ("fp8", self.k_fp8), ("nvfp4", self.k_nvfp4)):
            try:
                result[name] = dict(k.attributes)
            except Exception as exc:
                result[name] = {"error": f"{type(exc).__name__}: {exc}"}
        return result


def cpu_reduction_selftest(trials: int = 1000) -> dict:
    """Independent CPU emulation of the exact width-16 leaf reduction."""
    from ervf_dense import _reference_reduce

    rng = np.random.default_rng(0xD0A1BEEF)
    mismatches = 0
    examples = []
    for t in range(trials):
        leaves = (rng.standard_normal(256) * (10.0 ** rng.uniform(-5.0, 5.0))).astype(np.float32)
        ref = _reference_reduce(leaves)

        lane_warp = np.zeros((16, 8), dtype=np.float32)
        for lane in range(16):
            for w in range(8):
                a = leaves[lane + 32 * w]
                b = leaves[lane + 16 + 32 * w]
                lane_warp[lane, w] = np.float32(np.float64(a) + np.float64(b))
        for w in range(8):
            v = lane_warp[:, w].copy()
            for off in (8, 4, 2, 1):
                old = v.copy()
                for lane in range(16 - off):
                    v[lane] = np.float32(np.float64(old[lane]) + np.float64(old[lane + off]))
            lane_warp[:, w] = v
        s = lane_warp[0]
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
                examples.append({"trial": t, "ref_bits": int(ref.view(np.uint32)),
                                 "got_bits": int(got.view(np.uint32))})
    return {"trials": trials, "mismatches": mismatches,
            "examples": examples, "passed": mismatches == 0}
