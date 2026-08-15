"""E4 -- Attention Roofline Recovery runner (agent 21).

Component-level measurements only. Standalone harness: seeded random fp8-e4m3
KV caches + random q, no model weights needed. Compares the registered v1
kernel (attn_decode_warp_fp8_gqa) against the newly registered v2
(attn_decode_warp_fp8_gqa2) and decomposes v2 into bandwidth/QK/softmax/PV
stages. Gates are frozen in
reports/treesweep200/E4_ATTENTION_ROOFLINE_PREREGISTRATION_2026-08-15.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cupy as cp  # noqa: E402

from moe_lab.lightningstream_nemotron.gpu_kernels import GPUKernels  # noqa: E402

CONTEXTS = [64, 4096, 32768, 131072, 262144]
N_HEADS, N_KV, HEAD_DIM, GROUPS = 32, 2, 128, 16
MAX_CTX = max(CONTEXTS)
MAX_SPLITS = 256
REPS, WARMUP = 20, 3
SEEDS = [20260815, 20260816, 20260817]

_PROFILE_SRC = r"""
__device__ __forceinline__ float e4m3_decode_p(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = ldexpf((float)m / 8.0f, -6);
    else v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return s ? -v : v;
}

// raw contiguous scan: pure sequential read of K+V over the chunk range.
extern "C" __global__ void raw_scan(
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ sink, const int t, const int max_ctx, const int chunk)
{
    const int g = blockIdx.x, s = blockIdx.y;
    const int j0 = s * chunk, j1 = min(t, j0 + chunk);
    const unsigned char* kb = Kc + (size_t)g * max_ctx * 128;
    const unsigned char* vb = Vc + (size_t)g * max_ctx * 128;
    unsigned int acc = 0u;
    for (int i = j0 * 32 + threadIdx.x; i < j1 * 32; i += blockDim.x) {
        const uint4 k4 = reinterpret_cast<const uint4*>(kb)[i];
        const uint4 v4 = reinterpret_cast<const uint4*>(vb)[i];
        acc ^= k4.x ^ k4.y ^ k4.z ^ k4.w ^ v4.x ^ v4.y ^ v4.z ^ v4.w;
    }
    for (int o = 16; o > 0; o >>= 1)
        acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    __shared__ unsigned int red[4];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        sink[blockIdx.y * gridDim.x + blockIdx.x] =
            (float)(red[0] ^ red[1] ^ red[2] ^ red[3]);
}

// address scan: the exact gqa2 access pattern (warp-strided rows, per-lane
// 64-byte halves via uint4) without any decode or arithmetic.
extern "C" __global__ void addr_scan(
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ sink, const int t, const int max_ctx, const int chunk)
{
    const int g = blockIdx.x, s = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hf = lane & 1;
    const int j0 = s * chunk, j1 = min(t, j0 + chunk);
    const unsigned char* kb = Kc + (size_t)g * max_ctx * 128;
    const unsigned char* vb = Vc + (size_t)g * max_ctx * 128;
    unsigned int acc = 0u;
    for (int j = j0 + warp; j < j1; j += 4) {
        const uint4* kr = reinterpret_cast<const uint4*>(kb + (size_t)j * 128 + hf * 64);
        const uint4* vr = reinterpret_cast<const uint4*>(vb + (size_t)j * 128 + hf * 64);
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint4 k4 = kr[u], v4 = vr[u];
            acc ^= k4.x ^ k4.y ^ k4.z ^ k4.w ^ v4.x ^ v4.y ^ v4.z ^ v4.w;
        }
    }
    for (int o = 16; o > 0; o >>= 1)
        acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    __shared__ unsigned int red[4];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        sink[blockIdx.y * gridDim.x + blockIdx.x] =
            (float)(red[0] ^ red[1] ^ red[2] ^ red[3]);
}

// QK only: gqa2 lane assignment, K decode + dot + one shuffle, no softmax/PV.
extern "C" __global__ void qk_only(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc,
    float* __restrict__ sink, const int t, const int max_ctx,
    const float scale, const int chunk)
{
    const int g = blockIdx.x, s = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hh = lane >> 1, hf = lane & 1;
    __shared__ float lut[256];
    __shared__ float qs[16 * 132];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode_p((unsigned char)i);
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[(i >> 7) * 132 + (i & 127)] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();
    const int j0 = s * chunk, j1 = min(t, j0 + chunk);
    const unsigned char* kb = Kc + (size_t)g * max_ctx * 128;
    const float* qh = qs + hh * 132 + hf * 64;
    float acc = 0.0f;
    for (int j = j0 + warp; j < j1; j += 4) {
        const uint4* kr = reinterpret_cast<const uint4*>(kb + (size_t)j * 128 + hf * 64);
        float part = 0.0f;
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint4 kv = kr[u];
            const unsigned char* kb4 = reinterpret_cast<const unsigned char*>(&kv);
            #pragma unroll
            for (int b = 0; b < 16; b++)
                part = fmaf(qh[u * 16 + b], lut[kb4[b]], part);
        }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        acc += part * scale;
    }
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __shared__ float red[4];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        sink[blockIdx.y * gridDim.x + blockIdx.x] = red[0] + red[1] + red[2] + red[3];
}

// QK + online softmax update, no PV accumulation.
extern "C" __global__ void qk_softmax(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc,
    float* __restrict__ sink, const int t, const int max_ctx,
    const float scale, const int chunk)
{
    const int g = blockIdx.x, s = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hh = lane >> 1, hf = lane & 1;
    __shared__ float lut[256];
    __shared__ float qs[16 * 132];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode_p((unsigned char)i);
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[(i >> 7) * 132 + (i & 127)] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();
    const int j0 = s * chunk, j1 = min(t, j0 + chunk);
    const unsigned char* kb = Kc + (size_t)g * max_ctx * 128;
    const float* qh = qs + hh * 132 + hf * 64;
    float m = -3.0e38f, l = 0.0f, acc = 0.0f;
    for (int j = j0 + warp; j < j1; j += 4) {
        const uint4* kr = reinterpret_cast<const uint4*>(kb + (size_t)j * 128 + hf * 64);
        float part = 0.0f;
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint4 kv = kr[u];
            const unsigned char* kb4 = reinterpret_cast<const unsigned char*>(&kv);
            #pragma unroll
            for (int b = 0; b < 16; b++)
                part = fmaf(qh[u * 16 + b], lut[kb4[b]], part);
        }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        const float sc = part * scale;
        const float m_new = fmaxf(m, sc);
        const float corr = __expf(m - m_new);
        const float p = __expf(sc - m_new);
        l = l * corr + p;
        m = m_new;
        acc += p;
    }
    acc += l;
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __shared__ float red[4];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        sink[blockIdx.y * gridDim.x + blockIdx.x] = red[0] + red[1] + red[2] + red[3];
}

// ---- ablation kernels: v4 structure with one component removed each. ----
// MODE: 0 = full, 1 = no butterfly shuffle, 2 = no exp, 3 = no PV.
__device__ __forceinline__ void e4m3x4_f32_p(const uchar4 c, float* f) {
    const unsigned short p0 = (unsigned short)((unsigned short)c.x |
                                               ((unsigned short)c.y << 8));
    const unsigned short p1 = (unsigned short)((unsigned short)c.z |
                                               ((unsigned short)c.w << 8));
    unsigned h0, h1;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h0) : "h"(p0));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h1) : "h"(p1));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f[0]) : "h"((unsigned short)(h0 & 0xffffu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f[1]) : "h"((unsigned short)(h0 >> 16)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f[2]) : "h"((unsigned short)(h1 & 0xffffu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f[3]) : "h"((unsigned short)(h1 >> 16)));
}

template <int MODE>
__device__ __forceinline__ void abl_body(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ sink, const int t, const int max_ctx,
    const float scale, const int chunk)
{
    const int g = blockIdx.x, s = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float qs[16 * 128];
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[i] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();
    const int j0 = s * chunk, j1 = min(t, j0 + chunk);
    const uchar4* kb = reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * 128);
    const uchar4* vb = reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * 128);
    float m[16], l[16], a[16][4];
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        m[hh] = -3.0e38f; l[hh] = 0.0f;
        a[hh][0] = a[hh][1] = a[hh][2] = a[hh][3] = 0.0f;
    }
    const int d0 = lane << 2;
    for (int j = j0 + warp; j + 4 < j1; j += 8) {
        const uchar4 k4a = kb[(size_t)j * 32 + lane];
        const uchar4 v4a = vb[(size_t)j * 32 + lane];
        const uchar4 k4b = kb[(size_t)(j + 4) * 32 + lane];
        const uchar4 v4b = vb[(size_t)(j + 4) * 32 + lane];
        float kfa[4], vfa[4], kfb[4], vfb[4];
        e4m3x4_f32_p(k4a, kfa); e4m3x4_f32_p(k4b, kfb);
        float ca[16], pa[16], cb[16], pb[16];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float partA = qs[hh * 128 + d0]     * kfa[0] + qs[hh * 128 + d0 + 1] * kfa[1]
                        + qs[hh * 128 + d0 + 2] * kfa[2] + qs[hh * 128 + d0 + 3] * kfa[3];
            float partB = qs[hh * 128 + d0]     * kfb[0] + qs[hh * 128 + d0 + 1] * kfb[1]
                        + qs[hh * 128 + d0 + 2] * kfb[2] + qs[hh * 128 + d0 + 3] * kfb[3];
            if (MODE != 1) {
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    partA += __shfl_xor_sync(0xffffffffu, partA, o);
                    partB += __shfl_xor_sync(0xffffffffu, partB, o);
                }
            }
            const float scA = partA * scale, scB = partB * scale;
            const float mA = fmaxf(m[hh], scA);
            if (MODE == 2) { ca[hh] = m[hh] - mA; pa[hh] = scA - mA; }
            else { ca[hh] = __expf(m[hh] - mA); pa[hh] = __expf(scA - mA); }
            l[hh] = l[hh] * ca[hh] + pa[hh];
            m[hh] = mA;
            const float mB = fmaxf(m[hh], scB);
            if (MODE == 2) { cb[hh] = m[hh] - mB; pb[hh] = scB - mB; }
            else { cb[hh] = __expf(m[hh] - mB); pb[hh] = __expf(scB - mB); }
            l[hh] = l[hh] * cb[hh] + pb[hh];
            m[hh] = mB;
        }
        if (MODE != 3) {
            e4m3x4_f32_p(v4a, vfa); e4m3x4_f32_p(v4b, vfb);
            #pragma unroll
            for (int hh = 0; hh < 16; hh++) {
                a[hh][0] = a[hh][0] * ca[hh] + pa[hh] * vfa[0];
                a[hh][1] = a[hh][1] * ca[hh] + pa[hh] * vfa[1];
                a[hh][2] = a[hh][2] * ca[hh] + pa[hh] * vfa[2];
                a[hh][3] = a[hh][3] * ca[hh] + pa[hh] * vfa[3];
            }
            #pragma unroll
            for (int hh = 0; hh < 16; hh++) {
                a[hh][0] = a[hh][0] * cb[hh] + pb[hh] * vfb[0];
                a[hh][1] = a[hh][1] * cb[hh] + pb[hh] * vfb[1];
                a[hh][2] = a[hh][2] * cb[hh] + pb[hh] * vfb[2];
                a[hh][3] = a[hh][3] * cb[hh] + pb[hh] * vfb[3];
            }
        } else {
            vfa[0] = 0.0f; vfb[0] = 0.0f;  // keep loads alive
            a[0][0] += vfa[0] + vfb[0];
        }
    }
    float r = 0.0f;
    #pragma unroll
    for (int hh = 0; hh < 16; hh++)
        r += l[hh] + a[hh][0] + a[hh][1] + a[hh][2] + a[hh][3];
    for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    __shared__ float red[4];
    if (lane == 0) red[warp] = r;
    __syncthreads();
    if (threadIdx.x == 0)
        sink[blockIdx.y * gridDim.x + blockIdx.x] = red[0] + red[1] + red[2] + red[3];
}

extern "C" __global__ void abl_full(
    const float* __restrict__ q, const unsigned char* __restrict__ Kc,
    const unsigned char* __restrict__ Vc, float* __restrict__ sink,
    const int t, const int max_ctx, const float scale, const int chunk)
{ abl_body<0>(q, Kc, Vc, sink, t, max_ctx, scale, chunk); }

extern "C" __global__ void abl_noshuf(
    const float* __restrict__ q, const unsigned char* __restrict__ Kc,
    const unsigned char* __restrict__ Vc, float* __restrict__ sink,
    const int t, const int max_ctx, const float scale, const int chunk)
{ abl_body<1>(q, Kc, Vc, sink, t, max_ctx, scale, chunk); }

extern "C" __global__ void abl_noexp(
    const float* __restrict__ q, const unsigned char* __restrict__ Kc,
    const unsigned char* __restrict__ Vc, float* __restrict__ sink,
    const int t, const int max_ctx, const float scale, const int chunk)
{ abl_body<2>(q, Kc, Vc, sink, t, max_ctx, scale, chunk); }

extern "C" __global__ void abl_nopv(
    const float* __restrict__ q, const unsigned char* __restrict__ Kc,
    const unsigned char* __restrict__ Vc, float* __restrict__ sink,
    const int t, const int max_ctx, const float scale, const int chunk)
{ abl_body<3>(q, Kc, Vc, sink, t, max_ctx, scale, chunk); }
"""


def _fill_fp8(arr: cp.ndarray, seed: int) -> None:
    """Random bytes with the e4m3 NaN patterns (S.1111.111) remapped."""
    rs = cp.random.RandomState(seed)
    raw = rs.randint(0, 256, size=arr.size, dtype=np.uint8)
    b = raw.reshape(arr.shape)
    nan_pat = (b & 0x7F) == 0x7F
    b[nan_pat] &= 0xFE
    arr[:] = b


def _time(fn, reps=REPS):
    for _ in range(WARMUP):
        fn()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(reps):
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        times.append(cp.cuda.get_elapsed_time(e0, e1))
    return float(np.median(times))


def main() -> int:
    k = GPUKernels()
    prof = cp.RawModule(code=_PROFILE_SRC, options=("-std=c++14", "--use_fast_math"))
    raw_scan = prof.get_function("raw_scan")
    addr_scan = prof.get_function("addr_scan")
    qk_only_f = prof.get_function("qk_only")
    qk_softmax_f = prof.get_function("qk_softmax")

    rs = cp.random.RandomState(SEEDS[0])
    Kc = cp.zeros(N_KV * MAX_CTX * HEAD_DIM, dtype=cp.uint8)
    Vc = cp.zeros(N_KV * MAX_CTX * HEAD_DIM, dtype=cp.uint8)
    _fill_fp8(Kc, SEEDS[0] + 1)
    _fill_fp8(Vc, SEEDS[0] + 2)
    q = rs.standard_normal(N_HEADS * HEAD_DIM, dtype=np.float32)
    part_acc = cp.zeros(N_HEADS * MAX_SPLITS * 4 * HEAD_DIM, dtype=cp.float32)
    part_ml = cp.zeros(N_HEADS * MAX_SPLITS * 4 * 2, dtype=cp.float32)
    out = cp.zeros(N_HEADS * HEAD_DIM, dtype=cp.float32)
    sink = cp.zeros(N_KV * MAX_SPLITS, dtype=cp.float32)
    scale = 1.0 / float(np.sqrt(HEAD_DIM))

    results: dict = {"contexts": {}, "correctness": [], "determinism": None}

    for t in CONTEXTS:
        splits = max(1, min(MAX_SPLITS, (t + 511) // 512))
        chunk = (t + splits - 1) // splits
        grid = (N_KV, splits)
        nbytes = 2 * N_KV * t * HEAD_DIM  # K+V, both kv heads

        entry: dict = {"t": t, "splits": splits, "bytes": nbytes}

        entry["raw_scan_ms"] = _time(lambda: raw_scan(
            grid, (128,), (Kc, Vc, sink, np.int32(t), np.int32(MAX_CTX),
                           np.int32(chunk))))
        entry["addr_scan_ms"] = _time(lambda: addr_scan(
            grid, (128,), (Kc, Vc, sink, np.int32(t), np.int32(MAX_CTX),
                           np.int32(chunk))))
        entry["qk_only_ms"] = _time(lambda: qk_only_f(
            grid, (128,), (q, Kc, sink, np.int32(t), np.int32(MAX_CTX),
                           np.float32(scale), np.int32(chunk))))
        entry["qk_softmax_ms"] = _time(lambda: qk_softmax_f(
            grid, (128,), (q, Kc, sink, np.int32(t), np.int32(MAX_CTX),
                           np.float32(scale), np.int32(chunk))))

        if t == 262144:
            for name in ("abl_full", "abl_noshuf", "abl_noexp", "abl_nopv"):
                fn = prof.get_function(name)
                entry[name + "_ms"] = _time(lambda fn=fn: fn(
                    grid, (128,),
                    (q, Kc, Vc, sink, np.int32(t), np.int32(MAX_CTX),
                     np.float32(scale), np.int32(chunk))))

        # v1 / v2 full kernels (kernel only) and full path (+ combine).
        def v1_kernel():
            k.attn_decode_warp_fp8_gqa(
                grid, (128,),
                (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(HEAD_DIM),
                 np.int32(GROUPS), np.int32(MAX_CTX), np.float32(scale),
                 np.int32(chunk)))

        def v2_kernel():
            k.attn_decode_warp_fp8_gqa2(
                grid, (128,),
                (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(HEAD_DIM),
                 np.int32(GROUPS), np.int32(MAX_CTX), np.float32(scale),
                 np.int32(chunk)))

        def v3_kernel():
            k.attn_decode_warp_fp8_gqa3(
                grid, (128,),
                (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(HEAD_DIM),
                 np.int32(GROUPS), np.int32(MAX_CTX), np.float32(scale),
                 np.int32(chunk)))

        def v4_kernel():
            k.attn_decode_warp_fp8_gqa4(
                grid, (128,),
                (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(HEAD_DIM),
                 np.int32(GROUPS), np.int32(MAX_CTX), np.float32(scale),
                 np.int32(chunk)))

        def v6_kernel():
            k.attn_decode_warp_fp8_gqa6(
                grid, (128,),
                (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(HEAD_DIM),
                 np.int32(GROUPS), np.int32(MAX_CTX), np.float32(scale),
                 np.int32(chunk)))

        def v7_kernel():
            k.attn_decode_warp_fp8_gqa7(
                grid, (128,),
                (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(HEAD_DIM),
                 np.int32(GROUPS), np.int32(MAX_CTX), np.float32(scale),
                 np.int32(chunk)))

        def combine():
            k.attn_decode_combine(
                (N_HEADS,), (128,),
                (part_acc, part_ml, out, np.int32(splits * 4),
                 np.int32(HEAD_DIM)))

        entry["v1_kernel_ms"] = _time(v1_kernel)
        entry["v2_kernel_ms"] = _time(v2_kernel)
        entry["v3_kernel_ms"] = _time(v3_kernel)
        entry["v4_kernel_ms"] = _time(v4_kernel)
        entry["v6_kernel_ms"] = _time(v6_kernel)
        entry["v7_kernel_ms"] = _time(v7_kernel)
        entry["combine_ms"] = _time(combine)
        for v in ("v1", "v2", "v3", "v4", "v6", "v7"):
            entry[f"{v}_path_ms"] = entry[f"{v}_kernel_ms"] + entry["combine_ms"]
            entry[f"{v}_gbps"] = nbytes / entry[f"{v}_path_ms"] / 1e6
        entry["raw_gbps"] = nbytes / entry["raw_scan_ms"] / 1e6
        results["contexts"][str(t)] = entry

        # correctness per context, 3 seeds: regenerate data per seed.
        for seed in SEEDS:
            _fill_fp8(Kc, seed + 1)
            _fill_fp8(Vc, seed + 2)
            q_new = cp.random.RandomState(seed).standard_normal(
                N_HEADS * HEAD_DIM, dtype=np.float32)
            k.attention_fp8_gqa(out, q_new, Kc, Vc, t, N_HEADS, HEAD_DIM,
                                GROUPS, MAX_CTX, scale, part_acc, part_ml)
            out_v1 = out.copy()
            k.attention_fp8_gqa2(out, q_new, Kc, Vc, t, N_HEADS, HEAD_DIM,
                                 GROUPS, MAX_CTX, scale, part_acc, part_ml)
            out_v2 = out.copy()
            k.attention_fp8_gqa3(out, q_new, Kc, Vc, t, N_HEADS, HEAD_DIM,
                                 GROUPS, MAX_CTX, scale, part_acc, part_ml)
            out_v3 = out.copy()
            k.attention_fp8_gqa4(out, q_new, Kc, Vc, t, N_HEADS, HEAD_DIM,
                                 GROUPS, MAX_CTX, scale, part_acc, part_ml)
            out_v4 = out.copy()
            k.attention_fp8_gqa6(out, q_new, Kc, Vc, t, N_HEADS, HEAD_DIM,
                                 GROUPS, MAX_CTX, scale, part_acc, part_ml)
            out_v6 = out.copy()
            k.attention_fp8_gqa7(out, q_new, Kc, Vc, t, N_HEADS, HEAD_DIM,
                                 GROUPS, MAX_CTX, scale, part_acc, part_ml)
            out_v7 = out.copy()
            n1 = cp.linalg.norm(out_v1) + 1e-30
            results["correctness"].append(
                {"t": t, "seed": seed,
                 "rel_l2_v2": float(cp.linalg.norm(out_v2 - out_v1) / n1),
                 "rel_l2_v3": float(cp.linalg.norm(out_v3 - out_v1) / n1),
                 "rel_l2_v4": float(cp.linalg.norm(out_v4 - out_v1) / n1),
                 "rel_l2_v6": float(cp.linalg.norm(out_v6 - out_v1) / n1),
                 "rel_l2_v7": float(cp.linalg.norm(out_v7 - out_v1) / n1),
                 "v3_bitwise_v1": bool(cp.array_equal(out_v3, out_v1)),
                 "v4_bitwise_v1": bool(cp.array_equal(out_v4, out_v1))})
        _fill_fp8(Kc, SEEDS[0] + 1)
        _fill_fp8(Vc, SEEDS[0] + 2)

    # determinism: two identical v2 runs bitwise equal
    t = 262144
    k.attention_fp8_gqa2(out, q, Kc, Vc, t, N_HEADS, HEAD_DIM, GROUPS,
                         MAX_CTX, scale, part_acc, part_ml)
    a = out.copy()
    k.attention_fp8_gqa2(out, q, Kc, Vc, t, N_HEADS, HEAD_DIM, GROUPS,
                         MAX_CTX, scale, part_acc, part_ml)
    results["determinism"] = bool(cp.array_equal(a, out))

    # v1 byte-linear fit reproduction
    xs = np.array([results["contexts"][str(t)]["bytes"] for t in CONTEXTS],
                  dtype=np.float64)
    ys = np.array([results["contexts"][str(t)]["v1_kernel_ms"] for t in CONTEXTS],
                  dtype=np.float64)
    coef = np.polyfit(xs, ys, 1)
    pred = np.polyval(coef, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    results["v1_fit"] = {"ms_per_gb": float(coef[0] * 1e9 / 1e9),  # slope in ms/byte -> ms/GB below
                         "slope_ms_per_gbyte": float(coef[0] * 1e9),
                         "intercept_ms": float(coef[1]),
                         "r2": 1.0 - ss_res / ss_tot}

    out_path = REPO / "reports" / "treesweep200" / "E4_ATTENTION_ROOFLINE_RESULTS.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps({c: {"v1_ms": round(e["v1_path_ms"], 3),
                          "v4_ms": round(e["v4_path_ms"], 3),
                          "v6_ms": round(e["v6_path_ms"], 3),
                          "v7_ms": round(e["v7_path_ms"], 3),
                          "v7_gbps": round(e["v7_gbps"], 1),
                          "raw_gbps": round(e["raw_gbps"], 1)}
                      for c, e in results["contexts"].items()}, indent=2))
    e262 = results["contexts"]["262144"]
    print("ablations @262144:", {kk: round(e262[kk], 3) for kk in e262
                                 if kk.startswith("abl")})
    print("fit:", results["v1_fit"])
    for v in ("v2", "v3", "v4", "v6", "v7"):
        print(f"max rel_l2 {v}:",
              max(r[f"rel_l2_{v}"] for r in results["correctness"]))
    print("determinism:", results["determinism"])
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
