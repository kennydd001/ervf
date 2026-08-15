"""GPU kernels for the LIGHTNINGSTREAM decode runtime.

BF16 and FP32 weights stay in their stored form on device and are widened in
registers, so the resident shell occupies exactly the bytes N5 measured.  The
NVFP4 expert kernel lives in ``fused_nvfp4`` and is unchanged.

Nothing here makes a performance claim on its own.
"""

from __future__ import annotations

import numpy as np

_SOURCE = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    unsigned int u = ((unsigned int)h) << 16;
    return __uint_as_float(u);
}

// out[row] = sum_k W[row,k] * x[k]   with W stored as bf16
extern "C" __global__ void gemv_bf16(
    const unsigned short* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int rows, const int cols)
{
    extern __shared__ float sx[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();

    const unsigned short* __restrict__ w = W + (size_t)row * cols;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        acc = fmaf(bf16_to_f32(w[k]), sx[k], acc);
    }
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? ws[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[row] = v;
    }
}

extern "C" __global__ void gemv_f32(
    const float* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ out, const int rows, const int cols)
{
    extern __shared__ float sx[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __syncthreads();
    const float* __restrict__ w = W + (size_t)row * cols;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) acc = fmaf(w[k], sx[k], acc);
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? ws[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[row] = v;
    }
}

// RMSNorm over one vector, bf16 weight, float32 accumulation
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

// Grouped gated RMSNorm used by the Mamba mixer: y = rms_group(x * silu(z)) * w
extern "C" __global__ void gated_rmsnorm_grouped(
    const float* __restrict__ x, const float* __restrict__ z,
    const unsigned short* __restrict__ w, float* __restrict__ out,
    const int n, const int group_size, const float eps)
{
    const int g = blockIdx.x;
    const int base = g * group_size;
    extern __shared__ float red[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const float zi = z[base + i];
        const float gated = x[base + i] * (zi / (1.0f + __expf(-zi)));
        red[i] = gated;
        acc = fmaf(gated, gated, acc);
    }
    for (int o = warpSize >> 1; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) s += ws[i];
        scale = rsqrtf(s / (float)group_size + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < group_size; i += blockDim.x)
        out[base + i] = red[i] * scale * bf16_to_f32(w[base + i]);
}

// Depthwise causal conv1d decode step over a rolling state, then SiLU.
// conv_state layout [conv_dim, K]; newest sample written at K-1 after a shift.
extern "C" __global__ void conv1d_decode(
    float* __restrict__ conv_state, const float* __restrict__ xin,
    const unsigned short* __restrict__ wt, const unsigned short* __restrict__ bias,
    float* __restrict__ out, const int conv_dim, const int K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float* st = conv_state + (size_t)c * K;
    #pragma unroll
    for (int k = 0; k < 8; ++k) { if (k + 1 < K) st[k] = st[k + 1]; }
    st[K - 1] = xin[c];
    float acc = bf16_to_f32(bias[c]);
    for (int k = 0; k < K; ++k) acc = fmaf(bf16_to_f32(wt[(size_t)c * K + k]), st[k], acc);
    out[c] = acc / (1.0f + __expf(-acc));   // SiLU
}

// One Mamba-2 SSM decode step.
// state [H, P, N]; x [H, P]; B,C [G, N]; heads_per_group = H / G
extern "C" __global__ void ssm_decode_step(
    float* __restrict__ state, const float* __restrict__ x,
    const float* __restrict__ Bv, const float* __restrict__ Cv,
    const float* __restrict__ dt, const float* __restrict__ Alog,
    const unsigned short* __restrict__ Dv,
    float* __restrict__ y,
    const int H, const int P, const int N, const int heads_per_group)
{
    const int h = blockIdx.x;
    if (h >= H) return;
    const int g = h / heads_per_group;
    const float dth = dt[h];
    const float decay = __expf(-__expf(Alog[h]) * dth);
    const float Dh = bf16_to_f32(Dv[h]);

    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        float* srow = state + ((size_t)h * P + p) * N;
        const float xv = x[h * P + p];
        const float dx = dth * xv;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float s = fmaf(decay, srow[n], dx * Bv[g * N + n]);
            srow[n] = s;
            acc = fmaf(s, Cv[g * N + n], acc);
        }
        y[h * P + p] = acc + Dh * xv;
    }
}

extern "C" __global__ void softplus_clamp(
    const float* __restrict__ src, const unsigned short* __restrict__ bias,
    float* __restrict__ dst, const int n, const float lo, const float hi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = src[i] + bf16_to_f32(bias[i]);
    float sp = (v > 20.0f) ? v : __logf(1.0f + __expf(v));
    dst[i] = fminf(fmaxf(sp, lo), hi);
}

extern "C" __global__ void add_inplace(
    float* __restrict__ dst, const float* __restrict__ src, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// Append one token's K or V into a cache laid out [kv_head][max_ctx][head_dim].
// That layout makes the per-head history contiguous, which cuBLAS requires for
// the attention GEMMs; an interleaved [pos][kv_head][head_dim] cache produces a
// strided slice that cublasGemmEx rejects.
// ---- FP8 E4M3 KV cache -------------------------------------------------
// The checkpoint declares kv_cache_quant_algo: FP8, and N3 measured an FP8 KV
// round trip at rel_l2 2.454e-03. Storing KV as E4M3 cuts attention read
// traffic 4x versus fp32 -- 3.22 GB -> 805 MB per step at 262k context -- and
// frees VRAM that goes straight into expert-cache slots.

__device__ __forceinline__ float e4m3_decode(unsigned char x) {
    const int s = x >> 7, E = (x >> 3) & 0xF, m = x & 7;
    // E==0: subnormal 2^-6 * m/8 == m * 2^-9 ; else (8+m) * 2^(E-10)
    const float v = (E == 0) ? ((float)m * 1.953125e-3f)
                             : ((float)(8 + m) * exp2f((float)(E - 10)));
    return s ? -v : v;
}

__device__ __forceinline__ unsigned char e4m3_encode(float x) {
    const unsigned char s = (x < 0.0f) ? 0x80 : 0x00;
    float ax = fabsf(x);
    if (!(ax > 0.0f)) return s;                 // zero or NaN -> signed zero
    if (ax >= 448.0f) return s | 0x7E;          // saturate at max finite
    if (ax < 1.5625e-2f) {                      // below 2^-6: subnormal ladder
        int m = __float2int_rn(ax * 512.0f);    // ax / 2^-9
        if (m > 7) m = 7;
        return s | (unsigned char)m;
    }
    int e;
    const float mant = frexpf(ax, &e);          // ax = mant * 2^e, mant in [0.5,1)
    int E = e + 6;
    int m = __float2int_rn((mant * 2.0f - 1.0f) * 8.0f);
    if (m > 7) { m = 0; E += 1; }
    if (E > 15) return s | 0x7E;
    if (E < 1) { int sm = __float2int_rn(ax * 512.0f); if (sm > 7) sm = 7;
                 return s | (unsigned char)sm; }
    return s | (unsigned char)((E << 3) | m);
}

// Append one token's K or V into an FP8 cache laid out [kv_head][max_ctx][head_dim].
extern "C" __global__ void kv_append_fp8(
    unsigned char* __restrict__ cache, const float* __restrict__ src,
    const int pos, const int n_kv, const int head_dim, const int max_ctx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv * head_dim;
    if (i >= total) return;
    const int g = i / head_dim, d = i - g * head_dim;
    cache[((size_t)g * max_ctx + pos) * head_dim + d] = e4m3_encode(src[i]);
}


// FP8 per-tensor GEMV: W is F8_E4M3, one byte per weight, scaled by one scalar.
// Used by 3.5 Lightning's Mamba in_proj/out_proj.
extern "C" __global__ void gemv_fp8_tensor(
    const unsigned char* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float wscale,
    const int rows, const int cols)
{
    extern __shared__ float smem[];
    float* sx = smem;                 // cols floats
    float* lut = smem + cols;         // 256 floats

    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();

    const uchar4* __restrict__ w4 =
        reinterpret_cast<const uchar4*>(W + (size_t)row * cols);
    const int nvec = cols >> 2;
    float acc = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
        const uchar4 q = w4[v];
        const int k = v << 2;
        acc = fmaf(lut[q.x], sx[k],     acc);
        acc = fmaf(lut[q.y], sx[k + 1], acc);
        acc = fmaf(lut[q.z], sx[k + 2], acc);
        acc = fmaf(lut[q.w], sx[k + 3], acc);
    }
    for (int b = (nvec << 2) + threadIdx.x; b < cols; b += blockDim.x)
        acc = fmaf(lut[W[(size_t)row * cols + b]], sx[b], acc);

    for (int o = warpSize >> 1; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? ws[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[row] = v * wscale;
    }
}

// Warp-per-position flash decoding over an FP8 KV cache.
// Each lane loads a uchar4 (4 dims) instead of a float4: same coalescing, a
// quarter of the bytes.
extern "C" __global__ void attn_decode_warp_fp8(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // Decode through a shared 1 KB table. Computing E4M3 arithmetically needs an
    // exp2f per element -- eight per position per lane -- which turned the FP8
    // path compute-bound and made it SLOWER than fp32 despite 4x less traffic.
    __shared__ float lut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const int g = h / groups;

    const float4 qv = reinterpret_cast<const float4*>(q + (size_t)h * head_dim)[lane];
    const uchar4* __restrict__ kb =
        reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * head_dim);
    const uchar4* __restrict__ vb =
        reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * head_dim);
    const int vec_per_row = head_dim >> 2;

    float m = -3.0e38f, l = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (int j = j0 + warp; j < j1; j += 4) {
        const uchar4 k4 = kb[(size_t)j * vec_per_row + lane];
        float part = qv.x * lut[k4.x] + qv.y * lut[k4.y]
                   + qv.z * lut[k4.z] + qv.w * lut[k4.w];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            part += __shfl_xor_sync(0xffffffffu, part, o);
        const float sc = part * scale;
        const uchar4 v4 = vb[(size_t)j * vec_per_row + lane];

        // MEASURED: branching to skip the rescale exp when the running max does
        // not move made this SLOWER (262k: 13.225 -> 12.404 tok/s). With
        // --use_fast_math __expf is a ~4-cycle hardware instruction, so two of
        // them cost less than a conditional that breaks the compiler's software
        // pipelining of the loop. The straight-line two-exp form is kept.
        const float m_new = fmaxf(m, sc);
        const float corr = __expf(m - m_new);
        const float p = __expf(sc - m_new);
        l = l * corr + p;

        a0 = a0 * corr + p * lut[v4.x];
        a1 = a1 * corr + p * lut[v4.y];
        a2 = a2 * corr + p * lut[v4.z];
        a3 = a3 * corr + p * lut[v4.w];
        m = m_new;
    }

    const size_t slot = ((size_t)h * (gridDim.y << 2)) + ((size_t)s << 2) + warp;
    float4 out4; out4.x = a0; out4.y = a1; out4.z = a2; out4.w = a3;
    reinterpret_cast<float4*>(part_acc + slot * head_dim)[lane] = out4;
    if (lane == 0) {
        part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m : -3.0e38f;
        part_ml[slot * 2 + 1] = l;
    }
}

// S7: GQA-grouped flash decode. The warp-per-position kernel above launches
// one block per QUERY head, so all 16 q-heads of a KV group re-read the same
// K/V rows -- S7 step 1 measured 10.66x time ratio between heads=32 and
// heads=2 grids, and the heads=2 grid runs at 244.8 GB/s (roofline). This
// kernel launches one block per KV head: a position's K/V bytes are read once
// per warp and serve all 16 query heads of the group. Per-head reduction
// order and the two-exp online softmax are unchanged; partials are written in
// the existing [h][splits*4][head_dim] layout so attn_decode_combine is
// reused unmodified. Hardcoded GQ=16, head_dim=128 (asserted host-side).
extern "C" __global__ void attn_decode_warp_fp8_gqa(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int g = blockIdx.x;               // kv head
    const int s = blockIdx.y;               // split
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;      // 0..3

    __shared__ float lut[256];
    __shared__ float qs[16 * 128];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode((unsigned char)i);
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[i] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const uchar4* __restrict__ kb =
        reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * 128);
    const uchar4* __restrict__ vb =
        reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * 128);

    float m[16], l[16], a[16][4];
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        m[hh] = -3.0e38f; l[hh] = 0.0f;
        a[hh][0] = a[hh][1] = a[hh][2] = a[hh][3] = 0.0f;
    }

    const int d0 = lane << 2;
    for (int j = j0 + warp; j < j1; j += 4) {
        const uchar4 k4 = kb[(size_t)j * 32 + lane];
        const float kx = lut[k4.x], ky = lut[k4.y], kz = lut[k4.z], kw = lut[k4.w];
        float corrs[16], ps[16];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float part = qs[hh * 128 + d0]     * kx + qs[hh * 128 + d0 + 1] * ky
                       + qs[hh * 128 + d0 + 2] * kz + qs[hh * 128 + d0 + 3] * kw;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffffu, part, o);
            const float sc = part * scale;
            const float m_new = fmaxf(m[hh], sc);
            corrs[hh] = __expf(m[hh] - m_new);
            ps[hh] = __expf(sc - m_new);
            l[hh] = l[hh] * corrs[hh] + ps[hh];
            m[hh] = m_new;
        }
        const uchar4 v4 = vb[(size_t)j * 32 + lane];
        const float vx = lut[v4.x], vy = lut[v4.y], vz = lut[v4.z], vw = lut[v4.w];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            a[hh][0] = a[hh][0] * corrs[hh] + ps[hh] * vx;
            a[hh][1] = a[hh][1] * corrs[hh] + ps[hh] * vy;
            a[hh][2] = a[hh][2] * corrs[hh] + ps[hh] * vz;
            a[hh][3] = a[hh][3] * corrs[hh] + ps[hh] * vw;
        }
    }

    const int nsplit4 = gridDim.y << 2;
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        const int h = g * 16 + hh;
        const size_t slot = ((size_t)h * nsplit4) + ((size_t)s << 2) + warp;
        float4 out4; out4.x = a[hh][0]; out4.y = a[hh][1];
        out4.z = a[hh][2]; out4.w = a[hh][3];
        reinterpret_cast<float4*>(part_acc + slot * 128)[lane] = out4;
        if (lane == 0) {
            part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m[hh] : -3.0e38f;
            part_ml[slot * 2 + 1] = l[hh];
        }
    }
}

// S7 v2: same GQA grouping (one block per KV head, K/V bytes move once per
// position), different lane assignment that removes the per-head butterfly.
// v1 measured 3.07 ms/layer @262144 -- compute-bound on 80 warp-shuffles per
// position. Here TWO lanes own one query head (hh = lane>>1, half = lane&1);
// each lane directly loads its own 64-byte half of the K and V rows with
// uint4 loads. All 16 lanes with the same half read the SAME 64 B, which the
// coalescer broadcasts, so HBM traffic stays exactly one row per position.
// The dot needs a single shfl_xor instead of a 5-stage butterfly.
// Deterministic: per-head reduction is lane-local sequential + one exchange.
extern "C" __global__ void attn_decode_warp_fp8_gqa2(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int g = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int hh = lane >> 1;               // query head within the group
    const int hf = lane & 1;                // which 64-dim half

    __shared__ float lut[256];
    __shared__ float qs[16 * 132];          // padded stride kills bank conflicts
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        lut[i] = e4m3_decode((unsigned char)i);
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[(i >> 7) * 132 + (i & 127)] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const unsigned char* __restrict__ kbase = Kc + (size_t)g * max_ctx * 128;
    const unsigned char* __restrict__ vbase = Vc + (size_t)g * max_ctx * 128;
    const float* __restrict__ qh = qs + hh * 132 + hf * 64;

    float m = -3.0e38f, l = 0.0f;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;

    for (int j = j0 + warp; j < j1; j += 4) {
        const uint4* __restrict__ kr =
            reinterpret_cast<const uint4*>(kbase + (size_t)j * 128 + hf * 64);
        const uint4* __restrict__ vr =
            reinterpret_cast<const uint4*>(vbase + (size_t)j * 128 + hf * 64);
        float part = 0.0f;
        uint4 vtmp[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint4 kv = kr[u];
            vtmp[u] = vr[u];
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
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const unsigned char* vb4 = reinterpret_cast<const unsigned char*>(&vtmp[u]);
            #pragma unroll
            for (int b = 0; b < 16; b++)
                acc[u * 16 + b] = fmaf(p, lut[vb4[b]], acc[u * 16 + b] * corr);
        }
    }

    const int h = g * 16 + hh;
    const size_t slot = ((size_t)h * (gridDim.y << 2)) + ((size_t)s << 2) + warp;
    float4* __restrict__ out4 = reinterpret_cast<float4*>(part_acc + slot * 128);
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        // acc[i] holds dim hf*64 + i; float4 slot hf*16+u covers dims
        // hf*64 + u*4 .. +3, i.e. acc[u*4 .. u*4+3] regardless of hf.
        out4[hf * 16 + u] = make_float4(acc[u * 4], acc[u * 4 + 1],
                                        acc[u * 4 + 2], acc[u * 4 + 3]);
    }
    if (hf == 0) {
        part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m : -3.0e38f;
        part_ml[slot * 2 + 1] = l;
    }
}

// E4 v3: v1's proven structure (lane owns 4 dims, 16-head butterfly) with two
// measured bottlenecks removed: (1) the shared-memory LUT decode is replaced
// by the hardware fp8->f16x2 converter (sm_89+; e4m3 values are exact in f16
// and f16->f32 is exact, so results are bitwise identical to the LUT), and
// (2) K/V row loads are double-buffered so the next position's bytes are in
// flight while the current position computes.  Numerics: identical operation
// order to v1, so bitwise-equal output is expected (and gated).
__device__ __forceinline__ void e4m3x4_f32(const uchar4 c, float* f) {
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

extern "C" __global__ void attn_decode_warp_fp8_gqa3(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int g = blockIdx.x;               // kv head
    const int s = blockIdx.y;               // split
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;      // 0..3

    __shared__ float qs[16 * 128];
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[i] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const uchar4* __restrict__ kb =
        reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * 128);
    const uchar4* __restrict__ vb =
        reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * 128);

    float m[16], l[16], a[16][4];
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        m[hh] = -3.0e38f; l[hh] = 0.0f;
        a[hh][0] = a[hh][1] = a[hh][2] = a[hh][3] = 0.0f;
    }

    const int d0 = lane << 2;
    uchar4 k4 = make_uchar4(0, 0, 0, 0), v4 = k4;
    int j = j0 + warp;
    if (j < j1) {
        k4 = kb[(size_t)j * 32 + lane];
        v4 = vb[(size_t)j * 32 + lane];
    }
    for (; j < j1; j += 4) {
        uchar4 k4n = make_uchar4(0, 0, 0, 0), v4n = k4n;
        if (j + 4 < j1) {
            k4n = kb[(size_t)(j + 4) * 32 + lane];
            v4n = vb[(size_t)(j + 4) * 32 + lane];
        }
        float kf[4], vf[4];
        e4m3x4_f32(k4, kf);
        float corrs[16], ps[16];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float part = qs[hh * 128 + d0]     * kf[0] + qs[hh * 128 + d0 + 1] * kf[1]
                       + qs[hh * 128 + d0 + 2] * kf[2] + qs[hh * 128 + d0 + 3] * kf[3];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffffu, part, o);
            const float sc = part * scale;
            const float m_new = fmaxf(m[hh], sc);
            corrs[hh] = __expf(m[hh] - m_new);
            ps[hh] = __expf(sc - m_new);
            l[hh] = l[hh] * corrs[hh] + ps[hh];
            m[hh] = m_new;
        }
        e4m3x4_f32(v4, vf);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            a[hh][0] = a[hh][0] * corrs[hh] + ps[hh] * vf[0];
            a[hh][1] = a[hh][1] * corrs[hh] + ps[hh] * vf[1];
            a[hh][2] = a[hh][2] * corrs[hh] + ps[hh] * vf[2];
            a[hh][3] = a[hh][3] * corrs[hh] + ps[hh] * vf[3];
        }
        k4 = k4n; v4 = v4n;
    }

    const int nsplit4 = gridDim.y << 2;
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        const int h = g * 16 + hh;
        const size_t slot = ((size_t)h * nsplit4) + ((size_t)s << 2) + warp;
        float4 out4; out4.x = a[hh][0]; out4.y = a[hh][1];
        out4.z = a[hh][2]; out4.w = a[hh][3];
        reinterpret_cast<float4*>(part_acc + slot * 128)[lane] = out4;
        if (lane == 0) {
            part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m[hh] : -3.0e38f;
            part_ml[slot * 2 + 1] = l[hh];
        }
    }
}

extern "C" __global__ void attn_decode_warp_fp8_gqa4(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int g = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float qs[16 * 128];
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[i] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const uchar4* __restrict__ kb =
        reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * 128);
    const uchar4* __restrict__ vb =
        reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * 128);

    float m[16], l[16], a[16][4];
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        m[hh] = -3.0e38f; l[hh] = 0.0f;
        a[hh][0] = a[hh][1] = a[hh][2] = a[hh][3] = 0.0f;
    }

    const int d0 = lane << 2;
    int j = j0 + warp;
    // pairs while both positions are in range
    for (; j + 4 < j1; j += 8) {
        const uchar4 k4a = kb[(size_t)j * 32 + lane];
        const uchar4 v4a = vb[(size_t)j * 32 + lane];
        const uchar4 k4b = kb[(size_t)(j + 4) * 32 + lane];
        const uchar4 v4b = vb[(size_t)(j + 4) * 32 + lane];
        float kfa[4], vfa[4], kfb[4], vfb[4];
        e4m3x4_f32(k4a, kfa);
        e4m3x4_f32(k4b, kfb);
        float ca[16], pa[16], cb[16], pb[16];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float partA = qs[hh * 128 + d0]     * kfa[0] + qs[hh * 128 + d0 + 1] * kfa[1]
                        + qs[hh * 128 + d0 + 2] * kfa[2] + qs[hh * 128 + d0 + 3] * kfa[3];
            float partB = qs[hh * 128 + d0]     * kfb[0] + qs[hh * 128 + d0 + 1] * kfb[1]
                        + qs[hh * 128 + d0 + 2] * kfb[2] + qs[hh * 128 + d0 + 3] * kfb[3];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                partA += __shfl_xor_sync(0xffffffffu, partA, o);
                partB += __shfl_xor_sync(0xffffffffu, partB, o);
            }
            const float scA = partA * scale;
            const float mA = fmaxf(m[hh], scA);
            ca[hh] = __expf(m[hh] - mA);
            pa[hh] = __expf(scA - mA);
            l[hh] = l[hh] * ca[hh] + pa[hh];
            m[hh] = mA;
            const float scB = partB * scale;
            const float mB = fmaxf(m[hh], scB);
            cb[hh] = __expf(m[hh] - mB);
            pb[hh] = __expf(scB - mB);
            l[hh] = l[hh] * cb[hh] + pb[hh];
            m[hh] = mB;
        }
        e4m3x4_f32(v4a, vfa);
        e4m3x4_f32(v4b, vfb);
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
    }
    // tail: at most one position left for this warp
    if (j < j1) {
        const uchar4 k4 = kb[(size_t)j * 32 + lane];
        const uchar4 v4 = vb[(size_t)j * 32 + lane];
        float kf[4], vf[4], corrs[16], ps[16];
        e4m3x4_f32(k4, kf);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float part = qs[hh * 128 + d0]     * kf[0] + qs[hh * 128 + d0 + 1] * kf[1]
                       + qs[hh * 128 + d0 + 2] * kf[2] + qs[hh * 128 + d0 + 3] * kf[3];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffffu, part, o);
            const float sc = part * scale;
            const float m_new = fmaxf(m[hh], sc);
            corrs[hh] = __expf(m[hh] - m_new);
            ps[hh] = __expf(sc - m_new);
            l[hh] = l[hh] * corrs[hh] + ps[hh];
            m[hh] = m_new;
        }
        e4m3x4_f32(v4, vf);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            a[hh][0] = a[hh][0] * corrs[hh] + ps[hh] * vf[0];
            a[hh][1] = a[hh][1] * corrs[hh] + ps[hh] * vf[1];
            a[hh][2] = a[hh][2] * corrs[hh] + ps[hh] * vf[2];
            a[hh][3] = a[hh][3] * corrs[hh] + ps[hh] * vf[3];
        }
    }

    const int nsplit4 = gridDim.y << 2;
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        const int h = g * 16 + hh;
        const size_t slot = ((size_t)h * nsplit4) + ((size_t)s << 2) + warp;
        float4 out4; out4.x = a[hh][0]; out4.y = a[hh][1];
        out4.z = a[hh][2]; out4.w = a[hh][3];
        reinterpret_cast<float4*>(part_acc + slot * 128)[lane] = out4;
        if (lane == 0) {
            part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m[hh] : -3.0e38f;
            part_ml[slot * 2 + 1] = l[hh];
        }
    }
}

// E4 v6: packed-fp32 (Blackwell f32x2, measured 1.63x scalar-FP throughput and
// bitwise equal to scalar fmaf) + q hoisted into registers (kills 64 shared
// loads per position-visit).  v3 structure: lane owns 4 dims, 16-head
// butterfly, double-buffered loads.  Dot association differs from v1
// ((q0k0+q2k2)+(q1k1+q3k3)), so bitwise equality is NOT expected; rel_l2 ~1e-7.
__device__ __forceinline__ float2 fma_f32x2(const float2 a, const float2 b,
                                            const float2 c) {
    float2 d;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*reinterpret_cast<unsigned long long*>(&d))
        : "l"(*reinterpret_cast<const unsigned long long*>(&a)),
          "l"(*reinterpret_cast<const unsigned long long*>(&b)),
          "l"(*reinterpret_cast<const unsigned long long*>(&c)));
    return d;
}
__device__ __forceinline__ float2 mul_f32x2(const float2 a, const float2 b) {
    float2 d;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*reinterpret_cast<unsigned long long*>(&d))
        : "l"(*reinterpret_cast<const unsigned long long*>(&a)),
          "l"(*reinterpret_cast<const unsigned long long*>(&b)));
    return d;
}

extern "C" __global__ void attn_decode_warp_fp8_gqa6(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int g = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // q for this lane's 4 dims, all 16 heads, in registers (loop-invariant).
    float2 qr[16][2];
    const float2* __restrict__ q2 =
        reinterpret_cast<const float2*>(q + (size_t)g * 16 * 128);
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        qr[hh][0] = q2[hh * 64 + lane * 2];
        qr[hh][1] = q2[hh * 64 + lane * 2 + 1];
    }

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const uchar4* __restrict__ kb =
        reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * 128);
    const uchar4* __restrict__ vb =
        reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * 128);

    float m[16], l[16];
    float2 a2[16][2];
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        m[hh] = -3.0e38f; l[hh] = 0.0f;
        a2[hh][0] = make_float2(0.0f, 0.0f);
        a2[hh][1] = make_float2(0.0f, 0.0f);
    }

    uchar4 k4 = make_uchar4(0, 0, 0, 0), v4 = k4;
    int j = j0 + warp;
    if (j < j1) {
        k4 = kb[(size_t)j * 32 + lane];
        v4 = vb[(size_t)j * 32 + lane];
    }
    for (; j < j1; j += 4) {
        uchar4 k4n = make_uchar4(0, 0, 0, 0), v4n = k4n;
        if (j + 4 < j1) {
            k4n = kb[(size_t)(j + 4) * 32 + lane];
            v4n = vb[(size_t)(j + 4) * 32 + lane];
        }
        float kf[4];
        e4m3x4_f32(k4, kf);
        const float2 k01 = make_float2(kf[0], kf[1]);
        const float2 k23 = make_float2(kf[2], kf[3]);
        float corrs[16], ps[16];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float2 acc2 = mul_f32x2(k01, qr[hh][0]);
            acc2 = fma_f32x2(k23, qr[hh][1], acc2);
            float part = acc2.x + acc2.y;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffffu, part, o);
            const float sc = part * scale;
            const float m_new = fmaxf(m[hh], sc);
            corrs[hh] = __expf(m[hh] - m_new);
            ps[hh] = __expf(sc - m_new);
            l[hh] = l[hh] * corrs[hh] + ps[hh];
            m[hh] = m_new;
        }
        float vf[4];
        e4m3x4_f32(v4, vf);
        const float2 v01 = make_float2(vf[0], vf[1]);
        const float2 v23 = make_float2(vf[2], vf[3]);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            const float2 c2 = make_float2(corrs[hh], corrs[hh]);
            const float2 p2 = make_float2(ps[hh], ps[hh]);
            a2[hh][0] = fma_f32x2(a2[hh][0], c2, mul_f32x2(p2, v01));
            a2[hh][1] = fma_f32x2(a2[hh][1], c2, mul_f32x2(p2, v23));
        }
        k4 = k4n; v4 = v4n;
    }

    const int nsplit4 = gridDim.y << 2;
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        const int h = g * 16 + hh;
        const size_t slot = ((size_t)h * nsplit4) + ((size_t)s << 2) + warp;
        float4 out4;
        out4.x = a2[hh][0].x; out4.y = a2[hh][0].y;
        out4.z = a2[hh][1].x; out4.w = a2[hh][1].y;
        reinterpret_cast<float4*>(part_acc + slot * 128)[lane] = out4;
        if (lane == 0) {
            part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m[hh] : -3.0e38f;
            part_ml[slot * 2 + 1] = l[hh];
        }
    }
}

extern "C" __global__ void attn_decode_warp_fp8_gqa7(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int g = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float2 qs2[16 * 64];
    for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x)
        qs2[i] = reinterpret_cast<const float2*>(q + (size_t)g * 16 * 128)[i];
    __syncthreads();

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const uchar4* __restrict__ kb =
        reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * 128);
    const uchar4* __restrict__ vb =
        reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * 128);

    float m[16], l[16];
    float2 a2[16][2];
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        m[hh] = -3.0e38f; l[hh] = 0.0f;
        a2[hh][0] = make_float2(0.0f, 0.0f);
        a2[hh][1] = make_float2(0.0f, 0.0f);
    }

    const int q0 = lane * 2;
    int j = j0 + warp;
    for (; j + 4 < j1; j += 8) {
        const uchar4 k4a = kb[(size_t)j * 32 + lane];
        const uchar4 v4a = vb[(size_t)j * 32 + lane];
        const uchar4 k4b = kb[(size_t)(j + 4) * 32 + lane];
        const uchar4 v4b = vb[(size_t)(j + 4) * 32 + lane];
        float kfa[4], vfa[4], kfb[4], vfb[4];
        e4m3x4_f32(k4a, kfa);
        e4m3x4_f32(k4b, kfb);
        const float2 k01a = make_float2(kfa[0], kfa[1]);
        const float2 k23a = make_float2(kfa[2], kfa[3]);
        const float2 k01b = make_float2(kfb[0], kfb[1]);
        const float2 k23b = make_float2(kfb[2], kfb[3]);
        float ca[16], pa[16], cb[16], pb[16];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            const float2 qA0 = qs2[hh * 64 + q0];
            const float2 qA1 = qs2[hh * 64 + q0 + 1];
            float2 accA = mul_f32x2(k01a, qA0);
            accA = fma_f32x2(k23a, qA1, accA);
            float partA = accA.x + accA.y;
            float2 accB = mul_f32x2(k01b, qA0);
            accB = fma_f32x2(k23b, qA1, accB);
            float partB = accB.x + accB.y;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                partA += __shfl_xor_sync(0xffffffffu, partA, o);
                partB += __shfl_xor_sync(0xffffffffu, partB, o);
            }
            const float scA = partA * scale;
            const float mA = fmaxf(m[hh], scA);
            ca[hh] = __expf(m[hh] - mA);
            pa[hh] = __expf(scA - mA);
            l[hh] = l[hh] * ca[hh] + pa[hh];
            m[hh] = mA;
            const float scB = partB * scale;
            const float mB = fmaxf(m[hh], scB);
            cb[hh] = __expf(m[hh] - mB);
            pb[hh] = __expf(scB - mB);
            l[hh] = l[hh] * cb[hh] + pb[hh];
            m[hh] = mB;
        }
        e4m3x4_f32(v4a, vfa);
        e4m3x4_f32(v4b, vfb);
        const float2 v01a = make_float2(vfa[0], vfa[1]);
        const float2 v23a = make_float2(vfa[2], vfa[3]);
        const float2 v01b = make_float2(vfb[0], vfb[1]);
        const float2 v23b = make_float2(vfb[2], vfb[3]);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            const float2 c2a = make_float2(ca[hh], ca[hh]);
            const float2 p2a = make_float2(pa[hh], pa[hh]);
            a2[hh][0] = fma_f32x2(a2[hh][0], c2a, mul_f32x2(p2a, v01a));
            a2[hh][1] = fma_f32x2(a2[hh][1], c2a, mul_f32x2(p2a, v23a));
            const float2 c2b = make_float2(cb[hh], cb[hh]);
            const float2 p2b = make_float2(pb[hh], pb[hh]);
            a2[hh][0] = fma_f32x2(a2[hh][0], c2b, mul_f32x2(p2b, v01b));
            a2[hh][1] = fma_f32x2(a2[hh][1], c2b, mul_f32x2(p2b, v23b));
        }
    }
    if (j < j1) {
        const uchar4 k4 = kb[(size_t)j * 32 + lane];
        const uchar4 v4 = vb[(size_t)j * 32 + lane];
        float kf[4], vf[4], corrs[16], ps[16];
        e4m3x4_f32(k4, kf);
        const float2 k01 = make_float2(kf[0], kf[1]);
        const float2 k23 = make_float2(kf[2], kf[3]);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float2 acc2 = mul_f32x2(k01, qs2[hh * 64 + q0]);
            acc2 = fma_f32x2(k23, qs2[hh * 64 + q0 + 1], acc2);
            float part = acc2.x + acc2.y;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffffu, part, o);
            const float sc = part * scale;
            const float m_new = fmaxf(m[hh], sc);
            corrs[hh] = __expf(m[hh] - m_new);
            ps[hh] = __expf(sc - m_new);
            l[hh] = l[hh] * corrs[hh] + ps[hh];
            m[hh] = m_new;
        }
        e4m3x4_f32(v4, vf);
        const float2 v01 = make_float2(vf[0], vf[1]);
        const float2 v23 = make_float2(vf[2], vf[3]);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            const float2 c2 = make_float2(corrs[hh], corrs[hh]);
            const float2 p2 = make_float2(ps[hh], ps[hh]);
            a2[hh][0] = fma_f32x2(a2[hh][0], c2, mul_f32x2(p2, v01));
            a2[hh][1] = fma_f32x2(a2[hh][1], c2, mul_f32x2(p2, v23));
        }
    }

    const int nsplit4 = gridDim.y << 2;
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        const int h = g * 16 + hh;
        const size_t slot = ((size_t)h * nsplit4) + ((size_t)s << 2) + warp;
        float4 out4;
        out4.x = a2[hh][0].x; out4.y = a2[hh][0].y;
        out4.z = a2[hh][1].x; out4.w = a2[hh][1].y;
        reinterpret_cast<float4*>(part_acc + slot * 128)[lane] = out4;
        if (lane == 0) {
            part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m[hh] : -3.0e38f;
            part_ml[slot * 2 + 1] = l[hh];
        }
    }
}

// Warp-per-position flash decoding.
//
// The earlier split kernel still did a whole-block reduction with two
// __syncthreads() PER POSITION, for only 512 B of useful K data -- it ran ~12.6x
// off the memory roofline. Here each WARP owns a position and each lane owns 4
// dims via a float4 load, so the dot product is a pure warp shuffle: no
// __syncthreads in the inner loop at all, and fully coalesced 512 B/warp loads.
//
// blockDim must be 128 (4 warps); head_dim must be 128.
// Each (block, warp) pair writes one partial, so the caller treats
// splits*4 as the partial count and reuses attn_decode_combine unchanged.
extern "C" __global__ void attn_decode_warp(
    const float* __restrict__ q, const float* __restrict__ Kc,
    const float* __restrict__ Vc,
    float* __restrict__ part_acc,   // [n_heads, splits*4, head_dim]
    float* __restrict__ part_ml,    // [n_heads, splits*4, 2]
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;      // 0..3

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);

    const int g = h / groups;
    const int d0 = lane << 2;               // this lane owns dims [d0, d0+3]

    const float4 qv = reinterpret_cast<const float4*>(q + (size_t)h * head_dim)[lane];
    const float4* __restrict__ kb =
        reinterpret_cast<const float4*>(Kc + (size_t)g * max_ctx * head_dim);
    const float4* __restrict__ vb =
        reinterpret_cast<const float4*>(Vc + (size_t)g * max_ctx * head_dim);
    const int vec_per_row = head_dim >> 2;  // 32

    float m = -3.0e38f, l = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (int j = j0 + warp; j < j1; j += 4) {
        const float4 k4 = kb[(size_t)j * vec_per_row + lane];
        float part = qv.x * k4.x + qv.y * k4.y + qv.z * k4.z + qv.w * k4.w;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            part += __shfl_xor_sync(0xffffffffu, part, o);
        const float sc = part * scale;      // every lane now has the score

        const float m_new = fmaxf(m, sc);
        const float corr = __expf(m - m_new);
        const float p = __expf(sc - m_new);
        l = l * corr + p;

        const float4 v4 = vb[(size_t)j * vec_per_row + lane];
        a0 = a0 * corr + p * v4.x;
        a1 = a1 * corr + p * v4.y;
        a2 = a2 * corr + p * v4.z;
        a3 = a3 * corr + p * v4.w;
        m = m_new;
    }

    const size_t slot = ((size_t)h * (gridDim.y << 2)) + ((size_t)s << 2) + warp;
    float4 out4;
    out4.x = a0; out4.y = a1; out4.z = a2; out4.w = a3;
    reinterpret_cast<float4*>(part_acc + slot * head_dim)[lane] = out4;
    if (lane == 0) {
        part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m : -3.0e38f;
        part_ml[slot * 2 + 1] = l;
    }
}

// Flash-decoding: split the position range across many blocks.
// grid = (n_heads, splits); each block runs online softmax over its chunk and
// writes a partial (m, l, acc[head_dim]) triple. A combine kernel merges them.
// This removes the O(t) serial walk that made 262k context cost ~248 ms/layer.
extern "C" __global__ void attn_decode_split(
    const float* __restrict__ q, const float* __restrict__ Kc,
    const float* __restrict__ Vc,
    float* __restrict__ part_acc,   // [n_heads, splits, head_dim]
    float* __restrict__ part_ml,    // [n_heads, splits, 2]
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int chunk)
{
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);

    const int g = h / groups;
    const float qv = q[(size_t)h * head_dim + d];
    const float* __restrict__ kbase = Kc + (size_t)g * max_ctx * head_dim;
    const float* __restrict__ vbase = Vc + (size_t)g * max_ctx * head_dim;

    __shared__ float red[32];
    __shared__ float s_score;
    float m = -3.0e38f, l = 0.0f, acc = 0.0f;

    for (int j = j0; j < j1; ++j) {
        float part = qv * kbase[(size_t)j * head_dim + d];
        for (int o = warpSize >> 1; o > 0; o >>= 1)
            part += __shfl_down_sync(0xffffffffu, part, o);
        const int lane = d & 31, warp = d >> 5;
        if (lane == 0) red[warp] = part;
        __syncthreads();
        if (d == 0) {
            float sum = 0.0f;
            const int nw = (head_dim + 31) >> 5;
            for (int w = 0; w < nw; ++w) sum += red[w];
            s_score = sum * scale;
        }
        __syncthreads();
        const float sc = s_score;
        const float m_new = fmaxf(m, sc);
        const float corr = __expf(m - m_new);
        const float p = __expf(sc - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vbase[(size_t)j * head_dim + d];
        m = m_new;
        __syncthreads();
    }

    const size_t base = ((size_t)h * gridDim.y + s);
    part_acc[base * head_dim + d] = acc;
    if (d == 0) { part_ml[base * 2 + 0] = (j1 > j0) ? m : -3.0e38f;
                  part_ml[base * 2 + 1] = l; }
}

// Merge the per-split partials for one head.
extern "C" __global__ void attn_decode_combine(
    const float* __restrict__ part_acc, const float* __restrict__ part_ml,
    float* __restrict__ out, const int splits, const int head_dim)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    float m = -3.0e38f;
    for (int s = 0; s < splits; ++s) {
        const float ms = part_ml[((size_t)h * splits + s) * 2 + 0];
        m = fmaxf(m, ms);
    }
    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const size_t base = (size_t)h * splits + s;
        const float ms = part_ml[base * 2 + 0];
        const float ls = part_ml[base * 2 + 1];
        if (ls <= 0.0f) continue;
        const float w = __expf(ms - m);
        l += ls * w;
        acc += part_acc[base * head_dim + d] * w;
    }
    out[(size_t)h * head_dim + d] = (l > 0.0f) ? acc / l : 0.0f;
}

// Decode attention with online (streaming) softmax.
// One block per query head; blockDim must equal head_dim.
// Avoids cuBLAS entirely -- the decode shape is a GEMV, and cublasGemmEx
// rejects the degenerate k=1 case at the first position.
// Cache layout: [kv_head][max_ctx][head_dim].
extern "C" __global__ void attn_decode(
    const float* __restrict__ q,      // [n_heads, head_dim]
    const float* __restrict__ Kc,     // [n_kv, max_ctx, head_dim]
    const float* __restrict__ Vc,
    float* __restrict__ out,          // [n_heads, head_dim]
    const int t, const int head_dim, const int groups,
    const int max_ctx, const float scale)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const int g = h / groups;

    const float qv = q[(size_t)h * head_dim + d];
    const float* __restrict__ kbase = Kc + (size_t)g * max_ctx * head_dim;
    const float* __restrict__ vbase = Vc + (size_t)g * max_ctx * head_dim;

    __shared__ float red[32];
    __shared__ float s_score;
    float m = -3.0e38f, l = 0.0f, acc = 0.0f;

    for (int j = 0; j < t; ++j) {
        float part = qv * kbase[(size_t)j * head_dim + d];
        for (int o = warpSize >> 1; o > 0; o >>= 1)
            part += __shfl_down_sync(0xffffffffu, part, o);
        const int lane = d & 31, warp = d >> 5;
        if (lane == 0) red[warp] = part;
        __syncthreads();
        if (d == 0) {
            float sum = 0.0f;
            const int nw = (head_dim + 31) >> 5;
            for (int w = 0; w < nw; ++w) sum += red[w];
            s_score = sum * scale;
        }
        __syncthreads();

        const float s = s_score;
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vbase[(size_t)j * head_dim + d];
        m = m_new;
        __syncthreads();
    }
    out[(size_t)h * head_dim + d] = acc / l;
}

extern "C" __global__ void kv_append(
    float* __restrict__ cache, const float* __restrict__ src,
    const int pos, const int n_kv, const int head_dim, const int max_ctx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv * head_dim;
    if (i >= total) return;
    const int g = i / head_dim;
    const int d = i - g * head_dim;
    cache[((size_t)g * max_ctx + pos) * head_dim + d] = src[i];
}

// ---------------------------------------------------------------------------
// E1 fase 2.2: graph-capture-compatible variants. Every scalar that used to
// arrive as a by-value launch argument is read from a device buffer instead,
// so one captured graph replays the whole token without host input. Numerics
// are untouched: the arithmetic bodies are verbatim copies of the kernels they
// mirror, only the prologue (WHERE t/pos/the token id comes from) differs.
// ---------------------------------------------------------------------------

// Embedding gather straight from the mapped, pinned host table: bf16 row ->
// f32 via a 16-bit left shift, exactly what step() did with cupy temporaries.
extern "C" __global__ void embed_gather_bf16(
    const unsigned short* __restrict__ table, const int* __restrict__ tok,
    float* __restrict__ h, const int hidden)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    h[i] = __uint_as_float(
        ((unsigned int)table[(size_t)tok[0] * hidden + i]) << 16);
}

// kv_append_fp8 with the position read on device.
extern "C" __global__ void kv_append_fp8_dp(
    unsigned char* __restrict__ cache, const float* __restrict__ src,
    const int* __restrict__ pos_dp, const int n_kv, const int head_dim,
    const int max_ctx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv * head_dim;
    if (i >= total) return;
    const int pos = pos_dp[0];
    const int g = i / head_dim, d = i - g * head_dim;
    cache[((size_t)g * max_ctx + pos) * head_dim + d] = e4m3_encode(src[i]);
}

// attn_decode_warp_fp8_gqa4 with t and chunk computed on device from pos_dp,
// under a FIXED grid of (n_kv, max_splits): blocks whose split lies beyond the
// live range write a neutral partial (m=-inf, l=0) so no slot ever holds stale
// data, and attn_decode_combine -- which already skips l<=0 -- merges exactly
// the same non-neutral partials in the same order as the eager variant.
extern "C" __global__ void attn_decode_warp_fp8_gqa4_dp(
    const float* __restrict__ q,
    const unsigned char* __restrict__ Kc, const unsigned char* __restrict__ Vc,
    float* __restrict__ part_acc, float* __restrict__ part_ml,
    const int* __restrict__ pos_dp, const int head_dim, const int groups,
    const int max_ctx, const float scale, const int max_splits,
    const int split_threshold)
{
    const int t = pos_dp[0] + 1;
    int splits = (t + split_threshold - 1) / split_threshold;
    splits = min(max_splits, max(1, splits));
    const int chunk = (t + splits - 1) / splits;

    const int g = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float qs[16 * 128];
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        qs[i] = q[((size_t)g * 16) * 128 + i];
    __syncthreads();

    const int j0 = s * chunk;
    const int j1 = min(t, j0 + chunk);
    const uchar4* __restrict__ kb =
        reinterpret_cast<const uchar4*>(Kc + (size_t)g * max_ctx * 128);
    const uchar4* __restrict__ vb =
        reinterpret_cast<const uchar4*>(Vc + (size_t)g * max_ctx * 128);

    float m[16], l[16], a[16][4];
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        m[hh] = -3.0e38f; l[hh] = 0.0f;
        a[hh][0] = a[hh][1] = a[hh][2] = a[hh][3] = 0.0f;
    }

    const int d0 = lane << 2;
    int j = j0 + warp;
    // pairs while both positions are in range
    for (; j + 4 < j1; j += 8) {
        const uchar4 k4a = kb[(size_t)j * 32 + lane];
        const uchar4 v4a = vb[(size_t)j * 32 + lane];
        const uchar4 k4b = kb[(size_t)(j + 4) * 32 + lane];
        const uchar4 v4b = vb[(size_t)(j + 4) * 32 + lane];
        float kfa[4], vfa[4], kfb[4], vfb[4];
        e4m3x4_f32(k4a, kfa);
        e4m3x4_f32(k4b, kfb);
        float ca[16], pa[16], cb[16], pb[16];
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float partA = qs[hh * 128 + d0]     * kfa[0] + qs[hh * 128 + d0 + 1] * kfa[1]
                        + qs[hh * 128 + d0 + 2] * kfa[2] + qs[hh * 128 + d0 + 3] * kfa[3];
            float partB = qs[hh * 128 + d0]     * kfb[0] + qs[hh * 128 + d0 + 1] * kfb[1]
                        + qs[hh * 128 + d0 + 2] * kfb[2] + qs[hh * 128 + d0 + 3] * kfb[3];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                partA += __shfl_xor_sync(0xffffffffu, partA, o);
                partB += __shfl_xor_sync(0xffffffffu, partB, o);
            }
            const float scA = partA * scale;
            const float mA = fmaxf(m[hh], scA);
            ca[hh] = __expf(m[hh] - mA);
            pa[hh] = __expf(scA - mA);
            l[hh] = l[hh] * ca[hh] + pa[hh];
            m[hh] = mA;
            const float scB = partB * scale;
            const float mB = fmaxf(m[hh], scB);
            cb[hh] = __expf(m[hh] - mB);
            pb[hh] = __expf(scB - mB);
            l[hh] = l[hh] * cb[hh] + pb[hh];
            m[hh] = mB;
        }
        e4m3x4_f32(v4a, vfa);
        e4m3x4_f32(v4b, vfb);
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
    }
    // tail: at most one position left for this warp
    if (j < j1) {
        const uchar4 k4 = kb[(size_t)j * 32 + lane];
        const uchar4 v4 = vb[(size_t)j * 32 + lane];
        float kf[4], vf[4], corrs[16], ps[16];
        e4m3x4_f32(k4, kf);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            float part = qs[hh * 128 + d0]     * kf[0] + qs[hh * 128 + d0 + 1] * kf[1]
                       + qs[hh * 128 + d0 + 2] * kf[2] + qs[hh * 128 + d0 + 3] * kf[3];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffffu, part, o);
            const float sc = part * scale;
            const float m_new = fmaxf(m[hh], sc);
            corrs[hh] = __expf(m[hh] - m_new);
            ps[hh] = __expf(sc - m_new);
            l[hh] = l[hh] * corrs[hh] + ps[hh];
            m[hh] = m_new;
        }
        e4m3x4_f32(v4, vf);
        #pragma unroll
        for (int hh = 0; hh < 16; hh++) {
            a[hh][0] = a[hh][0] * corrs[hh] + ps[hh] * vf[0];
            a[hh][1] = a[hh][1] * corrs[hh] + ps[hh] * vf[1];
            a[hh][2] = a[hh][2] * corrs[hh] + ps[hh] * vf[2];
            a[hh][3] = a[hh][3] * corrs[hh] + ps[hh] * vf[3];
        }
    }

    const int nsplit4 = gridDim.y << 2;
    #pragma unroll
    for (int hh = 0; hh < 16; hh++) {
        const int h = g * 16 + hh;
        const size_t slot = ((size_t)h * nsplit4) + ((size_t)s << 2) + warp;
        float4 out4; out4.x = a[hh][0]; out4.y = a[hh][1];
        out4.z = a[hh][2]; out4.w = a[hh][3];
        reinterpret_cast<float4*>(part_acc + slot * 128)[lane] = out4;
        if (lane == 0) {
            part_ml[slot * 2 + 0] = (j1 > j0 + warp) ? m[hh] : -3.0e38f;
            part_ml[slot * 2 + 1] = l[hh];
        }
    }
}

// Two-pass argmax over the logits; low index wins ties, matching cp.argmax.
// (NaN would be skipped here while cupy propagates it; NaN logits are a model
// defect, not a tie, so the verifier tests ties, not NaNs.)
extern "C" __global__ void argmax_part(
    const float* __restrict__ x, const int n,
    float* __restrict__ pmax, int* __restrict__ pidx)
{
    __shared__ float sm[256];
    __shared__ int si[256];
    const int tid = threadIdx.x;
    const int chunk = (n + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * chunk;
    const int hi = min(n, lo + chunk);
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = lo + tid; i < hi; i += blockDim.x) {
        const float v = x[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    sm[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const float ov = sm[tid + off];
            const int oi = si[tid + off];
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            sm[tid] = bv; si[tid] = bi;
        }
        __syncthreads();
    }
    if (tid == 0) { pmax[blockIdx.x] = bv; pidx[blockIdx.x] = bi; }
}

extern "C" __global__ void argmax_final(
    const float* __restrict__ pmax, const int* __restrict__ pidx,
    const int nparts, int* __restrict__ tok_out)
{
    __shared__ float sm[256];
    __shared__ int si[256];
    const int tid = threadIdx.x;
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = tid; i < nparts; i += blockDim.x) {
        const float v = pmax[i];
        if (v > bv || (v == bv && pidx[i] < bi)) { bv = v; bi = pidx[i]; }
    }
    sm[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const float ov = sm[tid + off];
            const int oi = si[tid + off];
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            sm[tid] = bv; si[tid] = bi;
        }
        __syncthreads();
    }
    if (tid == 0) tok_out[0] = bi;
}

extern "C" __global__ void pos_inc(int* __restrict__ pos_dp)
{
    pos_dp[0] += 1;
}
"""


class GPUKernels:
    def __init__(self, block: int = 256):
        import cupy as cp

        self.cp = cp
        self.block = block
        self.mod = cp.RawModule(code=_SOURCE, options=("-std=c++14", "--use_fast_math"))
        for name in ("gemv_bf16", "gemv_f32", "rmsnorm_bf16w", "gated_rmsnorm_grouped",
                     "conv1d_decode", "ssm_decode_step", "softplus_clamp",
                     "add_inplace", "kv_append", "attn_decode",
                     "attn_decode_split", "attn_decode_combine",
                     "attn_decode_warp", "attn_decode_warp_fp8", "kv_append_fp8",
                     "attn_decode_warp_fp8_gqa", "attn_decode_warp_fp8_gqa2",
                     "attn_decode_warp_fp8_gqa3", "attn_decode_warp_fp8_gqa4",
                     "attn_decode_warp_fp8_gqa6", "attn_decode_warp_fp8_gqa7",
                     "gemv_fp8_tensor",
                     # E1 fase 2.2: graph-capture-compatible variants
                     "embed_gather_bf16", "kv_append_fp8_dp",
                     "attn_decode_warp_fp8_gqa4_dp",
                     "argmax_part", "argmax_final", "pos_inc"):
            setattr(self, name, self.mod.get_function(name))

    # -- thin wrappers ----------------------------------------------------
    def mv_fp8_tensor(self, out, W, x, wscale, rows, cols):
        """FP8-per-tensor GEMV; shared holds the activation plus a 256-entry LUT."""
        self.gemv_fp8_tensor((rows,), (self.block,),
                             (W, x, out, np.float32(wscale),
                              np.int32(rows), np.int32(cols)),
                             shared_mem=(cols + 256) * 4)

    def mv_bf16(self, out, W, x, rows, cols):
        self.gemv_bf16((rows,), (self.block,), (W, x, out, np.int32(rows), np.int32(cols)),
                       shared_mem=cols * 4)

    def mv_f32(self, out, W, x, rows, cols):
        self.gemv_f32((rows,), (self.block,), (W, x, out, np.int32(rows), np.int32(cols)),
                      shared_mem=cols * 4)

    def norm(self, out, x, w, n, eps):
        self.rmsnorm_bf16w((1,), (self.block,), (x, w, out, np.int32(n), np.float32(eps)),
                           shared_mem=32 * 4)

    def gated_norm(self, out, x, z, w, n, group_size, eps):
        self.gated_rmsnorm_grouped((n // group_size,), (self.block,),
                                   (x, z, w, out, np.int32(n), np.int32(group_size),
                                    np.float32(eps)), shared_mem=group_size * 4)

    def conv_step(self, out, state, xin, w, b, conv_dim, k):
        blocks = (conv_dim + self.block - 1) // self.block
        self.conv1d_decode((blocks,), (self.block,),
                           (state, xin, w, b, out, np.int32(conv_dim), np.int32(k)))

    def ssm_step(self, y, state, x, Bv, Cv, dt, Alog, Dv, H, P, N, hpg):
        self.ssm_decode_step((H,), (min(self.block, P),),
                             (state, x, Bv, Cv, dt, Alog, Dv, y,
                              np.int32(H), np.int32(P), np.int32(N), np.int32(hpg)))

    def dt_activate(self, dst, src, bias, n, lo, hi):
        blocks = (n + self.block - 1) // self.block
        self.softplus_clamp((blocks,), (self.block,),
                            (src, bias, dst, np.int32(n), np.float32(lo), np.float32(hi)))

    def add_(self, dst, src, n):
        blocks = (n + self.block - 1) // self.block
        self.add_inplace((blocks,), (self.block,), (dst, src, np.int32(n)))

    # Below this depth the single-block walk is cheaper than the split launch
    # plus combine; above it the serial walk dominates.
    SPLIT_THRESHOLD = 512
    MAX_SPLITS = 256

    def attention_fp8(self, out, q, Kc, Vc, t, n_heads, head_dim, groups, max_ctx,
                      scale, part_acc, part_ml):
        splits = max(1, min(self.MAX_SPLITS,
                            (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD))
        chunk = (t + splits - 1) // splits
        self.attn_decode_warp_fp8(
            (n_heads, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def attention_fp8_gqa(self, out, q, Kc, Vc, t, n_heads, head_dim, groups,
                          max_ctx, scale, part_acc, part_ml):
        """S7: one block per KV head instead of per query head; K/V bytes move
        once per position per group instead of once per query head."""
        assert head_dim == 128 and groups == 16 and n_heads == 32
        splits = max(1, min(self.MAX_SPLITS,
                            (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD))
        chunk = (t + splits - 1) // splits
        n_kv = n_heads // groups
        self.attn_decode_warp_fp8_gqa(
            (n_kv, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def attention_fp8_gqa2(self, out, q, Kc, Vc, t, n_heads, head_dim, groups,
                           max_ctx, scale, part_acc, part_ml):
        """E4: same launch geometry as attention_fp8_gqa but the gqa2 lane
        assignment (two lanes per query head, lane-local dot, one shuffle)."""
        assert head_dim == 128 and groups == 16 and n_heads == 32
        splits = max(1, min(self.MAX_SPLITS,
                            (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD))
        chunk = (t + splits - 1) // splits
        n_kv = n_heads // groups
        self.attn_decode_warp_fp8_gqa2(
            (n_kv, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def attention_fp8_gqa3(self, out, q, Kc, Vc, t, n_heads, head_dim, groups,
                           max_ctx, scale, part_acc, part_ml):
        """E4 v3: hardware fp8 decode + double-buffered loads, v1 numerics."""
        assert head_dim == 128 and groups == 16 and n_heads == 32
        splits = max(1, min(self.MAX_SPLITS,
                            (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD))
        chunk = (t + splits - 1) // splits
        n_kv = n_heads // groups
        self.attn_decode_warp_fp8_gqa3(
            (n_kv, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def attention_fp8_gqa4(self, out, q, Kc, Vc, t, n_heads, head_dim, groups,
                           max_ctx, scale, part_acc, part_ml):
        """E4 v4: v3 + two positions per warp iteration (ILP), v1 numerics."""
        assert head_dim == 128 and groups == 16 and n_heads == 32
        splits = max(1, min(self.MAX_SPLITS,
                            (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD))
        chunk = (t + splits - 1) // splits
        n_kv = n_heads // groups
        self.attn_decode_warp_fp8_gqa4(
            (n_kv, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def attention_fp8_gqa6(self, out, q, Kc, Vc, t, n_heads, head_dim, groups,
                           max_ctx, scale, part_acc, part_ml):
        """E4 v6: packed-fp32 dots/PV + register-resident q."""
        assert head_dim == 128 and groups == 16 and n_heads == 32
        splits = max(1, min(self.MAX_SPLITS,
                            (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD))
        chunk = (t + splits - 1) // splits
        n_kv = n_heads // groups
        self.attn_decode_warp_fp8_gqa6(
            (n_kv, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def attention_fp8_gqa7(self, out, q, Kc, Vc, t, n_heads, head_dim, groups,
                           max_ctx, scale, part_acc, part_ml):
        """E4 v7: v4 2-position ILP + packed-fp32 dots/PV + shared float2 q."""
        assert head_dim == 128 and groups == 16 and n_heads == 32
        splits = max(1, min(self.MAX_SPLITS,
                            (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD))
        chunk = (t + splits - 1) // splits
        n_kv = n_heads // groups
        self.attn_decode_warp_fp8_gqa7(
            (n_kv, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def kv_write_fp8(self, cache, src, pos, n_kv, head_dim, max_ctx):
        total = n_kv * head_dim
        blocks = (total + self.block - 1) // self.block
        self.kv_append_fp8((blocks,), (self.block,),
                           (cache, src, np.int32(pos), np.int32(n_kv),
                            np.int32(head_dim), np.int32(max_ctx)))

    # -- E1 fase 2.2 wrappers (graph-safe: scalars live on device) -----------
    def embed_gather(self, h, table_ptr: int, tok_dev, hidden: int):
        blocks = (hidden + self.block - 1) // self.block
        self.embed_gather_bf16((blocks,), (self.block,),
                               (np.uint64(table_ptr), tok_dev, h,
                                np.int32(hidden)))

    def kv_write_fp8_dp(self, cache, src, pos_dp, n_kv, head_dim, max_ctx):
        total = n_kv * head_dim
        blocks = (total + self.block - 1) // self.block
        self.kv_append_fp8_dp((blocks,), (self.block,),
                              (cache, src, pos_dp, np.int32(n_kv),
                               np.int32(head_dim), np.int32(max_ctx)))

    def attention_fp8_gqa4_dp(self, out, q, Kc, Vc, pos_dp, n_heads, head_dim,
                              groups, max_ctx, scale, part_acc, part_ml):
        """E1 fase 2.2: v4 numerics under a fixed grid; t/chunk on device."""
        assert head_dim == 128 and groups == 16 and n_heads == 32
        n_kv = n_heads // groups
        self.attn_decode_warp_fp8_gqa4_dp(
            (n_kv, self.MAX_SPLITS), (128,),
            (q, Kc, Vc, part_acc, part_ml, pos_dp, np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale),
             np.int32(self.MAX_SPLITS), np.int32(self.SPLIT_THRESHOLD)))
        self.attn_decode_combine(
            (n_heads,), (128,),
            (part_acc, part_ml, out, np.int32(self.MAX_SPLITS * 4),
             np.int32(head_dim)))

    def argmax_logits(self, tok_dev, logits, vocab: int, pmax, pidx,
                      nparts: int = 256):
        self.argmax_part((nparts,), (256,),
                         (logits, np.int32(vocab), pmax, pidx))
        self.argmax_final((1,), (256,),
                          (pmax, pidx, np.int32(nparts), tok_dev))

    def pos_increment(self, pos_dp):
        self.pos_inc((1,), (1,), (pos_dp,))

    def attention(self, out, q, Kc, Vc, t, n_heads, head_dim, groups, max_ctx, scale,
                  part_acc=None, part_ml=None):
        if t <= self.SPLIT_THRESHOLD or part_acc is None:
            self.attn_decode((n_heads,), (head_dim,),
                             (q, Kc, Vc, out, np.int32(t), np.int32(head_dim),
                              np.int32(groups), np.int32(max_ctx), np.float32(scale)))
            return
        splits = min(self.MAX_SPLITS, (t + self.SPLIT_THRESHOLD - 1) // self.SPLIT_THRESHOLD)
        chunk = (t + splits - 1) // splits
        # Warp-per-position: 4 warps per block, so splits*4 partials.
        self.attn_decode_warp(
            (n_heads, splits), (128,),
            (q, Kc, Vc, part_acc, part_ml, np.int32(t), np.int32(head_dim),
             np.int32(groups), np.int32(max_ctx), np.float32(scale), np.int32(chunk)))
        self.attn_decode_combine(
            (n_heads,), (head_dim,),
            (part_acc, part_ml, out, np.int32(splits * 4), np.int32(head_dim)))

    def kv_write(self, cache, src, pos, n_kv, head_dim, max_ctx):
        total = n_kv * head_dim
        blocks = (total + self.block - 1) // self.block
        self.kv_append((blocks,), (self.block,),
                       (cache, src, np.int32(pos), np.int32(n_kv),
                        np.int32(head_dim), np.int32(max_ctx)))




