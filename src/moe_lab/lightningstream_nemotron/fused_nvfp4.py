"""Fused NVFP4 decode+GEMV kernels for the Nemotron routed-expert path.

N4 measured the unfused path: transport 29.756 ms against a decode of 353.133 ms
that was 93.9% of the composed token.  The cause is not arithmetic but memory
traffic.  Decoding a matrix to float32 first materialises 9,977,856 weights --
about 40 MB written and 40 MB read back per matrix -- to feed a GEMV that
consumes each weight exactly once.

These kernels never materialise the dequantised matrix.  Each block owns one
output row, streams that row's packed bytes and block scales straight from
global memory, decodes in registers and accumulates.  Per-matrix device traffic
falls from roughly 120 MB to the record itself: 2,494,464 B of codes plus
311,808 B of scales.

Bit-exactness note: the reduction order here is a block-parallel tree, not the
sequential order of the numpy reference.  Cross-device reduction order is
explicitly not required to match (assignment H2), so these kernels are gated on
``rel_l2`` against the reference, while the *decode* itself remains exactly the
same integer unpack and table lookup that N4 proved bit-identical.
"""

from __future__ import annotations

import numpy as np

from . import nvfp4

GROUP_SIZE = nvfp4.GROUP_SIZE

_KERNEL_SOURCE = r"""
extern "C" __global__ void gemv_nvfp4_rows(
    const unsigned char* __restrict__ codes,    // [rows, cols/2]
    const unsigned char* __restrict__ scales,   // [rows, cols/16]
    const float*         __restrict__ e2m1_lut, // [16]
    const float*         __restrict__ e4m3_lut, // [256]
    const float*         __restrict__ x,        // [cols]
    float*               __restrict__ out,      // [rows]
    const float global_scale,
    const int rows,
    const int cols,
    const int apply_relu2,
    const float out_scale)
{
    extern __shared__ float sx[];

    const int row = blockIdx.x;
    if (row >= rows) return;

    // Stage the activation vector once per block.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sx[i] = x[i];
    }

    // 16 E2M1 magnitudes live in shared memory too; a dynamically indexed
    // local array would otherwise spill to local memory.
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int n_bytes = cols >> 1;              // two 4-bit codes per byte
    const int n_scales = cols >> 4;             // group size 16
    const unsigned char* __restrict__ crow = codes  + (size_t)row * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)row * n_scales;

    // Vectorised code loads: one uchar4 fetch covers 8 codes and needs a single
    // block-scale lookup, because 16 codes = 8 bytes so bytes [4v, 4v+3] all sit
    // in group (4v)>>3. Byte-at-a-time loads left this GEMV ~3.5x off roofline.
    float acc = 0.0f;
    const int n_vec = n_bytes >> 2;
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);

    for (int v = threadIdx.x; v < n_vec; v += blockDim.x) {
        const uchar4 q = crow4[v];
        const int b = v << 2;
        const float s = e4m3_lut[srow[b >> 3]] * global_scale;
        const int k = b << 1;

        acc = fmaf(s_e2m1[q.x & 0x0F] * s, sx[k],     acc);
        acc = fmaf(s_e2m1[q.x >> 4]   * s, sx[k + 1], acc);
        acc = fmaf(s_e2m1[q.y & 0x0F] * s, sx[k + 2], acc);
        acc = fmaf(s_e2m1[q.y >> 4]   * s, sx[k + 3], acc);
        acc = fmaf(s_e2m1[q.z & 0x0F] * s, sx[k + 4], acc);
        acc = fmaf(s_e2m1[q.z >> 4]   * s, sx[k + 5], acc);
        acc = fmaf(s_e2m1[q.w & 0x0F] * s, sx[k + 6], acc);
        acc = fmaf(s_e2m1[q.w >> 4]   * s, sx[k + 7], acc);
    }
    // Tail for a row whose byte count is not a multiple of 4.
    for (int b = (n_vec << 2) + threadIdx.x; b < n_bytes; b += blockDim.x) {
        const unsigned char byte = crow[b];
        const float s = e4m3_lut[srow[b >> 3]] * global_scale;
        const int k = b << 1;
        acc = fmaf(s_e2m1[byte & 0x0F] * s, sx[k],     acc);
        acc = fmaf(s_e2m1[byte >> 4]   * s, sx[k + 1], acc);
    }

    // Warp reduction, then across warps.
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
            if (apply_relu2) {
                const float r = fmaxf(v, 0.0f);
                out[row] = r * r;
            } else {
                out[row] = v * out_scale;
            }
        }
    }
}


extern "C" __global__ void weighted_accumulate(
    float*       __restrict__ dst,
    const float* __restrict__ src,
    const float  weight,
    const int    n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = fmaf(src[i], weight, dst[i]);
}

// ---------------------------------------------------------------------------
// S5: column-selective down_proj.
//
// ReLU^2 makes ~91% of the intermediates exactly zero (S2 census, 20/20
// verified); the zero columns of down_proj then contribute nothing and their
// bytes should not move at all. Because the zeros do not cluster (30.6%
// all-zero 16-column blocks), selection must be column-accurate.
//
// down_proj is therefore stored panel-major on the host: 116 panels of 16
// columns; per panel first the 2688 scale bytes (one per output row, shared
// by the panel's 16 columns), then the 16 columns, each 2688 nibbles =
// 1344 B contiguous. The masked GEMV reads directly from mapped pinned host
// memory and touches only the cachelines of nonzero columns plus the scale
// blocks of active panels.
//
// panel_scan is deterministic: atomicOr commutes, and the panel list is
// built by a single-thread ascending scan. gemv_down_masked_partial assigns
// panels to chunks by striding (pi = chunk; pi += nchunks) and the reduce
// adds chunk partials in fixed order, so the whole composition is
// deterministic and every skipped term is an exact +0.0 addend.
// ---------------------------------------------------------------------------

extern "C" __global__ void panel_scan(
    const float* __restrict__ act,
    const int inter,
    unsigned int* __restrict__ panel_masks,   // [inter/16], zeroed by us
    int* __restrict__ panel_list,             // [inter/16]
    int* __restrict__ panel_count,
    int* __restrict__ nz_list,                // [inter] ascending column ids
    int* __restrict__ nz_count)
{
    const int npanel = inter >> 4;
    if (threadIdx.x < npanel) panel_masks[threadIdx.x] = 0u;
    if (threadIdx.x == 0) { *panel_count = 0; *nz_count = 0; }
    __syncthreads();
    for (int j = threadIdx.x; j < inter; j += blockDim.x) {
        if (act[j] != 0.0f) atomicOr(&panel_masks[j >> 4], 1u << (j & 15));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0, m = 0;
        for (int p = 0; p < npanel; p++) {
            const unsigned int mk = panel_masks[p];
            if (mk) {
                panel_list[n++] = p;
                for (int c = 0; c < 16; c++)
                    if (mk & (1u << c)) nz_list[m++] = (p << 4) + c;
            }
        }
        *panel_count = n;
        *nz_count = m;
    }
}

// S5-R1 (A2): SM-side gather. The microbench measured byte-per-thread mapped
// reads at 1.78 GB/s but uchar4 coalesced reads at 25.05 GB/s -- the SMs can
// DMA at PCIe speed if the loads are wide. One warp per nonzero column (1344
// B) and one warp per active panel's scale block (2688 B), copied from the
// mapped host bank into a device mirror AT THE SAME OFFSETS, so the masked
// GEMV below runs unchanged on device memory.
extern "C" __global__ void gather_down_sparse(
    const unsigned char* __restrict__ src_base,   // mapped host, panel-major record
    unsigned char*       __restrict__ dst_base,   // device mirror of the record
    const int*           __restrict__ panel_list,
    const int*           __restrict__ panel_count,
    const int*           __restrict__ nz_list,
    const int*           __restrict__ nz_count,
    const int rows)                               // 2688
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int ncol = *nz_count;
    const int rowhalf = rows >> 1;                // 1344 B per column
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    if (warp < ncol) {
        const int j = nz_list[warp];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rowhalf / 4; k += 32) d[k] = s[k];
    } else if (warp < ncol + *panel_count) {
        const int p = panel_list[warp - ncol];
        const size_t off = (size_t)p * panel_stride;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rows / 4; k += 32) d[k] = s[k];
    }
}

extern "C" __global__ void gemv_down_masked_partial(
    const unsigned char* __restrict__ bank,   // panel-major down block (may be mapped host)
    const float*         __restrict__ act,    // [inter]
    const int*           __restrict__ panel_list,
    const unsigned int*  __restrict__ panel_masks,
    const int*           __restrict__ panel_count,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    float*               __restrict__ partials, // [nchunks, rows]
    const float global_scale,
    const int rows,
    const int inter)
{
    const int nchunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    __shared__ float s_e2m1[16];
    __shared__ float s_e4m3[256];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256) s_e4m3[threadIdx.x] = e4m3_lut[threadIdx.x];
    __syncthreads();

    const int hb = row >> 1;
    const int hi = row & 1;
    const int rowhalf = rows >> 1;
    const int pcount = *panel_count;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;

    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const unsigned char* __restrict__ pbase = bank + (size_t)p * panel_stride;
        const float s = s_e4m3[pbase[row]] * global_scale;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = panel_masks[p];
        while (m) {
            const int c = __ffs(m) - 1;
            m &= m - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float w = s_e2m1[hi ? (byte >> 4) : (byte & 15)] * s;
            acc = fmaf(w, act[(p << 4) + c], acc);
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}

// NERVF-3: ERVF form of gemv_nvfp4_rows (width 16), additive.

#define WIDTH 16
#define VIRTUAL (256 / WIDTH)
#define ROWS_PER_BLOCK (256 / WIDTH)

// ERVF form of gemv_nvfp4_rows.
//
// Reference: 256 threads per row; thread tid walks v = tid, tid+256, ...; then a
// 32-wide butterfly per warp, warp sums through shared memory, then a second
// butterfly over the 8 warp sums.
//
// Here: WIDTH physical lanes per row, ROWS_PER_BLOCK rows per 256-thread block.
// Lane L holds VIRTUAL accumulators for the virtual threads tid = L + WIDTH*vi,
// so no MAC moves and no accumulator merges early. The reference tree is then
// rebuilt exactly:
//   * its first step (offset 16 inside a 32-warp) pairs tid and tid+16, which in
//     this mapping are two virtual accumulators OF THE SAME PHYSICAL LANE ->
//     a lane-local add, no shuffle;
//   * offsets 8/4/2/1 stay shuffles, now within a WIDTH-wide subwarp;
//   * the 8 warp sums combine in registers in exactly the order the reference's
//     second butterfly imposes: ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)).
extern "C" __global__ void gemv_nvfp4_ervf(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    const float*         __restrict__ x,
    float*               __restrict__ out,
    const float global_scale,
    const int rows, const int cols,
    const int apply_relu2, const float out_scale)
{
    extern __shared__ float sx[];                 // [cols], shared by all rows
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int lane = (int)threadIdx.x & (WIDTH - 1);
    const int sub  = (int)threadIdx.x / WIDTH;
    const int row  = blockIdx.x * ROWS_PER_BLOCK + sub;
    if (row >= rows) return;

    const int n_bytes  = cols >> 1;
    const int n_vec    = n_bytes >> 2;
    const unsigned char* __restrict__ crow = codes  + (size_t)row * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)row * (cols >> 4);
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);

    float part[VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) part[vi] = 0.0f;

    // Each virtual thread walks exactly the stride the reference gave it.
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) {
        const int tid = lane + WIDTH * vi;
        float acc = 0.0f;
        for (int v = tid; v < n_vec; v += 256) {
            const uchar4 q = crow4[v];
            const int b = v << 2;
            const float s = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            acc = fmaf(s_e2m1[q.x & 0x0F] * s, sx[k],     acc);
            acc = fmaf(s_e2m1[q.x >> 4]   * s, sx[k + 1], acc);
            acc = fmaf(s_e2m1[q.y & 0x0F] * s, sx[k + 2], acc);
            acc = fmaf(s_e2m1[q.y >> 4]   * s, sx[k + 3], acc);
            acc = fmaf(s_e2m1[q.z & 0x0F] * s, sx[k + 4], acc);
            acc = fmaf(s_e2m1[q.z >> 4]   * s, sx[k + 5], acc);
            acc = fmaf(s_e2m1[q.w & 0x0F] * s, sx[k + 6], acc);
            acc = fmaf(s_e2m1[q.w >> 4]   * s, sx[k + 7], acc);
        }
        for (int b = (n_vec << 2) + tid; b < n_bytes; b += 256) {
            const unsigned char byte = crow[b];
            const float s = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            acc = fmaf(s_e2m1[byte & 0x0F] * s, sx[k],     acc);
            acc = fmaf(s_e2m1[byte >> 4]   * s, sx[k + 1], acc);
        }
        part[vi] = acc;
    }

    // ---- rebuild the reference reduction tree, exactly.
    // Virtual tid t sits in reference warp t/32 at intra-warp lane t%32.
    // With tid = lane + WIDTH*vi and lane < WIDTH <= 32, the reference's
    // offset-16 step pairs virtual accumulators of THIS lane whenever
    // WIDTH <= 16; for WIDTH == 32 lane and lane+16 are different lanes and the
    // step stays a shuffle. Both cases are handled below.
    float s8[8];
#if WIDTH <= 16
    // Reference offsets >= WIDTH act on the virtual index, not on lanes, so they
    // must be folded in BUTTERFLY order (16, 8, 4, ... scaled by WIDTH), not by
    // summing the virtual accumulators sequentially. Sequential folding is what
    // made w=4 and w=8 non-exact in the first pass while w=16 -- where the
    // butterfly happens to be a single step -- came out identical.
    const int per_warp = 32 / WIDTH;              // virtual indices per ref warp
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float loc[per_warp];
        #pragma unroll
        for (int u = 0; u < per_warp; ++u) loc[u] = part[w * per_warp + u];
        #pragma unroll
        for (int stride = per_warp >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int u = 0; u < per_warp; ++u)
                if (u < stride) loc[u] += loc[u + stride];
        }
        float v = loc[0];
        for (int off = WIDTH >> 1; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, WIDTH);
        s8[w] = v;
    }
#else
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float v = part[w];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, 32);
        s8[w] = v;
    }
#endif
    if (lane == 0) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        const float u0 = t0 + t2;
        const float u1 = t1 + t3;
        const float v  = u0 + u1;
        if (apply_relu2) { const float r = fmaxf(v, 0.0f); out[row] = r * r; }
        else             { out[row] = v * out_scale; }
    }
}

extern "C" __global__ void reduce_partials(
    const float* __restrict__ partials,
    float*       __restrict__ out,
    const int    rows,
    const int    nchunks)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float a = 0.0f;
    for (int c = 0; c < nchunks; c++) a += partials[(size_t)c * rows + row];
    out[row] = a;
}

// ---------------------------------------------------------------------------
// E1 fase 2.1: device-resident routing + device-LRU cache. These kernels let
// the whole MoE layer run without a single device->host synchronisation; the
// host only launches. All arithmetic bodies are identical to the kernels they
// mirror -- the only difference is WHERE scalar arguments are read from
// (device buffers instead of by-value launch arguments), so results are
// bit-comparable with the host-driven path.
// ---------------------------------------------------------------------------

// Device router head: sigmoid scores + bias, serial top-k (low-index wins
// ties), weights normalised in route order. bad_pick is the control-arm
// sabotage: slot top_k-1 takes the (top_k+1)-th best expert instead.
extern "C" __global__ void route_topk_f32(
    const float* __restrict__ rlog, const float* __restrict__ gate_b,
    int* __restrict__ ids, float* __restrict__ w,
    const int n, const int top_k, const float scaling, const int bad_pick)
{
    __shared__ float sc[128], ch[128];
    const int i = threadIdx.x;
    if (i < n) {
        const float s = 1.0f / (1.0f + expf(-rlog[i]));
        sc[i] = s;
        ch[i] = s + gate_b[i];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float wsum = 0.0f;
    for (int s = 0; s < top_k; s++) {
        int bi = 0;
        float bv = -3.0e38f;
        for (int e = 0; e < n; e++)
            if (ch[e] > bv) { bv = ch[e]; bi = e; }
        if (bad_pick && s == top_k - 1) {
            ch[bi] = -3.0e38f; bi = 0; bv = -3.0e38f;
            for (int e = 0; e < n; e++)
                if (ch[e] > bv) { bv = ch[e]; bi = e; }
        }
        ids[s] = bi;
        w[s] = sc[bi];
        ch[bi] = -3.0e38f;
        wsum += sc[bi];
    }
    for (int s = 0; s < top_k; s++)
        w[s] = w[s] / (wsum + 1e-20f) * scaling;
}

// Device LRU: one thread walks the top-k ids in route order, updating the
// slot tables exactly as the host LRU did (append while filling, then
// min-last-used eviction with lowest-slot tiebreak).
extern "C" __global__ void cache_assign(
    int* __restrict__ slot_of,      // [n_experts], init -1
    int* __restrict__ expert_of,    // [cap], init -1
    int* __restrict__ last_used,    // [cap], init -1
    int* __restrict__ state2,       // [0]=tick, [1]=filled
    const int* __restrict__ ids,
    int* __restrict__ slots, int* __restrict__ need,
    int* __restrict__ stats2,       // [0]=hits, [1]=misses
    const int cap, const int top_k)
{
    if (threadIdx.x != 0) return;
    int tick = state2[0], filled = state2[1];
    for (int s = 0; s < top_k; s++) {
        const int e = ids[s];
        const int sl = slot_of[e];
        if (sl >= 0) {
            last_used[sl] = ++tick;
            slots[s] = sl;
            need[s] = 0;
            stats2[0] += 1;
            continue;
        }
        stats2[1] += 1;
        int v;
        if (filled < cap) {
            v = filled++;
        } else {
            v = 0;
            int mnv = last_used[0];
            for (int cix = 1; cix < cap; cix++)
                if (last_used[cix] < mnv) { mnv = last_used[cix]; v = cix; }
            slot_of[expert_of[v]] = -1;
        }
        slot_of[e] = v;
        expert_of[v] = e;
        last_used[v] = ++tick;
        slots[s] = v;
        need[s] = 1;
    }
    state2[0] = tick;
    state2[1] = filled;
}

// Bulk staging of miss experts from the pinned host bank into the device
// cache: the M1 pattern (wide uint4, ~25 GB/s measured), not the DMA engine.
// One blockIdx.x per route slot; blocks that got a hit return immediately.
extern "C" __global__ void cache_fetch(
    const unsigned char* __restrict__ bank_c,
    const unsigned char* __restrict__ bank_s,
    unsigned char* __restrict__ cache_c,
    unsigned char* __restrict__ cache_s,
    const int* __restrict__ ids, const int* __restrict__ slots,
    const int* __restrict__ need,
    const size_t code_bytes, const size_t scale_bytes)
{
    const int s = blockIdx.x;
    if (!need[s]) return;
    const size_t nc4 = code_bytes >> 4, ns4 = scale_bytes >> 4;
    const uint4* csrc = reinterpret_cast<const uint4*>(
        bank_c + (size_t)ids[s] * code_bytes);
    const uint4* ssrc = reinterpret_cast<const uint4*>(
        bank_s + (size_t)ids[s] * scale_bytes);
    uint4* cdst = reinterpret_cast<uint4*>(
        cache_c + (size_t)slots[s] * code_bytes);
    uint4* sdst = reinterpret_cast<uint4*>(
        cache_s + (size_t)slots[s] * scale_bytes);
    const size_t total = nc4 + ns4;
    const size_t stride = (size_t)gridDim.y * blockDim.x;
    for (size_t v = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
         v < total; v += stride) {
        if (v < nc4) cdst[v] = csrc[v];
        else         sdst[v - nc4] = ssrc[v - nc4];
    }
}

// Indirect ERVF up-GEMV: identical body to gemv_nvfp4_ervf, but codes/scales
// are located through the device slot table and global_scale comes from the
// device globals table. Same bytes, same order, same values -> bit-comparable.
extern "C" __global__ void gemv_nvfp4_ervf_ind(
    const unsigned char* __restrict__ codes_base,
    const unsigned char* __restrict__ scales_base,
    const int*           __restrict__ slot_ptr,
    const int*           __restrict__ id_ptr,
    const float*         __restrict__ globals, const int gsel,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    const float*         __restrict__ x,
    float*               __restrict__ out,
    const int rows, const int cols,
    const int apply_relu2, const float out_scale,
    const size_t code_stride, const size_t scale_stride)
{
    const int slot = *slot_ptr;
    const int e = *id_ptr;
    const unsigned char* __restrict__ codes = codes_base + (size_t)slot * code_stride;
    const unsigned char* __restrict__ scales = scales_base + (size_t)slot * scale_stride;
    const float global_scale = globals[e * 2 + gsel];

    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int lane = (int)threadIdx.x & (WIDTH - 1);
    const int sub  = (int)threadIdx.x / WIDTH;
    const int row  = blockIdx.x * ROWS_PER_BLOCK + sub;
    if (row >= rows) return;

    const int n_bytes  = cols >> 1;
    const int n_vec    = n_bytes >> 2;
    const unsigned char* __restrict__ crow = codes  + (size_t)row * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)row * (cols >> 4);
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);

    float part[VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) part[vi] = 0.0f;

    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) {
        const int tid = lane + WIDTH * vi;
        float acc = 0.0f;
        for (int v = tid; v < n_vec; v += 256) {
            const uchar4 q = crow4[v];
            const int b = v << 2;
            const float s = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            acc = fmaf(s_e2m1[q.x & 0x0F] * s, sx[k],     acc);
            acc = fmaf(s_e2m1[q.x >> 4]   * s, sx[k + 1], acc);
            acc = fmaf(s_e2m1[q.y & 0x0F] * s, sx[k + 2], acc);
            acc = fmaf(s_e2m1[q.y >> 4]   * s, sx[k + 3], acc);
            acc = fmaf(s_e2m1[q.z & 0x0F] * s, sx[k + 4], acc);
            acc = fmaf(s_e2m1[q.z >> 4]   * s, sx[k + 5], acc);
            acc = fmaf(s_e2m1[q.w & 0x0F] * s, sx[k + 6], acc);
            acc = fmaf(s_e2m1[q.w >> 4]   * s, sx[k + 7], acc);
        }
        for (int b = (n_vec << 2) + tid; b < n_bytes; b += 256) {
            const unsigned char byte = crow[b];
            const float s = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            acc = fmaf(s_e2m1[byte & 0x0F] * s, sx[k],     acc);
            acc = fmaf(s_e2m1[byte >> 4]   * s, sx[k + 1], acc);
        }
        part[vi] = acc;
    }

    float s8[8];
    const int per_warp = 32 / WIDTH;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float loc[per_warp];
        #pragma unroll
        for (int u = 0; u < per_warp; ++u) loc[u] = part[w * per_warp + u];
        #pragma unroll
        for (int stride = per_warp >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int u = 0; u < per_warp; ++u)
                if (u < stride) loc[u] += loc[u + stride];
        }
        float v = loc[0];
        for (int off = WIDTH >> 1; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, WIDTH);
        s8[w] = v;
    }
    if (lane == 0) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        const float u0 = t0 + t2;
        const float u1 = t1 + t3;
        const float v  = u0 + u1;
        if (apply_relu2) { const float r = fmaxf(v, 0.0f); out[row] = r * r; }
        else             { out[row] = v * out_scale; }
    }
}

// Indirect sparse gather: the expert's panel-major down record is located via
// the device id buffer; body identical to gather_down_sparse.
extern "C" __global__ void gather_down_sparse_ind(
    const unsigned char* __restrict__ down_base,
    const int*           __restrict__ id_ptr,
    const size_t         panel_bytes,
    unsigned char*       __restrict__ dst_base,
    const int*           __restrict__ panel_list,
    const int*           __restrict__ panel_count,
    const int*           __restrict__ nz_list,
    const int*           __restrict__ nz_count,
    const int rows)
{
    const unsigned char* __restrict__ src_base =
        down_base + (size_t)(*id_ptr) * panel_bytes;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int ncol = *nz_count;
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    if (warp < ncol) {
        const int j = nz_list[warp];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rowhalf / 4; k += 32) d[k] = s[k];
    } else if (warp < ncol + *panel_count) {
        const int p = panel_list[warp - ncol];
        const size_t off = (size_t)p * panel_stride;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rows / 4; k += 32) d[k] = s[k];
    }
}

// Indirect masked down-GEMV: global_scale via the device globals table.
extern "C" __global__ void gemv_down_masked_partial_ind(
    const unsigned char* __restrict__ bank,
    const int*           __restrict__ id_ptr,
    const float*         __restrict__ globals,
    const float*         __restrict__ act,
    const int*           __restrict__ panel_list,
    const unsigned int*  __restrict__ panel_masks,
    const int*           __restrict__ panel_count,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    float*               __restrict__ partials,
    const int rows,
    const int inter)
{
    const float global_scale = globals[(*id_ptr) * 2 + 0];
    const int nchunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    __shared__ float s_e2m1[16];
    __shared__ float s_e4m3[256];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256) s_e4m3[threadIdx.x] = e4m3_lut[threadIdx.x];
    __syncthreads();

    const int hb = row >> 1;
    const int hi = row & 1;
    const int rowhalf = rows >> 1;
    const int pcount = *panel_count;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;

    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const unsigned char* __restrict__ pbase = bank + (size_t)p * panel_stride;
        const float s = s_e4m3[pbase[row]] * global_scale;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = panel_masks[p];
        while (m) {
            const int c = __ffs(m) - 1;
            m &= m - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float w = s_e2m1[hi ? (byte >> 4) : (byte & 15)] * s;
            acc = fmaf(w, act[(p << 4) + c], acc);
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}

// Weighted accumulate with the route weight read from device memory.
extern "C" __global__ void weighted_accumulate_ind(
    float*       __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ w_ptr,
    const int    n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = fmaf(src[i], *w_ptr, dst[i]);
}
"""


class FusedNVFP4:
    """Compiled fused kernels plus the constant lookup tables."""

    def __init__(self, block: int = 256):
        import cupy as cp

        self.cp = cp
        self.block = block
        self._module = cp.RawModule(code=_KERNEL_SOURCE, options=("-std=c++14",))
        self.gemv = self._module.get_function("gemv_nvfp4_rows")
        self.accumulate = self._module.get_function("weighted_accumulate")

        self.e2m1 = cp.asarray(nvfp4.E2M1_TABLE, dtype=cp.float32)
        self.e4m3 = cp.asarray(
            np.nan_to_num(nvfp4.E4M3_TABLE, nan=0.0), dtype=cp.float32)

        self.gemv_ervf = self._module.get_function("gemv_nvfp4_ervf")
        # NERVF-3: opt-in. Default off so every earlier measurement still
        # describes the kernel it measured.
        # A1 adoption (2026-08-15): ERVF is the default GEMV path. NERVF-2
        # proved it bitwise identical to the production kernel at all four
        # widths (0/72 mismatches each) and 1.936x faster at width 16; E5 showed
        # every real shape improves. Set False for a legacy arm.
        self.use_ervf = True
        self.ervf_width = 16
        # NERVF-4: skip gather_down_sparse and let the masked GEMV read the
        # panel-major record straight from the mapped host bank. Same bytes,
        # same panel walk, same accumulation order -- only the staging hop
        # disappears. Opt-in; default off.
        self.gatherless_down = False
        self.panel_scan_k = self._module.get_function("panel_scan")
        self.down_masked = self._module.get_function("gemv_down_masked_partial")
        self.reduce_partials_k = self._module.get_function("reduce_partials")
        self.gather_k = self._module.get_function("gather_down_sparse")
        self.nchunks = 8
        # E1 fase 2.1: device-resident routing + cache (opt-in via the
        # runtime's device_cache flag; all indirect kernels are bit-comparable
        # with the host-driven path they mirror).
        self.route_topk_k = self._module.get_function("route_topk_f32")
        self.cache_assign_k = self._module.get_function("cache_assign")
        self.cache_fetch_k = self._module.get_function("cache_fetch")
        self.gemv_ervf_ind_k = self._module.get_function("gemv_nvfp4_ervf_ind")
        self.gather_ind_k = self._module.get_function("gather_down_sparse_ind")
        self.down_masked_ind_k = self._module.get_function(
            "gemv_down_masked_partial_ind")
        self.accumulate_ind_k = self._module.get_function(
            "weighted_accumulate_ind")

    # -- E1 fase 2.1 wrappers ------------------------------------------------
    def route_topk(self, rlog, gate_b, ids, w, n, top_k, scaling,
                   bad_pick: int = 0) -> None:
        self.route_topk_k((1,), (128,),
                          (rlog, gate_b, ids, w, np.int32(n), np.int32(top_k),
                           np.float32(scaling), np.int32(bad_pick)))

    def cache_assign(self, dev, ids, cap: int, top_k: int) -> None:
        self.cache_assign_k((1,), (32,),
                            (dev["slot_of"], dev["expert_of"], dev["last_used"],
                             dev["state2"], ids, dev["slots"], dev["need"],
                             dev["stats2"], np.int32(cap), np.int32(top_k)))

    def cache_fetch(self, bank_c_ptr: int, bank_s_ptr: int, cache_c, cache_s,
                    dev, code_bytes: int, scale_bytes: int,
                    top_k: int) -> None:
        jblocks = 64
        self.cache_fetch_k((top_k, jblocks), (256,),
                           (np.uint64(bank_c_ptr), np.uint64(bank_s_ptr),
                            cache_c, cache_s, dev["ids"], dev["slots"],
                            dev["need"], np.uint64(code_bytes),
                            np.uint64(scale_bytes)))

    def gemv_ervf_indirect(self, out, cache_c, cache_s, dev, s: int,
                           globals_dev, gsel: int, x, rows: int, cols: int,
                           apply_relu2: bool, code_stride: int,
                           scale_stride: int) -> None:
        rpb = 256 // self.ervf_width
        self.gemv_ervf_ind_k(
            ((rows + rpb - 1) // rpb,), (self.block,),
            (cache_c, cache_s, dev["slots"][s:], dev["ids"][s:], globals_dev,
             np.int32(gsel), self.e2m1, self.e4m3, x, out,
             np.int32(rows), np.int32(cols),
             np.int32(1 if apply_relu2 else 0), np.float32(1.0),
             np.uint64(code_stride), np.uint64(scale_stride)),
            shared_mem=cols * 4)

    def down_masked_into_indirect(self, out, down_base_ptr: int, dev, s: int,
                                  globals_dev, act, state, hidden: int,
                                  intermediate: int, panel_bytes: int) -> None:
        """The masked down path with expert identity read on device."""
        npanel = intermediate // 16
        nc = self.nchunks
        self.panel_scan_k((1,), (256,),
                          (act, np.int32(intermediate), state["masks"],
                           state["plist"], state["pcount"],
                           state["nz"], state["nzc"]))
        max_warps = intermediate + npanel
        blocks = (max_warps * 32 + 255) // 256
        self.gather_ind_k((blocks,), (256,),
                          (np.uint64(down_base_ptr), dev["ids"][s:],
                           np.uint64(panel_bytes), state["mirror"],
                           state["plist"], state["pcount"],
                           state["nz"], state["nzc"], np.int32(hidden)))
        grid = ((hidden + 127) // 128, nc)
        self.down_masked_ind_k(grid, (128,),
                               (state["mirror"], dev["ids"][s:], globals_dev,
                                act, state["plist"], state["masks"],
                                state["pcount"], self.e2m1, self.e4m3,
                                state["partials"], np.int32(hidden),
                                np.int32(intermediate)))
        self.reduce_partials_k(((hidden + 255) // 256,), (256,),
                               (state["partials"], out, np.int32(hidden),
                                np.int32(nc)))

    def accumulate_indirect(self, dst, src, w_ptr_elem, n: int) -> None:
        threads = 256
        blocks = (n + threads - 1) // threads
        self.accumulate_ind_k((blocks,), (threads,),
                              (dst, src, w_ptr_elem, np.int32(n)))

    def alloc_device_cache(self, n_experts: int, cap: int, top_k: int,
                           globals_host: np.ndarray):
        """Per-layer device state for the graph-safe MoE path."""
        cp = self.cp
        return {
            "ids": cp.zeros(top_k, dtype=cp.int32),
            "w": cp.zeros(top_k, dtype=cp.float32),
            "slots": cp.zeros(top_k, dtype=cp.int32),
            "need": cp.zeros(top_k, dtype=cp.int32),
            "slot_of": cp.full(n_experts, -1, dtype=cp.int32),
            "expert_of": cp.full(cap, -1, dtype=cp.int32),
            "last_used": cp.full(cap, -1, dtype=cp.int32),
            "state2": cp.zeros(2, dtype=cp.int32),
            "stats2": cp.zeros(2, dtype=cp.int32),
            "globals": cp.asarray(globals_host),
        }


    def alloc_masked_state(self, hidden: int, intermediate: int):
        """Device scratch for the masked down path (panel metadata + partials
        + sparse mirror of one down_proj record)."""
        cp = self.cp
        npanel = intermediate // 16
        return {
            "masks": cp.zeros(npanel, dtype=cp.uint32),
            "plist": cp.zeros(npanel, dtype=cp.int32),
            "pcount": cp.zeros(1, dtype=cp.int32),
            "nz": cp.zeros(intermediate, dtype=cp.int32),
            "nzc": cp.zeros(1, dtype=cp.int32),
            "partials": cp.zeros(self.nchunks * hidden, dtype=cp.float32),
            # sparse device mirror of one panel-major down record
            "mirror": cp.zeros(npanel * (hidden + 16 * (hidden // 2)),
                               dtype=cp.uint8),
        }

    def down_masked_into(self, out, bank_ptr: int, act, state, global_scale: float,
                         hidden: int, intermediate: int,
                         nchunks: int | None = None,
                         gather_from_host: bool = True) -> None:
        """out[hidden] = sum over nonzero act columns of dequant(down_col)*act.

        ``bank_ptr`` is the address of the expert's panel-major down block in
        mapped pinned HOST memory. With gather_from_host the gather kernel
        (wide uchar4 loads, ~25 GB/s measured) first pulls exactly the nonzero
        columns and active scale blocks into the device mirror; the masked
        GEMV then runs on device memory. Only those bytes cross PCIe.
        """
        cp = self.cp
        nc = nchunks or self.nchunks
        npanel = intermediate // 16
        self.panel_scan_k((1,), (256,),
                          (act, np.int32(intermediate), state["masks"],
                           state["plist"], state["pcount"],
                           state["nz"], state["nzc"]))
        if self.gatherless_down:
            gather_from_host = False
        if gather_from_host:
            max_warps = intermediate + npanel
            blocks = (max_warps * 32 + 255) // 256
            self.gather_k((blocks,), (256,),
                          (np.uint64(bank_ptr), state["mirror"],
                           state["plist"], state["pcount"],
                           state["nz"], state["nzc"], np.int32(hidden)))
            src = state["mirror"].data.ptr
        else:
            src = bank_ptr
        grid = ((hidden + 127) // 128, nc)
        self.down_masked(grid, (128,),
                         (np.uint64(src), act, state["plist"],
                          state["masks"], state["pcount"],
                          self.e2m1, self.e4m3, state["partials"],
                          np.float32(global_scale), np.int32(hidden),
                          np.int32(intermediate)))
        self.reduce_partials_k(((hidden + 255) // 256,), (256,),
                               (state["partials"], out, np.int32(hidden),
                                np.int32(nc)))

    def gemv_into(self, out, codes, scales, x, global_scale: float,
                  rows: int, cols: int, apply_relu2: bool = False,
                  out_scale: float = 1.0) -> None:
        """out[rows] = op(W[rows, cols] @ x[cols]) without materialising W."""
        shared = cols * 4
        if self.use_ervf:
            # Exact-Reduction Virtual Fusion: 16-lane subwarps, 16 rows per
            # block, per-lane virtual accumulators, reference reduction tree
            # rebuilt exactly. Bitwise identical to the branch below (NERVF-2,
            # 0/72 mismatches over 4 widths).
            rpb = 256 // self.ervf_width
            self.gemv_ervf(
                ((rows + rpb - 1) // rpb,), (self.block,),
                (codes, scales, self.e2m1, self.e4m3, x, out,
                 np.float32(global_scale), np.int32(rows), np.int32(cols),
                 np.int32(1 if apply_relu2 else 0), np.float32(out_scale)),
                shared_mem=shared)
            return
        self.gemv(
            (rows,), (self.block,),
            (codes, scales, self.e2m1, self.e4m3, x, out,
             np.float32(global_scale), np.int32(rows), np.int32(cols),
             np.int32(1 if apply_relu2 else 0), np.float32(out_scale)),
            shared_mem=shared,
        )

    def expert(self, up_codes, up_scales, up_gscale: float,
               down_codes, down_scales, down_gscale: float,
               x, act_buf, out_buf,
               hidden: int, intermediate: int) -> None:
        """One routed expert: up -> ReLU^2 -> down, fully fused per stage."""
        self.gemv_into(act_buf, up_codes, up_scales, x, up_gscale,
                       rows=intermediate, cols=hidden, apply_relu2=True)
        self.gemv_into(out_buf, down_codes, down_scales, act_buf, down_gscale,
                       rows=hidden, cols=intermediate, apply_relu2=False)

    def accumulate_into(self, dst, src, weight: float, n: int) -> None:
        threads = 256
        blocks = (n + threads - 1) // threads
        self.accumulate((blocks,), (threads,),
                        (dst, src, np.float32(weight), np.int32(n)))
