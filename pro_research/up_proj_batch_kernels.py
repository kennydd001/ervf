"""Batched variant of gemv_nvfp4_ervf_ind (the up-proj ERVF GEMV used by
_moe_dev, called top_k=6 times per layer). Unlike weighted_accumulate_ind,
each slot's up-proj GEMV writes to an INDEPENDENT output region (no shared
accumulator, no race) -- so this is the same class of mechanical batching
as panel_scan/reduce_partials, not the harder accumulate case.

x (the normalized hidden state) is IDENTICAL across all top_k slots -- only
which expert's codes/scales/global_scale is read differs. The batched kernel
adds a slot dimension (blockIdx.y) that selects slot/id/output offset;
every per-thread instruction inside a block is byte-for-byte identical to
the reference kernel gemv_nvfp4_ervf_ind (fused_nvfp4.py line 548) --
copied verbatim below except for that one addition, specifically to avoid
transcription risk in the WIDTH=16 subwarp-butterfly reduction tree that
NERVF-2 hand-verified bit-exact against the production kernel.

grid = ((rows+ROWS_PER_BLOCK-1)//ROWS_PER_BLOCK, top_k), same block size and
shared_mem per launch as the reference -- shared memory is per-block, so
each block still loads its own copy of x; the win is launch-count reduction
(top_k launches -> 1), not shared-memory reuse.
"""

from __future__ import annotations

import numpy as np

WIDTH = 16
VIRTUAL = 256 // WIDTH
ROWS_PER_BLOCK = 256 // WIDTH

CUDA_SOURCE = r"""
#define WIDTH 16
#define VIRTUAL (256 / WIDTH)
#define ROWS_PER_BLOCK (256 / WIDTH)

extern "C" __global__ void gemv_nvfp4_ervf_ind_ref(
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

// Batched: blockIdx.y = slot s. Every per-thread instruction below is
// identical to gemv_nvfp4_ervf_ind_ref above -- only slot/id/output-offset
// addressing changes. x is the SAME activation for every slot (each block
// still loads its own shared-memory copy; the win is launch count, not
// shared-memory reuse).
extern "C" __global__ void gemv_nvfp4_ervf_ind_batched(
    const unsigned char* __restrict__ codes_base,
    const unsigned char* __restrict__ scales_base,
    const int*           __restrict__ slots,       // [top_k]
    const int*           __restrict__ ids,          // [top_k]
    const float*         __restrict__ globals, const int gsel,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    const float*         __restrict__ x,
    float*               __restrict__ out,          // [top_k, rows]
    const int rows, const int cols,
    const int apply_relu2, const float out_scale,
    const size_t code_stride, const size_t scale_stride)
{
    const int s = blockIdx.y;
    const int slot = slots[s];
    const int e = ids[s];
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
        if (apply_relu2) { const float r = fmaxf(v, 0.0f); out[(size_t)s * rows + row] = r * r; }
        else             { out[(size_t)s * rows + row] = v * out_scale; }
    }
}
"""


class UpProjBatchKernels:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14", "--use_fast_math"))
        self.ref = self.mod.get_function("gemv_nvfp4_ervf_ind_ref")
        self.batched = self.mod.get_function("gemv_nvfp4_ervf_ind_batched")
        self.block = 256

    def run_ref(self, out, codes_base, scales_base, slot_ptr, id_ptr, globals_dev, gsel,
                e2m1_lut, e4m3_lut, x, rows, cols, apply_relu2, code_stride, scale_stride):
        rpb = 256 // WIDTH
        self.ref(
            ((rows + rpb - 1) // rpb,), (self.block,),
            (codes_base, scales_base, slot_ptr, id_ptr, globals_dev, np.int32(gsel),
             e2m1_lut, e4m3_lut, x, out, np.int32(rows), np.int32(cols),
             np.int32(1 if apply_relu2 else 0), np.float32(1.0),
             np.uint64(code_stride), np.uint64(scale_stride)),
            shared_mem=cols * 4,
        )

    def run_batched(self, out, codes_base, scales_base, slots, ids, globals_dev, gsel,
                    e2m1_lut, e4m3_lut, x, rows, cols, apply_relu2, code_stride, scale_stride, top_k):
        rpb = 256 // WIDTH
        self.batched(
            ((rows + rpb - 1) // rpb, top_k), (self.block,),
            (codes_base, scales_base, slots, ids, globals_dev, np.int32(gsel),
             e2m1_lut, e4m3_lut, x, out, np.int32(rows), np.int32(cols),
             np.int32(1 if apply_relu2 else 0), np.float32(1.0),
             np.uint64(code_stride), np.uint64(scale_stride)),
            shared_mem=cols * 4,
        )
