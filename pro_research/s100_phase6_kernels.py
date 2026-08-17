"""Exact CUDA experiments for S100 phase 6.

- ballot panel scan removes atomicOr traffic while producing identical masks;
- direct down GEMV reads mapped host code bytes without a mirror copy;
- fused reduce/accumulate preserves chunk and route summation order.
"""
from __future__ import annotations
import numpy as np

CUDA_SOURCE = r"""
extern "C" __global__ void panel_scan_ballot_exact_batched(
    const float* __restrict__ act,
    const int inter,
    unsigned int* __restrict__ panel_masks,
    int* __restrict__ panel_list,
    int* __restrict__ panel_count,
    int* __restrict__ nz_list,
    int* __restrict__ nz_count)
{
    const int s = blockIdx.x;
    const int npanel = inter >> 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const float* act_s = act + (size_t)s * inter;
    unsigned int* masks_s = panel_masks + (size_t)s * npanel;
    int* list_s = panel_list + (size_t)s * npanel;
    int* nz_s = nz_list + (size_t)s * inter;

    for (int p = warp; p < npanel; p += nwarps) {
        bool keep = false;
        if (lane < 16) keep = act_s[(p << 4) + lane] != 0.0f;
        const unsigned int mk = __ballot_sync(0xffffffffu, keep) & 0xffffu;
        if (lane == 0) masks_s[p] = mk;
    }
    if (threadIdx.x == 0) { panel_count[s] = 0; nz_count[s] = 0; }
    __syncthreads();

    if (threadIdx.x == 0) {
        int n = 0, m = 0;
        for (int p = 0; p < npanel; ++p) {
            unsigned int mk = masks_s[p];
            if (mk) {
                list_s[n++] = p;
                for (int c = 0; c < 16; ++c)
                    if (mk & (1u << c)) nz_s[m++] = (p << 4) + c;
            }
        }
        panel_count[s] = n;
        nz_count[s] = m;
    }
}

extern "C" __global__ void panel_scan_ballot_threshold_batched(
    const float* __restrict__ act,
    const int inter,
    const float alpha,
    unsigned int* __restrict__ panel_masks,
    int* __restrict__ panel_list,
    int* __restrict__ panel_count,
    int* __restrict__ nz_list,
    int* __restrict__ nz_count,
    float* __restrict__ max_act)
{
    const int s = blockIdx.x;
    const int npanel = inter >> 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const float* act_s = act + (size_t)s * inter;
    unsigned int* masks_s = panel_masks + (size_t)s * npanel;
    int* list_s = panel_list + (size_t)s * npanel;
    int* nz_s = nz_list + (size_t)s * inter;
    __shared__ float red[256];

    float lm = 0.0f;
    for (int j = threadIdx.x; j < inter; j += blockDim.x)
        lm = fmaxf(lm, act_s[j]);
    red[threadIdx.x] = lm;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float cut = alpha * red[0];
    if (threadIdx.x == 0) {
        max_act[s] = red[0]; panel_count[s] = 0; nz_count[s] = 0;
    }

    for (int p = warp; p < npanel; p += nwarps) {
        bool keep = false;
        if (lane < 16) {
            const float v = act_s[(p << 4) + lane];
            keep = (v != 0.0f) && (v >= cut);
        }
        const unsigned int mk = __ballot_sync(0xffffffffu, keep) & 0xffffu;
        if (lane == 0) masks_s[p] = mk;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int n = 0, m = 0;
        for (int p = 0; p < npanel; ++p) {
            unsigned int mk = masks_s[p];
            if (mk) {
                list_s[n++] = p;
                for (int c = 0; c < 16; ++c)
                    if (mk & (1u << c)) nz_s[m++] = (p << 4) + c;
            }
        }
        panel_count[s] = n;
        nz_count[s] = m;
    }
}

extern "C" __global__ void gemv_down_masked_partial_direct_sres(
    const unsigned char* __restrict__ down_base,
    const int* __restrict__ id_ptr,
    const unsigned char* __restrict__ planes,
    const int* __restrict__ slot_ptr,
    const float* __restrict__ globals,
    const float* __restrict__ act,
    const int* __restrict__ panel_list,
    const unsigned int* __restrict__ panel_masks,
    const int* __restrict__ panel_count,
    const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut,
    float* __restrict__ partials,
    const size_t panel_bytes,
    const size_t plane_bytes,
    const int rows,
    const int inter)
{
    const int id = *id_ptr;
    const unsigned char* rec = down_base + (size_t)id * panel_bytes;
    const unsigned char* plane = planes + (size_t)(*slot_ptr) * plane_bytes;
    const float global_scale = globals[id * 2 + 0];
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
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    const int pcount = *panel_count;
    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const unsigned char* pbase = rec + (size_t)p * panel_stride;
        const float sc = s_e4m3[plane[(size_t)p * rows + row]] * global_scale;
        const unsigned char* pcodes = pbase + rows;
        unsigned int mk = panel_masks[p];
        while (mk) {
            const int c = __ffs(mk) - 1;
            mk &= mk - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float w = s_e2m1[hi ? (byte >> 4) : (byte & 15)] * sc;
            acc = fmaf(w, act[(p << 4) + c], acc);
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}

extern "C" __global__ void reduce_accumulate_fused(
    const float* __restrict__ partials,
    float* __restrict__ dst,
    const float* __restrict__ route_w,
    const int rows,
    const int nchunks,
    const int top_k)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float out = dst[row];
    for (int s = 0; s < top_k; ++s) {
        float a = 0.0f;
        const float* p = partials + (size_t)s * nchunks * rows;
        for (int c = 0; c < nchunks; ++c)
            a += p[(size_t)c * rows + row];
        out = fmaf(a, route_w[s], out);
    }
    dst[row] = out;
}
"""

class Phase6Kernels:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14",))
        self.scan_exact = self.mod.get_function("panel_scan_ballot_exact_batched")
        self.scan_threshold = self.mod.get_function("panel_scan_ballot_threshold_batched")
        self.down_direct = self.mod.get_function("gemv_down_masked_partial_direct_sres")
        self.reduce_accumulate = self.mod.get_function("reduce_accumulate_fused")
