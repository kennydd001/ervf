
"""Exact contiguous sparse-code mirror for routed down projection."""
from __future__ import annotations
import numpy as np

CUDA_SOURCE = r"""
extern "C" __global__ void panel_offsets_batched(
    const unsigned int* __restrict__ masks,
    const int npanel,
    int* __restrict__ offsets)
{
    const int s = blockIdx.x;
    if (threadIdx.x != 0) return;
    const unsigned int* m = masks + (size_t)s * npanel;
    int* o = offsets + (size_t)s * npanel;
    int n = 0;
    for (int p = 0; p < npanel; ++p) {
        o[p] = n;
        n += __popc(m[p]);
    }
}

extern "C" __global__ void gather_down_cols_packed_ind(
    const unsigned char* __restrict__ down_base,
    const int* __restrict__ id_ptr,
    const size_t panel_bytes,
    unsigned char* __restrict__ packed,
    const int* __restrict__ nz_list,
    const int* __restrict__ nz_count,
    const int rows)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= *nz_count) return;

    const int j = nz_list[warp];
    const int rowhalf = rows >> 1;
    const size_t panel_stride =
        (size_t)rows + 16u * (size_t)rowhalf;
    const unsigned char* rec =
        down_base + (size_t)(*id_ptr) * panel_bytes;
    const size_t src_off =
        (size_t)(j >> 4) * panel_stride + rows
        + (size_t)(j & 15) * rowhalf;

    const uchar4* src =
        reinterpret_cast<const uchar4*>(rec + src_off);
    uchar4* dst = reinterpret_cast<uchar4*>(
        packed + (size_t)warp * rowhalf
    );
    for (int q = lane; q < rowhalf / 4; q += 32)
        dst[q] = src[q];
}

extern "C" __global__ void gemv_down_packed_partial_ind_sres(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ planes,
    const int* __restrict__ slot_ptr,
    const int* __restrict__ id_ptr,
    const float* __restrict__ globals,
    const float* __restrict__ act,
    const int* __restrict__ panel_list,
    const unsigned int* __restrict__ panel_masks,
    const int* __restrict__ panel_offsets,
    const int* __restrict__ panel_count,
    const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut,
    float* __restrict__ partials,
    const size_t plane_bytes,
    const int rows,
    const int inter)
{
    const int id = *id_ptr;
    const float global_scale = globals[id * 2 + 0];
    const int nchunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    __shared__ float e2[16];
    __shared__ float e4[256];
    if (threadIdx.x < 16) e2[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256) e4[threadIdx.x] = e4m3_lut[threadIdx.x];
    __syncthreads();

    const int hb = row >> 1;
    const int hi = row & 1;
    const int rowhalf = rows >> 1;
    const int pcount = *panel_count;
    const unsigned char* plane =
        planes + (size_t)(*slot_ptr) * plane_bytes;

    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const float scale =
            e4[plane[(size_t)p * rows + row]] * global_scale;
        unsigned int mask = panel_masks[p];
        const int packed_base = panel_offsets[p];
        int rank = 0;
        while (mask) {
            const int c = __ffs(mask) - 1;
            mask &= mask - 1;
            const unsigned char byte = packed[
                (size_t)(packed_base + rank) * rowhalf + hb
            ];
            const float w =
                e2[hi ? (byte >> 4) : (byte & 15)] * scale;
            acc = fmaf(w, act[(p << 4) + c], acc);
            ++rank;
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}
"""


class Phase7PackedKernels:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.mod = cp.RawModule(
            code=CUDA_SOURCE, options=("-std=c++14",)
        )
        self.panel_offsets = self.mod.get_function(
            "panel_offsets_batched"
        )
        self.gather_packed = self.mod.get_function(
            "gather_down_cols_packed_ind"
        )
        self.down_packed = self.mod.get_function(
            "gemv_down_packed_partial_ind_sres"
        )
