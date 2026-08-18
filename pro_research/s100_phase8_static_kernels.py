
"""Preload and exact hybrid static routed-down record kernels."""
from __future__ import annotations

import numpy as np

CUDA_SOURCE = r"""
extern "C" __global__ void preload_down_records(
    const unsigned char* __restrict__ down_base,
    const int* __restrict__ expert_ids,
    unsigned char* __restrict__ records,
    const size_t record_bytes)
{
    const int slot = blockIdx.x;
    const int expert = expert_ids[slot];
    const unsigned char* src =
        down_base + (size_t)expert * record_bytes;
    unsigned char* dst =
        records + (size_t)slot * record_bytes;

    const size_t total4 = record_bytes >> 4;
    const size_t stride =
        (size_t)gridDim.y * blockDim.x;
    for (
        size_t index =
            (size_t)blockIdx.y * blockDim.x + threadIdx.x;
        index < total4;
        index += stride
    ) {
        reinterpret_cast<uint4*>(dst)[index] =
            reinterpret_cast<const uint4*>(src)[index];
    }
}

extern "C" __global__ void gather_down_cols_static_miss(
    const unsigned char* __restrict__ down_base,
    const int* __restrict__ id_ptr,
    const size_t panel_bytes,
    unsigned char* __restrict__ mirror,
    const int* __restrict__ nz_list,
    const int* __restrict__ nz_count,
    const int rows,
    const int* __restrict__ expert_to_static)
{
    const int expert = *id_ptr;
    if (expert_to_static[expert] >= 0) return;

    const unsigned char* src_base =
        down_base + (size_t)expert * panel_bytes;
    const int warp =
        (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rowhalf = rows >> 1;
    const size_t panel_stride =
        (size_t)rows + 16u * (size_t)rowhalf;

    if (warp < *nz_count) {
        const int column = nz_list[warp];
        const size_t offset =
            (size_t)(column >> 4) * panel_stride
            + rows
            + (size_t)(column & 15) * rowhalf;
        const uchar4* src =
            reinterpret_cast<const uchar4*>(
                src_base + offset
            );
        uchar4* dst =
            reinterpret_cast<uchar4*>(mirror + offset);
        for (int q = lane; q < rowhalf / 4; q += 32)
            dst[q] = src[q];
    }
}

extern "C" __global__ void gemv_down_static_or_mirror(
    const unsigned char* __restrict__ mirror,
    const unsigned char* __restrict__ static_records,
    const int* __restrict__ expert_to_static,
    const unsigned char* __restrict__ planes,
    const int* __restrict__ dynamic_slot_ptr,
    const int* __restrict__ id_ptr,
    const float* __restrict__ globals,
    const float* __restrict__ act,
    const int* __restrict__ panel_list,
    const unsigned int* __restrict__ panel_masks,
    const int* __restrict__ panel_count,
    const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut,
    float* __restrict__ partials,
    const size_t record_bytes,
    const size_t plane_bytes,
    const int rows,
    const int inter)
{
    const int expert = *id_ptr;
    const int static_slot = expert_to_static[expert];
    const unsigned char* bank =
        static_slot >= 0
        ? static_records + (size_t)static_slot * record_bytes
        : mirror;

    const float global_scale =
        globals[expert * 2 + 0];
    const int nchunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int row =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    __shared__ float e2[16];
    __shared__ float e4[256];
    if (threadIdx.x < 16)
        e2[threadIdx.x] = e2m1_lut[threadIdx.x];
    if (threadIdx.x < 256)
        e4[threadIdx.x] = e4m3_lut[threadIdx.x];
    __syncthreads();

    const int half_byte = row >> 1;
    const int high_nibble = row & 1;
    const int rowhalf = rows >> 1;
    const size_t panel_stride =
        (size_t)rows + 16u * (size_t)rowhalf;
    const unsigned char* plane =
        planes
        + (size_t)(*dynamic_slot_ptr) * plane_bytes;
    const int count = *panel_count;

    float acc = 0.0f;
    for (int pi = chunk; pi < count; pi += nchunks) {
        const int panel = panel_list[pi];
        const unsigned char* panel_base =
            bank + (size_t)panel * panel_stride;
        const float scale =
            e4[plane[(size_t)panel * rows + row]]
            * global_scale;
        const unsigned char* codes = panel_base + rows;
        unsigned int mask = panel_masks[panel];

        while (mask) {
            const int column = __ffs(mask) - 1;
            mask &= mask - 1;
            const unsigned char byte =
                codes[(size_t)column * rowhalf + half_byte];
            const float weight =
                e2[
                    high_nibble
                    ? (byte >> 4)
                    : (byte & 15)
                ] * scale;
            acc = fmaf(
                weight,
                act[(panel << 4) + column],
                acc
            );
        }
    }
    partials[(size_t)chunk * rows + row] = acc;
}
"""


class StaticDownKernels:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(
            code=CUDA_SOURCE, options=("-std=c++14",)
        )
        self.preload = self.mod.get_function(
            "preload_down_records"
        )
        self.gather_miss = self.mod.get_function(
            "gather_down_cols_static_miss"
        )
        self.down_hybrid = self.mod.get_function(
            "gemv_down_static_or_mirror"
        )
