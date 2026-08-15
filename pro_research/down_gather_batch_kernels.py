"""Batched variants of gather_down_sparse_ind and gemv_down_masked_partial_ind
-- the two down_proj sub-kernels V5 deliberately left per-slot, since
gather_down_sparse_ind reads from host-mapped memory (PCIe-bound) and both
were assumed harder to batch. Re-examination: BOTH have fixed (data-
independent) grid formulas -- gather's blocks = ((inter+npanel)*32+255)//256
depends only on inter/npanel, not actual sparsity; down_masked's grid =
((hidden+127)//128, nchunks) is also fixed. So the same mechanical,
independent-per-slot batching class as panel_scan/reduce_partials/up-proj
GEMV applies here too.

Reference kernels copied verbatim next to the batched ones (only slot-
indexed addressing added via blockIdx.y for gather, blockIdx.z for
down_masked since blockIdx.y is already the chunk dimension there) --
same transcription-avoidance discipline as up_proj_batch_kernels.py.

Batching these requires top_k independent mirror buffers instead of one
reused sequentially (the original design's whole point was avoiding that
allocation) -- the caller must budget top_k x ~2.68 MB per layer.
"""

from __future__ import annotations

import numpy as np

CUDA_SOURCE = r"""
extern "C" __global__ void gather_down_sparse_ind_ref(
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

// Batched: blockIdx.y = slot s. Per-thread body identical to the ref kernel;
// only id/panel-metadata/dst addressing gains a slot offset.
extern "C" __global__ void gather_down_sparse_ind_batched(
    const unsigned char* __restrict__ down_base,
    const int*           __restrict__ ids,           // [top_k]
    const size_t         panel_bytes,
    unsigned char*       __restrict__ dst_base,       // [top_k, mirror_bytes]
    const int*           __restrict__ panel_list,     // [top_k, npanel]
    const int*           __restrict__ panel_count,    // [top_k]
    const int*           __restrict__ nz_list,        // [top_k, inter]
    const int*           __restrict__ nz_count,       // [top_k]
    const int rows,
    const int npanel,
    const int inter,
    const size_t mirror_bytes)
{
    const int s = blockIdx.y;
    const unsigned char* __restrict__ src_base = down_base + (size_t)ids[s] * panel_bytes;
    unsigned char* __restrict__ dst_s = dst_base + (size_t)s * mirror_bytes;
    const int* __restrict__ panel_list_s = panel_list + (size_t)s * npanel;
    const int* __restrict__ nz_list_s = nz_list + (size_t)s * inter;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int ncol = nz_count[s];
    const int pcount = panel_count[s];
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    if (warp < ncol) {
        const int j = nz_list_s[warp];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        const uchar4* srcp = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* dstp = reinterpret_cast<uchar4*>(dst_s + off);
        for (int k = lane; k < rowhalf / 4; k += 32) dstp[k] = srcp[k];
    } else if (warp < ncol + pcount) {
        const int p = panel_list_s[warp - ncol];
        const size_t off = (size_t)p * panel_stride;
        const uchar4* srcp = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* dstp = reinterpret_cast<uchar4*>(dst_s + off);
        for (int k = lane; k < rows / 4; k += 32) dstp[k] = srcp[k];
    }
}

extern "C" __global__ void gemv_down_masked_partial_ind_ref(
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

// Batched: blockIdx.z = slot s (blockIdx.y stays the existing chunk
// dimension, unchanged). Per-thread body identical to the ref kernel.
extern "C" __global__ void gemv_down_masked_partial_ind_batched(
    const unsigned char* __restrict__ bank,          // [top_k, mirror_bytes]
    const int*           __restrict__ ids,            // [top_k]
    const float*         __restrict__ globals,
    const float*         __restrict__ act,             // [top_k, inter]
    const int*           __restrict__ panel_list,       // [top_k, npanel]
    const unsigned int*  __restrict__ panel_masks,      // [top_k, npanel]
    const int*           __restrict__ panel_count,      // [top_k]
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    float*               __restrict__ partials,          // [top_k, nchunks, rows]
    const int rows,
    const int inter,
    const int npanel,
    const size_t mirror_bytes)
{
    const int s = blockIdx.z;
    const float global_scale = globals[ids[s] * 2 + 0];
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
    const int pcount = panel_count[s];
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    const unsigned char* __restrict__ bank_s = bank + (size_t)s * mirror_bytes;
    const int* __restrict__ panel_list_s = panel_list + (size_t)s * npanel;
    const unsigned int* __restrict__ panel_masks_s = panel_masks + (size_t)s * npanel;
    const float* __restrict__ act_s = act + (size_t)s * inter;

    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list_s[pi];
        const unsigned char* __restrict__ pbase = bank_s + (size_t)p * panel_stride;
        const float sc = s_e4m3[pbase[row]] * global_scale;
        const unsigned char* __restrict__ pcodes = pbase + rows;
        unsigned int m = panel_masks_s[p];
        while (m) {
            const int c = __ffs(m) - 1;
            m &= m - 1;
            const unsigned char byte = pcodes[(size_t)c * rowhalf + hb];
            const float w = s_e2m1[hi ? (byte >> 4) : (byte & 15)] * sc;
            acc = fmaf(w, act_s[(p << 4) + c], acc);
        }
    }
    partials[((size_t)s * nchunks + chunk) * rows + row] = acc;
}
"""


class DownGatherBatchKernels:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14",))
        self.gather_ref = self.mod.get_function("gather_down_sparse_ind_ref")
        self.gather_batched = self.mod.get_function("gather_down_sparse_ind_batched")
        self.down_masked_ref = self.mod.get_function("gemv_down_masked_partial_ind_ref")
        self.down_masked_batched = self.mod.get_function("gemv_down_masked_partial_ind_batched")

    def run_gather_ref(self, down_base, id_ptr, panel_bytes, dst_base, panel_list, panel_count, nz_list, nz_count, rows, blocks):
        self.gather_ref((blocks,), (256,),
                        (down_base, id_ptr, np.uint64(panel_bytes), dst_base,
                         panel_list, panel_count, nz_list, nz_count, np.int32(rows)))

    def run_gather_batched(self, down_base, ids, panel_bytes, dst_base, panel_list, panel_count,
                           nz_list, nz_count, rows, npanel, inter, mirror_bytes, top_k, blocks):
        self.gather_batched(
            (blocks, top_k), (256,),
            (down_base, ids, np.uint64(panel_bytes), dst_base, panel_list, panel_count,
             nz_list, nz_count, np.int32(rows), np.int32(npanel), np.int32(inter), np.uint64(mirror_bytes)),
        )

    def run_down_masked_ref(self, bank, id_ptr, globals_dev, act, panel_list, panel_masks,
                            panel_count, e2m1, e4m3, partials, rows, inter, nchunks):
        grid = ((rows + 127) // 128, nchunks)
        self.down_masked_ref(grid, (128,),
                             (bank, id_ptr, globals_dev, act, panel_list, panel_masks,
                              panel_count, e2m1, e4m3, partials, np.int32(rows), np.int32(inter)))

    def run_down_masked_batched(self, bank, ids, globals_dev, act, panel_list, panel_masks,
                                panel_count, e2m1, e4m3, partials, rows, inter, npanel,
                                mirror_bytes, top_k, nchunks):
        grid = ((rows + 127) // 128, nchunks, top_k)
        self.down_masked_batched(grid, (128,),
                                 (bank, ids, globals_dev, act, panel_list, panel_masks,
                                  panel_count, e2m1, e4m3, partials, np.int32(rows),
                                  np.int32(inter), np.int32(npanel), np.uint64(mirror_bytes)))
