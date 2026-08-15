"""V5 (PRO_V5_PREREGISTRATION.md): batched variants of panel_scan and
reduce_partials, the two down_proj sub-kernels with FIXED (data-independent)
grid sizes -- unlike gather_down_sparse_ind/gemv_down_masked_partial_ind,
whose grid size depends on ReLU2 activation sparsity and are therefore out
of scope here (a harder, separate problem).

Each batched kernel is a mechanical transformation of the original: same
per-block logic, addressed by an added slot dimension (blockIdx.x or
blockIdx.y) instead of being launched once per slot. This file does not
modify runtime.py or fused_nvfp4.py -- it is a standalone kernel module for
isolated verification before any integration is attempted.

Original sources: src/moe_lab/lightningstream_nemotron/fused_nvfp4.py,
panel_scan (line 157) and reduce_partials (line 408).
"""

from __future__ import annotations

import numpy as np

CUDA_SOURCE = r"""
extern "C" __global__ void panel_scan_ref(
    const float* __restrict__ act,
    const int inter,
    unsigned int* __restrict__ panel_masks,
    int* __restrict__ panel_list,
    int* __restrict__ panel_count,
    int* __restrict__ nz_list,
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

// Batched: one block per slot (blockIdx.x = s), identical per-block body to
// panel_scan_ref, addressed into [top_k, ...] arrays via a slot offset.
extern "C" __global__ void panel_scan_batched(
    const float* __restrict__ act,           // [top_k, inter]
    const int inter,
    unsigned int* __restrict__ panel_masks,  // [top_k, inter/16]
    int* __restrict__ panel_list,            // [top_k, inter/16]
    int* __restrict__ panel_count,           // [top_k]
    int* __restrict__ nz_list,               // [top_k, inter]
    int* __restrict__ nz_count)              // [top_k]
{
    const int s = blockIdx.x;
    const int npanel = inter >> 4;
    const float* __restrict__ act_s = act + (size_t)s * inter;
    unsigned int* __restrict__ masks_s = panel_masks + (size_t)s * npanel;
    int* __restrict__ list_s = panel_list + (size_t)s * npanel;
    int* __restrict__ nz_s = nz_list + (size_t)s * inter;

    if (threadIdx.x < npanel) masks_s[threadIdx.x] = 0u;
    if (threadIdx.x == 0) { panel_count[s] = 0; nz_count[s] = 0; }
    __syncthreads();
    for (int j = threadIdx.x; j < inter; j += blockDim.x) {
        if (act_s[j] != 0.0f) atomicOr(&masks_s[j >> 4], 1u << (j & 15));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0, m = 0;
        for (int p = 0; p < npanel; p++) {
            const unsigned int mk = masks_s[p];
            if (mk) {
                list_s[n++] = p;
                for (int c = 0; c < 16; c++)
                    if (mk & (1u << c)) nz_s[m++] = (p << 4) + c;
            }
        }
        panel_count[s] = n;
        nz_count[s] = m;
    }
}

extern "C" __global__ void reduce_partials_ref(
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

// Batched: blockIdx.y = slot s, identical per-thread body to reduce_partials_ref.
extern "C" __global__ void reduce_partials_batched(
    const float* __restrict__ partials,  // [top_k, nchunks, rows]
    float*       __restrict__ out,       // [top_k, rows]
    const int    rows,
    const int    nchunks)
{
    const int s = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* __restrict__ partials_s = partials + (size_t)s * nchunks * rows;
    float a = 0.0f;
    for (int c = 0; c < nchunks; c++) a += partials_s[(size_t)c * rows + row];
    out[(size_t)s * rows + row] = a;
}

extern "C" __global__ void weighted_accumulate_ind_ref(
    float*       __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ w_ptr,
    const int    n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = fmaf(src[i], *w_ptr, dst[i]);
}

// Batched: ONE launch replaces top_k sequential weighted_accumulate_ind
// calls. Each thread walks s = 0..top_k-1 in the SAME fixed order the
// sequential calls used, accumulating into a local register before a single
// write-back -- bit-identical FP op sequence to top_k separate launches
// each doing dst[i] = fmaf(src[i], w, dst[i]), NOT a parallel/atomic
// reduction (which would change the summation order -- the exact class of
// bug D1 already found once in this project).
extern "C" __global__ void weighted_accumulate_ind_batched(
    float*       __restrict__ dst,        // [rows], already holds the shared-expert term
    const float* __restrict__ contrib,    // [top_k, rows]
    const float* __restrict__ w,          // [top_k]
    const int    rows,
    const int    top_k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float acc = dst[i];
    for (int s = 0; s < top_k; s++) {
        acc = fmaf(contrib[(size_t)s * rows + i], w[s], acc);
    }
    dst[i] = acc;
}
"""


class DownProjBatchKernels:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14",))
        self.panel_scan_ref = self.mod.get_function("panel_scan_ref")
        self.panel_scan_batched = self.mod.get_function("panel_scan_batched")
        self.reduce_partials_ref = self.mod.get_function("reduce_partials_ref")
        self.reduce_partials_batched = self.mod.get_function("reduce_partials_batched")
        self.weighted_accumulate_ind_ref = self.mod.get_function("weighted_accumulate_ind_ref")
        self.weighted_accumulate_ind_batched = self.mod.get_function("weighted_accumulate_ind_batched")

    def run_accumulate_ref(self, dst, src, w_scalar_ptr, n: int):
        self.weighted_accumulate_ind_ref((( n + 255) // 256,), (256,), (dst, src, w_scalar_ptr, np.int32(n)))

    def run_accumulate_batched(self, dst, contrib_batched, w, rows: int, top_k: int):
        self.weighted_accumulate_ind_batched(
            ((rows + 255) // 256,), (256,),
            (dst, contrib_batched, w, np.int32(rows), np.int32(top_k)),
        )

    def run_panel_scan_ref(self, act, inter: int):
        cp = self.cp
        npanel = inter >> 4
        panel_masks = cp.zeros(npanel, dtype=cp.uint32)
        panel_list = cp.zeros(npanel, dtype=cp.int32)
        panel_count = cp.zeros(1, dtype=cp.int32)
        nz_list = cp.zeros(inter, dtype=cp.int32)
        nz_count = cp.zeros(1, dtype=cp.int32)
        self.panel_scan_ref(
            (1,), (256,),
            (act, np.int32(inter), panel_masks, panel_list, panel_count, nz_list, nz_count),
        )
        return panel_masks, panel_list, panel_count, nz_list, nz_count

    def run_panel_scan_batched(self, act_batched, inter: int, top_k: int):
        cp = self.cp
        npanel = inter >> 4
        panel_masks = cp.zeros(top_k * npanel, dtype=cp.uint32)
        panel_list = cp.zeros(top_k * npanel, dtype=cp.int32)
        panel_count = cp.zeros(top_k, dtype=cp.int32)
        nz_list = cp.zeros(top_k * inter, dtype=cp.int32)
        nz_count = cp.zeros(top_k, dtype=cp.int32)
        self.panel_scan_batched(
            (top_k,), (256,),
            (act_batched, np.int32(inter), panel_masks, panel_list, panel_count, nz_list, nz_count),
        )
        return panel_masks, panel_list, panel_count, nz_list, nz_count

    def run_reduce_partials_ref(self, partials, rows: int, nchunks: int):
        cp = self.cp
        out = cp.zeros(rows, dtype=cp.float32)
        blocks = (rows + 255) // 256
        self.reduce_partials_ref((blocks,), (256,), (partials, out, np.int32(rows), np.int32(nchunks)))
        return out

    def run_reduce_partials_batched(self, partials_batched, rows: int, nchunks: int, top_k: int):
        cp = self.cp
        out = cp.zeros(top_k * rows, dtype=cp.float32)
        blocks_x = (rows + 255) // 256
        self.reduce_partials_batched(
            (blocks_x, top_k), (256,),
            (partials_batched, out, np.int32(rows), np.int32(nchunks)),
        )
        return out
