"""H-SCALE kernels: keep the down_proj FP8 block-scale planes resident in VRAM.

Grounded in a measurement, not a guess. `diag_gather_pcie_ceiling.json`:

  * per gather call the production path moves 164.7 nonzero weight columns
    x 1344 B = 221.3 KB, PLUS 90.1 active panels x 2688 B = 242.2 KB of block
    scales -- **52.2% of all down-path PCIe traffic is scale metadata**;
  * hypothesis arm v3 (skip the scale planes, verified strict subset of the
    production byte set) costs 2.475 ms/token instead of 3.855 ms:
    **-1.380 ms/token measured**;
  * the same run showed the gather already runs at 64.0% of the byte-verified
    25.908 GB/s link ceiling and that every concurrency variant (unroll x4,
    2/4/8/16 warps per column) is SLOWER -- so the byte count is the only
    remaining lever, and half of it is scales.

The panel-major host record interleaves them: panel p is [2688 scale bytes]
[16 columns x 1344 code bytes], stride 24192 B. Because a panel's 2688 scale
bytes are indexed by OUTPUT ROW they are all needed whenever the panel is
active, and at 9% ReLU^2 sparsity only ~1.8 of the panel's 16 columns are.
Hence the scales cost more PCIe than the weights they scale.

## What changes and what provably does not

Only where the scale byte is READ FROM. Same expert, same panel, same row,
same byte, same `e4m3_lut[byte] * global_scale`, same fmaf order. The output is
bit-identical by construction -- this is a data-placement change of exactly the
kind `enable_cache(mode="up_only")` and the panel-major repack already are, not
an arithmetic one. Nothing here touches routing, precision, masks or ordering.

## Three kernels

cache_fetch_scale_plane
    Runs alongside the existing `cache_fetch` on the copy stream, same
    need[]/slots[]/ids[] contract, same SM-side wide-uint4 staging pattern (the
    M1 pattern, not the DMA engine, so it stays graph-capturable). Gathers the
    116 strided 2688-byte scale blocks of a MISSED expert into a contiguous
    per-slot plane. Costs 311,808 B per miss; at the measured ~20.24
    misses/token that is 6.3 MB/token, ~0.24 ms -- the price of the -1.380 ms.

gather_down_cols_ind
    The production `gather_down_sparse_ind` with the panel-scale branch
    removed. Byte-for-byte a strict subset of what production copies (asserted
    in diag_gather_pcie_ceiling.py's v3 arm).

gemv_down_masked_partial_ind_sres
    The production masked GEMV with one line changed: the scale byte comes from
    `planes + slot*PLANE_BYTES + p*rows + row` instead of
    `mirror + p*panel_stride + row`. The production kernel is kept verbatim in
    this file directly above it so the diff is a one-line read, deliberately --
    the same discipline the V6 up-proj batching used to avoid transcription
    errors.

VRAM: cap x 311,808 B per layer. At the V6 budget of 1656 slots over 23 layers
that is 516.4 MB = 492.5 MiB, against 639 MiB measured free during the V12 full
run. That fit is a gate, not an assumption -- the runner checks it.
"""

from __future__ import annotations

import numpy as np

HIDDEN = 2688
INTER = 1856
NPANEL = INTER // 16                       # 116
ROWHALF = HIDDEN // 2                      # 1344
PANEL_STRIDE = HIDDEN + 16 * ROWHALF       # 24192
DOWN_PANEL_BYTES = NPANEL * PANEL_STRIDE   # 2806272
PLANE_BYTES = NPANEL * HIDDEN              # 311808

CUDA_SOURCE = r"""
// ---------------------------------------------------------------------------
// 1. Stage a missed expert's scale plane into its cache slot.
//    Same contract as fused_nvfp4.py's cache_fetch: one blockIdx.x per route
//    slot, blocks whose slot was a cache hit return immediately, gridDim.y
//    blocks stride over the payload, wide uint4 loads from mapped host.
//    Source is strided (panel p at p*panel_stride), destination is contiguous
//    (p*rows), so the GEMV can index it as a flat [npanel, rows] plane.
// ---------------------------------------------------------------------------
extern "C" __global__ void cache_fetch_scale_plane(
    const unsigned char* __restrict__ down_base,
    unsigned char*       __restrict__ planes,
    const int* __restrict__ ids,
    const int* __restrict__ slots,
    const int* __restrict__ need,
    const size_t panel_bytes,      // 2806272, per-expert record stride
    const size_t plane_bytes,      // 311808, per-slot plane stride
    const int rows,                // 2688
    const int npanel)              // 116
{
    const int s = blockIdx.x;
    if (!need[s]) return;
    const unsigned char* __restrict__ rec = down_base + (size_t)ids[s] * panel_bytes;
    unsigned char* __restrict__ dst = planes + (size_t)slots[s] * plane_bytes;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)(rows >> 1);
    const size_t per_panel4 = (size_t)rows >> 4;          // 168 uint4 per panel
    const size_t total4 = per_panel4 * (size_t)npanel;    // 19488
    const size_t stride = (size_t)gridDim.y * blockDim.x;
    for (size_t v = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
         v < total4; v += stride) {
        const size_t p = v / per_panel4;
        const size_t j = v - p * per_panel4;
        const uint4* src = reinterpret_cast<const uint4*>(rec + p * panel_stride);
        uint4* d = reinterpret_cast<uint4*>(dst + p * (size_t)rows);
        d[j] = src[j];
    }
}

// ---------------------------------------------------------------------------
// 2. gather_down_sparse_ind with the panel-scale branch removed.
//    The `else if (warp < ncol + *panel_count)` arm of the production kernel is
//    exactly what H-SCALE makes unnecessary; everything else is verbatim.
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_down_cols_ind(
    const unsigned char* __restrict__ down_base,
    const int*           __restrict__ id_ptr,
    const size_t         panel_bytes,
    unsigned char*       __restrict__ dst_base,
    const int*           __restrict__ nz_list,
    const int*           __restrict__ nz_count,
    const int rows)
{
    const unsigned char* __restrict__ src_base =
        down_base + (size_t)(*id_ptr) * panel_bytes;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;
    if (warp < *nz_count) {
        const int j = nz_list[warp];
        const size_t off = (size_t)(j >> 4) * panel_stride + rows
                         + (size_t)(j & 15) * rowhalf;
        const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
        uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
        for (int k = lane; k < rowhalf / 4; k += 32) d[k] = s[k];
    }
}

// ---------------------------------------------------------------------------
// 3. The masked down-GEMV.
//
//    REFERENCE (verbatim copy of fused_nvfp4.py gemv_down_masked_partial_ind,
//    kept here so the one-line difference below is readable as a diff and can
//    never drift silently):
//
//        const unsigned char* pbase = bank + (size_t)p * panel_stride;
//        const float s = s_e4m3[pbase[row]] * global_scale;
//        const unsigned char* pcodes = pbase + rows;
//
//    H-SCALE changes only the middle line: the scale byte is read from the
//    resident plane instead of from the freshly gathered mirror. Same byte,
//    same lut, same multiply, same accumulation order.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_down_masked_partial_ind_sres(
    const unsigned char* __restrict__ bank,      // mirror: columns only now
    const unsigned char* __restrict__ planes,    // [cap, npanel, rows]
    const int*           __restrict__ slot_ptr,  // &dev["slots"][s]
    const int*           __restrict__ id_ptr,
    const float*         __restrict__ globals,
    const float*         __restrict__ act,
    const int*           __restrict__ panel_list,
    const unsigned int*  __restrict__ panel_masks,
    const int*           __restrict__ panel_count,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    float*               __restrict__ partials,
    const size_t         plane_bytes,
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
    const unsigned char* __restrict__ plane =
        planes + (size_t)(*slot_ptr) * plane_bytes;

    float acc = 0.0f;
    for (int pi = chunk; pi < pcount; pi += nchunks) {
        const int p = panel_list[pi];
        const unsigned char* __restrict__ pbase = bank + (size_t)p * panel_stride;
        const float s = s_e4m3[plane[(size_t)p * rows + row]] * global_scale;
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
"""


class ScaleResidentKernels:
    """Compiled H-SCALE kernels plus the per-layer resident plane buffers."""

    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14",))
        self.fetch_plane_k = self.mod.get_function("cache_fetch_scale_plane")
        self.gather_cols_k = self.mod.get_function("gather_down_cols_ind")
        self.down_masked_sres_k = self.mod.get_function("gemv_down_masked_partial_ind_sres")
        self.planes: dict[int, object] = {}

    def plane_bytes_for(self, caps: dict[int, int]) -> int:
        return sum(int(c) for c in caps.values()) * PLANE_BYTES

    def alloc_planes(self, layer: int, cap: int):
        p = self.cp.zeros(cap * PLANE_BYTES, dtype=self.cp.uint8)
        self.planes[layer] = p
        return p

    def fetch_planes(self, down_base_ptr: int, planes, dev, top_k: int) -> None:
        self.fetch_plane_k(
            (top_k, 64), (256,),
            (np.uint64(down_base_ptr), planes, dev["ids"], dev["slots"],
             dev["need"], np.uint64(DOWN_PANEL_BYTES), np.uint64(PLANE_BYTES),
             np.int32(HIDDEN), np.int32(NPANEL)))

    def gather_cols(self, blocks: int, down_base_ptr: int, id_slice, mirror,
                    nz_s, nzc_s, hidden: int) -> None:
        self.gather_cols_k(
            (blocks,), (256,),
            (np.uint64(down_base_ptr), id_slice, np.uint64(DOWN_PANEL_BYTES),
             mirror, nz_s, nzc_s, np.int32(hidden)))

    def down_masked_sres(self, grid, mirror, planes, slot_slice, id_slice,
                         globals_dev, act_s, plist_s, masks_s, pcount_s,
                         e2m1, e4m3, partials_s, hidden: int, inter: int) -> None:
        self.down_masked_sres_k(
            grid, (128,),
            (mirror, planes, slot_slice, id_slice, globals_dev, act_s,
             plist_s, masks_s, pcount_s, e2m1, e4m3, partials_s,
             np.uint64(PLANE_BYTES), np.int32(hidden), np.int32(inter)))
