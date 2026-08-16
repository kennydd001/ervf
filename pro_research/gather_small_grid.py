"""A grid-stride version of gather_down_sparse_ind, so the overlap has SM room.

V14-G showed B3 works (bit-exact, -0.416 ms in-graph) but hides only 16.8% of
the 2.47 ms of PCIe time, because `gather_down_sparse_ind` is an SM-side
zero-copy kernel and therefore competes with `down_masked_ind_k` for the very
SMs it is supposed to overlap with.

The production launch makes that worse than it needs to be. Its grid is sized
for the worst case -- every one of the 1856 columns nonzero plus all 116 panels:

    max_warps = inter + npanel = 1972
    blocks    = (1972 * 32 + 255) / 256 = 247

At the measured sparsity only **164.7 columns + 90.1 panels ~ 255 warps** have
work, so ~87% of the launched warps take the early-exit branch and retire. They
still occupy scheduling slots on the way through, and the grid still claims SMs
that `down_masked` could be using.

This version keeps the identical per-warp body but walks the work list with a
grid-stride loop, so the grid can be small and static (capture needs static) and
still cover any sparsity. Same bytes, same source and destination offsets, same
order within a warp -- the copied result is identical by construction, and the
runner gates on it.

The right grid size is an empirical trade-off: too small and the gather itself
slows down (it is PCIe-latency-bound and needs enough requests in flight); too
large and it starves the compute it overlaps with. Hence `make_gather` takes the
block count and the runner sweeps it rather than guessing.
"""

from __future__ import annotations

import numpy as np

CUDA_SOURCE = r"""
// Identical per-warp body to gather_down_sparse_ind; only the work assignment
// changes, from "one warp per item, grid sized for the worst case" to
// "grid-stride over the work list, grid sized for the machine".
extern "C" __global__ void gather_down_sparse_ind_stride(
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
    const int lane = threadIdx.x & 31;
    const int warp0 = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    const int ncol = *nz_count;
    const int total = ncol + *panel_count;
    const int rowhalf = rows >> 1;
    const size_t panel_stride = (size_t)rows + 16u * (size_t)rowhalf;

    for (int w = warp0; w < total; w += nwarps) {
        if (w < ncol) {
            const int j = nz_list[w];
            const size_t off = (size_t)(j >> 4) * panel_stride + rows
                             + (size_t)(j & 15) * rowhalf;
            const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
            uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
            for (int k = lane; k < rowhalf / 4; k += 32) d[k] = s[k];
        } else {
            const int p = panel_list[w - ncol];
            const size_t off = (size_t)p * panel_stride;
            const uchar4* s = reinterpret_cast<const uchar4*>(src_base + off);
            uchar4* d = reinterpret_cast<uchar4*>(dst_base + off);
            for (int k = lane; k < rows / 4; k += 32) d[k] = s[k];
        }
    }
}
"""


class GatherSmallGrid:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++14",))
        self.k = self.mod.get_function("gather_down_sparse_ind_stride")

    def launch(self, blocks: int, down_base_ptr: int, id_slice, panel_bytes: int,
               mirror, plist_s, pcount_s, nz_s, nzc_s, rows: int) -> None:
        self.k((blocks,), (256,),
               (np.uint64(down_base_ptr), id_slice, np.uint64(panel_bytes),
                mirror, plist_s, pcount_s, nz_s, nzc_s, np.int32(rows)))
