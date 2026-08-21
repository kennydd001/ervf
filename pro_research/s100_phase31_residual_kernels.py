from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
#define H4 4
#define TOPK 6

// Exact replacement for:
//   copy(shared_out -> out)
//   accumulate_h4(out, route_down, route_w)
//   add_(residual[t], out[t]) for t=0..3
//
// The slot loop and explicit fmaf are identical to accumulate_h4.  The final
// residual addition has the same single FP32 addition as add_.  reduce_routes
// remains a separate producer, so its work can still overlap the shared branch.
extern "C" __global__ void residual_sink_h4(
    const float* __restrict__ route_down,
    const float* __restrict__ shared_out,
    const float* __restrict__ route_w,
    float* __restrict__ residual,
    const int hidden)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (row >= hidden || token >= H4) return;

    float routed_shared = shared_out[(size_t)token * hidden + row];
    #pragma unroll
    for (int slot = 0; slot < TOPK; ++slot) {
        const int route = token * TOPK + slot;
        routed_shared = fmaf(
            route_down[(size_t)route * hidden + row],
            route_w[route],
            routed_shared);
    }
    const size_t index = (size_t)token * hidden + row;
    residual[index] = residual[index] + routed_shared;
}

// More aggressive exact replacement for reduce_routes + residual_sink_h4.
// Chunk reduction is kept in c=0..nchunks-1 order and route accumulation in
// slot=0..5 order.  This arm removes one additional launch and route_down
// round-trip, but it intentionally waits for shared_out before reducing; the
// matched sink arm determines whether that lost overlap matters.
extern "C" __global__ void reduce_residual_sink_h4(
    const float* __restrict__ partials,
    const float* __restrict__ shared_out,
    const float* __restrict__ route_w,
    float* __restrict__ residual,
    const int hidden,
    const int nchunks)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (row >= hidden || token >= H4) return;

    float routed_shared = shared_out[(size_t)token * hidden + row];
    #pragma unroll
    for (int slot = 0; slot < TOPK; ++slot) {
        const int route = token * TOPK + slot;
        const float* route_partials =
            partials + (size_t)route * nchunks * hidden;
        float down = 0.0f;
        for (int chunk = 0; chunk < nchunks; ++chunk)
            down += route_partials[(size_t)chunk * hidden + row];
        routed_shared = fmaf(down, route_w[route], routed_shared);
    }
    const size_t index = (size_t)token * hidden + row;
    residual[index] = residual[index] + routed_shared;
}
"""


class Phase31ResidualKernels:
    def __init__(self):
        import cupy as cp

        names = ("residual_sink_h4", "reduce_residual_sink_h4")
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.f = {name: self.mod.get_function(name) for name in names}

    @staticmethod
    def _grid(hidden: int):
        return ((int(hidden) + 255) // 256, 4), (256,)

    def sink(self, route_down, shared_out, route_w, residual, hidden: int) -> None:
        grid, block = self._grid(hidden)
        self.f["residual_sink_h4"](
            grid,
            block,
            (route_down, shared_out, route_w, residual, np.int32(hidden)),
        )

    def reduce_sink(
        self,
        partials,
        shared_out,
        route_w,
        residual,
        hidden: int,
        nchunks: int,
    ) -> None:
        grid, block = self._grid(hidden)
        self.f["reduce_residual_sink_h4"](
            grid,
            block,
            (
                partials,
                shared_out,
                route_w,
                residual,
                np.int32(hidden),
                np.int32(nchunks),
            ),
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        result = {}
        for name, fn in self.f.items():
            fn.compile()
            attrs = getattr(fn, "attributes", {}) or {}
            result[name] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result
