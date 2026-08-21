"""Bulk group-major NVFP4 M1 kernel for independent Ornith expert assignments."""
from __future__ import annotations

import numpy as np


NVFP4_BULK_SOURCE = r"""
extern "C" __global__ void nvfp4_bulk_m1_warp32_direct_l2(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ global_scales,
    const int rows,
    const int cols)
{
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = e2m1[threadIdx.x];
    __syncthreads();

    const int group = (int)blockIdx.y;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int row = (int)blockIdx.x * 8 + warp;
    if (row >= rows) return;

    const int nbytes = cols >> 1;
    const int nvec = nbytes >> 2;
    const size_t code_group_stride = (size_t)rows * nbytes;
    const size_t scale_group_stride = (size_t)rows * (cols >> 4);
    const unsigned char* crow = codes + (size_t)group * code_group_stride
        + (size_t)row * nbytes;
    const unsigned char* srow = scales + (size_t)group * scale_group_stride
        + (size_t)row * (cols >> 4);
    const uchar4* c4 = reinterpret_cast<const uchar4*>(crow);
    const float* xg = x + (size_t)group * cols;
    const float global_scale = global_scales[group];

    float part[8];
    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) part[wi] = 0.0f;

    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
        const int tid = lane + 32 * wi;
        float acc = 0.0f;
        for (int v = tid; v < nvec; v += 256) {
            const uchar4 q = c4[v];
            const int b = v << 2;
            const int k = b << 1;
            const float sc = e4m3[srow[b >> 3]] * global_scale;
            const float w[8] = {
                lut[q.x & 15] * sc, lut[q.x >> 4] * sc,
                lut[q.y & 15] * sc, lut[q.y >> 4] * sc,
                lut[q.z & 15] * sc, lut[q.z >> 4] * sc,
                lut[q.w & 15] * sc, lut[q.w >> 4] * sc
            };
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(w[j], xg[k + j], acc);
        }
        part[wi] = acc;
    }

    float s8[8];
    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
        float value = part[wi];
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        s8[wi] = value;
    }
    if (lane == 0) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        out[(size_t)group * rows + row] = (t0 + t2) + (t1 + t3);
    }
}

extern "C" __global__ void swiglu_bulk_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int n)
{
    const int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
"""


class OrnithNVFP4BulkM1:
    def __init__(self):
        import cupy as cp

        names = ("nvfp4_bulk_m1_warp32_direct_l2", "swiglu_bulk_f32")
        self.module = cp.RawModule(
            code=NVFP4_BULK_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.gemv = self.module.get_function(names[0])
        self.swiglu_kernel = self.module.get_function(names[1])

    def nvfp4(
        self,
        codes,
        scales,
        e2,
        e4,
        x,
        out,
        global_scales,
        groups: int,
        rows: int,
        cols: int,
    ) -> None:
        groups, rows, cols = int(groups), int(rows), int(cols)
        if tuple(x.shape) != (groups, cols):
            raise ValueError(f"x shape mismatch: {x.shape} != {(groups, cols)}")
        if tuple(out.shape) != (groups, rows):
            raise ValueError(f"out shape mismatch: {out.shape} != {(groups, rows)}")
        if int(global_scales.size) != groups:
            raise ValueError(f"global scale mismatch: {global_scales.size} != {groups}")
        self.gemv(
            ((rows + 7) // 8, groups),
            (256,),
            (
                codes,
                scales,
                e2,
                e4,
                x,
                out,
                global_scales,
                np.int32(rows),
                np.int32(cols),
            ),
        )

    def swiglu(self, gate, up, out, groups: int, width: int = 512) -> None:
        n = int(groups) * int(width)
        self.swiglu_kernel(((n + 255) // 256,), (256,), (gate, up, out, np.int32(n)))

    def nvfp4_ptr(
        self,
        codes_ptr: int,
        scales_ptr: int,
        e2,
        e4,
        x,
        out,
        global_scales,
        groups: int,
        rows: int,
        cols: int,
    ) -> None:
        """Launch against contiguous group-major UVA pointers in pinned RAM."""
        groups, rows, cols = int(groups), int(rows), int(cols)
        if tuple(x.shape) != (groups, cols):
            raise ValueError(f"x shape mismatch: {x.shape} != {(groups, cols)}")
        if tuple(out.shape) != (groups, rows):
            raise ValueError(f"out shape mismatch: {out.shape} != {(groups, rows)}")
        self.gemv(
            ((rows + 7) // 8, groups),
            (256,),
            (
                np.uint64(codes_ptr),
                np.uint64(scales_ptr),
                e2,
                e4,
                x,
                out,
                global_scales,
                np.int32(rows),
                np.int32(cols),
            ),
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        result = {}
        for name, fn in (("nvfp4_bulk_m1_warp32_direct_l2", self.gemv),
                         ("swiglu_bulk_f32", self.swiglu_kernel)):
            fn.compile()
            attrs = getattr(fn, "attributes", {}) or {}
            result[name] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result
