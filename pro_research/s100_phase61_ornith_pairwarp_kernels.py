"""Two-warp-per-row indirect NVFP4 exact-M kernels for Ornith."""
from __future__ import annotations

import numpy as np


_TEMPLATE = r"""
extern "C" __global__ void nvfp4_route_m__BATCH___pairwarp_indirect_l2(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ global_scales,
    const int* __restrict__ slots,
    const int* __restrict__ input_ids,
    const int rows,
    const int cols)
{
    __shared__ float lut[16];
    __shared__ float partial[4 * __BATCH__ * 8];
    if (threadIdx.x < 16) lut[threadIdx.x] = e2m1[threadIdx.x];
    __syncthreads();

    const int group = (int)blockIdx.y;
    const int slot = slots[group];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int pair = warp & 1;
    const int row_local = warp >> 1;
    const int row = (int)blockIdx.x * 4 + row_local;
    const bool active = row < rows;

    const int nbytes = cols >> 1;
    const int nvec = nbytes >> 2;
    const size_t code_slot_stride = (size_t)rows * nbytes;
    const size_t scale_slot_stride = (size_t)rows * (cols >> 4);
    const unsigned char* crow = codes + (size_t)slot * code_slot_stride
        + (size_t)row * nbytes;
    const unsigned char* srow = scales + (size_t)slot * scale_slot_stride
        + (size_t)row * (cols >> 4);
    const uchar4* c4 = reinterpret_cast<const uchar4*>(crow);
    const float global_scale = global_scales[slot];

    float sums[__BATCH__][4];
    #pragma unroll
    for (int m = 0; m < __BATCH__; ++m)
        for (int wi = 0; wi < 4; ++wi) sums[m][wi] = 0.0f;

    if (active) {
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const int tid = lane + 32 * wi + 128 * pair;
            float acc[__BATCH__];
            #pragma unroll
            for (int m = 0; m < __BATCH__; ++m) acc[m] = 0.0f;
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
                for (int m = 0; m < __BATCH__; ++m) {
                    const float* xm = x + (size_t)input_ids[group * __BATCH__ + m] * cols;
                    float value = acc[m];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) value = fmaf(w[j], xm[k + j], value);
                    acc[m] = value;
                }
            }
            #pragma unroll
            for (int m = 0; m < __BATCH__; ++m) sums[m][wi] = acc[m];
        }
    }

    #pragma unroll
    for (int m = 0; m < __BATCH__; ++m) {
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            float value = sums[m][wi];
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffffu, value, offset);
            if (lane == 0 && active)
                partial[(row_local * __BATCH__ + m) * 8 + pair * 4 + wi] = value;
        }
    }
    __syncthreads();

    if (active && pair == 0 && lane == 0) {
        #pragma unroll
        for (int m = 0; m < __BATCH__; ++m) {
            const float* s8 = partial + (row_local * __BATCH__ + m) * 8;
            const float t0 = s8[0] + s8[4];
            const float t1 = s8[1] + s8[5];
            const float t2 = s8[2] + s8[6];
            const float t3 = s8[3] + s8[7];
            out[((size_t)group * __BATCH__ + m) * rows + row] =
                (t0 + t2) + (t1 + t3);
        }
    }
}
"""


def _source() -> str:
    return "\n".join(_TEMPLATE.replace("__BATCH__", str(m)) for m in range(1, 5))


class OrnithNVFP4RoutePairWarp:
    def __init__(self):
        import cupy as cp

        names = tuple(f"nvfp4_route_m{m}_pairwarp_indirect_l2" for m in range(1, 5))
        self.module = cp.RawModule(
            code=_source(), options=("-std=c++14",), name_expressions=names
        )
        self.functions = {
            m: self.module.get_function(f"nvfp4_route_m{m}_pairwarp_indirect_l2")
            for m in range(1, 5)
        }

    def nvfp4(
        self, multiplicity: int, codes, scales, e2, e4, x, out,
        global_scales, slots, input_ids, groups: int, rows: int, cols: int,
    ) -> None:
        multiplicity, groups = int(multiplicity), int(groups)
        rows, cols = int(rows), int(cols)
        self.functions[multiplicity](
            ((rows + 3) // 4, groups), (256,),
            (codes, scales, e2, e4, x, out, global_scales, slots, input_ids,
             np.int32(rows), np.int32(cols)),
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        result = {}
        for multiplicity, fn in self.functions.items():
            fn.compile()
            attrs = getattr(fn, "attributes", {}) or {}
            result[f"M{multiplicity}"] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result
