"""Exact-size M2..M8 variants of the Phase33 direct NVFP4 kernel."""
from __future__ import annotations

import numpy as np


_TEMPLATE = r"""
extern "C" __global__ void nvfp4_m__BATCH___warp32_direct_l2(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float global_scale,
    const int rows,
    const int cols)
{
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = e2m1[threadIdx.x];
    __syncthreads();

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int row = (int)blockIdx.x * 8 + warp;
    if (row >= rows) return;

    const int nbytes = cols >> 1;
    const int nvec = nbytes >> 2;
    const unsigned char* crow = codes + (size_t)row * nbytes;
    const unsigned char* srow = scales + (size_t)row * (cols >> 4);
    const uchar4* c4 = reinterpret_cast<const uchar4*>(crow);

    float part[__BATCH__][8];
    #pragma unroll
    for (int m = 0; m < __BATCH__; ++m)
        for (int wi = 0; wi < 8; ++wi) part[m][wi] = 0.0f;

    #pragma unroll
    for (int wi = 0; wi < 8; ++wi) {
        const int tid = lane + 32 * wi;
        float a[__BATCH__];
        #pragma unroll
        for (int m = 0; m < __BATCH__; ++m) a[m] = 0.0f;
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
                const float* xm = x + (size_t)m * cols;
                float z = a[m];
                #pragma unroll
                for (int j = 0; j < 8; ++j) z = fmaf(w[j], xm[k + j], z);
                a[m] = z;
            }
        }
        #pragma unroll
        for (int m = 0; m < __BATCH__; ++m) part[m][wi] = a[m];
    }

    #pragma unroll
    for (int m = 0; m < __BATCH__; ++m) {
        float s8[8];
        #pragma unroll
        for (int wi = 0; wi < 8; ++wi) {
            float value = part[m][wi];
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffffu, value, offset);
            s8[wi] = value;
        }
        if (lane == 0) {
            const float t0 = s8[0] + s8[4];
            const float t1 = s8[1] + s8[5];
            const float t2 = s8[2] + s8[6];
            const float t3 = s8[3] + s8[7];
            out[(size_t)m * rows + row] = (t0 + t2) + (t1 + t3);
        }
    }
}
"""


def _source() -> str:
    return "\n".join(
        _TEMPLATE.replace("__BATCH__", str(batch)) for batch in range(2, 9)
    )


class NVFP4MFamilyWarp32:
    def __init__(self):
        import cupy as cp

        names = tuple(
            f"nvfp4_m{batch}_warp32_direct_l2" for batch in range(2, 9)
        )
        self.module = cp.RawModule(
            code=_source(), options=("-std=c++14",), name_expressions=names
        )
        self.functions = {
            batch: self.module.get_function(f"nvfp4_m{batch}_warp32_direct_l2")
            for batch in range(2, 9)
        }

    def nvfp4(
        self,
        batch: int,
        codes,
        scales,
        e2,
        e4,
        x,
        out,
        global_scale: float,
        rows: int,
        cols: int,
    ) -> None:
        batch, rows, cols = int(batch), int(rows), int(cols)
        if batch not in self.functions:
            raise ValueError(f"unsupported batch {batch}")
        if tuple(x.shape) != (batch, cols):
            raise ValueError(f"x shape mismatch: {x.shape} != {(batch, cols)}")
        if tuple(out.shape) != (batch, rows):
            raise ValueError(f"out shape mismatch: {out.shape} != {(batch, rows)}")
        self.functions[batch](
            ((rows + 7) // 8,),
            (256,),
            (
                codes,
                scales,
                e2,
                e4,
                x,
                out,
                np.float32(global_scale),
                np.int32(rows),
                np.int32(cols),
            ),
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        result = {}
        for batch, fn in self.functions.items():
            fn.compile()
            attrs = getattr(fn, "attributes", {}) or {}
            result[f"M{batch}"] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result

    def nvfp4_ptr(
        self,
        batch: int,
        codes_ptr: int,
        scales_ptr: int,
        e2,
        e4,
        x,
        out,
        global_scale: float,
        rows: int,
        cols: int,
    ) -> None:
        """Launch against raw UVA pointers into mapped pinned host memory."""
        batch, rows, cols = int(batch), int(rows), int(cols)
        if batch not in self.functions:
            raise ValueError(f"unsupported batch {batch}")
        if tuple(x.shape) != (batch, cols):
            raise ValueError(f"x shape mismatch: {x.shape} != {(batch, cols)}")
        if tuple(out.shape) != (batch, rows):
            raise ValueError(f"out shape mismatch: {out.shape} != {(batch, rows)}")
        self.functions[batch](
            ((rows + 7) // 8,),
            (256,),
            (
                np.uint64(codes_ptr),
                np.uint64(scales_ptr),
                e2,
                e4,
                x,
                out,
                np.float32(global_scale),
                np.int32(rows),
                np.int32(cols),
            ),
        )
