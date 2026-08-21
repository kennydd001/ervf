"""Indexed exact-ERVF rerank kernel for token-major LM-head shortlists."""
from __future__ import annotations

import numpy as np


SOURCE = r"""
extern "C" __global__ void nvfp4_ervf_shortlist_w16(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1_lut,
    const float* __restrict__ e4m3_lut,
    const float* __restrict__ x4,
    const long long* __restrict__ row_ids,
    float* __restrict__ out,
    const float global_scale,
    const int shortlist,
    const int cols)
{
    extern __shared__ float sx[];
    const int lane = (int)threadIdx.x & 15;
    const int sub = (int)threadIdx.x >> 4;
    const int item = (int)blockIdx.x * 16 + sub;
    const int total = 4 * shortlist;
    const int token = min(item / shortlist, 3);
    for (int i = (int)threadIdx.x; i < cols; i += (int)blockDim.x)
        sx[i] = x4[(size_t)token * cols + i];
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();
    if (item >= total) return;

    const long long row = row_ids[item];
    const int nbytes = cols >> 1;
    const int nvec = nbytes >> 2;
    const unsigned char* crow = codes + (size_t)row * nbytes;
    const unsigned char* srow = scales + (size_t)row * (cols >> 4);
    const uchar4* c4 = reinterpret_cast<const uchar4*>(crow);

    float part[16];
    #pragma unroll
    for (int vi = 0; vi < 16; ++vi) part[vi] = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < 16; ++vi) {
        const int tid = lane + 16 * vi;
        float acc = 0.0f;
        for (int v = tid; v < nvec; v += 256) {
            const uchar4 q = c4[v];
            const int b = v << 2;
            const int k = b << 1;
            const float sc = e4m3_lut[srow[b >> 3]] * global_scale;
            acc = fmaf(lut[q.x & 15] * sc, sx[k], acc);
            acc = fmaf(lut[q.x >> 4] * sc, sx[k + 1], acc);
            acc = fmaf(lut[q.y & 15] * sc, sx[k + 2], acc);
            acc = fmaf(lut[q.y >> 4] * sc, sx[k + 3], acc);
            acc = fmaf(lut[q.z & 15] * sc, sx[k + 4], acc);
            acc = fmaf(lut[q.z >> 4] * sc, sx[k + 5], acc);
            acc = fmaf(lut[q.w & 15] * sc, sx[k + 6], acc);
            acc = fmaf(lut[q.w >> 4] * sc, sx[k + 7], acc);
        }
        part[vi] = acc;
    }

    float s8[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float loc0 = part[w * 2];
        float loc1 = part[w * 2 + 1];
        loc0 += loc1;
        for (int off = 8; off > 0; off >>= 1)
            loc0 += __shfl_down_sync(0xffffffffu, loc0, off, 16);
        s8[w] = loc0;
    }
    if (lane == 0) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        out[item] = (t0 + t2) + (t1 + t3);
    }
}
"""


class ExactERVFShortlist:
    def __init__(self):
        import cupy as cp

        self.module = cp.RawModule(
            code=SOURCE, options=("-std=c++14",),
            name_expressions=("nvfp4_ervf_shortlist_w16",),
        )
        self.function = self.module.get_function("nvfp4_ervf_shortlist_w16")

    def __call__(
        self, codes, scales, e2, e4, x4, row_ids_ptr: int, out_ptr: int,
        global_scale: float, shortlist: int, cols: int,
    ) -> None:
        total = 4 * int(shortlist)
        self.function(
            ((total + 15) // 16,), (256,),
            (codes, scales, e2, e4, x4, np.uint64(row_ids_ptr), np.uint64(out_ptr),
             np.float32(global_scale), np.int32(shortlist), np.int32(cols)),
            shared_mem=int(cols) * 4,
        )

    def resource_audit(self) -> dict[str, int | None]:
        self.function.compile()
        attrs = getattr(self.function, "attributes", {}) or {}
        return {
            "num_regs": attrs.get("num_regs"),
            "shared_size_bytes_static": attrs.get("shared_size_bytes"),
            "local_size_bytes": attrs.get("local_size_bytes"),
            "max_threads_per_block": attrs.get("max_threads_per_block"),
        }
