"""Direct-L2 FP8 E4M3 M1/M4 kernels for the Ornith H4 verifier port."""
from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
__device__ __forceinline__ float p58_e4m3(const unsigned char raw) {
    const unsigned int sign = ((unsigned int) raw) >> 7;
    const unsigned int exp  = (((unsigned int) raw) >> 3) & 15u;
    const unsigned int mant = ((unsigned int) raw) & 7u;
    if (exp == 0u) {
        const float value = ((float) mant) * 0.001953125f;
        return sign ? -value : value;
    }
    if (exp == 15u && mant == 7u) {
        return __int_as_float(0x7fffffff);
    }
    const unsigned int bits = (sign << 31) | ((exp + 120u) << 23) | (mant << 20);
    return __uint_as_float(bits);
}

extern "C" __global__ void fp8_e4m3_m1_direct_l2(
    const unsigned char* __restrict__ weight,
    const unsigned char* __restrict__ x,
    float* __restrict__ out,
    const int rows,
    const int cols,
    const float scale)
{
    const int row = (int) blockIdx.x;
    if (row >= rows) return;
    const unsigned char* w = weight + (size_t) row * cols;
    float acc = 0.0f;
    for (int k = (int) threadIdx.x; k < cols; k += (int) blockDim.x) {
        acc = fmaf(p58_e4m3(w[k]), p58_e4m3(x[k]), acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    __shared__ float warp_sum[32];
    const int lane = (int) threadIdx.x & 31;
    const int warp = (int) threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nwarp = ((int) blockDim.x + 31) >> 5;
        float value = lane < nwarp ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) out[row] = value * scale;
    }
}

extern "C" __global__ void fp8_e4m3_m4_direct_l2(
    const unsigned char* __restrict__ weight,
    const unsigned char* __restrict__ x4,
    float* __restrict__ out4,
    const int rows,
    const int cols,
    const float scale)
{
    const int row = (int) blockIdx.x;
    if (row >= rows) return;
    const unsigned char* w = weight + (size_t) row * cols;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = (int) threadIdx.x; k < cols; k += (int) blockDim.x) {
        const float ww = p58_e4m3(w[k]);
        a0 = fmaf(ww, p58_e4m3(x4[k]),                    a0);
        a1 = fmaf(ww, p58_e4m3(x4[(size_t) cols + k]),    a1);
        a2 = fmaf(ww, p58_e4m3(x4[(size_t) 2*cols + k]),  a2);
        a3 = fmaf(ww, p58_e4m3(x4[(size_t) 3*cols + k]),  a3);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        a0 += __shfl_down_sync(0xffffffffu, a0, offset);
        a1 += __shfl_down_sync(0xffffffffu, a1, offset);
        a2 += __shfl_down_sync(0xffffffffu, a2, offset);
        a3 += __shfl_down_sync(0xffffffffu, a3, offset);
    }
    __shared__ float warp_sum[4][32];
    const int lane = (int) threadIdx.x & 31;
    const int warp = (int) threadIdx.x >> 5;
    if (lane == 0) {
        warp_sum[0][warp] = a0;
        warp_sum[1][warp] = a1;
        warp_sum[2][warp] = a2;
        warp_sum[3][warp] = a3;
    }
    __syncthreads();
    if (warp == 0) {
        const int nwarp = ((int) blockDim.x + 31) >> 5;
        float v0 = lane < nwarp ? warp_sum[0][lane] : 0.0f;
        float v1 = lane < nwarp ? warp_sum[1][lane] : 0.0f;
        float v2 = lane < nwarp ? warp_sum[2][lane] : 0.0f;
        float v3 = lane < nwarp ? warp_sum[3][lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v0 += __shfl_down_sync(0xffffffffu, v0, offset);
            v1 += __shfl_down_sync(0xffffffffu, v1, offset);
            v2 += __shfl_down_sync(0xffffffffu, v2, offset);
            v3 += __shfl_down_sync(0xffffffffu, v3, offset);
        }
        if (lane == 0) {
            out4[row] = v0 * scale;
            out4[(size_t) rows + row] = v1 * scale;
            out4[(size_t) 2*rows + row] = v2 * scale;
            out4[(size_t) 3*rows + row] = v3 * scale;
        }
    }
}
"""


def decode_e4m3_host(raw: np.ndarray) -> np.ndarray:
    """Decode torch.float8_e4m3fn bytes to float32 without lookup tables."""
    value = np.asarray(raw, dtype=np.uint8)
    sign = value >> np.uint8(7)
    exp = (value >> np.uint8(3)) & np.uint8(15)
    mant = value & np.uint8(7)
    out = np.empty(value.shape, dtype=np.float32)
    subnormal = exp == 0
    out[subnormal] = mant[subnormal].astype(np.float32) * np.float32(2.0**-9)
    normal = ~subnormal
    out[normal] = np.ldexp(
        np.float32(1.0) + mant[normal].astype(np.float32) * np.float32(0.125),
        exp[normal].astype(np.int16) - 7,
    ).astype(np.float32)
    out[sign.astype(bool)] *= np.float32(-1.0)
    out[(exp == 15) & (mant == 7)] = np.nan
    return out


class OrnithFP8H4Kernels:
    def __init__(self) -> None:
        import cupy as cp

        names = ("fp8_e4m3_m1_direct_l2", "fp8_e4m3_m4_direct_l2")
        self.cp = cp
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.functions = {name: self.mod.get_function(name) for name in names}

    def m1(self, weight, x, out, rows: int, cols: int, scale: float) -> None:
        if tuple(weight.shape) != (int(rows), int(cols)):
            raise ValueError(f"weight shape mismatch: {weight.shape}")
        if tuple(x.shape) != (int(cols),):
            raise ValueError(f"x shape mismatch: {x.shape}")
        if tuple(out.shape) != (int(rows),):
            raise ValueError(f"out shape mismatch: {out.shape}")
        self.functions["fp8_e4m3_m1_direct_l2"](
            (int(rows),),
            (256,),
            (weight, x, out, np.int32(rows), np.int32(cols), np.float32(scale)),
        )

    def m4(self, weight, x4, out4, rows: int, cols: int, scale: float) -> None:
        if tuple(weight.shape) != (int(rows), int(cols)):
            raise ValueError(f"weight shape mismatch: {weight.shape}")
        if tuple(x4.shape) != (4, int(cols)):
            raise ValueError(f"x4 shape mismatch: {x4.shape}")
        if tuple(out4.shape) != (4, int(rows)):
            raise ValueError(f"out4 shape mismatch: {out4.shape}")
        self.functions["fp8_e4m3_m4_direct_l2"](
            (int(rows),),
            (256,),
            (weight, x4, out4, np.int32(rows), np.int32(cols), np.float32(scale)),
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        result = {}
        for name, function in self.functions.items():
            function.compile()
            attrs = getattr(function, "attributes", {}) or {}
            result[name] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result
