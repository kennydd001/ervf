from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned char p35_e2m1_rne(float x) {
    const float a=fabsf(x);
    unsigned char c=7;
    if(a<=5.0f)c=6;
    if(a<3.5f)c=5;
    if(a<=2.5f)c=4;
    if(a<1.75f)c=3;
    if(a<=1.25f)c=2;
    if(a<0.75f)c=1;
    if(a<=0.25f)c=0;
    if(x<0.0f)c|=8;
    return c;
}

extern "C" __global__ void quantize_static_nvfp4_m8(
    const float* __restrict__ x,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ blocked_scales,
    const float tensor_scale,
    const int m,
    const int k,
    const int sfp)
{
    const int sfk=k>>4;
    const int task=(int)blockIdx.x;
    const int row=task/sfk;
    const int col=task-row*sfk;
    if(row>=m)return;
    const int lane=(int)threadIdx.x;
    const int base=(row*k)+(col<<4);

    float value=lane<16?fabsf(x[base+lane]):0.0f;
    for(int offset=16;offset>0;offset>>=1)
        value=fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));

    __shared__ float reciprocal;
    if(lane==0){
        float desired=(value/6.0f)/tensor_scale;
        desired=fminf(448.0f,fmaxf(0.015625f,desired));
        const __nv_fp8_e4m3 q=__nv_fp8_e4m3(desired);
        const unsigned char raw=*reinterpret_cast<const unsigned char*>(&q);
        reciprocal=(1.0f/tensor_scale)/(float)q;

        const int rb=row>>7;
        const int cb=col>>2;
        const int r=row&127;
        const int c=col&3;
        const int ncb=sfp>>2;
        const size_t scale_offset=(size_t)(rb*ncb+cb)*512u
            +(size_t)(r&31)*16u+(size_t)(r>>5)*4u+(size_t)c;
        blocked_scales[scale_offset]=raw;
    }
    __syncwarp();

    if(lane<8){
        const float lo=fminf(6.0f,fmaxf(-6.0f,x[base+(lane<<1)]*reciprocal));
        const float hi=fminf(6.0f,fmaxf(-6.0f,x[base+(lane<<1)+1]*reciprocal));
        packed[(size_t)row*(k>>1)+(col<<3)+lane]
            =p35_e2m1_rne(lo)|(p35_e2m1_rne(hi)<<4);
    }
}
"""


class FusedStaticNVFP4Quantizer:
    def __init__(self, k: int, m: int = 8):
        import cupy as cp

        self.cp = cp
        self.k = int(k)
        self.m = int(m)
        if self.k % 16 or not (1 <= self.m <= 8):
            raise ValueError((k, m))
        self.sfk = self.k // 16
        self.sfp = ((self.sfk + 3) // 4) * 4
        self.packed = cp.empty((self.m, self.k // 2), cp.uint8)
        self.blocked_scales = cp.zeros((128, self.sfp), cp.uint8)
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=("quantize_static_nvfp4_m8",),
        )
        self.fn = self.mod.get_function("quantize_static_nvfp4_m8")

    def quantize(self, x, tensor_scale: float) -> tuple:
        if tuple(x.shape) != (self.m, self.k):
            raise ValueError(f"expected {(self.m, self.k)}, got {x.shape}")
        self.fn(
            (self.m * self.sfk,),
            (32,),
            (
                x, self.packed, self.blocked_scales,
                np.float32(tensor_scale), np.int32(self.m),
                np.int32(self.k), np.int32(self.sfp),
            ),
        )
        return self.packed, self.blocked_scales

    def resource_audit(self) -> dict[str, int | None]:
        self.fn.compile()
        attrs = getattr(self.fn, "attributes", {}) or {}
        return {
            "num_regs": attrs.get("num_regs"),
            "shared_size_bytes_static": attrs.get("shared_size_bytes"),
            "local_size_bytes": attrs.get("local_size_bytes"),
            "max_threads_per_block": attrs.get("max_threads_per_block"),
        }
