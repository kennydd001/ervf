from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
__device__ __forceinline__ float p31_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// Four independent production-order BF16 GEMVs sharing each weight load.
// x4 is intentionally read through L2: no 4*cols dynamic shared allocation.
extern "C" __global__ void gemv_bf16_m4_direct_l2(
    const unsigned short* __restrict__ W,
    const float* __restrict__ x4,
    float* __restrict__ out4,
    const int rows,
    const int cols)
{
    const int row=(int)blockIdx.x;
    if(row>=rows)return;
    const unsigned short* w=W+(size_t)row*cols;

    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
    for(int k=(int)threadIdx.x;k<cols;k+=(int)blockDim.x){
        const float ww=p31_bf16(w[k]);
        a0=fmaf(ww,x4[k],a0);
        a1=fmaf(ww,x4[(size_t)cols+k],a1);
        a2=fmaf(ww,x4[(size_t)2*cols+k],a2);
        a3=fmaf(ww,x4[(size_t)3*cols+k],a3);
    }
    for(int offset=16;offset>0;offset>>=1){
        a0+=__shfl_down_sync(0xffffffffu,a0,offset);
        a1+=__shfl_down_sync(0xffffffffu,a1,offset);
        a2+=__shfl_down_sync(0xffffffffu,a2,offset);
        a3+=__shfl_down_sync(0xffffffffu,a3,offset);
    }

    __shared__ float warp_sum[4][32];
    const int lane=(int)threadIdx.x&31;
    const int warp=(int)threadIdx.x>>5;
    if(lane==0){
        warp_sum[0][warp]=a0;
        warp_sum[1][warp]=a1;
        warp_sum[2][warp]=a2;
        warp_sum[3][warp]=a3;
    }
    __syncthreads();
    if(warp==0){
        const int nwarp=((int)blockDim.x+31)>>5;
        float v0=lane<nwarp?warp_sum[0][lane]:0.0f;
        float v1=lane<nwarp?warp_sum[1][lane]:0.0f;
        float v2=lane<nwarp?warp_sum[2][lane]:0.0f;
        float v3=lane<nwarp?warp_sum[3][lane]:0.0f;
        for(int offset=16;offset>0;offset>>=1){
            v0+=__shfl_down_sync(0xffffffffu,v0,offset);
            v1+=__shfl_down_sync(0xffffffffu,v1,offset);
            v2+=__shfl_down_sync(0xffffffffu,v2,offset);
            v3+=__shfl_down_sync(0xffffffffu,v3,offset);
        }
        if(lane==0){
            out4[row]=v0;
            out4[(size_t)rows+row]=v1;
            out4[(size_t)2*rows+row]=v2;
            out4[(size_t)3*rows+row]=v3;
        }
    }
}

// FP32 counterpart for the router.  Thread mapping and reduction match
// production gemv_f32 exactly for each of the four independent rows.
extern "C" __global__ void gemv_f32_m4_direct_l2(
    const float* __restrict__ W,
    const float* __restrict__ x4,
    float* __restrict__ out4,
    const int rows,
    const int cols)
{
    const int row=(int)blockIdx.x;
    if(row>=rows)return;
    const float* w=W+(size_t)row*cols;

    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
    for(int k=(int)threadIdx.x;k<cols;k+=(int)blockDim.x){
        const float ww=w[k];
        a0=fmaf(ww,x4[k],a0);
        a1=fmaf(ww,x4[(size_t)cols+k],a1);
        a2=fmaf(ww,x4[(size_t)2*cols+k],a2);
        a3=fmaf(ww,x4[(size_t)3*cols+k],a3);
    }
    for(int offset=16;offset>0;offset>>=1){
        a0+=__shfl_down_sync(0xffffffffu,a0,offset);
        a1+=__shfl_down_sync(0xffffffffu,a1,offset);
        a2+=__shfl_down_sync(0xffffffffu,a2,offset);
        a3+=__shfl_down_sync(0xffffffffu,a3,offset);
    }

    __shared__ float warp_sum[4][32];
    const int lane=(int)threadIdx.x&31;
    const int warp=(int)threadIdx.x>>5;
    if(lane==0){
        warp_sum[0][warp]=a0;
        warp_sum[1][warp]=a1;
        warp_sum[2][warp]=a2;
        warp_sum[3][warp]=a3;
    }
    __syncthreads();
    if(warp==0){
        const int nwarp=((int)blockDim.x+31)>>5;
        float v0=lane<nwarp?warp_sum[0][lane]:0.0f;
        float v1=lane<nwarp?warp_sum[1][lane]:0.0f;
        float v2=lane<nwarp?warp_sum[2][lane]:0.0f;
        float v3=lane<nwarp?warp_sum[3][lane]:0.0f;
        for(int offset=16;offset>0;offset>>=1){
            v0+=__shfl_down_sync(0xffffffffu,v0,offset);
            v1+=__shfl_down_sync(0xffffffffu,v1,offset);
            v2+=__shfl_down_sync(0xffffffffu,v2,offset);
            v3+=__shfl_down_sync(0xffffffffu,v3,offset);
        }
        if(lane==0){
            out4[row]=v0;
            out4[(size_t)rows+row]=v1;
            out4[(size_t)2*rows+row]=v2;
            out4[(size_t)3*rows+row]=v3;
        }
    }
}
"""


class Phase31DenseDirectKernels:
    def __init__(self):
        import cupy as cp

        names = ("gemv_bf16_m4_direct_l2", "gemv_f32_m4_direct_l2")
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.f = {name: self.mod.get_function(name) for name in names}

    def bf16(self, weights, x4, out4, rows: int, cols: int) -> None:
        if tuple(x4.shape) != (4, int(cols)):
            raise ValueError(f"x4 shape mismatch: {x4.shape}")
        if tuple(out4.shape) != (4, int(rows)):
            raise ValueError(f"out4 shape mismatch: {out4.shape}")
        self.f["gemv_bf16_m4_direct_l2"](
            (int(rows),),
            (256,),
            (weights, x4, out4, np.int32(rows), np.int32(cols)),
        )

    def f32(self, weights, x4, out4, rows: int, cols: int) -> None:
        if tuple(x4.shape) != (4, int(cols)):
            raise ValueError(f"x4 shape mismatch: {x4.shape}")
        if tuple(out4.shape) != (4, int(rows)):
            raise ValueError(f"out4 shape mismatch: {out4.shape}")
        self.f["gemv_f32_m4_direct_l2"](
            (int(rows),),
            (256,),
            (weights, x4, out4, np.int32(rows), np.int32(cols)),
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
