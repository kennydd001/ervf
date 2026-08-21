from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
__device__ __forceinline__ float p32_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

extern "C" __global__ void gemv_bf16_m8_direct_l2(
    const unsigned short* __restrict__ W,
    const float* __restrict__ x8,
    float* __restrict__ out8,
    const int rows,
    const int cols)
{
    const int row=(int)blockIdx.x;
    if(row>=rows)return;
    const unsigned short* w=W+(size_t)row*cols;
    float a[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    for(int k=(int)threadIdx.x;k<cols;k+=(int)blockDim.x){
        const float ww=p32_bf16(w[k]);
        #pragma unroll
        for(int m=0;m<8;++m)a[m]=fmaf(ww,x8[(size_t)m*cols+k],a[m]);
    }
    for(int offset=16;offset>0;offset>>=1){
        #pragma unroll
        for(int m=0;m<8;++m)a[m]+=__shfl_down_sync(0xffffffffu,a[m],offset);
    }
    __shared__ float warp_sum[8][32];
    const int lane=(int)threadIdx.x&31;
    const int warp=(int)threadIdx.x>>5;
    if(lane==0){
        #pragma unroll
        for(int m=0;m<8;++m)warp_sum[m][warp]=a[m];
    }
    __syncthreads();
    if(warp==0){
        const int nwarp=((int)blockDim.x+31)>>5;
        float v[8];
        #pragma unroll
        for(int m=0;m<8;++m)v[m]=lane<nwarp?warp_sum[m][lane]:0.0f;
        for(int offset=16;offset>0;offset>>=1){
            #pragma unroll
            for(int m=0;m<8;++m)v[m]+=__shfl_down_sync(0xffffffffu,v[m],offset);
        }
        if(lane==0){
            #pragma unroll
            for(int m=0;m<8;++m)out8[(size_t)m*rows+row]=v[m];
        }
    }
}

extern "C" __global__ void gemv_f32_m8_direct_l2(
    const float* __restrict__ W,
    const float* __restrict__ x8,
    float* __restrict__ out8,
    const int rows,
    const int cols)
{
    const int row=(int)blockIdx.x;
    if(row>=rows)return;
    const float* w=W+(size_t)row*cols;
    float a[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    for(int k=(int)threadIdx.x;k<cols;k+=(int)blockDim.x){
        const float ww=w[k];
        #pragma unroll
        for(int m=0;m<8;++m)a[m]=fmaf(ww,x8[(size_t)m*cols+k],a[m]);
    }
    for(int offset=16;offset>0;offset>>=1){
        #pragma unroll
        for(int m=0;m<8;++m)a[m]+=__shfl_down_sync(0xffffffffu,a[m],offset);
    }
    __shared__ float warp_sum[8][32];
    const int lane=(int)threadIdx.x&31;
    const int warp=(int)threadIdx.x>>5;
    if(lane==0){
        #pragma unroll
        for(int m=0;m<8;++m)warp_sum[m][warp]=a[m];
    }
    __syncthreads();
    if(warp==0){
        const int nwarp=((int)blockDim.x+31)>>5;
        float v[8];
        #pragma unroll
        for(int m=0;m<8;++m)v[m]=lane<nwarp?warp_sum[m][lane]:0.0f;
        for(int offset=16;offset>0;offset>>=1){
            #pragma unroll
            for(int m=0;m<8;++m)v[m]+=__shfl_down_sync(0xffffffffu,v[m],offset);
        }
        if(lane==0){
            #pragma unroll
            for(int m=0;m<8;++m)out8[(size_t)m*rows+row]=v[m];
        }
    }
}
"""


class Phase32DenseM8Kernels:
    def __init__(self):
        import cupy as cp

        names = ("gemv_bf16_m8_direct_l2", "gemv_f32_m8_direct_l2")
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.f = {name: self.mod.get_function(name) for name in names}

    def _launch(self, name, weights, x8, out8, rows: int, cols: int) -> None:
        rows, cols = int(rows), int(cols)
        if tuple(x8.shape) != (8, cols):
            raise ValueError(f"x8 shape mismatch: {x8.shape}")
        if tuple(out8.shape) != (8, rows):
            raise ValueError(f"out8 shape mismatch: {out8.shape}")
        self.f[name](
            (rows,),
            (256,),
            (weights, x8, out8, np.int32(rows), np.int32(cols)),
        )

    def bf16(self, weights, x8, out8, rows: int, cols: int) -> None:
        self._launch("gemv_bf16_m8_direct_l2", weights, x8, out8, rows, cols)

    def f32(self, weights, x8, out8, rows: int, cols: int) -> None:
        self._launch("gemv_f32_m8_direct_l2", weights, x8, out8, rows, cols)

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
