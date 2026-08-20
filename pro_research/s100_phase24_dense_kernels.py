from __future__ import annotations

import numpy as np

_SOURCE = r"""
__device__ __forceinline__ float p24_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

/* Same thread/k mapping and same two-level reduction as production
   gpu_kernels.py::gemv_bf16. The only change is four independent accumulators
   sharing each BF16 weight load. */
extern "C" __global__ void gemv_bf16_m4(
    const unsigned short* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int rows,
    const int cols)
{
    extern __shared__ float sx[];
    int row=blockIdx.x;
    if(row>=rows)return;
    for(int i=threadIdx.x;i<4*cols;i+=blockDim.x)sx[i]=x[i];
    __syncthreads();

    const unsigned short* w=W+(size_t)row*cols;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
    for(int k=threadIdx.x;k<cols;k+=blockDim.x){
        float ww=p24_bf16(w[k]);
        a0=fmaf(ww,sx[k],a0);
        a1=fmaf(ww,sx[(size_t)cols+k],a1);
        a2=fmaf(ww,sx[(size_t)2*cols+k],a2);
        a3=fmaf(ww,sx[(size_t)3*cols+k],a3);
    }

    for(int o=16;o>0;o>>=1){
        a0+=__shfl_down_sync(0xffffffffu,a0,o);
        a1+=__shfl_down_sync(0xffffffffu,a1,o);
        a2+=__shfl_down_sync(0xffffffffu,a2,o);
        a3+=__shfl_down_sync(0xffffffffu,a3,o);
    }
    __shared__ float ws[4][32];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(lane==0){ws[0][warp]=a0;ws[1][warp]=a1;ws[2][warp]=a2;ws[3][warp]=a3;}
    __syncthreads();
    if(warp==0){
        int nw=(blockDim.x+31)>>5;
        float v0=lane<nw?ws[0][lane]:0.0f;
        float v1=lane<nw?ws[1][lane]:0.0f;
        float v2=lane<nw?ws[2][lane]:0.0f;
        float v3=lane<nw?ws[3][lane]:0.0f;
        for(int o=16;o>0;o>>=1){
            v0+=__shfl_down_sync(0xffffffffu,v0,o);
            v1+=__shfl_down_sync(0xffffffffu,v1,o);
            v2+=__shfl_down_sync(0xffffffffu,v2,o);
            v3+=__shfl_down_sync(0xffffffffu,v3,o);
        }
        if(lane==0){
            out[row]=v0;
            out[(size_t)rows+row]=v1;
            out[(size_t)2*rows+row]=v2;
            out[(size_t)3*rows+row]=v3;
        }
    }
}

/* Exact M4 analogue of production gemv_f32. */
extern "C" __global__ void gemv_f32_m4(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int rows,
    const int cols)
{
    extern __shared__ float sx[];
    int row=blockIdx.x;
    if(row>=rows)return;
    for(int i=threadIdx.x;i<4*cols;i+=blockDim.x)sx[i]=x[i];
    __syncthreads();

    const float* w=W+(size_t)row*cols;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
    for(int k=threadIdx.x;k<cols;k+=blockDim.x){
        float ww=w[k];
        a0=fmaf(ww,sx[k],a0);
        a1=fmaf(ww,sx[(size_t)cols+k],a1);
        a2=fmaf(ww,sx[(size_t)2*cols+k],a2);
        a3=fmaf(ww,sx[(size_t)3*cols+k],a3);
    }
    for(int o=16;o>0;o>>=1){
        a0+=__shfl_down_sync(0xffffffffu,a0,o);
        a1+=__shfl_down_sync(0xffffffffu,a1,o);
        a2+=__shfl_down_sync(0xffffffffu,a2,o);
        a3+=__shfl_down_sync(0xffffffffu,a3,o);
    }
    __shared__ float ws[4][32];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(lane==0){ws[0][warp]=a0;ws[1][warp]=a1;ws[2][warp]=a2;ws[3][warp]=a3;}
    __syncthreads();
    if(warp==0){
        int nw=(blockDim.x+31)>>5;
        float v0=lane<nw?ws[0][lane]:0.0f;
        float v1=lane<nw?ws[1][lane]:0.0f;
        float v2=lane<nw?ws[2][lane]:0.0f;
        float v3=lane<nw?ws[3][lane]:0.0f;
        for(int o=16;o>0;o>>=1){
            v0+=__shfl_down_sync(0xffffffffu,v0,o);
            v1+=__shfl_down_sync(0xffffffffu,v1,o);
            v2+=__shfl_down_sync(0xffffffffu,v2,o);
            v3+=__shfl_down_sync(0xffffffffu,v3,o);
        }
        if(lane==0){
            out[row]=v0;
            out[(size_t)rows+row]=v1;
            out[(size_t)2*rows+row]=v2;
            out[(size_t)3*rows+row]=v3;
        }
    }
}
"""

class DenseM4Kernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.mod=cp.RawModule(
            code=_SOURCE,
            options=("-std=c++14",),
            name_expressions=("gemv_bf16_m4","gemv_f32_m4"),
        )
        self.bf16_k=self.mod.get_function("gemv_bf16_m4")
        self.f32_k=self.mod.get_function("gemv_f32_m4")

    def bf16(self,W,x,out,rows,cols):
        shared=4*int(cols)*4
        if shared>48*1024:
            self.bf16_k.max_dynamic_shared_size_bytes=shared
        self.bf16_k((int(rows),),(256,),
            (W,x,out,np.int32(rows),np.int32(cols)),
            shared_mem=shared)

    def f32(self,W,x,out,rows,cols):
        shared=4*int(cols)*4
        if shared>48*1024:
            self.f32_k.max_dynamic_shared_size_bytes=shared
        self.f32_k((int(rows),),(256,),
            (W,x,out,np.int32(rows),np.int32(cols)),
            shared_mem=shared)
