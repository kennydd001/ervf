from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
extern "C" __global__ void nvfp4_m8_warp32_direct_l2(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x8,
    float* __restrict__ out8,
    const float global_scale,
    const int rows,
    const int cols,
    const int apply_relu2)
{
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2m1[threadIdx.x];
    __syncthreads();

    const int lane=(int)threadIdx.x&31;
    const int warp=(int)threadIdx.x>>5;
    const int row=(int)blockIdx.x*8+warp;
    if(row>=rows)return;

    const int nbytes=cols>>1;
    const int nvec=nbytes>>2;
    const unsigned char* crow=codes+(size_t)row*nbytes;
    const unsigned char* srow=scales+(size_t)row*(cols>>4);
    const uchar4* c4=reinterpret_cast<const uchar4*>(crow);

    float part[8][8];
    #pragma unroll
    for(int m=0;m<8;++m)
        for(int wi=0;wi<8;++wi)part[m][wi]=0.0f;

    #pragma unroll
    for(int wi=0;wi<8;++wi){
        const int tid=lane+32*wi;
        float a[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        for(int v=tid;v<nvec;v+=256){
            const uchar4 q=c4[v];
            const int b=v<<2;
            const int k=b<<1;
            const float sc=e4m3[srow[b>>3]]*global_scale;
            const float w[8]={
                lut[q.x&15]*sc,lut[q.x>>4]*sc,
                lut[q.y&15]*sc,lut[q.y>>4]*sc,
                lut[q.z&15]*sc,lut[q.z>>4]*sc,
                lut[q.w&15]*sc,lut[q.w>>4]*sc
            };
            #pragma unroll
            for(int m=0;m<8;++m){
                const float* xm=x8+(size_t)m*cols;
                float z=a[m];
                #pragma unroll
                for(int j=0;j<8;++j)z=fmaf(w[j],xm[k+j],z);
                a[m]=z;
            }
        }
        for(int b=(nvec<<2)+tid;b<nbytes;b+=256){
            const unsigned char q=crow[b];
            const float sc=e4m3[srow[b>>3]]*global_scale;
            const int k=b<<1;
            const float w0=lut[q&15]*sc;
            const float w1=lut[q>>4]*sc;
            #pragma unroll
            for(int m=0;m<8;++m){
                const float* xm=x8+(size_t)m*cols;
                a[m]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],a[m]));
            }
        }
        #pragma unroll
        for(int m=0;m<8;++m)part[m][wi]=a[m];
    }

    #pragma unroll
    for(int m=0;m<8;++m){
        float s8[8];
        #pragma unroll
        for(int wi=0;wi<8;++wi){
            float value=part[m][wi];
            for(int offset=16;offset>0;offset>>=1)
                value+=__shfl_down_sync(0xffffffffu,value,offset);
            s8[wi]=value;
        }
        if(lane==0){
            const float t0=s8[0]+s8[4];
            const float t1=s8[1]+s8[5];
            const float t2=s8[2]+s8[6];
            const float t3=s8[3]+s8[7];
            float value=(t0+t2)+(t1+t3);
            if(apply_relu2){
                const float relu=fmaxf(value,0.0f);
                value=relu*relu;
            }
            out8[(size_t)m*rows+row]=value;
        }
    }
}
"""


class NVFP4M8Warp32:
    def __init__(self):
        import cupy as cp

        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=("nvfp4_m8_warp32_direct_l2",),
        )
        self.fn = self.mod.get_function("nvfp4_m8_warp32_direct_l2")

    def nvfp4(
        self,
        codes,
        scales,
        e2,
        e4,
        x8,
        out8,
        global_scale,
        rows: int,
        cols: int,
        apply_relu2: bool = False,
    ) -> None:
        rows, cols = int(rows), int(cols)
        if tuple(x8.shape) != (8, cols):
            raise ValueError(f"x8 shape mismatch: {x8.shape}")
        if tuple(out8.shape) != (8, rows):
            raise ValueError(f"out8 shape mismatch: {out8.shape}")
        self.fn(
            ((rows + 7) // 8,),
            (256,),
            (
                codes, scales, e2, e4, x8, out8,
                np.float32(global_scale), np.int32(rows), np.int32(cols),
                np.int32(1 if apply_relu2 else 0),
            ),
        )

    def resource_audit(self) -> dict[str, int | None]:
        self.fn.compile()
        attrs = getattr(self.fn, "attributes", {}) or {}
        return {
            "num_regs": attrs.get("num_regs"),
            "shared_size_bytes_static": attrs.get("shared_size_bytes"),
            "local_size_bytes": attrs.get("local_size_bytes"),
            "max_threads_per_block": attrs.get("max_threads_per_block"),
        }
