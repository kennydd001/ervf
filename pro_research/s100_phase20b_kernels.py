from __future__ import annotations

import numpy as np

H = 4
MAXM = 4

_SOURCE = r'''
#define MAXM 4

extern "C" __global__ void batched_nvfp4_m4(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float global_scale,
    const int rows,
    const int cols,
    const int M,
    const int apply_relu2)
{
    extern __shared__ float sx[]; // M*cols floats
    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < M * cols; i += blockDim.x) sx[i] = x[i];
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = e2m1[threadIdx.x];
    __syncthreads();

    const int nbytes = cols >> 1;
    const unsigned char* crow = codes + (size_t)row * nbytes;
    const unsigned char* srow = scales + (size_t)row * (cols >> 4);
    const uchar4* c4 = reinterpret_cast<const uchar4*>(crow);
    const int nvec = nbytes >> 2;
    float acc[MAXM] = {0,0,0,0};

    for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
        const uchar4 q = c4[v];
        const int b = v << 2;
        const float sc = e4m3[srow[b >> 3]] * global_scale;
        const int k = b << 1;
        const float w0=lut[q.x&15]*sc, w1=lut[q.x>>4]*sc;
        const float w2=lut[q.y&15]*sc, w3=lut[q.y>>4]*sc;
        const float w4=lut[q.z&15]*sc, w5=lut[q.z>>4]*sc;
        const float w6=lut[q.w&15]*sc, w7=lut[q.w>>4]*sc;
        #pragma unroll
        for (int m=0; m<MAXM; ++m) if (m < M) {
            const float* xm = sx + (size_t)m * cols;
            float a = acc[m];
            a=fmaf(w0,xm[k],a); a=fmaf(w1,xm[k+1],a);
            a=fmaf(w2,xm[k+2],a); a=fmaf(w3,xm[k+3],a);
            a=fmaf(w4,xm[k+4],a); a=fmaf(w5,xm[k+5],a);
            a=fmaf(w6,xm[k+6],a); a=fmaf(w7,xm[k+7],a);
            acc[m]=a;
        }
    }
    for (int b=(nvec<<2)+threadIdx.x; b<nbytes; b+=blockDim.x) {
        const unsigned char q=crow[b];
        const float sc=e4m3[srow[b>>3]]*global_scale;
        const int k=b<<1;
        const float w0=lut[q&15]*sc, w1=lut[q>>4]*sc;
        #pragma unroll
        for (int m=0; m<MAXM; ++m) if (m < M) {
            const float* xm=sx+(size_t)m*cols;
            acc[m]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],acc[m]));
        }
    }

    __shared__ float ws[MAXM][32];
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    #pragma unroll
    for (int m=0;m<MAXM;++m) if (m<M) {
        float a=acc[m];
        for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffffu,a,o);
        if(lane==0) ws[m][warp]=a;
    }
    __syncthreads();
    if(warp==0) {
        const int nw=(blockDim.x+31)>>5;
        #pragma unroll
        for(int m=0;m<MAXM;++m) if(m<M) {
            float a=(lane<nw)?ws[m][lane]:0.0f;
            for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffffu,a,o);
            if(lane==0) {
                if(apply_relu2){float r=fmaxf(a,0.0f);a=r*r;}
                out[(size_t)m*rows+row]=a;
            }
        }
    }
}

extern "C" __global__ void panel_scan_batched_m4(
    const float* __restrict__ act,
    const int M,
    const int inter,
    unsigned int* __restrict__ masks,
    int* __restrict__ plist,
    int* __restrict__ pcount,
    int* __restrict__ nz,
    int* __restrict__ nzc)
{
    const int np=inter>>4;
    for(int p=threadIdx.x;p<np;p+=blockDim.x)masks[p]=0u;
    if(threadIdx.x==0){*pcount=0;*nzc=0;}
    __syncthreads();
    for(int j=threadIdx.x;j<inter;j+=blockDim.x){
        bool any=false;
        #pragma unroll
        for(int m=0;m<MAXM;++m) if(m<M && act[(size_t)m*inter+j]!=0.0f) any=true;
        if(any) atomicOr(&masks[j>>4],1u<<(j&15));
    }
    __syncthreads();
    if(threadIdx.x==0){
        int a=0,b=0;
        for(int p=0;p<np;++p){unsigned int mk=masks[p];if(mk){plist[a++]=p;
            for(int c=0;c<16;++c)if(mk&(1u<<c))nz[b++]=(p<<4)+c;}}
        *pcount=a;*nzc=b;
    }
}

extern "C" __global__ void down_batched_partial_m4(
    const unsigned char* __restrict__ bank,
    const float* __restrict__ act,
    const int M,
    const int* __restrict__ plist,
    const unsigned int* __restrict__ masks,
    const int* __restrict__ pcount,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    float* __restrict__ partials,
    const float global_scale,
    const int rows,
    const int inter)
{
    const int nc=gridDim.y, chunk=blockIdx.y;
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)return;
    __shared__ float l2[16];__shared__ float l4[256];
    if(threadIdx.x<16)l2[threadIdx.x]=e2m1[threadIdx.x];
    if(threadIdx.x<256)l4[threadIdx.x]=e4m3[threadIdx.x];
    __syncthreads();
    const int hb=row>>1,hi=row&1,rowhalf=rows>>1;
    const size_t stride=(size_t)rows+16u*(size_t)rowhalf;
    float acc[MAXM]={0,0,0,0};
    for(int pi=chunk;pi<*pcount;pi+=nc){
        const int p=plist[pi];
        const unsigned char* pb=bank+(size_t)p*stride;
        const float sc=l4[pb[row]]*global_scale;
        const unsigned char* pc=pb+rows;
        unsigned int mk=masks[p];
        while(mk){int c=__ffs(mk)-1;mk&=mk-1;
            const unsigned char q=pc[(size_t)c*rowhalf+hb];
            const float w=l2[hi?(q>>4):(q&15)]*sc;
            const int j=(p<<4)+c;
            #pragma unroll
            for(int m=0;m<MAXM;++m)if(m<M)acc[m]=fmaf(w,act[(size_t)m*inter+j],acc[m]);
        }
    }
    #pragma unroll
    for(int m=0;m<MAXM;++m)if(m<M)
        partials[((size_t)m*nc+chunk)*rows+row]=acc[m];
}

extern "C" __global__ void reduce_down_batched_m4(
    const float* __restrict__ partials,
    float* __restrict__ out,
    const int M,
    const int rows,
    const int nc)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    const int m=blockIdx.y;
    if(row>=rows || m>=M)return;
    float a=0.0f;
    for(int c=0;c<nc;++c)a+=partials[((size_t)m*nc+c)*rows+row];
    out[(size_t)m*rows+row]=a;
}
'''

class Phase20BKernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.mod=cp.RawModule(code=_SOURCE,options=("-std=c++14",),name_expressions=(
            "batched_nvfp4_m4","panel_scan_batched_m4","down_batched_partial_m4","reduce_down_batched_m4"))
        self.batched=self.mod.get_function("batched_nvfp4_m4")
        self.scan=self.mod.get_function("panel_scan_batched_m4")
        self.down=self.mod.get_function("down_batched_partial_m4")
        self.reduce=self.mod.get_function("reduce_down_batched_m4")

    def nvfp4(self,codes,scales,e2,e4,x,out,gscale,rows,cols,M,apply_relu2=False):
        shared=int(M)*int(cols)*4
        # H4 down projections can require >48 KiB dynamic shared memory
        # (4 * 3712 FP32 activations = 59,392 bytes).  CUDA rejects that
        # launch unless the function opts into the larger per-block budget.
        if shared > 48 * 1024:
            self.batched.max_dynamic_shared_size_bytes = int(shared)
        self.batched((int(rows),),(256,),
                     (codes,scales,e2,e4,x,out,np.float32(gscale),np.int32(rows),
                      np.int32(cols),np.int32(M),np.int32(1 if apply_relu2 else 0)),
                     shared_mem=shared)

    def down_group(self,rt,bank_ptr,acts,out,gscale,M,state,hidden,inter):
        cp=self.cp;nc=int(rt.fused.nchunks)
        self.scan((1,),(256,),
                  (acts,np.int32(M),np.int32(inter),state["masks"],state["plist"],
                   state["pcount"],state["nz"],state["nzc"]))
        max_warps=int(inter)+int(inter)//16
        blocks=(max_warps*32+255)//256
        rt.fused.gather_k((blocks,),(256,),
                          (np.uint64(bank_ptr),state["mirror"],state["plist"],
                           state["pcount"],state["nz"],state["nzc"],np.int32(hidden)))
        grid=((int(hidden)+127)//128,nc)
        self.down(grid,(128,),
                  (state["mirror"],acts,np.int32(M),state["plist"],state["masks"],
                   state["pcount"],rt.fused.e2m1,rt.fused.e4m3,state["partials_b"],
                   np.float32(gscale),np.int32(hidden),np.int32(inter)))
        self.reduce(((int(hidden)+255)//256,int(M)),(256,),
                    (state["partials_b"],out,np.int32(M),np.int32(hidden),np.int32(nc)))

    def alloc_down_state(self,rt,hidden,inter):
        cp=self.cp;npanel=int(inter)//16;nc=int(rt.fused.nchunks)
        return {
            "masks":cp.zeros(npanel,cp.uint32),"plist":cp.zeros(npanel,cp.int32),
            "pcount":cp.zeros(1,cp.int32),"nz":cp.zeros(inter,cp.int32),
            "nzc":cp.zeros(1,cp.int32),
            "mirror":cp.zeros(npanel*(int(hidden)+16*(int(hidden)//2)),cp.uint8),
            "partials_b":cp.zeros(MAXM*nc*int(hidden),cp.float32),
        }
