from __future__ import annotations

import numpy as np

HS = (2, 4, 8)

HEADER = r"""
__device__ __forceinline__ float p17_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
"""

def block_conv(T: int) -> str:
    return f"""
extern "C" __global__ void block_conv_t{T}(
    const float* __restrict__ state0,
    const float* __restrict__ xseq,
    const unsigned short* __restrict__ wt,
    const unsigned short* __restrict__ bias,
    float* __restrict__ outseq,
    float* __restrict__ state_final,
    const int conv_dim,
    const int K,
    const int x_stride,
    const int x_offset)
{{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    #pragma unroll
    for (int k=0;k<8;++k) st[k]=0.0f;
    for (int k=0;k<K;++k) st[k]=state0[(size_t)c*K+k];

    #pragma unroll
    for (int t=0;t<{T};++t) {{
        for (int k=0;k<K-1;++k) st[k]=st[k+1];
        st[K-1]=xseq[(size_t)t*x_stride+x_offset+c];
        float acc=p17_bf16(bias[c]);
        for (int k=0;k<K;++k)
            acc=fmaf(p17_bf16(wt[(size_t)c*K+k]),st[k],acc);
        outseq[(size_t)t*conv_dim+c]=acc/(1.0f+__expf(-acc));
    }}
    for (int k=0;k<K;++k)
        state_final[(size_t)c*K+k]=st[k];
}}
"""

def block_dt(T: int) -> str:
    return f"""
extern "C" __global__ void block_dt_t{T}(
    const float* __restrict__ dtr,
    const unsigned short* __restrict__ bias,
    float* __restrict__ dt,
    const int Hh,
    const int dtr_stride,
    const int dtr_offset)
{{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int total={T}*Hh;
    if(i>=total)return;
    int h=i%Hh;
    int t=i/Hh;
    float v=dtr[(size_t)t*dtr_stride+dtr_offset+h]+p17_bf16(bias[h]);
    float sp=(v>20.0f)?v:__logf(1.0f+__expf(v));
    dt[i]=sp;
}}
"""

def prepare_affine(T: int) -> str:
    return f"""
extern "C" __global__ void prepare_affine_t{T}(
    const float* __restrict__ x,
    const float* __restrict__ dt,
    const float* __restrict__ Alog,
    float* __restrict__ dx,
    float* __restrict__ decay,
    const int Hh,
    const int P,
    const int x_stride,
    const int x_offset)
{{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int total={T}*Hh*P;
    if(i>=total)return;
    int p=i%P;
    int hp=i/P;
    int h=hp%Hh;
    int t=hp/Hh;
    float d=dt[(size_t)t*Hh+h];
    dx[i]=d*x[(size_t)t*x_stride+x_offset+h*P+p];
    if(p==0)
        decay[(size_t)t*Hh+h]=__expf(-__expf(Alog[h])*d);
}}
"""

def scan_prefix(T: int) -> str:
    return f"""
extern "C" __global__ void ssm_prefix_t{T}(
    const float* __restrict__ state0,
    const float* __restrict__ dx,
    const float* __restrict__ Bseq,
    const float* __restrict__ decay,
    float* __restrict__ states,
    const int Hh,
    const int P,
    const int N,
    const int hpg,
    const int b_stride,
    const int b_offset)
{{
    const int T={T};
    size_t tid=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t elems=(size_t)Hh*P*N;
    size_t elem=tid/T;
    int lane=(int)(tid&(T-1));
    if(elem>=elems)return;

    int n=(int)(elem%N);
    size_t q=elem/N;
    int p=(int)(q%P);
    int h=(int)(q/P);
    int g=h/hpg;
    int G=Hh/hpg;

    float A=decay[(size_t)lane*Hh+h];
    float bb=dx[((size_t)lane*Hh+h)*P+p] *
             Bseq[(size_t)lane*b_stride+b_offset+g*N+n];

    unsigned mask=__activemask();
    #pragma unroll
    for(int off=1;off<T;off<<=1){{
        float pa=__shfl_up_sync(mask,A,off,T);
        float pb=__shfl_up_sync(mask,bb,off,T);
        if(lane>=off){{
            bb=fmaf(A,pb,bb);
            A=A*pa;
        }}
    }}
    states[(size_t)lane*elems+elem]=fmaf(A,state0[elem],bb);
}}
"""

def scan_serial(T: int) -> str:
    return f"""
extern "C" __global__ void ssm_serial_block_t{T}(
    const float* __restrict__ state0,
    const float* __restrict__ dx,
    const float* __restrict__ Bseq,
    const float* __restrict__ decay,
    float* __restrict__ states,
    const int Hh,
    const int P,
    const int N,
    const int hpg,
    const int b_stride,
    const int b_offset)
{{
    size_t elem=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t elems=(size_t)Hh*P*N;
    if(elem>=elems)return;
    int n=(int)(elem%N);
    size_t q=elem/N;
    int p=(int)(q%P);
    int h=(int)(q/P);
    int g=h/hpg;
    int G=Hh/hpg;
    float s=state0[elem];
    #pragma unroll
    for(int t=0;t<{T};++t){{
        float a=decay[(size_t)t*Hh+h];
        float b=dx[((size_t)t*Hh+h)*P+p] *
                Bseq[(size_t)t*b_stride+b_offset+g*N+n];
        s=fmaf(a,s,b);
        states[(size_t)t*elems+elem]=s;
    }}
}}
"""

def ssm_y(T: int) -> str:
    return f"""
extern "C" __global__ void ssm_y_t{T}(
    const float* __restrict__ states,
    const float* __restrict__ Cseq,
    const float* __restrict__ x,
    const unsigned short* __restrict__ D,
    float* __restrict__ y,
    const int Hh,
    const int P,
    const int N,
    const int hpg,
    const int c_stride,
    const int c_offset,
    const int x_stride,
    const int x_offset)
{{
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t total=(size_t){T}*Hh*P;
    if(i>=total)return;
    int p=(int)(i%P);
    size_t hp=i/P;
    int h=(int)(hp%Hh);
    int t=(int)(hp/Hh);
    int g=h/hpg;
    int G=Hh/hpg;
    size_t elems=(size_t)Hh*P*N;
    const float* s=states+(size_t)t*elems+((size_t)h*P+p)*N;
    const float* c=Cseq+(size_t)t*c_stride+c_offset+g*N;
    float acc=0.0f;
    for(int n=0;n<N;++n) acc=fmaf(s[n],c[n],acc);
    y[i]=fmaf(p17_bf16(D[h]),
               x[(size_t)t*x_stride+x_offset+h*P+p],acc);
}}
"""

def gated_norm(T: int) -> str:
    return f"""
extern "C" __global__ void gated_norm_t{T}(
    const float* __restrict__ y,
    const float* __restrict__ z,
    const unsigned short* __restrict__ w,
    float* __restrict__ out,
    const int n,
    const int group_size,
    const float eps,
    const int z_stride,
    const int z_offset)
{{
    int groups=n/group_size;
    int bg=blockIdx.x;
    int t=bg/groups;
    int g=bg-t*groups;
    if(t>={T})return;
    int base=t*n+g*group_size;

    extern __shared__ float vals[];
    __shared__ float ws[32];
    __shared__ float scale;

    float acc=0.0f;
    for(int j=threadIdx.x;j<group_size;j+=blockDim.x){{
        int i=base+j;
        float zz=z[(size_t)t*z_stride+z_offset+g*group_size+j];
        float gated=y[i]*(zz/(1.0f+__expf(-zz)));
        vals[j]=gated;
        acc=fmaf(gated,gated,acc);
    }}
    for(int o=16;o>0;o>>=1)
        acc+=__shfl_down_sync(0xffffffffu,acc,o);
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    if(lane==0)ws[warp]=acc;
    __syncthreads();
    if(threadIdx.x==0){{
        float s=0.0f;
        int nw=(blockDim.x+31)>>5;
        for(int q=0;q<nw;++q)s+=ws[q];
        scale=rsqrtf(s/(float)group_size+eps);
    }}
    __syncthreads();
    for(int j=threadIdx.x;j<group_size;j+=blockDim.x){{
        int i=base+j;
        out[i]=vals[j]*scale*p17_bf16(w[g*group_size+j]);
    }}
}}
"""

SOURCE = HEADER + "\n".join(
    block_conv(T)+block_dt(T)+prepare_affine(T)+scan_prefix(T)+scan_serial(T)+ssm_y(T)+gated_norm(T)
    for T in HS
)

class Phase17Kernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        names=[]
        for T in HS:
            names += [
                f"block_conv_t{T}", f"block_dt_t{T}", f"prepare_affine_t{T}",
                f"ssm_prefix_t{T}", f"ssm_serial_block_t{T}",
                f"ssm_y_t{T}", f"gated_norm_t{T}",
            ]
        self.mod=cp.RawModule(code=SOURCE,options=("-std=c++14",),name_expressions=names)
        self.f={n:self.mod.get_function(n) for n in names}

    def block_conv(self,T,state0,xseq,wt,bias,outseq,state_final,conv_dim,K,
                   x_stride=None,x_offset=0):
        if x_stride is None: x_stride=conv_dim
        self.f[f"block_conv_t{T}"](
            ((int(conv_dim)+255)//256,), (256,),
            (state0,xseq,wt,bias,outseq,state_final,np.int32(conv_dim),np.int32(K),
             np.int32(x_stride),np.int32(x_offset))
        )

    def block_dt(self,T,dtr,bias,dt,Hh,dtr_stride=None,dtr_offset=0):
        if dtr_stride is None: dtr_stride=Hh
        total=int(T)*int(Hh)
        self.f[f"block_dt_t{T}"](
            ((total+255)//256,), (256,),
            (dtr,bias,dt,np.int32(Hh),np.int32(dtr_stride),np.int32(dtr_offset))
        )

    def prepare(self,T,x,dt,Alog,dx,decay,Hh,P,x_stride=None,x_offset=0):
        if x_stride is None: x_stride=Hh*P
        total=int(T)*int(Hh)*int(P)
        self.f[f"prepare_affine_t{T}"](
            ((total+255)//256,), (256,),
            (x,dt,Alog,dx,decay,np.int32(Hh),np.int32(P),
             np.int32(x_stride),np.int32(x_offset))
        )

    def scan(self,kind,T,state0,dx,Bseq,decay,states,Hh,P,N,hpg,
             b_stride=None,b_offset=0):
        if b_stride is None: b_stride=(Hh//hpg)*N
        elems=int(Hh)*int(P)*int(N)
        if kind=="prefix":
            total=elems*int(T)
            name=f"ssm_prefix_t{T}"
        elif kind=="serial":
            total=elems
            name=f"ssm_serial_block_t{T}"
        else:
            raise ValueError(kind)
        self.f[name](
            ((total+255)//256,), (256,),
            (state0,dx,Bseq,decay,states,np.int32(Hh),np.int32(P),
             np.int32(N),np.int32(hpg),np.int32(b_stride),np.int32(b_offset))
        )

    def y(self,T,states,Cseq,x,D,y,Hh,P,N,hpg,
          c_stride=None,c_offset=0,x_stride=None,x_offset=0):
        if c_stride is None: c_stride=(Hh//hpg)*N
        if x_stride is None: x_stride=Hh*P
        total=int(T)*int(Hh)*int(P)
        self.f[f"ssm_y_t{T}"](
            ((total+255)//256,), (256,),
            (states,Cseq,x,D,y,np.int32(Hh),np.int32(P),
             np.int32(N),np.int32(hpg),np.int32(c_stride),np.int32(c_offset),
             np.int32(x_stride),np.int32(x_offset))
        )

    def gated(self,T,y,z,w,out,n,group_size,eps,z_stride=None,z_offset=0):
        if z_stride is None: z_stride=n
        groups=int(n)//int(group_size)
        self.f[f"gated_norm_t{T}"](
            (int(T)*groups,), (256,),
            (y,z,w,out,np.int32(n),np.int32(group_size),np.float32(eps),
             np.int32(z_stride),np.int32(z_offset)),
            shared_mem=int(group_size)*4,
        )
