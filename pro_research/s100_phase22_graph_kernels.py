from __future__ import annotations
import numpy as np

_SOURCE=r"""
extern "C" __global__ void embed4_bf16(
    const unsigned short* __restrict__ table,
    const int* __restrict__ toks,
    float* __restrict__ h4,
    const int hidden)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int total=4*hidden;
    if(i>=total)return;
    int t=i/hidden, d=i-t*hidden;
    unsigned int u=((unsigned int)table[(size_t)toks[t]*hidden+d])<<16;
    h4[i]=__uint_as_float(u);
}

extern "C" __global__ void kv_append_f32_dp(
    float* __restrict__ cache,
    const float* __restrict__ src,
    const int* __restrict__ base_pos,
    const int offset,
    const int n_kv,
    const int head_dim,
    const int max_ctx)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int total=n_kv*head_dim;
    if(i>=total)return;
    int pos=base_pos[0]+offset;
    int g=i/head_dim,d=i-g*head_dim;
    cache[((size_t)g*max_ctx+pos)*head_dim+d]=src[i];
}

/* Verbatim production attn_decode arithmetic; only t source differs. */
extern "C" __global__ void attn_decode_f32_dp(
    const float* __restrict__ q,
    const float* __restrict__ Kc,
    const float* __restrict__ Vc,
    float* __restrict__ out,
    const int* __restrict__ base_pos,
    const int offset,
    const int head_dim,
    const int groups,
    const int max_ctx,
    const float scale)
{
    int h=blockIdx.x,d=threadIdx.x;
    if(d>=head_dim)return;
    int t=base_pos[0]+offset+1;
    int g=h/groups;
    float qv=q[(size_t)h*head_dim+d];
    const float* kbase=Kc+(size_t)g*max_ctx*head_dim;
    const float* vbase=Vc+(size_t)g*max_ctx*head_dim;
    __shared__ float red[32];
    __shared__ float s_score;
    float m=-3.0e38f,l=0.0f,acc=0.0f;
    for(int j=0;j<t;++j){
        float part=qv*kbase[(size_t)j*head_dim+d];
        for(int o=16;o>0;o>>=1)
            part+=__shfl_down_sync(0xffffffffu,part,o);
        int lane=d&31,warp=d>>5;
        if(lane==0)red[warp]=part;
        __syncthreads();
        if(d==0){
            float sum=0.0f;
            int nw=(head_dim+31)>>5;
            for(int w=0;w<nw;++w)sum+=red[w];
            s_score=sum*scale;
        }
        __syncthreads();
        float s=s_score;
        float m_new=fmaxf(m,s);
        float corr=__expf(m-m_new);
        float pp=__expf(s-m_new);
        l=l*corr+pp;
        acc=acc*corr+pp*vbase[(size_t)j*head_dim+d];
        m=m_new;
        __syncthreads();
    }
    out[(size_t)h*head_dim+d]=acc/l;
}

/* Production attn_decode_warp arithmetic under fixed graph grid.
   Dynamic real split count/chunk remain identical to eager. */
extern "C" __global__ void attn_decode_warp_f32_dp(
    const float* __restrict__ q,
    const float* __restrict__ Kc,
    const float* __restrict__ Vc,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    const int* __restrict__ base_pos,
    const int offset,
    const int head_dim,
    const int groups,
    const int max_ctx,
    const float scale,
    const int split_threshold,
    const int max_splits)
{
    int h=blockIdx.x,s=blockIdx.y;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int t=base_pos[0]+offset+1;
    int splits=(t+split_threshold-1)/split_threshold;
    splits=min(max_splits,max(1,splits));

    const int fixed4=gridDim.y<<2;
    size_t slot=((size_t)h*fixed4)+((size_t)s<<2)+warp;

    if(s>=splits){
        reinterpret_cast<float4*>(part_acc+slot*head_dim)[lane]
            =make_float4(0,0,0,0);
        if(lane==0){
            part_ml[slot*2+0]=-3.0e38f;
            part_ml[slot*2+1]=0.0f;
        }
        return;
    }

    int chunk=(t+splits-1)/splits;
    int j0=s*chunk,j1=min(t,j0+chunk);
    int g=h/groups;
    const float4 qv=reinterpret_cast<const float4*>(
        q+(size_t)h*head_dim)[lane];
    const float4* kb=reinterpret_cast<const float4*>(
        Kc+(size_t)g*max_ctx*head_dim);
    const float4* vb=reinterpret_cast<const float4*>(
        Vc+(size_t)g*max_ctx*head_dim);
    int vec_per_row=head_dim>>2;
    float m=-3.0e38f,l=0.0f;
    float a0=0,a1=0,a2=0,a3=0;

    for(int j=j0+warp;j<j1;j+=4){
        float4 k4=kb[(size_t)j*vec_per_row+lane];
        float part=qv.x*k4.x+qv.y*k4.y+qv.z*k4.z+qv.w*k4.w;
        for(int o=16;o>0;o>>=1)
            part+=__shfl_xor_sync(0xffffffffu,part,o);
        float sc=part*scale;
        float m_new=fmaxf(m,sc);
        float corr=__expf(m-m_new);
        float pp=__expf(sc-m_new);
        l=l*corr+pp;
        float4 v4=vb[(size_t)j*vec_per_row+lane];
        a0=a0*corr+pp*v4.x;
        a1=a1*corr+pp*v4.y;
        a2=a2*corr+pp*v4.z;
        a3=a3*corr+pp*v4.w;
        m=m_new;
    }
    reinterpret_cast<float4*>(part_acc+slot*head_dim)[lane]
        =make_float4(a0,a1,a2,a3);
    if(lane==0){
        part_ml[slot*2+0]=(j1>j0+warp)?m:-3.0e38f;
        part_ml[slot*2+1]=l;
    }
}

/* Same production combine arithmetic, fixed neutral slots included. */
extern "C" __global__ void attn_combine_f32_fixed(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    const int splits4,
    const int head_dim)
{
    int h=blockIdx.x,d=threadIdx.x;
    if(d>=head_dim)return;
    float m=-3.0e38f;
    for(int s=0;s<splits4;++s){
        float ms=part_ml[((size_t)h*splits4+s)*2+0];
        m=fmaxf(m,ms);
    }
    float l=0.0f,acc=0.0f;
    for(int s=0;s<splits4;++s){
        size_t base=(size_t)h*splits4+s;
        float ms=part_ml[base*2+0];
        float ls=part_ml[base*2+1];
        if(ls<=0.0f)continue;
        float w=__expf(ms-m);
        l+=ls*w;
        acc+=part_acc[base*head_dim+d]*w;
    }
    out[(size_t)h*head_dim+d]=(l>0.0f)?acc/l:0.0f;
}

extern "C" __global__ void pos_add4(int* p){if(threadIdx.x==0)p[0]+=4;}

extern "C" __global__ void argmax4_part(
    const float* __restrict__ logits,
    const int vocab,
    float* __restrict__ pmax,
    int* __restrict__ pidx,
    const int nparts)
{
    int row=blockIdx.y,partid=blockIdx.x,tid=threadIdx.x;
    int chunk=(vocab+nparts-1)/nparts;
    int lo=partid*chunk,hi=min(vocab,lo+chunk);
    float bv=-3.0e38f;int bi=0x7fffffff;
    const float* x=logits+(size_t)row*vocab;
    for(int i=lo+tid;i<hi;i+=blockDim.x){
        float v=x[i];
        if(v>bv || (v==bv && i<bi)){bv=v;bi=i;}
    }
    __shared__ float sm[256];__shared__ int si[256];
    sm[tid]=bv;si[tid]=bi;__syncthreads();
    for(int off=128;off>0;off>>=1){
        if(tid<off){
            float ov=sm[tid+off];int oi=si[tid+off];
            if(ov>sm[tid] || (ov==sm[tid] && oi<si[tid])){
                sm[tid]=ov;si[tid]=oi;
            }
        }
        __syncthreads();
    }
    if(tid==0){
        size_t z=(size_t)row*nparts+partid;
        pmax[z]=sm[0];pidx[z]=si[0];
    }
}

extern "C" __global__ void argmax4_final(
    const float* __restrict__ pmax,
    const int* __restrict__ pidx,
    const int nparts,
    int* __restrict__ out)
{
    int row=blockIdx.x,tid=threadIdx.x;
    float bv=-3.0e38f;int bi=0x7fffffff;
    const float* pm=pmax+(size_t)row*nparts;
    const int* pi=pidx+(size_t)row*nparts;
    for(int i=tid;i<nparts;i+=blockDim.x){
        float v=pm[i];int ix=pi[i];
        if(v>bv || (v==bv && ix<bi)){bv=v;bi=ix;}
    }
    __shared__ float sm[256];__shared__ int si[256];
    sm[tid]=bv;si[tid]=bi;__syncthreads();
    for(int off=128;off>0;off>>=1){
        if(tid<off){
            float ov=sm[tid+off];int oi=si[tid+off];
            if(ov>sm[tid] || (ov==sm[tid] && oi<si[tid]){
                sm[tid]=ov;si[tid]=oi;
            }
        }
        __syncthreads();
    }
    if(tid==0)out[row]=si[0];
}
"""

# Fix one generated-CUDA typo defensively before compilation.
_SOURCE=_SOURCE.replace(
    "if(ov>sm[tid] || (ov==sm[tid] && oi<si[tid]){",
    "if(ov>sm[tid] || (ov==sm[tid] && oi<si[tid])){"
)

class Phase22GraphKernels:
    SPLIT_THRESHOLD=512
    def __init__(self,cp,max_ctx):
        self.cp=cp
        self.max_splits=max(1,(int(max_ctx)+self.SPLIT_THRESHOLD-1)//self.SPLIT_THRESHOLD)
        names=(
          "embed4_bf16","kv_append_f32_dp","attn_decode_f32_dp",
          "attn_decode_warp_f32_dp","attn_combine_f32_fixed","pos_add4",
          "argmax4_part","argmax4_final",
        )
        self.mod=cp.RawModule(code=_SOURCE,options=("-std=c++14","--use_fast_math"),
                              name_expressions=names)
        self.f={n:self.mod.get_function(n) for n in names}

    def embed4(self,table_ptr,toks,h4,hidden):
        total=4*int(hidden)
        self.f["embed4_bf16"](((total+255)//256,),(256,),
            (np.uint64(table_ptr),toks,h4,np.int32(hidden)))

    def kv_write(self,cache,src,pos,offset,n_kv,head_dim,max_ctx):
        total=int(n_kv)*int(head_dim)
        self.f["kv_append_f32_dp"](((total+255)//256,),(256,),
            (cache,src,pos,np.int32(offset),np.int32(n_kv),
             np.int32(head_dim),np.int32(max_ctx)))

    def attention(self,out,q,K,V,pos,offset,n_heads,head_dim,groups,max_ctx,
                  scale,part_acc,part_ml):
        tmax=int(max_ctx)
        if tmax<=512:
            self.f["attn_decode_f32_dp"]((int(n_heads),),(int(head_dim),),
                (q,K,V,out,pos,np.int32(offset),np.int32(head_dim),
                 np.int32(groups),np.int32(max_ctx),np.float32(scale)))
        else:
            self.f["attn_decode_warp_f32_dp"](
                (int(n_heads),int(self.max_splits)),(128,),
                (q,K,V,part_acc,part_ml,pos,np.int32(offset),
                 np.int32(head_dim),np.int32(groups),np.int32(max_ctx),
                 np.float32(scale),np.int32(self.SPLIT_THRESHOLD),
                 np.int32(self.max_splits)))
            self.f["attn_combine_f32_fixed"](
                (int(n_heads),),(int(head_dim),),
                (part_acc,part_ml,out,np.int32(self.max_splits*4),
                 np.int32(head_dim)))

    def argmax4(self,logits,vocab,pmax,pidx,out,nparts=256):
        self.f["argmax4_part"]((int(nparts),4),(256,),
            (logits,np.int32(vocab),pmax,pidx,np.int32(nparts)))
        self.f["argmax4_final"]((4,),(256,),
            (pmax,pidx,np.int32(nparts),out))

    def add4pos(self,pos):
        self.f["pos_add4"]((1,),(1,),(pos,))
