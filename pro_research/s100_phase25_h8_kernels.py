from __future__ import annotations

import numpy as np

H=8
TOPK=6
ROUTES=H*TOPK
GROUPS=ROUTES
MAXM=H
LOW_WIDTH=16
LOW_VIRTUAL=16
LOW_ROWS_PER_BLOCK=16
HIGH_WIDTH=32
HIGH_VIRTUAL=8
HIGH_ROWS_PER_BLOCK=8

HEAD=r"""
#define H 8
#define TOPK 6
#define ROUTES 48
#define GROUPS 48
#define MAXM 8

extern "C" __global__ void cache_assign_h8(
    int* slot_of,int* expert_of,int* last_used,int* state2,
    const int* ids,int* slots,int* need,int* stats2,
    const int cap)
{
    if(threadIdx.x!=0)return;
    int tick=state2[0],filled=state2[1];
    for(int r=0;r<ROUTES;++r){
        int e=ids[r],sl=slot_of[e];
        if(sl>=0){
            last_used[sl]=++tick;slots[r]=sl;need[r]=0;stats2[0]+=1;
        }else{
            stats2[1]+=1;int v;
            if(filled<cap)v=filled++;
            else{
                v=0;int mn=last_used[0];
                for(int c=1;c<cap;++c)if(last_used[c]<mn){mn=last_used[c];v=c;}
                int old=expert_of[v];
                if(old>=0)slot_of[old]=-1;
            }
            slot_of[e]=v;expert_of[v]=e;last_used[v]=++tick;
            slots[r]=v;need[r]=1;
        }
    }
    state2[0]=tick;state2[1]=filled;
}

extern "C" __global__ void group_routes48(
    const int* ids,int* route_group,int* group_ids,int* group_count,
    int* group_refs,int* ngroups)
{
    if(threadIdx.x!=0)return;
    for(int g=0;g<GROUPS;++g){
        group_ids[g]=-1;group_count[g]=0;
        for(int m=0;m<MAXM;++m)group_refs[g*MAXM+m]=-1;
    }
    int ng=0;
    for(int r=0;r<ROUTES;++r){
        int e=ids[r],g=-1;
        for(int q=0;q<ng;++q)if(group_ids[q]==e){g=q;break;}
        if(g<0){g=ng++;group_ids[g]=e;}
        int m=group_count[g]++;
        if(m>=MAXM){group_count[g]=99;continue;}
        group_refs[g*MAXM+m]=r;route_group[r]=g;
    }
    *ngroups=ng;
}
"""


def low_up_kernel(M:int)->str:
    return f"""
extern "C" __global__ void grouped_up_h8_m{M}(
    const unsigned char* codes_base,const unsigned char* scales_base,
    const int* slots,const int* ids,const float* globals,
    const int* group_count,const int* group_refs,
    const float* e2,const float* e4,const float* x8,float* route_act,
    const int rows,const int cols,const size_t code_stride,const size_t scale_stride)
{{
    int g=blockIdx.y;
    if(g>=GROUPS || group_count[g]!={M})return;
    int anchor=group_refs[g*MAXM+0];
    if(anchor<0)return;
    int slot=slots[anchor],e=ids[anchor];
    const unsigned char* codes=codes_base+(size_t)slot*code_stride;
    const unsigned char* scales=scales_base+(size_t)slot*scale_stride;
    float gs=globals[e*2+1];

    extern __shared__ float sx[];
    int refs[{M}];
    #pragma unroll
    for(int m=0;m<{M};++m)refs[m]=group_refs[g*MAXM+m];
    for(int i=threadIdx.x;i<{M}*cols;i+=blockDim.x){{
        int m=i/cols,k=i-m*cols,token=refs[m]/TOPK;
        sx[i]=x8[(size_t)token*cols+k];
    }}
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2[threadIdx.x];
    __syncthreads();

    int lane=threadIdx.x&15,sub=threadIdx.x>>4;
    int row=blockIdx.x*16+sub;
    if(row>=rows)return;
    int nbytes=cols>>1,nvec=nbytes>>2;
    const unsigned char* crow=codes+(size_t)row*nbytes;
    const unsigned char* srow=scales+(size_t)row*(cols>>4);
    const uchar4* c4=reinterpret_cast<const uchar4*>(crow);

    float part[{M}][16];
    #pragma unroll
    for(int m=0;m<{M};++m)for(int vi=0;vi<16;++vi)part[m][vi]=0.0f;

    #pragma unroll
    for(int vi=0;vi<16;++vi){{
        int tid=lane+16*vi;
        float a[{M}];
        #pragma unroll
        for(int m=0;m<{M};++m)a[m]=0.0f;
        for(int v=tid;v<nvec;v+=256){{
            uchar4 q=c4[v];int b=v<<2,k=b<<1;
            float sc=e4[srow[b>>3]]*gs;
            float w[8]={{lut[q.x&15]*sc,lut[q.x>>4]*sc,lut[q.y&15]*sc,lut[q.y>>4]*sc,
                        lut[q.z&15]*sc,lut[q.z>>4]*sc,lut[q.w&15]*sc,lut[q.w>>4]*sc}};
            #pragma unroll
            for(int m=0;m<{M};++m){{
                const float* xm=sx+(size_t)m*cols;
                float z=a[m];
                #pragma unroll
                for(int j=0;j<8;++j)z=fmaf(w[j],xm[k+j],z);
                a[m]=z;
            }}
        }}
        for(int b=(nvec<<2)+tid;b<nbytes;b+=256){{
            unsigned char q=crow[b];float sc=e4[srow[b>>3]]*gs;int k=b<<1;
            float w0=lut[q&15]*sc,w1=lut[q>>4]*sc;
            #pragma unroll
            for(int m=0;m<{M};++m){{
                const float* xm=sx+(size_t)m*cols;
                a[m]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],a[m]));
            }}
        }}
        #pragma unroll
        for(int m=0;m<{M};++m)part[m][vi]=a[m];
    }}

    #pragma unroll
    for(int m=0;m<{M};++m){{
        float s8[8];
        #pragma unroll
        for(int w=0;w<8;++w){{
            float v=part[m][w*2+0]+part[m][w*2+1];
            for(int off=8;off>0;off>>=1)v+=__shfl_down_sync(0xffffffffu,v,off,16);
            s8[w]=v;
        }}
        if(lane==0){{
            float t0=s8[0]+s8[4],t1=s8[1]+s8[5],t2=s8[2]+s8[6],t3=s8[3]+s8[7];
            float v=(t0+t2)+(t1+t3);
            float rr=fmaxf(v,0.0f);v=rr*rr;
            route_act[(size_t)refs[m]*rows+row]=v;
        }}
    }}
}}
"""


def high_up_kernel(M:int)->str:
    return f"""
extern "C" __global__ void grouped_up_h8_m{M}(
    const unsigned char* codes_base,const unsigned char* scales_base,
    const int* slots,const int* ids,const float* globals,
    const int* group_count,const int* group_refs,
    const float* e2,const float* e4,const float* x8,float* route_act,
    const int rows,const int cols,const size_t code_stride,const size_t scale_stride)
{{
    int g=blockIdx.y;
    if(g>=GROUPS || group_count[g]!={M})return;
    int anchor=group_refs[g*MAXM+0];
    if(anchor<0)return;
    int slot=slots[anchor],e=ids[anchor];
    const unsigned char* codes=codes_base+(size_t)slot*code_stride;
    const unsigned char* scales=scales_base+(size_t)slot*scale_stride;
    float gs=globals[e*2+1];
    int refs[{M}];
    #pragma unroll
    for(int m=0;m<{M};++m)refs[m]=group_refs[g*MAXM+m];
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2[threadIdx.x];
    __syncthreads();

    int lane=threadIdx.x&31,sub=threadIdx.x>>5;
    int row=blockIdx.x*8+sub;
    if(row>=rows)return;
    int nbytes=cols>>1,nvec=nbytes>>2;
    const unsigned char* crow=codes+(size_t)row*nbytes;
    const unsigned char* srow=scales+(size_t)row*(cols>>4);
    const uchar4* c4=reinterpret_cast<const uchar4*>(crow);

    float part[{M}][8];
    #pragma unroll
    for(int m=0;m<{M};++m)for(int vi=0;vi<8;++vi)part[m][vi]=0.0f;

    #pragma unroll
    for(int vi=0;vi<8;++vi){{
        int tid=lane+32*vi;
        float a[{M}];
        #pragma unroll
        for(int m=0;m<{M};++m)a[m]=0.0f;
        for(int v=tid;v<nvec;v+=256){{
            uchar4 q=c4[v];int b=v<<2,k=b<<1;
            float sc=e4[srow[b>>3]]*gs;
            float w[8]={{lut[q.x&15]*sc,lut[q.x>>4]*sc,lut[q.y&15]*sc,lut[q.y>>4]*sc,
                        lut[q.z&15]*sc,lut[q.z>>4]*sc,lut[q.w&15]*sc,lut[q.w>>4]*sc}};
            #pragma unroll
            for(int m=0;m<{M};++m){{
                int token=refs[m]/TOPK;
                const float* xm=x8+(size_t)token*cols;
                float z=a[m];
                #pragma unroll
                for(int j=0;j<8;++j)z=fmaf(w[j],xm[k+j],z);
                a[m]=z;
            }}
        }}
        for(int b=(nvec<<2)+tid;b<nbytes;b+=256){{
            unsigned char q=crow[b];float sc=e4[srow[b>>3]]*gs;int k=b<<1;
            float w0=lut[q&15]*sc,w1=lut[q>>4]*sc;
            #pragma unroll
            for(int m=0;m<{M};++m){{
                int token=refs[m]/TOPK;
                const float* xm=x8+(size_t)token*cols;
                a[m]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],a[m]));
            }}
        }}
        #pragma unroll
        for(int m=0;m<{M};++m)part[m][vi]=a[m];
    }}

    #pragma unroll
    for(int m=0;m<{M};++m){{
        float s8[8];
        #pragma unroll
        for(int vi=0;vi<8;++vi){{
            float v=part[m][vi];
            for(int off=16;off>0;off>>=1)v+=__shfl_down_sync(0xffffffffu,v,off,32);
            s8[vi]=v;
        }}
        if(lane==0){{
            float t0=s8[0]+s8[4],t1=s8[1]+s8[5],t2=s8[2]+s8[6],t3=s8[3]+s8[7];
            float v=(t0+t2)+(t1+t3);
            float rr=fmaxf(v,0.0f);v=rr*rr;
            route_act[(size_t)refs[m]*rows+row]=v;
        }}
    }}
}}
"""

SPLIT4=r"""
extern "C" __global__ void grouped_up_h8_split4(
    const unsigned char* codes_base,const unsigned char* scales_base,
    const int* slots,const int* ids,const float* globals,
    const int* group_count,const int* group_refs,
    const float* e2,const float* e4,const float* x8,float* route_act,
    const int rows,const int cols,const size_t code_stride,const size_t scale_stride,
    const int start)
{
    int g=blockIdx.y,cnt=(g<GROUPS)?group_count[g]:0;
    if(cnt<=start || cnt>MAXM)return;
    int mcount=min(4,cnt-start);
    int anchor=group_refs[g*MAXM+start];
    if(anchor<0)return;
    int slot=slots[anchor],e=ids[anchor];
    const unsigned char* codes=codes_base+(size_t)slot*code_stride;
    const unsigned char* scales=scales_base+(size_t)slot*scale_stride;
    float gs=globals[e*2+1];
    int refs[4]={-1,-1,-1,-1};
    for(int m=0;m<mcount;++m)refs[m]=group_refs[g*MAXM+start+m];

    extern __shared__ float sx[];
    for(int i=threadIdx.x;i<4*cols;i+=blockDim.x){
        int m=i/cols,k=i-m*cols;
        sx[i]=(m<mcount)?x8[(size_t)(refs[m]/TOPK)*cols+k]:0.0f;
    }
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2[threadIdx.x];
    __syncthreads();
    int lane=threadIdx.x&15,sub=threadIdx.x>>4;
    int row=blockIdx.x*16+sub;
    if(row>=rows)return;
    int nbytes=cols>>1,nvec=nbytes>>2;
    const unsigned char* crow=codes+(size_t)row*nbytes;
    const unsigned char* srow=scales+(size_t)row*(cols>>4);
    const uchar4* c4=reinterpret_cast<const uchar4*>(crow);
    float part[4][16];
    #pragma unroll
    for(int m=0;m<4;++m)for(int vi=0;vi<16;++vi)part[m][vi]=0.0f;
    #pragma unroll
    for(int vi=0;vi<16;++vi){
        int tid=lane+16*vi;float a[4]={0,0,0,0};
        for(int v=tid;v<nvec;v+=256){
            uchar4 q=c4[v];int b=v<<2,k=b<<1;
            float sc=e4[srow[b>>3]]*gs;
            float w[8]={lut[q.x&15]*sc,lut[q.x>>4]*sc,lut[q.y&15]*sc,lut[q.y>>4]*sc,
                        lut[q.z&15]*sc,lut[q.z>>4]*sc,lut[q.w&15]*sc,lut[q.w>>4]*sc};
            #pragma unroll
            for(int m=0;m<4;++m){
                const float* xm=sx+(size_t)m*cols;float z=a[m];
                #pragma unroll
                for(int j=0;j<8;++j)z=fmaf(w[j],xm[k+j],z);
                a[m]=z;
            }
        }
        for(int b=(nvec<<2)+tid;b<nbytes;b+=256){
            unsigned char q=crow[b];float sc=e4[srow[b>>3]]*gs;int k=b<<1;
            float w0=lut[q&15]*sc,w1=lut[q>>4]*sc;
            #pragma unroll
            for(int m=0;m<4;++m){
                const float* xm=sx+(size_t)m*cols;
                a[m]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],a[m]));
            }
        }
        #pragma unroll
        for(int m=0;m<4;++m)part[m][vi]=a[m];
    }
    for(int m=0;m<mcount;++m){
        float s8[8];
        #pragma unroll
        for(int w=0;w<8;++w){
            float v=part[m][w*2]+part[m][w*2+1];
            for(int off=8;off>0;off>>=1)v+=__shfl_down_sync(0xffffffffu,v,off,16);
            s8[w]=v;
        }
        if(lane==0){
            float t0=s8[0]+s8[4],t1=s8[1]+s8[5],t2=s8[2]+s8[6],t3=s8[3]+s8[7];
            float v=(t0+t2)+(t1+t3);float rr=fmaxf(v,0.0f);v=rr*rr;
            route_act[(size_t)refs[m]*rows+row]=v;
        }
    }
}
"""

TAIL=r"""
extern "C" __global__ void scan_group_masks_h8(
    const float* act,const int* group_count,const int* group_refs,
    unsigned int* route_masks,int* route_plist,int* route_pcount,
    unsigned int* union_masks,int* union_plist,int* union_pcount,
    int* union_nz,int* union_nzc,const int inter)
{
    int g=blockIdx.x,cnt=group_count[g],np=inter>>4;
    if(cnt<=0 || cnt>MAXM)return;
    for(int p=threadIdx.x;p<np;p+=blockDim.x){
        union_masks[g*np+p]=0;
        for(int m=0;m<cnt;++m)route_masks[group_refs[g*MAXM+m]*np+p]=0;
    }
    if(threadIdx.x==0){
        union_pcount[g]=0;union_nzc[g]=0;
        for(int m=0;m<cnt;++m)route_pcount[group_refs[g*MAXM+m]]=0;
    }
    __syncthreads();
    for(int j=threadIdx.x;j<inter;j+=blockDim.x){
        bool any=false;int p=j>>4;unsigned bit=1u<<(j&15);
        for(int m=0;m<cnt;++m){
            int r=group_refs[g*MAXM+m];
            if(act[(size_t)r*inter+j]!=0.0f){atomicOr(&route_masks[r*np+p],bit);any=true;}
        }
        if(any)atomicOr(&union_masks[g*np+p],bit);
    }
    __syncthreads();
    if(threadIdx.x==0){
        int up=0,un=0;
        for(int p=0;p<np;++p){
            unsigned mk=union_masks[g*np+p];
            if(mk){
                union_plist[g*np+up++]=p;
                for(int c=0;c<16;++c)if(mk&(1u<<c))union_nz[g*inter+un++]=(p<<4)+c;
            }
        }
        union_pcount[g]=up;union_nzc[g]=un;
        for(int m=0;m<cnt;++m){
            int r=group_refs[g*MAXM+m],rp=0;
            for(int p=0;p<np;++p)if(route_masks[r*np+p])route_plist[r*np+rp++]=p;
            route_pcount[r]=rp;
        }
    }
}

extern "C" __global__ void down_routes_partial_sres_h8(
    const unsigned char* mirrors,const unsigned char* planes,
    const int* slots,const int* ids,const int* route_group,
    const float* globals,const float* act,const int* route_plist,
    const unsigned int* route_masks,const int* route_pcount,
    const float* e2,const float* e4,float* partials,
    const size_t panel_bytes,const size_t plane_bytes,
    const int rows,const int inter,const int nchunks)
{
    int y=blockIdx.y,r=y/nchunks,chunk=y-r*nchunks;
    if(r>=ROUTES)return;
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    int g=route_group[r],e=ids[r],slot=slots[r];
    int np=inter>>4,rowhalf=rows>>1;
    const unsigned char* bank=mirrors+(size_t)g*panel_bytes;
    const unsigned char* plane=planes+(size_t)slot*plane_bytes;
    __shared__ float l2[16],l4[256];
    if(threadIdx.x<16)l2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256)l4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();
    int hb=row>>1,hi=row&1,pc=route_pcount[r];
    size_t ps=(size_t)rows+16u*(size_t)rowhalf;
    float acc=0.0f,gs=globals[e*2+0];
    for(int pi=chunk;pi<pc;pi+=nchunks){
        int p=route_plist[r*np+pi];
        const unsigned char* pb=bank+(size_t)p*ps;
        float sc=l4[plane[(size_t)p*rows+row]]*gs;
        const unsigned char* codes=pb+rows;
        unsigned mk=route_masks[r*np+p];
        while(mk){
            int c=__ffs(mk)-1;mk&=mk-1;
            unsigned char q=codes[(size_t)c*rowhalf+hb];
            float w=l2[hi?(q>>4):(q&15)]*sc;
            acc=fmaf(w,act[(size_t)r*inter+(p<<4)+c],acc);
        }
    }
    partials[((size_t)r*nchunks+chunk)*rows+row]=acc;
}

extern "C" __global__ void down_grouped_sres_h8(
    const unsigned char* mirrors,const unsigned char* planes,
    const int* slots,const int* ids,const int* group_count,const int* group_refs,
    const float* globals,const float* act,const unsigned int* route_masks,
    const int* union_nz,const int* union_nzc,
    const float* e2,const float* e4,float* down,
    const size_t panel_bytes,const size_t plane_bytes,
    const int rows,const int inter)
{
    int g=blockIdx.y,cnt=(g<GROUPS)?group_count[g]:0;
    if(cnt<=0 || cnt>MAXM)return;
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    int refs[MAXM];
    #pragma unroll
    for(int m=0;m<MAXM;++m)refs[m]=(m<cnt)?group_refs[g*MAXM+m]:-1;
    int anchor=refs[0],e=ids[anchor],slot=slots[anchor];
    int np=inter>>4,rowhalf=rows>>1;
    const unsigned char* bank=mirrors+(size_t)g*panel_bytes;
    const unsigned char* plane=planes+(size_t)slot*plane_bytes;
    __shared__ float l2[16],l4[256];
    if(threadIdx.x<16)l2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256)l4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();
    int hb=row>>1,hi=row&1;
    size_t ps=(size_t)rows+16u*(size_t)rowhalf;
    float acc[MAXM];
    #pragma unroll
    for(int m=0;m<MAXM;++m)acc[m]=0.0f;
    float gs=globals[e*2+0];
    int un=union_nzc[g];
    for(int qi=0;qi<un;++qi){
        int j=union_nz[g*inter+qi],p=j>>4,c=j&15;
        const unsigned char* pb=bank+(size_t)p*ps;
        float sc=l4[plane[(size_t)p*rows+row]]*gs;
        unsigned char q=(pb+rows)[(size_t)c*rowhalf+hb];
        float w=l2[hi?(q>>4):(q&15)]*sc;
        unsigned bit=1u<<c;
        #pragma unroll
        for(int m=0;m<MAXM;++m){
            if(m<cnt){
                int r=refs[m];
                if(route_masks[r*np+p]&bit)
                    acc[m]=fmaf(w,act[(size_t)r*inter+j],acc[m]);
            }
        }
    }
    #pragma unroll
    for(int m=0;m<MAXM;++m)if(m<cnt)down[(size_t)refs[m]*rows+row]=acc[m];
}

extern "C" __global__ void reduce_routes_h8(
    const float* partials,float* down,const int rows,const int nchunks)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(row>=rows || r>=ROUTES)return;
    float a=0.0f;
    for(int c=0;c<nchunks;++c)a+=partials[((size_t)r*nchunks+c)*rows+row];
    down[(size_t)r*rows+row]=a;
}

extern "C" __global__ void accumulate_h8(
    float* out,const float* down,const float* w,const int hidden)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    if(i>=hidden || t>=H)return;
    float a=out[(size_t)t*hidden+i];
    #pragma unroll
    for(int s=0;s<TOPK;++s){
        int r=t*TOPK+s;
        a=fmaf(down[(size_t)r*hidden+i],w[r],a);
    }
    out[(size_t)t*hidden+i]=a;
}
"""

SOURCE=HEAD+"".join(low_up_kernel(m) for m in range(1,5))+"".join(high_up_kernel(m) for m in range(5,9))+SPLIT4+TAIL


class H8GroupedKernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        names=["cache_assign_h8","group_routes48","grouped_up_h8_split4"]
        names += [f"grouped_up_h8_m{m}" for m in range(1,9)]
        names += ["scan_group_masks_h8","down_routes_partial_sres_h8",
                  "down_grouped_sres_h8","reduce_routes_h8","accumulate_h8"]
        self.mod=cp.RawModule(code=SOURCE,options=("-std=c++14",),name_expressions=names)
        self.f={n:self.mod.get_function(n) for n in names}

    def cache_assign(self,dev,ids,slots,need,cap):
        self.f["cache_assign_h8"]((1,),(32,),
            (dev["slot_of"],dev["expert_of"],dev["last_used"],dev["state2"],
             ids,slots,need,dev["stats2"],np.int32(cap)))

    def group(self,ids,route_group,group_ids,group_count,group_refs,ngroups):
        self.f["group_routes48"]((1,),(32,),
            (ids,route_group,group_ids,group_count,group_refs,ngroups))

    def up_direct(self,m,cache_c,cache_s,slots,ids,globals_dev,e2,e4,normed,
                  route_act,rows,cols,code_stride,scale_stride,group_count,group_refs):
        m=int(m)
        fn=self.f[f"grouped_up_h8_m{m}"]
        if m<=4:
            rpb=LOW_ROWS_PER_BLOCK; sh=m*int(cols)*4
            if sh>48*1024: fn.max_dynamic_shared_size_bytes=sh
        else:
            rpb=HIGH_ROWS_PER_BLOCK; sh=0
        fn(((int(rows)+rpb-1)//rpb,GROUPS),(256,),
           (cache_c,cache_s,slots,ids,globals_dev,group_count,group_refs,e2,e4,
            normed,route_act,np.int32(rows),np.int32(cols),
            np.uint64(code_stride),np.uint64(scale_stride)),shared_mem=sh)

    def up_split4(self,cache_c,cache_s,slots,ids,globals_dev,e2,e4,normed,
                  route_act,rows,cols,code_stride,scale_stride,group_count,group_refs):
        fn=self.f["grouped_up_h8_split4"]
        sh=4*int(cols)*4
        if sh>48*1024: fn.max_dynamic_shared_size_bytes=sh
        grid=((int(rows)+15)//16,GROUPS)
        common=(cache_c,cache_s,slots,ids,globals_dev,group_count,group_refs,e2,e4,
                normed,route_act,np.int32(rows),np.int32(cols),
                np.uint64(code_stride),np.uint64(scale_stride))
        fn(grid,(256,),common+(np.int32(0),),shared_mem=sh)
        fn(grid,(256,),common+(np.int32(4),),shared_mem=sh)

    def scan(self,act,group_count,group_refs,route_masks,route_plist,route_pcount,
             union_masks,union_plist,union_pcount,union_nz,union_nzc,inter):
        self.f["scan_group_masks_h8"]((GROUPS,),(256,),
            (act,group_count,group_refs,route_masks,route_plist,route_pcount,
             union_masks,union_plist,union_pcount,union_nz,union_nzc,np.int32(inter)))

    def down_route(self,mirrors,planes,slots,ids,route_group,globals_dev,act,
                   route_plist,route_masks,route_pcount,e2,e4,partials,
                   panel_bytes,plane_bytes,rows,inter,nchunks):
        self.f["down_routes_partial_sres_h8"](
            ((int(rows)+127)//128,ROUTES*int(nchunks)),(128,),
            (mirrors,planes,slots,ids,route_group,globals_dev,act,route_plist,
             route_masks,route_pcount,e2,e4,partials,np.uint64(panel_bytes),
             np.uint64(plane_bytes),np.int32(rows),np.int32(inter),np.int32(nchunks)))

    def down_grouped(self,mirrors,planes,slots,ids,group_count,group_refs,
                     globals_dev,act,route_masks,union_nz,union_nzc,e2,e4,down,
                     panel_bytes,plane_bytes,rows,inter):
        self.f["down_grouped_sres_h8"](
            ((int(rows)+127)//128,GROUPS),(128,),
            (mirrors,planes,slots,ids,group_count,group_refs,globals_dev,act,
             route_masks,union_nz,union_nzc,e2,e4,down,np.uint64(panel_bytes),
             np.uint64(plane_bytes),np.int32(rows),np.int32(inter)))

    def reduce(self,partials,down,rows,nchunks):
        self.f["reduce_routes_h8"](((int(rows)+255)//256,ROUTES),(256,),
            (partials,down,np.int32(rows),np.int32(nchunks)))

    def accumulate(self,out,down,w,hidden):
        self.f["accumulate_h8"](((int(hidden)+255)//256,H),(256,),
            (out,down,w,np.int32(hidden)))
