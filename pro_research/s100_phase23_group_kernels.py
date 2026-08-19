from __future__ import annotations
import numpy as np

H=4
TOPK=6
ROUTES=24
GROUPS=24
MAXM=4
WIDTH=16
VIRTUAL=16
ROWS_PER_BLOCK=16

HEAD=r"""
#define ROUTES 24
#define GROUPS 24
#define TOPK 6
#define MAXM 4
#define WIDTH 16
#define VIRTUAL 16
#define ROWS_PER_BLOCK 16

extern "C" __global__ void cache_assign_h4(
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
                slot_of[expert_of[v]]=-1;
            }
            slot_of[e]=v;expert_of[v]=e;last_used[v]=++tick;
            slots[r]=v;need[r]=1;
        }
    }
    state2[0]=tick;state2[1]=filled;
}

extern "C" __global__ void group_routes24(
    const int* ids,int* route_group,int* group_ids,int* group_count,
    int* group_refs,int* ngroups)
{
    if(threadIdx.x!=0)return;
    for(int g=0;g<GROUPS;++g){group_ids[g]=-1;group_count[g]=0;
        for(int m=0;m<MAXM;++m)group_refs[g*MAXM+m]=-1;}
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

def up_kernel(M):
    return f"""
extern "C" __global__ void grouped_up_m{M}(
    const unsigned char* codes_base,const unsigned char* scales_base,
    const int* slots,const int* ids,const float* globals,
    const int* group_count,const int* group_refs,
    const float* e2,const float* e4,const float* x4,float* route_act,
    const int rows,const int cols,const size_t code_stride,const size_t scale_stride)
{{
    int anchor=blockIdx.y;
    int g=-1;
    for(int q=0;q<GROUPS;++q)
        if(group_refs[q*MAXM]==anchor){{g=q;break;}}
    if(g<0 || group_count[g]!={M})return;

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
        sx[i]=x4[(size_t)token*cols+k];
    }}
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2[threadIdx.x];
    __syncthreads();

    int lane=threadIdx.x&(WIDTH-1),sub=threadIdx.x/WIDTH;
    int row=blockIdx.x*ROWS_PER_BLOCK+sub;
    if(row>=rows)return;
    int nbytes=cols>>1,nvec=nbytes>>2;
    const unsigned char* crow=codes+(size_t)row*nbytes;
    const unsigned char* srow=scales+(size_t)row*(cols>>4);
    const uchar4* c4=reinterpret_cast<const uchar4*>(crow);

    float part[{M}][VIRTUAL];
    #pragma unroll
    for(int m=0;m<{M};++m)for(int vi=0;vi<VIRTUAL;++vi)part[m][vi]=0.0f;

    #pragma unroll
    for(int vi=0;vi<VIRTUAL;++vi){{
        int tid=lane+WIDTH*vi;
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
            float loc0=part[m][w*2+0];
            float loc1=part[m][w*2+1];
            float v=loc0+loc1;
            for(int off=WIDTH>>1;off>0;off>>=1)
                v+=__shfl_down_sync(0xffffffffu,v,off,WIDTH);
            s8[w]=v;
        }}
        if(lane==0){{
            float t0=s8[0]+s8[4],t1=s8[1]+s8[5],t2=s8[2]+s8[6],t3=s8[3]+s8[7];
            float v=(t0+t2)+(t1+t3);
            float rr=fmaxf(v,0.0f);v=rr*rr;
            int ref=refs[m];
            route_act[(size_t)ref*rows+row]=v;
        }}
    }}
}}
"""

TAIL=r"""
extern "C" __global__ void scan_group_masks(
    const float* act,const int* group_count,const int* group_refs,
    unsigned int* route_masks,int* route_plist,int* route_pcount,
    unsigned int* union_masks,int* union_plist,int* union_pcount,
    int* union_nz,int* union_nzc,const int inter)
{
    int g=blockIdx.x,cnt=group_count[g],np=inter>>4;
    if(cnt<=0 || cnt>4)return;
    for(int p=threadIdx.x;p<np;p+=blockDim.x){
        union_masks[g*np+p]=0;
        for(int m=0;m<cnt;++m)route_masks[group_refs[g*4+m]*np+p]=0;
    }
    if(threadIdx.x==0){
        union_pcount[g]=0;union_nzc[g]=0;
        for(int m=0;m<cnt;++m)route_pcount[group_refs[g*4+m]]=0;
    }
    __syncthreads();
    for(int j=threadIdx.x;j<inter;j+=blockDim.x){
        bool any=false;int p=j>>4;unsigned bit=1u<<(j&15);
        for(int m=0;m<cnt;++m){
            int r=group_refs[g*4+m];
            if(act[(size_t)r*inter+j]!=0.0f){atomicOr(&route_masks[r*np+p],bit);any=true;}
        }
        if(any)atomicOr(&union_masks[g*np+p],bit);
    }
    __syncthreads();
    if(threadIdx.x==0){
        int up=0,un=0;
        for(int p=0;p<np;++p){
            unsigned mk=union_masks[g*np+p];
            if(mk){union_plist[g*np+up++]=p;
                for(int c=0;c<16;++c)if(mk&(1u<<c))union_nz[g*inter+un++]=(p<<4)+c;}
        }
        union_pcount[g]=up;union_nzc[g]=un;
        for(int m=0;m<cnt;++m){
            int r=group_refs[g*4+m],rp=0;
            for(int p=0;p<np;++p)if(route_masks[r*np+p])route_plist[r*np+rp++]=p;
            route_pcount[r]=rp;
        }
    }
}

extern "C" __global__ void gather_group_union(
    const unsigned char* down_base,const int* group_ids,const int* group_count,
    const int* union_plist,const int* union_pcount,const int* union_nz,const int* union_nzc,
    unsigned char* mirrors,const size_t panel_bytes,const int rows,const int inter)
{
    int g=blockIdx.x;if(group_count[g]<=0 || group_count[g]>4)return;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int task=blockIdx.y*8+warp,stride=gridDim.y*8;
    int ncol=union_nzc[g],pc=union_pcount[g],rowhalf=rows>>1,np=inter>>4;
    size_t ps=(size_t)rows+16u*(size_t)rowhalf;
    const unsigned char* src=down_base+(size_t)group_ids[g]*panel_bytes;
    unsigned char* dst=mirrors+(size_t)g*panel_bytes;
    for(int q=task;q<ncol+pc;q+=stride){
        size_t off;int bytes;
        if(q<ncol){
            int j=union_nz[g*inter+q];
            off=(size_t)(j>>4)*ps+rows+(size_t)(j&15)*rowhalf;bytes=rowhalf;
        }else{
            int p=union_plist[g*np+(q-ncol)];
            off=(size_t)p*ps;bytes=rows;
        }
        const uchar4* s=reinterpret_cast<const uchar4*>(src+off);
        uchar4* d=reinterpret_cast<uchar4*>(dst+off);
        for(int k=lane;k<bytes/4;k+=32)d[k]=s[k];
    }
}

extern "C" __global__ void down_routes_partial(
    const unsigned char* mirrors,const int* ids,const int* route_group,
    const float* globals,const float* act,const int* route_plist,
    const unsigned int* route_masks,const int* route_pcount,
    const float* e2,const float* e4,float* partials,
    const size_t panel_bytes,const int rows,const int inter,const int nchunks)
{
    int y=blockIdx.y,r=y/nchunks,chunk=y-r*nchunks;
    if(r>=ROUTES)return;
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    int g=route_group[r],e=ids[r],np=inter>>4,rowhalf=rows>>1;
    const unsigned char* bank=mirrors+(size_t)g*panel_bytes;
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
        float sc=l4[pb[row]]*gs;
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

extern "C" __global__ void reduce_routes(
    const float* partials,float* down,const int rows,const int nchunks)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(row>=rows || r>=ROUTES)return;
    float a=0.0f;
    for(int c=0;c<nchunks;++c)a+=partials[((size_t)r*nchunks+c)*rows+row];
    down[(size_t)r*rows+row]=a;
}

extern "C" __global__ void accumulate_h4(
    float* out,const float* down,const float* w,const int hidden)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    if(i>=hidden || t>=4)return;
    float a=out[(size_t)t*hidden+i];
    #pragma unroll
    for(int s=0;s<TOPK;++s){
        int r=t*TOPK+s;
        a=fmaf(down[(size_t)r*hidden+i],w[r],a);
    }
    out[(size_t)t*hidden+i]=a;
}
"""

SOURCE=HEAD+"".join(up_kernel(m) for m in (1,2,3,4))+TAIL

class Phase23Kernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        names=["cache_assign_h4","group_routes24"]+[f"grouped_up_m{m}" for m in (1,2,3,4)]+[
            "scan_group_masks","gather_group_union","down_routes_partial",
            "reduce_routes","accumulate_h4"]
        self.mod=cp.RawModule(code=SOURCE,options=("-std=c++14",),name_expressions=names)
        self.f={n:self.mod.get_function(n) for n in names}

    def cache_assign(self,dev,ids,slots,need,cap):
        self.f["cache_assign_h4"]((1,),(32,),
            (dev["slot_of"],dev["expert_of"],dev["last_used"],dev["state2"],
             ids,slots,need,dev["stats2"],np.int32(cap)))

    def group(self,ids,route_group,group_ids,group_count,group_refs,ngroups):
        self.f["group_routes24"]((1,),(32,),
            (ids,route_group,group_ids,group_count,group_refs,ngroups))

    def up(self,m,cache_c,cache_s,slots,ids,globals_dev,e2,e4,normed,route_act,
           rows,cols,code_stride,scale_stride):
        rpb=16
        sh=int(m)*int(cols)*4
        fn=self.f[f"grouped_up_m{m}"]
        if sh>48*1024: fn.max_dynamic_shared_size_bytes=sh
        fn(((int(rows)+rpb-1)//rpb,ROUTES),(256,),
           (cache_c,cache_s,slots,ids,globals_dev,
            self.group_count,self.group_refs,e2,e4,normed,route_act,
            np.int32(rows),np.int32(cols),np.uint64(code_stride),np.uint64(scale_stride)),
           shared_mem=sh)

    def bind_group_arrays(self,group_count,group_refs):
        self.group_count=group_count;self.group_refs=group_refs

    def scan(self,act,group_count,group_refs,route_masks,route_plist,route_pcount,
             union_masks,union_plist,union_pcount,union_nz,union_nzc,inter):
        self.f["scan_group_masks"]((GROUPS,),(256,),
            (act,group_count,group_refs,route_masks,route_plist,route_pcount,
             union_masks,union_plist,union_pcount,union_nz,union_nzc,np.int32(inter)))

    def gather(self,down_base,group_ids,group_count,union_plist,union_pcount,
               union_nz,union_nzc,mirrors,panel_bytes,rows,inter):
        self.f["gather_group_union"]((GROUPS,32),(256,),
            (np.uint64(down_base),group_ids,group_count,union_plist,union_pcount,
             union_nz,union_nzc,mirrors,np.uint64(panel_bytes),
             np.int32(rows),np.int32(inter)))

    def down(self,mirrors,ids,route_group,globals_dev,act,route_plist,route_masks,
             route_pcount,e2,e4,partials,panel_bytes,rows,inter,nchunks):
        self.f["down_routes_partial"](
            ((int(rows)+127)//128,ROUTES*int(nchunks)),(128,),
            (mirrors,ids,route_group,globals_dev,act,route_plist,route_masks,
             route_pcount,e2,e4,partials,np.uint64(panel_bytes),
             np.int32(rows),np.int32(inter),np.int32(nchunks)))

    def reduce(self,partials,down,rows,nchunks):
        self.f["reduce_routes"](((int(rows)+255)//256,ROUTES),(256,),
            (partials,down,np.int32(rows),np.int32(nchunks)))

    def accumulate(self,out,down,w,hidden):
        self.f["accumulate_h4"](((int(hidden)+255)//256,4),(256,),
            (out,down,w,np.int32(hidden)))
