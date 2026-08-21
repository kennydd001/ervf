from __future__ import annotations

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES


GROUPS = 24
ROUTES = 24


CUDA_SOURCE = r"""
#define GROUPS 24
#define ROUTES 24

// Exact scan_group_masks body with one ownership predicate.  Count ranges
// [1,2] and [3,4] are disjoint and cover every valid H4 expert group.
extern "C" __global__ void scan_group_masks_count_range(
    const float* act,const int* group_count,const int* group_refs,
    unsigned int* route_masks,int* route_plist,int* route_pcount,
    unsigned int* union_masks,int* union_plist,int* union_pcount,
    int* union_nz,int* union_nzc,const int inter,const int count_lo,
    const int count_hi)
{
    int g=blockIdx.x,cnt=group_count[g],np=inter>>4;
    if(cnt<count_lo || cnt>count_hi)return;
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
            if(act[(size_t)r*inter+j]!=0.0f){
                atomicOr(&route_masks[r*np+p],bit);any=true;
            }
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
                for(int c=0;c<16;++c)
                    if(mk&(1u<<c))union_nz[g*inter+un++]=(p<<4)+c;
            }
        }
        union_pcount[g]=up;union_nzc[g]=un;
        for(int m=0;m<cnt;++m){
            int r=group_refs[g*4+m],rp=0;
            for(int p=0;p<np;++p)
                if(route_masks[r*np+p])route_plist[r*np+rp++]=p;
            route_pcount[r]=rp;
        }
    }
}

// Exact Phase27R gather body with multiplicity ownership added to its existing
// contiguous gather-batch predicate.
extern "C" __global__ void gather_count_range(
    const unsigned char* down_base,const int* group_ids,
    const int* group_count,const int* union_nz,const int* union_nzc,
    unsigned char* mirrors,const size_t panel_bytes,const int rows,
    const int inter,const int g0,const int g1,const int count_lo,
    const int count_hi)
{
    int g=g0+(int)blockIdx.x;
    if(g>=g1 || g>=GROUPS)return;
    int cnt=group_count[g];
    if(cnt<count_lo || cnt>count_hi)return;

    int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31;
    int task=(int)blockIdx.y*8+warp,stride=(int)gridDim.y*8;
    int ncol=union_nzc[g],rowhalf=rows>>1;
    size_t ps=(size_t)rows+16u*(size_t)rowhalf;
    const unsigned char* src=down_base+(size_t)group_ids[g]*panel_bytes;
    unsigned char* dst=mirrors+(size_t)g*panel_bytes;
    for(int q=task;q<ncol;q+=stride){
        int j=union_nz[g*inter+q];
        size_t off=(size_t)(j>>4)*ps+rows+(size_t)(j&15)*rowhalf;
        const uchar4* s=reinterpret_cast<const uchar4*>(src+off);
        uchar4* d=reinterpret_cast<uchar4*>(dst+off);
        for(int k=lane;k<rowhalf/4;k+=32)d[k]=s[k];
    }
}

// Exact Phase27R sparse-down body.  The ownership predicates only prevent the
// other lane from executing the same route; arithmetic is unchanged.
extern "C" __global__ void down_count_range(
    const unsigned char* mirrors,const unsigned char* planes,const int* slots,
    const int* ids,const int* route_group,const int* group_count,
    const float* globals,const float* act,const int* route_plist,
    const unsigned int* route_masks,const int* route_pcount,const float* e2,
    const float* e4,float* partials,const size_t panel_bytes,
    const size_t plane_bytes,const int rows,const int inter,const int nchunks,
    const int g0,const int g1,const int count_lo,const int count_hi)
{
    int y=(int)blockIdx.y,r=y/nchunks,chunk=y-r*nchunks;
    if(r>=ROUTES)return;
    int g=route_group[r];
    if(g<g0 || g>=g1)return;
    int cnt=group_count[g];
    if(cnt<count_lo || cnt>count_hi)return;

    int row=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
    if(row>=rows)return;
    int e=ids[r],slot=slots[r],np=inter>>4,rowhalf=rows>>1;
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
"""


class Phase31StagedKernels:
    def __init__(self):
        import cupy as cp

        names = (
            "scan_group_masks_count_range",
            "gather_count_range",
            "down_count_range",
        )
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.f = {name: self.mod.get_function(name) for name in names}

    def scan(
        self,
        act,
        group_count,
        group_refs,
        route_masks,
        route_plist,
        route_pcount,
        union_masks,
        union_plist,
        union_pcount,
        union_nz,
        union_nzc,
        inter: int,
        count_lo: int,
        count_hi: int,
    ) -> None:
        self.f["scan_group_masks_count_range"](
            (GROUPS,),
            (256,),
            (
                act,
                group_count,
                group_refs,
                route_masks,
                route_plist,
                route_pcount,
                union_masks,
                union_plist,
                union_pcount,
                union_nz,
                union_nzc,
                np.int32(inter),
                np.int32(count_lo),
                np.int32(count_hi),
            ),
        )

    def gather(
        self,
        down_base: int,
        group_ids,
        group_count,
        union_nz,
        union_nzc,
        mirrors,
        rows: int,
        inter: int,
        g0: int,
        g1: int,
        gather_y: int,
        count_lo: int,
        count_hi: int,
    ) -> None:
        self.f["gather_count_range"](
            (int(g1) - int(g0), int(gather_y)),
            (256,),
            (
                np.uint64(down_base),
                group_ids,
                group_count,
                union_nz,
                union_nzc,
                mirrors,
                np.uint64(DOWN_PANEL_BYTES),
                np.int32(rows),
                np.int32(inter),
                np.int32(g0),
                np.int32(g1),
                np.int32(count_lo),
                np.int32(count_hi),
            ),
        )

    def down(
        self,
        mirrors,
        planes,
        slots,
        ids,
        route_group,
        group_count,
        globals_dev,
        act,
        route_plist,
        route_masks,
        route_pcount,
        e2,
        e4,
        partials,
        rows: int,
        inter: int,
        nchunks: int,
        g0: int,
        g1: int,
        count_lo: int,
        count_hi: int,
    ) -> None:
        self.f["down_count_range"](
            ((int(rows) + 127) // 128, ROUTES * int(nchunks)),
            (128,),
            (
                mirrors,
                planes,
                slots,
                ids,
                route_group,
                group_count,
                globals_dev,
                act,
                route_plist,
                route_masks,
                route_pcount,
                e2,
                e4,
                partials,
                np.uint64(DOWN_PANEL_BYTES),
                np.uint64(PLANE_BYTES),
                np.int32(rows),
                np.int32(inter),
                np.int32(nchunks),
                np.int32(g0),
                np.int32(g1),
                np.int32(count_lo),
                np.int32(count_hi),
            ),
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
