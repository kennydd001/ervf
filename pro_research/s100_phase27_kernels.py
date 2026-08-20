from __future__ import annotations

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES

H=4
TOPK=6
ROUTES=24
GROUPS=24
NCHUNKS=8

_SOURCE=r"""
#define ROUTES 24
#define GROUPS 24

/* Exact subset of Phase24 gather_group_union_cols.
   The only semantic difference is the fixed [g0,g1) group range and
   configurable gridDim.y chosen by the host. */
extern "C" __global__ void gather_group_union_cols_range(
    const unsigned char* down_base,
    const int* group_ids,
    const int* group_count,
    const int* union_nz,
    const int* union_nzc,
    unsigned char* mirrors,
    const size_t panel_bytes,
    const int rows,
    const int inter,
    const int g0,
    const int g1)
{
    int g=g0+(int)blockIdx.x;
    if(g>=g1 || g>=GROUPS)return;
    if(group_count[g]<=0 || group_count[g]>4)return;

    int warp=(int)threadIdx.x>>5;
    int lane=(int)threadIdx.x&31;
    int task=(int)blockIdx.y*8+warp;
    int stride=(int)gridDim.y*8;
    int ncol=union_nzc[g];
    int rowhalf=rows>>1;
    size_t ps=(size_t)rows+16u*(size_t)rowhalf;

    const unsigned char* src=
        down_base+(size_t)group_ids[g]*panel_bytes;
    unsigned char* dst=
        mirrors+(size_t)g*panel_bytes;

    for(int q=task;q<ncol;q+=stride){
        int j=union_nz[g*inter+q];
        size_t off=(size_t)(j>>4)*ps+rows+
                   (size_t)(j&15)*rowhalf;
        const uchar4* s=
            reinterpret_cast<const uchar4*>(src+off);
        uchar4* d=
            reinterpret_cast<uchar4*>(dst+off);
        for(int k=lane;k<rowhalf/4;k+=32)d[k]=s[k];
    }
}

/* Exact subset of Phase24 down_routes_partial_sres.
   Every active route/chunk executes the identical parent body. The added
   [g0,g1) predicate only decides which pipeline stage launches that body. */
extern "C" __global__ void down_routes_partial_sres_range(
    const unsigned char* mirrors,
    const unsigned char* planes,
    const int* slots,
    const int* ids,
    const int* route_group,
    const float* globals,
    const float* act,
    const int* route_plist,
    const unsigned int* route_masks,
    const int* route_pcount,
    const float* e2,
    const float* e4,
    float* partials,
    const size_t panel_bytes,
    const size_t plane_bytes,
    const int rows,
    const int inter,
    const int nchunks,
    const int g0,
    const int g1)
{
    int y=(int)blockIdx.y;
    int r=y/nchunks;
    int chunk=y-r*nchunks;
    if(r>=ROUTES)return;

    int g=route_group[r];
    if(g<g0 || g>=g1)return;

    int row=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
    if(row>=rows)return;

    int e=ids[r];
    int slot=slots[r];
    int np=inter>>4;
    int rowhalf=rows>>1;

    const unsigned char* bank=
        mirrors+(size_t)g*panel_bytes;
    const unsigned char* plane=
        planes+(size_t)slot*plane_bytes;

    __shared__ float l2[16],l4[256];
    if(threadIdx.x<16)l2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256)l4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();

    int hb=row>>1;
    int hi=row&1;
    int pc=route_pcount[r];
    size_t ps=(size_t)rows+16u*(size_t)rowhalf;
    float acc=0.0f;
    float gs=globals[e*2+0];

    for(int pi=chunk;pi<pc;pi+=nchunks){
        int p=route_plist[r*np+pi];
        const unsigned char* pb=
            bank+(size_t)p*ps;
        float sc=l4[plane[(size_t)p*rows+row]]*gs;
        const unsigned char* codes=pb+rows;
        unsigned mk=route_masks[r*np+p];
        while(mk){
            int c=__ffs(mk)-1;
            mk&=mk-1;
            unsigned char q=codes[(size_t)c*rowhalf+hb];
            float w=l2[hi?(q>>4):(q&15)]*sc;
            acc=fmaf(w,act[(size_t)r*inter+(p<<4)+c],acc);
        }
    }

    partials[((size_t)r*nchunks+chunk)*rows+row]=acc;
}
"""

class Phase27DownPipelineKernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.mod=cp.RawModule(
            code=_SOURCE,
            options=("-std=c++14",),
            name_expressions=(
                "gather_group_union_cols_range",
                "down_routes_partial_sres_range",
            ),
        )
        self.gather_k=self.mod.get_function(
            "gather_group_union_cols_range"
        )
        self.down_k=self.mod.get_function(
            "down_routes_partial_sres_range"
        )

    @staticmethod
    def ranges(n_batches:int):
        n=int(n_batches)
        if n<1 or n>GROUPS:
            raise ValueError(n_batches)
        out=[]
        base=GROUPS//n
        rem=GROUPS%n
        g=0
        for b in range(n):
            width=base+(1 if b<rem else 0)
            out.append((g,g+width))
            g+=width
        if g!=GROUPS:
            raise AssertionError(out)
        return tuple(out)

    def gather_range(
        self,down_base,group_ids,group_count,union_nz,union_nzc,
        mirrors,rows,inter,g0,g1,gather_y,
    ):
        g0=int(g0);g1=int(g1);gy=int(gather_y)
        if not (0<=g0<g1<=GROUPS):
            raise ValueError((g0,g1))
        if gy<1 or gy>64:
            raise ValueError(gather_y)
        self.gather_k(
            (g1-g0,gy),(256,),
            (
                np.uint64(down_base),
                group_ids,group_count,union_nz,union_nzc,
                mirrors,np.uint64(DOWN_PANEL_BYTES),
                np.int32(rows),np.int32(inter),
                np.int32(g0),np.int32(g1),
            ),
        )

    def down_range(
        self,mirrors,planes,slots,ids,route_group,globals_dev,
        act,route_plist,route_masks,route_pcount,e2,e4,partials,
        rows,inter,nchunks,g0,g1,
    ):
        g0=int(g0);g1=int(g1)
        if not (0<=g0<g1<=GROUPS):
            raise ValueError((g0,g1))
        self.down_k(
            ((int(rows)+127)//128,ROUTES*int(nchunks)),
            (128,),
            (
                mirrors,planes,slots,ids,route_group,globals_dev,
                act,route_plist,route_masks,route_pcount,
                e2,e4,partials,
                np.uint64(DOWN_PANEL_BYTES),np.uint64(PLANE_BYTES),
                np.int32(rows),np.int32(inter),np.int32(nchunks),
                np.int32(g0),np.int32(g1),
            ),
        )
