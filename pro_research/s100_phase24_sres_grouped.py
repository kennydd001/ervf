from __future__ import annotations

import numpy as np

from scale_resident_kernels import PLANE_BYTES
from moe_dev_batched import DOWN_PANEL_BYTES

ROUTES=24
GROUPS=24

_SOURCE=r"""
#define ROUTES 24
#define GROUPS 24

/* Group-union gather with the panel-scale branch removed.
   Scale bytes are resident per cache slot. */
extern "C" __global__ void gather_group_union_cols(
    const unsigned char* down_base,
    const int* group_ids,
    const int* group_count,
    const int* union_nz,
    const int* union_nzc,
    unsigned char* mirrors,
    const size_t panel_bytes,
    const int rows,
    const int inter)
{
    int g=blockIdx.x;
    if(group_count[g]<=0 || group_count[g]>4)return;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int task=blockIdx.y*8+warp,stride=gridDim.y*8;
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

/* Arithmetic/order is Phase23 down_routes_partial. Only the scale byte source
   changes to planes[route_slot,panel,row]. */
extern "C" __global__ void down_routes_partial_sres(
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
    const int nchunks)
{
    int y=blockIdx.y,r=y/nchunks,chunk=y-r*nchunks;
    if(r>=ROUTES)return;
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)return;
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

class GroupedScaleResidentKernels:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.mod=cp.RawModule(
            code=_SOURCE,options=("-std=c++14",),
            name_expressions=(
                "gather_group_union_cols",
                "down_routes_partial_sres",
            ),
        )
        self.gather_k=self.mod.get_function("gather_group_union_cols")
        self.down_k=self.mod.get_function("down_routes_partial_sres")

    def gather(self,down_base,group_ids,group_count,union_nz,union_nzc,
               mirrors,rows,inter):
        self.gather_k((GROUPS,32),(256,),
            (np.uint64(down_base),group_ids,group_count,union_nz,union_nzc,
             mirrors,np.uint64(DOWN_PANEL_BYTES),np.int32(rows),np.int32(inter)))

    def down(self,mirrors,planes,slots,ids,route_group,globals_dev,act,
             route_plist,route_masks,route_pcount,e2,e4,partials,
             rows,inter,nchunks):
        self.down_k(
            ((int(rows)+127)//128,ROUTES*int(nchunks)),(128,),
            (mirrors,planes,slots,ids,route_group,globals_dev,act,
             route_plist,route_masks,route_pcount,e2,e4,partials,
             np.uint64(DOWN_PANEL_BYTES),np.uint64(PLANE_BYTES),
             np.int32(rows),np.int32(inter),np.int32(nchunks))
        )
