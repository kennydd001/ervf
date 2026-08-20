from __future__ import annotations

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES

H=8
ROUTES=48
GROUPS=48
MAXM=8

_SOURCE=r"""
#define GROUPS 48
#define MAXM 8

extern "C" __global__ void gather_group_union_cols_h8(
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
    if(group_count[g]<=0 || group_count[g]>MAXM)return;
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
"""

class H8ScaleResidentGather:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.mod=cp.RawModule(
            code=_SOURCE,options=("-std=c++14",),
            name_expressions=("gather_group_union_cols_h8",),
        )
        self.gather_k=self.mod.get_function("gather_group_union_cols_h8")

    def gather(self,down_base,group_ids,group_count,union_nz,union_nzc,
               mirrors,rows,inter):
        self.gather_k((GROUPS,32),(256,),
            (np.uint64(down_base),group_ids,group_count,union_nz,union_nzc,
             mirrors,np.uint64(DOWN_PANEL_BYTES),np.int32(rows),np.int32(inter)))
