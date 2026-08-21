"""Exact range-subset kernels for pipelining H8 sparse-down."""
from __future__ import annotations

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES

ROUTES = 48
GROUPS = 48

CUDA_SOURCE = r"""
#define ROUTES 48
#define GROUPS 48
#define MAXM 8

extern "C" __global__ void gather_group_union_cols_h8_range(
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
    const int g=g0+(int)blockIdx.x;
    if(g>=g1 || g>=GROUPS)return;
    if(group_count[g]<=0 || group_count[g]>MAXM)return;
    const int warp=(int)threadIdx.x>>5;
    const int lane=(int)threadIdx.x&31;
    const int task=(int)blockIdx.y*8+warp;
    const int stride=(int)gridDim.y*8;
    const int ncol=union_nzc[g];
    const int rowhalf=rows>>1;
    const size_t ps=(size_t)rows+16u*(size_t)rowhalf;
    const unsigned char* src=down_base+(size_t)group_ids[g]*panel_bytes;
    unsigned char* dst=mirrors+(size_t)g*panel_bytes;
    for(int q=task;q<ncol;q+=stride){
        const int j=union_nz[g*inter+q];
        const size_t off=(size_t)(j>>4)*ps+rows+(size_t)(j&15)*rowhalf;
        const uchar4* s=reinterpret_cast<const uchar4*>(src+off);
        uchar4* d=reinterpret_cast<uchar4*>(dst+off);
        for(int k=lane;k<rowhalf/4;k+=32)d[k]=s[k];
    }
}

extern "C" __global__ void down_routes_partial_sres_h8_range(
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
    const int y=(int)blockIdx.y;
    const int r=y/nchunks;
    const int chunk=y-r*nchunks;
    if(r>=ROUTES)return;
    const int g=route_group[r];
    if(g<g0 || g>=g1)return;
    const int row=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
    if(row>=rows)return;
    const int e=ids[r],slot=slots[r];
    const int np=inter>>4,rowhalf=rows>>1;
    const unsigned char* bank=mirrors+(size_t)g*panel_bytes;
    const unsigned char* plane=planes+(size_t)slot*plane_bytes;
    __shared__ float l2[16],l4[256];
    if(threadIdx.x<16)l2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256)l4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();
    const int hb=row>>1,hi=row&1,pc=route_pcount[r];
    const size_t ps=(size_t)rows+16u*(size_t)rowhalf;
    float acc=0.0f;
    const float gs=globals[e*2+0];
    for(int pi=chunk;pi<pc;pi+=nchunks){
        const int p=route_plist[r*np+pi];
        const unsigned char* pb=bank+(size_t)p*ps;
        const float sc=l4[plane[(size_t)p*rows+row]]*gs;
        const unsigned char* codes=pb+rows;
        unsigned mk=route_masks[r*np+p];
        while(mk){
            const int c=__ffs(mk)-1;
            mk&=mk-1;
            const unsigned char q=codes[(size_t)c*rowhalf+hb];
            const float w=l2[hi?(q>>4):(q&15)]*sc;
            acc=fmaf(w,act[(size_t)r*inter+(p<<4)+c],acc);
        }
    }
    partials[((size_t)r*nchunks+chunk)*rows+row]=acc;
}
"""


class Phase40H8PipelineKernels:
    def __init__(self):
        import cupy as cp

        names = (
            "gather_group_union_cols_h8_range",
            "down_routes_partial_sres_h8_range",
        )
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.gather_k = self.mod.get_function(names[0])
        self.down_k = self.mod.get_function(names[1])

    @staticmethod
    def ranges(batches: int) -> tuple[tuple[int, int], ...]:
        batches = int(batches)
        if batches < 1 or batches > GROUPS:
            raise ValueError(batches)
        width, remainder = divmod(GROUPS, batches)
        result = []
        begin = 0
        for index in range(batches):
            end = begin + width + (1 if index < remainder else 0)
            result.append((begin, end))
            begin = end
        if begin != GROUPS:
            raise AssertionError(result)
        return tuple(result)

    def gather_range(
        self, down_base, group_ids, group_count, union_nz, union_nzc,
        mirrors, rows: int, inter: int, begin: int, end: int, gather_y: int,
    ) -> None:
        begin, end, gather_y = int(begin), int(end), int(gather_y)
        self.gather_k(
            (end - begin, gather_y),
            (256,),
            (
                np.uint64(down_base), group_ids, group_count, union_nz,
                union_nzc, mirrors, np.uint64(DOWN_PANEL_BYTES),
                np.int32(rows), np.int32(inter), np.int32(begin), np.int32(end),
            ),
        )

    def down_range(
        self, mirrors, planes, slots, ids, route_group, globals_dev, act,
        route_plist, route_masks, route_pcount, e2, e4, partials,
        rows: int, inter: int, nchunks: int, begin: int, end: int,
    ) -> None:
        self.down_k(
            ((int(rows) + 127) // 128, ROUTES * int(nchunks)),
            (128,),
            (
                mirrors, planes, slots, ids, route_group, globals_dev, act,
                route_plist, route_masks, route_pcount, e2, e4, partials,
                np.uint64(DOWN_PANEL_BYTES), np.uint64(PLANE_BYTES),
                np.int32(rows), np.int32(inter), np.int32(nchunks),
                np.int32(begin), np.int32(end),
            ),
        )

    def resource_audit(self) -> dict:
        result = {}
        for name, fn in (
            ("gather_group_union_cols_h8_range", self.gather_k),
            ("down_routes_partial_sres_h8_range", self.down_k),
        ):
            fn.compile()
            attrs = getattr(fn, "attributes", {}) or {}
            result[name] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result

