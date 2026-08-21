"""Explicit-global group-range variants of Phase25 H8 UP and scan kernels."""
from __future__ import annotations

import numpy as np

from s100_phase25_h8_kernels import (
    GROUPS,
    HIGH_ROWS_PER_BLOCK,
    LOW_ROWS_PER_BLOCK,
    MAXM,
    high_up_kernel,
    low_up_kernel,
)


def _up_range_source(multiplicity: int) -> str:
    multiplicity = int(multiplicity)
    source = (
        low_up_kernel(multiplicity)
        if multiplicity <= 4
        else high_up_kernel(multiplicity)
    )
    old_name = f"grouped_up_h8_m{multiplicity}"
    new_name = f"grouped_up_h8_m{multiplicity}_range"
    source = source.replace(old_name, new_name, 1)
    old_tail = (
        "const int rows,const int cols,const size_t code_stride,"
        "const size_t scale_stride)"
    )
    new_tail = (
        "const int rows,const int cols,const size_t code_stride,"
        "const size_t scale_stride,const int g0)"
    )
    if old_tail not in source or "int g=blockIdx.y;" not in source:
        raise RuntimeError(f"Phase25 UP template changed for M={multiplicity}")
    source = source.replace(old_tail, new_tail, 1)
    source = source.replace("int g=blockIdx.y;", "int g=g0+(int)blockIdx.y;", 1)
    return source


SCAN_SOURCE = r"""
#define ROUTES 48
#define GROUPS 48
#define MAXM 8

extern "C" __global__ void scan_group_masks_h8_range(
    const float* act,const int* group_count,const int* group_refs,
    unsigned int* route_masks,int* route_plist,int* route_pcount,
    unsigned int* union_masks,int* union_plist,int* union_pcount,
    int* union_nz,int* union_nzc,const int inter,const int g0)
{
    const int g=g0+(int)blockIdx.x;
    const int cnt=group_count[g],np=inter>>4;
    if(g>=GROUPS || cnt<=0 || cnt>MAXM)return;
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
        bool any=false;const int p=j>>4;const unsigned bit=1u<<(j&15);
        for(int m=0;m<cnt;++m){
            const int r=group_refs[g*MAXM+m];
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
            const unsigned mk=union_masks[g*np+p];
            if(mk){
                union_plist[g*np+up++]=p;
                for(int c=0;c<16;++c)
                    if(mk&(1u<<c))union_nz[g*inter+un++]=(p<<4)+c;
            }
        }
        union_pcount[g]=up;union_nzc[g]=un;
        for(int m=0;m<cnt;++m){
            const int r=group_refs[g*MAXM+m];int rp=0;
            for(int p=0;p<np;++p)
                if(route_masks[r*np+p])route_plist[r*np+rp++]=p;
            route_pcount[r]=rp;
        }
    }
}
"""

CUDA_SOURCE = (
    "#define H 8\n#define TOPK 6\n#define ROUTES 48\n"
    "#define GROUPS 48\n#define MAXM 8\n"
    + "".join(_up_range_source(m) for m in range(1, 9))
    + SCAN_SOURCE
)


class Phase42RangeDispatchKernels:
    def __init__(self):
        import cupy as cp

        names = tuple(f"grouped_up_h8_m{m}_range" for m in range(1, 9)) + (
            "scan_group_masks_h8_range",
        )
        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.f = {name: self.mod.get_function(name) for name in names}

    def up_range(
        self, multiplicity: int, cache_c, cache_s, slots, ids, globals_dev,
        group_count, group_refs, e2, e4, normed, route_act,
        rows: int, cols: int, code_stride: int, scale_stride: int,
        begin: int, end: int,
    ) -> None:
        multiplicity = int(multiplicity)
        fn = self.f[f"grouped_up_h8_m{multiplicity}_range"]
        if multiplicity <= 4:
            rows_per_block = LOW_ROWS_PER_BLOCK
            shared = multiplicity * int(cols) * 4
            if shared > 48 * 1024:
                fn.max_dynamic_shared_size_bytes = shared
        else:
            rows_per_block = HIGH_ROWS_PER_BLOCK
            shared = 0
        fn(
            ((int(rows) + rows_per_block - 1) // rows_per_block, int(end) - int(begin)),
            (256,),
            (
                cache_c, cache_s, slots, ids, globals_dev,
                group_count, group_refs, e2, e4, normed, route_act,
                np.int32(rows), np.int32(cols),
                np.uint64(code_stride), np.uint64(scale_stride), np.int32(begin),
            ),
            shared_mem=shared,
        )

    def scan_range(
        self, act, group_count, group_refs, route_masks, route_plist,
        route_pcount, union_masks, union_plist, union_pcount, union_nz,
        union_nzc, inter: int, begin: int, end: int,
    ) -> None:
        self.f["scan_group_masks_h8_range"](
            (int(end) - int(begin),),
            (256,),
            (
                act, group_count, group_refs, route_masks, route_plist,
                route_pcount, union_masks, union_plist, union_pcount,
                union_nz, union_nzc, np.int32(inter), np.int32(begin),
            ),
        )

    def resource_audit(self) -> dict:
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

