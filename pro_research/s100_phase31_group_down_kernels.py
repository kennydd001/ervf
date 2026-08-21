from __future__ import annotations

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES


GROUPS = 24
MAX_REFS = 4
NCHUNKS = 8
ROW_TILE = 128
PAIR_BYTES = ROW_TILE // 2


def _kernel(name: str, max_m: int, count_lo: int, count_hi: int) -> str:
    return f"""
// Exact per-route/chunk arithmetic with gathered code bytes shared across
// routes selecting the same expert.  Only data movement is coalesced.
extern "C" __global__ void {name}(
    const unsigned char* __restrict__ mirrors,
    const unsigned char* __restrict__ planes,
    const int* __restrict__ slots,
    const int* __restrict__ ids,
    const int* __restrict__ group_count,
    const int* __restrict__ group_refs,
    const float* __restrict__ globals,
    const float* __restrict__ act,
    const int* __restrict__ route_plist,
    const unsigned int* __restrict__ route_masks,
    const int* __restrict__ route_pcount,
    const float* __restrict__ e2,
    const float* __restrict__ e4,
    float* __restrict__ partials,
    const size_t panel_bytes,
    const size_t plane_bytes,
    const int rows,
    const int inter,
    const int nchunks,
    const int g0,
    const int g1)
{{
    const int gy=(int)blockIdx.y;
    const int g=g0+gy/nchunks;
    const int chunk=gy-(gy/nchunks)*nchunks;
    if(g>=g1 || g>=GROUPS)return;

    const int count=group_count[g];
    if(count<{count_lo} || count>{count_hi})return;

    const int row_base=(int)blockIdx.x*ROW_TILE;
    const int row=row_base+(int)threadIdx.x;
    if(row_base+ROW_TILE>rows)return;

    int refs[{max_m}];
    #pragma unroll
    for(int q=0;q<{max_m};++q)
        refs[q]=(q<count)?group_refs[g*{MAX_REFS}+q]:-1;

    const int anchor=refs[0];
    const int expert=ids[anchor];
    const int slot=slots[anchor];
    const int np=inter>>4;
    const int rowhalf=rows>>1;
    const size_t panel_stride=(size_t)rows+16u*(size_t)rowhalf;
    const unsigned char* bank=mirrors+(size_t)g*panel_bytes;
    const unsigned char* plane=planes+(size_t)slot*plane_bytes;
    const float gs=globals[expert*2+0];

    __shared__ float s_l2[16];
    __shared__ float s_l4[256];
    __shared__ int s_cols[16];
    __shared__ int s_ncols;
    __shared__ __align__(16) unsigned char s_qbuf[16*{PAIR_BYTES}];

    if(threadIdx.x<16)s_l2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256)s_l4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();

    float acc[{max_m}];
    #pragma unroll
    for(int q=0;q<{max_m};++q)acc[q]=0.0f;

    for(int step=0;;++step){{
        int panels[{max_m}];
        bool any=false;
        #pragma unroll
        for(int q=0;q<{max_m};++q){{
            int p=-1;
            if(q<count){{
                const int r=refs[q];
                const int pi=chunk+step*nchunks;
                p=(pi<route_pcount[r])?route_plist[r*np+pi]:-1;
            }}
            panels[q]=p;
            any|=(p>=0);
        }}
        if(!any)break;

        #pragma unroll
        for(int lead=0;lead<{max_m};++lead){{
            const int p=panels[lead];
            if(p<0)continue;

            bool seen=false;
            #pragma unroll
            for(int q=0;q<lead;++q)seen|=(panels[q]==p);
            if(seen)continue;

            unsigned int union_mask=0;
            #pragma unroll
            for(int q=0;q<{max_m};++q)
                if(q<count && panels[q]==p)
                    union_mask|=route_masks[refs[q]*np+p];

            if(threadIdx.x==0){{
                int n=0;
                unsigned int mask=union_mask;
                while(mask){{
                    const int c=__ffs((int)mask)-1;
                    mask&=mask-1;
                    s_cols[n++]=c;
                }}
                s_ncols=n;
            }}
            __syncthreads();

            const int segments={PAIR_BYTES}/16;
            const int total=s_ncols*segments;
            for(int z=(int)threadIdx.x;z<total;z+=(int)blockDim.x){{
                const int ci=z/segments;
                const int seg=z-ci*segments;
                const int c=s_cols[ci];
                const unsigned char* src=
                    bank+(size_t)p*panel_stride+rows+
                    (size_t)c*rowhalf+(size_t)(row_base>>1)+seg*16;
                const uint4 packed=*reinterpret_cast<const uint4*>(src);
                reinterpret_cast<uint4*>(
                    s_qbuf+(size_t)c*{PAIR_BYTES}
                )[seg]=packed;
            }}
            __syncthreads();

            const int local_pair=(int)threadIdx.x>>1;
            const int hi=row&1;
            const float sc=s_l4[plane[(size_t)p*rows+row]]*gs;
            #pragma unroll
            for(int q=0;q<{max_m};++q){{
                if(q>=count || panels[q]!=p)continue;
                const int r=refs[q];
                unsigned int mask=route_masks[r*np+p];
                while(mask){{
                    const int c=__ffs((int)mask)-1;
                    mask&=mask-1;
                    const unsigned char packed=
                        s_qbuf[(size_t)c*{PAIR_BYTES}+local_pair];
                    const float w=s_l2[hi?(packed>>4):(packed&15)]*sc;
                    acc[q]=fmaf(
                        w,act[(size_t)r*inter+(p<<4)+c],acc[q]
                    );
                }}
            }}
            __syncthreads();
        }}
    }}

    #pragma unroll
    for(int q=0;q<{max_m};++q){{
        if(q<count){{
            const int r=refs[q];
            partials[((size_t)r*nchunks+chunk)*rows+row]=acc[q];
        }}
    }}
}}
"""


SPECS = {
    "group_down_m12": (2, 1, 2),
    "group_down_m34": (4, 3, 4),
}
CUDA_SOURCE = f"#define GROUPS {GROUPS}\n#define ROW_TILE {ROW_TILE}\n" + "\n".join(
    _kernel(name, max_m, count_lo, count_hi)
    for name, (max_m, count_lo, count_hi) in SPECS.items()
)


class Phase31GroupDownKernels:
    def __init__(self):
        import cupy as cp

        self.mod = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=tuple(SPECS),
        )
        self.f = {name: self.mod.get_function(name) for name in SPECS}

    def launch_range(
        self,
        mirrors,
        planes,
        slots,
        ids,
        group_count,
        group_refs,
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
    ) -> None:
        if int(rows) % ROW_TILE:
            raise ValueError(f"rows must be divisible by {ROW_TILE}; got {rows}")
        if int(nchunks) != NCHUNKS:
            raise ValueError(f"expected nchunks={NCHUNKS}; got {nchunks}")
        grid = (int(rows) // ROW_TILE, (int(g1) - int(g0)) * int(nchunks))
        args = (
            mirrors,
            planes,
            slots,
            ids,
            group_count,
            group_refs,
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
        )
        for name in SPECS:
            self.f[name](grid, (ROW_TILE,), args)

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
