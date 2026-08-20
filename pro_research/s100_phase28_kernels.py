from __future__ import annotations

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES

H = 4
TOPK = 6
ROUTES = 24
GROUPS = 24
MAXM = 4
NCHUNKS = 8
ROW_TILE = 128
PAIR_BYTES = ROW_TILE // 2

_HEADER = r"""
#define ROUTES 24
#define GROUPS 24
#define MAXM 4
#define NCHUNKS 8
#define ROW_TILE 128
#define PAIR_BYTES 64

/* Exact Phase24 route/chunk arithmetic reading the mapped host expert record
   directly. No mirror and no cross-route code reuse. */
extern "C" __global__ void mirrorless_direct_route(
    const unsigned char* __restrict__ down_base,
    const unsigned char* __restrict__ planes,
    const int* __restrict__ slots,
    const int* __restrict__ ids,
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
    const int nchunks)
{
    int y=(int)blockIdx.y;
    int r=y/nchunks;
    int chunk=y-r*nchunks;
    if(r>=ROUTES)return;

    int row=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
    if(row>=rows)return;

    int expert=ids[r];
    int slot=slots[r];
    int np=inter>>4;
    int rowhalf=rows>>1;
    const unsigned char* bank=
        down_base+(size_t)expert*panel_bytes;
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
    float gs=globals[expert*2+0];

    for(int pi=chunk;pi<pc;pi+=nchunks){
        int p=route_plist[r*np+pi];
        const unsigned char* pb=bank+(size_t)p*ps;
        float sc=l4[plane[(size_t)p*rows+row]]*gs;
        const unsigned char* codes=pb+rows;
        unsigned int mk=route_masks[r*np+p];
        while(mk){
            int c=__ffs((int)mk)-1;
            mk&=mk-1;
            unsigned char q=codes[(size_t)c*rowhalf+hb];
            float w=l2[hi?(q>>4):(q&15)]*sc;
            acc=fmaf(w,act[(size_t)r*inter+(p<<4)+c],acc);
        }
    }

    partials[((size_t)r*nchunks+chunk)*rows+row]=acc;
}

/* route_plist is ascending. panel_chunk[r,p]=pi mod 8 therefore identifies
   the exact parent chunk accumulator for every active route panel. */
extern "C" __global__ void build_panel_chunk(
    const int* __restrict__ route_plist,
    const int* __restrict__ route_pcount,
    signed char* __restrict__ panel_chunk,
    const int inter,
    const int nchunks)
{
    int r=(int)blockIdx.x;
    int np=inter>>4;

    for(int p=(int)threadIdx.x;p<np;p+=(int)blockDim.x)
        panel_chunk[r*np+p]=(signed char)-1;
    __syncthreads();

    int pc=route_pcount[r];
    for(int pi=(int)threadIdx.x;pi<pc;pi+=(int)blockDim.x){
        int p=route_plist[r*np+pi];
        panel_chunk[r*np+p]=(signed char)(pi%nchunks);
    }
}
"""


def _shared_load(vector_bytes: int) -> str:
    if vector_bytes == 16:
        return r"""
            int segments=PAIR_BYTES/16;
            int total=ncols*segments;
            for(int z=(int)threadIdx.x;z<total;z+=(int)blockDim.x){
                int ci=z/segments;
                int seg=z-ci*segments;
                int c=s_cols[ci];
                const unsigned char* src=
                    bank+(size_t)p*panel_stride+rows+
                    (size_t)c*rowhalf+(size_t)(row_base>>1)+seg*16;
                uint4 q=*reinterpret_cast<const uint4*>(src);
                reinterpret_cast<uint4*>(
                    s_qbuf+(size_t)c*PAIR_BYTES
                )[seg]=q;
            }
"""
    if vector_bytes == 4:
        return r"""
            int segments=PAIR_BYTES/4;
            int total=ncols*segments;
            for(int z=(int)threadIdx.x;z<total;z+=(int)blockDim.x){
                int ci=z/segments;
                int seg=z-ci*segments;
                int c=s_cols[ci];
                const unsigned char* src=
                    bank+(size_t)p*panel_stride+rows+
                    (size_t)c*rowhalf+(size_t)(row_base>>1)+seg*4;
                uchar4 q=*reinterpret_cast<const uchar4*>(src);
                reinterpret_cast<uchar4*>(
                    s_qbuf+(size_t)c*PAIR_BYTES
                )[seg]=q;
            }
"""
    raise ValueError(vector_bytes)


def _group_chunk_kernel(multiplicity: int) -> str:
    m = int(multiplicity)
    load = _shared_load(16)
    return f"""
/* Group/chunk fusion. Each route keeps the exact parent pi sequence.
   Routes only share a host load when they use the same panel at the same
   chunk step. */
extern "C" __global__ void mirrorless_group_chunk_m{m}_v16(
    const unsigned char* __restrict__ down_base,
    const unsigned char* __restrict__ planes,
    const int* __restrict__ slots,
    const int* __restrict__ group_ids,
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
    const int nchunks)
{{
    int gy=(int)blockIdx.y;
    int g=gy/nchunks;
    int chunk=gy-g*nchunks;
    if(g>=GROUPS || group_count[g]!={m})return;

    int row_base=(int)blockIdx.x*ROW_TILE;
    int row=row_base+(int)threadIdx.x;
    if(row_base+ROW_TILE>rows)return;

    int refs[{m}];
    #pragma unroll
    for(int q=0;q<{m};++q)refs[q]=group_refs[g*MAXM+q];

    int expert=group_ids[g];
    int slot=slots[refs[0]];
    int np=inter>>4;
    int rowhalf=rows>>1;
    size_t panel_stride=(size_t)rows+16u*(size_t)rowhalf;
    const unsigned char* bank=
        down_base+(size_t)expert*panel_bytes;
    const unsigned char* plane=
        planes+(size_t)slot*plane_bytes;
    float gs=globals[expert*2+0];

    __shared__ float s_l2[16];
    __shared__ float s_l4[256];
    __shared__ int s_cols[16];
    __shared__ int s_ncols;
    __shared__ __align__(16) unsigned char s_qbuf[16*PAIR_BYTES];

    if(threadIdx.x<16)s_l2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256)s_l4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();

    float acc[{m}];
    #pragma unroll
    for(int q=0;q<{m};++q)acc[q]=0.0f;

    for(int step=0;;++step){{
        int panels[{m}];
        bool any=false;

        #pragma unroll
        for(int q=0;q<{m};++q){{
            int r=refs[q];
            int pi=chunk+step*nchunks;
            int p=(pi<route_pcount[r])?route_plist[r*np+pi]:-1;
            panels[q]=p;
            any|=(p>=0);
        }}
        if(!any)break;

        #pragma unroll
        for(int lead=0;lead<{m};++lead){{
            int p=panels[lead];
            if(p<0)continue;

            bool seen=false;
            #pragma unroll
            for(int q=0;q<lead;++q)seen|=(panels[q]==p);
            if(seen)continue;

            unsigned int union_mask=0;
            #pragma unroll
            for(int q=0;q<{m};++q){{
                if(panels[q]==p)
                    union_mask|=route_masks[refs[q]*np+p];
            }}

            if(threadIdx.x==0){{
                int count=0;
                unsigned int z=union_mask;
                while(z){{
                    int c=__ffs((int)z)-1;
                    z&=z-1;
                    s_cols[count++]=c;
                }}
                s_ncols=count;
            }}
            __syncthreads();

            int ncols=s_ncols;
{load}
            __syncthreads();

            if(row<rows){{
                int local_pair=(int)threadIdx.x>>1;
                int hi=row&1;
                float sc=s_l4[plane[(size_t)p*rows+row]]*gs;

                #pragma unroll
                for(int q=0;q<{m};++q){{
                    if(panels[q]!=p)continue;
                    int r=refs[q];
                    unsigned int mk=route_masks[r*np+p];
                    while(mk){{
                        int c=__ffs((int)mk)-1;
                        mk&=mk-1;
                        unsigned char packed=
                            s_qbuf[(size_t)c*PAIR_BYTES+local_pair];
                        float w=s_l2[hi?(packed>>4):(packed&15)]*sc;
                        acc[q]=fmaf(
                            w,act[(size_t)r*inter+(p<<4)+c],acc[q]
                        );
                    }}
                }}
            }}
            __syncthreads();
        }}
    }}

    if(row<rows){{
        #pragma unroll
        for(int q=0;q<{m};++q){{
            int r=refs[q];
            partials[((size_t)r*nchunks+chunk)*rows+row]=acc[q];
        }}
    }}
}}
"""


def _allchunks_kernel(multiplicity: int, vector_bytes: int) -> str:
    m = int(multiplicity)
    v = int(vector_bytes)
    load = _shared_load(v)
    return f"""
/* All eight original chunk accumulators remain independent. Panels are
   visited in ascending union-panel order; panel_chunk maps each route panel
   to the parent's original pi mod 8 accumulator. */
extern "C" __global__ void mirrorless_allchunks_m{m}_v{v}(
    const unsigned char* __restrict__ down_base,
    const unsigned char* __restrict__ planes,
    const int* __restrict__ slots,
    const int* __restrict__ group_ids,
    const int* __restrict__ group_count,
    const int* __restrict__ group_refs,
    const float* __restrict__ globals,
    const float* __restrict__ act,
    const unsigned int* __restrict__ route_masks,
    const int* __restrict__ union_plist,
    const int* __restrict__ union_pcount,
    const signed char* __restrict__ panel_chunk,
    const float* __restrict__ e2,
    const float* __restrict__ e4,
    float* __restrict__ partials,
    const size_t panel_bytes,
    const size_t plane_bytes,
    const int rows,
    const int inter,
    const int nchunks)
{{
    int g=(int)blockIdx.y;
    if(g>=GROUPS || group_count[g]!={m})return;

    int row_base=(int)blockIdx.x*ROW_TILE;
    int row=row_base+(int)threadIdx.x;
    if(row_base+ROW_TILE>rows)return;

    int refs[{m}];
    #pragma unroll
    for(int q=0;q<{m};++q)refs[q]=group_refs[g*MAXM+q];

    int expert=group_ids[g];
    int slot=slots[refs[0]];
    int np=inter>>4;
    int rowhalf=rows>>1;
    size_t panel_stride=(size_t)rows+16u*(size_t)rowhalf;
    const unsigned char* bank=
        down_base+(size_t)expert*panel_bytes;
    const unsigned char* plane=
        planes+(size_t)slot*plane_bytes;
    float gs=globals[expert*2+0];

    __shared__ float s_l2[16];
    __shared__ float s_l4[256];
    __shared__ int s_cols[16];
    __shared__ int s_ncols;
    __shared__ __align__(16) unsigned char s_qbuf[16*PAIR_BYTES];

    if(threadIdx.x<16)s_l2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256)s_l4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();

    float acc[{m}][NCHUNKS];
    #pragma unroll
    for(int q=0;q<{m};++q){{
        #pragma unroll
        for(int chunk=0;chunk<NCHUNKS;++chunk)
            acc[q][chunk]=0.0f;
    }}

    int union_count=union_pcount[g];
    for(int upi=0;upi<union_count;++upi){{
        int p=union_plist[g*np+upi];

        unsigned int union_mask=0;
        #pragma unroll
        for(int q=0;q<{m};++q)
            union_mask|=route_masks[refs[q]*np+p];

        if(threadIdx.x==0){{
            int count=0;
            unsigned int z=union_mask;
            while(z){{
                int c=__ffs((int)z)-1;
                z&=z-1;
                s_cols[count++]=c;
            }}
            s_ncols=count;
        }}
        __syncthreads();

        int ncols=s_ncols;
{load}
        __syncthreads();

        if(row<rows){{
            int local_pair=(int)threadIdx.x>>1;
            int hi=row&1;
            float sc=s_l4[plane[(size_t)p*rows+row]]*gs;

            #pragma unroll
            for(int q=0;q<{m};++q){{
                int r=refs[q];
                unsigned int mk=route_masks[r*np+p];
                if(!mk)continue;

                int chunk=(int)panel_chunk[r*np+p];
                if(chunk<0 || chunk>=nchunks)continue;

                while(mk){{
                    int c=__ffs((int)mk)-1;
                    mk&=mk-1;
                    unsigned char packed=
                        s_qbuf[(size_t)c*PAIR_BYTES+local_pair];
                    float w=s_l2[hi?(packed>>4):(packed&15)]*sc;
                    acc[q][chunk]=fmaf(
                        w,act[(size_t)r*inter+(p<<4)+c],
                        acc[q][chunk]
                    );
                }}
            }}
        }}
        __syncthreads();
    }}

    if(row<rows){{
        #pragma unroll
        for(int q=0;q<{m};++q){{
            int r=refs[q];
            #pragma unroll
            for(int chunk=0;chunk<NCHUNKS;++chunk){{
                partials[((size_t)r*nchunks+chunk)*rows+row]
                    =acc[q][chunk];
            }}
        }}
    }}
}}
"""


SOURCE = _HEADER
for _m in (1, 2, 3, 4):
    SOURCE += _group_chunk_kernel(_m)
for _v in (4, 16):
    for _m in (1, 2, 3, 4):
        SOURCE += _allchunks_kernel(_m, _v)


class Phase28MirrorlessKernels:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        names = [
            "mirrorless_direct_route",
            "build_panel_chunk",
        ]
        names += [
            f"mirrorless_group_chunk_m{m}_v16"
            for m in (1, 2, 3, 4)
        ]
        names += [
            f"mirrorless_allchunks_m{m}_v{v}"
            for v in (4, 16)
            for m in (1, 2, 3, 4)
        ]

        self.mod = cp.RawModule(
            code=SOURCE,
            options=("-std=c++14",),
            name_expressions=tuple(names),
        )
        self.f = {
            name: self.mod.get_function(name)
            for name in names
        }

    @staticmethod
    def validate_shape(rows: int, inter: int, nchunks: int):
        if int(rows) % ROW_TILE:
            raise ValueError(
                f"rows must be divisible by {ROW_TILE}; got {rows}"
            )
        if int(inter) % 16:
            raise ValueError(
                f"intermediate size must be divisible by 16; got {inter}"
            )
        if int(nchunks) != NCHUNKS:
            raise ValueError(
                f"exact parent requires nchunks={NCHUNKS}; got {nchunks}"
            )

    def direct_route(
        self,
        down_base,
        planes,
        slots,
        ids,
        globals_dev,
        act,
        route_plist,
        route_masks,
        route_pcount,
        e2,
        e4,
        partials,
        rows,
        inter,
        nchunks,
    ):
        self.validate_shape(rows, inter, nchunks)
        self.f["mirrorless_direct_route"](
            (
                (int(rows) + 127) // 128,
                ROUTES * int(nchunks),
            ),
            (128,),
            (
                np.uint64(down_base),
                planes,
                slots,
                ids,
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
            ),
        )

    def build_panel_chunk(
        self,
        route_plist,
        route_pcount,
        panel_chunk,
        inter,
        nchunks,
    ):
        self.f["build_panel_chunk"](
            (ROUTES,),
            (128,),
            (
                route_plist,
                route_pcount,
                panel_chunk,
                np.int32(inter),
                np.int32(nchunks),
            ),
        )

    def group_chunk_v16(
        self,
        down_base,
        planes,
        slots,
        group_ids,
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
        rows,
        inter,
        nchunks,
    ):
        self.validate_shape(rows, inter, nchunks)
        grid = (
            int(rows) // ROW_TILE,
            GROUPS * int(nchunks),
        )
        for m in (1, 2, 3, 4):
            self.f[f"mirrorless_group_chunk_m{m}_v16"](
                grid,
                (ROW_TILE,),
                (
                    np.uint64(down_base),
                    planes,
                    slots,
                    group_ids,
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
                ),
            )

    def allchunks(
        self,
        vector_bytes,
        down_base,
        planes,
        slots,
        group_ids,
        group_count,
        group_refs,
        globals_dev,
        act,
        route_masks,
        union_plist,
        union_pcount,
        panel_chunk,
        e2,
        e4,
        partials,
        rows,
        inter,
        nchunks,
    ):
        self.validate_shape(rows, inter, nchunks)
        vector_bytes = int(vector_bytes)
        if vector_bytes not in (4, 16):
            raise ValueError(vector_bytes)

        grid = (
            int(rows) // ROW_TILE,
            GROUPS,
        )
        for m in (1, 2, 3, 4):
            self.f[
                f"mirrorless_allchunks_m{m}_v{vector_bytes}"
            ](
                grid,
                (ROW_TILE,),
                (
                    np.uint64(down_base),
                    planes,
                    slots,
                    group_ids,
                    group_count,
                    group_refs,
                    globals_dev,
                    act,
                    route_masks,
                    union_plist,
                    union_pcount,
                    panel_chunk,
                    e2,
                    e4,
                    partials,
                    np.uint64(DOWN_PANEL_BYTES),
                    np.uint64(PLANE_BYTES),
                    np.int32(rows),
                    np.int32(inter),
                    np.int32(nchunks),
                ),
            )

    def attributes(self) -> dict:
        out = {}
        for name, fn in self.f.items():
            try:
                out[name] = dict(fn.attributes)
            except Exception as exc:
                out[name] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        return out
