from __future__ import annotations

"""Exact H4 shared-expert NVFP4 kernels with production ERVF geometry.

The rejected Phase20B shared-M4 screen assigned one output row to an entire
256-thread CTA.  Phase23's exact routed-UP path instead assigns one 16-lane
sub-warp to each of sixteen output rows.  This module ports that proven
arithmetic and reduction order to the fixed shared expert and optionally
serialises 2 or 4 row tiles per CTA so the four H4 activations are staged only
once.
"""

import numpy as np

M = 4
WIDTH = 16
VIRTUAL = 16
ROWS_PER_WAVE = 16
TILES = (1, 2, 4)


def _kernel(tile: int) -> str:
    if tile not in TILES:
        raise ValueError(f"unsupported tile {tile}")
    return f"""
extern "C" __global__ void shared_nvfp4_m4_r16_t{tile}(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x4,
    float* __restrict__ out4,
    const float global_scale,
    const int rows,
    const int cols,
    const int apply_relu2)
{{
    extern __shared__ float sx[];
    for (int i=threadIdx.x; i<{M}*cols; i+=blockDim.x) sx[i]=x4[i];
    __shared__ float lut[16];
    if (threadIdx.x<16) lut[threadIdx.x]=e2m1[threadIdx.x];
    __syncthreads();

    const int lane=threadIdx.x&({WIDTH}-1);
    const int sub=threadIdx.x/{WIDTH};
    const int nbytes=cols>>1;
    const int nvec=nbytes>>2;

    #pragma unroll 1
    for (int tile_i=0; tile_i<{tile}; ++tile_i) {{
        const int row=blockIdx.x*({ROWS_PER_WAVE}*{tile})
                     +tile_i*{ROWS_PER_WAVE}+sub;
        if (row>=rows) continue;

        const unsigned char* crow=codes+(size_t)row*nbytes;
        const unsigned char* srow=scales+(size_t)row*(cols>>4);
        const uchar4* c4=reinterpret_cast<const uchar4*>(crow);

        float part[{M}][{VIRTUAL}];
        #pragma unroll
        for (int m=0;m<{M};++m)
            for (int vi=0;vi<{VIRTUAL};++vi) part[m][vi]=0.0f;

        #pragma unroll
        for (int vi=0;vi<{VIRTUAL};++vi) {{
            const int tid=lane+{WIDTH}*vi;
            float a[{M}]={{0.0f,0.0f,0.0f,0.0f}};
            for (int v=tid;v<nvec;v+=256) {{
                const uchar4 q=c4[v];
                const int b=v<<2;
                const int k=b<<1;
                const float sc=e4m3[srow[b>>3]]*global_scale;
                const float w[8]={{
                    lut[q.x&15]*sc,lut[q.x>>4]*sc,
                    lut[q.y&15]*sc,lut[q.y>>4]*sc,
                    lut[q.z&15]*sc,lut[q.z>>4]*sc,
                    lut[q.w&15]*sc,lut[q.w>>4]*sc
                }};
                #pragma unroll
                for (int m=0;m<{M};++m) {{
                    const float* xm=sx+(size_t)m*cols;
                    float z=a[m];
                    #pragma unroll
                    for (int j=0;j<8;++j) z=fmaf(w[j],xm[k+j],z);
                    a[m]=z;
                }}
            }}
            for (int b=(nvec<<2)+tid;b<nbytes;b+=256) {{
                const unsigned char q=crow[b];
                const float sc=e4m3[srow[b>>3]]*global_scale;
                const int k=b<<1;
                const float w0=lut[q&15]*sc;
                const float w1=lut[q>>4]*sc;
                #pragma unroll
                for (int m=0;m<{M};++m) {{
                    const float* xm=sx+(size_t)m*cols;
                    a[m]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],a[m]));
                }}
            }}
            #pragma unroll
            for (int m=0;m<{M};++m) part[m][vi]=a[m];
        }}

        #pragma unroll
        for (int m=0;m<{M};++m) {{
            float s8[8];
            #pragma unroll
            for (int w=0;w<8;++w) {{
                float v=part[m][w*2]+part[m][w*2+1];
                for (int off={WIDTH}>>1;off>0;off>>=1)
                    v+=__shfl_down_sync(0xffffffffu,v,off,{WIDTH});
                s8[w]=v;
            }}
            if (lane==0) {{
                const float t0=s8[0]+s8[4];
                const float t1=s8[1]+s8[5];
                const float t2=s8[2]+s8[6];
                const float t3=s8[3]+s8[7];
                float v=(t0+t2)+(t1+t3);
                if (apply_relu2) {{
                    const float r=fmaxf(v,0.0f);
                    v=r*r;
                }}
                out4[(size_t)m*rows+row]=v;
            }}
        }}
    }}
}}
"""


NAMES = tuple(f"shared_nvfp4_m4_r16_t{tile}" for tile in TILES)
SOURCE = "\n".join(_kernel(tile) for tile in TILES)


class SharedM4R16Kernels:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(
            code=SOURCE,
            options=("-std=c++14",),
            name_expressions=NAMES,
        )
        self.f = {
            tile: self.mod.get_function(f"shared_nvfp4_m4_r16_t{tile}")
            for tile in TILES
        }

    def nvfp4(
        self,
        codes,
        scales,
        e2,
        e4,
        x4,
        out4,
        global_scale,
        rows: int,
        cols: int,
        tile: int,
        apply_relu2: bool = False,
    ) -> None:
        tile = int(tile)
        if tile not in self.f:
            raise ValueError(f"tile must be one of {TILES}, got {tile}")
        if int(x4.shape[0]) != M or int(out4.shape[0]) != M:
            raise ValueError("shared M4 kernel requires exactly four rows")
        if int(x4.shape[1]) != int(cols) or int(out4.shape[1]) != int(rows):
            raise ValueError("x4/out4 shapes do not match rows and cols")

        shared = M * int(cols) * 4
        fn = self.f[tile]
        if shared > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = shared
        blocks = (int(rows) + ROWS_PER_WAVE * tile - 1) // (
            ROWS_PER_WAVE * tile
        )
        fn(
            (blocks,),
            (256,),
            (
                codes,
                scales,
                e2,
                e4,
                x4,
                out4,
                np.float32(global_scale),
                np.int32(rows),
                np.int32(cols),
                np.int32(1 if apply_relu2 else 0),
            ),
            shared_mem=shared,
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        """Return best-effort function attributes after NVRTC compilation."""
        result: dict[str, dict[str, int | None]] = {}
        for tile, fn in self.f.items():
            attrs = getattr(fn, "attributes", {}) or {}
            result[str(tile)] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result


def _occupancy_kernel(m: int, staged: bool) -> tuple[str, str]:
    if m not in (2, 4):
        raise ValueError(m)
    mode = "stage" if staged else "direct"
    name = f"shared_nvfp4_m{m}_r16_{mode}"
    stage_decl = "extern __shared__ float sx[];" if staged else ""
    stage_copy = (
        f"for(int i=threadIdx.x;i<{m}*cols;i+=blockDim.x) sx[i]=x[i];"
        if staged else ""
    )
    xbase = "sx" if staged else "x"
    source = f"""
extern "C" __global__ void {name}(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ e2m1,
    const float* __restrict__ e4m3,
    const float* __restrict__ x,
    float* __restrict__ out,
    const float global_scale,
    const int rows,
    const int cols,
    const int apply_relu2)
{{
    {stage_decl}
    {stage_copy}
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2m1[threadIdx.x];
    __syncthreads();

    const int lane=threadIdx.x&15;
    const int sub=threadIdx.x/16;
    const int row=blockIdx.x*16+sub;
    if(row>=rows)return;
    const int nbytes=cols>>1;
    const int nvec=nbytes>>2;
    const unsigned char* crow=codes+(size_t)row*nbytes;
    const unsigned char* srow=scales+(size_t)row*(cols>>4);
    const uchar4* c4=reinterpret_cast<const uchar4*>(crow);

    float part[{m}][16];
    #pragma unroll
    for(int mi=0;mi<{m};++mi)
        for(int vi=0;vi<16;++vi)part[mi][vi]=0.0f;

    #pragma unroll
    for(int vi=0;vi<16;++vi){{
        const int tid=lane+16*vi;
        float a[{m}];
        #pragma unroll
        for(int mi=0;mi<{m};++mi)a[mi]=0.0f;
        for(int v=tid;v<nvec;v+=256){{
            const uchar4 q=c4[v];
            const int b=v<<2;
            const int k=b<<1;
            const float sc=e4m3[srow[b>>3]]*global_scale;
            const float w[8]={{
                lut[q.x&15]*sc,lut[q.x>>4]*sc,
                lut[q.y&15]*sc,lut[q.y>>4]*sc,
                lut[q.z&15]*sc,lut[q.z>>4]*sc,
                lut[q.w&15]*sc,lut[q.w>>4]*sc
            }};
            #pragma unroll
            for(int mi=0;mi<{m};++mi){{
                const float* xm={xbase}+(size_t)mi*cols;
                float z=a[mi];
                #pragma unroll
                for(int j=0;j<8;++j)z=fmaf(w[j],xm[k+j],z);
                a[mi]=z;
            }}
        }}
        for(int b=(nvec<<2)+tid;b<nbytes;b+=256){{
            const unsigned char q=crow[b];
            const float sc=e4m3[srow[b>>3]]*global_scale;
            const int k=b<<1;
            const float w0=lut[q&15]*sc,w1=lut[q>>4]*sc;
            #pragma unroll
            for(int mi=0;mi<{m};++mi){{
                const float* xm={xbase}+(size_t)mi*cols;
                a[mi]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],a[mi]));
            }}
        }}
        #pragma unroll
        for(int mi=0;mi<{m};++mi)part[mi][vi]=a[mi];
    }}

    #pragma unroll
    for(int mi=0;mi<{m};++mi){{
        float s8[8];
        #pragma unroll
        for(int w=0;w<8;++w){{
            float v=part[mi][w*2]+part[mi][w*2+1];
            for(int off=8;off>0;off>>=1)
                v+=__shfl_down_sync(0xffffffffu,v,off,16);
            s8[w]=v;
        }}
        if(lane==0){{
            const float t0=s8[0]+s8[4],t1=s8[1]+s8[5];
            const float t2=s8[2]+s8[6],t3=s8[3]+s8[7];
            float v=(t0+t2)+(t1+t3);
            if(apply_relu2){{const float r=fmaxf(v,0.0f);v=r*r;}}
            out[(size_t)mi*rows+row]=v;
        }}
    }}
}}
"""
    return name, source


OCCUPANCY_VARIANTS = (
    "m2_stage",
    "m2_direct",
    "m4_direct",
)


class SharedOccupancyKernels:
    """M2/M4 occupancy ladder, using one 16-row wave per CTA."""

    def __init__(self):
        import cupy as cp

        specs = ((2, True), (2, False), (4, False))
        built = [_occupancy_kernel(m, staged) for m, staged in specs]
        names = tuple(name for name, _ in built)
        source = "\n".join(src for _, src in built)
        self.cp = cp
        self.mod = cp.RawModule(
            code=source,
            options=("-std=c++14",),
            name_expressions=names,
        )
        self.f = {}
        for (m, staged), (name, _) in zip(specs, built):
            self.f[(m, staged)] = self.mod.get_function(name)

    def nvfp4(
        self,
        codes,
        scales,
        e2,
        e4,
        x,
        out,
        global_scale,
        rows: int,
        cols: int,
        m: int,
        staged: bool,
        apply_relu2: bool = False,
    ) -> None:
        key = (int(m), bool(staged))
        if key not in self.f:
            raise ValueError(f"unsupported occupancy kernel {key}")
        if int(x.shape[0]) != key[0] or int(out.shape[0]) != key[0]:
            raise ValueError(f"kernel {key} requires exactly {key[0]} rows")
        shared = key[0] * int(cols) * 4 if key[1] else 0
        fn = self.f[key]
        if shared > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = shared
        fn(
            ((int(rows) + 15) // 16,),
            (256,),
            (
                codes, scales, e2, e4, x, out,
                np.float32(global_scale), np.int32(rows), np.int32(cols),
                np.int32(1 if apply_relu2 else 0),
            ),
            shared_mem=shared,
        )

    def resource_audit(self) -> dict[str, dict[str, int | None]]:
        result = {}
        for (m, staged), fn in self.f.items():
            attrs = getattr(fn, "attributes", {}) or {}
            name = f"m{m}_{'stage' if staged else 'direct'}"
            result[name] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result
