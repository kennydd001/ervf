"""Exact multi-row ERVF kernels.

Each physical subwarp owns one output row. A thread loads each weight element
once and updates M independent virtual-accumulator sets. For every activation
row, the FMA and reduction order is identical to the existing M=1 ERVF kernel.
"""
from __future__ import annotations

import numpy as np

WIDTH = 16
VIRTUAL = 16
ROWS_PER_BLOCK = 16
MS = (1, 2, 3, 4, 6, 8)

HEADER = r"""
__device__ __forceinline__ float p12_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
#define WIDTH 16
#define VIRTUAL 16
#define RPB 16
"""

def _reduce_and_store(M: int, scale_expr: str = "1.0f",
                      relu2: bool = False) -> str:
    lines = [
        f"    float s8[{M}][8];",
        "    #pragma unroll",
        f"    for (int mm=0; mm<{M}; ++mm) {{",
        "      #pragma unroll",
        "      for (int g=0; g<8; ++g) {",
        "        float v=acc[mm][g*2]+acc[mm][g*2+1];",
        "        #pragma unroll",
        "        for (int o=8; o>0; o>>=1)",
        "          v += __shfl_down_sync(0xffffffffu,v,o,16);",
        "        s8[mm][g]=v;",
        "      }",
        "    }",
        "    if (lane==0 && valid) {",
        "      #pragma unroll",
        f"      for (int mm=0; mm<{M}; ++mm) {{",
        "        float a0=s8[mm][0]+s8[mm][4];",
        "        float a1=s8[mm][1]+s8[mm][5];",
        "        float a2=s8[mm][2]+s8[mm][6];",
        "        float a3=s8[mm][3]+s8[mm][7];",
        "        float u0=a0+a2, u1=a1+a3;",
        f"        float v=(u0+u1)*({scale_expr});",
    ]
    if relu2:
        lines += [
            "        float r=fmaxf(v,0.0f);",
            "        out[(size_t)mm*rows+row]=r*r;",
        ]
    else:
        lines += ["        out[(size_t)mm*rows+row]=v;"]
    lines += ["      }", "    }"]
    return "\n".join(lines)

def _bf16(M: int) -> str:
    return f"""
extern "C" __global__ void ervfm_bf16_m{M}(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const int rows,const int cols)
{{
    int sub=threadIdx.x/WIDTH,lane=threadIdx.x&(WIDTH-1);
    int row=blockIdx.x*RPB+sub;
    bool valid=row<rows;
    const unsigned short* w=W+(size_t)(valid?row:0)*cols;
    float acc[{M}][VIRTUAL];
    #pragma unroll
    for(int mm=0;mm<{M};++mm)
      #pragma unroll
      for(int vi=0;vi<VIRTUAL;++vi) acc[mm][vi]=0.0f;
    #pragma unroll
    for(int vi=0;vi<VIRTUAL;++vi){{
      int tid=lane+WIDTH*vi;
      if(valid) for(int k=tid;k<cols;k+=256){{
        float ww=p12_bf16(w[k]);
        #pragma unroll
        for(int mm=0;mm<{M};++mm)
          acc[mm][vi]=fmaf(ww,X[(size_t)mm*cols+k],acc[mm][vi]);
      }}
    }}
{_reduce_and_store(M)}
}}
"""

def _f32(M: int) -> str:
    return f"""
extern "C" __global__ void ervfm_f32_m{M}(
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const int rows,const int cols)
{{
    int sub=threadIdx.x/WIDTH,lane=threadIdx.x&(WIDTH-1);
    int row=blockIdx.x*RPB+sub;
    bool valid=row<rows;
    const float* w=W+(size_t)(valid?row:0)*cols;
    float acc[{M}][VIRTUAL];
    #pragma unroll
    for(int mm=0;mm<{M};++mm)
      #pragma unroll
      for(int vi=0;vi<VIRTUAL;++vi) acc[mm][vi]=0.0f;
    #pragma unroll
    for(int vi=0;vi<VIRTUAL;++vi){{
      int tid=lane+WIDTH*vi;
      if(valid) for(int k=tid;k<cols;k+=256){{
        float ww=w[k];
        #pragma unroll
        for(int mm=0;mm<{M};++mm)
          acc[mm][vi]=fmaf(ww,X[(size_t)mm*cols+k],acc[mm][vi]);
      }}
    }}
{_reduce_and_store(M)}
}}
"""

def _fp8(M: int) -> str:
    return f"""
extern "C" __global__ void ervfm_fp8_m{M}(
    const unsigned char* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ out,
    const float* __restrict__ e4,
    const float wscale,
    const int rows,const int cols)
{{
    __shared__ float lut[256];
    if(threadIdx.x<256) lut[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();
    int sub=threadIdx.x/WIDTH,lane=threadIdx.x&(WIDTH-1);
    int row=blockIdx.x*RPB+sub;
    bool valid=row<rows;
    const unsigned char* w=W+(size_t)(valid?row:0)*cols;
    const uchar4* w4=(const uchar4*)w;
    int nvec=cols>>2;
    float acc[{M}][VIRTUAL];
    #pragma unroll
    for(int mm=0;mm<{M};++mm)
      #pragma unroll
      for(int vi=0;vi<VIRTUAL;++vi) acc[mm][vi]=0.0f;
    #pragma unroll
    for(int vi=0;vi<VIRTUAL;++vi){{
      int tid=lane+WIDTH*vi;
      if(valid){{
        for(int q=tid;q<nvec;q+=256){{
          uchar4 z=w4[q]; int k=q<<2;
          float w0=lut[z.x],w1=lut[z.y],w2=lut[z.z],w3=lut[z.w];
          #pragma unroll
          for(int mm=0;mm<{M};++mm){{
            const float* x=X+(size_t)mm*cols;
            acc[mm][vi]=fmaf(w0,x[k],acc[mm][vi]);
            acc[mm][vi]=fmaf(w1,x[k+1],acc[mm][vi]);
            acc[mm][vi]=fmaf(w2,x[k+2],acc[mm][vi]);
            acc[mm][vi]=fmaf(w3,x[k+3],acc[mm][vi]);
          }}
        }}
        for(int k=(nvec<<2)+tid;k<cols;k+=256){{
          float ww=lut[w[k]];
          #pragma unroll
          for(int mm=0;mm<{M};++mm)
            acc[mm][vi]=fmaf(ww,X[(size_t)mm*cols+k],acc[mm][vi]);
        }}
      }}
    }}
{_reduce_and_store(M, "wscale")}
}}
"""

def _nvfp4(M: int, name: str | None = None, relu2: bool = False) -> str:
    fname = name or f"ervfm_nvfp4_m{M}"
    return f"""
extern "C" __global__ void {fname}(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ X,
    float* __restrict__ out,
    const float* __restrict__ e2,
    const float* __restrict__ e4,
    const float global_scale,
    const int rows,const int cols)
{{
    __shared__ float se2[16],se4[256];
    if(threadIdx.x<16) se2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256) se4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();
    int sub=threadIdx.x/WIDTH,lane=threadIdx.x&(WIDTH-1);
    int row=blockIdx.x*RPB+sub;
    bool valid=row<rows;
    int nbytes=cols>>1,nvec=nbytes>>2;
    const unsigned char* crow=codes+(size_t)(valid?row:0)*nbytes;
    const unsigned char* srow=scales+(size_t)(valid?row:0)*(cols>>4);
    const uchar4* c4=(const uchar4*)crow;
    float acc[{M}][VIRTUAL];
    #pragma unroll
    for(int mm=0;mm<{M};++mm)
      #pragma unroll
      for(int vi=0;vi<VIRTUAL;++vi) acc[mm][vi]=0.0f;
    #pragma unroll
    for(int vi=0;vi<VIRTUAL;++vi){{
      int tid=lane+WIDTH*vi;
      if(valid){{
        for(int v=tid;v<nvec;v+=256){{
          uchar4 q=c4[v]; int b=v<<2,k=b<<1;
          float ss=se4[srow[b>>3]]*global_scale;
          float w0=se2[q.x&15]*ss,w1=se2[q.x>>4]*ss;
          float w2=se2[q.y&15]*ss,w3=se2[q.y>>4]*ss;
          float w4=se2[q.z&15]*ss,w5=se2[q.z>>4]*ss;
          float w6=se2[q.w&15]*ss,w7=se2[q.w>>4]*ss;
          #pragma unroll
          for(int mm=0;mm<{M};++mm){{
            const float* x=X+(size_t)mm*cols;
            acc[mm][vi]=fmaf(w0,x[k],acc[mm][vi]);
            acc[mm][vi]=fmaf(w1,x[k+1],acc[mm][vi]);
            acc[mm][vi]=fmaf(w2,x[k+2],acc[mm][vi]);
            acc[mm][vi]=fmaf(w3,x[k+3],acc[mm][vi]);
            acc[mm][vi]=fmaf(w4,x[k+4],acc[mm][vi]);
            acc[mm][vi]=fmaf(w5,x[k+5],acc[mm][vi]);
            acc[mm][vi]=fmaf(w6,x[k+6],acc[mm][vi]);
            acc[mm][vi]=fmaf(w7,x[k+7],acc[mm][vi]);
          }}
        }}
        for(int b=(nvec<<2)+tid;b<nbytes;b+=256){{
          unsigned char q=crow[b]; int k=b<<1;
          float ss=se4[srow[b>>3]]*global_scale;
          float w0=se2[q&15]*ss,w1=se2[q>>4]*ss;
          #pragma unroll
          for(int mm=0;mm<{M};++mm){{
            const float* x=X+(size_t)mm*cols;
            acc[mm][vi]=fmaf(w0,x[k],acc[mm][vi]);
            acc[mm][vi]=fmaf(w1,x[k+1],acc[mm][vi]);
          }}
        }}
      }}
    }}
{_reduce_and_store(M, "1.0f", relu2)}
}}
"""

SOURCE = HEADER + "\n".join(
    _bf16(m) + _f32(m) + _fp8(m) + _nvfp4(m)
    + _nvfp4(m, f"ervfm_nvfp4_relu2_m{m}", True)
    for m in MS
)

class ERVFM:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.mod = cp.RawModule(code=SOURCE, options=("-std=c++14",))
        self.functions = {}
        for kind in ("bf16", "f32", "fp8", "nvfp4", "nvfp4_relu2"):
            for m in MS:
                self.functions[(kind, m)] = self.mod.get_function(
                    f"ervfm_{kind}_m{m}"
                )

    def run(self, kind, m, out, W, X, rows, cols, *,
            scale=1.0, scales=None, e2=None, e4=None):
        if m not in MS:
            raise ValueError(f"unsupported M={m}")
        grid = ((int(rows) + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK,)
        args = None
        if kind == "bf16":
            args = (W, X, out, np.int32(rows), np.int32(cols))
        elif kind == "f32":
            args = (W, X, out, np.int32(rows), np.int32(cols))
        elif kind == "fp8":
            args = (
                W, X, out, e4, np.float32(scale),
                np.int32(rows), np.int32(cols),
            )
        elif kind in {"nvfp4", "nvfp4_relu2"}:
            args = (
                W, scales, X, out, e2, e4, np.float32(scale),
                np.int32(rows), np.int32(cols),
            )
        else:
            raise ValueError(kind)
        self.functions[(kind, m)](grid, (256,), args)
