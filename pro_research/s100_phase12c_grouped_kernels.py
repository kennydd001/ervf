"""Exact same-expert multi-row routed-down kernels."""
from __future__ import annotations

import numpy as np

MS = (1, 2, 3, 4, 6, 8)
HIDDEN = 2688
INTER = 1856
NPANEL = 116
ROWHALF = 1344
PANEL_STRIDE = 24192
PLANE_BYTES = 311808

HEADER = r"""
#define ROWS 2688
#define INTER 1856
#define NPANEL 116
#define ROWHALF 1344
#define PANEL_STRIDE 24192
"""

def make_down(M: int) -> str:
    return f"""
extern "C" __global__ void grouped_down_m{M}(
    const unsigned char* __restrict__ record,
    const unsigned char* __restrict__ plane,
    const float* __restrict__ acts,
    const unsigned int* __restrict__ masks,
    const float* __restrict__ e2,
    const float* __restrict__ e4,
    const float global_scale,
    float* __restrict__ partials,
    const int nchunks)
{{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int chunk=blockIdx.y;
    if(row>=ROWS) return;
    __shared__ float se2[16],se4[256];
    if(threadIdx.x<16) se2[threadIdx.x]=e2[threadIdx.x];
    if(threadIdx.x<256) se4[threadIdx.x]=e4[threadIdx.x];
    __syncthreads();

    int hb=row>>1,hi=row&1;
    float acc[{M}];
    int rank[{M}];
    #pragma unroll
    for(int mm=0;mm<{M};++mm){{acc[mm]=0.0f;rank[mm]=0;}}

    for(int p=0;p<NPANEL;++p){{
        unsigned int mmask[{M}];
        bool take[{M}];
        unsigned int uni=0u;
        #pragma unroll
        for(int mm=0;mm<{M};++mm){{
            unsigned int mk=masks[(size_t)mm*NPANEL+p];
            mmask[mm]=mk;
            take[mm]=(mk!=0u && (rank[mm]%nchunks)==chunk);
            if(take[mm]) uni|=mk;
            if(mk!=0u) rank[mm]++;
        }}
        if(uni==0u) continue;
        float ss=se4[plane[(size_t)p*ROWS+row]]*global_scale;
        const unsigned char* pc=
            record+(size_t)p*PANEL_STRIDE+ROWS;
        while(uni){{
            int c=__ffs(uni)-1;
            uni&=uni-1;
            unsigned char q=pc[(size_t)c*ROWHALF+hb];
            float ww=se2[hi?(q>>4):(q&15)]*ss;
            #pragma unroll
            for(int mm=0;mm<{M};++mm)
                if(take[mm] && (mmask[mm]&(1u<<c)))
                    acc[mm]=fmaf(
                        ww,acts[(size_t)mm*INTER+(p<<4)+c],acc[mm]
                    );
        }}
    }}
    #pragma unroll
    for(int mm=0;mm<{M};++mm)
        partials[((size_t)mm*nchunks+chunk)*ROWS+row]=acc[mm];
}}
"""

SOURCE = HEADER + "\n".join(make_down(m) for m in MS)

class GroupedDown:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.mod=cp.RawModule(code=SOURCE, options=("-std=c++14",))
        self.k={
            m:self.mod.get_function(f"grouped_down_m{m}")
            for m in MS
        }

    def run(self, m, record, plane, acts, masks, e2, e4,
            global_scale, partials, nchunks):
        self.k[int(m)](
            ((HIDDEN+127)//128, int(nchunks)), (128,),
            (
                record, plane, acts, masks, e2, e4,
                np.float32(global_scale), partials, np.int32(nchunks),
            ),
        )
