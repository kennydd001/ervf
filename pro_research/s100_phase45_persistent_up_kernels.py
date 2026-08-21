from __future__ import annotations

import numpy as np


GROUPS = 24
MAX_REFS = 4
TOPK = 6
ROWS_PER_TILE = 16


def _kernel(name: str, max_m: int, lo: int, hi: int, workers: int) -> str:
    return f"""
extern "C" __global__ void {name}(
    const unsigned char* codes_base,const unsigned char* scales_base,
    const int* slots,const int* ids,const float* globals,
    const int* group_count,const int* group_refs,
    const float* e2,const float* e4,const float* x4,float* route_act,
    const int rows,const int cols,const size_t code_stride,const size_t scale_stride)
{{
    const int g=(int)blockIdx.y;
    const int count=group_count[g];
    if(count<{lo} || count>{hi})return;
    const int anchor=group_refs[g*{MAX_REFS}];
    if(anchor<0)return;
    const int slot=slots[anchor],expert=ids[anchor];
    const unsigned char* codes=codes_base+(size_t)slot*code_stride;
    const unsigned char* scales=scales_base+(size_t)slot*scale_stride;
    const float gs=globals[expert*2+1];

    extern __shared__ float sx[];
    int refs[{max_m}];
    #pragma unroll
    for(int m=0;m<{max_m};++m)refs[m]=(m<count)?group_refs[g*{MAX_REFS}+m]:-1;
    for(int i=(int)threadIdx.x;i<count*cols;i+=(int)blockDim.x){{
        const int m=i/cols,k=i-m*cols,token=refs[m]/{TOPK};
        sx[i]=x4[(size_t)token*cols+k];
    }}
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2[threadIdx.x];
    __syncthreads();

    const int lane=(int)threadIdx.x&15,sub=(int)threadIdx.x/16;
    const int nbytes=cols>>1,nvec=nbytes>>2;
    const int ntiles=(rows+{ROWS_PER_TILE}-1)/{ROWS_PER_TILE};

    for(int tile=(int)blockIdx.x;tile<ntiles;tile+={workers}){{
        const int row=tile*{ROWS_PER_TILE}+sub;
        if(row>=rows)continue;
        const unsigned char* crow=codes+(size_t)row*nbytes;
        const unsigned char* srow=scales+(size_t)row*(cols>>4);
        const uchar4* c4=reinterpret_cast<const uchar4*>(crow);

        float part[{max_m}][16];
        #pragma unroll
        for(int m=0;m<{max_m};++m)
            for(int vi=0;vi<16;++vi)part[m][vi]=0.0f;

        #pragma unroll
        for(int vi=0;vi<16;++vi){{
            const int tid=lane+16*vi;
            float a[{max_m}];
            #pragma unroll
            for(int m=0;m<{max_m};++m)a[m]=0.0f;
            for(int v=tid;v<nvec;v+=256){{
                const uchar4 q=c4[v];const int b=v<<2,k=b<<1;
                const float sc=e4[srow[b>>3]]*gs;
                const float w[8]={{
                    lut[q.x&15]*sc,lut[q.x>>4]*sc,
                    lut[q.y&15]*sc,lut[q.y>>4]*sc,
                    lut[q.z&15]*sc,lut[q.z>>4]*sc,
                    lut[q.w&15]*sc,lut[q.w>>4]*sc
                }};
                #pragma unroll
                for(int m=0;m<{max_m};++m)if(m<count){{
                    const float* xm=sx+(size_t)m*cols;float z=a[m];
                    #pragma unroll
                    for(int j=0;j<8;++j)z=fmaf(w[j],xm[k+j],z);
                    a[m]=z;
                }}
            }}
            for(int b=(nvec<<2)+tid;b<nbytes;b+=256){{
                const unsigned char q=crow[b];
                const float sc=e4[srow[b>>3]]*gs;const int k=b<<1;
                const float w0=lut[q&15]*sc,w1=lut[q>>4]*sc;
                #pragma unroll
                for(int m=0;m<{max_m};++m)if(m<count){{
                    const float* xm=sx+(size_t)m*cols;
                    a[m]=fmaf(w1,xm[k+1],fmaf(w0,xm[k],a[m]));
                }}
            }}
            #pragma unroll
            for(int m=0;m<{max_m};++m)part[m][vi]=a[m];
        }}

        #pragma unroll
        for(int m=0;m<{max_m};++m)if(m<count){{
            float s8[8];
            #pragma unroll
            for(int w=0;w<8;++w){{
                float v=part[m][w*2]+part[m][w*2+1];
                for(int off=8;off>0;off>>=1)
                    v+=__shfl_down_sync(0xffffffffu,v,off,16);
                s8[w]=v;
            }}
            if(lane==0){{
                const float t0=s8[0]+s8[4],t1=s8[1]+s8[5];
                const float t2=s8[2]+s8[6],t3=s8[3]+s8[7];
                float v=(t0+t2)+(t1+t3);
                const float r=fmaxf(v,0.0f);v=r*r;
                route_act[(size_t)refs[m]*rows+row]=v;
            }}
        }}
    }}
}}
"""


SCHEDULES = {
    "p2_p4": (2, 4),
    "p4_p4": (4, 4),
    "p4_p8": (4, 8),
}


class PersistentSplitUpKernels:
    def __init__(self, schedule: str):
        import cupy as cp

        if schedule not in SCHEDULES:
            raise ValueError(schedule)
        self.schedule = schedule
        self.workers12, self.workers34 = SCHEDULES[schedule]
        names = (f"p45_m12_w{self.workers12}", f"p45_m34_w{self.workers34}")
        source = _kernel(names[0], 2, 1, 2, self.workers12)
        source += _kernel(names[1], 4, 3, 4, self.workers34)
        module = cp.RawModule(
            code=source,
            options=("-std=c++17",),
            name_expressions=names,
        )
        self.functions = {
            "m12": module.get_function(names[0]),
            "m34": module.get_function(names[1]),
        }

    def _launch(self, key: str, workers: int, max_m: int, *args):
        cols = int(args[-3])
        shared = max_m * cols * 4
        function = self.functions[key]
        if shared > 48 * 1024:
            function.max_dynamic_shared_size_bytes = shared
        function(
            (workers, GROUPS),
            (256,),
            tuple(args[:-4])
            + (
                np.int32(args[-4]),
                np.int32(args[-3]),
                np.uint64(args[-2]),
                np.uint64(args[-1]),
            ),
            shared_mem=shared,
        )

    def split2(self, *args):
        self._launch("m12", self.workers12, 2, *args)
        self._launch("m34", self.workers34, 4, *args)

    def resource_audit(self):
        result = {}
        for key, function in self.functions.items():
            function.compile()
            attrs = getattr(function, "attributes", {}) or {}
            result[key] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result
