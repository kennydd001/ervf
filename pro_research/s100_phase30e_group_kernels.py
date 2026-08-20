from __future__ import annotations

import numpy as np

GROUPS = 24
MAX_REFS = 4
TOPK = 6
ROWS_PER_BLOCK = 16


def _kernel(name: str, max_m: int, lo: int, hi: int) -> str:
    return f"""
extern "C" __global__ void {name}(
    const unsigned char* codes_base,const unsigned char* scales_base,
    const int* slots,const int* ids,const float* globals,
    const int* group_count,const int* group_refs,
    const float* e2,const float* e4,const float* x4,float* route_act,
    const int rows,const int cols,const size_t code_stride,const size_t scale_stride)
{{
    const int g=blockIdx.y;
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
    for(int i=threadIdx.x;i<count*cols;i+=blockDim.x){{
        const int m=i/cols,k=i-m*cols,token=refs[m]/{TOPK};
        sx[i]=x4[(size_t)token*cols+k];
    }}
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=e2[threadIdx.x];
    __syncthreads();

    const int lane=threadIdx.x&15,sub=threadIdx.x/16;
    const int row=blockIdx.x*{ROWS_PER_BLOCK}+sub;
    if(row>=rows)return;
    const int nbytes=cols>>1,nvec=nbytes>>2;
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
                lut[q.x&15]*sc,lut[q.x>>4]*sc,lut[q.y&15]*sc,lut[q.y>>4]*sc,
                lut[q.z&15]*sc,lut[q.z>>4]*sc,lut[q.w&15]*sc,lut[q.w>>4]*sc
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
            float v=(t0+t2)+(t1+t3);const float r=fmaxf(v,0.0f);v=r*r;
            route_act[(size_t)refs[m]*rows+row]=v;
        }}
    }}
}}
"""


SPECS = {
    "direct_m1": (1, 1, 1),
    "direct_m2": (2, 2, 2),
    "direct_m3": (3, 3, 3),
    "direct_m4": (4, 4, 4),
    "split_m12": (2, 1, 2),
    "split_m34": (4, 3, 4),
    "unified_m14": (4, 1, 4),
}
SOURCE = "\n".join(
    _kernel(name, max_m, lo, hi)
    for name, (max_m, lo, hi) in SPECS.items()
)


class GroupDispatchKernels:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.mod = cp.RawModule(
            code=SOURCE,
            options=("-std=c++14",),
            name_expressions=tuple(SPECS),
        )
        self.f = {name: self.mod.get_function(name) for name in SPECS}

    def _launch(
        self, name, cache_c, cache_s, slots, ids, globals_dev,
        group_count, group_refs, e2, e4, normed, route_act,
        rows, cols, code_stride, scale_stride,
    ):
        max_m = SPECS[name][0]
        shared = max_m * int(cols) * 4
        fn = self.f[name]
        if shared > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = shared
        fn(
            ((int(rows) + 15) // 16, GROUPS),
            (256,),
            (
                cache_c, cache_s, slots, ids, globals_dev,
                group_count, group_refs, e2, e4, normed, route_act,
                np.int32(rows), np.int32(cols),
                np.uint64(code_stride), np.uint64(scale_stride),
            ),
            shared_mem=shared,
        )

    def direct4(self, *args):
        for m in (1, 2, 3, 4):
            self._launch(f"direct_m{m}", *args)

    def split2(self, *args):
        self._launch("split_m12", *args)
        self._launch("split_m34", *args)

    def unified(self, *args):
        self._launch("unified_m14", *args)

    def resource_audit(self):
        result = {}
        for name, fn in self.f.items():
            attrs = getattr(fn, "attributes", {}) or {}
            result[name] = {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            }
        return result
