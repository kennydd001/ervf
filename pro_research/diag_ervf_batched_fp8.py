import json, sys, numpy as np
from pathlib import Path
REPO = Path("C:/Users/de_do/Documents/ChatGPT/New project")
sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"pro_research"))
from common import require_gpu_free, utc_now, write_json_atomic
require_gpu_free()
import cupy as cp

HEAD = open(REPO/"pro_research/diag_batched_vs_ervf_baseline.py").read()
BASE = HEAD[HEAD.index('BASE = r"""')+11:]; BASE = BASE[:BASE.index('"""')]

ERVF_B = r"""
extern "C" __global__ void ervf_batched_n%(N)d(
    const unsigned char* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, const float wscale, const int rows, const int cols)
{
    const int N = %(N)d;
    // X is read straight from global: N*cols floats is up to 43 KB and would
    // blow the 48 KB dynamic shared limit at N=4, cols=4096. Same values in the
    // same order, so the bit-exactness gate against production ERVF still holds.
    extern __shared__ float lut[];
    const float* __restrict__ sx = X;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_decode((unsigned char)i);
    __syncthreads();
    const int sub = threadIdx.x / W16;
    const int lane = threadIdx.x & (W16 - 1);
    const int row = blockIdx.x * (256 / W16) + sub;
    const bool valid = row < rows;
    const unsigned char* __restrict__ w = W + (size_t)(valid ? row : 0) * cols;
    float acc[V16][%(N)d];
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi)
        #pragma unroll
        for (int n = 0; n < N; ++n) acc[vi][n] = 0.0f;
    const int nvec = cols >> 2;
    const uchar4* w4 = reinterpret_cast<const uchar4*>(w);
    #pragma unroll
    for (int vi = 0; vi < V16; ++vi) {
        const int tid = lane + W16 * vi;
        if (valid) {
            for (int q = tid; q < nvec; q += 256) {
                const uchar4 c = w4[q];
                const int k = q << 2;
                const float d0 = lut[c.x], d1 = lut[c.y], d2 = lut[c.z], d3 = lut[c.w];
                #pragma unroll
                for (int n = 0; n < N; ++n) {
                    const float* __restrict__ sxn = sx + (size_t)n * cols;
                    acc[vi][n] = fmaf(d0, sxn[k],     acc[vi][n]);
                    acc[vi][n] = fmaf(d1, sxn[k + 1], acc[vi][n]);
                    acc[vi][n] = fmaf(d2, sxn[k + 2], acc[vi][n]);
                    acc[vi][n] = fmaf(d3, sxn[k + 3], acc[vi][n]);
                }
            }
            for (int b = (nvec << 2) + tid; b < cols; b += 256) {
                const float d = lut[w[b]];
                #pragma unroll
                for (int n = 0; n < N; ++n)
                    acc[vi][n] = fmaf(d, sx[(size_t)n * cols + b], acc[vi][n]);
            }
        }
    }
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        float a[V16];
        #pragma unroll
        for (int vi = 0; vi < V16; ++vi) a[vi] = acc[vi][n];
        const float v = reduce16(a);
        if (lane == 0 && valid) Y[(size_t)n * rows + row] = v * wscale;
    }
}
"""
NS=[1,2,4]
mod = cp.RawModule(code=BASE + "".join(ERVF_B % {"N":n} for n in NS), options=("-std=c++14",))
k_erv = mod.get_function("prod_ervf16")
k_eb = {n: mod.get_function(f"ervf_batched_n{n}") for n in NS}
rng = np.random.default_rng(20260816)
out={}
for label, rows, cols, mb in (("mamba_in_proj",10304,2688,637.4),("mamba_out_proj",2688,4096,253.2)):
    mats=[cp.asarray(rng.integers(0,256,size=rows*cols,dtype=np.uint8)) for _ in range(8)]
    X=cp.asarray(rng.standard_normal((max(NS),cols)).astype(np.float32).ravel())
    oe=cp.zeros(rows,dtype=cp.float32); Y=cp.zeros(max(NS)*rows,dtype=cp.float32)
    ws=np.float32(0.0123); blocks=(rows+15)//16; wb=rows*cols
    sm_ref=(cols+256)*4
    k_erv((blocks,),(256,),(mats[0],X,oe,ws,np.int32(rows),np.int32(cols)),shared_mem=sm_ref)
    k_eb[1]((blocks,),(256,),(mats[0],X,Y,ws,np.int32(rows),np.int32(cols)),shared_mem=256*4)
    cp.cuda.Device(0).synchronize()
    exact=bool(np.array_equal(cp.asnumpy(oe).view(np.uint32),cp.asnumpy(Y[:rows]).view(np.uint32)))
    def timed(fn):
        fn(0); cp.cuda.Device(0).synchronize()
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record()
        for i in range(100): fn(i)
        e1.record(); e1.synchronize(); return cp.cuda.get_elapsed_time(e0,e1)/100
    ms_ref=timed(lambda i: k_erv((blocks,),(256,),(mats[i%8],X,oe,ws,np.int32(rows),np.int32(cols)),shared_mem=sm_ref))
    per={}
    for N in NS:
        sm=256*4
        ms=timed(lambda i,N=N,sm=sm: k_eb[N]((blocks,),(256,),(mats[i%8],X,Y,ws,np.int32(rows),np.int32(cols)),shared_mem=sm))
        per[str(N)]={"ms_per_token":ms/N,"speedup_vs_production_ervf":ms_ref/(ms/N),
                     "smem_bytes":sm,"gb_s_per_step":wb/(ms*1e-3)/1e9}
    out[label]={"rows":rows,"cols":cols,"mb_per_token":mb,"n1_bitexact_vs_production_ervf":exact,
                "production_ervf_ms":ms_ref,"production_ervf_gb_s":wb/(ms_ref*1e-3)/1e9,"by_N":per}
    del mats,X,oe,Y; cp.get_default_memory_pool().free_all_blocks()

tot=sum(v["mb_per_token"] for v in out.values())
wsp={str(N): sum(v["mb_per_token"]*v["by_N"][str(N)]["speedup_vs_production_ervf"] for v in out.values())/tot for N in NS}
payload={"kind":"diag_ervf_batched_fp8","created_utc":utc_now(),
 "note":"Batching built ON the ERVF-16 geometry (N accumulator sets) instead of on the slow one-block-per-row geometry. The earlier batch numbers used row-block as their N=1 baseline, but production routes these shapes to ERVF, which is 3.4-3.7x faster -- so those speedups were largely recovering ground ERVF already held. This measures batching against what production actually runs.",
 "shapes":out,"all_n1_bitexact_vs_production_ervf":all(v["n1_bitexact_vs_production_ervf"] for v in out.values()),
 "mb_weighted_speedup_vs_production_ervf":wsp}
write_json_atomic(REPO/"pro_research/diag_ervf_batched_fp8.json",payload,archive=False)
print(json.dumps({"n1_bitexact":payload["all_n1_bitexact_vs_production_ervf"],
  "mb_weighted_vs_production_ervf":{k:round(v,3) for k,v in wsp.items()},
  "per_shape":{k:{"prod_ervf_gb_s":round(v["production_ervf_gb_s"],1),
                  **{f"N{n}":round(v["by_N"][str(n)]["speedup_vs_production_ervf"],3) for n in NS}}
               for k,v in out.items()}},indent=2))
