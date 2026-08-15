from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import LAYERS
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q5, load_q8
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, measure

R = ROOT / "reports/streamq5_moe"
PREREG = R / "N2A_TEMPORAL_ERVF_ORACLE_PREREGISTRATION.md"
OUTPUT = R / "n2a_temporal_ervf_oracle.json"
SEED = 120821
SIZES = (2, 4, 8)

SOURCE = r'''
template<int S>
__device__ __forceinline__ void q8_temporal_row(
    const float* x, const signed char* codes, const unsigned short* scales,
    int row, int cols, int lane, float* result) {
    float partial[S][16];
    int groups = cols >> 7;
    #pragma unroll
    for (int t = 0; t < S; ++t) {
        #pragma unroll
        for (int v = 0; v < 16; ++v) partial[t][v] = 0.0f;
    }
    #pragma unroll
    for (int v = 0; v < 16; ++v) {
        int tid = lane + 16 * v;
        for (int col = tid; col < cols; col += 256) {
            float scale = bf16_to_float(scales[row * groups + (col >> 7)]);
            float weight = round_bf16(((float)codes[(long long)row * cols + col]) * scale);
            #pragma unroll
            for (int t = 0; t < S; ++t) partial[t][v] += weight * x[t * 4096 + col];
        }
    }
    #pragma unroll
    for (int stride = 128; stride >= 16; stride >>= 1) {
        #pragma unroll
        for (int index = 0; index < stride / 16; ++index) {
            #pragma unroll
            for (int t = 0; t < S; ++t) partial[t][index] += partial[t][index + stride / 16];
        }
    }
    #pragma unroll
    for (int t = 0; t < S; ++t) {
        float value = partial[t][0];
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset, 16);
        result[t] = value;
    }
}

template<int S>
__device__ __forceinline__ void q5_temporal_row(
    const float* x, int xstride, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int lane, float* result) {
    float partial[S][16];
    int packs = cols >> 3; int groups = cols >> 7;
    #pragma unroll
    for (int t = 0; t < S; ++t) {
        #pragma unroll
        for (int v = 0; v < 16; ++v) partial[t][v] = 0.0f;
    }
    #pragma unroll
    for (int v = 0; v < 16; ++v) {
        int tid = lane + 16 * v;
        for (int pack = tid; pack < packs; pack += 256) {
            const unsigned char* source = packed + ((long long)row * packs + pack) * 5LL;
            unsigned long long word = ((unsigned long long)source[0]) | ((unsigned long long)source[1] << 8)
                | ((unsigned long long)source[2] << 16) | ((unsigned long long)source[3] << 24)
                | ((unsigned long long)source[4] << 32);
            int column = pack << 3;
            float scale = bf16_to_float(scales[row * groups + (column >> 7)]);
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                int code = ((word >> (item * 5)) & 31ULL) - 15;
                float weight = round_bf16(((float)code) * scale);
                #pragma unroll
                for (int t = 0; t < S; ++t) partial[t][v] += weight * x[t * xstride + column + item];
            }
        }
    }
    #pragma unroll
    for (int stride = 128; stride >= 16; stride >>= 1) {
        #pragma unroll
        for (int index = 0; index < stride / 16; ++index) {
            #pragma unroll
            for (int t = 0; t < S; ++t) partial[t][index] += partial[t][index + stride / 16];
        }
    }
    #pragma unroll
    for (int t = 0; t < S; ++t) {
        float value = partial[t][0];
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset, 16);
        result[t] = value;
    }
}

#define DEFINE_TEMP(S) \
extern "C" __global__ void q8_temporal##S(const float* x, const unsigned char* bank, long long base, long long code_bytes, int rows, int cols, float* output) { \
    int group = (int)threadIdx.x >> 4; int lane = (int)threadIdx.x & 15; int row = (int)blockIdx.x * 16 + group; if (row >= rows) return; \
    const signed char* codes = (const signed char*)(bank + base); const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes); float values[S]; \
    q8_temporal_row<S>(x, codes, scales, row, cols, lane, values); if (lane == 0) { for (int t=0;t<S;++t) output[t*rows+row]=round_bf16(values[t]); } } \
extern "C" __global__ void q5_gate_up_temporal##S(const float* x, const unsigned char* cache, const int* slots, const int* positions, float* gate, float* up) { \
    int group=(int)threadIdx.x>>4; int lane=(int)threadIdx.x&15; int gr=(int)blockIdx.x*16+group; if(gr>=8*1536)return; int expert=gr/1536; int local=gr-expert*1536; int proj=local>=768; int row=local-proj*768; int oe=positions[expert]; \
    long long base=(long long)slots[expert]*3035136LL+(long long)proj*1011712LL; const unsigned char* packed=cache+base+64; const unsigned short* scales=(const unsigned short*)(cache+base+64+983040); float values[S]; \
    q5_temporal_row<S>(x,4096,packed,scales,row,2048,lane,values); if(lane==0){for(int t=0;t<S;++t){if(proj)up[t*6144+oe*768+row]=round_bf16(values[t]);else gate[t*6144+oe*768+row]=round_bf16(values[t]);}} } \
extern "C" __global__ void q5_down_temporal##S(const float* activation, const unsigned char* cache, const int* slots, const int* positions, float* down) { \
    int group=(int)threadIdx.x>>4; int lane=(int)threadIdx.x&15; int gr=(int)blockIdx.x*16+group; if(gr>=8*2048)return; int expert=gr/2048; int row=gr-expert*2048; int oe=positions[expert]; \
    long long base=(long long)slots[expert]*3035136LL+2LL*1011712LL; const unsigned char* packed=cache+base+64; const unsigned short* scales=(const unsigned short*)(cache+base+64+983040); float values[S]; \
    q5_temporal_row<S>(activation+oe*768,6144,packed,scales,row,768,lane,values); if(lane==0){for(int t=0;t<S;++t)down[t*16384+oe*2048+row]=round_bf16(values[t]);} }

DEFINE_TEMP(2)
DEFINE_TEMP(4)
DEFINE_TEMP(8)
'''


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def compact(x): return x["stats"] | {"iterations": len(x["event_ms"])}


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    manifest, q8_pin, q8_host, q8_mem, q8, records, q8_sha = load_q8(); q5_mem, q5 = load_q5()
    names = ["q8_ervf16", "q5_gate_up_ervf16", "q5_down_ervf16"]
    for s in SIZES: names += [f"q8_temporal{s}", f"q5_gate_up_temporal{s}", f"q5_down_temporal{s}"]
    module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + SOURCE, options=("--std=c++11",), name_expressions=tuple(names))
    k={name:module.get_function(name) for name in names}; stream=cp.cuda.Stream(non_blocking=True)
    rng=np.random.default_rng(SEED); x=cp.asarray(rng.standard_normal((8,4096),dtype=np.float32))
    positions=cp.asarray(np.arange(8,dtype=np.int32)); slots=[cp.asarray(np.arange(i*8,i*8+8,dtype=np.int32)) for i in range(LAYERS)]
    max_rows=max(r["rows"] for _,r in records); q8out=cp.empty(8*max_rows,dtype=cp.float32)
    gate=cp.empty(8*6144,dtype=cp.float32); up=cp.empty_like(gate); down=cp.empty(8*16384,dtype=cp.float32)

    def q8_plane(s, temporal):
        for base,record in records:
            if temporal:
                k[f"q8_temporal{s}"](((record["rows"]+15)//16,),(256,),(x,q8,np.int64(base),np.int64(record["code_bytes"]),np.int32(record["rows"]),np.int32(record["cols"]),q8out),stream=stream)
            else:
                for t in range(s): k["q8_ervf16"](((record["rows"]+15)//16,),(256,),(x[t],q8,np.int64(base),np.int64(record["code_bytes"]),np.int32(record["rows"]),np.int32(record["cols"]),q8out[t*max_rows:]),stream=stream)

    def q5_plane(s, temporal):
        for layer in range(LAYERS):
            if temporal:
                k[f"q5_gate_up_temporal{s}"]((768,),(256,),(x,q5,slots[layer],positions,gate,up),stream=stream)
                k[f"q5_down_temporal{s}"]((1024,),(256,),(gate,q5,slots[layer],positions,down),stream=stream)
            else:
                for t in range(s):
                    k["q5_gate_up_ervf16"]((768,),(256,),(x[t],q5,slots[layer],positions,gate[t*6144:],up[t*6144:]),stream=stream)
                    k["q5_down_ervf16"]((1024,),(256,),(gate[t*6144:],q5,slots[layer],positions,down[t*16384:]),stream=stream)

    def capture_q8(s, temporal):
        total_rows=sum(record["rows"] for _,record in records); captured=np.empty((s,total_rows),dtype=np.float32); cursor=0
        for base,record in records:
            rows=record["rows"]
            if temporal:
                k[f"q8_temporal{s}"](((rows+15)//16,),(256,),(x,q8,np.int64(base),np.int64(record["code_bytes"]),np.int32(rows),np.int32(record["cols"]),q8out),stream=stream)
            else:
                for t in range(s): k["q8_ervf16"](((rows+15)//16,),(256,),(x[t],q8,np.int64(base),np.int64(record["code_bytes"]),np.int32(rows),np.int32(record["cols"]),q8out[t*max_rows:]),stream=stream)
            stream.synchronize(); raw=cp.asnumpy(q8out)
            for t in range(s):
                begin=t*(rows if temporal else max_rows); captured[t,cursor:cursor+rows]=raw[begin:begin+rows]
            cursor+=rows
        return captured

    correctness={}; validation={}
    for s in SIZES:
        q8ref=capture_q8(s,False); q8obs=capture_q8(s,True)
        q5_plane(s,False); stream.synchronize(); q5ref=np.concatenate((cp.asnumpy(gate[:s*6144]),cp.asnumpy(up[:s*6144]),cp.asnumpy(down[:s*16384])))
        q5_plane(s,True); stream.synchronize(); q5obs=np.concatenate((cp.asnumpy(gate[:s*6144]),cp.asnumpy(up[:s*6144]),cp.asnumpy(down[:s*16384])))
        correctness[str(s)]={"q8":comparison(q8obs,q8ref),"q5":comparison(q5obs,q5ref)}
        validation[str(s)]={}
        for name,fn in (("q8",q8_plane),("q5",q5_plane)):
            base=measure(stream,lambda f=fn,n=s:f(n,False),5,30); cand=measure(stream,lambda f=fn,n=s:f(n,True),5,30)
            validation[str(s)][name]={"sequential":compact(base),"temporal":compact(cand),"p50_ratio":cand["stats"]["p50"]/base["stats"]["p50"],"p95_ratio":cand["stats"]["p95"]/base["stats"]["p95"]}
        b=sum(validation[str(s)][n]["sequential"]["p50"] for n in ("q8","q5")); c=sum(validation[str(s)][n]["temporal"]["p50"] for n in ("q8","q5"))
        validation[str(s)]["combined_p50_ratio"]=c/b
    open_test=all(correctness["4"][name]["bitwise_equal"] for name in ("q8","q5")) and validation["4"]["combined_p50_ratio"]<=0.80
    test=None
    if open_test:
        test={}
        for name,fn in (("q8",q8_plane),("q5",q5_plane)):
            base=measure(stream,lambda f=fn:f(4,False),10,120); cand=measure(stream,lambda f=fn:f(4,True),10,120)
            test[name]={"sequential":compact(base),"temporal":compact(cand),"p50_ratio":cand["stats"]["p50"]/base["stats"]["p50"],"p95_ratio":cand["stats"]["p95"]/base["stats"]["p95"]}
        b50=sum(test[n]["sequential"]["p50"] for n in ("q8","q5"));c50=sum(test[n]["temporal"]["p50"] for n in ("q8","q5"));b95=sum(test[n]["sequential"]["p95"] for n in ("q8","q5"));c95=sum(test[n]["temporal"]["p95"] for n in ("q8","q5"))
        test["combined"]={"p50_ratio":c50/b50,"p95_ratio":c95/b95,"pass":c50/b50<=0.75 and c95/b95<=0.80}
    result={"kind":"streamq5_moe_n2a_temporal_ervf_oracle","completed_utc":datetime.now(timezone.utc).isoformat(),"inputs":{"preregistration_sha256":sha256(PREREG),"q8_manifest_sha256":sha256(R/"p6a_exact_runtime_bank_result.json"),"q8_aggregate_sha256":q8_sha},"seed":SEED,"correctness":correctness,"validation":validation,"test_opened":open_test,"test":test,"overall_pass":bool(test and test["combined"]["pass"]),"claim_boundary":"Isolated same-expert target weight-reuse oracle; no acceptance, route-union, causal attention or end-to-end claim."}
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps({"correctness":correctness,"validation_ratios":{s:{"q8":validation[s]["q8"]["p50_ratio"],"q5":validation[s]["q5"]["p50_ratio"],"combined":validation[s]["combined_p50_ratio"]}for s in validation},"test_opened":open_test,"test":test,"overall_pass":result["overall_pass"]},indent=2),flush=True)


if __name__=="__main__":main()
