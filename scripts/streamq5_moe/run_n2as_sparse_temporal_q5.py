from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
from safetensors import safe_open

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import BANK_DIR, EXPERT_BYTES, LAYERS
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison

R = ROOT / "reports/streamq5_moe"
ROUTES = ROOT / "reports/runs/streamq5_moe/p4d_routes"
CAPTURE = R / "p4d_route_capture_result.json"
PREREG = R / "N2AS_SPARSE_TEMPORAL_Q5_PREREGISTRATION.md"
OUTPUT = R / "n2as_sparse_temporal_q5.json"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
S = 4
SLOTS_PER_LAYER = 32
SEED = 120822
VALIDATION_STARTS = (512, 576, 640, 704)
TEST_STARTS = (768, 832, 896, 960)

SOURCE = r'''
template<int S>
__device__ __forceinline__ void q5_sparse_row(
    const float* x, int xstride, const unsigned char* packed,
    const unsigned short* scales, const int* ranks, int expert,
    int row, int cols, int lane, float* result) {
    float partial[S][16];
    int packs = cols >> 3; int groups = cols >> 7;
    #pragma unroll
    for (int t=0;t<S;++t) {
        #pragma unroll
        for (int v=0;v<16;++v) partial[t][v]=0.0f;
    }
    #pragma unroll
    for (int v=0;v<16;++v) {
        int tid=lane+16*v;
        for (int pack=tid;pack<packs;pack+=256) {
            const unsigned char* source=packed+((long long)row*packs+pack)*5LL;
            unsigned long long word=((unsigned long long)source[0])|((unsigned long long)source[1]<<8)
                |((unsigned long long)source[2]<<16)|((unsigned long long)source[3]<<24)
                |((unsigned long long)source[4]<<32);
            int column=pack<<3;
            float scale=bf16_to_float(scales[row*groups+(column>>7)]);
            #pragma unroll
            for (int item=0;item<8;++item) {
                int code=((word>>(item*5))&31ULL)-15;
                float weight=round_bf16(((float)code)*scale);
                #pragma unroll
                for (int t=0;t<S;++t) {
                    if (ranks[expert*S+t]>=0) partial[t][v]+=weight*x[t*xstride+column+item];
                }
            }
        }
    }
    #pragma unroll
    for (int stride=128;stride>=16;stride>>=1) {
        #pragma unroll
        for (int index=0;index<stride/16;++index) {
            #pragma unroll
            for (int t=0;t<S;++t) partial[t][index]+=partial[t][index+stride/16];
        }
    }
    #pragma unroll
    for (int t=0;t<S;++t) {
        float value=partial[t][0];
        #pragma unroll
        for (int offset=8;offset>0;offset>>=1) value+=__shfl_down_sync(0xffffffffU,value,offset,16);
        result[t]=value;
    }
}

extern "C" __global__ void q5_gate_up_sparse4(
    const float* x, const unsigned char* cache, int base_slot, int union_count,
    const int* ranks, float* gate, float* up) {
    int group=(int)threadIdx.x>>4; int lane=(int)threadIdx.x&15;
    int gr=(int)blockIdx.x*16+group; if(gr>=union_count*1536)return;
    int expert=gr/1536; int local=gr-expert*1536; int proj=local>=768;
    int row=local-proj*768; int slot=base_slot+expert;
    long long base=(long long)slot*3035136LL+(long long)proj*1011712LL;
    const unsigned char* packed=cache+base+64;
    const unsigned short* scales=(const unsigned short*)(cache+base+64+983040);
    float values[4]; q5_sparse_row<4>(x,4096,packed,scales,ranks,expert,row,2048,lane,values);
    if(lane==0){
        #pragma unroll
        for(int t=0;t<4;++t){int rank=ranks[expert*4+t];if(rank>=0){int out=(t*8+rank)*768+row;if(proj)up[out]=round_bf16(values[t]);else gate[out]=round_bf16(values[t]);}}
    }
}

template<int S>
__device__ __forceinline__ void q5_sparse_down_row(
    const float* activation, const unsigned char* packed,
    const unsigned short* scales, const int* ranks, int expert,
    int row, int lane, float* result) {
    float partial[S][16];
    #pragma unroll
    for(int t=0;t<S;++t){
        #pragma unroll
        for(int v=0;v<16;++v)partial[t][v]=0.0f;
    }
    #pragma unroll
    for(int v=0;v<16;++v){int tid=lane+16*v;for(int pack=tid;pack<96;pack+=256){
        const unsigned char* source=packed+((long long)row*96+pack)*5LL;
        unsigned long long word=((unsigned long long)source[0])|((unsigned long long)source[1]<<8)
            |((unsigned long long)source[2]<<16)|((unsigned long long)source[3]<<24)
            |((unsigned long long)source[4]<<32);
        int column=pack<<3;float scale=bf16_to_float(scales[row*6+(column>>7)]);
        #pragma unroll
        for(int item=0;item<8;++item){int code=((word>>(item*5))&31ULL)-15;float weight=round_bf16(((float)code)*scale);
            #pragma unroll
            for(int t=0;t<S;++t){int rank=ranks[expert*S+t];if(rank>=0)partial[t][v]+=weight*activation[(t*8+rank)*768+column+item];}
        }
    }}
    #pragma unroll
    for(int stride=128;stride>=16;stride>>=1){
        #pragma unroll
        for(int index=0;index<stride/16;++index){
            #pragma unroll
            for(int t=0;t<S;++t)partial[t][index]+=partial[t][index+stride/16];
        }
    }
    #pragma unroll
    for(int t=0;t<S;++t){
        float value=partial[t][0];
        #pragma unroll
        for(int offset=8;offset>0;offset>>=1)value+=__shfl_down_sync(0xffffffffU,value,offset,16);
        result[t]=value;
    }
}

extern "C" __global__ void q5_down_sparse4_real(
    const float* activation, const unsigned char* cache, int base_slot,
    int union_count, const int* ranks, float* down) {
    int group=(int)threadIdx.x>>4; int lane=(int)threadIdx.x&15;
    int gr=(int)blockIdx.x*16+group; if(gr>=union_count*2048)return;
    int expert=gr/2048; int row=gr-expert*2048; int slot=base_slot+expert;
    long long base=(long long)slot*3035136LL+2LL*1011712LL;
    const unsigned char* packed=cache+base+64;
    const unsigned short* scales=(const unsigned short*)(cache+base+64+983040);
    float values[4];q5_sparse_down_row<4>(activation,packed,scales,ranks,expert,row,lane,values);
    if(lane==0){
        #pragma unroll
        for(int t=0;t<4;++t){int rank=ranks[expert*4+t];if(rank>=0)down[(t*8+rank)*2048+row]=round_bf16(values[t]);}
    }
}
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def load_q5_32() -> tuple[cp.cuda.Memory, cp.ndarray]:
    total = LAYERS * SLOTS_PER_LAYER * EXPERT_BYTES
    memory = cp.cuda.alloc(total)
    device = cp.ndarray((total,), dtype=cp.uint8, memptr=memory)
    staging = cp.cuda.alloc_pinned_memory(EXPERT_BYTES)
    host = np.frombuffer(staging, dtype=np.uint8, count=EXPERT_BYTES)
    stream = cp.cuda.Stream(non_blocking=True)
    for layer in range(LAYERS):
        with (BANK_DIR / f"layer_{layer:02d}.q5bin").open("rb", buffering=8 * 2**20) as handle:
            for expert in range(SLOTS_PER_LAYER):
                handle.seek(expert * EXPERT_BYTES)
                if handle.readinto(host) != EXPERT_BYTES:
                    raise RuntimeError("short Q5 record")
                slot = layer * SLOTS_PER_LAYER + expert
                cp.cuda.runtime.memcpyAsync(memory.ptr + slot * EXPERT_BYTES, staging.ptr, EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                stream.synchronize()
        print(json.dumps({"q5_32_layers_loaded": layer + 1}), flush=True)
    return memory, device


def metadata(starts: tuple[int, ...]):
    result = []
    for domain in DOMAINS:
        for layer in range(LAYERS):
            with safe_open(ROUTES / f"layer_{layer:02d}.safetensors", framework="numpy") as handle:
                route = handle.get_tensor(f"{domain}_router_ids").astype(np.int32)
            for start in starts:
                block = route[start:start + S]
                union = np.unique(block)
                index = {int(value): i for i, value in enumerate(union)}
                mapped = np.vectorize(lambda value: index[int(value)], otypes=[np.int32])(block)
                ranks = np.full((len(union), S), -1, dtype=np.int32)
                for token in range(S):
                    for rank in range(8):
                        ranks[mapped[token, rank], token] = rank
                result.append({"domain": domain, "layer": layer, "start": start,
                               "union_count": int(len(union)), "slots": cp.asarray(layer * SLOTS_PER_LAYER + mapped),
                               "ranks": cp.asarray(ranks.reshape(-1))})
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    for layer in range(LAYERS):
        path = ROUTES / f"layer_{layer:02d}.safetensors"
        if sha256(path) != capture["manifests"][str(layer)]["artifact_sha256"]:
            raise RuntimeError("route hash mismatch")
    q5_mem, q5 = load_q5_32()
    names = ("q5_gate_up_ervf16", "q5_down_ervf16", "q5_gate_up_sparse4", "q5_down_sparse4_real")
    module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + SOURCE, options=("--std=c++11",), name_expressions=names)
    k = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x = cp.asarray(rng.standard_normal((S, 4096), dtype=np.float32))
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    gate = cp.empty(S * 8 * 768, dtype=cp.float32); up = cp.empty_like(gate)
    down = cp.empty(S * 8 * 2048, dtype=cp.float32)

    validation_meta = metadata(VALIDATION_STARTS)
    test_meta = metadata(TEST_STARTS)

    def reference(item):
        for token in range(S):
            slots = item["slots"][token]
            k["q5_gate_up_ervf16"]((768,), (256,), (x[token], q5, slots, positions,
                gate[token * 8 * 768:], up[token * 8 * 768:]), stream=stream)
            k["q5_down_ervf16"]((1024,), (256,), (gate[token * 8 * 768:], q5, slots,
                positions, down[token * 8 * 2048:]), stream=stream)

    def candidate(item):
        count = item["union_count"]
        k["q5_gate_up_sparse4"](((count * 1536 + 15) // 16,), (256,),
            (x, q5, np.int32(item["layer"] * SLOTS_PER_LAYER), np.int32(count), item["ranks"], gate, up), stream=stream)
        k["q5_down_sparse4_real"](((count * 2048 + 15) // 16,), (256,),
            (gate, q5, np.int32(item["layer"] * SLOTS_PER_LAYER), np.int32(count), item["ranks"], down), stream=stream)

    def correctness_check(items):
        checks = []
        chosen = [item for item in items if item["start"] == items[0]["start"]]
        for item in chosen:
            reference(item); stream.synchronize()
            expected = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))
            candidate(item); stream.synchronize()
            observed = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))
            checks.append(comparison(observed, expected))
        return {"planes": len(checks), "elements": int(sum(x["elements"] for x in checks)),
                "different": int(sum(x["different"] for x in checks)),
                "bitwise_equal": all(x["bitwise_equal"] for x in checks),
                "max_abs": max(x["max_abs"] for x in checks)}

    def run_suite(items, kind):
        fn = reference if kind == "reference" else candidate
        for item in items: fn(item)

    def measure_pair(items, warmups, iterations):
        for i in range(warmups):
            for kind in (("reference", "candidate") if i % 2 == 0 else ("candidate", "reference")):
                run_suite(items, kind)
        stream.synchronize()
        values = {"reference": [], "candidate": []}
        for i in range(iterations):
            order = ("reference", "candidate") if i % 2 == 0 else ("candidate", "reference")
            for kind in order:
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record(stream); run_suite(items, kind); end.record(stream); end.synchronize()
                values[kind].append(float(cp.cuda.get_elapsed_time(begin, end)))
        return {kind: {"event_ms": vals, "stats": stats(vals)} for kind, vals in values.items()}

    valid_correct = correctness_check(validation_meta)
    validation = measure_pair(validation_meta, 2, 12)
    validation["p50_ratio"] = validation["candidate"]["stats"]["p50"] / validation["reference"]["stats"]["p50"]
    validation["p95_ratio"] = validation["candidate"]["stats"]["p95"] / validation["reference"]["stats"]["p95"]
    test_opened = valid_correct["bitwise_equal"] and validation["p50_ratio"] <= 0.90
    test_correct = None; test = None
    if test_opened:
        test_correct = correctness_check(test_meta)
        test = measure_pair(test_meta, 3, 30)
        test["p50_ratio"] = test["candidate"]["stats"]["p50"] / test["reference"]["stats"]["p50"]
        test["p95_ratio"] = test["candidate"]["stats"]["p95"] / test["reference"]["stats"]["p95"]
        test["pass"] = test_correct["bitwise_equal"] and test["p50_ratio"] <= 0.85 and test["p95_ratio"] <= 0.90
    result = {"kind": "streamq5_moe_n2as_sparse_temporal_q5", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"preregistration_sha256": sha256(PREREG), "route_capture_sha256": sha256(CAPTURE),
                         "domains": DOMAINS, "validation_starts": VALIDATION_STARTS, "test_starts": TEST_STARTS,
                         "seed": SEED, "physical_slots_per_layer": SLOTS_PER_LAYER},
              "validation_correctness": valid_correct, "validation": validation, "test_opened": test_opened,
              "test_correctness": test_correct, "test": test, "overall_pass": bool(test and test["pass"]),
              "claim_boundary": "Physical sparse temporal Q5 kernel on actual P4D route patterns; no causal availability, acceptance, quality or end-to-end claim."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_correctness": valid_correct, "validation_ratios": {"p50": validation["p50_ratio"], "p95": validation["p95_ratio"]},
                      "test_opened": test_opened, "test_correctness": test_correct,
                      "test_ratios": None if test is None else {"p50": test["p50_ratio"], "p95": test["p95_ratio"]}, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
