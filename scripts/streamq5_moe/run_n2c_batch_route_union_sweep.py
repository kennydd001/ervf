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
from scripts.streamq5_moe.run_n1b_q5_vectorized_loads import VECTOR_SOURCE
from scripts.streamq5_moe.run_p3a_integrated_expert import BANK_DIR, EXPERT_BYTES, LAYERS
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q8
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, stats


R = ROOT / "reports/streamq5_moe"
ROUTES = ROOT / "reports/runs/streamq5_moe/p4d_routes"
CAPTURE = R / "p4d_route_capture_result.json"
N1B = R / "n1b_q5_vectorized_loads.json"
PREREG = R / "N2C_BATCH_ROUTE_UNION_SWEEP_PREREGISTRATION.md"
OUTPUT = R / "n2c_batch_route_union_sweep.json"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
SIZES = (2, 4, 8, 16)
EXECUTABLE_SIZES = (2, 4, 8)
VALIDATION_STARTS = (512, 576, 640, 704)
TEST_STARTS = (768, 832, 896, 960)
PHYSICAL_SLOTS = 80
SEED = 120823


BASE_SOURCE = r'''
__device__ __forceinline__ unsigned long long n2c_q5_word(
    const unsigned char* source) {
    unsigned long long address = (unsigned long long)source;
    int shift = (int)(address & 3ULL) * 8;
    const unsigned int* aligned = (const unsigned int*)(address & ~3ULL);
    unsigned long long window = (unsigned long long)aligned[0]
        | ((unsigned long long)aligned[1] << 32);
    return (window >> shift) & 0xffffffffffULL;
}

template<int S>
__device__ __forceinline__ void n2c_q8_row(
    const float* x, const signed char* codes, const unsigned short* scales,
    int row, int cols, int lane, float* result) {
    float partial[S][16]; int groups=cols>>7;
    #pragma unroll
    for(int t=0;t<S;++t){
        #pragma unroll
        for(int v=0;v<16;++v)partial[t][v]=0.0f;
    }
    #pragma unroll
    for(int v=0;v<16;++v){int tid=lane+16*v;for(int col=tid;col<cols;col+=256){
        float scale=bf16_to_float(scales[row*groups+(col>>7)]);
        float weight=round_bf16(((float)codes[(long long)row*cols+col])*scale);
        #pragma unroll
        for(int t=0;t<S;++t)partial[t][v]+=weight*x[t*4096+col];
    }}
    #pragma unroll
    for(int stride=128;stride>=16;stride>>=1){
        #pragma unroll
        for(int i=0;i<stride/16;++i){
            #pragma unroll
            for(int t=0;t<S;++t)partial[t][i]+=partial[t][i+stride/16];
        }
    }
    #pragma unroll
    for(int t=0;t<S;++t){float value=partial[t][0];
        #pragma unroll
        for(int offset=8;offset>0;offset>>=1)value+=__shfl_down_sync(0xffffffffU,value,offset,16);
        result[t]=value;
    }
}

template<int S, bool RANKED>
__device__ __forceinline__ void n2c_q5_sparse_row(
    const float* x, int xstride, const unsigned char* packed,
    const unsigned short* scales, const int* active_counts,
    const int* active_tokens, const int* active_ranks,
    int expert, int row, int cols, int lane,
    float* result) {
    float partial[S][16]; int packs=cols>>3;int groups=cols>>7;
    int active=active_counts[expert];
    #pragma unroll
    for(int j=0;j<S;++j){
        #pragma unroll
        for(int v=0;v<16;++v)partial[j][v]=0.0f;
    }
    #pragma unroll
    for(int v=0;v<16;++v){int tid=lane+16*v;for(int pack=tid;pack<packs;pack+=256){
        const unsigned char* source=packed+((long long)row*packs+pack)*5LL;
        unsigned long long word=n2c_q5_word(source);int column=pack<<3;
        float scale=bf16_to_float(scales[row*groups+(column>>7)]);
        #pragma unroll
        for(int item=0;item<8;++item){int code=((word>>(item*5))&31ULL)-15;
            float weight=round_bf16(((float)code)*scale);
            for(int j=0;j<active;++j){int token=active_tokens[expert*S+j];
                int base=RANKED?(token*8+active_ranks[expert*S+j])*768:token*xstride;
                partial[j][v]+=weight*x[base+column+item];}
        }
    }}
    #pragma unroll
    for(int stride=128;stride>=16;stride>>=1){
        #pragma unroll
        for(int i=0;i<stride/16;++i)for(int j=0;j<active;++j)
            partial[j][i]+=partial[j][i+stride/16];
    }
    for(int j=0;j<active;++j){float value=partial[j][0];
        #pragma unroll
        for(int offset=8;offset>0;offset>>=1)value+=__shfl_down_sync(0xffffffffU,value,offset,16);
        result[j]=value;
    }
}
'''


def wrappers(size: int) -> str:
    return f'''
extern "C" __global__ void n2c_q8_temporal{size}(
    const float* x,const unsigned char* bank,long long base,long long code_bytes,
    int rows,int cols,float* output){{
    int group=(int)threadIdx.x>>4;int lane=(int)threadIdx.x&15;
    int row=(int)blockIdx.x*16+group;if(row>=rows)return;
    const signed char* codes=(const signed char*)(bank+base);
    const unsigned short* scales=(const unsigned short*)(bank+base+code_bytes);
    float values[{size}];n2c_q8_row<{size}>(x,codes,scales,row,cols,lane,values);
    if(lane==0)for(int t=0;t<{size};++t)output[t*rows+row]=round_bf16(values[t]);
}}
extern "C" __global__ void n2c_q5_gate_up_sparse{size}(
    const float* x,const unsigned char* cache,int union_count,
    const int* active_counts,const int* active_tokens,const int* active_ranks,
    float* gate,float* up){{
    int group=(int)threadIdx.x>>4;int lane=(int)threadIdx.x&15;
    int gr=(int)blockIdx.x*16+group;if(gr>=union_count*1536)return;
    int expert=gr/1536;int local=gr-expert*1536;int proj=local>=768;
    int row=local-proj*768;long long base=(long long)expert*3035136LL+(long long)proj*1011712LL;
    const unsigned char* packed=cache+base+64;
    const unsigned short* scales=(const unsigned short*)(cache+base+64+983040);
    float values[{size}];n2c_q5_sparse_row<{size},false>(x,4096,packed,scales,active_counts,active_tokens,active_ranks,expert,row,2048,lane,values);
    if(lane==0){{int active=active_counts[expert];for(int j=0;j<active;++j){{
        int token=active_tokens[expert*{size}+j];int rank=active_ranks[expert*{size}+j];
        int out=(token*8+rank)*768+row;if(proj)up[out]=round_bf16(values[j]);else gate[out]=round_bf16(values[j]);
    }}}}
}}
extern "C" __global__ void n2c_q5_down_sparse{size}(
    const float* activation,const unsigned char* cache,int union_count,
    const int* active_counts,const int* active_tokens,const int* active_ranks,
    float* down){{
    int group=(int)threadIdx.x>>4;int lane=(int)threadIdx.x&15;
    int gr=(int)blockIdx.x*16+group;if(gr>=union_count*2048)return;
    int expert=gr/2048;int row=gr-expert*2048;
    long long base=(long long)expert*3035136LL+2LL*1011712LL;
    const unsigned char* packed=cache+base+64;
    const unsigned short* scales=(const unsigned short*)(cache+base+64+983040);
    float values[{size}];n2c_q5_sparse_row<{size},true>(activation,6144,packed,scales,active_counts,active_tokens,active_ranks,expert,row,768,lane,values);
    if(lane==0){{int active=active_counts[expert];for(int j=0;j<active;++j){{
        int token=active_tokens[expert*{size}+j];int rank=active_ranks[expert*{size}+j];
        down[(token*8+rank)*2048+row]=round_bf16(values[j]);
    }}}}
}}
'''


SOURCE = BASE_SOURCE + "\n".join(wrappers(size) for size in SIZES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_q5_payload() -> tuple[cp.cuda.Memory, cp.ndarray]:
    total = PHYSICAL_SLOTS * EXPERT_BYTES
    memory = cp.cuda.alloc(total)
    device = cp.ndarray((total,), dtype=cp.uint8, memptr=memory)
    staging = cp.cuda.alloc_pinned_memory(EXPERT_BYTES)
    host = np.frombuffer(staging, dtype=np.uint8, count=EXPERT_BYTES)
    stream = cp.cuda.Stream(non_blocking=True)
    with (BANK_DIR / "layer_00.q5bin").open("rb", buffering=8 * 2**20) as handle:
        for expert in range(PHYSICAL_SLOTS):
            handle.seek(expert * EXPERT_BYTES)
            if handle.readinto(host) != EXPERT_BYTES:
                raise RuntimeError("short Q5 record")
            cp.cuda.runtime.memcpyAsync(
                memory.ptr + expert * EXPERT_BYTES, staging.ptr, EXPERT_BYTES,
                cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
            )
            stream.synchronize()
    return memory, device


def load_routes() -> dict[str, list[np.ndarray]]:
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    result = {domain: [] for domain in DOMAINS}
    for layer in range(LAYERS):
        path = ROUTES / f"layer_{layer:02d}.safetensors"
        if sha256(path) != capture["manifests"][str(layer)]["artifact_sha256"]:
            raise RuntimeError(f"route hash mismatch layer {layer}")
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                result[domain].append(handle.get_tensor(f"{domain}_router_ids").astype(np.int32))
    return result


def build_blocks(routes, size: int, starts: tuple[int, ...]):
    blocks = []
    for domain in DOMAINS:
        for start in starts:
            layers = []
            for layer in range(LAYERS):
                block = routes[domain][layer][start:start + size]
                union = np.unique(block)
                if len(union) > PHYSICAL_SLOTS:
                    raise RuntimeError("physical slot capacity exceeded")
                index = {int(value): i for i, value in enumerate(union)}
                mapped = np.vectorize(lambda value: index[int(value)], otypes=[np.int32])(block)
                counts = np.zeros(len(union), dtype=np.int32)
                tokens = np.full((len(union), size), -1, dtype=np.int32)
                ranks = np.full((len(union), size), -1, dtype=np.int32)
                for token in range(size):
                    for rank in range(8):
                        expert = mapped[token, rank]
                        cursor = counts[expert]
                        tokens[expert, cursor] = token
                        ranks[expert, cursor] = rank
                        counts[expert] += 1
                layers.append({
                    "union_count": int(len(union)),
                    "slots": cp.asarray(mapped),
                    "counts": cp.asarray(counts),
                    "tokens": cp.asarray(tokens.reshape(-1)),
                    "ranks": cp.asarray(ranks.reshape(-1)),
                })
            blocks.append({"domain": domain, "start": start, "layers": layers})
    return blocks


def paired_measure(stream, reference, candidate, warmups: int, iterations: int):
    for iteration in range(warmups):
        order = (reference, candidate) if iteration % 2 == 0 else (candidate, reference)
        for launch in order:
            launch()
    stream.synchronize()
    values = {"reference": [], "candidate": []}
    for iteration in range(iterations):
        order = (("reference", reference), ("candidate", candidate)) if iteration % 2 == 0 else (("candidate", candidate), ("reference", reference))
        for name, launch in order:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record(stream); launch(); end.record(stream); end.synchronize()
            values[name].append(float(cp.cuda.get_elapsed_time(begin, end)))
    result = {name: {"event_ms": rows, "stats": stats(rows)} for name, rows in values.items()}
    result["p50_ratio"] = result["candidate"]["stats"]["p50"] / result["reference"]["stats"]["p50"]
    result["p95_ratio"] = result["candidate"]["stats"]["p95"] / result["reference"]["stats"]["p95"]
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    n1b = json.loads(N1B.read_text(encoding="utf-8"))
    if not n1b["overall_pass"] or n1b["selected"] != "aligned32x2":
        raise RuntimeError("N1B aligned32x2 pass required")
    routes = load_routes()
    q5_mem, q5 = load_q5_payload()
    manifest, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    names = ["q8_ervf16", "q5_gate_up_aligned32x2", "q5_down_aligned32x2"]
    for size in SIZES:
        names += [f"n2c_q8_temporal{size}", f"n2c_q5_gate_up_sparse{size}", f"n2c_q5_down_sparse{size}"]
    module = cp.RawModule(
        code=CUDA_SOURCE + ERVF_SOURCE + VECTOR_SOURCE + SOURCE,
        options=("--std=c++11",), name_expressions=tuple(names),
    )
    k = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x = cp.asarray(rng.standard_normal((max(SIZES), 4096), dtype=np.float32))
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    gate = cp.empty(max(SIZES) * 8 * 768, dtype=cp.float32)
    up = cp.empty_like(gate)
    down = cp.empty(max(SIZES) * 8 * 2048, dtype=cp.float32)
    q8_out = cp.empty(max(SIZES) * max(record["rows"] for _, record in q8_records), dtype=cp.float32)

    def q8_plane(size: int, temporal: bool) -> None:
        for base, record in q8_records:
            if temporal:
                k[f"n2c_q8_temporal{size}"](
                    ((record["rows"] + 15) // 16,), (256,),
                    (x, q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), q8_out), stream=stream,
                )
            else:
                for token in range(size):
                    k["q8_ervf16"](
                        ((record["rows"] + 15) // 16,), (256,),
                        (x[token], q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), q8_out[token * record["rows"]:]), stream=stream,
                    )

    def q5_layer(size: int, item, temporal: bool) -> None:
        if temporal:
            count = item["union_count"]
            k[f"n2c_q5_gate_up_sparse{size}"](
                ((count * 1536 + 15) // 16,), (256,),
                (x, q5, np.int32(count), item["counts"], item["tokens"], item["ranks"], gate, up), stream=stream,
            )
            k[f"n2c_q5_down_sparse{size}"](
                ((count * 2048 + 15) // 16,), (256,),
                (gate, q5, np.int32(count), item["counts"], item["tokens"], item["ranks"], down), stream=stream,
            )
        else:
            for token in range(size):
                slots = item["slots"][token]
                k["q5_gate_up_aligned32x2"](
                    (768,), (256,), (x[token], q5, slots, positions, gate[token * 6144:], up[token * 6144:]), stream=stream,
                )
                k["q5_down_aligned32x2"](
                    (1024,), (256,), (gate[token * 6144:], q5, slots, positions, down[token * 16384:]), stream=stream,
                )

    def q5_suite(size: int, blocks, temporal: bool) -> None:
        for block in blocks:
            for item in block["layers"]:
                q5_layer(size, item, temporal)

    def combined_suite(size: int, blocks, temporal: bool) -> None:
        for block in blocks:
            q8_plane(size, temporal)
            for item in block["layers"]:
                q5_layer(size, item, temporal)

    results = {
        "16": {
            "status": "blocked_by_resource_spill_timeout",
            "correctness": None,
            "validation": None,
            "test_opened": False,
            "test": None,
            "note": "Original preregistered S=16 run was controlled-aborted after sustained 97-100% GPU utilization and impractical phase time; see N2C_S16_RESOURCE_ABORT_2026-08-12.md.",
        }
    }
    test_passes = []
    for size in EXECUTABLE_SIZES:
        validation_blocks = build_blocks(routes, size, VALIDATION_STARTS)
        test_blocks = build_blocks(routes, size, TEST_STARTS)
        # Q8 correctness over every physical projection.
        q8_checks = []
        for base, record in q8_records:
            for token in range(size):
                k["q8_ervf16"](((record["rows"] + 15) // 16,), (256,),
                    (x[token], q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), q8_out[token * record["rows"]:]), stream=stream)
            stream.synchronize(); expected = cp.asnumpy(q8_out[:size * record["rows"]])
            k[f"n2c_q8_temporal{size}"](((record["rows"] + 15) // 16,), (256,),
                (x, q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), q8_out), stream=stream)
            stream.synchronize(); q8_checks.append(comparison(cp.asnumpy(q8_out[:size * record["rows"]]), expected))
        q8_correct = {"planes": len(q8_checks), "elements": sum(row["elements"] for row in q8_checks),
                      "different": sum(row["different"] for row in q8_checks), "bitwise_equal": all(row["bitwise_equal"] for row in q8_checks),
                      "max_abs": max(row["max_abs"] for row in q8_checks)}
        # Q5 correctness: first validation start, five domains, all 48 layers.
        q5_checks = []
        chosen = [block for block in validation_blocks if block["start"] == VALIDATION_STARTS[0]]
        for block in chosen:
            for item in block["layers"]:
                q5_layer(size, item, False); stream.synchronize()
                expected = np.concatenate((cp.asnumpy(gate[:size * 6144]), cp.asnumpy(up[:size * 6144]), cp.asnumpy(down[:size * 16384])))
                q5_layer(size, item, True); stream.synchronize()
                observed = np.concatenate((cp.asnumpy(gate[:size * 6144]), cp.asnumpy(up[:size * 6144]), cp.asnumpy(down[:size * 16384])))
                q5_checks.append(comparison(observed, expected))
        q5_correct = {"planes": len(q5_checks), "elements": sum(row["elements"] for row in q5_checks),
                      "different": sum(row["different"] for row in q5_checks), "bitwise_equal": all(row["bitwise_equal"] for row in q5_checks),
                      "max_abs": max(row["max_abs"] for row in q5_checks)}
        union_values = [item["union_count"] for block in validation_blocks + test_blocks for item in block["layers"]]
        validation_q5 = paired_measure(stream, lambda: q5_suite(size, validation_blocks, False), lambda: q5_suite(size, validation_blocks, True), 2, 12)
        validation_combined = paired_measure(stream, lambda: combined_suite(size, validation_blocks, False), lambda: combined_suite(size, validation_blocks, True), 2, 12)
        test_opened = bool(q8_correct["bitwise_equal"] and q5_correct["bitwise_equal"] and validation_combined["p50_ratio"] <= 0.98)
        test = None
        if test_opened:
            q5_test = paired_measure(stream, lambda: q5_suite(size, test_blocks, False), lambda: q5_suite(size, test_blocks, True), 3, 30)
            combined_test = paired_measure(stream, lambda: combined_suite(size, test_blocks, False), lambda: combined_suite(size, test_blocks, True), 3, 30)
            passed = bool(combined_test["p50_ratio"] <= 0.95 and combined_test["p95_ratio"] <= 1.00)
            test = {"q5": q5_test, "combined": combined_test, "pass": passed,
                    "candidate_ms_per_token_p50": combined_test["candidate"]["stats"]["p50"] / (len(test_blocks) * size)}
            if passed:
                test_passes.append(size)
        results[str(size)] = {
            "union": stats(union_values), "q8_correctness": q8_correct, "q5_correctness": q5_correct,
            "validation": {"q5": validation_q5, "combined": validation_combined},
            "test_opened": test_opened, "test": test,
        }
        checkpoint = {
            "kind": "streamq5_moe_n2c_batch_route_union_sweep_checkpoint",
            "completed_sizes": sorted(int(key) for key, row in results.items() if row.get("validation") is not None),
            "results": results,
        }
        OUTPUT.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"size": size, "union_p50": results[str(size)]["union"]["p50"],
                          "q8_exact": q8_correct["bitwise_equal"], "q5_exact": q5_correct["bitwise_equal"],
                          "validation_q5_ratio": validation_q5["p50_ratio"],
                          "validation_combined_ratio": validation_combined["p50_ratio"], "test_opened": test_opened,
                          "test_pass": None if test is None else test["pass"]}), flush=True)
    winner = min(test_passes, key=lambda size: results[str(size)]["test"]["candidate_ms_per_token_p50"]) if test_passes else None
    result = {
        "kind": "streamq5_moe_n2c_batch_route_union_sweep", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
                   "route_capture_sha256": sha256(CAPTURE), "n1b_sha256": sha256(N1B), "q8_manifest_sha256": sha256(R / "p6a_exact_runtime_bank_result.json"),
                   "q8_pinned_aggregate_sha256": q8_sha, "sizes": SIZES, "domains": DOMAINS,
                   "validation_starts": VALIDATION_STARTS, "test_starts": TEST_STARTS,
                   "physical_q5_slots": PHYSICAL_SLOTS, "physical_q5_bytes": int(q5.size), "seed": SEED},
        "results": results, "passing_sizes": test_passes, "winner": winner, "overall_pass": bool(test_passes),
        "sweep_complete": False,
        "incomplete_reason": "S=16 blocked_by_resource_spill_timeout; S=2/4/8 completed.",
        "claim_boundary": "Physical resident Q8 plus sparse temporal Q5 component sweep on actual P4D route patterns; no causal availability, acceptance, quality, host overlap, end-to-end or external SOTA claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "passing_sizes": test_passes, "winner": winner, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
