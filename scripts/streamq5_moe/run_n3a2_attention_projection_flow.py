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
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE, KV_BYTES
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q8
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, stats


R = ROOT / "reports/streamq5_moe"
BANK = R / "p6a_exact_runtime_bank_result.json"
PREREG = R / "N3A2_ATTENTION_PROJECTION_FLOW_PREREGISTRATION.md"
OUTPUT = R / "n3a2_attention_projection_flow.json"
LAYERS = 48
SEED = 120824
VALIDATION_POSITION = 1237
TEST_POSITION = 3079
CANDIDATES = ("concat_qkv", "head_flow")


SOURCE = r'''
extern "C" __global__ void n3a2_qkv_concat(
    const float* x, const unsigned char* bank,
    long long q_base, long long q_codes, long long k_base, long long k_codes,
    long long v_base, long long v_codes, float* q, float* k, float* v) {
    int group=(int)threadIdx.x>>4; int lane=(int)threadIdx.x&15;
    int global_row=(int)blockIdx.x*16+group;
    if(global_row>=5120)return;
    long long base; long long code_bytes; int row; float* output;
    if(global_row<4096){base=q_base;code_bytes=q_codes;row=global_row;output=q;}
    else if(global_row<4608){base=k_base;code_bytes=k_codes;row=global_row-4096;output=k;}
    else{base=v_base;code_bytes=v_codes;row=global_row-4608;output=v;}
    const signed char* codes=(const signed char*)(bank+base);
    const unsigned short* scales=(const unsigned short*)(bank+base+code_bytes);
    float value=q8_ervf_row<16>(x,codes,scales,row,2048,lane);
    if(lane==0)output[row]=round_bf16(value);
}

extern "C" __global__ void n3a2_head_flow(
    const float* x, const unsigned char* bank,
    long long q_base, long long q_codes, long long k_base, long long k_codes,
    long long v_base, long long v_codes,
    const unsigned short* q_weight, const unsigned short* k_weight,
    float* q, float* k, float* v, unsigned short* kv,
    int layer, int position) {
    int head=(int)blockIdx.x; int tid=(int)threadIdx.x;
    int group=tid>>4; int lane=tid&15; bool is_q=head<32;
    int local_head=is_q?head:head-32;
    long long proj_base=is_q?q_base:k_base;
    long long proj_codes=is_q?q_codes:k_codes;
    const signed char* codes=(const signed char*)(bank+proj_base);
    const unsigned short* scales=(const unsigned short*)(bank+proj_base+proj_codes);
    __shared__ float projected[128];
    __shared__ float squares[128];
    __shared__ float normed[128];
    for(int wave=0;wave<8;++wave){
        int d=wave*16+group; int row=local_head*128+d;
        float value=q8_ervf_row<16>(x,codes,scales,row,2048,lane);
        if(lane==0)projected[d]=round_bf16(value);
    }
    __syncthreads();
    if(tid<128)squares[tid]=projected[tid]*projected[tid];
    __syncthreads();
    for(int stride=64;stride>0;stride>>=1){
        if(tid<stride)squares[tid]+=squares[tid+stride];
        __syncthreads();
    }
    if(tid<128){
        const unsigned short* weight=is_q?q_weight:k_weight;
        float normalized=round_bf16(projected[tid]*rsqrtf(squares[0]/128.0f+1.0e-6f));
        normed[tid]=round_bf16(normalized*bf16_to_float(weight[tid]));
    }
    __syncthreads();
    if(tid<128){
        int frequency=tid&63;
        float angle=((float)position)/powf(1000000.0f,((float)(2*frequency))/128.0f);
        float cosine=round_bf16(cosf(angle));float sine=round_bf16(sinf(angle));
        int partner=tid<64?tid+64:tid-64;
        float rotated=tid<64?-normed[partner]:normed[partner];
        float output=round_bf16(round_bf16(normed[tid]*cosine)+round_bf16(rotated*sine));
        if(is_q)q[local_head*128+tid]=output;
        else{
            k[local_head*128+tid]=output;
            long long target=(((((long long)layer*2LL)*4LL+local_head)*4096LL+position)*128LL+tid);
            kv[target]=float_to_bf16(output);
        }
    }
    if(is_q)return;
    __syncthreads();
    codes=(const signed char*)(bank+v_base);
    scales=(const unsigned short*)(bank+v_base+v_codes);
    for(int wave=0;wave<8;++wave){
        int d=wave*16+group;int row=local_head*128+d;
        float value=q8_ervf_row<16>(x,codes,scales,row,2048,lane);
        if(lane==0)projected[d]=round_bf16(value);
    }
    __syncthreads();
    if(tid<128){
        float output=projected[tid];v[local_head*128+tid]=output;
        long long target=(((((long long)layer*2LL+1LL)*4LL+local_head)*4096LL+position)*128LL+tid);
        kv[target]=float_to_bf16(output);
    }
}
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def rotated_measure(stream, launches, warmups: int, iterations: int):
    names = list(launches)
    for iteration in range(warmups):
        for shift in range(len(names)):
            launches[names[(iteration + shift) % len(names)]]()
    stream.synchronize()
    values = {name: [] for name in names}
    for iteration in range(iterations):
        order = [names[(iteration + shift) % len(names)] for shift in range(len(names))]
        if iteration % 2:
            order.reverse()
        for name in order:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record(stream); launches[name](); end.record(stream); end.synchronize()
            values[name].append(float(cp.cuda.get_elapsed_time(begin, end)))
    return {name: {"event_ms": rows, "stats": stats(rows)} for name, rows in values.items()}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    records = {(row["layer"], row["name"]): (base, row) for base, row in q8_records}
    norm_bits = np.fromfile(ROOT / manifest["norm_bank"]["artifact"], dtype="<u2")
    norms = cp.asarray(norm_bits)
    norm_records = {(row["layer"], row["name"]): row for row in manifest["norm_bank"]["records"]}

    def norm(layer: int, name: str):
        return norms[norm_records[(layer, name)]["offset"] // 2:]

    names = ("rmsnorm", "q8_ervf16", "qk_norm_rope_write", "write_v", "n3a2_qkv_concat", "n3a2_head_flow")
    module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + SOURCE, options=("--std=c++11",), name_expressions=names)
    kfun = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    states = cp.asarray(rng.standard_normal((LAYERS, 2048), dtype=np.float32))
    normed = cp.empty(2048, dtype=cp.float32)
    q = cp.empty(4096, dtype=cp.float32); key = cp.empty(512, dtype=cp.float32); value = cp.empty(512, dtype=cp.float32)
    kv_memory = cp.cuda.alloc(KV_BYTES)
    kv = cp.ndarray((KV_BYTES // 2,), dtype=cp.uint16, memptr=kv_memory)

    def record_args(layer: int):
        qb, qr = records[(layer, "q")]; kb, kr = records[(layer, "k")]; vb, vr = records[(layer, "v")]
        return qb, qr, kb, kr, vb, vr

    def layer_flow(kind: str, layer: int, position: int) -> None:
        qb, qr, kb, kr, vb, vr = record_args(layer)
        kfun["rmsnorm"]((1,), (256,), (states[layer], norm(layer, "input"), normed, np.int32(2048)), stream=stream)
        if kind == "baseline":
            for base, row, output in ((qb, qr, q), (kb, kr, key), (vb, vr, value)):
                kfun["q8_ervf16"](((row["rows"] + 15) // 16,), (256,),
                    (normed, q8, np.int64(base), np.int64(row["code_bytes"]), np.int32(row["rows"]), np.int32(2048), output), stream=stream)
            kfun["qk_norm_rope_write"]((36,), (128,), (q, key, norm(layer, "q_norm"), norm(layer, "k_norm"), kv, np.int32(layer), np.int32(position)), stream=stream)
            kfun["write_v"]((2,), (256,), (value, kv, np.int32(layer), np.int32(position)), stream=stream)
        elif kind == "concat_qkv":
            kfun["n3a2_qkv_concat"]((320,), (256,),
                (normed, q8, np.int64(qb), np.int64(qr["code_bytes"]), np.int64(kb), np.int64(kr["code_bytes"]), np.int64(vb), np.int64(vr["code_bytes"]), q, key, value), stream=stream)
            kfun["qk_norm_rope_write"]((36,), (128,), (q, key, norm(layer, "q_norm"), norm(layer, "k_norm"), kv, np.int32(layer), np.int32(position)), stream=stream)
            kfun["write_v"]((2,), (256,), (value, kv, np.int32(layer), np.int32(position)), stream=stream)
        else:
            kfun["n3a2_head_flow"]((36,), (256,),
                (normed, q8, np.int64(qb), np.int64(qr["code_bytes"]), np.int64(kb), np.int64(kr["code_bytes"]), np.int64(vb), np.int64(vr["code_bytes"]), norm(layer, "q_norm"), norm(layer, "k_norm"), q, key, value, kv, np.int32(layer), np.int32(position)), stream=stream)

    def plane(kind: str, position: int) -> None:
        for layer in range(LAYERS):
            layer_flow(kind, layer, position)

    def capture(kind: str, position: int):
        outputs = np.empty((LAYERS, 4096 + 512 + 512), dtype=np.float32)
        kvbits = np.empty((LAYERS, 2, 4, 128), dtype=np.uint16)
        view = kv.reshape(LAYERS, 2, 4, 4096, 128)
        for layer in range(LAYERS):
            layer_flow(kind, layer, position); stream.synchronize()
            outputs[layer] = np.concatenate((cp.asnumpy(q), cp.asnumpy(key), cp.asnumpy(value)))
            kvbits[layer] = cp.asnumpy(view[layer, :, :, position, :])
        return outputs, kvbits

    baseline_outputs, baseline_kv = capture("baseline", VALIDATION_POSITION)
    correctness = {}
    for candidate in CANDIDATES:
        outputs, kvbits = capture(candidate, VALIDATION_POSITION)
        correctness[candidate] = {
            "outputs": comparison(outputs, baseline_outputs),
            "kv_bitwise_equal": bool(np.array_equal(kvbits, baseline_kv)),
            "kv_elements": int(baseline_kv.size),
            "kv_different": int(np.count_nonzero(kvbits != baseline_kv)),
        }
    launches = {name: (lambda n=name: plane(n, VALIDATION_POSITION)) for name in ("baseline",) + CANDIDATES}
    validation = rotated_measure(stream, launches, 5, 30)
    eligible = [candidate for candidate in CANDIDATES if correctness[candidate]["outputs"]["bitwise_equal"] and correctness[candidate]["kv_bitwise_equal"]]
    selected = min(eligible, key=lambda name: validation[name]["stats"]["p50"]) if eligible else None
    validation_ratio = validation[selected]["stats"]["p50"] / validation["baseline"]["stats"]["p50"] if selected else None
    test_opened = bool(selected and validation_ratio <= 0.98)
    test_correctness = None; test = None
    if test_opened:
        baseline_outputs, baseline_kv = capture("baseline", TEST_POSITION)
        outputs, kvbits = capture(selected, TEST_POSITION)
        test_correctness = {"outputs": comparison(outputs, baseline_outputs),
                            "kv_bitwise_equal": bool(np.array_equal(kvbits, baseline_kv)),
                            "kv_elements": int(baseline_kv.size), "kv_different": int(np.count_nonzero(kvbits != baseline_kv))}
        measurements = rotated_measure(stream, {"baseline": lambda: plane("baseline", TEST_POSITION), selected: lambda: plane(selected, TEST_POSITION)}, 10, 120)
        p50_ratio = measurements[selected]["stats"]["p50"] / measurements["baseline"]["stats"]["p50"]
        p95_ratio = measurements[selected]["stats"]["p95"] / measurements["baseline"]["stats"]["p95"]
        test = {"measurements": measurements, "p50_ratio": p50_ratio, "p95_ratio": p95_ratio,
                "speedup_p50": 1.0 / p50_ratio,
                "pass": bool(p50_ratio <= 0.97 and p95_ratio <= 1.00 and test_correctness["outputs"]["bitwise_equal"] and test_correctness["kv_bitwise_equal"])}
    props = cp.cuda.runtime.getDeviceProperties(0)
    result = {"kind": "streamq5_moe_n3a2_attention_projection_flow", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)), "bank_sha256": sha256(BANK),
                         "q8_pinned_aggregate_sha256": q8_sha, "seed": SEED, "layers": LAYERS,
                         "validation_position": VALIDATION_POSITION, "test_position": TEST_POSITION},
              "device": {"name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"]},
              "correctness": correctness, "validation": validation, "selected": selected,
              "validation_p50_ratio": validation_ratio, "test_opened": test_opened,
              "test_correctness": test_correctness, "test": test, "overall_pass": bool(test and test["pass"]),
              "claim_boundary": "Physical resident 48-layer RMSNorm/QKV/QK-norm/RoPE/KV-write component flow only; no attention values, expert path, full decoder, quality, cross-GPU or SOTA claim."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "correctness": correctness, "selected": selected,
                      "validation_p50_ratio": validation_ratio, "test_opened": test_opened,
                      "test": None if test is None else {key: value for key, value in test.items() if key != "measurements"},
                      "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
