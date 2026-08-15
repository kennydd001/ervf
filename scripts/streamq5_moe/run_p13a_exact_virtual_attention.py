from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7b_ervf_kernel import comparison, measure


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P13A_EXACT_VIRTUAL_ATTENTION_PREREGISTRATION.md"
OUTPUT = R / "p13a_exact_virtual_attention.json"
LAYERS, HEADS, KV_HEADS, HEAD_DIM, MAX_CONTEXT = 48, 32, 4, 128, 4096
CONTEXTS = (128, 512, 1024, 4096)
SEED = 130812


SOURCE = r'''
extern "C" __global__ void attention_scores_evt8(
    const float* q, const unsigned short* kv, float* scores,
    int layer, int context) {
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int item = (int)blockIdx.x * 8 + warp;
    if (item >= 32 * context) return;
    int head = item / context;
    int position = item - head * context;
    int kv_head = head >> 3;
    float partial[4];
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 4; ++virtual_index) {
        int d = lane + 32 * virtual_index;
        long long key_index = (((((long long)layer * 2LL) * 4LL + kv_head) * 4096LL + position) * 128LL + d);
        partial[virtual_index] = q[head * 128 + d] * bf16_to_float(kv[key_index]);
    }
    partial[0] += partial[2];
    partial[1] += partial[3];
    float value = partial[0] + partial[1];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffU, value, offset, 32);
    if (lane == 0) {
        float dot = round_bf16(value);
        scores[head * 4096 + position] = round_bf16(dot * 0.08838834764831845f);
    }
}

extern "C" __global__ void attention_softmax_materialize(
    float* scores, int context) {
    int head = (int)blockIdx.x;
    int d = (int)threadIdx.x;
    __shared__ float reduction[128];
    float local_max = -3.402823466e+38F;
    for (int p = d; p < context; p += 128)
        local_max = fmaxf(local_max, scores[head * 4096 + p]);
    reduction[d] = local_max; __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) reduction[d] = fmaxf(reduction[d], reduction[d + stride]);
        __syncthreads();
    }
    float maximum = reduction[0];
    float local_sum = 0.0f;
    for (int p = d; p < context; p += 128)
        local_sum += expf(scores[head * 4096 + p] - maximum);
    reduction[d] = local_sum; __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) reduction[d] += reduction[d + stride];
        __syncthreads();
    }
    float denominator = reduction[0];
    for (int p = d; p < context; p += 128)
        scores[head * 4096 + p] = round_bf16(expf(scores[head * 4096 + p] - maximum) / denominator);
}

extern "C" __global__ void attention_values_materialized(
    const float* probabilities, const unsigned short* kv, float* output,
    int layer, int context) {
    int head = (int)blockIdx.x;
    int d = (int)threadIdx.x;
    int kv_head = head >> 3;
    float value = 0.0f;
    for (int p = 0; p < context; ++p) {
        float probability = probabilities[head * 4096 + p];
        long long source = (((((long long)layer * 2LL + 1LL) * 4LL + kv_head) * 4096LL + p) * 128LL + d);
        value += probability * bf16_to_float(kv[source]);
    }
    output[head * 128 + d] = round_bf16(value);
}
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value):
    return value["stats"] | {"iterations": len(value["event_ms"])}


def main():
    names = ("attention_scores", "attention_values", "attention_scores_evt8", "attention_softmax_materialize", "attention_values_materialized")
    module = cp.RawModule(code=CUDA_SOURCE + SOURCE, options=("--std=c++11",), name_expressions=names)
    kernels = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = cp.random.RandomState(SEED)
    q = rng.standard_normal(HEADS * HEAD_DIM, dtype=cp.float32)
    kv = rng.randint(0x3E00, 0x4000, size=LAYERS * 2 * KV_HEADS * MAX_CONTEXT * HEAD_DIM, dtype=cp.uint16)
    scores = cp.empty(HEADS * MAX_CONTEXT, dtype=cp.float32)
    output = cp.empty(HEADS * HEAD_DIM, dtype=cp.float32)
    correctness = {}; validation = {}

    def original_layer(layer, context):
        kernels["attention_scores"]((HEADS * context,), (128,), (q, kv, scores, np.int32(layer), np.int32(context)), stream=stream)
        kernels["attention_values"]((HEADS,), (128,), (scores, kv, output, np.int32(layer), np.int32(context)), stream=stream)

    def evt_layer(layer, context):
        kernels["attention_scores_evt8"](((HEADS * context + 7) // 8,), (256,), (q, kv, scores, np.int32(layer), np.int32(context)), stream=stream)
        kernels["attention_softmax_materialize"]((HEADS,), (128,), (scores, np.int32(context)), stream=stream)
        kernels["attention_values_materialized"]((HEADS,), (128,), (scores, kv, output, np.int32(layer), np.int32(context)), stream=stream)

    def plane(kind, context):
        launch = original_layer if kind == "original" else evt_layer
        for layer in range(LAYERS): launch(layer, context)

    for context in CONTEXTS:
        original_values = np.empty((LAYERS, HEADS * HEAD_DIM), dtype=np.float32)
        evt_values = np.empty_like(original_values)
        score_checks = []
        for layer in range(LAYERS):
            kernels["attention_scores"]((HEADS * context,), (128,), (q, kv, scores, np.int32(layer), np.int32(context)), stream=stream)
            stream.synchronize(); original_scores = cp.asnumpy(scores.reshape(HEADS, MAX_CONTEXT)[:, :context]).copy()
            kernels["attention_values"]((HEADS,), (128,), (scores, kv, output, np.int32(layer), np.int32(context)), stream=stream)
            stream.synchronize(); original_values[layer] = cp.asnumpy(output)
            kernels["attention_scores_evt8"](((HEADS * context + 7) // 8,), (256,), (q, kv, scores, np.int32(layer), np.int32(context)), stream=stream)
            stream.synchronize(); evt_scores = cp.asnumpy(scores.reshape(HEADS, MAX_CONTEXT)[:, :context]).copy()
            score_checks.append(comparison(evt_scores, original_scores))
            kernels["attention_softmax_materialize"]((HEADS,), (128,), (scores, np.int32(context)), stream=stream)
            kernels["attention_values_materialized"]((HEADS,), (128,), (scores, kv, output, np.int32(layer), np.int32(context)), stream=stream)
            stream.synchronize(); evt_values[layer] = cp.asnumpy(output)
        correctness[str(context)] = {
            "scores_bitwise_equal_all_layers": all(item["bitwise_equal"] for item in score_checks),
            "score_differences": sum(item["different"] for item in score_checks),
            "values": comparison(evt_values, original_values),
        }
        original = measure(stream, lambda c=context: plane("original", c), 3, 30)
        evt = measure(stream, lambda c=context: plane("evt", c), 3, 30)
        validation[str(context)] = {
            "original": compact(original), "evt_pm": compact(evt),
            "p50_ratio": evt["stats"]["p50"] / original["stats"]["p50"],
            "p95_ratio": evt["stats"]["p95"] / original["stats"]["p95"],
        }
        print(json.dumps({"context": context, "correctness": correctness[str(context)], "validation": validation[str(context)]}), flush=True)
    eligible = all(value["scores_bitwise_equal_all_layers"] and value["values"]["bitwise_equal"] for value in correctness.values())
    validation_gate = validation["1024"]["p50_ratio"] <= 0.80 and validation["1024"]["p95_ratio"] <= 0.80 and validation["4096"]["p50_ratio"] <= 0.50 and validation["4096"]["p95_ratio"] <= 0.50
    test = None
    if eligible and validation_gate:
        test = {}
        for context in (1024, 4096):
            original = measure(stream, lambda c=context: plane("original", c), 10, 120)
            evt = measure(stream, lambda c=context: plane("evt", c), 10, 120)
            test[str(context)] = {
                "original": compact(original), "evt_pm": compact(evt),
                "p50_ratio": evt["stats"]["p50"] / original["stats"]["p50"],
                "p95_ratio": evt["stats"]["p95"] / original["stats"]["p95"],
            }
    test_pass = bool(test and test["1024"]["p50_ratio"] <= 0.80 and test["1024"]["p95_ratio"] <= 0.80 and test["4096"]["p50_ratio"] <= 0.50 and test["4096"]["p95_ratio"] <= 0.50)
    result = {
        "kind": "streamq5_moe_p13a_exact_virtual_attention", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)), "seed": SEED,
        "correctness": correctness, "validation": validation, "eligible": eligible,
        "validation_gate": validation_gate, "test": test, "overall_pass": test_pass,
        "claim_boundary": "Full 48-layer isolated attention-plane result; exact decoder integration remains required.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"eligible": eligible, "validation_gate": validation_gate, "test": test, "overall_pass": test_pass}, indent=2), flush=True)


if __name__ == "__main__":
    main()
