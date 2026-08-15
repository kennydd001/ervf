from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_qwen_decoder_layer
from scripts.streamq5_moe.run_p0c_model_quality import forward_layer, selected_embeddings


MODEL = ROOT / "models/qwen3-30b-a3b-base"
DATA = ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors"
DATA_LOCK = ROOT / "reports/streamq5_moe/p0c_input_lock.json"
PREREG = ROOT / "reports/streamq5_moe/P10D_GPU_ROUTER_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/streamq5_moe/p10d_gpu_router.json"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
REPEATS = 20


CUDA = r'''
__device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int lsb = (bits >> 16) & 1U;
    return (unsigned short)((bits + 0x7FFFU + lsb) >> 16);
}
__device__ __forceinline__ float round_bf16(float value) {
    return __uint_as_float(((unsigned int)float_to_bf16(value)) << 16);
}
__device__ __forceinline__ bool greater_pair(float av, int ai, float bv, int bi) {
    return (av > bv) || (av == bv && ai < bi);
}
extern "C" __global__ void route_top8(
    const float* logits, int* output_ids, float* output_weights) {
    __shared__ float values[128];
    __shared__ int ids[128];
    int i = (int)threadIdx.x;
    values[i] = logits[i]; ids[i] = i; __syncthreads();
    for (int size = 2; size <= 128; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int partner = i ^ stride;
            if (partner > i) {
                bool ascending = (i & size) == 0;
                float av = values[i], bv = values[partner];
                int ai = ids[i], bi = ids[partner];
                bool a_greater = greater_pair(av, ai, bv, bi);
                bool b_greater = greater_pair(bv, bi, av, ai);
                if ((ascending && a_greater) || (!ascending && b_greater)) {
                    values[i] = bv; ids[i] = bi;
                    values[partner] = av; ids[partner] = ai;
                }
            }
            __syncthreads();
        }
    }
    if (i == 0) {
        float maximum = values[127];
        float denominator = 0.0f;
        float probabilities[8];
        for (int j = 0; j < 128; ++j)
            denominator += expf(logits[j] - maximum);
        float selected_sum = 0.0f;
        for (int rank = 0; rank < 8; ++rank) {
            int source = 127 - rank;
            output_ids[rank] = ids[source];
            probabilities[rank] = expf(values[source] - maximum) / denominator;
            selected_sum += probabilities[rank];
        }
        for (int rank = 0; rank < 8; ++rank)
            output_weights[rank] = round_bf16(probabilities[rank] / selected_sum);
    }
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_round(values: np.ndarray) -> np.ndarray:
    work = np.asarray(values, dtype=np.float32).copy()
    bits = work.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return ((rounded >> 16) << 16).view(np.float32)


def cpu_route(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.exp(logits - np.max(logits), dtype=np.float32)
    probabilities /= np.sum(probabilities, dtype=np.float32)
    ids = np.argsort(-probabilities, kind="stable")[:8].astype(np.int32)
    weights = probabilities[ids]
    weights /= np.sum(weights, dtype=np.float32)
    return ids, bf16_round(weights)


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
        "max": float(array.max()), "samples": int(array.size),
    }


@torch.inference_mode()
def capture_logits() -> np.ndarray:
    lock = json.loads(DATA_LOCK.read_text(encoding="utf-8"))
    if sha256(DATA) != lock["artifact_sha256"]:
        raise ValueError("P0C data provenance mismatch")
    source = load_file(DATA)
    ids = torch.cat([source[f"test_{domain}"] for domain in DOMAINS], 0).long()
    if ids.shape != (10, 128):
        raise RuntimeError(tuple(ids.shape))
    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    weight_map = checkpoint_weight_map(MODEL)
    hidden = selected_embeddings(MODEL, ids, device, weight_map, None)
    captured = []
    for layer_index in range(48):
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        sample = hidden[:, -1, :].to(device)
        captured.append(layer.mlp.gate(sample).float().cpu().numpy())
        hidden = forward_layer(layer, rotary, hidden, device)
        print(json.dumps({"captured_layer": layer_index}), flush=True)
        del layer, rotary, sample
        gc.collect(); torch.cuda.empty_cache()
    result = np.concatenate(captured, axis=0).astype(np.float32, copy=False)
    if result.shape != (480, 128) or not np.isfinite(result).all():
        raise RuntimeError("invalid router capture")
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    logits = capture_logits()
    logits_sha = hashlib.sha256(logits.tobytes()).hexdigest()
    module = cp.RawModule(code=CUDA, options=("--std=c++11",), name_expressions=("route_top8",))
    kernel = module.get_function("route_top8")
    device_logits = cp.asarray(logits)
    device_ids = cp.empty(8, dtype=cp.int32)
    device_weights = cp.empty(8, dtype=cp.float32)
    cpu_memory = cp.cuda.alloc_pinned_memory(128 * 4)
    cpu_host = np.frombuffer(cpu_memory, dtype=np.float32, count=128)
    ids_memory = cp.cuda.alloc_pinned_memory(8 * 4)
    weights_memory = cp.cuda.alloc_pinned_memory(8 * 4)
    gpu_ids = np.frombuffer(ids_memory, dtype=np.int32, count=8)
    gpu_weights = np.frombuffer(weights_memory, dtype=np.float32, count=8)
    stream = cp.cuda.Stream(non_blocking=True)

    exact_ids = 0; exact_weights = 0; maximum_weight_error = 0.0; maximum_sum_error = 0.0
    for index in range(logits.shape[0]):
        row = device_logits[index]
        expected_ids, expected_weights = cpu_route(logits[index])
        kernel((1,), (128,), (row, device_ids, device_weights), stream=stream)
        cp.cuda.runtime.memcpyAsync(ids_memory.ptr, device_ids.data.ptr, 32, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        cp.cuda.runtime.memcpyAsync(weights_memory.ptr, device_weights.data.ptr, 32, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        stream.synchronize()
        exact_ids += int(np.array_equal(gpu_ids, expected_ids))
        exact_weights += int(np.array_equal(gpu_weights.view(np.uint32), expected_weights.view(np.uint32)))
        maximum_weight_error = max(maximum_weight_error, float(np.max(np.abs(gpu_weights - expected_weights))))
        maximum_sum_error = max(maximum_sum_error, abs(float(gpu_weights.sum(dtype=np.float32)) - 1.0))

    cpu_ms: list[float] = []; gpu_ms: list[float] = []; event_ms: list[float] = []
    order = np.arange(logits.shape[0])
    for warmup in range(3 + REPEATS):
        for index in order:
            row = device_logits[index]
            started = time.perf_counter_ns()
            cp.cuda.runtime.memcpyAsync(cpu_memory.ptr, row.data.ptr, 512, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
            stream.synchronize(); cpu_route(cpu_host)
            elapsed_cpu = (time.perf_counter_ns() - started) / 1e6

            begin = cp.cuda.Event(); end = cp.cuda.Event()
            started = time.perf_counter_ns(); begin.record(stream)
            kernel((1,), (128,), (row, device_ids, device_weights), stream=stream)
            end.record(stream)
            cp.cuda.runtime.memcpyAsync(ids_memory.ptr, device_ids.data.ptr, 32, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
            cp.cuda.runtime.memcpyAsync(weights_memory.ptr, device_weights.data.ptr, 32, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
            stream.synchronize()
            elapsed_gpu = (time.perf_counter_ns() - started) / 1e6
            elapsed_event = cp.cuda.get_elapsed_time(begin, end)
            if warmup >= 3:
                cpu_ms.append(elapsed_cpu); gpu_ms.append(elapsed_gpu); event_ms.append(elapsed_event)

    cpu_stats, gpu_stats, event_stats = stats(cpu_ms), stats(gpu_ms), stats(event_ms)
    gates = {
        "ids_exact_all_480": exact_ids == 480,
        "bf16_weights_exact_all_480": exact_weights == 480,
        "weight_sum_error_le_0_02": maximum_sum_error <= 0.02,
        "host_p50_le_90pct": gpu_stats["p50"] <= 0.90 * cpu_stats["p50"],
        "host_p95_le_90pct": gpu_stats["p95"] <= 0.90 * cpu_stats["p95"],
    }
    result = {
        "kind": "streamq5_moe_p10d_gpu_router", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"preregistration_sha256": sha256(PREREG), "data_sha256": sha256(DATA),
                   "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"), "captured_logits_sha256": logits_sha},
        "workload": {"vectors": 480, "logits_per_vector": 128, "top_k": 8, "repeats": REPEATS,
                     "timed_samples_per_path": len(cpu_ms)},
        "correctness": {"exact_id_vectors": exact_ids, "exact_bf16_weight_vectors": exact_weights,
                        "maximum_weight_abs_error": maximum_weight_error, "maximum_weight_sum_error": maximum_sum_error},
        "timing_ms": {"cpu_route_host": cpu_stats, "gpu_route_host": gpu_stats, "gpu_kernel_event": event_stats,
                      "host_p50_ratio": gpu_stats["p50"] / cpu_stats["p50"],
                      "host_p95_ratio": gpu_stats["p95"] / cpu_stats["p95"]},
        "gates": gates, "overall_pass": all(gates.values()),
        "claim_boundary": "Real-logit isolated route barrier only; cache decisions, expert copies and end-to-end prediction/timing are not part of this result.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "correctness": result["correctness"], "timing_ms": result["timing_ms"], "gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
