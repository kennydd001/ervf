from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import (
    BANK_BYTES,
    BANK_RESULT_PATH,
    CACHE_BYTES,
    EXPERT_BYTES,
    LAYER_BYTES,
    LAYERS,
    STATIC_SLOTS,
    bases,
    copy_expert,
    dynamic_slots,
    pin_bank as pin_expert_bank,
)


MODEL = ROOT / "models/qwen3-30b-a3b-base"
R = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe"
PREREG = R / "P6A_END_TO_END_DECODE_PREREGISTRATION.md"
BANK_RESULT = R / "p6a_exact_runtime_bank_result.json"
BANK_VERIFY = R / "p6a_exact_runtime_bank_verification.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
P0C_INPUT_LOCK = R / "p0c_input_lock.json"
P0C_DATA = RUNS / "p0c_fresh_input_ids.safetensors"
P0C_VALIDATION = R / "p0c_validation_model_quality.json"
P0C_TEST = R / "p0c_test_model_quality.json"
P4D_CAPTURE = R / "p4d_route_capture_result.json"
P4D_ROUTES = RUNS / "p4d_routes"
INPUT_LOCK = R / "p6a_end_to_end_input_lock.json"
EVALUATOR_LOCK = R / "p6a_end_to_end_evaluator_lock.json"

HIDDEN = 2048
Q_SIZE = 4096
KV_SIZE = 512
HEAD_DIM = 128
Q_HEADS = 32
KV_HEADS = 4
EXPERTS = 128
TOP_K = 8
INTERMEDIATE = 768
VOCAB = 151936
MAX_CONTEXT = 4096
KV_BYTES = LAYERS * 2 * KV_HEADS * MAX_CONTEXT * HEAD_DIM * 2
DOMAINS = ("general", "code", "math", "multilingual", "instruction")


CUDA_SOURCE = r'''
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}
__device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int lsb = (bits >> 16) & 1U;
    return (unsigned short)((bits + 0x7FFFU + lsb) >> 16);
}
__device__ __forceinline__ float round_bf16(float value) {
    return bf16_to_float(float_to_bf16(value));
}
__device__ __forceinline__ float q5_dot(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int tid) {
    int packs_per_row = cols >> 3;
    int groups_per_row = cols >> 7;
    float sum = 0.0f;
    for (int pack = tid; pack < packs_per_row; pack += blockDim.x) {
        const unsigned char* source = packed + ((long long)row * packs_per_row + pack) * 5;
        unsigned long long word = ((unsigned long long)source[0])
            | ((unsigned long long)source[1] << 8)
            | ((unsigned long long)source[2] << 16)
            | ((unsigned long long)source[3] << 24)
            | ((unsigned long long)source[4] << 32);
        int column = pack << 3;
        float scale = bf16_to_float(scales[row * groups_per_row + (column >> 7)]);
        #pragma unroll
        for (int item = 0; item < 8; ++item) {
            int code = ((word >> (item * 5)) & 31ULL) - 15;
            float weight = round_bf16(((float)code) * scale);
            sum += weight * x[column + item];
        }
    }
    return sum;
}
extern "C" __global__ void q8_gemv(
    const float* x, const unsigned char* bank, long long base,
    long long code_bytes, int rows, int cols, float* output) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    const signed char* codes = (const signed char*)(bank + base);
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes);
    int groups_per_row = cols >> 7;
    float sum = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += blockDim.x) {
        float scale = bf16_to_float(scales[row * groups_per_row + (col >> 7)]);
        float weight = round_bf16(((float)codes[(long long)row * cols + col]) * scale);
        sum += weight * x[col];
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = round_bf16(reduction[0]);
}
extern "C" __global__ void rmsnorm(
    const float* input, const unsigned short* weight, float* output, int n) {
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < n; i += blockDim.x) sum += input[i] * input[i];
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reduction[0] / ((float)n) + 1.0e-6f);
    for (int i = (int)threadIdx.x; i < n; i += blockDim.x) {
        float normalized = round_bf16(input[i] * inverse);
        output[i] = round_bf16(normalized * bf16_to_float(weight[i]));
    }
}
extern "C" __global__ void qk_norm_rope_write(
    float* q, float* k, const unsigned short* q_weight, const unsigned short* k_weight,
    unsigned short* kv, int layer, int position) {
    int head = (int)blockIdx.x;
    bool is_q = head < 32;
    int local_head = is_q ? head : head - 32;
    float* source = is_q ? q + local_head * 128 : k + local_head * 128;
    const unsigned short* weight = is_q ? q_weight : k_weight;
    int d = (int)threadIdx.x;
    __shared__ float values[128];
    __shared__ float squares[128];
    float value = source[d];
    squares[d] = value * value; __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) squares[d] += squares[d + stride];
        __syncthreads();
    }
    float normalized = round_bf16(value * rsqrtf(squares[0] / 128.0f + 1.0e-6f));
    values[d] = round_bf16(normalized * bf16_to_float(weight[d])); __syncthreads();
    int frequency = d & 63;
    float angle = ((float)position) / powf(1000000.0f, ((float)(2 * frequency)) / 128.0f);
    float cosine = round_bf16(cosf(angle));
    float sine = round_bf16(sinf(angle));
    int partner = d < 64 ? d + 64 : d - 64;
    float rotated = d < 64 ? -values[partner] : values[partner];
    float output = round_bf16(round_bf16(values[d] * cosine) + round_bf16(rotated * sine));
    source[d] = output;
    if (!is_q) {
        long long index = (((((long long)layer * 2LL) * 4LL + local_head) * 4096LL + position) * 128LL + d);
        kv[index] = float_to_bf16(output);
    }
}
extern "C" __global__ void write_v(
    const float* v, unsigned short* kv, int layer, int position) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < 512) {
        int head = index >> 7;
        int d = index & 127;
        long long target = (((((long long)layer * 2LL + 1LL) * 4LL + head) * 4096LL + position) * 128LL + d);
        kv[target] = float_to_bf16(v[index]);
    }
}
extern "C" __global__ void attention_scores(
    const float* q, const unsigned short* kv, float* scores,
    int layer, int context) {
    int item = (int)blockIdx.x;
    int head = item / context;
    int position = item - head * context;
    int kv_head = head >> 3;
    int d = (int)threadIdx.x;
    long long key_index = (((((long long)layer * 2LL) * 4LL + kv_head) * 4096LL + position) * 128LL + d);
    float part = q[head * 128 + d] * bf16_to_float(kv[key_index]);
    __shared__ float reduction[128];
    reduction[d] = part; __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) reduction[d] += reduction[d + stride];
        __syncthreads();
    }
    if (d == 0) {
        float dot = round_bf16(reduction[0]);
        scores[head * 4096 + position] = round_bf16(dot * 0.08838834764831845f);
    }
}
extern "C" __global__ void attention_values(
    const float* scores, const unsigned short* kv, float* output,
    int layer, int context) {
    int head = (int)blockIdx.x;
    int d = (int)threadIdx.x;
    int kv_head = head >> 3;
    __shared__ float reduction[128];
    float local_max = -3.402823466e+38F;
    for (int p = d; p < context; p += 128) local_max = fmaxf(local_max, scores[head * 4096 + p]);
    reduction[d] = local_max; __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) reduction[d] = fmaxf(reduction[d], reduction[d + stride]);
        __syncthreads();
    }
    float maximum = reduction[0];
    float local_sum = 0.0f;
    for (int p = d; p < context; p += 128) local_sum += expf(scores[head * 4096 + p] - maximum);
    reduction[d] = local_sum; __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) reduction[d] += reduction[d + stride];
        __syncthreads();
    }
    float denominator = reduction[0];
    float value = 0.0f;
    for (int p = 0; p < context; ++p) {
        float probability = round_bf16(expf(scores[head * 4096 + p] - maximum) / denominator);
        long long source = (((((long long)layer * 2LL + 1LL) * 4LL + kv_head) * 4096LL + p) * 128LL + d);
        value += probability * bf16_to_float(kv[source]);
    }
    output[head * 128 + d] = round_bf16(value);
}
extern "C" __global__ void residual_add(
    const float* residual, const float* update, float* output, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) output[i] = round_bf16(residual[i] + update[i]);
}
extern "C" __global__ void q5_gate_up_n(
    const float* x, const unsigned char* cache, const int* slots,
    const int* positions, float* gate, float* up) {
    int local_expert = (int)blockIdx.x / 1536;
    int local = (int)blockIdx.x - local_expert * 1536;
    int projection = local >= 768;
    int row = local - projection * 768;
    int output_expert = positions[local_expert];
    long long base = (long long)slots[local_expert] * 3035136LL + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float sum = q5_dot(x, packed, scales, row, 2048, (int)threadIdx.x);
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (projection) up[output_expert * 768 + row] = round_bf16(reduction[0]);
        else gate[output_expert * 768 + row] = round_bf16(reduction[0]);
    }
}
extern "C" __global__ void swiglu_n(float* gate, const float* up, const int* positions) {
    int local = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int local_expert = local / 768;
    int column = local - local_expert * 768;
    int output_index = positions[local_expert] * 768 + column;
    float value = gate[output_index];
    float silu = round_bf16(value / (1.0f + expf(-value)));
    gate[output_index] = round_bf16(silu * up[output_index]);
}
extern "C" __global__ void q5_down_n(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {
    int local_expert = (int)blockIdx.x / 2048;
    int row = (int)blockIdx.x - local_expert * 2048;
    int output_expert = positions[local_expert];
    long long base = (long long)slots[local_expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float sum = q5_dot(activation + output_expert * 768, packed, scales, row, 768, (int)threadIdx.x);
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down[output_expert * 2048 + row] = round_bf16(reduction[0]);
}
extern "C" __global__ void weighted_residual(
    const float* down, const int* order, const float* weights,
    const float* residual, float* state) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < 2048) {
        float sum = 0.0f;
        #pragma unroll
        for (int item = 0; item < 8; ++item) {
            int position = order[item];
            float term = round_bf16(down[position * 2048 + index] * weights[position]);
            sum = round_bf16(sum + term);
        }
        state[index] = round_bf16(residual[index] + sum);
    }
}
extern "C" __global__ void logits_stats(
    const float* logits, int target, float* values, int* argmax_out) {
    int tid = (int)threadIdx.x;
    float local_max = -3.402823466e+38F;
    int local_index = 0;
    for (int i = tid; i < 151936; i += 256) {
        float value = logits[i];
        if (value > local_max || (value == local_max && i < local_index)) { local_max = value; local_index = i; }
    }
    __shared__ float maxima[256];
    __shared__ int indices[256];
    maxima[tid] = local_max; indices[tid] = local_index; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = maxima[tid + stride]; int other_index = indices[tid + stride];
            if (other > maxima[tid] || (other == maxima[tid] && other_index < indices[tid])) {
                maxima[tid] = other; indices[tid] = other_index;
            }
        }
        __syncthreads();
    }
    float maximum = maxima[0];
    float sum = 0.0f;
    for (int i = tid; i < 151936; i += 256) sum += expf(logits[i] - maximum);
    maxima[tid] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) maxima[tid] += maxima[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        values[0] = maximum + logf(maxima[0]);
        values[1] = target >= 0 ? logits[target] : 0.0f;
        argmax_out[0] = indices[0];
    }
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def stats(values) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def bf16_round(value):
    x = np.asarray(value, dtype=np.float32).copy()
    bits = x.view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def bits_to_float(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def load_static_sets():
    routes = {domain: [] for domain in DOMAINS}
    hashes = {}
    for layer in range(LAYERS):
        path = P4D_ROUTES / f"layer_{layer:02d}.safetensors"
        hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS:
                routes[domain].append(handle.get_tensor(f"{domain}_router_ids").astype(np.int64))
    selected = {domain: [] for domain in DOMAINS}
    for domain in DOMAINS:
        for layer in range(LAYERS):
            counts = np.bincount(routes[domain][layer][:512].reshape(-1), minlength=EXPERTS)
            selected[domain].append(tuple(int(x) for x in np.lexsort((np.arange(EXPERTS), -counts))[:STATIC_SLOTS]))
    return selected, hashes


def pin_q8_bank(bank):
    total = bank["aggregate"]["bytes"]
    memory = cp.cuda.alloc_pinned_memory(total)
    host = np.frombuffer(memory, dtype=np.uint8, count=total)
    host_offsets = {}
    cursor = 0
    digest = hashlib.sha256()
    for index, record in enumerate(bank["records"]):
        path = ROOT / record["artifact"]
        target = memoryview(host[cursor:cursor + record["bytes"]])
        with path.open("rb") as handle:
            if handle.readinto(target) != record["bytes"]:
                raise RuntimeError("short P6 bank read")
        raw = memoryview(host[cursor:cursor + record["bytes"]])
        if hashlib.sha256(raw).hexdigest() != record["artifact_sha256"]:
            raise ValueError("P6 pinned record hash mismatch")
        digest.update(raw)
        host_offsets[index] = cursor
        cursor += record["bytes"]
    if cursor != total:
        raise RuntimeError("P6 pinned byte mismatch")
    return memory, host, host_offsets, digest.hexdigest()


class Runtime:
    def __init__(self, lock):
        self.lock = lock
        self.bank = json.loads(BANK_RESULT.read_text(encoding="utf-8"))
        self.expert_bank = json.loads(BANK_RESULT_PATH.read_text(encoding="utf-8"))
        self.fixed, self.route_hashes = load_static_sets()
        if self.route_hashes != lock["p4d_route_artifact_sha256"]:
            raise ValueError("P4D route artifact provenance mismatch")
        started = time.perf_counter()
        self.expert_pinned, self.expert_hashes, expert_pin_ms = pin_expert_bank(self.expert_bank)
        self.q8_pinned, self.q8_host, self.q8_host_offsets, self.q8_pinned_sha = pin_q8_bank(self.bank)
        self.pin_ms = (time.perf_counter() - started) * 1000.0
        self.expert_pin_ms = expert_pin_ms
        cp.get_default_memory_pool().free_all_blocks()
        self.free_before, self.total_vram = cp.cuda.runtime.memGetInfo()
        self.compute = cp.cuda.Stream(non_blocking=True)
        self.copy = cp.cuda.Stream(non_blocking=True)
        self.expert_cache_memory = cp.cuda.alloc(CACHE_BYTES)
        self.expert_cache = cp.ndarray((CACHE_BYTES,), dtype=cp.uint8, memptr=self.expert_cache_memory)
        self.trunk_memory = cp.cuda.alloc(self.bank["aggregate"]["device_bytes"])
        self.trunk = cp.ndarray((self.bank["aggregate"]["device_bytes"],), dtype=cp.uint8, memptr=self.trunk_memory)
        self.kv_memory = cp.cuda.alloc(KV_BYTES)
        self.kv = cp.ndarray((KV_BYTES // 2,), dtype=cp.uint16, memptr=self.kv_memory)
        self.layer_bases = bases()
        self.record_by_key = {}
        self.device_offsets = {}
        device_cursor = 0
        for index, record in enumerate(self.bank["records"]):
            self.record_by_key[(record["layer"], record["name"])] = (index, record)
            if record["residency"] == "device":
                self.device_offsets[(record["layer"], record["name"])] = device_cursor
                cp.cuda.runtime.memcpyAsync(
                    self.trunk_memory.ptr + device_cursor,
                    self.q8_pinned.ptr + self.q8_host_offsets[index],
                    record["bytes"], cp.cuda.runtime.memcpyHostToDevice, self.compute.ptr,
                )
                device_cursor += record["bytes"]
        if device_cursor != self.bank["aggregate"]["device_bytes"]:
            raise RuntimeError("device trunk byte mismatch")
        cp.cuda.runtime.memsetAsync(self.expert_cache_memory.ptr, 0, CACHE_BYTES, self.compute.ptr)
        cp.cuda.runtime.memsetAsync(self.kv_memory.ptr, 0, KV_BYTES, self.compute.ptr)
        norm_bits = np.fromfile(ROOT / self.bank["norm_bank"]["artifact"], dtype="<u2")
        self.norms = cp.asarray(norm_bits)
        self.norm_by_key = {(r["layer"], r["name"]): r for r in self.bank["norm_bank"]["records"]}
        self.compute.synchronize()
        self.free_after_fixed, _ = cp.cuda.runtime.memGetInfo()
        module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11",), name_expressions=(
            "q8_gemv", "rmsnorm", "qk_norm_rope_write", "write_v", "attention_scores",
            "attention_values", "residual_add", "q5_gate_up_n", "swiglu_n", "q5_down_n",
            "weighted_residual", "logits_stats",
        ))
        self.k = {name: module.get_function(name) for name in (
            "q8_gemv", "rmsnorm", "qk_norm_rope_write", "write_v", "attention_scores",
            "attention_values", "residual_add", "q5_gate_up_n", "swiglu_n", "q5_down_n",
            "weighted_residual", "logits_stats",
        )}
        self.state = cp.empty(HIDDEN, dtype=cp.float32)
        self.residual = cp.empty(HIDDEN, dtype=cp.float32)
        self.normed = cp.empty(HIDDEN, dtype=cp.float32)
        self.q = cp.empty(Q_SIZE, dtype=cp.float32)
        self.key = cp.empty(KV_SIZE, dtype=cp.float32)
        self.value = cp.empty(KV_SIZE, dtype=cp.float32)
        self.attention = cp.empty(Q_SIZE, dtype=cp.float32)
        self.projected = cp.empty(HIDDEN, dtype=cp.float32)
        self.router = cp.empty(EXPERTS, dtype=cp.float32)
        self.gate = cp.empty(TOP_K * INTERMEDIATE, dtype=cp.float32)
        self.up = cp.empty(TOP_K * INTERMEDIATE, dtype=cp.float32)
        self.down = cp.empty(TOP_K * HIDDEN, dtype=cp.float32)
        self.scores = cp.empty(Q_HEADS * MAX_CONTEXT, dtype=cp.float32)
        self.logits = cp.empty(VOCAB, dtype=cp.float32)
        self.logit_values = cp.empty(2, dtype=cp.float32)
        self.argmax = cp.empty(1, dtype=cp.int32)
        self.hit_slots = cp.empty(TOP_K, dtype=cp.int32)
        self.hit_positions = cp.empty(TOP_K, dtype=cp.int32)
        self.miss_slots = cp.empty(TOP_K, dtype=cp.int32)
        self.miss_positions = cp.empty(TOP_K, dtype=cp.int32)
        self.order_device = cp.empty(TOP_K, dtype=cp.int32)
        self.weights_device = cp.empty(TOP_K, dtype=cp.float32)
        self.router_pinned = cp.cuda.alloc_pinned_memory(EXPERTS * 4)
        self.router_host = np.frombuffer(self.router_pinned, dtype=np.float32, count=EXPERTS)
        self.events = [cp.cuda.Event() for _ in range(LAYERS)]
        self.dynamic = [OrderedDict() for _ in range(LAYERS)]
        self.active_domain = None
        self.total_misses = 0
        self.total_miss_bytes = 0
        self.route_weight_error_max = 0.0
        self.route_unique_failures = 0
        self.kv_layer_position_writes = 0
        self.free_after_scratch, _ = cp.cuda.runtime.memGetInfo()

    def norm(self, layer, name):
        record = self.norm_by_key[(layer, name)]
        return self.norms[record["offset"] // 2:]

    def q8(self, layer, name, source, output):
        _index, record = self.record_by_key[(layer, name)]
        base = self.device_offsets[(layer, name)]
        self.k["q8_gemv"]((record["rows"],), (256,), (
            source, self.trunk, np.int64(base), np.int64(record["code_bytes"]),
            np.int32(record["rows"]), np.int32(record["cols"]), output,
        ), stream=self.compute)

    def embedding(self, token):
        index, record = self.record_by_key[(49, "embed")]
        if not 0 <= token < record["rows"]:
            raise ValueError(f"token {token} outside embedding vocabulary")
        base = self.q8_host_offsets[index]
        code_offset = base + token * record["cols"]
        scale_offset = base + record["code_bytes"] + token * (record["cols"] // 128) * 2
        codes = self.q8_host[code_offset:code_offset + record["cols"]].view(np.int8).astype(np.float32)
        scale_bits = self.q8_host[scale_offset:scale_offset + (record["cols"] // 128) * 2].view("<u2")
        scales = bits_to_float(scale_bits)
        return bf16_round(codes * np.repeat(scales, 128))

    def activate_domain(self, domain):
        self.compute.synchronize(); self.copy.synchronize()
        started = time.perf_counter_ns()
        for layer in range(LAYERS):
            for slot, expert in enumerate(self.fixed[domain][layer]):
                copy_expert(self.compute, self.expert_pinned, self.expert_cache_memory, self.layer_bases, layer, expert, slot)
        self.compute.synchronize()
        self.dynamic = [OrderedDict() for _ in range(LAYERS)]
        self.active_domain = domain
        return (time.perf_counter_ns() - started) / 1e6

    def reset_context(self):
        self.compute.synchronize(); self.copy.synchronize()
        self.dynamic = [OrderedDict() for _ in range(LAYERS)]

    def plan(self, route_ids, layer):
        slots = np.empty(TOP_K, dtype=np.int32)
        hits, misses, copies = [], [], []
        fixed_ids = self.fixed[self.active_domain][layer]
        fixed_set = frozenset(fixed_ids)
        lru = self.dynamic[layer]
        for position, raw in enumerate(route_ids):
            expert = int(raw)
            if expert in fixed_set:
                slot = fixed_ids.index(expert); hits.append(position)
            elif expert in lru:
                slot = lru[expert]; lru.move_to_end(expert); hits.append(position)
            else:
                if len(lru) < dynamic_slots(layer):
                    slot = STATIC_SLOTS + len(lru)
                else:
                    _old, slot = lru.popitem(last=False)
                lru[expert] = slot; misses.append(position); copies.append((expert, slot))
            slots[position] = self.layer_bases[layer] + slot
        return slots, hits, misses, copies

    def launch_expert_group(self, slots, positions, count):
        if count == 0:
            return
        self.k["q5_gate_up_n"]((count * 1536,), (256,), (
            self.normed, self.expert_cache, slots, positions, self.gate, self.up,
        ), stream=self.compute)
        self.k["swiglu_n"]((count * 3,), (256,), (self.gate, self.up, positions), stream=self.compute)
        self.k["q5_down_n"]((count * 2048,), (256,), (
            self.gate, self.expert_cache, slots, positions, self.down,
        ), stream=self.compute)

    def route(self, layer):
        cp.cuda.runtime.memcpyAsync(self.router_pinned.ptr, self.router.data.ptr, self.router.nbytes, cp.cuda.runtime.memcpyDeviceToHost, self.compute.ptr)
        self.compute.synchronize()
        logits = self.router_host.copy()
        probabilities = np.exp(logits - np.max(logits), dtype=np.float32)
        probabilities /= np.sum(probabilities, dtype=np.float32)
        ids = np.argsort(-probabilities, kind="stable")[:TOP_K].astype(np.int32)
        weights = probabilities[ids]
        weights /= np.sum(weights, dtype=np.float32)
        weights = bf16_round(weights).astype(np.float32)
        unique = len(set(int(x) for x in ids)) == TOP_K
        if not unique:
            self.route_unique_failures += 1
        self.route_weight_error_max = max(self.route_weight_error_max, abs(float(weights.sum(dtype=np.float32)) - 1.0))
        return ids, weights

    def decode(self, token, position, target=-1):
        if self.active_domain is None:
            raise RuntimeError("activate a cache domain first")
        if not 0 <= position < MAX_CONTEXT:
            raise ValueError("position outside physical KV cache")
        state_host = self.embedding(int(token))
        wall_start = time.perf_counter_ns()
        self.state.set(state_host, stream=self.compute)
        token_misses = 0
        for layer in range(LAYERS):
            cp.cuda.runtime.memcpyAsync(self.residual.data.ptr, self.state.data.ptr, self.state.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, self.compute.ptr)
            self.k["rmsnorm"]((1,), (256,), (self.state, self.norm(layer, "input"), self.normed, np.int32(HIDDEN)), stream=self.compute)
            self.q8(layer, "q", self.normed, self.q)
            self.q8(layer, "k", self.normed, self.key)
            self.q8(layer, "v", self.normed, self.value)
            self.k["qk_norm_rope_write"]((Q_HEADS + KV_HEADS,), (HEAD_DIM,), (
                self.q, self.key, self.norm(layer, "q_norm"), self.norm(layer, "k_norm"),
                self.kv, np.int32(layer), np.int32(position),
            ), stream=self.compute)
            self.k["write_v"]((2,), (256,), (self.value, self.kv, np.int32(layer), np.int32(position)), stream=self.compute)
            context = position + 1
            self.k["attention_scores"]((Q_HEADS * context,), (HEAD_DIM,), (
                self.q, self.kv, self.scores, np.int32(layer), np.int32(context),
            ), stream=self.compute)
            self.k["attention_values"]((Q_HEADS,), (HEAD_DIM,), (
                self.scores, self.kv, self.attention, np.int32(layer), np.int32(context),
            ), stream=self.compute)
            self.q8(layer, "o", self.attention, self.projected)
            self.k["residual_add"]((8,), (256,), (self.residual, self.projected, self.state, np.int32(HIDDEN)), stream=self.compute)
            cp.cuda.runtime.memcpyAsync(self.residual.data.ptr, self.state.data.ptr, self.state.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, self.compute.ptr)
            self.k["rmsnorm"]((1,), (256,), (self.state, self.norm(layer, "post"), self.normed, np.int32(HIDDEN)), stream=self.compute)
            self.q8(layer, "router", self.normed, self.router)
            route_ids, weights = self.route(layer)
            slots, hits, misses, copies = self.plan(route_ids, layer)
            token_misses += len(misses)
            for expert, slot in copies:
                copy_expert(self.copy, self.expert_pinned, self.expert_cache_memory, self.layer_bases, layer, expert, slot)
            if copies:
                self.events[layer].record(self.copy)
            if hits:
                hp = np.asarray(hits, dtype=np.int32); hs = slots[hp]
                self.hit_slots[:len(hp)].set(hs, stream=self.compute)
                self.hit_positions[:len(hp)].set(hp, stream=self.compute)
                self.launch_expert_group(self.hit_slots, self.hit_positions, len(hp))
            if misses:
                mp = np.asarray(misses, dtype=np.int32); ms = slots[mp]
                self.miss_slots[:len(mp)].set(ms, stream=self.compute)
                self.miss_positions[:len(mp)].set(mp, stream=self.compute)
                self.compute.wait_event(self.events[layer])
                self.launch_expert_group(self.miss_slots, self.miss_positions, len(mp))
            order = np.argsort(route_ids, kind="stable").astype(np.int32)
            self.order_device.set(order, stream=self.compute)
            self.weights_device.set(weights, stream=self.compute)
            self.k["weighted_residual"]((8,), (256,), (
                self.down, self.order_device, self.weights_device, self.residual, self.state,
            ), stream=self.compute)
            self.kv_layer_position_writes += 1
        self.k["rmsnorm"]((1,), (256,), (self.state, self.norm(48, "final"), self.normed, np.int32(HIDDEN)), stream=self.compute)
        self.q8(48, "head", self.normed, self.logits)
        self.k["logits_stats"]((1,), (256,), (self.logits, np.int32(target), self.logit_values, self.argmax), stream=self.compute)
        self.compute.synchronize(); self.copy.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        values = cp.asnumpy(self.logit_values)
        predicted = int(cp.asnumpy(self.argmax)[0])
        ce = None if target < 0 else float(values[0] - values[1])
        self.total_misses += token_misses
        self.total_miss_bytes += token_misses * EXPERT_BYTES
        finite = bool(np.isfinite(values).all())
        return {"prediction": predicted, "ce": ce, "wall_ms": wall_ms, "misses": token_misses, "finite": finite}

    def kv_digest(self, context):
        view = self.kv.reshape(LAYERS, 2, KV_HEADS, MAX_CONTEXT, HEAD_DIM)
        observed = cp.asnumpy(view[:, :, :, :context, :])
        nonzero = int(np.count_nonzero(observed))
        return {"context": context, "bytes": int(observed.nbytes), "sha256": array_sha(observed), "nonzero": nonzero, "elements": int(observed.size)}

    def physical(self):
        return {
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "total_vram_bytes": int(self.total_vram), "free_before_bytes": int(self.free_before),
            "free_after_fixed_bytes": int(self.free_after_fixed), "free_after_scratch_bytes": int(self.free_after_scratch),
            "expert_cache_bytes": CACHE_BYTES, "trunk_device_bytes": self.bank["aggregate"]["device_bytes"],
            "embedding_host_bytes": self.bank["aggregate"]["host_embedding_bytes"], "kv_bytes": KV_BYTES,
            "expert_bank_pinned_bytes": BANK_BYTES, "q8_bank_pinned_bytes": self.bank["aggregate"]["bytes"],
            "pin_ms": self.pin_ms, "expert_pin_ms": self.expert_pin_ms,
        }


def quality_phase(runtime, split, tensors, teacher):
    per_domain = {}
    all_ce, all_times, all_misses, all_predictions = [], [], [], []
    cache_init_ms = {}
    kv_digests = []
    for domain in DOMAINS:
        cache_init_ms[domain] = runtime.activate_domain(domain)
        domain_ce, domain_times, domain_misses, domain_predictions = [], [], [], []
        ids = tensors[f"{split}_{domain}"].numpy().astype(np.int64)
        for context_index, context_ids in enumerate(ids):
            runtime.reset_context()
            for position in range(context_ids.size - 1):
                row = runtime.decode(int(context_ids[position]), position, int(context_ids[position + 1]))
                domain_ce.append(row["ce"]); domain_times.append(row["wall_ms"])
                domain_misses.append(row["misses"]); domain_predictions.append(row["prediction"])
                if position % 32 == 0:
                    print(json.dumps({"phase": split, "domain": domain, "context": context_index, "position": position, "ms": row["wall_ms"], "misses": row["misses"], "ce": row["ce"]}), flush=True)
            kv_digests.append({"domain": domain, "context_index": context_index, **runtime.kv_digest(context_ids.size - 1)})
        teacher_domain = teacher["variants"]["bf16_teacher"]["domains"][domain]["next_token_cross_entropy"]
        domain_mean_ce = float(np.mean(domain_ce))
        per_domain[domain] = {
            "labels": len(domain_ce), "next_token_cross_entropy": domain_mean_ce,
            "teacher_cross_entropy": teacher_domain,
            "relative_cross_entropy_increase": (domain_mean_ce - teacher_domain) / teacher_domain,
            "wall_ms": domain_times, "wall_ms_stats": stats(domain_times),
            "misses": domain_misses, "miss_stats": stats(domain_misses),
            "predictions": domain_predictions, "prediction_sha256": array_sha(np.asarray(domain_predictions, dtype=np.int32)),
        }
        all_ce.extend(domain_ce); all_times.extend(domain_times); all_misses.extend(domain_misses); all_predictions.extend(domain_predictions)
    aggregate_teacher = teacher["variants"]["bf16_teacher"]["next_token_cross_entropy"]
    aggregate_ce = float(np.mean(all_ce))
    return {
        "per_domain": per_domain,
        "aggregate": {
            "labels": len(all_ce), "next_token_cross_entropy": aggregate_ce,
            "teacher_cross_entropy": aggregate_teacher,
            "relative_cross_entropy_increase": (aggregate_ce - aggregate_teacher) / aggregate_teacher,
            "wall_ms": all_times, "wall_ms_stats": stats(all_times),
            "misses": all_misses, "miss_stats": stats(all_misses),
            "predictions_sha256": array_sha(np.asarray(all_predictions, dtype=np.int32)),
            "finite": bool(np.isfinite(all_ce).all() and np.isfinite(all_times).all()),
        },
        "cache_initialization_ms": cache_init_ms,
        "kv_digests": kv_digests,
    }


def rollout(runtime, tokenizer, prompt, count):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("rollout prompt tokenized empty")
    cache_init_ms = runtime.activate_domain("general")
    runtime.reset_context()
    prefill = []
    for position, token in enumerate(ids[:-1]):
        row = runtime.decode(token, position, -1)
        prefill.append(row["wall_ms"])
    generated, feedback, times, misses = [], [], [], []
    current = int(ids[-1])
    position = len(ids) - 1
    for step in range(count):
        feedback.append(current)
        row = runtime.decode(current, position, -1)
        generated.append(row["prediction"]); times.append(row["wall_ms"]); misses.append(row["misses"])
        current = row["prediction"]; position += 1
        if step % 32 == 0:
            print(json.dumps({"phase": "rollout", "step": step, "token": generated[-1], "ms": times[-1], "misses": misses[-1]}), flush=True)
    return {
        "prompt": prompt, "prompt_ids": ids, "prefill_tokens": len(prefill),
        "prefill_wall_ms": prefill, "cache_initialization_ms": cache_init_ms,
        "generated_ids": generated, "feedback_ids": feedback,
        "generated_ids_sha256": array_sha(np.asarray(generated, dtype=np.int32)),
        "wall_ms": times, "wall_ms_stats": stats(times), "misses": misses, "miss_stats": stats(misses),
        "tokens_per_second": 1000.0 / float(np.mean(times)),
        "text": tokenizer.decode(generated, skip_special_tokens=False),
        "kv_digest": runtime.kv_digest(position),
    }


def smoke(runtime, tensors):
    init_ms = runtime.activate_domain("general")
    runtime.reset_context()
    ids = tensors["validation_general"].numpy()[0, :9].astype(np.int64)
    rows = [runtime.decode(int(ids[p]), p, int(ids[p + 1])) for p in range(8)]
    q_index, q_record = runtime.record_by_key[(0, "q")]
    base = runtime.q8_host_offsets[q_index]
    source = runtime.embedding(int(ids[0]))
    sample_rows = [0, q_record["rows"] // 3, q_record["rows"] // 2, q_record["rows"] - 1]
    codes = runtime.q8_host[base:base + q_record["code_bytes"]].view(np.int8).reshape(q_record["rows"], q_record["cols"])
    scale_bits = runtime.q8_host[base + q_record["code_bytes"]:base + q_record["bytes"]].view("<u2").reshape(q_record["rows"], q_record["cols"] // 128)
    expected = []
    for row in sample_rows:
        weights = bf16_round(codes[row].astype(np.float32) * np.repeat(bits_to_float(scale_bits[row]), 128))
        expected.append(float(weights @ source))
    runtime.state.set(source); runtime.q8(0, "q", runtime.state, runtime.q); runtime.compute.synchronize()
    observed = cp.asnumpy(runtime.q)[sample_rows]
    q8_max_abs = float(np.max(np.abs(observed - bf16_round(np.asarray(expected, dtype=np.float32)))))
    kv = runtime.kv_digest(8)
    gates = {
        "eight_full_tokens": len(rows) == 8,
        "all_finite": all(r["finite"] and math.isfinite(r["ce"]) and math.isfinite(r["wall_ms"]) for r in rows),
        "q8_sample_max_abs_le_0_25": q8_max_abs <= 0.25,
        "router_unique": runtime.route_unique_failures == 0,
        "router_weight_sum_error_le_0_02": runtime.route_weight_error_max <= 0.02,
        "kv_all_layers_written": runtime.kv_layer_position_writes == 8 * LAYERS,
        "kv_nonzero": kv["nonzero"] > kv["elements"] // 2,
        "resident_scratch_ge_192mib": runtime.free_after_fixed >= 192 * 2**20,
    }
    return {
        "kind": "streamq5_moe_p6a_end_to_end_smoke", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p6a_smoke_pass" if all(gates.values()) else "p6a_smoke_fail",
        "tokens": rows, "cache_initialization_ms": init_ms, "q8_sample_rows": sample_rows,
        "q8_sample_max_abs": q8_max_abs, "kv_digest": kv, "physical": runtime.physical(), "gates": gates,
        "claim_boundary": "Eight-token full physical smoke only; validation/test quality and rollout remain unopened.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "validation", "test"), required=True)
    args = parser.parse_args()
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    evaluator = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    required = {
        PREREG: lock["preregistration_sha256"], BANK_RESULT: lock["bank_result_sha256"],
        BANK_VERIFY: lock["bank_verification_sha256"], BANK_RESULT_PATH: lock["expert_bank_result_sha256"],
        P1D_VERIFY: lock["expert_bank_verification_sha256"], P0C_INPUT_LOCK: lock["p0c_input_lock_sha256"],
        P0C_DATA: lock["p0c_data_sha256"], P4D_CAPTURE: lock["p4d_capture_sha256"],
        P0C_VALIDATION: lock["p0c_validation_sha256"], P0C_TEST: lock["p0c_test_sha256"],
    }
    if any(sha256(path) != digest for path, digest in required.items()):
        raise ValueError("P6A input provenance mismatch")
    if sha256(Path(__file__)) != evaluator["evaluator_sha256"] or sha256(INPUT_LOCK) != evaluator["input_lock_sha256"]:
        raise ValueError("P6A evaluator provenance mismatch")
    if json.loads(BANK_VERIFY.read_text(encoding="utf-8"))["status"] != "p6a_exact_runtime_bank_verification_pass":
        raise RuntimeError("verified exact P6 bank required")
    output = R / f"p6a_end_to_end_{args.phase}.json"
    report = R / f"P6A_END_TO_END_{args.phase.upper()}.md"
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite P6A phase output")
    smoke_path = R / "p6a_end_to_end_smoke.json"
    validation_path = R / "p6a_end_to_end_validation.json"
    if args.phase != "smoke" and (not smoke_path.exists() or json.loads(smoke_path.read_text(encoding="utf-8"))["status"] != "p6a_smoke_pass"):
        raise RuntimeError("P6A smoke pass required")
    if args.phase == "test" and (not validation_path.exists() or json.loads(validation_path.read_text(encoding="utf-8"))["status"] != "p6a_validation_pass_test_authorized"):
        raise RuntimeError("P6A test not authorized")
    tensors = load_file(P0C_DATA)
    runtime = Runtime(lock)
    if args.phase == "smoke":
        payload = smoke(runtime, tensors)
    else:
        teacher_path = P0C_VALIDATION if args.phase == "validation" else P0C_TEST
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        quality = quality_phase(runtime, args.phase, tensors, teacher)
        aggregate = quality["aggregate"]
        gates = {
            "labels_1270": aggregate["labels"] == 1270,
            "finite": aggregate["finite"],
            "relative_ce_le_0_02": aggregate["relative_cross_entropy_increase"] <= lock["gates"]["relative_ce_max"],
            "mean_ms_le_100": aggregate["wall_ms_stats"]["mean"] <= lock["gates"]["mean_ms_max"],
            "p95_ms_le_150": aggregate["wall_ms_stats"]["p95"] <= lock["gates"]["p95_ms_max"],
            "all_domain_mean_le_100": all(row["wall_ms_stats"]["mean"] <= lock["gates"]["mean_ms_max"] for row in quality["per_domain"].values()),
            "router_unique": runtime.route_unique_failures == 0,
            "router_weight_sum_error_le_0_02": runtime.route_weight_error_max <= lock["gates"]["router_weight_sum_error_max"],
            "expert_miss_bytes_exact": runtime.total_miss_bytes == runtime.total_misses * EXPERT_BYTES,
            "kv_write_count_exact": runtime.kv_layer_position_writes == aggregate["labels"] * LAYERS,
            "kv_nonzero_all_contexts": all(row["nonzero"] > row["elements"] // 2 for row in quality["kv_digests"]),
            "resident_scratch_ge_192mib": runtime.free_after_fixed >= lock["gates"]["minimum_scratch_bytes"],
            "pinned_banks_exact": runtime.physical()["expert_bank_pinned_bytes"] == BANK_BYTES and runtime.physical()["q8_bank_pinned_bytes"] == runtime.bank["aggregate"]["bytes"],
        }
        rollout_result = None
        if args.phase == "validation":
            status = "p6a_validation_pass_test_authorized" if all(gates.values()) else "p6a_validation_closed_test_unopened"
        else:
            if all(gates.values()):
                tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
                before_writes = runtime.kv_layer_position_writes
                rollout_result = rollout(runtime, tokenizer, lock["rollout"]["prompt"], lock["rollout"]["tokens"])
                rollout_gates = {
                    "tokens_512": len(rollout_result["generated_ids"]) == lock["rollout"]["tokens"],
                    "feedback_exact": rollout_result["feedback_ids"][1:] == rollout_result["generated_ids"][:-1],
                    "finite": bool(np.isfinite(rollout_result["wall_ms"]).all()),
                    "mean_ms_le_100": rollout_result["wall_ms_stats"]["mean"] <= lock["gates"]["mean_ms_max"],
                    "p95_ms_le_150": rollout_result["wall_ms_stats"]["p95"] <= lock["gates"]["p95_ms_max"],
                    "kv_write_count_exact": runtime.kv_layer_position_writes - before_writes == (len(rollout_result["prompt_ids"]) - 1 + lock["rollout"]["tokens"]) * LAYERS,
                    "text_decodable": isinstance(rollout_result["text"], str) and len(rollout_result["text"]) > 0,
                }
                rollout_result["gates"] = rollout_gates
                gates["rollout_all_gates"] = all(rollout_gates.values())
            status = "p6a_end_to_end_eureka_pass" if all(gates.values()) else "p6a_end_to_end_closed"
        payload = {
            "kind": "streamq5_moe_p6a_physical_end_to_end_decode", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "phase": args.phase, "status": status,
            "inputs": {"input_lock_sha256": sha256(INPUT_LOCK), "evaluator_lock_sha256": sha256(EVALUATOR_LOCK), "evaluator_sha256": sha256(Path(__file__)), **{path.name: digest for path, digest in required.items()}},
            "physical": runtime.physical(), "quality": quality, "rollout": rollout_result,
            "runtime_invariants": {"total_misses": runtime.total_misses, "total_miss_bytes": runtime.total_miss_bytes, "route_unique_failures": runtime.route_unique_failures, "route_weight_sum_abs_error_max": runtime.route_weight_error_max, "kv_layer_position_writes": runtime.kv_layer_position_writes},
            "gates": {key: bool(value) for key, value in gates.items()},
            "claim_boundary": "Physical full-depth quality and batch-1 decode on this exact machine/artifact; no cross-hardware, >4096-context, batch>1, or other-model claim.",
        }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.phase == "smoke":
        summary = f"Status **{payload['status']}**; 8 fysieke end-to-end tokens."
    else:
        a = payload["quality"]["aggregate"]
        summary = f"Status **{payload['status']}**; CE-relatief {a['relative_cross_entropy_increase']:.6%}; mean/p95 {a['wall_ms_stats']['mean']:.3f}/{a['wall_ms_stats']['p95']:.3f} ms."
        if payload.get("rollout"):
            rr = payload["rollout"]
            summary += f" Rollout {len(rr['generated_ids'])} tokens, {rr['tokens_per_second']:.3f} tok/s."
    report.write_text(f"# P6A fysieke end-to-end decode — {args.phase}\n\n{summary}\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "physical": payload.get("physical"), "gates": payload["gates"], "quality_aggregate": None if args.phase == "smoke" else payload["quality"]["aggregate"], "rollout": None if not payload.get("rollout") else {"tokens": len(payload["rollout"]["generated_ids"]), "timing": payload["rollout"]["wall_ms_stats"], "tokens_per_second": payload["rollout"]["tokens_per_second"], "gates": payload["rollout"]["gates"]}}, indent=2))
    if "fail" in payload["status"] or "closed" in payload["status"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
