from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import zlib
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P3A_INTEGRATED_EXPERT_DATAPLANE_PREREGISTRATION.md"
INPUT_LOCK = R / "p3a_dataplane_input_lock.json"
ROUTE_CAPTURE = R / "p3a_route_capture_result.json"
P2C_VERIFY = R / "p2c_physical_h2d_verification.json"
P2A_VERIFY = R / "p2a_kernel_verification.json"
BANK_RESULT_PATH = R / "p1d_physical_bank_result.json"
SMOKE_LOCK = R / "p3a_smoke_evaluator_lock.json"
BENCHMARK_LOCK = R / "p3a_benchmark_evaluator_lock.json"
SMOKE = R / "p3a_dataplane_smoke.json"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p3a_routes"
BANK_DIR = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
LAYERS, EXPERTS, TOP_K = 48, 128, 8
STATIC_SLOTS, TOTAL_SLOTS = 20, 1640
HEADER_BYTES, RECORD_BYTES, EXPERT_BYTES = 64, 1_011_712, 3_035_136
CODE_BYTES, SCALE_BYTES, LAYER_BYTES, BANK_BYTES = 983_040, 24_576, 388_497_408, 18_647_875_584
CACHE_BYTES, TRUNK_BYTES, KV_BYTES, MIN_SCRATCH = 4_977_623_040, 1_541_093_376, 402_653_184, 402_653_184
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s")


CUDA_SOURCE = r'''
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
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
        for (int slot = 0; slot < 8; ++slot) {
            int code = ((word >> (slot * 5)) & 31ULL) - 15;
            sum += ((float)code) * scale * x[column + slot];
        }
    }
    return sum;
}
extern "C" __global__ void q5_gate_up_8(
    const float* x, const unsigned char* cache, const int* slots,
    float* gate, float* up) {
    int expert = (int)blockIdx.x / 1536;
    int local = (int)blockIdx.x - expert * 1536;
    int projection = local >= 768;
    int row = local - projection * 768;
    long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL;
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
        if (projection) up[expert * 768 + row] = reduction[0];
        else gate[expert * 768 + row] = reduction[0];
    }
}
extern "C" __global__ void swiglu_8(float* gate, const float* up) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < 6144) {
        float value = gate[index];
        gate[index] = (value / (1.0f + expf(-value))) * up[index];
    }
}
extern "C" __global__ void q5_down_8(
    const float* activation, const unsigned char* cache, const int* slots,
    float* down) {
    int expert = (int)blockIdx.x / 2048;
    int row = (int)blockIdx.x - expert * 2048;
    long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float sum = q5_dot(activation + expert * 768, packed, scales, row, 768, (int)threadIdx.x);
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down[expert * 2048 + row] = reduction[0];
}
extern "C" __global__ void reduce_experts(const float* down, float* state) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < 2048) {
        float sum = 0.0f;
        #pragma unroll
        for (int expert = 0; expert < 8; ++expert) sum += down[expert * 2048 + index];
        state[index] = sum * 0.125f;
    }
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_array(value: np.ndarray) -> str:
    digest = hashlib.sha256(); raw = memoryview(value).cast("B")
    for begin in range(0, len(raw), 8 * 2**20): digest.update(raw[begin:begin + 8 * 2**20])
    return digest.hexdigest()


def dynamic_slots(layer: int) -> int:
    return 15 if layer <= 7 else 14


def bases():
    result, cursor = [], 0
    for layer in range(LAYERS):
        result.append(cursor); cursor += STATIC_SLOTS + dynamic_slots(layer)
    if cursor != TOTAL_SLOTS: raise RuntimeError("slot partition mismatch")
    return result


def load_routes():
    routes = {domain: [] for domain in DOMAINS}; hashes = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"; hashes[str(layer)] = sha256(path)
        with safe_open(path, framework="numpy") as handle:
            for domain in DOMAINS: routes[domain].append(handle.get_tensor(f"{domain}_router_ids").astype(np.int64))
    return {domain: np.stack(values, axis=1) for domain, values in routes.items()}, hashes


def static_sets(routes):
    selected = {}
    for domain in DOMAINS:
        selected[domain] = []
        for layer in range(LAYERS):
            counts = np.bincount(routes[domain][:512, layer, :].reshape(-1), minlength=EXPERTS)
            selected[domain].append(tuple(int(value) for value in np.lexsort((np.arange(EXPERTS), -counts))[:STATIC_SLOTS]))
    return selected


def pin_bank(bank):
    layers, hashes = [], {}; started = time.perf_counter()
    for layer in range(LAYERS):
        memory = cp.cuda.alloc_pinned_memory(LAYER_BYTES); host = np.frombuffer(memory, dtype=np.uint8, count=LAYER_BYTES)
        with (BANK_DIR / f"layer_{layer:02d}.q5bin").open("rb", buffering=8 * 2**20) as handle:
            if handle.readinto(host) != LAYER_BYTES: raise RuntimeError("bank read failed")
        observed = sha_array(host)
        if observed != bank["manifests"][str(layer)]["artifact_sha256"]: raise ValueError("pinned bank hash mismatch")
        hashes[str(layer)] = observed; layers.append(memory)
        if layer % 8 == 7: print(json.dumps({"pinned_layers": layer + 1}), flush=True)
    return layers, hashes, (time.perf_counter() - started) * 1000


def copy_expert(stream, pinned, cache, layer_bases, layer, expert, slot):
    cp.cuda.runtime.memcpyAsync(cache.ptr + (layer_bases[layer] + slot) * EXPERT_BYTES, pinned[layer].ptr + expert * EXPERT_BYTES, EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)


def decode_record(layer, expert, projection):
    rows, cols = ((768, 2048), (768, 2048), (2048, 768))[projection]
    path = BANK_DIR / f"layer_{layer:02d}.q5bin"; offset = (expert * 3 + projection) * RECORD_BYTES
    with path.open("rb") as handle:
        handle.seek(offset); fields = HEADER.unpack(handle.read(HEADER_BYTES)); packed = np.frombuffer(handle.read(CODE_BYTES), dtype=np.uint8).copy(); bits = np.frombuffer(handle.read(SCALE_BYTES), dtype="<u2").copy()
    if fields[0] != b"SQ5M" or fields[2] != layer or fields[3] != expert or fields[4] != projection: raise ValueError("smoke header mismatch")
    chunks = packed.reshape(-1, 5).astype(np.uint64)
    words = chunks[:, 0] | chunks[:, 1] << 8 | chunks[:, 2] << 16 | chunks[:, 3] << 24 | chunks[:, 4] << 32
    codes = np.stack([((words >> (5 * slot)) & 31).astype(np.int8) - 15 for slot in range(8)], axis=-1).reshape(rows, cols)
    scales = (bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols // 128)
    return codes.astype(np.float32) * scales[:, np.arange(cols) // 128]


def kernels():
    return (
        cp.RawKernel(CUDA_SOURCE, "q5_gate_up_8", options=("--std=c++11",)),
        cp.RawKernel(CUDA_SOURCE, "swiglu_8", options=("--std=c++11",)),
        cp.RawKernel(CUDA_SOURCE, "q5_down_8", options=("--std=c++11",)),
        cp.RawKernel(CUDA_SOURCE, "reduce_experts", options=("--std=c++11",)),
    )


def launch_layer(kernel_set, stream, state, cache, slot_ids, gate, up, down):
    gate_up, swiglu, down_kernel, reduce = kernel_set
    gate_up((8 * 1536,), (256,), (state, cache, slot_ids, gate, up), stream=stream)
    swiglu((24,), (256,), (gate, up), stream=stream)
    down_kernel((8 * 2048,), (256,), (gate, cache, slot_ids, down), stream=stream)
    reduce((8,), (256,), (down, state), stream=stream)


def error(observed, expected):
    delta = observed - expected
    return {"max_abs": float(np.abs(delta).max()), "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30)), "finite": bool(np.isfinite(observed).all())}


def smoke_run(lock, route, kernel_set):
    layer, token = 0, 0; experts = [int(value) for value in route[token, layer]]
    cache = cp.cuda.alloc(8 * EXPERT_BYTES); stream = cp.cuda.Stream(); slots = cp.asarray(np.arange(8, dtype=np.int32))
    pinned = []
    for index, expert in enumerate(experts):
        memory = cp.cuda.alloc_pinned_memory(EXPERT_BYTES); host = np.frombuffer(memory, dtype=np.uint8, count=EXPERT_BYTES)
        with (BANK_DIR / "layer_00.q5bin").open("rb") as handle:
            handle.seek(expert * EXPERT_BYTES); handle.readinto(host)
        pinned.append(memory); cp.cuda.runtime.memcpyAsync(cache.ptr + index * EXPERT_BYTES, memory.ptr, EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
    rng = np.random.default_rng(lock["initial_state_seed"]); x = rng.standard_normal(2048, dtype=np.float32)
    state = cp.asarray(x); gate = cp.empty(6144, dtype=cp.float32); up = cp.empty(6144, dtype=cp.float32); down = cp.empty(16384, dtype=cp.float32)
    cache_view = cp.ndarray((8 * EXPERT_BYTES,), dtype=cp.uint8, memptr=cache)
    gate_up, swiglu, down_kernel, reduce = kernel_set
    gate_up((8 * 1536,), (256,), (state, cache_view, slots, gate, up), stream=stream); stream.synchronize()
    observed_gate, observed_up = cp.asnumpy(gate), cp.asnumpy(up)
    swiglu((24,), (256,), (gate, up), stream=stream); stream.synchronize(); observed_activation = cp.asnumpy(gate)
    down_kernel((8 * 2048,), (256,), (gate, cache_view, slots, down), stream=stream); stream.synchronize(); observed_down = cp.asnumpy(down)
    reduce((8,), (256,), (down, state), stream=stream); stream.synchronize(); observed_state = cp.asnumpy(state)
    ref_gate, ref_up, ref_down = [], [], []
    for expert in experts:
        g = decode_record(layer, expert, 0) @ x; u = decode_record(layer, expert, 1) @ x
        activation = (g / (1.0 + np.exp(-g))) * u; d = decode_record(layer, expert, 2) @ activation
        ref_gate.append(g); ref_up.append(u); ref_down.append(d)
    ref_gate = np.stack(ref_gate); ref_up = np.stack(ref_up); ref_activation = (ref_gate / (1.0 + np.exp(-ref_gate))) * ref_up; ref_down = np.stack(ref_down); ref_state = ref_down.mean(axis=0)
    errors = {"gate": error(observed_gate.reshape(8, 768), ref_gate), "up": error(observed_up.reshape(8, 768), ref_up), "swiglu": error(observed_activation.reshape(8, 768), ref_activation), "down": error(observed_down.reshape(8, 2048), ref_down), "reduced": error(observed_state, ref_state)}
    correct = all(row["finite"] and row["max_abs"] <= 0.02 and row["relative_l2"] <= 1e-4 for row in errors.values())
    return {"kind": "streamq5_moe_p3a_integrated_expert_smoke", "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "smoke_pass" if correct else "smoke_fail", "layer": layer, "token": token, "experts": experts, "errors": errors, "claim_boundary": "Untimed one-layer fused dataplane correctness only."}


def percentile(values):
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def run_domain(domain, route, fixed_ids, begin, end, pinned, cache, layer_bases, stream, kernel_set, seed):
    fixed = [frozenset(values) for values in fixed_ids]; dynamic = [OrderedDict() for _ in range(LAYERS)]
    for layer in range(LAYERS):
        for slot, expert in enumerate(fixed_ids[layer]): copy_expert(stream, pinned, cache, layer_bases, layer, expert, slot)
    stream.synchronize()
    cache_view = cp.ndarray((CACHE_BYTES,), dtype=cp.uint8, memptr=cache)
    state = cp.empty(2048, dtype=cp.float32); gate = cp.empty(6144, dtype=cp.float32); up = cp.empty(6144, dtype=cp.float32); down = cp.empty(16384, dtype=cp.float32); slot_ids = cp.empty(8, dtype=cp.int32)
    rng = np.random.default_rng(seed); times = np.empty(end - begin); misses = np.zeros(end - begin, dtype=np.int64); finite = True
    for local, token in enumerate(range(begin, end)):
        initial = rng.standard_normal(2048, dtype=np.float32); state.set(initial, stream=stream)
        wall_begin = time.perf_counter_ns()
        for layer in range(LAYERS):
            slots = np.empty(8, dtype=np.int32); lru = dynamic[layer]
            for index, raw in enumerate(route[token, layer]):
                expert = int(raw)
                if expert in fixed[layer]: slot = fixed_ids[layer].index(expert)
                elif expert in lru: slot = lru[expert]; lru.move_to_end(expert)
                else:
                    misses[local] += 1
                    if len(lru) < dynamic_slots(layer): slot = STATIC_SLOTS + len(lru)
                    else: _evicted, slot = lru.popitem(last=False)
                    lru[expert] = slot; copy_expert(stream, pinned, cache, layer_bases, layer, expert, slot)
                slots[index] = layer_bases[layer] + slot
            slot_ids.set(slots, stream=stream); launch_layer(kernel_set, stream, state, cache_view, slot_ids, gate, up, down)
        stream.synchronize(); times[local] = (time.perf_counter_ns() - wall_begin) / 1e6
        if local % 64 == 0: print(json.dumps({"domain": domain, "token": token, "misses": int(misses[local]), "integrated_ms": float(times[local])}), flush=True)
        if local in (0, end - begin - 1): finite &= bool(cp.isfinite(state).all().get())
    return {"tokens": end - begin, "misses": misses.tolist(), "wall_ms": times.tolist(), "miss_stats": percentile(misses.astype(np.float64)), "wall_ms_stats": percentile(times), "finite_outputs": finite}


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("smoke", "validation", "test"), required=True); return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8")); routes, route_hashes = load_routes(); kernel_set = kernels()
    active_lock = SMOKE_LOCK if args.phase == "smoke" else BENCHMARK_LOCK
    evaluator_lock = json.loads(active_lock.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != evaluator_lock["evaluator_sha256"] or sha256(INPUT_LOCK) != evaluator_lock["input_lock_sha256"] or sha256(PREREG) != lock["preregistration_sha256"] or sha256(ROUTE_CAPTURE) != lock["route_capture_sha256"] or sha256(P2C_VERIFY) != lock["p2c_verification_sha256"] or sha256(P2A_VERIFY) != lock["p2a_verification_sha256"]: raise ValueError("P3A provenance mismatch")
    if args.phase == "smoke":
        if SMOKE.exists(): raise FileExistsError("refusing to overwrite P3A smoke")
        payload = smoke_run(lock, routes["general"], kernel_set); SMOKE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(payload, indent=2)); raise SystemExit(0 if payload["status"] == "smoke_pass" else 1)
    output = R / f"p3a_integrated_expert_{args.phase}.json"; report = R / f"P3A_INTEGRATED_EXPERT_{args.phase.upper()}.md"
    if output.exists() or report.exists(): raise FileExistsError("refusing to overwrite P3A benchmark")
    if sha256(SMOKE) != evaluator_lock["smoke_sha256"] or json.loads(SMOKE.read_text(encoding="utf-8"))["status"] != "smoke_pass": raise RuntimeError("locked smoke pass required")
    validation_path = R / "p3a_integrated_expert_validation.json"
    if args.phase == "test" and (not validation_path.exists() or json.loads(validation_path.read_text(encoding="utf-8"))["status"] != "p3a_validation_pass_test_authorized"): raise RuntimeError("P3A test not authorized")
    bank = json.loads(BANK_RESULT_PATH.read_text(encoding="utf-8")); fixed = static_sets(routes); layer_bases = bases(); pinned, pinned_hashes, pin_ms = pin_bank(bank)
    cp.get_default_memory_pool().free_all_blocks(); free_before, total_vram = cp.cuda.runtime.memGetInfo()
    if free_before < CACHE_BYTES + TRUNK_BYTES + KV_BYTES + MIN_SCRATCH: raise MemoryError("P3A co-residency unavailable")
    cache = cp.cuda.alloc(CACHE_BYTES); cache_view = cp.ndarray((CACHE_BYTES,), dtype=cp.uint8, memptr=cache); trunk = cp.cuda.alloc(TRUNK_BYTES); kv = cp.cuda.alloc(KV_BYTES); stream = cp.cuda.Stream()
    with stream: cp.cuda.runtime.memsetAsync(cache.ptr, 0, CACHE_BYTES, stream.ptr); cp.cuda.runtime.memsetAsync(trunk.ptr, 0, TRUNK_BYTES, stream.ptr); cp.cuda.runtime.memsetAsync(kv.ptr, 0, KV_BYTES, stream.ptr)
    stream.synchronize(); free_after, _ = cp.cuda.runtime.memGetInfo(); begin, end = lock["partitions"][args.phase]
    per_domain = {domain: run_domain(domain, routes[domain], fixed[domain], begin, end, pinned, cache, layer_bases, stream, kernel_set, lock["initial_state_seed"] + index * 100000 + begin) for index, domain in enumerate(DOMAINS)}
    all_times = np.concatenate([np.asarray(per_domain[domain]["wall_ms"]) for domain in DOMAINS]); all_misses = np.concatenate([np.asarray(per_domain[domain]["misses"]) for domain in DOMAINS]); aggregate = {"tokens": int(all_times.size), "wall_ms": percentile(all_times), "misses": percentile(all_misses.astype(np.float64))}
    gates = {"full_bank_pinned": len(pinned) * LAYER_BYTES == BANK_BYTES and pinned_hashes == {str(layer): bank["manifests"][str(layer)]["artifact_sha256"] for layer in range(LAYERS)}, "device_co_resident_and_scratch": free_after >= MIN_SCRATCH, "aggregate_mean_le_60": aggregate["wall_ms"]["mean"] <= 60, "aggregate_p95_le_75": aggregate["wall_ms"]["p95"] <= 75, "all_domain_mean_le_60": all(row["wall_ms_stats"]["mean"] <= 60 for row in per_domain.values()), "all_domain_p95_le_75": all(row["wall_ms_stats"]["p95"] <= 75 for row in per_domain.values()), "all_outputs_finite": all(row["finite_outputs"] for row in per_domain.values())}
    all_pass = all(gates.values())
    if args.phase == "validation": status, phase_pass = ("p3a_validation_pass_test_authorized", False) if all_pass else ("p3a_validation_closed", False)
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8")); status = "p3a_integrated_expert_pass" if all_pass and all(validation["gates"].values()) else "p3a_integrated_expert_closed"; phase_pass = status == "p3a_integrated_expert_pass"
    payload = {"kind": "streamq5_moe_p3a_integrated_physical_expert_dataplane", "completed_utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase, "status": status, "inputs": {"preregistration_sha256": sha256(PREREG), "input_lock_sha256": sha256(INPUT_LOCK), "evaluator_lock_sha256": sha256(active_lock), "evaluator_sha256": sha256(Path(__file__)), "route_capture_sha256": sha256(ROUTE_CAPTURE), "smoke_sha256": sha256(SMOKE), "route_artifact_sha256": route_hashes}, "physical": {"pinned_bank_bytes": len(pinned) * LAYER_BYTES, "pin_and_hash_ms": pin_ms, "cache_bytes": CACHE_BYTES, "trunk_bytes": TRUNK_BYTES, "kv_bytes": KV_BYTES, "free_before_bytes": int(free_before), "free_after_bytes": int(free_after), "total_vram_bytes": int(total_vram)}, "policy": {"static": 20, "dynamic_layers_0_7": 15, "dynamic_layers_8_47": 14, "reduction": lock["expert_reduction"]}, "per_domain": per_domain, "aggregate": aggregate, "gates": gates, "p3a_pass": phase_pass, "claim_boundary": "Integrated routed-expert data plane only; attention, router/trunk, embeddings/head, KV update, sampling, and end-to-end model token timing remain unproven."}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); report.write_text(f"# STREAMQ5-MoE P3A - {args.phase}\n\nUitkomst: **{status}**. Mean/p95 {aggregate['wall_ms']['mean']:.3f}/{aggregate['wall_ms']['p95']:.3f} ms/token.\n", encoding="utf-8"); print(json.dumps({"status": status, "aggregate": aggregate, "physical": payload["physical"], "gates": gates}, indent=2)); raise SystemExit(0 if not status.endswith("closed") else 1)
