from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import (
    BANK_DIR, EXPERT_BYTES, LAYERS, RECORD_BYTES,
)
from scripts.streamq5_moe.run_p6a_end_to_end_decode import (
    CUDA_SOURCE, pin_q8_bank,
)


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P7A_KERNEL_ROOFLINE_PREREGISTRATION.md"
Q8_MANIFEST = R / "p6a_exact_runtime_bank_result.json"
OUTPUT = R / "p7a_kernel_roofline.json"
SEED = 270812
WARMUPS = 5
ITERATIONS = 60
Q5_EXPERTS_PER_LAYER = 8
Q5_SLOTS = LAYERS * Q5_EXPERTS_PER_LAYER


DIAG_SOURCE = r'''
extern "C" __global__ void dispatch_noop() { }

extern "C" __global__ void raw_scan(
    const unsigned char* data, long long bytes, unsigned int* checksum) {
    unsigned int value = 0U;
    long long global = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long index = global * 4LL; index + 3LL < bytes; index += stride * 4LL) {
        unsigned int word = *((const unsigned int*)(data + index));
        value ^= word + (unsigned int)index;
    }
    for (int delta = 16; delta > 0; delta >>= 1) value ^= __shfl_down_sync(0xffffffffU, value, delta);
    if ((threadIdx.x & 31) == 0) atomicXor(checksum + blockIdx.x, value);
}

extern "C" __global__ void q8_pattern_read(
    const unsigned char* bank, long long base, long long code_bytes,
    int rows, int cols, unsigned int* checksum) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    const signed char* codes = (const signed char*)(bank + base);
    const unsigned short* scales = (const unsigned short*)(bank + base + code_bytes);
    int groups = cols >> 7;
    unsigned int value = 0U;
    for (int col = (int)threadIdx.x; col < cols; col += blockDim.x) {
        unsigned int code = (unsigned int)(unsigned char)codes[(long long)row * cols + col];
        unsigned int scale = (unsigned int)scales[row * groups + (col >> 7)];
        value ^= (code | (scale << 8)) + (unsigned int)col;
    }
    __shared__ unsigned int reduction[256];
    reduction[threadIdx.x] = value; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] ^= reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) checksum[row] ^= reduction[0];
}

__device__ __forceinline__ unsigned int q5_read_row(
    const unsigned char* packed, const unsigned short* scales,
    int row, int cols, int tid) {
    int packs = cols >> 3;
    int groups = cols >> 7;
    unsigned int value = 0U;
    for (int pack = tid; pack < packs; pack += blockDim.x) {
        const unsigned char* source = packed + ((long long)row * packs + pack) * 5LL;
        unsigned int lo = (unsigned int)source[0]
            | ((unsigned int)source[1] << 8)
            | ((unsigned int)source[2] << 16)
            | ((unsigned int)source[3] << 24);
        unsigned int hi = (unsigned int)source[4];
        unsigned int scale = (unsigned int)scales[row * groups + (pack >> 4)];
        value ^= lo ^ (hi << 3) ^ (scale + (unsigned int)pack);
    }
    return value;
}

extern "C" __global__ void q5_pattern_gate_up(
    const unsigned char* cache, const int* slots, unsigned int* checksum) {
    int expert = (int)blockIdx.x / 1536;
    int local = (int)blockIdx.x - expert * 1536;
    int projection = local >= 768;
    int row = local - projection * 768;
    long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    unsigned int value = q5_read_row(packed, scales, row, 2048, (int)threadIdx.x);
    __shared__ unsigned int reduction[256];
    reduction[threadIdx.x] = value; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] ^= reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) checksum[expert * 1536 + local] ^= reduction[0];
}

extern "C" __global__ void q5_pattern_down(
    const unsigned char* cache, const int* slots, unsigned int* checksum) {
    int expert = (int)blockIdx.x / 2048;
    int row = (int)blockIdx.x - expert * 2048;
    long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    unsigned int value = q5_read_row(packed, scales, row, 768, (int)threadIdx.x);
    __shared__ unsigned int reduction[256];
    reduction[threadIdx.x] = value; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] ^= reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) checksum[expert * 2048 + row] ^= reduction[0];
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)), "min": float(x.min()),
        "max": float(x.max()),
    }


def measure(stream: cp.cuda.Stream, launch) -> dict:
    for _ in range(WARMUPS):
        launch()
    stream.synchronize()
    values = []
    for _ in range(ITERATIONS):
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record(stream); launch(); end.record(stream); end.synchronize()
        values.append(float(cp.cuda.get_elapsed_time(begin, end)))
    return {"event_ms": values, "stats": summary(values)}


def load_q8() -> tuple[dict, object, np.ndarray, cp.cuda.Memory, cp.ndarray, list]:
    manifest = json.loads(Q8_MANIFEST.read_text(encoding="utf-8"))
    pinned, host, offsets, aggregate_sha = pin_q8_bank(manifest)
    memory = cp.cuda.alloc(manifest["aggregate"]["device_bytes"])
    device = cp.ndarray((manifest["aggregate"]["device_bytes"],), dtype=cp.uint8, memptr=memory)
    records, cursor = [], 0
    stream = cp.cuda.Stream(non_blocking=True)
    for index, record in enumerate(manifest["records"]):
        if record["residency"] != "device":
            continue
        cp.cuda.runtime.memcpyAsync(memory.ptr + cursor, pinned.ptr + offsets[index], record["bytes"], cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
        records.append((cursor, record))
        cursor += record["bytes"]
    stream.synchronize()
    if cursor != manifest["aggregate"]["device_bytes"]:
        raise RuntimeError("Q8 device-byte mismatch")
    return manifest, pinned, host, memory, device, records, aggregate_sha


def load_q5() -> tuple[cp.cuda.Memory, cp.ndarray]:
    total = Q5_SLOTS * EXPERT_BYTES
    memory = cp.cuda.alloc(total)
    device = cp.ndarray((total,), dtype=cp.uint8, memptr=memory)
    staging = cp.cuda.alloc_pinned_memory(EXPERT_BYTES)
    host = np.frombuffer(staging, dtype=np.uint8, count=EXPERT_BYTES)
    stream = cp.cuda.Stream(non_blocking=True)
    for layer in range(LAYERS):
        path = BANK_DIR / f"layer_{layer:02d}.q5bin"
        with path.open("rb", buffering=8 * 2**20) as handle:
            for expert in range(Q5_EXPERTS_PER_LAYER):
                handle.seek(expert * EXPERT_BYTES)
                if handle.readinto(host) != EXPERT_BYTES:
                    raise RuntimeError("short Q5 record")
                slot = layer * Q5_EXPERTS_PER_LAYER + expert
                cp.cuda.runtime.memcpyAsync(memory.ptr + slot * EXPERT_BYTES, staging.ptr, EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                stream.synchronize()
        print(json.dumps({"q5_layers_loaded": layer + 1}), flush=True)
    return memory, device


def adjusted_gbps(byte_count: int, measured: dict, noop: dict) -> tuple[float, float]:
    raw_ms = measured["stats"]["p50"]
    adjusted_ms = max(raw_ms - noop["stats"]["p50"], 1.0e-6)
    return adjusted_ms, byte_count / adjusted_ms / 1.0e6


def classify(raw_gbps: float, pattern_gbps: float, gemv_gbps: float) -> str:
    if pattern_gbps / raw_gbps < 0.60:
        return "row_geometry_reduction_or_launch_dominant"
    if gemv_gbps / pattern_gbps < 0.50:
        return "decode_rounding_or_mac_dominant"
    return "mixed"


def main() -> None:
    started = time.perf_counter()
    cp.cuda.runtime.deviceSynchronize()
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    manifest, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    q5_mem, q5 = load_q5()
    free_resident, _ = cp.cuda.runtime.memGetInfo()

    names = ("q8_gemv", "q5_gate_up_n", "q5_down_n", "dispatch_noop", "raw_scan", "q8_pattern_read", "q5_pattern_gate_up", "q5_pattern_down")
    module = cp.RawModule(code=CUDA_SOURCE + DIAG_SOURCE, options=("--std=c++11",), name_expressions=names)
    kernels = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    x4096 = cp.asarray(rng.standard_normal(4096, dtype=np.float32))
    q8_output = cp.empty(151936, dtype=cp.float32)
    q8_check = cp.zeros(151936, dtype=cp.uint32)
    raw_check = cp.zeros(4096, dtype=cp.uint32)
    positions = cp.asarray(np.arange(8, dtype=np.int32))
    all_slots = [cp.asarray(np.arange(layer * 8, layer * 8 + 8, dtype=np.int32)) for layer in range(LAYERS)]
    gate = cp.empty(8 * 768, dtype=cp.float32)
    up = cp.empty_like(gate)
    down = cp.empty(8 * 2048, dtype=cp.float32)
    q5_check = cp.zeros(8 * 2048, dtype=cp.uint32)

    def launch_noop(count):
        def inner():
            for _ in range(count): kernels["dispatch_noop"]((1,), (1,), (), stream=stream)
        return inner

    def q8_raw():
        kernels["raw_scan"]((4096,), (256,), (q8, np.int64(q8.size), raw_check), stream=stream)

    def q8_pattern():
        for base, record in q8_records:
            kernels["q8_pattern_read"]((record["rows"],), (256,), (q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), q8_check), stream=stream)

    def q8_gemv():
        for base, record in q8_records:
            kernels["q8_gemv"]((record["rows"],), (256,), (x4096, q8, np.int64(base), np.int64(record["code_bytes"]), np.int32(record["rows"]), np.int32(record["cols"]), q8_output), stream=stream)

    def q5_raw():
        kernels["raw_scan"]((4096,), (256,), (q5, np.int64(q5.size), raw_check), stream=stream)

    def q5_pattern():
        for layer in range(LAYERS):
            slots = all_slots[layer]
            kernels["q5_pattern_gate_up"]((8 * 1536,), (256,), (q5, slots, q5_check), stream=stream)
            kernels["q5_pattern_down"]((8 * 2048,), (256,), (q5, slots, q5_check), stream=stream)

    def q5_gemv():
        for layer in range(LAYERS):
            slots = all_slots[layer]
            kernels["q5_gate_up_n"]((8 * 1536,), (256,), (x4096, q5, slots, positions, gate, up), stream=stream)
            kernels["q5_down_n"]((8 * 2048,), (256,), (gate, q5, slots, positions, down), stream=stream)

    measurements = {}
    for name, launch in (
        ("q8_noop_241", launch_noop(len(q8_records))), ("q8_raw", q8_raw),
        ("q8_pattern", q8_pattern), ("q8_gemv", q8_gemv),
        ("q5_noop_96", launch_noop(LAYERS * 2)), ("q5_raw", q5_raw),
        ("q5_pattern", q5_pattern), ("q5_gemv", q5_gemv),
    ):
        print(json.dumps({"measuring": name}), flush=True)
        measurements[name] = measure(stream, launch)

    q8_bytes = int(manifest["aggregate"]["device_bytes"])
    q5_payload = Q5_SLOTS * 3 * (983040 + 24576)
    q8_raw_ms, q8_raw_bw = adjusted_gbps(q8_bytes, measurements["q8_raw"], {"stats": {"p50": 0.0}})
    q8_pattern_ms, q8_pattern_bw = adjusted_gbps(q8_bytes, measurements["q8_pattern"], measurements["q8_noop_241"])
    q8_gemv_ms, q8_gemv_bw = adjusted_gbps(q8_bytes, measurements["q8_gemv"], measurements["q8_noop_241"])
    q5_raw_ms, q5_raw_bw = adjusted_gbps(q5_payload, measurements["q5_raw"], {"stats": {"p50": 0.0}})
    q5_pattern_ms, q5_pattern_bw = adjusted_gbps(q5_payload, measurements["q5_pattern"], measurements["q5_noop_96"])
    q5_gemv_ms, q5_gemv_bw = adjusted_gbps(q5_payload, measurements["q5_gemv"], measurements["q5_noop_96"])
    diagnosis = {
        "q8": {"bytes": q8_bytes, "adjusted_ms": {"raw": q8_raw_ms, "pattern": q8_pattern_ms, "gemv": q8_gemv_ms}, "effective_gbps": {"raw": q8_raw_bw, "pattern": q8_pattern_bw, "gemv": q8_gemv_bw}, "pattern_over_raw": q8_pattern_bw / q8_raw_bw, "gemv_over_pattern": q8_gemv_bw / q8_pattern_bw, "classification": classify(q8_raw_bw, q8_pattern_bw, q8_gemv_bw)},
        "q5": {"bytes": q5_payload, "resident_record_bytes": int(q5.size), "adjusted_ms": {"raw": q5_raw_ms, "pattern": q5_pattern_ms, "gemv": q5_gemv_ms}, "effective_gbps": {"raw": q5_raw_bw, "pattern": q5_pattern_bw, "gemv": q5_gemv_bw}, "pattern_over_raw": q5_pattern_bw / q5_raw_bw, "gemv_over_pattern": q5_gemv_bw / q5_pattern_bw, "classification": classify(q5_raw_bw, q5_pattern_bw, q5_gemv_bw)},
    }
    props = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "kind": "streamq5_moe_p7a_kernel_roofline", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
        "seed": SEED, "warmups": WARMUPS, "iterations": ITERATIONS,
        "device": {"name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"], "total_vram": int(total_vram), "free_before": int(free_before), "free_resident": int(free_resident)},
        "inputs": {"q8_manifest_sha256": sha256(Q8_MANIFEST), "q8_pinned_aggregate_sha256": q8_sha, "q5_layers": LAYERS, "q5_experts_per_layer": Q5_EXPERTS_PER_LAYER},
        "source_audit": {"dequantized_weight_scratch_bytes": 0, "weights_reconstructed_from_codes_and_scales_inside_mac_kernel": True},
        "measurements": measurements, "diagnosis": diagnosis,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Physical local kernel diagnosis only; no quality, end-to-end, cross-runtime, or SOTA claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "diagnosis": diagnosis}, indent=2), flush=True)


if __name__ == "__main__":
    main()
