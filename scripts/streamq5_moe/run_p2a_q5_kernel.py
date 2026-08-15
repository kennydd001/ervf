from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import torch

from moe_lab.reporting import ROOT


REPORT_DIR = ROOT / "reports/streamq5_moe"
PREREG = REPORT_DIR / "P2A_Q5_KERNEL_PREREGISTRATION.md"
LOCK = REPORT_DIR / "p2a_kernel_input_lock.json"
SMOKE_EVALUATOR_LOCK = REPORT_DIR / "p2a_kernel_smoke_evaluator_lock.json"
BENCHMARK_EVALUATOR_LOCK = REPORT_DIR / "p2a_kernel_benchmark_evaluator_lock.json"
P1D_VERIFY = REPORT_DIR / "p1d_physical_bank_verification.json"
BANK_DIR = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
SMOKE = REPORT_DIR / "p2a_kernel_smoke.json"
SMOKE_REPORT = REPORT_DIR / "P2A_KERNEL_SMOKE.md"
RESULT = REPORT_DIR / "p2a_kernel_benchmark.json"
REPORT = REPORT_DIR / "P2A_KERNEL_BENCHMARK.md"

HEADER = struct.Struct("<4sHHHBBIIH2xIII28s")
HEADER_BYTES, RECORD_BYTES = 64, 1_011_712
CODE_BYTES, SCALE_BYTES = 983_040, 24_576
MATRICES = (("gate", 768, 2048, 0), ("up", 768, 2048, 1), ("down", 2048, 768, 2))
FULL_TOKEN_EXPERT_WEIGHTS = 1_811_939_328


CUDA_SOURCE = r'''
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

extern "C" __global__ void physical_q5_gemv(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    float* output, int rows, int cols) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
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
    __shared__ float reduction[256];
    reduction[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = reduction[0];
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_state() -> dict:
    fields = "name,memory.used,memory.total,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,pstate"
    completed = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    return dict(zip(fields.split(","), (value.strip() for value in completed.stdout.strip().split(","))))


def bf16_to_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def unpack_q5(packed: np.ndarray, rows: int, columns: int) -> np.ndarray:
    chunks = packed.reshape(-1, 5).astype(np.uint64)
    words = chunks[:, 0] | (chunks[:, 1] << 8) | (chunks[:, 2] << 16) | (chunks[:, 3] << 24) | (chunks[:, 4] << 32)
    values = np.stack([((words >> (slot * 5)) & 31).astype(np.int8) - 15 for slot in range(8)], axis=-1)
    return values.reshape(rows, columns)


def load_record(layer: int, expert: int, matrix_index: int):
    name, rows, columns, projection = MATRICES[matrix_index]
    path = BANK_DIR / f"layer_{layer:02d}.q5bin"
    offset = (expert * 3 + matrix_index) * RECORD_BYTES
    with path.open("rb") as handle:
        handle.seek(offset)
        raw_header = handle.read(HEADER_BYTES)
        fields = HEADER.unpack(raw_header)
        expected = (b"SQ5M", 1, layer, expert, projection, 5, rows, columns, 128, CODE_BYTES, SCALE_BYTES)
        if fields[:11] != expected or fields[12] != b"\x00" * 28:
            raise ValueError(f"physical header mismatch {layer}:{expert}:{name}")
        packed = np.frombuffer(handle.read(CODE_BYTES), dtype=np.uint8).copy()
        scale_bits = np.frombuffer(handle.read(SCALE_BYTES), dtype="<u2").copy()
        padding = handle.read(RECORD_BYTES - HEADER_BYTES - CODE_BYTES - SCALE_BYTES)
    crc = zlib.crc32(packed); crc = zlib.crc32(scale_bits, crc) & 0xFFFFFFFF
    if crc != fields[11] or any(padding):
        raise ValueError(f"physical CRC/padding mismatch {layer}:{expert}:{name}")
    return name, rows, columns, packed, scale_bits.reshape(rows, columns // 128)


def reference(packed: np.ndarray, scale_bits: np.ndarray, rows: int, columns: int, vector: np.ndarray):
    codes = unpack_q5(packed, rows, columns)
    scales = bf16_to_float(scale_bits)
    weights = codes.astype(np.float32) * scales[:, np.arange(columns) // 128]
    return weights @ vector, weights


def errors(observed: np.ndarray, expected: np.ndarray) -> dict:
    delta = observed - expected
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30)),
        "finite": bool(np.isfinite(observed).all()),
    }


def time_kernel(kernel, args, rows: int, warmup: int, iterations: int) -> np.ndarray:
    for _ in range(warmup):
        kernel((rows,), (256,), args)
    cp.cuda.get_current_stream().synchronize()
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record(); kernel((rows,), (256,), args); end.record(); end.synchronize()
        samples[index] = cp.cuda.get_elapsed_time(begin, end)
    return samples


def time_bf16(weight: np.ndarray, vector: np.ndarray, warmup: int, iterations: int) -> np.ndarray:
    matrix = torch.from_numpy(weight).to("cuda", dtype=torch.bfloat16)
    x = torch.from_numpy(vector).to("cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.mv(matrix, x)
    torch.cuda.synchronize()
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(); torch.mv(matrix, x); end.record(); end.synchronize()
        samples[index] = begin.elapsed_time(end)
    del matrix, x
    return samples


def stats(samples: np.ndarray, weights: int) -> dict:
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "p50_weight_applications_per_second": float(weights / (np.percentile(samples, 50) / 1000)),
        "p95_weight_applications_per_second": float(weights / (np.percentile(samples, 95) / 1000)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "benchmark"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output, report = (SMOKE, SMOKE_REPORT) if args.phase == "smoke" else (RESULT, REPORT)
    if output.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite P2A {args.phase}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    active_evaluator_lock = SMOKE_EVALUATOR_LOCK if args.phase == "smoke" else BENCHMARK_EVALUATOR_LOCK
    evaluator_lock = json.loads(active_evaluator_lock.read_text(encoding="utf-8"))
    p1d = json.loads(P1D_VERIFY.read_text(encoding="utf-8"))
    if p1d.get("status") != "p1d_physical_bank_verification_pass":
        raise RuntimeError("P1D pass required")
    if sha256(Path(__file__)) != evaluator_lock["evaluator_sha256"] or sha256(LOCK) != evaluator_lock["input_lock_sha256"] or sha256(PREREG) != lock["preregistration_sha256"] or sha256(P1D_VERIFY) != lock["p1d_verification_sha256"]:
        raise ValueError("P2A evaluator/input provenance mismatch")
    if args.phase == "benchmark":
        smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
        if smoke.get("status") != "smoke_pass" or sha256(SMOKE) != evaluator_lock.get("smoke_sha256"):
            raise RuntimeError("locked smoke pass required before benchmark")

    vectors = np.load(ROOT / lock["input_vectors"])
    if sha256(ROOT / lock["input_vectors"]) != lock["input_vectors_sha256"]:
        raise ValueError("vector artifact mismatch")
    kernel = cp.RawKernel(CUDA_SOURCE, "physical_q5_gemv", options=("--std=c++11",))
    tolerance = lock["tolerances"]
    warmup, iterations = lock["kernel"]["warmup"], lock["kernel"]["iterations"]
    cases = []
    selected = [(lock["layers"][0], lock["experts"][str(lock["layers"][0])][0], 0)] if args.phase == "smoke" else [
        (layer, expert, matrix_index)
        for layer in lock["layers"]
        for expert in lock["experts"][str(layer)]
        for matrix_index in range(3)
    ]
    before = gpu_state(); started = time.perf_counter()
    for layer, expert, matrix_index in selected:
        name, rows, columns, packed, scale_bits = load_record(layer, expert, matrix_index)
        key = f"layer_{layer:02d}_expert_{expert:03d}_{name}"
        vector = vectors[key]
        expected, dequantized_baseline = reference(packed, scale_bits, rows, columns, vector)
        x_gpu = cp.asarray(vector); packed_gpu = cp.asarray(packed); scales_gpu = cp.asarray(scale_bits); out_gpu = cp.empty(rows, dtype=cp.float32)
        kernel_args = (x_gpu, packed_gpu, scales_gpu, out_gpu, np.int32(rows), np.int32(columns))
        kernel((rows,), (256,), kernel_args); cp.cuda.get_current_stream().synchronize()
        error = errors(cp.asnumpy(out_gpu), expected)
        correct = error["finite"] and error["max_abs"] <= tolerance["max_abs"] and error["relative_l2"] <= tolerance["relative_l2"]
        row = {"key": key, "layer": layer, "expert": expert, "matrix": name, "rows": rows, "columns": columns, "weights": rows * columns, "correct": bool(correct), "error": error, "physical_bytes": int(CODE_BYTES + SCALE_BYTES)}
        if args.phase == "benchmark":
            q5_samples = time_kernel(kernel, kernel_args, rows, warmup, iterations)
            bf16_samples = time_bf16(dequantized_baseline, vector, warmup, iterations)
            row["timing"] = {"physical_q5": stats(q5_samples, rows * columns), "bf16_dequantized_baseline": stats(bf16_samples, rows * columns)}
        cases.append(row)
        print(json.dumps({"key": key, "correct": bool(correct), "error": error, "q5_p50_ms": row.get("timing", {}).get("physical_q5", {}).get("p50_ms")}), flush=True)
        del x_gpu, packed_gpu, scales_gpu, out_gpu, expected, dequantized_baseline
        cp.get_default_memory_pool().free_all_blocks(); torch.cuda.empty_cache()

    all_correct = all(row["correct"] for row in cases)
    if args.phase == "smoke":
        status = "smoke_pass" if all_correct and len(cases) == 1 else "smoke_fail"
        payload = {"kind": "streamq5_moe_p2a_q5_kernel_smoke", "completed_utc": datetime.now(timezone.utc).isoformat(), "status": status, "cases": cases, "claim_boundary": "Untimed single-case toolchain and correctness smoke only."}
    else:
        total_weights = sum(row["weights"] for row in cases)
        p50_seconds = sum(row["timing"]["physical_q5"]["p50_ms"] for row in cases) / 1000
        p95_seconds = sum(row["timing"]["physical_q5"]["p95_ms"] for row in cases) / 1000
        bf16_seconds = sum(row["timing"]["bf16_dequantized_baseline"]["p50_ms"] for row in cases) / 1000
        p50_throughput = total_weights / p50_seconds
        p95_throughput = total_weights / p95_seconds
        full_token_p50_ms = FULL_TOKEN_EXPERT_WEIGHTS / p50_throughput * 1000
        full_token_p95_ms = FULL_TOKEN_EXPERT_WEIGHTS / p95_throughput * 1000
        aggregate = {"cases": len(cases), "sample_weights": total_weights, "full_token_expert_weights": FULL_TOKEN_EXPERT_WEIGHTS, "q5_p50_weight_applications_per_second": p50_throughput, "q5_summed_p95_weight_applications_per_second": p95_throughput, "bf16_p50_weight_applications_per_second": total_weights / bf16_seconds, "full_token_q5_p50_compute_ms": full_token_p50_ms, "full_token_q5_summed_p95_compute_ms": full_token_p95_ms}
        gates = {"all_72_correct": all_correct and len(cases) == 72, "p50_throughput_ge_27_2_gweights_s": p50_throughput >= lock["gates"]["weights_per_second"], "summed_p95_throughput_ge_27_2_gweights_s": p95_throughput >= lock["gates"]["weights_per_second"], "full_token_p95_compute_ms_le_66_615": full_token_p95_ms <= lock["gates"]["full_token_p95_compute_ms_max"], "direct_physical_q5_no_dequantized_candidate_matrix": "float* dequantized" not in CUDA_SOURCE and "output[row]" in CUDA_SOURCE}
        status = "p2a_kernel_pass" if all(gates.values()) else "p2a_kernel_closed"
        payload = {"kind": "streamq5_moe_p2a_physical_q5_kernel_benchmark", "completed_utc": datetime.now(timezone.utc).isoformat(), "status": status, "inputs": {"preregistration_sha256": sha256(PREREG), "input_lock_sha256": sha256(LOCK), "evaluator_lock_sha256": sha256(active_evaluator_lock), "evaluator_sha256": sha256(Path(__file__)), "p1d_verification_sha256": sha256(P1D_VERIFY), "smoke_sha256": sha256(SMOKE), "vectors_sha256": sha256(ROOT / lock["input_vectors"])}, "environment": {"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "cupy": cp.__version__, "torch": torch.__version__, "before": before, "after": gpu_state()}, "aggregate": aggregate, "gates": gates, "cases": cases, "runtime_seconds": time.perf_counter() - started, "claim_boundary": "Physical Q5 GEMV correctness and microkernel throughput only; fragmented H2D and integrated full-token wall-clock remain unproven."}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(f"# STREAMQ5-MoE P2A - {args.phase}\n\nUitkomst: **{payload['status']}**.\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "aggregate", "gates") if key in payload}, indent=2), flush=True)
    if payload["status"].endswith("fail") or payload["status"].endswith("closed"):
        raise SystemExit(1)
