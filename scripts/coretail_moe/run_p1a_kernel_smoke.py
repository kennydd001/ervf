from __future__ import annotations

import binascii
import hashlib
import json
import math
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from safetensors import safe_open

from moe_lab.reporting import ROOT


LOCK = ROOT / "reports/coretail_moe/p1_fused_kernel_input_lock.json"
P0_RESULT = ROOT / "reports/coretail_moe/p0_full_bank_format_result.json"
P0_VERIFY = ROOT / "reports/coretail_moe/p0_full_bank_format_verification.json"
OUT_JSON = ROOT / "reports/coretail_moe/p1a_kernel_smoke_result.json"
OUT_MD = ROOT / "reports/coretail_moe/P1A_KERNEL_SMOKE_REPORT.md"
CORE_HEADER = struct.Struct("<8sHHB3xIIHHQQQQI")
TAIL_HEADER = struct.Struct("<8sHHB3xIIHHQIIQQI")
TAIL_INDEX = struct.Struct("<QIIIIB7x")


CUDA_SOURCE = r'''
__device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

extern "C" __global__ void fixed_uint2_gemv(
    const float* x, const unsigned char* packed, const unsigned short* scales,
    float* output, int rows, int cols) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int packed_cols = cols >> 2;
    int groups = cols >> 7;
    float sum = 0.0f;
    for (int byte_id = tid; byte_id < packed_cols; byte_id += blockDim.x) {
        unsigned char value = packed[row * packed_cols + byte_id];
        int column = byte_id << 2;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int code = ((value >> (2 * k)) & 3) - 2;
            float scale = bf16_to_float(scales[row * groups + ((column + k) >> 7)]);
            sum += ((float)code) * scale * x[column + k];
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

extern "C" __global__ void coretail_gemv(
    const float* x, const unsigned short* scales, const unsigned char* bitmap,
    const unsigned int* sign_offsets, const unsigned char* signs,
    const unsigned int* tail_offsets, const unsigned char* tail_flags,
    float* output, int rows, int cols) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bitmap_cols = cols >> 3;
    int groups = cols >> 7;
    unsigned char mask = tid < bitmap_cols ? bitmap[row * bitmap_cols + tid] : 0;
    int nonzeros = __popc((unsigned int)mask);
    __shared__ int scan[256];
    __shared__ int negative_scan[256];
    scan[tid] = nonzeros;
    __syncthreads();
    for (int offset = 1; offset < 256; offset <<= 1) {
        int add = tid >= offset ? scan[tid - offset] : 0;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    int nonzero_base = scan[tid] - nonzeros;
    unsigned int sign_base = sign_offsets[row] << 3;
    int local_nonzero = 0;
    int negatives = 0;
    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        if (mask & (1u << bit)) {
            unsigned int index = sign_base + nonzero_base + local_nonzero;
            int positive = (signs[index >> 3] >> (index & 7)) & 1;
            negatives += 1 - positive;
            local_nonzero += 1;
        }
    }
    negative_scan[tid] = negatives;
    __syncthreads();
    for (int offset = 1; offset < 256; offset <<= 1) {
        int add = tid >= offset ? negative_scan[tid - offset] : 0;
        __syncthreads();
        negative_scan[tid] += add;
        __syncthreads();
    }
    int negative_base = negative_scan[tid] - negatives;
    unsigned int tail_base = tail_offsets[row] << 3;
    float sum = 0.0f;
    local_nonzero = 0;
    int local_negative = 0;
    if (tid < bitmap_cols) {
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            if (mask & (1u << bit)) {
                unsigned int sign_index = sign_base + nonzero_base + local_nonzero;
                int positive = (signs[sign_index >> 3] >> (sign_index & 7)) & 1;
                int code;
                if (positive) {
                    code = 1;
                } else {
                    unsigned int tail_index = tail_base + negative_base + local_negative;
                    int extreme = (tail_flags[tail_index >> 3] >> (tail_index & 7)) & 1;
                    code = -1 - extreme;
                    local_negative += 1;
                }
                int column = (tid << 3) + bit;
                float scale = bf16_to_float(scales[row * groups + (column >> 7)]);
                sum += ((float)code) * scale * x[column];
                local_nonzero += 1;
            }
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


def bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def unpack_codes(packed: np.ndarray) -> np.ndarray:
    values = np.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), axis=-1)
    return values.reshape(*packed.shape[:-1], -1).astype(np.int8) - 2


def load_physical_record(result: dict, key: str):
    record = next(item for item in result["records"] if item["key"] == key)
    core_path = ROOT / result["actual_full_bank"]["core_file"]
    tail_path = ROOT / result["actual_full_bank"]["tail_file"]
    with core_path.open("rb") as handle:
        handle.seek(record["core"]["start"])
        fields = CORE_HEADER.unpack(handle.read(CORE_HEADER.size))
        _, _, _, _, rows, cols, _, _, scale_n, bitmap_n, offsets_n, sign_n, crc = fields
        payload = handle.read(scale_n + bitmap_n + offsets_n + sign_n)
    if (binascii.crc32(payload) & 0xFFFFFFFF) != crc:
        raise ValueError("core CRC mismatch")
    cursor = 0
    scales = np.frombuffer(payload[cursor : cursor + scale_n], dtype="<u2").copy(); cursor += scale_n
    bitmap = np.frombuffer(payload[cursor : cursor + bitmap_n], dtype=np.uint8).copy(); cursor += bitmap_n
    sign_offsets = np.frombuffer(payload[cursor : cursor + offsets_n], dtype="<u4").copy(); cursor += offsets_n
    signs = np.frombuffer(payload[cursor : cursor + sign_n], dtype=np.uint8).copy()
    bitmap_matrix = np.unpackbits(
        bitmap.reshape(rows, math.ceil(cols / 8)), axis=1, bitorder="little"
    )[:, :cols].astype(bool)
    negative_counts = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        count = int(bitmap_matrix[row].sum())
        row_signs = np.unpackbits(
            signs[sign_offsets[row] : sign_offsets[row + 1]], bitorder="little"
        )[:count]
        negative_counts[row] = count - int(row_signs.sum())

    with tail_path.open("rb") as handle:
        handle.seek(record["tail"]["start"])
        fields = TAIL_HEADER.unpack(handle.read(TAIL_HEADER.size))
        _, _, _, _, tail_rows, tail_cols, block_rows, _, negative_bits, blocks, index_n, payload_n, raw_n, tail_crc = fields
        tail_record = handle.read(index_n + payload_n)
    if (binascii.crc32(tail_record) & 0xFFFFFFFF) != tail_crc:
        raise ValueError("tail CRC mismatch")
    index, compressed_payload = tail_record[:index_n], tail_record[index_n:]
    row_tail_parts: list[bytes] = []
    tail_offsets = np.empty(rows + 1, dtype="<u4")
    tail_offsets[0] = 0
    observed_bits = observed_raw = 0
    for block_id in range(blocks):
        entry = TAIL_INDEX.unpack(index[block_id * TAIL_INDEX.size : (block_id + 1) * TAIL_INDEX.size])
        position, stored_n, raw_len, bit_count, block_crc, codec = entry
        stored = compressed_payload[position : position + stored_n]
        raw = zlib.decompress(stored) if codec == 1 else stored
        if len(raw) != raw_len or (binascii.crc32(raw) & 0xFFFFFFFF) != block_crc:
            raise ValueError("tail block mismatch")
        flags = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:bit_count]
        flag_cursor = 0
        row_start = block_id * block_rows
        for row in range(row_start, min(rows, row_start + block_rows)):
            count = int(negative_counts[row])
            row_raw = np.packbits(flags[flag_cursor : flag_cursor + count], bitorder="little").tobytes()
            row_tail_parts.append(row_raw)
            tail_offsets[row + 1] = tail_offsets[row] + len(row_raw)
            flag_cursor += count
        if flag_cursor != bit_count:
            raise ValueError("tail row split mismatch")
        observed_bits += bit_count
        observed_raw += raw_len
    if observed_bits != negative_bits or observed_raw != raw_n or tail_rows != rows or tail_cols != cols:
        raise ValueError("tail totals mismatch")
    return record, rows, cols, scales, bitmap, sign_offsets, signs, tail_offsets, np.frombuffer(b"".join(row_tail_parts), dtype=np.uint8).copy()


def time_kernel(kernel, grid, block, arguments, warmup: int, iterations: int):
    for _ in range(warmup):
        kernel(grid, block, arguments)
    cp.cuda.get_current_stream().synchronize()
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        kernel(grid, block, arguments)
        end.record(); end.synchronize()
        samples[index] = cp.cuda.get_elapsed_time(start, end)
    return samples


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite P1A smoke")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    p0_verify = json.loads(P0_VERIFY.read_text(encoding="utf-8"))
    if p0_verify.get("status") != "p0_pass" or sha256(P0_VERIFY) != lock["p0_verification_sha256"]:
        raise ValueError("P0 verification lock mismatch")
    result = json.loads(P0_RESULT.read_text(encoding="utf-8"))
    layer, expert, name = 0, lock["experts"]["0"][0], "gate"
    key = f"{layer}:{expert}:{name}"
    vector_key = f"layer_{layer:02d}_expert_{expert:03d}_{name}"
    vectors = np.load(ROOT / lock["input_vectors"])
    x = vectors[vector_key]
    record, rows, cols, physical_scales, bitmap, sign_offsets, signs, tail_offsets, tail_flags = load_physical_record(result, key)
    source_path = ROOT / record["source"]
    with safe_open(source_path, framework="pt", device="cpu") as source:
        packed = source.get_tensor(f"{name}_codes_packed")[expert].contiguous().numpy()
        source_scale_bits = source.get_tensor(f"{name}_scales")[expert].contiguous().view(torch.uint16).numpy()
    if not np.array_equal(physical_scales.reshape(source_scale_bits.shape), source_scale_bits):
        raise ValueError("physical/source scales differ")
    codes = unpack_codes(packed)
    scale_float = bf16_bits_to_float(source_scale_bits)
    reference = (codes.astype(np.float32) * scale_float[:, np.arange(cols) // 128]) @ x

    fixed_kernel = cp.RawKernel(CUDA_SOURCE, "fixed_uint2_gemv", options=("--std=c++11",))
    coretail_kernel = cp.RawKernel(CUDA_SOURCE, "coretail_gemv", options=("--std=c++11",))
    x_gpu = cp.asarray(x)
    packed_gpu = cp.asarray(packed)
    scales_gpu = cp.asarray(source_scale_bits)
    bitmap_gpu = cp.asarray(bitmap)
    sign_offsets_gpu = cp.asarray(sign_offsets)
    signs_gpu = cp.asarray(signs)
    tail_offsets_gpu = cp.asarray(tail_offsets)
    tail_flags_gpu = cp.asarray(tail_flags)
    fixed_output = cp.empty(rows, dtype=cp.float32)
    coretail_output = cp.empty(rows, dtype=cp.float32)
    fixed_args = (x_gpu, packed_gpu, scales_gpu, fixed_output, np.int32(rows), np.int32(cols))
    coretail_args = (
        x_gpu, scales_gpu, bitmap_gpu, sign_offsets_gpu, signs_gpu,
        tail_offsets_gpu, tail_flags_gpu, coretail_output, np.int32(rows), np.int32(cols),
    )
    fixed_kernel((rows,), (256,), fixed_args)
    coretail_kernel((rows,), (256,), coretail_args)
    cp.cuda.get_current_stream().synchronize()
    fixed_host = cp.asnumpy(fixed_output)
    coretail_host = cp.asnumpy(coretail_output)

    def errors(observed: np.ndarray):
        delta = observed - reference
        return {
            "max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
            "finite": bool(np.isfinite(observed).all()),
        }

    fixed_error = errors(fixed_host)
    coretail_error = errors(coretail_host)
    cross_error = {
        "max_abs": float(np.max(np.abs(coretail_host - fixed_host))),
        "relative_l2": float(np.linalg.norm(coretail_host - fixed_host) / max(np.linalg.norm(fixed_host), 1e-30)),
        "finite": bool(np.isfinite(coretail_host).all() and np.isfinite(fixed_host).all()),
    }
    tolerance = lock["tolerances"]
    correctness = {
        "fixed_within_tolerance": fixed_error["finite"] and fixed_error["max_abs"] <= tolerance["max_abs"] and fixed_error["relative_l2"] <= tolerance["relative_l2"],
        "coretail_within_tolerance": coretail_error["finite"] and coretail_error["max_abs"] <= tolerance["max_abs"] and coretail_error["relative_l2"] <= tolerance["relative_l2"],
        "fixed_coretail_within_tolerance": cross_error["finite"] and cross_error["max_abs"] <= tolerance["max_abs"] and cross_error["relative_l2"] <= tolerance["relative_l2"],
    }
    warmup, iterations = lock["kernel"]["warmup"], lock["kernel"]["iterations"]
    fixed_ms = time_kernel(fixed_kernel, (rows,), (256,), fixed_args, warmup, iterations)
    coretail_ms = time_kernel(coretail_kernel, (rows,), (256,), coretail_args, warmup, iterations)
    weights = rows * cols

    def timing(samples: np.ndarray):
        return {
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)),
            "median_weight_applications_per_second": float(weights / (np.median(samples) / 1000)),
        }

    payload = {
        "kind": "coretail_moe_p1a_kernel_toolchain_correctness_smoke",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_pass" if all(correctness.values()) else "smoke_fail",
        "claim_boundary": "Toolchain and one locked correctness case only; timings are diagnostic and cannot satisfy P1.",
        "inputs": {"lock_sha256": sha256(LOCK), "key": key, "vector_key": vector_key},
        "environment": {
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__, "torch": torch.__version__,
        },
        "physical_runtime_inputs": {
            "core_bitmap_bytes": int(bitmap.nbytes), "core_sign_bytes": int(signs.nbytes),
            "core_row_offset_bytes": int(sign_offsets.nbytes),
            "scale_bytes": int(source_scale_bits.nbytes),
            "decoded_tail_flag_bytes": int(tail_flags.nbytes),
            "decoded_tail_row_offset_bytes": int(tail_offsets.nbytes),
        },
        "correctness": correctness,
        "errors": {"fixed_vs_reference": fixed_error, "coretail_vs_reference": coretail_error, "coretail_vs_fixed": cross_error},
        "diagnostic_timing": {"fixed_uint2": timing(fixed_ms), "coretail": timing(coretail_ms)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# CORETAIL-MoE P1A — NVRTC-kernelsmoke", "",
        f"Uitkomst: **{payload['status']}** op `{key}`.", "",
        f"Fixed uint2 fout: max_abs={fixed_error['max_abs']:.6g}, relative_l2={fixed_error['relative_l2']:.6g}.",
        f"CORETAIL fout: max_abs={coretail_error['max_abs']:.6g}, relative_l2={coretail_error['relative_l2']:.6g}.",
        f"Onderlinge fout: max_abs={cross_error['max_abs']:.6g}, relative_l2={cross_error['relative_l2']:.6g}.", "",
        "Dit is uitsluitend de vooraf toegestane toolchain-/correctheidssmoke. De diagnostische timing opent of sluit P1 niet.", "",
    ]), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "correctness": correctness, "errors": payload["errors"], "diagnostic_timing": payload["diagnostic_timing"]}, indent=2))
