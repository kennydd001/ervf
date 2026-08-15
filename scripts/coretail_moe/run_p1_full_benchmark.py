from __future__ import annotations

import hashlib
import json
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from safetensors import safe_open

from moe_lab.reporting import ROOT
from run_p1a_kernel_smoke import (
    CUDA_SOURCE,
    bf16_bits_to_float,
    load_physical_record,
    time_kernel,
    unpack_codes,
)


PREREG = ROOT / "reports/coretail_moe/P1_FUSED_KERNEL_PREREGISTRATION.md"
LOCK = ROOT / "reports/coretail_moe/p1_fused_kernel_input_lock.json"
P0_RESULT = ROOT / "reports/coretail_moe/p0_full_bank_format_result.json"
P0_VERIFY = ROOT / "reports/coretail_moe/p0_full_bank_format_verification.json"
SMOKE = ROOT / "reports/coretail_moe/p1a_kernel_smoke_result.json"
OUT_JSON = ROOT / "reports/coretail_moe/p1_full_benchmark_result.json"
OUT_MD = ROOT / "reports/coretail_moe/P1_FULL_BENCHMARK_REPORT.md"
DOMAINS = ("general", "instruction", "code", "math", "multilingual")
MATRICES = ("gate", "up", "down")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_state() -> dict:
    fields = "name,memory.used,memory.total,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,pstate"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    return dict(zip(fields.split(","), values))


def errors(observed: np.ndarray, reference: np.ndarray) -> dict:
    delta = observed - reference
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
        "finite": bool(np.isfinite(observed).all()),
    }


def timing(samples: np.ndarray, weights: int) -> dict:
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "p50_weight_applications_per_second": float(weights / (np.percentile(samples, 50) / 1000)),
        "p95_weight_applications_per_second": float(weights / (np.percentile(samples, 95) / 1000)),
    }


def time_torch_mv(weight: torch.Tensor, vector: torch.Tensor, warmup: int, iterations: int):
    for _ in range(warmup):
        torch.mv(weight, vector)
    torch.cuda.synchronize()
    samples = np.empty(iterations, dtype=np.float64)
    output = None
    for index in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = torch.mv(weight, vector)
        end.record(); end.synchronize()
        samples[index] = start.elapsed_time(end)
    return samples, output


def time_entropy_decode_and_copy(packed: np.ndarray, iterations: int = 100):
    encoded = zlib.compress(packed.tobytes(), level=9)
    raw = zlib.decompress(encoded)
    if raw != packed.tobytes():
        raise ValueError("entropy baseline round trip failed")
    decode_ms = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start = time.perf_counter_ns()
        decoded = zlib.decompress(encoded)
        decode_ms[index] = (time.perf_counter_ns() - start) / 1e6
    pinned = cp.cuda.alloc_pinned_memory(len(raw))
    host = np.frombuffer(pinned, dtype=np.uint8)[: len(raw)]
    host[:] = np.frombuffer(raw, dtype=np.uint8)
    device = cp.empty(len(raw), dtype=cp.uint8)
    copy_ms = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record(); device.set(host); end.record(); end.synchronize()
        copy_ms[index] = cp.cuda.get_elapsed_time(start, end)
    del device
    return {
        "raw_bytes": len(raw), "encoded_bytes": len(encoded),
        "decode_p50_ms": float(np.percentile(decode_ms, 50)),
        "decode_p95_ms": float(np.percentile(decode_ms, 95)),
        "h2d_p50_ms": float(np.percentile(copy_ms, 50)),
        "h2d_p95_ms": float(np.percentile(copy_ms, 95)),
    }


def tail_traffic_distribution(result: dict, tokens: int):
    record_map = {record["key"]: record for record in result["records"]}
    raw_cost = np.zeros((48, 128), dtype=np.int64)
    compressed_cost = np.zeros((48, 128), dtype=np.int64)
    for layer in range(48):
        for expert in range(128):
            for name in MATRICES:
                record = record_map[f"{layer}:{expert}:{name}"]
                tail = record["tail"]
                # The measured kernel consumes row-aligned flags. Adding one
                # byte per row is a strict upper bound on repacking padding.
                raw_cost[layer, expert] += tail["raw_flag_bytes"] + tail["rows"] + 4 * (tail["rows"] + 1)
                compressed_cost[layer, expert] += tail["header_bytes"] + tail["index_bytes"] + tail["payload_bytes"]
    raw_totals = []
    compressed_totals = []
    route_hashes = {}
    for domain in DOMAINS:
        domain_raw = np.zeros(tokens, dtype=np.int64)
        domain_compressed = np.zeros(tokens, dtype=np.int64)
        for layer in range(48):
            path = ROOT / f"reports/runs/qwen_gptq_bank/p0_supplement_routes/layer_{layer:02d}.safetensors"
            route_hashes[str(layer)] = sha256(path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                route_ids = handle.get_tensor(f"{domain}_router_ids")[:tokens].numpy().astype(np.int64)
            domain_raw += raw_cost[layer][route_ids].sum(axis=1)
            domain_compressed += compressed_cost[layer][route_ids].sum(axis=1)
        raw_totals.append(domain_raw)
        compressed_totals.append(domain_compressed)
    return np.concatenate(raw_totals), np.concatenate(compressed_totals), route_hashes


def time_pinned_copy(byte_count: int, warmup: int, iterations: int):
    pinned = cp.cuda.alloc_pinned_memory(byte_count)
    host = np.frombuffer(pinned, dtype=np.uint8)[:byte_count]
    host.fill(0xA5)
    device = cp.empty(byte_count, dtype=cp.uint8)
    for _ in range(warmup):
        device.set(host)
    cp.cuda.get_current_stream().synchronize()
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record(); device.set(host); end.record(); end.synchronize()
        samples[index] = cp.cuda.get_elapsed_time(start, end)
    del device
    return samples


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite P1 benchmark")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    p0 = json.loads(P0_VERIFY.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    if p0.get("status") != "p0_pass" or smoke.get("status") != "smoke_pass":
        raise ValueError("P0 and P1A smoke passes are required")
    if sha256(PREREG) != lock["preregistration_sha256"] or sha256(P0_VERIFY) != lock["p0_verification_sha256"]:
        raise ValueError("P1 lock mismatch")
    if sha256(ROOT / lock["input_vectors"]) != lock["input_vectors_sha256"]:
        raise ValueError("input vector artifact changed")
    result = json.loads(P0_RESULT.read_text(encoding="utf-8"))
    vectors = np.load(ROOT / lock["input_vectors"])
    fixed_kernel = cp.RawKernel(CUDA_SOURCE, "fixed_uint2_gemv", options=("--std=c++11",))
    coretail_kernel = cp.RawKernel(CUDA_SOURCE, "coretail_gemv", options=("--std=c++11",))
    before = nvidia_state()
    cp.get_default_memory_pool().free_all_blocks()
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    records = []
    all_correct = True
    total_weights = 0
    fixed_p50_seconds = coretail_p50_seconds = coretail_p95_seconds = bf16_p50_seconds = 0.0
    warmup, iterations = lock["kernel"]["warmup"], lock["kernel"]["iterations"]
    tolerance = lock["tolerances"]

    for layer in lock["layers"]:
        for expert in lock["experts"][str(layer)]:
            for name in MATRICES:
                key = f"{layer}:{expert}:{name}"
                vector_key = f"layer_{layer:02d}_expert_{expert:03d}_{name}"
                x = vectors[vector_key]
                physical_record, rows, cols, physical_scales, bitmap, sign_offsets, signs, tail_offsets, tail_flags = load_physical_record(result, key)
                source_path = ROOT / physical_record["source"]
                with safe_open(source_path, framework="pt", device="cpu") as source:
                    packed = source.get_tensor(f"{name}_codes_packed")[expert].contiguous().numpy()
                    scale_bits = source.get_tensor(f"{name}_scales")[expert].contiguous().view(torch.uint16).numpy()
                if not np.array_equal(physical_scales.reshape(scale_bits.shape), scale_bits):
                    raise ValueError(f"scale mismatch for {key}")
                codes = unpack_codes(packed)
                scale_float = bf16_bits_to_float(scale_bits)
                dequantized = codes.astype(np.float32) * scale_float[:, np.arange(cols) // 128]
                reference = dequantized @ x

                x_gpu = cp.asarray(x); packed_gpu = cp.asarray(packed); scales_gpu = cp.asarray(scale_bits)
                bitmap_gpu = cp.asarray(bitmap); sign_offsets_gpu = cp.asarray(sign_offsets); signs_gpu = cp.asarray(signs)
                tail_offsets_gpu = cp.asarray(tail_offsets); tail_flags_gpu = cp.asarray(tail_flags)
                fixed_output = cp.empty(rows, dtype=cp.float32); coretail_output = cp.empty(rows, dtype=cp.float32)
                fixed_args = (x_gpu, packed_gpu, scales_gpu, fixed_output, np.int32(rows), np.int32(cols))
                coretail_args = (x_gpu, scales_gpu, bitmap_gpu, sign_offsets_gpu, signs_gpu, tail_offsets_gpu, tail_flags_gpu, coretail_output, np.int32(rows), np.int32(cols))
                fixed_kernel((rows,), (256,), fixed_args); coretail_kernel((rows,), (256,), coretail_args)
                cp.cuda.get_current_stream().synchronize()
                fixed_error = errors(cp.asnumpy(fixed_output), reference)
                coretail_error = errors(cp.asnumpy(coretail_output), reference)
                cross_error = errors(cp.asnumpy(coretail_output), cp.asnumpy(fixed_output))
                correct = all(
                    item["finite"] and item["max_abs"] <= tolerance["max_abs"] and item["relative_l2"] <= tolerance["relative_l2"]
                    for item in (fixed_error, coretail_error, cross_error)
                )
                all_correct &= correct
                fixed_samples = time_kernel(fixed_kernel, (rows,), (256,), fixed_args, warmup, iterations)
                coretail_samples = time_kernel(coretail_kernel, (rows,), (256,), coretail_args, warmup, iterations)
                entropy = time_entropy_decode_and_copy(packed)

                cp.cuda.get_current_stream().synchronize()
                weight_torch = torch.from_numpy(dequantized).to(device="cuda", dtype=torch.bfloat16)
                x_torch = torch.from_numpy(x).to(device="cuda", dtype=torch.bfloat16)
                bf16_samples, bf16_output = time_torch_mv(weight_torch, x_torch, warmup, iterations)
                bf16_error = errors(bf16_output.float().cpu().numpy(), reference)
                weight_count = rows * cols
                total_weights += weight_count
                fixed_p50_seconds += np.percentile(fixed_samples, 50) / 1000
                coretail_p50_seconds += np.percentile(coretail_samples, 50) / 1000
                coretail_p95_seconds += np.percentile(coretail_samples, 95) / 1000
                bf16_p50_seconds += np.percentile(bf16_samples, 50) / 1000
                records.append({
                    "key": key, "rows": rows, "cols": cols, "weights": weight_count,
                    "correct": bool(correct),
                    "errors": {"fixed": fixed_error, "coretail": coretail_error, "cross": cross_error, "bf16": bf16_error},
                    "timing": {
                        "bf16_dequantized_reference": timing(bf16_samples, weight_count),
                        "fixed_uint2": timing(fixed_samples, weight_count),
                        "coretail": timing(coretail_samples, weight_count),
                        "entropy_gptq_host_decode_h2d": entropy,
                    },
                    "runtime_bytes": {
                        "fixed_codes": int(packed.nbytes),
                        "scales": int(scale_bits.nbytes),
                        "core_bitmap": int(bitmap.nbytes), "core_signs": int(signs.nbytes),
                        "core_row_offsets": int(sign_offsets.nbytes),
                        "tail_flags_row_aligned": int(tail_flags.nbytes),
                        "tail_row_offsets": int(tail_offsets.nbytes),
                    },
                })
                print(json.dumps({"key": key, "correct": bool(correct), "coretail_gweights_s": weight_count / (np.percentile(coretail_samples, 50) / 1000) / 1e9}), flush=True)
                del weight_torch, x_torch, bf16_output, dequantized
                del x_gpu, packed_gpu, scales_gpu, bitmap_gpu, sign_offsets_gpu, signs_gpu, tail_offsets_gpu, tail_flags_gpu, fixed_output, coretail_output
                torch.cuda.empty_cache(); cp.get_default_memory_pool().free_all_blocks()

    raw_tail_bytes, compressed_tail_bytes, route_hashes = tail_traffic_distribution(result, lock["tail_trace"]["tokens_per_domain"])
    p95_raw_bytes = int(np.percentile(raw_tail_bytes, 95, method="higher"))
    tail_copy_samples = time_pinned_copy(p95_raw_bytes, warmup, iterations)
    tail_copy_p95 = float(np.percentile(tail_copy_samples, 95))
    after = nvidia_state()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    aggregate = {
        "bf16_p50_weight_applications_per_second": total_weights / bf16_p50_seconds,
        "fixed_uint2_p50_weight_applications_per_second": total_weights / fixed_p50_seconds,
        "coretail_p50_weight_applications_per_second": total_weights / coretail_p50_seconds,
        "coretail_p95_weight_applications_per_second": total_weights / coretail_p95_seconds,
    }
    tail = {
        "tokens": int(raw_tail_bytes.size), "domains": list(DOMAINS),
        "raw_runtime_bytes_conservative_upper_bound": {name: int(np.percentile(raw_tail_bytes, percentile, method="higher")) for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))},
        "compressed_physical_bytes": {name: int(np.percentile(compressed_tail_bytes, percentile, method="higher")) for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))},
        "runtime_policy": "full tail is decompressed and row-aligned once at model load; selected flags and row offsets are copied from pinned host RAM per token",
        "one_time_host_cache_bytes_conservative_upper_bound": int(sum(record["tail"]["raw_flag_bytes"] + record["tail"]["rows"] + 4 * (record["tail"]["rows"] + 1) for record in result["records"])),
        "p95_size_copy_timing_ms": {"p50": float(np.percentile(tail_copy_samples, 50)), "p95": tail_copy_p95, "p99": float(np.percentile(tail_copy_samples, 99))},
        "per_token_decode_ms": 0.0,
        "route_artifact_sha256": route_hashes,
    }
    gates = {
        "all_72_correct": bool(all_correct and len(records) == 72),
        "coretail_p50_throughput_ge_27_2_gweights_s": bool(aggregate["coretail_p50_weight_applications_per_second"] >= lock["gates"]["weights_per_second"]),
        "coretail_p95_throughput_ge_27_2_gweights_s": bool(aggregate["coretail_p95_weight_applications_per_second"] >= lock["gates"]["weights_per_second"]),
        "tail_decode_h2d_p95_le_33_3_ms": bool(tail_copy_p95 <= lock["gates"]["tail_decode_h2d_p95_ms"]),
        "no_full_dequantized_matrix_in_custom_kernels": bool("dequantized" not in CUDA_SOURCE),
    }
    passed = all(gates.values())
    payload = {
        "kind": "coretail_moe_p1_full_fused_kernel_benchmark",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p1_pass" if passed else "p1_fail",
        "inputs": {"preregistration_sha256": sha256(PREREG), "lock_sha256": sha256(LOCK), "p0_verification_sha256": sha256(P0_VERIFY), "smoke_sha256": sha256(SMOKE)},
        "environment": {"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "cupy": cp.__version__, "torch": torch.__version__, "before": before, "after": after, "vram_total_bytes": int(total_vram), "free_before_bytes": int(free_before), "free_after_bytes": int(free_after)},
        "aggregate": aggregate, "tail": tail, "gates": gates, "records": records,
        "claim_boundary": "P1 measures exact microkernel throughput and pinned-tail traffic for the locked construction. It does not prove full-model quality or end-to-end tokens per second; those require P2 and the later wall-clock run.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join([
        "# CORETAIL-MoE P1 — volledige fused-kernelbenchmark", "",
        f"Uitkomst: **{payload['status']}** ({sum(gates.values())}/{len(gates)} gates).", "",
        f"CORETAIL aggregate p50: {aggregate['coretail_p50_weight_applications_per_second']/1e9:.3f} Gweight/s; p95: {aggregate['coretail_p95_weight_applications_per_second']/1e9:.3f} Gweight/s; gate: 27,2.",
        f"Pinned tail H2D bij de werkelijke p95-tokenomvang van {p95_raw_bytes/2**20:.3f} MiB: p95 {tail_copy_p95:.3f} ms; gate: 33,3 ms.",
        f"Correctheid: {sum(record['correct'] for record in records)}/{len(records)} gelockte matrixgevallen.", "",
        "De runtime houdt de volledige tail eenmaal gedecomprimeerd in host-RAM; per token worden uitsluitend geselecteerde raw flags en row offsets gekopieerd. P1 bewijst geen modelkwaliteit of end-to-end tok/s.", "",
    ]), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gates": gates, "aggregate": aggregate, "tail": tail}, indent=2), flush=True)
