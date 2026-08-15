from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import psutil

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_port80b_d2_registered_scatter import (
    EXPECTED_BANK_SHA256,
    REGISTER_FLAGS,
    TOKEN_BYTES,
    full_verify,
    header_reference,
    record_offset,
    routes,
    stats,
    unregister_ranges,
)
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import (
    BANK, BANK_BYTES, EXPERT_BYTES, LAYERS, MANIFEST, HardPageReadSampler,
)

R = ROOT / "reports/streamq5_moe"
PREREG = R / "PORT80B_D3R_MAPPED_HOST_KERNEL_PREREGISTRATION.md"
OUTPUT = R / "port80b_d3r_mapped_host_kernel.json"
REPORT = R / "PORT80B_D3R_MAPPED_HOST_KERNEL_REPORT_2026-08-12.md"
EXPERTS = 307
SCHEDULES = (512, 1024, 2048, 4096)
THREADS = 256
VALIDATION_WARMUPS = 6
VALIDATION_ROUNDS = 24
TEST_ROUNDS = 120


KERNEL_SOURCE = r'''
extern "C" __global__ void mapped_host_scatter(
    const unsigned long long* source_pointers,
    unsigned char* destination,
    unsigned long long total) {
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  const unsigned long long record_bytes = 2027520ULL;
  for (; index < total; index += stride) {
    unsigned long long record = index / record_bytes;
    unsigned long long within = index - record * record_bytes;
    const unsigned char* source = (const unsigned char*)source_pointers[record];
    destination[index] = source[within];
  }
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_and_alias(mapped: np.memmap) -> tuple[list[int], list[int]]:
    host_pointers: list[int] = []
    device_pointers: list[int] = []
    size = EXPERTS * EXPERT_BYTES
    try:
        for layer in range(LAYERS):
            host_pointer = int(mapped.ctypes.data) + record_offset(layer, 0)
            cp.cuda.runtime.hostRegister(host_pointer, size, REGISTER_FLAGS)
            host_pointers.append(host_pointer)
            attrs = cp.cuda.runtime.pointerGetAttributes(host_pointer)
            device_pointer = int(attrs.devicePointer)
            if not device_pointer:
                raise RuntimeError(f"layer {layer} returned a null mapped device pointer")
            device_pointers.append(device_pointer)
    except Exception:
        unregister_ranges(host_pointers)
        raise
    return host_pointers, device_pointers


def pointer_row(token: int, aliases: list[int]) -> np.ndarray:
    result = np.empty(LAYERS * 10, dtype=np.uint64)
    for index, (layer, expert) in enumerate(routes(token, EXPERTS)):
        result[index] = np.uint64(aliases[layer] + expert * EXPERT_BYTES)
    return result


def launch(kernel: cp.RawKernel, blocks: int, pointer_row_device: cp.ndarray, destination: cp.ndarray, stream: cp.cuda.Stream) -> None:
    kernel((blocks,), (THREADS,), (pointer_row_device, destination, np.uint64(TOKEN_BYTES)), stream=stream)


def timed(kernel: cp.RawKernel, blocks: int, pointer_row_device: cp.ndarray, destination: cp.ndarray, stream: cp.cuda.Stream) -> float:
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record(stream)
    launch(kernel, blocks, pointer_row_device, destination, stream)
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(begin, end))


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D3 result")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not BANK.is_file() or BANK.stat().st_size != BANK_BYTES or manifest.get("bank_sha256") != EXPECTED_BANK_SHA256:
        raise RuntimeError("immutable P0 bank/manifest contract failed")

    started = time.perf_counter()
    try:
        cp.cuda.runtime.setDeviceFlags(0x08)
        map_flag = {"success": True}
    except Exception as exc:
        map_flag = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    cp.cuda.Device(0).use()
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    kernel = cp.RawKernel(KERNEL_SOURCE, "mapped_host_scatter", options=("--std=c++14",))
    # Reuse D2's independent full-byte structural verifier kernel source.
    from scripts.streamq5_moe.run_port80b_d2_registered_scatter import VERIFY_SOURCE
    verify_kernel = cp.RawKernel(VERIFY_SOURCE, "verify_record_bytes", options=("--std=c++14",))
    destination = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)
    host_pointers: list[int] = []
    sampler = HardPageReadSampler()
    error = None
    result_payload: dict[str, object] = {}
    unregister_failures: list[str] = []
    try:
        registration_started = time.perf_counter()
        host_pointers, aliases = register_and_alias(mapped)
        registration_seconds = time.perf_counter() - registration_started
        memory_after_registration = dict(psutil.virtual_memory()._asdict())

        validation_tokens = list(range(50_000, 50_000 + VALIDATION_ROUNDS))
        test_tokens = list(range(60_000, 60_000 + TEST_ROUNDS))
        all_tokens = validation_tokens + test_tokens + [49_999]
        host_table = np.stack([pointer_row(token, aliases) for token in all_tokens], axis=0)
        device_table = cp.asarray(host_table)
        token_to_row = {token: index for index, token in enumerate(all_tokens)}

        correctness_token = 49_999
        launch(kernel, 4096, device_table[token_to_row[correctness_token]], destination, stream)
        stream.synchronize()
        mismatch_count = full_verify(verify_kernel, destination, routes(correctness_token, EXPERTS), stream)

        for blocks in SCHEDULES:
            for warmup in range(VALIDATION_WARMUPS):
                token = validation_tokens[warmup % len(validation_tokens)]
                launch(kernel, blocks, device_table[token_to_row[token]], destination, stream)
        stream.synchronize()

        raw_validation = {str(blocks): [] for blocks in SCHEDULES}
        orders = []
        for round_index, token in enumerate(validation_tokens):
            rotation = round_index % len(SCHEDULES)
            order = list(SCHEDULES[rotation:] + SCHEDULES[:rotation])
            if round_index & 1:
                order.reverse()
            orders.append(order)
            for blocks in order:
                raw_validation[str(blocks)].append(timed(kernel, blocks, device_table[token_to_row[token]], destination, stream))
        validation = {key: {"raw_ms": values, "stats": stats(values)} for key, values in raw_validation.items()}
        selected_blocks = min(SCHEDULES, key=lambda value: (float(validation[str(value)]["stats"]["p50"]), value))
        validation_open = mismatch_count == 0 and float(validation[str(selected_blocks)]["stats"]["p50"]) <= 65.0

        raw_test: list[float] = []
        page_rows = []
        if validation_open:
            sampler.start()
            for token in test_tokens:
                raw_test.append(timed(kernel, selected_blocks, device_table[token_to_row[token]], destination, stream))
            sampler.stop()
            page_rows = sampler.samples
        test_stats = stats(raw_test) if raw_test else None
        effective = TOKEN_BYTES / (float(test_stats["p95"]) / 1000.0) / 1e9 if test_stats else None
        gates = {
            "registration_48_ranges": len(host_pointers) == LAYERS,
            "full_destination_zero_mismatches": mismatch_count == 0,
            "validation_open": validation_open,
            "test_120_finite": len(raw_test) == TEST_ROUNDS and bool(np.isfinite(raw_test).all()),
            "test_p95_le_65ms": bool(test_stats and float(test_stats["p95"]) <= 65.0),
            "effective_gb_s_at_p95_ge_15": bool(effective is not None and effective >= 15.0),
            "strong_test_p95_le_45ms": bool(test_stats and float(test_stats["p95"]) <= 45.0),
            "strong_effective_gb_s_at_p95_ge_21_627": bool(effective is not None and effective >= 21.627),
            "no_cuda_or_runner_error": True,
        }
        mechanism_pass = all(gates[name] for name in (
            "registration_48_ranges", "full_destination_zero_mismatches", "validation_open",
            "test_120_finite", "test_p95_le_65ms", "effective_gb_s_at_p95_ge_15",
            "no_cuda_or_runner_error",
        ))
        strong_pass = mechanism_pass and gates["strong_test_p95_le_45ms"] and gates["strong_effective_gb_s_at_p95_ge_21_627"]
        result_payload = {
            "registration_seconds": registration_seconds,
            "registered_bytes": LAYERS * EXPERTS * EXPERT_BYTES,
            "registered_gib": LAYERS * EXPERTS * EXPERT_BYTES / 2**30,
            "memory_after_registration": memory_after_registration,
            "mapped_aliases_nonzero": all(value != 0 for value in aliases),
            "full_destination_mismatch_count": mismatch_count,
            "validation": {"tokens": validation_tokens, "orders": orders, "schedules": validation},
            "selected_blocks": selected_blocks,
            "test": {"tokens": test_tokens if validation_open else [], "raw_ms": raw_test, "stats": test_stats},
            "effective_gb_s_at_p95": effective,
            "page_telemetry": {"available": sampler.error is None, "error": sampler.error, "samples": page_rows},
            "gates": gates,
            "mechanism_pass": mechanism_pass,
            "strong_pass": strong_pass,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            sampler.stop()
        except Exception:
            pass
    finally:
        unregister_failures = unregister_ranges(host_pointers)

    mechanism_pass = bool(result_payload.get("mechanism_pass")) and error is None and not unregister_failures
    strong_pass = bool(result_payload.get("strong_pass")) and mechanism_pass
    result = {
        "kind": "port80b_d3r_mapped_host_kernel",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "strong_transport_pass" if strong_pass else ("mapped_host_mechanism_pass" if mechanism_pass else "mapped_host_kernel_negative"),
        "mechanism_pass": mechanism_pass,
        "strong_pass": strong_pass,
        "full_bank_pass": False,
        "set_device_map_host": map_flag,
        "inputs": {"preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(MANIFEST), "bank_sha256_from_manifest": manifest["bank_sha256"]},
        "protocol": {"experts_per_layer": EXPERTS, "schedules_blocks": list(SCHEDULES), "threads": THREADS, "validation_warmups": VALIDATION_WARMUPS, "validation_rounds": VALIDATION_ROUNDS, "test_rounds": TEST_ROUNDS},
        **result_payload,
        "error": error,
        "unregister_failures": unregister_failures,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "60%-bank mapped-host byte-copy kernel only; no Q5 arithmetic, full-bank capacity, real model, quality, dense shell, tok/s or endurance claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    test_stats = result_payload.get("test", {}).get("stats") if result_payload else None
    REPORT.write_text(
        "# PORT80B-D3R — mapped-host one-kernel replication report\n\n"
        f"Verdict: **{result['status']}**. Selected blocks: {result_payload.get('selected_blocks', '—')}. "
        f"Test p50/p95: {test_stats.get('p50') if test_stats else '—'} / {test_stats.get('p95') if test_stats else '—'} ms. "
        f"Effective p95 bandwidth: {result_payload.get('effective_gb_s_at_p95', '—')} GB/s. "
        f"Full-byte mismatches: {result_payload.get('full_destination_mismatch_count', '—')}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "mechanism_pass": mechanism_pass, "strong_pass": strong_pass, "selected_blocks": result_payload.get("selected_blocks"), "validation": {key: value["stats"] for key, value in result_payload.get("validation", {}).get("schedules", {}).items()}, "test": test_stats, "effective_gb_s_at_p95": result_payload.get("effective_gb_s_at_p95"), "mismatches": result_payload.get("full_destination_mismatch_count"), "error": error, "unregister_failures": unregister_failures}, indent=2))


if __name__ == "__main__":
    main()
