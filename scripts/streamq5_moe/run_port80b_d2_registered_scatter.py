from __future__ import annotations

import hashlib
import json
import os
import struct
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
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import (
    BANK,
    BANK_BYTES,
    CODE_BYTES,
    EXPERT_BYTES,
    EXPERTS_WITH_SHARED,
    HEADER_BYTES,
    LAYERS,
    MANIFEST,
    MATRIX_BYTES,
    PADDING_BYTES,
    PROJECTIONS,
    SCALE_BYTES,
    TRACE_SEED,
    HardPageReadSampler,
    q5_header,
    splitmix64,
)

R = ROOT / "reports/streamq5_moe"
PREREG = R / "PORT80B_D2_REGISTERED_SCATTER_PREREGISTRATION.md"
OUTPUT = R / "port80b_d2_registered_scatter.json"
REPORT = R / "PORT80B_D2_REGISTERED_SCATTER_REPORT_2026-08-12.md"
EXPECTED_BANK_SHA256 = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"
TOP_K = 10
TOKEN_BYTES = LAYERS * TOP_K * EXPERT_BYTES
PREFIX_EXPERTS = (307, 358, 410, 512)
WARMUPS = 10
ROUNDS = 120
REGISTER_FLAGS = 0x02 | 0x08  # cudaHostRegisterMapped | cudaHostRegisterReadOnly
MIN_AVAILABLE_AFTER_REGISTER = 2 * 2**30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def attributes() -> dict[str, int | str]:
    runtime = cp.cuda.runtime
    names = (
        "cudaDevAttrCanMapHostMemory",
        "cudaDevAttrHostRegisterSupported",
        "cudaDevAttrHostRegisterReadOnlySupported",
        "cudaDevAttrCanUseHostPointerForRegisteredMem",
        "cudaDevAttrPageableMemoryAccessUsesHostPageTables",
    )
    props = runtime.getDeviceProperties(0)
    return {
        "name": props["name"].decode("utf-8"),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "integrated": int(props["integrated"]),
        "unified_addressing": int(props["unifiedAddressing"]),
        "async_engine_count": int(props["asyncEngineCount"]),
        **{name: int(runtime.deviceGetAttribute(getattr(runtime, name), 0)) for name in names},
    }


def routes(token: int, experts: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for layer in range(LAYERS):
        state = (TRACE_SEED ^ (token * 0xD6E8FEB86659FD93) ^ (layer * 0xA5A3564E27F8862D)) & ((1 << 64) - 1)
        layer_values: list[int] = []
        while len(layer_values) < TOP_K:
            state = splitmix64(state)
            value = int(state % experts)
            if value not in layer_values:
                layer_values.append(value)
        selected.extend((layer, expert) for expert in layer_values)
    return selected


def record_offset(layer: int, expert: int) -> int:
    return (layer * EXPERTS_WITH_SHARED + expert) * EXPERT_BYTES


def header_reference(selected: list[tuple[int, int]]) -> np.ndarray:
    result = bytearray(len(selected) * len(PROJECTIONS) * HEADER_BYTES)
    import zlib
    codes = bytes([0x55]) * CODE_BYTES
    scales = struct.pack("<H", 0x3C00) * (SCALE_BYTES // 2)
    crc = zlib.crc32(scales, zlib.crc32(codes)) & 0xFFFFFFFF
    for record_index, (layer, expert) in enumerate(selected):
        for projection_index, (projection, rows, columns) in enumerate(PROJECTIONS):
            # CRC is invariant because every physical synthetic payload is identical.
            value = q5_header(layer, expert, projection, rows, columns, crc)
            begin = (record_index * len(PROJECTIONS) + projection_index) * HEADER_BYTES
            result[begin:begin + HEADER_BYTES] = value
    return np.frombuffer(result, dtype=np.uint8)


VERIFY_SOURCE = r'''
extern "C" __global__ void verify_record_bytes(
    const unsigned char* data, const unsigned char* headers,
    unsigned long long total, unsigned long long* mismatches) {
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  const unsigned long long expert_bytes = 2027520ULL;
  const unsigned long long matrix_bytes = 675840ULL;
  const unsigned long long header_bytes = 64ULL;
  const unsigned long long code_bytes = 655360ULL;
  const unsigned long long scale_bytes = 16384ULL;
  unsigned long long local_errors = 0;
  for (; index < total; index += stride) {
    unsigned long long record = index / expert_bytes;
    unsigned long long within = index - record * expert_bytes;
    unsigned long long matrix = within / matrix_bytes;
    unsigned long long pos = within - matrix * matrix_bytes;
    unsigned char expected;
    if (pos < header_bytes) {
      expected = headers[(record * 3ULL + matrix) * header_bytes + pos];
    } else if (pos < header_bytes + code_bytes) {
      expected = 0x55;
    } else if (pos < header_bytes + code_bytes + scale_bytes) {
      expected = ((pos - header_bytes - code_bytes) & 1ULL) ? 0x3c : 0x00;
    } else {
      expected = 0x00;
    }
    local_errors += (unsigned long long)(data[index] != expected);
  }
  if (local_errors) atomicAdd(mismatches, local_errors);
}

extern "C" __global__ void mapped_copy_probe(
    const unsigned char* source, unsigned char* destination, unsigned long long size) {
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  for (; index < size; index += stride) destination[index] = source[index];
}
'''


def register_ranges(mapped: np.memmap, experts: int) -> list[int]:
    pointers: list[int] = []
    size = experts * EXPERT_BYTES
    try:
        for layer in range(LAYERS):
            pointer = int(mapped.ctypes.data) + record_offset(layer, 0)
            cp.cuda.runtime.hostRegister(pointer, size, REGISTER_FLAGS)
            pointers.append(pointer)
    except Exception:
        for pointer in reversed(pointers):
            cp.cuda.runtime.hostUnregister(pointer)
        raise
    return pointers


def unregister_ranges(pointers: list[int]) -> list[str]:
    failures = []
    for pointer in reversed(pointers):
        try:
            cp.cuda.runtime.hostUnregister(pointer)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    return failures


def copy_selected(mapped: np.memmap, destination: cp.ndarray, selected: list[tuple[int, int]], stream: cp.cuda.Stream) -> None:
    for record_index, (layer, expert) in enumerate(selected):
        cp.cuda.runtime.memcpyAsync(
            int(destination.data.ptr) + record_index * EXPERT_BYTES,
            int(mapped.ctypes.data) + record_offset(layer, expert),
            EXPERT_BYTES,
            cp.cuda.runtime.memcpyHostToDevice,
            stream.ptr,
        )


def timed_copy(mapped: np.memmap, destination: cp.ndarray, selected: list[tuple[int, int]], stream: cp.cuda.Stream) -> float:
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record(stream)
    copy_selected(mapped, destination, selected, stream)
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(begin, end))


def full_verify(kernel: cp.RawKernel, destination: cp.ndarray, selected: list[tuple[int, int]], stream: cp.cuda.Stream) -> int:
    headers = cp.asarray(header_reference(selected))
    mismatches = cp.zeros(1, dtype=cp.uint64)
    kernel((4096,), (256,), (destination, headers, np.uint64(TOKEN_BYTES), mismatches), stream=stream)
    stream.synchronize()
    return int(mismatches.get()[0])


def small_mapped_probe(mapped: np.memmap, kernel: cp.RawKernel, stream: cp.cuda.Stream) -> dict[str, object]:
    size = 64 * 2**20
    pointer = int(mapped.ctypes.data)
    cp.cuda.runtime.hostRegister(pointer, size, REGISTER_FLAGS)
    try:
        attrs = cp.cuda.runtime.pointerGetAttributes(pointer)
        device_pointer = int(attrs.devicePointer)
        destination = cp.empty(size, dtype=cp.uint8)
        kernel((1024,), (256,), (np.uint64(device_pointer), destination, np.uint64(size)), stream=stream)
        stream.synchronize()
        source_head = bytes(memoryview(mapped)[:4096])
        source_tail = bytes(memoryview(mapped)[size - 4096:size])
        destination_head = bytes(cp.asnumpy(destination[:4096]))
        destination_tail = bytes(cp.asnumpy(destination[-4096:]))
        return {
            "registered_bytes": size,
            "device_pointer_nonzero": device_pointer != 0,
            "pointer_attributes": {
                "device": int(attrs.device),
                "devicePointer": device_pointer,
                "hostPointer": int(attrs.hostPointer),
                "type": int(attrs.type),
            },
            "edge_bytes_equal": source_head == destination_head and source_tail == destination_tail,
        }
    finally:
        cp.cuda.runtime.hostUnregister(pointer)


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D2 result")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not BANK.is_file() or BANK.stat().st_size != BANK_BYTES:
        raise RuntimeError("locked P0 bank missing")
    if manifest.get("bank_sha256") != EXPECTED_BANK_SHA256:
        raise RuntimeError("bank manifest SHA contract mismatch")
    if TOKEN_BYTES != 973_209_600 or HEADER_BYTES != 64:
        raise RuntimeError("frozen byte geometry mismatch")

    started = time.perf_counter()
    capability = attributes()
    # This flag must be requested before the process creates its primary context.
    try:
        cp.cuda.runtime.setDeviceFlags(0x08)  # cudaDeviceMapHost
        set_device_flags = {"success": True}
    except Exception as exc:
        set_device_flags = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    cp.cuda.Device(0).use()
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    if mapped.flags.writeable:
        raise RuntimeError("bank must remain read-only")
    stream = cp.cuda.Stream(non_blocking=True)
    verify_kernel = cp.RawKernel(VERIFY_SOURCE, "verify_record_bytes", options=("--std=c++14",))
    mapped_kernel = cp.RawKernel(VERIFY_SOURCE, "mapped_copy_probe", options=("--std=c++14",))
    small_probe: dict[str, object]
    try:
        small_probe = small_mapped_probe(mapped, mapped_kernel, stream)
    except Exception as exc:
        small_probe = {"registered_bytes": 64 * 2**20, "success": False, "error": f"{type(exc).__name__}: {exc}"}

    destination = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
    sweep = []
    expansion_stopped = False
    for experts in PREFIX_EXPERTS:
        fraction = experts / 512.0
        registered_bytes = LAYERS * experts * EXPERT_BYTES
        row: dict[str, object] = {
            "experts_per_layer": experts,
            "fraction": fraction,
            "registered_bytes": registered_bytes,
            "registered_gib": registered_bytes / 2**30,
            "system_before": dict(psutil.virtual_memory()._asdict()),
        }
        pointers: list[int] = []
        sampler = HardPageReadSampler()
        try:
            register_started = time.perf_counter()
            pointers = register_ranges(mapped, experts)
            row["registration_seconds"] = time.perf_counter() - register_started
            row["registered_ranges"] = len(pointers)
            row["system_after_registration"] = dict(psutil.virtual_memory()._asdict())
            if int(psutil.virtual_memory().available) < MIN_AVAILABLE_AFTER_REGISTER:
                row["status"] = "safety_stop_low_available_memory"
                expansion_stopped = True
                sweep.append(row)
                break

            selected = routes(20_000 + experts, experts)
            copy_selected(mapped, destination, selected, stream)
            stream.synchronize()
            mismatch_count = full_verify(verify_kernel, destination, selected, stream)
            row["full_destination_mismatch_count"] = mismatch_count
            for warmup in range(WARMUPS):
                copy_selected(mapped, destination, routes(30_000 + warmup, experts), stream)
            stream.synchronize()

            sampler.start()
            raw_ms = []
            for round_index in range(ROUNDS):
                raw_ms.append(timed_copy(mapped, destination, routes(40_000 + round_index, experts), stream))
            sampler.stop()
            timing = stats(raw_ms)
            page_reads = [float(value["page_reads_per_sec"]) for value in sampler.samples]
            bandwidth = TOKEN_BYTES / (float(timing["p95"]) / 1000.0) / 1e9
            row.update({
                "status": "timed",
                "raw_ms": raw_ms,
                "timing": timing,
                "effective_gb_s_at_p95": bandwidth,
                "page_telemetry": {"available": sampler.error is None, "error": sampler.error, "samples": sampler.samples, "page_reads_max": max(page_reads) if page_reads else None},
            })
            row["gates"] = {
                "registration_48_ranges": len(pointers) == LAYERS,
                "full_destination_zero_mismatches": mismatch_count == 0,
                "samples_120_finite": len(raw_ms) == ROUNDS and bool(np.isfinite(raw_ms).all()),
                "p95_le_45ms": float(timing["p95"]) <= 45.0,
                "effective_gb_s_at_p95_ge_21_627": bandwidth >= 21.627,
                "page_reads_zero_when_available": sampler.error is not None or (bool(page_reads) and max(page_reads) == 0.0),
                "no_cuda_or_runner_error": True,
            }
            row["mechanism_pass"] = all(row["gates"][name] for name in (
                "registration_48_ranges", "full_destination_zero_mismatches", "samples_120_finite",
                "p95_le_45ms", "effective_gb_s_at_p95_ge_21_627", "no_cuda_or_runner_error",
            ))
            row["all_gates_pass"] = all(row["gates"].values())
        except Exception as exc:
            try:
                sampler.stop()
            except Exception:
                pass
            row.update({"status": "registration_or_timing_failed", "error": f"{type(exc).__name__}: {exc}", "mechanism_pass": False, "all_gates_pass": False})
            expansion_stopped = True
        finally:
            row["unregister_failures"] = unregister_ranges(pointers)
            row["system_after_unregister"] = dict(psutil.virtual_memory()._asdict())
        if not sweep or sweep[-1] is not row:
            sweep.append(row)
        if expansion_stopped:
            break

    full = next((row for row in sweep if row["experts_per_layer"] == 512), None)
    any_mechanism = any(bool(row.get("mechanism_pass")) for row in sweep)
    full_pass = bool(full and full.get("all_gates_pass") and not full.get("unregister_failures"))
    result = {
        "kind": "port80b_d2_registered_scatter",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_bank_pass" if full_pass else ("prefix_mechanism_pass_full_bank_unproven" if any_mechanism else "registered_scatter_negative"),
        "mechanism_pass": any_mechanism,
        "full_bank_pass": full_pass,
        "capability": capability,
        "set_device_map_host": set_device_flags,
        "small_mapped_host_probe": small_probe,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "evaluator_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST),
            "bank_sha256_from_manifest": manifest["bank_sha256"],
        },
        "protocol": {"prefix_experts": list(PREFIX_EXPERTS), "warmups": WARMUPS, "rounds": ROUNDS, "registration_flags": REGISTER_FLAGS},
        "sweep": sweep,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Synthetic registered mmap-to-device scatter transport only; no real 80B model, Q5 math, quality, dense shell, tok/s or endurance claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PORT80B-D2 — registered-bank scatter report",
        "",
        f"Verdict: **{result['status']}**. Mechanism pass: **{any_mechanism}**; full-bank pass: **{full_pass}**.",
        "",
        "| prefix | registered GiB | status | p50 ms | p95 ms | GB/s at p95 | mismatches |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sweep:
        timing = row.get("timing", {})
        lines.append(
            f"| {100 * float(row['fraction']):.1f}% | {float(row['registered_gib']):.3f} | {row['status']} | "
            f"{timing.get('p50', '—')} | {timing.get('p95', '—')} | {row.get('effective_gb_s_at_p95', '—')} | "
            f"{row.get('full_destination_mismatch_count', '—')} |"
        )
    lines.extend(["", "The immutable P0 negative result is unchanged. See the JSON and independent verifier for raw samples and gates.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "mechanism_pass": any_mechanism, "full_bank_pass": full_pass, "small_probe": small_probe, "sweep": [{key: row.get(key) for key in ("experts_per_layer", "registered_gib", "status", "timing", "effective_gb_s_at_p95", "full_destination_mismatch_count", "gates", "error")} for row in sweep]}, indent=2))


if __name__ == "__main__":
    main()
