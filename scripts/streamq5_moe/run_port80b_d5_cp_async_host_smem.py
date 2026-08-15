from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_port80b_d2_registered_scatter import (
    EXPECTED_BANK_SHA256, REGISTER_FLAGS, TOKEN_BYTES, VERIFY_SOURCE,
    full_verify, record_offset, routes, stats, unregister_ranges,
)
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import (
    BANK, BANK_BYTES, EXPERT_BYTES, LAYERS, MANIFEST,
)

R = ROOT / "reports/streamq5_moe"
PREREG = R / "PORT80B_D5_CP_ASYNC_HOST_SMEM_PREREGISTRATION.md"
OUTPUT = R / "port80b_d5_cp_async_host_smem.json"
REPORT = R / "PORT80B_D5_CP_ASYNC_HOST_SMEM_REPORT_2026-08-12.md"
EXPERTS = 307
SCHEDULES = (256, 512, 1024, 2048)
THREADS = 256
WARMUPS = 4
VALIDATION_ROUNDS = 16
TEST_ROUNDS = 120
TILE_BYTES = 4096
TILES_PER_RECORD = EXPERT_BYTES // TILE_BYTES


SOURCE = r'''
#include <cuda_pipeline.h>
extern "C" __global__ void host_to_smem_pipeline(
    const unsigned long long* source_pointers,
    unsigned char* destination,
    unsigned long long total_tiles) {
  extern __shared__ unsigned char tile[];
  const unsigned long long record_bytes = 2027520ULL;
  const unsigned long long tile_bytes = 4096ULL;
  const unsigned long long tiles_per_record = 495ULL;
  for (unsigned long long tile_index = blockIdx.x; tile_index < total_tiles; tile_index += gridDim.x) {
    unsigned long long record = tile_index / tiles_per_record;
    unsigned long long tile_in_record = tile_index - record * tiles_per_record;
    const unsigned char* source = (const unsigned char*)source_pointers[record] + tile_in_record * tile_bytes;
    unsigned char* target = destination + record * record_bytes + tile_in_record * tile_bytes;
    for (unsigned int offset = threadIdx.x * 16U; offset < 4096U; offset += blockDim.x * 16U) {
      __pipeline_memcpy_async(tile + offset, source + offset, 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    for (unsigned int offset = threadIdx.x * 16U; offset < 4096U; offset += blockDim.x * 16U) {
      *(uint4*)(target + offset) = *(const uint4*)(tile + offset);
    }
    __syncthreads();
  }
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register(mapped: np.memmap) -> tuple[list[int], list[int]]:
    hosts, aliases = [], []
    size = EXPERTS * EXPERT_BYTES
    try:
        for layer in range(LAYERS):
            host = int(mapped.ctypes.data) + record_offset(layer, 0)
            cp.cuda.runtime.hostRegister(host, size, REGISTER_FLAGS)
            hosts.append(host)
            alias = int(cp.cuda.runtime.pointerGetAttributes(host).devicePointer)
            if not alias:
                raise RuntimeError(f"layer {layer}: null mapped alias")
            aliases.append(alias)
    except Exception:
        unregister_ranges(hosts)
        raise
    return hosts, aliases


def pointer_row(token: int, aliases: list[int]) -> np.ndarray:
    return np.asarray([aliases[layer] + expert * EXPERT_BYTES for layer, expert in routes(token, EXPERTS)], dtype=np.uint64)


def launch(kernel: cp.RawKernel, blocks: int, pointers: cp.ndarray, destination: cp.ndarray, stream: cp.cuda.Stream) -> None:
    kernel((blocks,), (THREADS,), (pointers, destination, np.uint64(480 * TILES_PER_RECORD)), shared_mem=TILE_BYTES, stream=stream)


def timed(kernel: cp.RawKernel, blocks: int, pointers: cp.ndarray, destination: cp.ndarray, stream: cp.cuda.Stream) -> float:
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record(stream)
    launch(kernel, blocks, pointers, destination, stream)
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(begin, end))


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D5 result")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not BANK.is_file() or BANK.stat().st_size != BANK_BYTES or manifest.get("bank_sha256") != EXPECTED_BANK_SHA256:
        raise RuntimeError("immutable bank/manifest contract failed")
    if TILES_PER_RECORD != 495:
        raise RuntimeError("record is not exactly 495 tiles")
    started = time.perf_counter()
    cuda_include = ROOT / ".venv/Lib/site-packages/nvidia/cu13/include"
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    options = ("--std=c++14", f"--include-path={cuda_include}")
    kernel = cp.RawKernel(SOURCE, "host_to_smem_pipeline", options=options)
    verify_kernel = cp.RawKernel(VERIFY_SOURCE, "verify_record_bytes", options=("--std=c++14",))
    destination = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)
    hosts: list[int] = []
    payload: dict[str, object] = {}
    error = None
    unregister_failures: list[str] = []
    try:
        hosts, aliases = register(mapped)
        tokens = [89_999] + list(range(90_000, 90_000 + VALIDATION_ROUNDS)) + list(range(91_000, 91_000 + TEST_ROUNDS))
        table = cp.asarray(np.stack([pointer_row(token, aliases) for token in tokens]))
        rows = {token: index for index, token in enumerate(tokens)}

        launch(kernel, 512, table[rows[89_999]], destination, stream)
        stream.synchronize()
        mismatches = full_verify(verify_kernel, destination, routes(89_999, EXPERTS), stream)
        for blocks in SCHEDULES:
            for warmup in range(WARMUPS):
                token = 90_000 + warmup
                launch(kernel, blocks, table[rows[token]], destination, stream)
        stream.synchronize()

        raw_validation = {str(blocks): [] for blocks in SCHEDULES}
        orders = []
        validation_tokens = list(range(90_000, 90_000 + VALIDATION_ROUNDS))
        for round_index, token in enumerate(validation_tokens):
            rotation = round_index % len(SCHEDULES)
            order = list(SCHEDULES[rotation:] + SCHEDULES[:rotation])
            if round_index & 1:
                order.reverse()
            orders.append(order)
            for blocks in order:
                raw_validation[str(blocks)].append(timed(kernel, blocks, table[rows[token]], destination, stream))
        validation = {key: {"raw_ms": values, "stats": stats(values)} for key, values in raw_validation.items()}
        selected = min(SCHEDULES, key=lambda blocks: (float(validation[str(blocks)]["stats"]["p50"]), blocks))
        validation_open = mismatches == 0 and float(validation[str(selected)]["stats"]["p50"]) <= 65.0
        raw_test = []
        test_tokens = list(range(91_000, 91_000 + TEST_ROUNDS))
        if validation_open:
            for token in test_tokens:
                raw_test.append(timed(kernel, selected, table[rows[token]], destination, stream))
        test_stats = stats(raw_test) if raw_test else None
        effective = TOKEN_BYTES / (float(test_stats["p95"]) / 1000.0) / 1e9 if test_stats else None
        gates = {
            "full_destination_zero_mismatches": mismatches == 0,
            "test_120_finite": len(raw_test) == TEST_ROUNDS and bool(np.isfinite(raw_test).all()),
            "test_p95_le_65ms": bool(test_stats and float(test_stats["p95"]) <= 65.0),
            "effective_gb_s_at_p95_ge_15": bool(effective is not None and effective >= 15.0),
            "strong_test_p95_le_45ms": bool(test_stats and float(test_stats["p95"]) <= 45.0),
            "strong_effective_gb_s_at_p95_ge_21_627": bool(effective is not None and effective >= 21.627),
            "registration_48_ranges": len(hosts) == LAYERS,
            "no_cuda_or_runner_error": True,
        }
        mechanism_pass = all(gates[name] for name in ("full_destination_zero_mismatches", "test_120_finite", "test_p95_le_65ms", "effective_gb_s_at_p95_ge_15", "registration_48_ranges", "no_cuda_or_runner_error"))
        strong_pass = mechanism_pass and gates["strong_test_p95_le_45ms"] and gates["strong_effective_gb_s_at_p95_ge_21_627"]
        payload = {
            "full_destination_mismatch_count": mismatches,
            "validation": {"tokens": validation_tokens, "orders": orders, "schedules": validation},
            "selected_blocks": selected,
            "validation_open": validation_open,
            "test": {"tokens": test_tokens if validation_open else [], "raw_ms": raw_test, "stats": test_stats},
            "effective_gb_s_at_p95": effective,
            "gates": gates,
            "mechanism_pass": mechanism_pass,
            "strong_pass": strong_pass,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            stream.synchronize()
        except Exception:
            pass
        unregister_failures = unregister_ranges(hosts)

    mechanism_pass = bool(payload.get("mechanism_pass")) and error is None and not unregister_failures
    strong_pass = bool(payload.get("strong_pass")) and mechanism_pass
    result = {
        "kind": "port80b_d5_cp_async_host_smem",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "strong_transport_pass" if strong_pass else ("cp_async_mechanism_pass" if mechanism_pass else "cp_async_negative"),
        "mechanism_pass": mechanism_pass,
        "strong_pass": strong_pass,
        "full_bank_pass": False,
        "inputs": {"preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(MANIFEST), "bank_sha256_from_manifest": manifest["bank_sha256"]},
        "protocol": {"experts_per_layer": EXPERTS, "tile_bytes": TILE_BYTES, "tiles_per_record": TILES_PER_RECORD, "total_tiles": 480 * TILES_PER_RECORD, "threads": THREADS, "schedules": list(SCHEDULES), "warmups": WARMUPS, "validation_rounds": VALIDATION_ROUNDS, "test_rounds": TEST_ROUNDS},
        **payload,
        "error": error,
        "unregister_failures": unregister_failures,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "60%-bank cp.async byte pipeline only; no descriptor TMA, Q5 math, full bank, model, quality, tok/s or endurance claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    test_stats = payload.get("test", {}).get("stats") if payload else None
    REPORT.write_text(
        "# PORT80B-D5 — mapped-host cp.async-to-SMEM report\n\n"
        f"Verdict: **{result['status']}**. Selected blocks: {payload.get('selected_blocks', '—')}. "
        f"Test p50/p95: {test_stats.get('p50') if test_stats else '—'} / {test_stats.get('p95') if test_stats else '—'} ms. "
        f"Effective p95 bandwidth: {payload.get('effective_gb_s_at_p95', '—')} GB/s. Mismatches: {payload.get('full_destination_mismatch_count', '—')}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "mechanism_pass": mechanism_pass, "strong_pass": strong_pass, "validation": {key: row["stats"] for key, row in payload.get("validation", {}).get("schedules", {}).items()}, "selected": payload.get("selected_blocks"), "test": test_stats, "effective": payload.get("effective_gb_s_at_p95"), "mismatches": payload.get("full_destination_mismatch_count"), "error": error, "unregister_failures": unregister_failures}, indent=2))


if __name__ == "__main__":
    main()
