from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import (
    BANK, BANK_BYTES, EXPERT_BYTES, LAYERS, MANIFEST, route,
)

R = ROOT / "reports/streamq5_moe"
PREREG = R / "PORT80B_D1_TOKEN_BATCH_DIRECTPATH_PREREGISTRATION.md"
OUTPUT = R / "port80b_d1_token_batch_directpath.json"
REPORT = R / "PORT80B_D1_TOKEN_BATCH_DIRECTPATH_REPORT_2026-08-12.md"
TOKEN = 10_000
EXPERTS_PER_TOKEN = LAYERS * 10
TOKEN_BYTES = EXPERTS_PER_TOKEN * EXPERT_BYTES
ARMS = ("record480", "layer48", "token1")
WARMUPS = 10
ROUNDS = 120
STAGE_TOKENS = tuple(range(10_001, 10_033))
EXPECTED_BANK_SHA256 = "4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)), "min": float(array.min()), "max": float(array.max()),
    }


def offsets_for(token: int) -> list[int]:
    return [((layer * 513 + expert) * EXPERT_BYTES) for layer in range(LAYERS) for expert in route(token, layer)]


def gather(mapped: np.memmap, target: np.ndarray, token: int) -> None:
    cursor = 0
    for source in offsets_for(token):
        np.copyto(target[cursor:cursor + EXPERT_BYTES], mapped[source:source + EXPERT_BYTES])
        cursor += EXPERT_BYTES
    if cursor != TOKEN_BYTES:
        raise RuntimeError("token gather byte contract failed")


def launch_copy(name: str, source_ptr: int, destination_ptr: int, stream: cp.cuda.Stream) -> None:
    chunks = {"record480": 480, "layer48": 48, "token1": 1}[name]
    chunk_bytes = TOKEN_BYTES // chunks
    if chunk_bytes * chunks != TOKEN_BYTES:
        raise RuntimeError("copy chunk divisibility failed")
    for index in range(chunks):
        offset = index * chunk_bytes
        cp.cuda.runtime.memcpyAsync(
            destination_ptr + offset, source_ptr + offset, chunk_bytes,
            cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
        )


def timed_copy(name: str, source_ptr: int, destination_ptr: int, stream: cp.cuda.Stream) -> float:
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record(stream); launch_copy(name, source_ptr, destination_ptr, stream); end.record(stream); end.synchronize()
    return float(cp.cuda.get_elapsed_time(begin, end))


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D1 output")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not BANK.is_file() or BANK.stat().st_size != BANK_BYTES or manifest["bank_sha256"] != EXPECTED_BANK_SHA256:
        raise RuntimeError("locked physical P0 bank missing or provenance mismatch")
    if TOKEN_BYTES != 973_209_600:
        raise RuntimeError(TOKEN_BYTES)

    started = time.perf_counter()
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    if mapped.flags.writeable:
        raise RuntimeError("mapping is not read-only")
    pinned = cp.cuda.alloc_pinned_memory(TOKEN_BYTES)
    pinned_array = np.frombuffer(pinned, dtype=np.uint8, count=TOKEN_BYTES)
    gather(mapped, pinned_array, TOKEN)
    pinned_sha = hashlib.sha256(memoryview(pinned_array)).hexdigest()

    reference = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
    destination = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)
    launch_copy("token1", pinned.ptr, reference.data.ptr, stream); stream.synchronize()
    correctness = {}
    for name in ARMS:
        destination.fill(0)
        launch_copy(name, pinned.ptr, destination.data.ptr, stream); stream.synchronize()
        equal = bool(cp.array_equal(reference, destination).item())
        correctness[name] = {"byte_equal": equal, "bytes": TOKEN_BYTES}
    if not all(row["byte_equal"] for row in correctness.values()):
        raise RuntimeError("full-buffer correctness failed")

    for name in ARMS:
        for _ in range(WARMUPS):
            launch_copy(name, pinned.ptr, destination.data.ptr, stream)
        stream.synchronize()
    raw = {name: [] for name in ARMS}
    orders = []
    for round_index in range(ROUNDS):
        rotation = round_index % len(ARMS)
        order = list(ARMS[rotation:] + ARMS[:rotation])
        if round_index & 1:
            order.reverse()
        orders.append(order)
        for name in order:
            raw[name].append(timed_copy(name, pinned.ptr, destination.data.ptr, stream))

    # Warm exactly the pages that the measured page-resident staging pass uses.
    for token in STAGE_TOKENS:
        gather(mapped, pinned_array, token)
    stage_ms = []
    stage_hash = hashlib.sha256()
    for token in STAGE_TOKENS:
        begin = time.perf_counter(); gather(mapped, pinned_array, token); stage_ms.append((time.perf_counter() - begin) * 1000.0)
        stage_hash.update(memoryview(pinned_array)[:4096]); stage_hash.update(memoryview(pinned_array)[-4096:])

    timing = {name: {"raw_ms": raw[name], "stats": stats(raw[name])} for name in ARMS}
    stage = {"tokens": list(STAGE_TOKENS), "raw_ms": stage_ms, "stats": stats(stage_ms), "edge_digest": stage_hash.hexdigest()}
    token_stats, record_stats = timing["token1"]["stats"], timing["record480"]["stats"]
    ratios = {
        "token1_over_record480_p50": token_stats["p50"] / record_stats["p50"],
        "token1_over_record480_p95": token_stats["p95"] / record_stats["p95"],
    }
    ideal_overlap_p95 = max(stage["stats"]["p95"], token_stats["p95"])
    serial_p95 = stage["stats"]["p95"] + token_stats["p95"]
    gates = {
        "all_full_buffers_byte_equal": all(row["byte_equal"] for row in correctness.values()),
        "all_arms_120_finite_samples": all(len(raw[name]) == ROUNDS and np.isfinite(raw[name]).all() for name in ARMS),
        "token1_p95_le_45ms": token_stats["p95"] <= 45.0,
        "token1_p50_ratio_le_0_80": ratios["token1_over_record480_p50"] <= 0.80,
        "token1_p95_ratio_le_0_90": ratios["token1_over_record480_p95"] <= 0.90,
        "ideal_overlap_p95_le_45ms": ideal_overlap_p95 <= 45.0,
    }
    component_pass = all(gates[name] for name in ("all_full_buffers_byte_equal", "all_arms_120_finite_samples", "token1_p95_le_45ms", "token1_p50_ratio_le_0_80", "token1_p95_ratio_le_0_90"))
    feasibility_pass = component_pass and gates["ideal_overlap_p95_le_45ms"]
    result = {
        "kind": "port80b_d1_token_batch_directpath", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "directpath_feasibility_pass" if feasibility_pass else ("h2d_component_pass_staging_closed" if component_pass else "directpath_closed"),
        "component_pass": component_pass, "overall_pass": feasibility_pass,
        "inputs": {"preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)), "bank_manifest_sha256": sha256(MANIFEST), "bank_sha256_from_verified_manifest": manifest["bank_sha256"], "source_token": TOKEN, "source_pinned_sha256": pinned_sha},
        "physical": {"bank_bytes": BANK_BYTES, "token_bytes": TOKEN_BYTES, "records_per_token": EXPERTS_PER_TOKEN, "pinned_bytes": TOKEN_BYTES, "device_bytes": 2 * TOKEN_BYTES, "pid": os.getpid()},
        "correctness": correctness, "protocol": {"warmups_per_arm": WARMUPS, "rounds": ROUNDS, "orders": orders},
        "timing": timing, "staging": stage, "ratios": ratios,
        "projection": {"ideal_overlap_p95_ms": ideal_overlap_p95, "fully_serial_p95_ms": serial_p95},
        "gates": gates, "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Resident pinned-buffer H2D dispatch decomposition plus page-resident mmap gather; no actual overlap, model compute, quality, routing, dense shell or end-to-end tok/s.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# PORT80B-D1 — token-batch direct-path report\n\n"
        f"Verdict: **{result['status']}**. token1 p95 {token_stats['p95']:.3f} ms; "
        f"record480 p95 {record_stats['p95']:.3f} ms; staging p95 {stage['stats']['p95']:.3f} ms; "
        f"ideal-overlapgrens {ideal_overlap_p95:.3f} ms.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "component_pass": component_pass, "overall_pass": feasibility_pass, "timing": {name: timing[name]["stats"] for name in ARMS}, "staging": stage["stats"], "ratios": ratios, "projection": result["projection"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
