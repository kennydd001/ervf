from __future__ import annotations

import argparse
from collections import OrderedDict
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading
import time
import zlib

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUN_DIR = ROOT / "reports/runs/streamq5_moe/port80b_p0"
PREREG = REPORTS / "PORT80B_P0_PHYSICAL_HOST_BANK_PREREGISTRATION.md"
REGISTRY = REPORTS / "PORT80B_P0_ACTIVE_SET_REGISTRY_2026-08-12.yaml"
N4A = REPORTS / "n4a_synthetic_80b_shape_capacity.json"
N4BR = REPORTS / "n4br_synthetic_80b_exact_replication.json"
CAPACITY = ROOT / "info/BREAKTHROUGH_NEXT_PHASE_PACK_2026-08-12/PORT80B_CAPACITY_AND_LATENCY.json"
PREFLIGHT = REPORTS / "port80b_p0_preflight_dry_run.json"
BANK = RUN_DIR / "port80b_p0_full_q5_bank.bin"
MANIFEST = RUN_DIR / "port80b_p0_full_q5_bank_manifest.json"
RESULT = REPORTS / "port80b_p0_physical_host_bank_result.json"

ACKNOWLEDGEMENT = "PORT80B_P0_49925652480"
LAYERS = 48
ROUTED_EXPERTS = 512
EXPERTS_WITH_SHARED = 513
TOP_K = 10
PROJECTIONS = ((0, 512, 2048), (1, 512, 2048), (2, 2048, 512))
GROUP = 128
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
ALIGNMENT = 4096
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
PADDING_BYTES = 4_032
MATRIX_BYTES = 675_840
EXPERT_BYTES = 2_027_520
BANK_BYTES = 49_925_652_480
MATRIX_RECORDS = LAYERS * EXPERTS_WITH_SHARED * 3
EXPERT_RECORDS = LAYERS * EXPERTS_WITH_SHARED
PINNED_WINDOWS = 8
PINNED_BYTES = PINNED_WINDOWS * EXPERT_BYTES
CACHE_SLOTS = {"cache_4k": 2_420, "cache_32k": 2_072}
TOKENS_PER_SCENARIO = 10_000
STABILITY_SECONDS = 3_600
COMMIT_LIMIT_BYTES = 58 * 2**30
H2D_P95_LIMIT_MS = 45.0
TRACE_SEED = 0x80B0120826
MASK64 = (1 << 64) - 1


def sha256(path: Path, chunk_bytes: int = 64 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timing_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def process_memory(process: psutil.Process) -> dict[str, int]:
    value = process.memory_info()
    return {
        "rss": int(value.rss),
        "peak_wset": int(getattr(value, "peak_wset", value.rss)),
        "pagefile": int(getattr(value, "pagefile", 0)),
        "peak_pagefile": int(getattr(value, "peak_pagefile", 0)),
        "private": int(getattr(value, "private", 0)),
        "num_page_faults_hard_plus_soft": int(getattr(value, "num_page_faults", 0)),
    }


def system_memory() -> dict[str, int]:
    value = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_physical": int(value.total),
        "available_physical": int(value.available),
        "used_physical": int(value.used),
        "swap_total": int(swap.total),
        "swap_used": int(swap.used),
    }


def disk_contract(path: Path) -> dict[str, int | bool]:
    if os.name != "nt":
        raise RuntimeError("PORT80B_P0 physical allocation verification requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetFileAttributesW.restype = wintypes.DWORD
    kernel32.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetCompressedFileSizeW.restype = wintypes.DWORD
    attributes = int(kernel32.GetFileAttributesW(str(path)))
    if attributes == 0xFFFFFFFF:
        raise ctypes.WinError(ctypes.get_last_error())
    high = wintypes.DWORD()
    ctypes.set_last_error(0)
    low = int(kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high)))
    error = ctypes.get_last_error()
    if low == 0xFFFFFFFF and error:
        raise ctypes.WinError(error)
    allocated = (int(high.value) << 32) | low
    sparse = bool(attributes & 0x00000200)
    compressed = bool(attributes & 0x00000800)
    return {
        "logical_bytes": int(path.stat().st_size),
        "allocated_bytes_getcompressedfilesize": allocated,
        "file_attributes": attributes,
        "sparse_attribute": sparse,
        "compressed_attribute": compressed,
        "non_sparse_fully_allocated": (
            path.stat().st_size == BANK_BYTES and allocated >= BANK_BYTES and not sparse and not compressed
        ),
    }


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def route(token: int, layer: int) -> tuple[int, ...]:
    counter = (TRACE_SEED ^ (token * 0xD6E8FEB86659FD93) ^ (layer * 0xA5A3564E27F8862D)) & MASK64
    first = splitmix64(counter)
    second = splitmix64(first)
    start = first & 511
    stride = ((second & 255) << 1) | 1
    result = tuple(int((start + rank * stride) & 511) for rank in range(TOP_K))
    if len(set(result)) != TOP_K:
        raise AssertionError("top-10 generator produced duplicates")
    return result


def expert_offset(layer: int, expert: int) -> int:
    return (layer * EXPERTS_WITH_SHARED + expert) * EXPERT_BYTES


def record_offset(layer: int, expert: int, projection: int) -> int:
    return expert_offset(layer, expert) + projection * MATRIX_BYTES


def gpu_telemetry() -> dict[str, float | str]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        ).strip().split(",")
        return {
            "temperature_c": float(raw[0]),
            "power_w": float(raw[1]),
            "sm_clock_mhz": float(raw[2]),
            "memory_used_mib": float(raw[3]),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


class _PDHValueUnion(ctypes.Union):
    _fields_ = [("long_value", wintypes.LONG), ("double_value", ctypes.c_double), ("large_value", ctypes.c_longlong)]


class _PDHValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("status", wintypes.DWORD), ("value", _PDHValueUnion)]


class HardPageReadSampler:
    """System-wide physical page-in counters; intentionally conservative."""

    def __init__(self) -> None:
        self.samples: list[dict[str, float | str]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="port80b-pdh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        if os.name != "nt":
            self.error = "PDH hard-page-read sampling requires Windows"
            return
        query = ctypes.c_void_p()
        page_reads = ctypes.c_void_p()
        pages_input = ctypes.c_void_p()
        try:
            pdh = ctypes.WinDLL("pdh", use_last_error=True)
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            pdh.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(_PDHValue)]
            pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)):
                raise RuntimeError("PdhOpenQueryW failed")
            for counter, name in ((page_reads, r"\Memory\Page Reads/sec"), (pages_input, r"\Memory\Pages Input/sec")):
                if pdh.PdhAddEnglishCounterW(query, name, None, ctypes.byref(counter)):
                    raise RuntimeError(f"PdhAddEnglishCounterW failed for {name}")
            if pdh.PdhCollectQueryData(query):
                raise RuntimeError("initial PdhCollectQueryData failed")
            while not self._stop.wait(1.0):
                if pdh.PdhCollectQueryData(query):
                    raise RuntimeError("PdhCollectQueryData failed")
                row: dict[str, float | str] = {"utc": utc_now(), "monotonic_seconds": time.perf_counter()}
                for counter, key in ((page_reads, "page_reads_per_sec"), (pages_input, "pages_input_per_sec")):
                    value = _PDHValue()
                    if pdh.PdhGetFormattedCounterValue(counter, 0x00000200, None, ctypes.byref(value)) or value.status:
                        raise RuntimeError(f"PdhGetFormattedCounterValue failed for {key}")
                    row[key] = float(value.double_value)
                self.samples.append(row)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if query:
                try:
                    ctypes.WinDLL("pdh").PdhCloseQuery(query)
                except Exception:
                    pass


def expected_contract() -> dict[str, int | float]:
    return {
        "layers": LAYERS,
        "routed_experts_per_layer": ROUTED_EXPERTS,
        "shared_experts_per_layer": 1,
        "expert_records": EXPERT_RECORDS,
        "matrix_records": MATRIX_RECORDS,
        "header_bytes": HEADER_BYTES,
        "code_bytes_per_matrix": CODE_BYTES,
        "scale_bytes_per_matrix": SCALE_BYTES,
        "padding_bytes_per_matrix": PADDING_BYTES,
        "matrix_bytes": MATRIX_BYTES,
        "expert_bytes": EXPERT_BYTES,
        "bank_bytes": BANK_BYTES,
        "bank_gib": BANK_BYTES / 2**30,
        "pinned_windows": PINNED_WINDOWS,
        "pinned_bytes": PINNED_BYTES,
        "cache_4k_slots": CACHE_SLOTS["cache_4k"],
        "cache_4k_bytes": CACHE_SLOTS["cache_4k"] * EXPERT_BYTES,
        "cache_32k_slots": CACHE_SLOTS["cache_32k"],
        "cache_32k_bytes": CACHE_SLOTS["cache_32k"] * EXPERT_BYTES,
    }


def validate_constant_math() -> None:
    assert HEADER_BYTES == 64
    assert HEADER_BYTES + CODE_BYTES + SCALE_BYTES + PADDING_BYTES == MATRIX_BYTES
    assert MATRIX_BYTES % ALIGNMENT == 0
    assert EXPERT_BYTES == 3 * MATRIX_BYTES
    assert EXPERT_RECORDS == 24_624
    assert MATRIX_RECORDS == 73_872
    assert BANK_BYTES == EXPERT_RECORDS * EXPERT_BYTES
    n4a = json.loads(N4A.read_text(encoding="utf-8"))
    n4br = json.loads(N4BR.read_text(encoding="utf-8"))
    capacity = json.loads(CAPACITY.read_text(encoding="utf-8"))
    assert n4a["expert_accounting"]["full_q5_bank_bytes"] == BANK_BYTES
    assert n4a["q5_record_contract"]["expert_record_bytes"] == EXPERT_BYTES
    assert n4br["physical"]["q5_bank_bytes"] // (
        n4br["physical"]["layers"] * n4br["physical"]["active_experts"]
    ) == EXPERT_BYTES
    assert capacity["locked_projection"]["custom_q5_bank_gib"] == 46.497


def preflight() -> dict[str, object]:
    validate_constant_math()
    source_text = Path(__file__).read_text(encoding="utf-8")
    compile(source_text, str(Path(__file__)), "exec")
    pdh_smoke = HardPageReadSampler()
    pdh_smoke.start()
    time.sleep(2.2)
    pdh_smoke.stop()
    disk = shutil.disk_usage(ROOT)
    memory = system_memory()
    contract = expected_contract()
    disk_after = disk.free - BANK_BYTES
    result: dict[str, object] = {
        "kind": "port80b_p0_preflight_dry_run",
        "completed_utc": utc_now(),
        "status": "safe_dry_run_complete_waiting_for_timing_go",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "registry_sha256": sha256(REGISTRY),
            "n4a_sha256": sha256(N4A),
            "n4br_sha256": sha256(N4BR),
            "capacity_sha256": sha256(CAPACITY),
        },
        "compile_and_contract": {
            "compile_ok": True,
            "constant_math_matches_n4a_n4br": True,
            "windows": os.name == "nt",
            "numpy_version": np.__version__,
            "psutil_version": psutil.__version__,
            "cupy_discoverable_without_import": importlib.util.find_spec("cupy") is not None,
            "nvidia_smi_discoverable_without_execution": shutil.which("nvidia-smi") is not None,
            "pdh_available": os.name == "nt",
            "pdh_smoke_error": pdh_smoke.error,
            "pdh_smoke_samples": pdh_smoke.samples,
            "wpr_available_for_optional_raw_trace": shutil.which("wpr") is not None,
        },
        "physical_contract": contract,
        "disk_impact": {
            "volume_total_bytes": disk.total,
            "volume_free_before_bytes": disk.free,
            "bank_logical_and_minimum_allocated_bytes": BANK_BYTES,
            "free_after_build_bytes": disk_after,
            "free_after_build_gib": disk_after / 2**30,
            "sufficient_for_bank_plus_100gib_reserve": disk_after >= 100 * 2**30,
            "bank_preexists": BANK.exists(),
            "partial_preexists": BANK.with_suffix(BANK.suffix + ".inprogress").exists(),
        },
        "ram_and_device_impact": {
            **memory,
            "readonly_mapping_virtual_address_bytes": BANK_BYTES,
            "maximum_clean_file_working_set_bytes": BANK_BYTES,
            "eight_pinned_host_windows_bytes": PINNED_BYTES,
            "physical_ram_left_if_entire_bank_and_pinned_resident_bytes": memory["total_physical"] - BANK_BYTES - PINNED_BYTES,
            "current_available_minus_bank_and_pinned_bytes": memory["available_physical"] - BANK_BYTES - PINNED_BYTES,
            "zero_cache_device_ring_bytes": PINNED_BYTES,
            "cache_4k_device_bytes": CACHE_SLOTS["cache_4k"] * EXPERT_BYTES,
            "cache_32k_device_bytes": CACHE_SLOTS["cache_32k"] * EXPERT_BYTES,
            "reference_host_accounted_plus_reserve_bytes_from_n4a": json.loads(N4A.read_text(encoding="utf-8"))["host_budget"]["accounted_plus_reserve_bytes"],
            "process_commit_gate_bytes": COMMIT_LIMIT_BYTES,
        },
        "planned_work": {
            "primary_tokens": 3 * TOKENS_PER_SCENARIO,
            "zero_cache_tokens": TOKENS_PER_SCENARIO,
            "cache_4k_tokens": TOKENS_PER_SCENARIO,
            "cache_32k_tokens": TOKENS_PER_SCENARIO,
            "minimum_uninterrupted_benchmark_seconds": STABILITY_SECONDS,
            "zero_cache_bytes_per_token": LAYERS * TOP_K * EXPERT_BYTES,
            "zero_cache_primary_transfer_bytes": TOKENS_PER_SCENARIO * LAYERS * TOP_K * EXPERT_BYTES,
            "hard_fault_measurement": "system-wide PDH Page Reads/sec + Pages Input/sec; process total page faults logged separately",
        },
        "safety": {
            "physical_file_created_by_this_run": False,
            "cupy_imported_by_this_run": False,
            "pinned_or_device_memory_allocated_by_this_run": False,
            "gpu_work_executed_by_this_run": False,
            "required_acknowledgement_for_build_or_benchmark": ACKNOWLEDGEMENT,
            "timing_go_received": False,
        },
        "next_action": "report exact impact and wait for a separate timing-go; do not build or use CUDA yet",
        "claim_boundary": "Compile/safety/disk/RAM dry-run only; no physical bank, CUDA work, performance or 80B claim.",
    }
    PREFLIGHT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def q5_tail() -> tuple[bytes, int]:
    codes = bytes([0x55]) * CODE_BYTES
    scales = struct.pack("<H", 0x3C00) * (SCALE_BYTES // 2)
    crc = zlib.crc32(scales, zlib.crc32(codes)) & 0xFFFFFFFF
    return codes + scales + bytes(PADDING_BYTES), crc


def q5_header(layer: int, expert: int, projection: int, rows: int, columns: int, crc: int) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        b"SQ5M", 1, layer, expert, projection, 5, rows, columns, GROUP,
        CODE_BYTES, SCALE_BYTES, crc, bytes(28),
    )


def build_bank() -> dict[str, object]:
    validate_constant_math()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    partial = BANK.with_suffix(BANK.suffix + ".inprogress")
    if BANK.exists() or MANIFEST.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite an existing PORT80B_P0 bank, manifest or partial file")
    free_before = shutil.disk_usage(ROOT).free
    if free_before - BANK_BYTES < 100 * 2**30:
        raise RuntimeError("build safety gate requires at least 100 GiB free after the final bank")
    tail, crc = q5_tail()
    digest = hashlib.sha256()
    started = time.perf_counter()
    with partial.open("xb", buffering=8 * 2**20) as output:
        for layer in range(LAYERS):
            for expert in range(EXPERTS_WITH_SHARED):
                for projection, rows, columns in PROJECTIONS:
                    header = q5_header(layer, expert, projection, rows, columns, crc)
                    output.write(header)
                    output.write(tail)
                    digest.update(header)
                    digest.update(tail)
            output.flush()
            print(json.dumps({"built_layers": layer + 1, "bytes": output.tell()}), flush=True)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != BANK_BYTES:
        raise RuntimeError(f"built size {partial.stat().st_size} != {BANK_BYTES}")
    allocation = disk_contract(partial)
    if not allocation["non_sparse_fully_allocated"]:
        raise RuntimeError(f"physical allocation contract failed: {allocation}")
    os.replace(partial, BANK)
    manifest = {
        "kind": "port80b_p0_full_q5_bank_manifest",
        "completed_utc": utc_now(),
        "build_seconds": time.perf_counter() - started,
        "bank": str(BANK),
        "bank_sha256": digest.hexdigest(),
        "contract": expected_contract(),
        "allocation": disk_contract(BANK),
        "payload": {"codes": "0x55", "bf16_scale_word": "0x3c00", "payload_crc32": crc, "padding": "zero"},
        "order": "layer-major; expert 0..511 routed then 512 shared; gate/up/down",
        "inputs": {"preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__))},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def sample_indices() -> list[tuple[int, int, int]]:
    values = {(0, 0, 0), (0, 512, 2), (47, 0, 0), (47, 512, 2)}
    state = TRACE_SEED
    while len(values) < 132:
        state = splitmix64(state)
        values.add((int(state % LAYERS), int((state >> 8) % EXPERTS_WITH_SHARED), int((state >> 24) % 3)))
    return sorted(values)


def verify_and_warm_bank(mapped: np.memmap, manifest: dict[str, object]) -> dict[str, object]:
    allocation = disk_contract(BANK)
    if not allocation["non_sparse_fully_allocated"]:
        raise RuntimeError("bank is sparse, compressed, undersized or not fully allocated")
    failures = []
    checked_payload_bytes = 0
    for layer, expert, projection in sample_indices():
        offset = record_offset(layer, expert, projection)
        fields = struct.unpack_from(HEADER_FORMAT, mapped, offset)
        expected_projection, rows, columns = PROJECTIONS[projection]
        expected = (b"SQ5M", 1, layer, expert, expected_projection, 5, rows, columns, GROUP, CODE_BYTES, SCALE_BYTES)
        if fields[:11] != expected or fields[12] != bytes(28):
            failures.append(f"header:{layer}:{expert}:{projection}")
            continue
        code_begin = offset + HEADER_BYTES
        scale_begin = code_begin + CODE_BYTES
        padding_begin = scale_begin + SCALE_BYTES
        crc = zlib.crc32(memoryview(mapped)[scale_begin:padding_begin], zlib.crc32(memoryview(mapped)[code_begin:scale_begin])) & 0xFFFFFFFF
        if crc != fields[11]:
            failures.append(f"crc:{layer}:{expert}:{projection}")
        if np.any(mapped[padding_begin:offset + MATRIX_BYTES]):
            failures.append(f"padding:{layer}:{expert}:{projection}")
        checked_payload_bytes += CODE_BYTES + SCALE_BYTES
    if failures:
        raise RuntimeError(f"sampled bank verification failed: {failures[:8]}")
    digest = hashlib.sha256()
    view = memoryview(mapped)
    for begin in range(0, BANK_BYTES, 64 * 2**20):
        digest.update(view[begin:min(begin + 64 * 2**20, BANK_BYTES)])
    del view
    observed = digest.hexdigest()
    if observed != manifest["bank_sha256"]:
        raise RuntimeError("full bank SHA256 differs from build manifest")
    return {
        "full_sha256": observed,
        "full_sha256_matches_manifest": True,
        "sampled_records": len(sample_indices()),
        "sampled_payload_bytes": checked_payload_bytes,
        "allocation": allocation,
        "warmup": "full sequential SHA256 sweep over read-only mapping",
    }


def make_telemetry_point(process: psutil.Process, scenario: str, tokens: int, started: float) -> dict[str, object]:
    return {
        "utc": utc_now(),
        "scenario": scenario,
        "tokens": tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "process": process_memory(process),
        "system": system_memory(),
        "gpu": gpu_telemetry(),
    }


def run_transfer_scenario(
    cp: object,
    mapped: np.memmap,
    pinned_memories: list[object],
    pinned_arrays: list[np.ndarray],
    process: psutil.Process,
    name: str,
    token_start: int,
    tokens: int | None,
    deadline: float | None,
    global_started: float,
) -> dict[str, object]:
    if (tokens is None) == (deadline is None):
        raise ValueError("exactly one of tokens or deadline must be provided")
    slots = CACHE_SLOTS.get(name, PINNED_WINDOWS)
    cp.get_default_memory_pool().free_all_blocks()
    device = cp.empty(slots * EXPERT_BYTES, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)
    window_events = [cp.cuda.Event() for _ in range(PINNED_WINDOWS)]
    window_used = [False] * PINNED_WINDOWS
    cache: OrderedDict[tuple[int, int], int] = OrderedDict()
    free_slots = list(range(slots - 1, -1, -1))
    h2d_ms: list[float] = []
    wall_ms: list[float] = []
    misses_per_token: list[int] = []
    telemetry: list[dict[str, object]] = []
    route_digest = hashlib.sha256()
    transferred_bytes = 0
    hits = 0
    misses = 0
    local_index = 0
    last_telemetry = time.perf_counter()
    while (tokens is not None and local_index < tokens) or (deadline is not None and time.perf_counter() < deadline):
        token = token_start + local_index
        token_started = time.perf_counter()
        start_event = cp.cuda.Event()
        stop_event = cp.cuda.Event()
        start_event.record(stream)
        token_misses = 0
        for layer in range(LAYERS):
            selected = route(token, layer)
            route_digest.update(struct.pack("<IH", token & 0xFFFFFFFF, layer))
            route_digest.update(struct.pack("<10H", *selected))
            for expert in selected:
                key = (layer, expert)
                if name != "zero_cache" and key in cache:
                    hits += 1
                    cache.move_to_end(key)
                    continue
                misses += 1
                token_misses += 1
                if name == "zero_cache":
                    slot = misses % PINNED_WINDOWS
                elif free_slots:
                    slot = free_slots.pop()
                    cache[key] = slot
                else:
                    _, slot = cache.popitem(last=False)
                    cache[key] = slot
                window = (misses - 1) % PINNED_WINDOWS
                if window_used[window]:
                    window_events[window].synchronize()
                begin = expert_offset(layer, expert)
                np.copyto(pinned_arrays[window], mapped[begin:begin + EXPERT_BYTES])
                cp.cuda.runtime.memcpyAsync(
                    device.data.ptr + slot * EXPERT_BYTES,
                    pinned_memories[window].ptr,
                    EXPERT_BYTES,
                    cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
                window_events[window].record(stream)
                window_used[window] = True
                transferred_bytes += EXPERT_BYTES
        stop_event.record(stream)
        stop_event.synchronize()
        h2d_ms.append(float(cp.cuda.get_elapsed_time(start_event, stop_event)))
        wall_ms.append((time.perf_counter() - token_started) * 1000.0)
        misses_per_token.append(token_misses)
        local_index += 1
        now = time.perf_counter()
        if local_index % 250 == 0 or now - last_telemetry >= 30.0:
            point = make_telemetry_point(process, name, local_index, global_started)
            telemetry.append(point)
            print(json.dumps({"scenario": name, "tokens": local_index, "recent_h2d": timing_stats(h2d_ms[-250:]), "telemetry": point}), flush=True)
            last_telemetry = now
    stream.synchronize()
    final_point = make_telemetry_point(process, name, local_index, global_started)
    telemetry.append(final_point)
    del device
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "name": name,
        "token_start": token_start,
        "tokens": local_index,
        "cache_slots": 0 if name == "zero_cache" else slots,
        "route_sha256": route_digest.hexdigest(),
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / (hits + misses),
        "transferred_bytes": transferred_bytes,
        "h2d_ms": h2d_ms,
        "h2d": timing_stats(h2d_ms),
        "stage_plus_h2d_wall_ms": wall_ms,
        "stage_plus_h2d_wall": timing_stats(wall_ms),
        "misses_per_token": misses_per_token,
        "misses_per_token_stats": timing_stats([float(value) for value in misses_per_token]),
        "telemetry": telemetry,
    }


def benchmark(attempt_full_host_register: bool) -> dict[str, object]:
    validate_constant_math()
    if not BANK.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("physical bank and manifest are required")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    process = psutil.Process(os.getpid())
    memory_before_map = process_memory(process)
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    if mapped.flags.writeable:
        raise RuntimeError("bank mapping unexpectedly writable")
    warm_started = time.perf_counter()
    verification = verify_and_warm_bank(mapped, manifest)
    verification["warmup_seconds"] = time.perf_counter() - warm_started
    memory_after_warm = process_memory(process)

    full_register = {"attempted": False, "required": False}
    sampler = HardPageReadSampler()
    sampler.start()
    global_started = time.perf_counter()
    started_utc = utc_now()
    scenario_results: dict[str, dict[str, object]] = {}
    error: str | None = None
    cp = None
    try:
        import cupy as cp_module

        cp = cp_module
        cp.cuda.Device().use()
        if attempt_full_host_register:
            full_register["attempted"] = True
            try:
                cp.cuda.runtime.hostRegister(int(mapped.ctypes.data), BANK_BYTES, 0)
                full_register["success"] = True
                cp.cuda.runtime.hostUnregister(int(mapped.ctypes.data))
                full_register["immediately_unregistered"] = True
            except Exception as exc:
                full_register["success"] = False
                full_register["error"] = f"{type(exc).__name__}: {exc}"
        pinned_memories = [cp.cuda.alloc_pinned_memory(EXPERT_BYTES) for _ in range(PINNED_WINDOWS)]
        pinned_arrays = [np.frombuffer(item, dtype=np.uint8, count=EXPERT_BYTES) for item in pinned_memories]
        for name in ("zero_cache", "cache_4k", "cache_32k"):
            scenario_results[name] = run_transfer_scenario(
                cp, mapped, pinned_memories, pinned_arrays, process, name, 0,
                TOKENS_PER_SCENARIO, None, global_started,
            )
        deadline = global_started + STABILITY_SECONDS
        if time.perf_counter() < deadline:
            scenario_results["zero_cache_stability_extension"] = run_transfer_scenario(
                cp, mapped, pinned_memories, pinned_arrays, process, "zero_cache",
                TOKENS_PER_SCENARIO, None, deadline, global_started,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sampler.stop()
        if cp is not None:
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
    elapsed = time.perf_counter() - global_started
    final_memory = process_memory(process)
    all_telemetry = [point for result in scenario_results.values() for point in result["telemetry"]]
    telemetry_gaps = []
    monotonic_points = sorted(float(point["elapsed_seconds"]) for point in all_telemetry)
    for left, right in zip(monotonic_points, monotonic_points[1:]):
        telemetry_gaps.append(right - left)
    page_reads = [float(row["page_reads_per_sec"]) for row in sampler.samples]
    page_inputs = [float(row["pages_input_per_sec"]) for row in sampler.samples]
    zero = scenario_results.get("zero_cache")
    primary_complete = all(scenario_results.get(name, {}).get("tokens") == TOKENS_PER_SCENARIO for name in ("zero_cache", "cache_4k", "cache_32k"))
    gates = {
        "bank_physical_contract": bool(verification["allocation"]["non_sparse_fully_allocated"]),
        "full_sha256_verified": bool(verification["full_sha256_matches_manifest"]),
        "three_scenarios_10000_tokens": primary_complete,
        "one_hour_uninterrupted": elapsed >= STABILITY_SECONDS,
        "pdh_available_and_sampled": sampler.error is None and bool(sampler.samples),
        "no_system_page_reads_after_warmup": bool(page_reads) and max(page_reads) == 0.0,
        "peak_process_commit_le_58gib": final_memory["peak_pagefile"] <= COMMIT_LIMIT_BYTES,
        "zero_cache_h2d_p95_le_45ms": bool(zero) and float(zero["h2d"]["p95"]) <= H2D_P95_LIMIT_MS,
        "no_cuda_or_runner_error": error is None,
        "telemetry_gap_le_45s": bool(monotonic_points) and (not telemetry_gaps or max(telemetry_gaps) <= 45.0),
        "no_thermal_or_driver_error": bool(all_telemetry) and all("error" not in point["gpu"] for point in all_telemetry),
    }
    result = {
        "kind": "port80b_p0_physical_host_bank_gate",
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "status": "pass" if all(gates.values()) else "fail",
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST),
        },
        "contract": expected_contract(),
        "verification_and_warmup": verification,
        "mapping_readonly": not mapped.flags.writeable,
        "full_host_registration": full_register,
        "memory_before_map": memory_before_map,
        "memory_after_warmup": memory_after_warm,
        "memory_final": final_memory,
        "scenarios": scenario_results,
        "hard_page_read_telemetry": {
            "scope": "system-wide conservative PDH counters after full-bank warmup",
            "error": sampler.error,
            "samples": sampler.samples,
            "page_reads_per_sec": timing_stats(page_reads),
            "pages_input_per_sec": timing_stats(page_inputs),
        },
        "runner_error": error,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "claim_boundary": "Synthetic final-size host-bank residency/H2D gate only; no real weights, quality, shell, decode or 80B tokens/s claim.",
    }
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite {RESULT}")
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def require_acknowledgement(value: str | None) -> None:
    if value != ACKNOWLEDGEMENT:
        raise PermissionError(f"this phase requires --timing-go {ACKNOWLEDGEMENT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PORT80B_P0 full-size physical host-bank gate")
    parser.add_argument("--phase", choices=("preflight", "build", "benchmark", "full"), default="preflight")
    parser.add_argument("--timing-go")
    parser.add_argument("--attempt-full-host-register", action="store_true")
    args = parser.parse_args()
    if args.phase == "preflight":
        result = preflight()
    else:
        require_acknowledgement(args.timing_go)
        result = None
        if args.phase in ("build", "full"):
            result = build_bank()
        if args.phase in ("benchmark", "full"):
            result = benchmark(args.attempt_full_host_register)
        assert result is not None
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
