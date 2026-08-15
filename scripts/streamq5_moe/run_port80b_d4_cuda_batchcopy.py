from __future__ import annotations

import ctypes
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
PREREG = R / "PORT80B_D4R_CUDA_BATCHCOPY_PREREGISTRATION.md"
OUTPUT = R / "port80b_d4r_cuda_batchcopy.json"
REPORT = R / "PORT80B_D4R_CUDA_BATCHCOPY_REPORT_2026-08-12.md"
EXPERTS = 307
ARMS = ("ordinary480", "batch48x10", "batch1x480")
CANDIDATES = ("batch48x10", "batch1x480")
WARMUPS = 6
VALIDATION_ROUNDS = 24
TEST_ROUNDS = 120


class CudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", CudaMemLocation),
        ("dstLocHint", CudaMemLocation),
        ("flags", ctypes.c_uint),
    ]


class Batch:
    def __init__(self, destinations: list[int], sources: list[int]):
        if len(destinations) != len(sources):
            raise ValueError("batch arrays differ")
        self.count = len(destinations)
        self.destinations = (ctypes.c_void_p * self.count)(*destinations)
        self.sources = (ctypes.c_void_p * self.count)(*sources)
        self.sizes = (ctypes.c_size_t * self.count)(*([EXPERT_BYTES] * self.count))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_native() -> tuple[ctypes.WinDLL, object]:
    dll_path = ROOT / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/cudart64_13.dll"
    if not dll_path.is_file():
        raise FileNotFoundError(dll_path)
    dll = ctypes.WinDLL(str(dll_path))
    function = dll.cudaMemcpyBatchAsync
    function.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
        ctypes.POINTER(CudaMemcpyAttributes), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t, ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return dll, function


def register_ranges(mapped: np.memmap) -> tuple[list[int], list[int]]:
    result = []
    aliases = []
    size = EXPERTS * EXPERT_BYTES
    try:
        for layer in range(LAYERS):
            pointer = int(mapped.ctypes.data) + record_offset(layer, 0)
            cp.cuda.runtime.hostRegister(pointer, size, REGISTER_FLAGS)
            result.append(pointer)
            alias = int(cp.cuda.runtime.pointerGetAttributes(pointer).devicePointer)
            if not alias:
                raise RuntimeError(f"registered layer {layer} returned null device alias")
            aliases.append(alias)
    except Exception:
        unregister_ranges(result)
        raise
    return result, aliases


def make_batches(mapped: np.memmap, destination: cp.ndarray, token: int, mode: str, aliases: list[int]) -> tuple[list[Batch], list[tuple[int, int]]]:
    selected = routes(token, EXPERTS)
    destinations = [int(destination.data.ptr) + index * EXPERT_BYTES for index in range(len(selected))]
    sources = [aliases[layer] + expert * EXPERT_BYTES for layer, expert in selected]
    if mode == "batch48x10":
        return [Batch(destinations[index:index + 10], sources[index:index + 10]) for index in range(0, 480, 10)], selected
    if mode == "batch1x480":
        return [Batch(destinations, sources)], selected
    return [], selected


def native_launch(function: object, batches: list[Batch], attributes: CudaMemcpyAttributes, attr_index: ctypes.c_size_t, stream: cp.cuda.Stream) -> None:
    for batch in batches:
        status = int(function(
            batch.destinations, batch.sources, batch.sizes, batch.count,
            ctypes.byref(attributes), ctypes.byref(attr_index), 1,
            ctypes.c_void_p(stream.ptr),
        ))
        if status != 0:
            raise RuntimeError(f"cudaMemcpyBatchAsync status={status}")


def ordinary_launch(mapped: np.memmap, destination: cp.ndarray, selected: list[tuple[int, int]], stream: cp.cuda.Stream) -> None:
    for index, (layer, expert) in enumerate(selected):
        cp.cuda.runtime.memcpyAsync(
            int(destination.data.ptr) + index * EXPERT_BYTES,
            int(mapped.ctypes.data) + record_offset(layer, expert),
            EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
        )


def launch(arm: str, function: object, mapped: np.memmap, destination: cp.ndarray, selected: list[tuple[int, int]], batches: list[Batch], attributes: CudaMemcpyAttributes, attr_index: ctypes.c_size_t, stream: cp.cuda.Stream) -> None:
    if arm == "ordinary480":
        ordinary_launch(mapped, destination, selected, stream)
    else:
        native_launch(function, batches, attributes, attr_index, stream)


def timed(*args: object) -> float:
    stream = args[-1]
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record(stream)
    launch(*args)
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(begin, end))


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D4 result")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not BANK.is_file() or BANK.stat().st_size != BANK_BYTES or manifest.get("bank_sha256") != EXPECTED_BANK_SHA256:
        raise RuntimeError("immutable P0 bank/manifest contract failed")
    if ctypes.sizeof(CudaMemLocation) != 8 or ctypes.sizeof(CudaMemcpyAttributes) != 24:
        raise RuntimeError("CUDA 13.2 bundled-header ABI size mismatch")

    started = time.perf_counter()
    dll, batch_function = load_native()
    try:
        cp.cuda.runtime.setDeviceFlags(0x08)
    except Exception:
        pass
    cp.cuda.Device(0).use()
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    destination = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)
    verify_kernel = cp.RawKernel(VERIFY_SOURCE, "verify_record_bytes", options=("--std=c++14",))
    attributes = CudaMemcpyAttributes(3, CudaMemLocation(0, 0), CudaMemLocation(0, 0), 0)
    attr_index = ctypes.c_size_t(0)
    registered: list[int] = []
    payload: dict[str, object] = {}
    error = None
    unregister_failures = []
    try:
        registered, aliases = register_ranges(mapped)
        tokens = list(range(70_000, 70_000 + VALIDATION_ROUNDS)) + list(range(80_000, 80_000 + TEST_ROUNDS)) + [69_999]
        descriptors: dict[int, dict[str, tuple[list[Batch], list[tuple[int, int]]]]] = {}
        for token in tokens:
            descriptors[token] = {arm: make_batches(mapped, destination, token, arm, aliases) for arm in CANDIDATES}

        correctness = {}
        check_token = 69_999
        for arm in ARMS:
            batches, selected = descriptors[check_token].get(arm, ([], routes(check_token, EXPERTS)))
            launch(arm, batch_function, mapped, destination, selected, batches, attributes, attr_index, stream)
            stream.synchronize()
            correctness[arm] = {"full_destination_mismatch_count": full_verify(verify_kernel, destination, selected, stream)}

        for arm in ARMS:
            for warmup in range(WARMUPS):
                token = 70_000 + warmup
                batches, selected = descriptors[token].get(arm, ([], routes(token, EXPERTS)))
                launch(arm, batch_function, mapped, destination, selected, batches, attributes, attr_index, stream)
        stream.synchronize()

        raw_validation = {arm: [] for arm in ARMS}
        orders = []
        validation_tokens = list(range(70_000, 70_000 + VALIDATION_ROUNDS))
        for round_index, token in enumerate(validation_tokens):
            rotation = round_index % len(ARMS)
            order = list(ARMS[rotation:] + ARMS[:rotation])
            if round_index & 1:
                order.reverse()
            orders.append(order)
            for arm in order:
                batches, selected = descriptors[token].get(arm, ([], routes(token, EXPERTS)))
                raw_validation[arm].append(timed(arm, batch_function, mapped, destination, selected, batches, attributes, attr_index, stream))
        validation = {arm: {"raw_ms": values, "stats": stats(values)} for arm, values in raw_validation.items()}
        selected_arm = min(CANDIDATES, key=lambda arm: (float(validation[arm]["stats"]["p50"]), CANDIDATES.index(arm)))
        correctness_pass = all(row["full_destination_mismatch_count"] == 0 for row in correctness.values())
        validation_open = correctness_pass and float(validation[selected_arm]["stats"]["p50"]) <= 1.05 * float(validation["ordinary480"]["stats"]["p50"])

        raw_test = []
        test_tokens = list(range(80_000, 80_000 + TEST_ROUNDS))
        if validation_open:
            for token in test_tokens:
                batches, selected = descriptors[token][selected_arm]
                raw_test.append(timed(selected_arm, batch_function, mapped, destination, selected, batches, attributes, attr_index, stream))
        test_stats = stats(raw_test) if raw_test else None
        effective = TOKEN_BYTES / (float(test_stats["p95"]) / 1000.0) / 1e9 if test_stats else None
        ratios = {
            "selected_validation_p50_over_ordinary": float(validation[selected_arm]["stats"]["p50"]) / float(validation["ordinary480"]["stats"]["p50"]),
            "selected_validation_p95_over_ordinary": float(validation[selected_arm]["stats"]["p95"]) / float(validation["ordinary480"]["stats"]["p95"]),
        }
        gates = {
            "native_symbol_and_abi": True,
            "all_arms_zero_mismatches": correctness_pass,
            "test_120_finite": len(raw_test) == TEST_ROUNDS and bool(np.isfinite(raw_test).all()),
            "test_p95_le_45ms": bool(test_stats and float(test_stats["p95"]) <= 45.0),
            "effective_gb_s_at_p95_ge_21_627": bool(effective is not None and effective >= 21.627),
            "validation_p50_ratio_le_0_90": ratios["selected_validation_p50_over_ordinary"] <= 0.90,
            "validation_p95_ratio_le_0_90": ratios["selected_validation_p95_over_ordinary"] <= 0.90,
            "registration_48_ranges": len(registered) == LAYERS,
            "no_cuda_or_runner_error": True,
        }
        payload = {
            "correctness": correctness,
            "validation": {"tokens": validation_tokens, "orders": orders, "arms": validation},
            "selected_arm": selected_arm,
            "validation_open": validation_open,
            "test": {"tokens": test_tokens if validation_open else [], "raw_ms": raw_test, "stats": test_stats},
            "effective_gb_s_at_p95": effective,
            "ratios": ratios,
            "gates": gates,
            "pass": all(gates.values()),
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            stream.synchronize()
        except Exception:
            pass
        unregister_failures = unregister_ranges(registered)

    passed = bool(payload.get("pass")) and error is None and not unregister_failures
    result = {
        "kind": "port80b_d4r_cuda_batchcopy",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "native_batchcopy_pass" if passed else "native_batchcopy_negative",
        "pass": passed,
        "full_bank_pass": False,
        "native": {"dll": str(dll._name), "symbol": "cudaMemcpyBatchAsync", "mem_location_size": ctypes.sizeof(CudaMemLocation), "attributes_size": ctypes.sizeof(CudaMemcpyAttributes), "src_access_order": "Any"},
        "inputs": {"preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(MANIFEST), "bank_sha256_from_manifest": manifest["bank_sha256"]},
        "protocol": {"experts_per_layer": EXPERTS, "arms": list(ARMS), "warmups": WARMUPS, "validation_rounds": VALIDATION_ROUNDS, "test_rounds": TEST_ROUNDS},
        **payload,
        "error": error,
        "unregister_failures": unregister_failures,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "60%-bank native batch-copy transport only; no Q5 arithmetic, full-bank, model, quality, dense shell, tok/s or endurance claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    validation = payload.get("validation", {}).get("arms", {})
    REPORT.write_text(
        "# PORT80B-D4R — native CUDA batch-copy alias-repair report\n\n"
        f"Verdict: **{result['status']}**. Selected: {payload.get('selected_arm', '—')}. "
        f"Validation p50 ordinary/batch48/batch480: "
        f"{validation.get('ordinary480', {}).get('stats', {}).get('p50', '—')} / "
        f"{validation.get('batch48x10', {}).get('stats', {}).get('p50', '—')} / "
        f"{validation.get('batch1x480', {}).get('stats', {}).get('p50', '—')} ms.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "pass": passed, "selected": payload.get("selected_arm"), "validation": {arm: row["stats"] for arm, row in validation.items()}, "test": payload.get("test", {}).get("stats"), "ratios": payload.get("ratios"), "gates": payload.get("gates"), "error": error, "unregister_failures": unregister_failures}, indent=2))


if __name__ == "__main__":
    main()
