from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import psutil

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE
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
from scripts.streamq5_moe.run_port80b_d5_cp_async_host_smem import SOURCE as STAGE_SOURCE
from scripts.streamq5_moe.run_port80b_d7_staged_exact_q5_plane import COMPUTE_SOURCE
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import (
    BANK,
    BANK_BYTES,
    EXPERT_BYTES,
    LAYERS,
    MANIFEST,
)


R = ROOT / "reports" / "streamq5_moe"
PREREG = R / "PORT80B_D9_CAPACITY_AWARE_BANK_BRIDGE_PREREGISTRATION.md"
COMPILE_OUT = R / "port80b_d9_capacity_aware_bank_bridge_compile.json"
OUTPUT = R / "port80b_d9_capacity_aware_bank_bridge.json"
REPORT = R / "PORT80B_D9_CAPACITY_AWARE_BANK_BRIDGE_REPORT_2026-08-12.md"
D7_RESULT = R / "port80b_d7_staged_exact_q5_plane.json"
D8_VERIFICATION = R / "port80b_d8_registration_capacity_independent_verification.json"

PREFIX = 499
COLD_BEGIN = 499
COLD_END = 512
ACTIVE = 10
HIDDEN = 2048
INTER = 512
SEED = 120_829
STAGE_BLOCKS = 1024
STAGE_THREADS = 256
TILES_PER_RECORD = 495
TILE_BYTES = 4096
WARMUPS = 4
VALIDATION_ROUNDS = 24
TEST_ROUNDS = 60
MIN_AVAILABLE = 2 * 2**30
VALIDATION_P50_LIMITS = {"all_hot": 65.0, "mixed_5_hot_5_cold": 100.0, "all_cold_tail": 135.0}
TEST_P95_LIMITS = {"all_hot": 65.0, "mixed_5_hot_5_cold": 100.0, "all_cold_tail": 135.0}
CASE_NAMES = tuple(VALIDATION_P50_LIMITS)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def build_module() -> tuple[cp.RawModule, dict[str, cp.RawKernel]]:
    cuda_include = ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "include"
    if not cuda_include.is_dir():
        raise RuntimeError(f"CUDA include directory missing: {cuda_include}")
    names = (
        "host_to_smem_pipeline",
        "staged_q5_gate_up",
        "staged_q5_down",
        "canonical_swiglu_d7",
        "verify_record_bytes",
    )
    # The D2 verifier source is obtained through the function's defining module
    # to keep the exact differentiated byte oracle used by earlier planes.
    from scripts.streamq5_moe.run_port80b_d2_registered_scatter import VERIFY_SOURCE

    module = cp.RawModule(
        code=CUDA_SOURCE + ERVF_SOURCE + STAGE_SOURCE + COMPUTE_SOURCE + VERIFY_SOURCE,
        options=("--std=c++14", f"--include-path={cuda_include}"),
        name_expressions=names,
    )
    return module, {name: module.get_function(name) for name in names}


def immutable_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d7 = json.loads(D7_RESULT.read_text(encoding="utf-8"))
    d8 = json.loads(D8_VERIFICATION.read_text(encoding="utf-8"))
    mapped_readonly = False
    if BANK.is_file() and BANK.stat().st_size == BANK_BYTES:
        mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
        mapped_readonly = not mapped.flags.writeable
        del mapped
    checks = {
        "bank_exists_and_exact_size": BANK.is_file() and BANK.stat().st_size == BANK_BYTES,
        "manifest_bank_sha_locked": manifest.get("bank_sha256") == EXPECTED_BANK_SHA256,
        "mapping_readonly": mapped_readonly,
        "d7_strong_exact_pass": bool(d7.get("strong_pass")) and bool(d7.get("correctness", {}).get("bitwise_equal")),
        "d8_independent_replayable": bool(d8.get("all_replayable_checks_pass")),
        "d8_largest_clean_prefix_is_499": d8.get("capacity", {}).get("largest_clean_prefix_experts_per_layer") == PREFIX,
        "d8_512_is_not_clean": d8.get("capacity", {}).get("raw_largest_claim_is_protocol_valid") is False,
        "d8_raw_512_unregister_failures_44": d8.get("protocol_checks", {}).get("full_prefix_has_exactly_44_raw_unregister_failures") is True,
        "constant_math": PREFIX + (COLD_END - COLD_BEGIN) == 512
        and TOKEN_BYTES == LAYERS * ACTIVE * EXPERT_BYTES
        and EXPERT_BYTES == TILES_PER_RECORD * TILE_BYTES,
    }
    return manifest, {
        "checks": checks,
        "pass": all(checks.values()),
        "bank_size": BANK.stat().st_size if BANK.is_file() else None,
        "available_ram_bytes": int(psutil.virtual_memory().available),
        "expected_registered_bytes": LAYERS * PREFIX * EXPERT_BYTES,
        "expected_registered_gib": LAYERS * PREFIX * EXPERT_BYTES / 2**30,
        "d8_cumulative_ram_caveat": d8.get("cumulative_ram_caveat"),
    }


def compile_phase() -> None:
    if COMPILE_OUT.exists():
        raise FileExistsError(f"refusing to overwrite {COMPILE_OUT}")
    started = time.perf_counter()
    error = None
    manifest: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    symbols: list[str] = []
    try:
        manifest, audit = immutable_audit()
        if not audit["pass"]:
            raise RuntimeError("immutable compile audit failed")
        compile(Path(__file__).read_text(encoding="utf-8"), str(Path(__file__)), "exec")
        _, kernels = build_module()
        symbols = sorted(kernels)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "kind": "port80b_d9_capacity_aware_bank_bridge_compile",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "compile_preflight_pass" if error is None else "compile_preflight_fail",
        "pass": error is None and bool(audit.get("pass")),
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST),
            "manifest_bank_sha256": manifest.get("bank_sha256"),
            "d7_result_sha256": sha256(D7_RESULT),
            "d8_independent_verification_sha256": sha256(D8_VERIFICATION),
        },
        "audit": audit,
        "cuda": {
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8"),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cupy_version": cp.__version__,
            "resolved_symbols": symbols,
        },
        "physical_actions": {
            "host_registration": False,
            "large_hbm_allocation": False,
            "bank_sweep": False,
            "timing": False,
        },
        "error": error,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Compile and read-only contract audit only; no 499-prefix registration, cold escape, correctness capture or timing.",
    }
    COMPILE_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def frozen_cases() -> dict[str, list[tuple[int, int]]]:
    hot_1 = routes(130_001, PREFIX)
    hot_2 = routes(130_002, PREFIX)
    by_layer_1 = {layer: [] for layer in range(LAYERS)}
    by_layer_2 = {layer: [] for layer in range(LAYERS)}
    for layer, expert in hot_1:
        by_layer_1[layer].append(expert)
    for layer, expert in hot_2:
        by_layer_2[layer].append(expert)

    cases: dict[str, list[tuple[int, int]]] = {name: [] for name in CASE_NAMES}
    for layer in range(LAYERS):
        cases["all_hot"].extend((layer, expert) for expert in by_layer_1[layer])
        cases["mixed_5_hot_5_cold"].extend((layer, expert) for expert in by_layer_2[layer][:5])
        cases["mixed_5_hot_5_cold"].extend((layer, COLD_BEGIN + ((layer + rank) % 13)) for rank in range(5))
        cases["all_cold_tail"].extend((layer, COLD_BEGIN + ((layer + rank) % 13)) for rank in range(10))
    return cases


def case_contract(cases: dict[str, list[tuple[int, int]]]) -> dict[str, Any]:
    expected_counts = {
        "all_hot": (480, 0),
        "mixed_5_hot_5_cold": (240, 240),
        "all_cold_tail": (0, 480),
    }
    result: dict[str, Any] = {}
    for name, selected in cases.items():
        hot = sum(expert < PREFIX for _, expert in selected)
        cold = len(selected) - hot
        valid = len(selected) == 480 and (hot, cold) == expected_counts[name]
        for layer in range(LAYERS):
            values = [expert for route_layer, expert in selected if route_layer == layer]
            valid &= len(values) == ACTIVE and len(set(values)) == ACTIVE and all(0 <= expert < 512 for expert in values)
        result[name] = {
            "records": len(selected),
            "hot_records": hot,
            "cold_escape_records": cold,
            "route_sha256": hashlib.sha256(np.asarray(selected, dtype=np.int16).tobytes()).hexdigest(),
            "pass": bool(valid),
        }
    return result


def register_prefix(mapped: np.memmap) -> tuple[list[int], list[int]]:
    hosts: list[int] = []
    aliases: list[int] = []
    size = PREFIX * EXPERT_BYTES
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


def compare(observed: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    left, right = observed.view(np.uint32), expected.view(np.uint32)
    return {
        "elements": int(expected.size),
        "different_bits": int(np.count_nonzero(left != right)),
        "bitwise_equal": bool(np.array_equal(left, right)),
        "max_abs": float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64)), initial=0.0)),
        "finite": bool(np.isfinite(observed).all()),
        "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
    }


def fixed_order(round_index: int) -> list[str]:
    rotation = round_index % len(CASE_NAMES)
    order = list(CASE_NAMES[rotation:] + CASE_NAMES[:rotation])
    if round_index & 1:
        order.reverse()
    return order


def run_phase() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite an existing D9 result/report")
    if not COMPILE_OUT.is_file():
        raise RuntimeError("compile/preflight evidence is missing")
    compile_evidence = json.loads(COMPILE_OUT.read_text(encoding="utf-8"))
    if not compile_evidence.get("pass"):
        raise RuntimeError("compile/preflight did not pass")
    if compile_evidence["inputs"]["preregistration_sha256"] != sha256(PREREG):
        raise RuntimeError("preregistration changed after compile/preflight")
    if compile_evidence["inputs"]["runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("runner changed after compile/preflight")
    available_before = int(psutil.virtual_memory().available)
    if available_before < MIN_AVAILABLE:
        raise RuntimeError(f"safety stop: available RAM {available_before} < {MIN_AVAILABLE}")

    manifest, audit = immutable_audit()
    if not audit["pass"]:
        raise RuntimeError("immutable run audit failed")
    cases = frozen_cases()
    route_contract = case_contract(cases)
    if not all(row["pass"] for row in route_contract.values()):
        raise RuntimeError(f"frozen route contract failed: {route_contract}")

    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    if mapped.flags.writeable:
        raise RuntimeError("bank mapping is unexpectedly writeable")
    _, kernels = build_module()
    stream = cp.cuda.Stream(non_blocking=True)
    started = time.perf_counter()
    hosts: list[int] = []
    aliases: list[int] = []
    payload: dict[str, Any] = {}
    error = None
    unregister_failures: list[str] = []
    available_after_registration: int | None = None
    available_after_unregister: int | None = None
    try:
        hosts, aliases = register_prefix(mapped)
        available_after_registration = int(psutil.virtual_memory().available)
        if len(hosts) != LAYERS or len(aliases) != LAYERS:
            raise RuntimeError("did not obtain exactly 48 registered prefix aliases")
        if available_after_registration < MIN_AVAILABLE:
            raise RuntimeError(
                f"post-registration safety stop: available RAM {available_after_registration} < {MIN_AVAILABLE}"
            )

        staging = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
        cold_escape = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
        reference = cp.empty(ACTIVE * EXPERT_BYTES, dtype=cp.uint8)
        pointer_host: dict[str, np.ndarray] = {}
        cold_rows: dict[str, list[tuple[int, int, int]]] = {}
        for name, selected in cases.items():
            values: list[int] = []
            cold: list[tuple[int, int, int]] = []
            for record_index, (layer, expert) in enumerate(selected):
                if expert < PREFIX:
                    values.append(aliases[layer] + expert * EXPERT_BYTES)
                else:
                    values.append(int(cold_escape.data.ptr) + record_index * EXPERT_BYTES)
                    cold.append((record_index, layer, expert))
            pointer_host[name] = np.asarray(values, dtype=np.uint64)
            cold_rows[name] = cold
        pointer_device = {name: cp.asarray(values) for name, values in pointer_host.items()}

        cp.cuda.runtime.memcpyAsync(
            reference.data.ptr,
            int(mapped.ctypes.data),
            ACTIVE * EXPERT_BYTES,
            cp.cuda.runtime.memcpyHostToDevice,
            stream.ptr,
        )
        rng = np.random.default_rng(SEED)
        x_host = rng.standard_normal(HIDDEN, dtype=np.float32)
        x = cp.asarray(x_host)
        gate = cp.empty(ACTIVE * INTER, dtype=cp.float32)
        up = cp.empty_like(gate)
        down = cp.empty(ACTIVE * HIDDEN, dtype=cp.float32)

        def copy_cold(name: str) -> None:
            for record_index, layer, expert in cold_rows[name]:
                cp.cuda.runtime.memcpyAsync(
                    int(cold_escape.data.ptr) + record_index * EXPERT_BYTES,
                    int(mapped.ctypes.data) + record_offset(layer, expert),
                    EXPERT_BYTES,
                    cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )

        def stage(name: str, pointers: cp.ndarray | None = None) -> None:
            copy_cold(name)
            kernels["host_to_smem_pipeline"](
                (STAGE_BLOCKS,),
                (STAGE_THREADS,),
                (pointer_device[name] if pointers is None else pointers, staging, np.uint64(480 * TILES_PER_RECORD)),
                shared_mem=TILE_BYTES,
                stream=stream,
            )

        def compute(bank: cp.ndarray, layer: int) -> None:
            kernels["staged_q5_gate_up"](
                (ACTIVE * 1024 // 32,), (256,), (x, bank, np.int32(layer), gate, up), stream=stream
            )
            kernels["canonical_swiglu_d7"](
                ((ACTIVE * INTER + 255) // 256,), (256,), (gate, up), stream=stream
            )
            kernels["staged_q5_down"](
                (ACTIVE * HIDDEN // 32,), (256,), (gate, bank, np.int32(layer), down), stream=stream
            )

        def capture(bank: cp.ndarray, layer_argument: bool) -> np.ndarray:
            output = np.empty((LAYERS, ACTIVE * (INTER + INTER + HIDDEN)), dtype=np.float32)
            for layer in range(LAYERS):
                compute(bank, layer if layer_argument else 0)
                stream.synchronize()
                output[layer] = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))
            return output

        def execute(name: str, timed: bool) -> tuple[float | None, float | None]:
            begin_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            begin_wall = time.perf_counter()
            begin_event.record(stream)
            stage(name)
            for layer in range(LAYERS):
                compute(staging, layer)
            end_event.record(stream)
            end_event.synchronize()
            wall_ms = (time.perf_counter() - begin_wall) * 1000.0
            event_ms = float(cp.cuda.get_elapsed_time(begin_event, end_event))
            return (wall_ms, event_ms) if timed else (None, None)

        expected = capture(reference, False)
        integrity: dict[str, Any] = {"positive": {}}
        correctness: dict[str, Any] = {}
        for name in CASE_NAMES:
            stage(name)
            stream.synchronize()
            mismatch = full_verify(kernels["verify_record_bytes"], staging, cases[name], stream)
            integrity["positive"][name] = {"full_image_byte_mismatches": mismatch, "pass": mismatch == 0}
            stage(name)
            stream.synchronize()
            observed = capture(staging, True)
            correctness[name] = compare(observed, expected)

        original = pointer_host["all_hot"]
        selected = cases["all_hot"]
        first_layer, first_expert = selected[0]
        wrong_expert = (first_expert + 1) % PREFIX
        if wrong_expert == first_expert:
            raise AssertionError("wrong-expert control did not change expert")
        corrupted_expert = original.copy()
        corrupted_expert[0] = aliases[first_layer] + wrong_expert * EXPERT_BYTES
        stage("all_hot", cp.asarray(corrupted_expert))
        stream.synchronize()
        expert_mismatches = full_verify(kernels["verify_record_bytes"], staging, selected, stream)

        wrong_layer = (first_layer + 1) % LAYERS
        corrupted_layer = original.copy()
        corrupted_layer[0] = aliases[wrong_layer] + first_expert * EXPERT_BYTES
        stage("all_hot", cp.asarray(corrupted_layer))
        stream.synchronize()
        layer_mismatches = full_verify(kernels["verify_record_bytes"], staging, selected, stream)
        integrity["negative_controls"] = {
            "wrong_expert": {
                "expected": [first_layer, first_expert],
                "substituted": [first_layer, wrong_expert],
                "detected_mismatches": expert_mismatches,
                "pass": expert_mismatches > 0,
            },
            "wrong_layer": {
                "expected": [first_layer, first_expert],
                "substituted": [wrong_layer, first_expert],
                "detected_mismatches": layer_mismatches,
                "pass": layer_mismatches > 0,
            },
        }

        exact = all(row["bitwise_equal"] and row["finite"] and row["expected_sha256"] == row["observed_sha256"] for row in correctness.values())
        integrity_pass = all(row["pass"] for row in integrity["positive"].values()) and all(
            row["pass"] for row in integrity["negative_controls"].values()
        )

        if exact and integrity_pass:
            for name in CASE_NAMES:
                for _ in range(WARMUPS):
                    execute(name, False)
        stream.synchronize()

        validation_wall = {name: [] for name in CASE_NAMES}
        validation_event = {name: [] for name in CASE_NAMES}
        validation_orders: list[list[str]] = []
        if exact and integrity_pass:
            for round_index in range(VALIDATION_ROUNDS):
                order = fixed_order(round_index)
                validation_orders.append(order)
                for name in order:
                    wall_ms, event_ms = execute(name, True)
                    assert wall_ms is not None and event_ms is not None
                    validation_wall[name].append(wall_ms)
                    validation_event[name].append(event_ms)
        validation = {
            name: {
                "wall_ms": validation_wall[name],
                "wall_stats": stats(validation_wall[name]) if validation_wall[name] else None,
                "cuda_event_ms": validation_event[name],
                "cuda_event_stats": stats(validation_event[name]) if validation_event[name] else None,
            }
            for name in CASE_NAMES
        }
        validation_finite = all(
            len(validation_wall[name]) == VALIDATION_ROUNDS and bool(np.isfinite(validation_wall[name]).all())
            for name in CASE_NAMES
        )
        validation_latency = all(
            validation[name]["wall_stats"] is not None
            and float(validation[name]["wall_stats"]["p50"]) <= VALIDATION_P50_LIMITS[name]
            for name in CASE_NAMES
        )
        validation_open = exact and integrity_pass and validation_finite and validation_latency

        test_wall = {name: [] for name in CASE_NAMES}
        test_event = {name: [] for name in CASE_NAMES}
        test_orders: list[list[str]] = []
        if validation_open:
            for round_index in range(TEST_ROUNDS):
                order = fixed_order(VALIDATION_ROUNDS + round_index)
                test_orders.append(order)
                for name in order:
                    wall_ms, event_ms = execute(name, True)
                    assert wall_ms is not None and event_ms is not None
                    test_wall[name].append(wall_ms)
                    test_event[name].append(event_ms)
        test = {
            name: {
                "wall_ms": test_wall[name],
                "wall_stats": stats(test_wall[name]) if test_wall[name] else None,
                "cuda_event_ms": test_event[name],
                "cuda_event_stats": stats(test_event[name]) if test_event[name] else None,
            }
            for name in CASE_NAMES
        }
        gates = {
            "mapping_readonly": not mapped.flags.writeable,
            "prefix_exactly_499_and_tail_exactly_13": PREFIX == 499 and COLD_END - COLD_BEGIN == 13,
            "route_source_provenance_exact": all(row["pass"] for row in route_contract.values()),
            "positive_route_integrity_zero_mismatch": all(row["pass"] for row in integrity["positive"].values()),
            "wrong_expert_detected": integrity["negative_controls"]["wrong_expert"]["pass"],
            "wrong_layer_detected": integrity["negative_controls"]["wrong_layer"]["pass"],
            "all_outputs_bitexact_finite_digest_equal": exact,
            "validation_24_each_finite": validation_finite,
            "validation_p50_case_limits": validation_latency,
            "test_60_each_finite": all(
                len(test_wall[name]) == TEST_ROUNDS and bool(np.isfinite(test_wall[name]).all()) for name in CASE_NAMES
            ),
            **{
                f"test_{name}_wall_p95_le_{int(TEST_P95_LIMITS[name])}ms": bool(
                    test[name]["wall_stats"] and float(test[name]["wall_stats"]["p95"]) <= TEST_P95_LIMITS[name]
                )
                for name in CASE_NAMES
            },
            "strong_mixed_wall_p95_le_80ms": bool(
                test["mixed_5_hot_5_cold"]["wall_stats"]
                and float(test["mixed_5_hot_5_cold"]["wall_stats"]["p95"]) <= 80.0
            ),
            "strong_all_cold_wall_p95_le_110ms": bool(
                test["all_cold_tail"]["wall_stats"]
                and float(test["all_cold_tail"]["wall_stats"]["p95"]) <= 110.0
            ),
            "registration_48_ranges": len(hosts) == LAYERS,
            "post_registration_available_ram_ge_2gib": available_after_registration >= MIN_AVAILABLE,
            "no_cuda_or_runner_error": True,
        }
        payload = {
            "route_contract": route_contract,
            "pointer_tables": {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in pointer_host.items()},
            "integrity": integrity,
            "correctness": correctness,
            "validation": {"orders": validation_orders, "cases": validation, "open": validation_open},
            "test": {"orders": test_orders, "cases": test, "opened": validation_open},
            "gates": gates,
            "input_sha256": hashlib.sha256(x_host.tobytes()).hexdigest(),
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            stream.synchronize()
        except Exception:
            pass
        unregister_failures = unregister_ranges(hosts)
        available_after_unregister = int(psutil.virtual_memory().available)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    gates = payload.setdefault("gates", {})
    gates["clean_unregister_48_ranges"] = len(hosts) == LAYERS and not unregister_failures
    gates["no_cuda_or_runner_error"] = error is None
    primary_names = (
        "mapping_readonly",
        "prefix_exactly_499_and_tail_exactly_13",
        "route_source_provenance_exact",
        "positive_route_integrity_zero_mismatch",
        "wrong_expert_detected",
        "wrong_layer_detected",
        "all_outputs_bitexact_finite_digest_equal",
        "validation_24_each_finite",
        "validation_p50_case_limits",
        "test_60_each_finite",
        "test_all_hot_wall_p95_le_65ms",
        "test_mixed_5_hot_5_cold_wall_p95_le_100ms",
        "test_all_cold_tail_wall_p95_le_135ms",
        "registration_48_ranges",
        "post_registration_available_ram_ge_2gib",
        "clean_unregister_48_ranges",
        "no_cuda_or_runner_error",
    )
    primary = all(gates.get(name) is True for name in primary_names)
    strong = primary and gates.get("strong_mixed_wall_p95_le_80ms") is True and gates.get("strong_all_cold_wall_p95_le_110ms") is True
    result = {
        "kind": "port80b_d9_capacity_aware_bank_bridge",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "capacity_bridge_strong_pass" if strong else ("capacity_bridge_primary_pass" if primary else "capacity_bridge_negative"),
        "primary_pass": primary,
        "strong_pass": strong,
        "full_bank_registration_pass": False,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "compile_evidence_sha256": sha256(COMPILE_OUT),
            "manifest_sha256": sha256(MANIFEST),
            "bank_sha256_from_manifest": manifest["bank_sha256"],
            "d7_result_sha256": sha256(D7_RESULT),
            "d8_independent_verification_sha256": sha256(D8_VERIFICATION),
            "seed": SEED,
        },
        "physical": {
            "registered_experts_per_layer": PREFIX,
            "cold_escape_experts_per_layer": COLD_END - COLD_BEGIN,
            "registered_ranges": len(hosts),
            "registered_bytes": LAYERS * PREFIX * EXPERT_BYTES,
            "registered_gib": LAYERS * PREFIX * EXPERT_BYTES / 2**30,
            "staging_hbm_bytes": TOKEN_BYTES,
            "cold_escape_hbm_bytes": TOKEN_BYTES,
            "available_ram_before_registration": available_before,
            "available_ram_after_registration": available_after_registration,
            "available_ram_after_unregister": available_after_unregister,
        },
        "protocol": {
            "stage_blocks": STAGE_BLOCKS,
            "stage_threads": STAGE_THREADS,
            "q5_width": 8,
            "warmups_per_case": WARMUPS,
            "validation_rounds_per_case": VALIDATION_ROUNDS,
            "test_rounds_per_case": TEST_ROUNDS,
            "validation_p50_limits_ms": VALIDATION_P50_LIMITS,
            "test_p95_limits_ms": TEST_P95_LIMITS,
            "pass_timing": "inclusive wall clock; CUDA events diagnostic only",
        },
        **payload,
        "error": error,
        "unregister_failures": unregister_failures,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Exact synthetic 499+13 active-plane bridge only; no 512/full-bank registration, stable capacity, real checkpoint, natural traffic, quality, physical dense shell, end-to-end tok/s or endurance claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    case_lines = []
    for name in CASE_NAMES:
        validation_stats = result.get("validation", {}).get("cases", {}).get(name, {}).get("wall_stats")
        test_stats = result.get("test", {}).get("cases", {}).get(name, {}).get("wall_stats")
        case_lines.append(
            f"| {name} | {validation_stats.get('p50') if validation_stats else '—'} | "
            f"{test_stats.get('p50') if test_stats else '—'} | {test_stats.get('p95') if test_stats else '—'} |"
        )
    REPORT.write_text(
        "# PORT80B-D9 — capacity-aware 499+13 bank bridge report\n\n"
        f"Verdict: **{result['status']}**. Primary pass: **{primary}**. Strong pass: **{strong}**. "
        f"Clean unregister: **{gates.get('clean_unregister_48_ranges')}**.\n\n"
        "| case | validation wall p50 ms | test wall p50 ms | test wall p95 ms |\n"
        "|---|---:|---:|---:|\n" + "\n".join(case_lines) + "\n\n"
        "The differentiated header oracle includes positive all-hot/mixed/all-cold checks and deliberate wrong-expert/wrong-layer controls. Pass/fail timing is inclusive wall time; CUDA events are diagnostic.\n\n"
        f"Claim boundary: {result['claim_boundary']}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_pass": primary,
                "strong_pass": strong,
                "validation_open": result.get("validation", {}).get("open"),
                "validation": {name: result.get("validation", {}).get("cases", {}).get(name, {}).get("wall_stats") for name in CASE_NAMES},
                "test": {name: result.get("test", {}).get("cases", {}).get(name, {}).get("wall_stats") for name in CASE_NAMES},
                "integrity": result.get("integrity"),
                "correctness": result.get("correctness"),
                "error": error,
                "unregister_failures": unregister_failures,
                "available_ram": result["physical"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("compile", "run"), required=True)
    args = parser.parse_args()
    if args.phase == "compile":
        compile_phase()
    else:
        run_phase()


if __name__ == "__main__":
    main()
