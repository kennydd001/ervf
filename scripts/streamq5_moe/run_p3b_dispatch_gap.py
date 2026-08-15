from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p3a_integrated_expert import (
    BANK_DIR,
    CACHE_BYTES,
    EXPERT_BYTES,
    HEADER,
    HEADER_BYTES,
    KV_BYTES,
    LAYERS,
    MIN_SCRATCH,
    RECORD_BYTES,
    TRUNK_BYTES,
    bases,
    kernels,
    launch_layer,
    load_routes,
)


R = ROOT / "reports/streamq5_moe"
PREREG = R / "P3B_DISPATCH_GAP_PREREGISTRATION.md"
INPUT_LOCK = R / "p3b_dispatch_input_lock.json"
EVALUATOR_LOCK = R / "p3b_dispatch_evaluator_lock.json"
P1D_VERIFY = R / "p1d_physical_bank_verification.json"
P3A_ROUTE_CAPTURE = R / "p3a_route_capture_result.json"

NOOP_SOURCE = r'''
extern "C" __global__ void dispatch_noop() { }
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def copy_state(stream: cp.cuda.Stream, state: cp.ndarray, initial: cp.ndarray) -> None:
    cp.cuda.runtime.memcpyAsync(
        state.data.ptr,
        initial.data.ptr,
        state.nbytes,
        cp.cuda.runtime.memcpyDeviceToDevice,
        stream.ptr,
    )


def launch_token(
    stream: cp.cuda.Stream,
    kernel_set,
    state: cp.ndarray,
    cache: cp.ndarray,
    slot_table: cp.ndarray,
    gate: cp.ndarray,
    up: cp.ndarray,
    down: cp.ndarray,
) -> None:
    for layer in range(LAYERS):
        launch_layer(
            kernel_set,
            stream,
            state,
            cache,
            slot_table[layer],
            gate,
            up,
            down,
        )


def capture_graph(stream: cp.cuda.Stream, enqueue) -> cp.cuda.graph.Graph:
    stream.begin_capture()
    enqueue()
    graph = stream.end_capture()
    graph.upload(stream)
    stream.synchronize()
    return graph


def measure(stream: cp.cuda.Stream, enqueue, warmups: int, iterations: int) -> dict:
    for _ in range(warmups):
        enqueue()
    stream.synchronize()
    host_ms: list[float] = []
    event_ms: list[float] = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(stream)
        enqueue()
        end.record(stream)
        end.synchronize()
        host_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        event_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
    return {"iterations": iterations, "host_ms": host_ms, "event_ms": event_ms,
            "host_stats": stats(host_ms), "event_stats": stats(event_ms)}


def load_selected_experts(
    route: np.ndarray,
    cache,
    cache_view: cp.ndarray,
    stream: cp.cuda.Stream,
) -> tuple[np.ndarray, dict]:
    layer_bases = bases()
    slot_table = np.empty((LAYERS, 8), dtype=np.int32)
    pinned = cp.cuda.alloc_pinned_memory(EXPERT_BYTES)
    host = np.frombuffer(pinned, dtype=np.uint8, count=EXPERT_BYTES)
    loaded_records = 0
    loaded_bytes = 0
    selected_digest = hashlib.sha256()
    for layer in range(LAYERS):
        path = BANK_DIR / f"layer_{layer:02d}.q5bin"
        with path.open("rb", buffering=8 * 2**20) as handle:
            for local_slot, expert_value in enumerate(route[layer]):
                expert = int(expert_value)
                handle.seek(expert * EXPERT_BYTES)
                if handle.readinto(host) != EXPERT_BYTES:
                    raise RuntimeError("short physical expert read")
                for projection in range(3):
                    offset = projection * RECORD_BYTES
                    fields = HEADER.unpack(bytes(host[offset:offset + HEADER_BYTES]))
                    if (fields[0] != b"SQ5M" or fields[2] != layer
                            or fields[3] != expert or fields[4] != projection):
                        raise ValueError("physical expert header mismatch")
                    loaded_records += 1
                selected_digest.update(memoryview(host))
                absolute_slot = layer_bases[layer] + local_slot
                cp.cuda.runtime.memcpyAsync(
                    cache.ptr + absolute_slot * EXPERT_BYTES,
                    pinned.ptr,
                    EXPERT_BYTES,
                    cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
                stream.synchronize()
                slot_table[layer, local_slot] = absolute_slot
                loaded_bytes += EXPERT_BYTES
    return slot_table, {
        "experts_loaded": LAYERS * 8,
        "records_header_checked": loaded_records,
        "bytes_loaded": loaded_bytes,
        "selected_expert_bytes_sha256": selected_digest.hexdigest(),
        "cache_view_bytes": int(cache_view.nbytes),
    }


def run(phase: str) -> dict:
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    evaluator = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    if sha256(PREREG) != lock["preregistration_sha256"]:
        raise ValueError("P3B preregistration mismatch")
    if sha256(P1D_VERIFY) != lock["p1d_verification_sha256"]:
        raise ValueError("P3B P1D verification mismatch")
    if sha256(P3A_ROUTE_CAPTURE) != lock["p3a_route_capture_sha256"]:
        raise ValueError("P3B route capture mismatch")
    if sha256(INPUT_LOCK) != evaluator["input_lock_sha256"]:
        raise ValueError("P3B input lock mismatch")
    if sha256(Path(__file__)) != evaluator["evaluator_sha256"]:
        raise ValueError("P3B evaluator mismatch")

    output = R / f"p3b_dispatch_gap_{phase}.json"
    report = R / f"P3B_DISPATCH_GAP_{phase.upper()}.md"
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite P3B output")
    if phase == "test":
        validation_path = R / "p3b_dispatch_gap_validation.json"
        if (not validation_path.exists()
                or json.loads(validation_path.read_text(encoding="utf-8"))["status"]
                != "p3b_validation_pass_test_authorized"):
            raise RuntimeError("P3B test is not authorized")

    routes, route_hashes = load_routes()
    case = lock["route_cases"][phase]
    selected_route = routes[case["domain"]][case["token"]]
    kernel_set = kernels()
    noop = cp.RawKernel(NOOP_SOURCE, "dispatch_noop", options=("--std=c++11",))
    cp.get_default_memory_pool().free_all_blocks()
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    cache = cp.cuda.alloc(CACHE_BYTES)
    cache_view = cp.ndarray((CACHE_BYTES,), dtype=cp.uint8, memptr=cache)
    trunk = cp.cuda.alloc(TRUNK_BYTES)
    kv = cp.cuda.alloc(KV_BYTES)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        cp.cuda.runtime.memsetAsync(cache.ptr, 0, CACHE_BYTES, stream.ptr)
        cp.cuda.runtime.memsetAsync(trunk.ptr, 0, TRUNK_BYTES, stream.ptr)
        cp.cuda.runtime.memsetAsync(kv.ptr, 0, KV_BYTES, stream.ptr)
    stream.synchronize()
    free_after_alloc, _ = cp.cuda.runtime.memGetInfo()

    slot_table_host, physical = load_selected_experts(
        selected_route, cache, cache_view, stream
    )
    slot_table = cp.asarray(slot_table_host)
    rng = np.random.default_rng(lock["initial_state_seed"])
    initial = cp.asarray(rng.standard_normal(2048, dtype=np.float32))
    state = cp.empty(2048, dtype=cp.float32)
    gate = cp.empty(6144, dtype=cp.float32)
    up = cp.empty(6144, dtype=cp.float32)
    down = cp.empty(16384, dtype=cp.float32)

    def eager_enqueue() -> None:
        copy_state(stream, state, initial)
        launch_token(stream, kernel_set, state, cache_view, slot_table, gate, up, down)

    graph = capture_graph(stream, eager_enqueue)

    eager_enqueue()
    stream.synchronize()
    eager_state = cp.asnumpy(state)
    graph.launch(stream)
    stream.synchronize()
    graph_state = cp.asnumpy(state)
    exact_output = bool(np.array_equal(eager_state, graph_state))
    max_abs = float(np.max(np.abs(eager_state - graph_state)))
    relative_l2 = float(
        np.linalg.norm(eager_state - graph_state)
        / max(float(np.linalg.norm(eager_state)), 1e-30)
    )

    iterations = int(lock["measured_iterations"][phase])
    warmups = 2 if phase == "smoke" else int(lock["warmup_iterations"])
    eager = measure(stream, eager_enqueue, warmups, iterations)
    graph_result = measure(stream, lambda: graph.launch(stream), warmups, iterations)

    def noop_eager_enqueue() -> None:
        for _ in range(lock["layers"] * lock["launches_per_layer"]):
            noop((1,), (1,), (), stream=stream)

    noop_graph = capture_graph(stream, noop_eager_enqueue)
    noop_eager = measure(stream, noop_eager_enqueue, warmups, iterations)
    noop_graph_result = measure(
        stream, lambda: noop_graph.launch(stream), warmups, iterations
    )

    host_ratio = graph_result["host_stats"]["p50"] / eager["host_stats"]["p50"]
    event_ratio = graph_result["event_stats"]["p50"] / eager["event_stats"]["p50"]
    gates = {
        "physical_experts_loaded": physical["experts_loaded"] == LAYERS * 8
        and physical["records_header_checked"] == LAYERS * 8 * 3
        and physical["bytes_loaded"] == LAYERS * 8 * EXPERT_BYTES,
        "device_co_resident_and_scratch": free_after_alloc >= MIN_SCRATCH,
        "exact_output_match": exact_output,
        "finite_output": bool(np.isfinite(eager_state).all())
        and bool(np.isfinite(graph_state).all()),
        "finite_timings": all(
            np.isfinite(row).all()
            for row in (
                eager["host_ms"], eager["event_ms"],
                graph_result["host_ms"], graph_result["event_ms"],
                noop_eager["host_ms"], noop_eager["event_ms"],
                noop_graph_result["host_ms"], noop_graph_result["event_ms"],
            )
        ),
        "graph_host_p50_ratio_le_0_90": host_ratio
        <= lock["gates"]["graph_to_eager_host_p50_max"],
        "graph_event_p50_ratio_ge_0_90": event_ratio
        >= lock["gates"]["graph_to_eager_event_p50_min"],
        "graph_event_p50_ratio_le_1_05": event_ratio
        <= lock["gates"]["graph_to_eager_event_p50_max"],
    }
    passed = all(gates.values())
    if phase == "smoke":
        status = "p3b_smoke_complete"
    elif phase == "validation":
        status = (
            "p3b_validation_pass_test_authorized"
            if passed else "p3b_validation_closed_test_unopened"
        )
    else:
        status = "p3b_dispatch_gap_pass" if passed else "p3b_dispatch_gap_closed"

    result = {
        "kind": "streamq5_moe_p3b_dispatch_gap",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "status": status,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "input_lock_sha256": sha256(INPUT_LOCK),
            "evaluator_lock_sha256": sha256(EVALUATOR_LOCK),
            "evaluator_sha256": sha256(Path(__file__)),
            "p1d_verification_sha256": sha256(P1D_VERIFY),
            "p3a_route_capture_sha256": sha256(P3A_ROUTE_CAPTURE),
            "route_artifact_sha256": route_hashes,
            "case": case,
            "selected_route": selected_route.tolist(),
        },
        "environment": {
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "total_vram_bytes": int(total_vram),
            "free_before_bytes": int(free_before),
            "free_after_alloc_bytes": int(free_after_alloc),
        },
        "workload": {
            "layers": LAYERS,
            "experts_per_layer": 8,
            "kernel_launches_per_layer": 4,
            "kernel_launches_per_token": LAYERS * 4,
            "cache_misses_timed": 0,
            "warmups": warmups,
            "measured_iterations": iterations,
        },
        "physical": physical,
        "correctness": {
            "exact_output_match": exact_output,
            "max_abs": max_abs,
            "relative_l2": relative_l2,
        },
        "eager": eager,
        "graph": graph_result,
        "noop_eager": noop_eager,
        "noop_graph": noop_graph_result,
        "ratios": {
            "graph_to_eager_host_p50": host_ratio,
            "graph_to_eager_event_p50": event_ratio,
            "noop_graph_to_eager_host_p50": (
                noop_graph_result["host_stats"]["p50"]
                / noop_eager["host_stats"]["p50"]
            ),
        },
        "gates": gates,
        "claim_boundary": (
            "All-hit physical expert dispatch audit only; no H2D misses, "
            "attention/router/trunk compute, KV mutation, LM head, sampling, "
            "autoregressive routing, or full-model tok/s."
        ),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(
        "# STREAMQ5-MoE P3B — dispatch gap " + phase + "\n\n"
        f"Status: **{status}**.\n\n"
        f"Eager host/event p50: {eager['host_stats']['p50']:.4f}/"
        f"{eager['event_stats']['p50']:.4f} ms. Graph: "
        f"{graph_result['host_stats']['p50']:.4f}/"
        f"{graph_result['event_stats']['p50']:.4f} ms. "
        f"Hostratio: {host_ratio:.4f}.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "validation", "test"), required=True)
    args = parser.parse_args()
    result = run(args.phase)
    print(json.dumps({
        "status": result["status"],
        "eager": {"host": result["eager"]["host_stats"], "event": result["eager"]["event_stats"]},
        "graph": {"host": result["graph"]["host_stats"], "event": result["graph"]["event_stats"]},
        "ratios": result["ratios"],
        "gates": result["gates"],
    }, indent=2))
    if result["status"].endswith("closed") or "closed_test" in result["status"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
