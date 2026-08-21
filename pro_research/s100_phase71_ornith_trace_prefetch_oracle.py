"""Phase71 real-trace H2D copy-engine prefetch oracle for Ornith."""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import REPO, environment_snapshot, percentiles, utc_now, write_json_atomic


RESULTS = REPO / "pro_research" / "results" / "s100_phase71"
PREREG = REPO / "pro_research" / "S100_PHASE71_ORNITH_TRACE_PREFETCH_ORACLE_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase71_ornith_trace_prefetch_oracle.py"
PHASE70 = REPO / "pro_research" / "results" / "s100_phase70" / "S100_PHASE70_ORNITH_REAL_TRACE.json"
EXPERT_BYTES = 1_769_472
FLOOR_MS = 60.095487602
HEAD_MS = 1.574815989
LAYER_MS = (FLOOR_MS - HEAD_MS) / 40.0
RESIDUAL_MS = 4000.0 / 65.0 - FLOOR_MS
LEADS = (1, 2, 4, 8)


CUDA_SOURCE = r"""
extern "C" __global__ void p71_wait_cycles(
    const unsigned long long cycles,
    unsigned long long * sink)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned long long begin = clock64();
    while (clock64() - begin < cycles) {
        __nanosleep(256);
    }
    sink[0] ^= clock64();
}
"""


def _measure(cp, function: Callable[[], None], warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    cp.cuda.get_current_stream().synchronize()
    samples = []
    for _ in range(repeats):
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        begin.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(begin, end)))
    return percentiles(samples)


def _calibrate_wait(cp, kernel, sink, target_ms: float, clock_khz: int) -> tuple[int, dict[str, Any]]:
    cycles = max(1, int(round(clock_khz * target_ms)))

    def launch() -> None:
        kernel((1,), (1,), (np.uint64(cycles), sink))

    for _ in range(3):
        timing = _measure(cp, launch, 2, 5)
        measured = max(float(timing["p50"]), 1.0e-6)
        cycles = max(1, int(round(cycles * target_ms / measured)))
    timing = _measure(cp, launch, 3, 11)
    return cycles, timing


def _warm_counts(payload: dict[str, Any], policy: str) -> list[list[int]]:
    blocks = payload["cache"]["long"][policy]["h4_miss_groups"]["blocks"]
    selected = [
        [int(value) for value in block["unique_miss_groups_by_layer"]]
        for block in blocks
        if int(block["begin_token"]) >= 16
    ]
    if len(selected) != 28 or any(len(row) != 40 for row in selected):
        raise ValueError(f"unexpected {policy} warm trace shape")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE71_ORNITH_TRACE_PREFETCH_ORACLE.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase71_ornith_trace_prefetch_oracle",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "targets_ms": {
            "all_hot_floor_h4": FLOOR_MS,
            "head_h4": HEAD_MS,
            "per_layer_h4": LAYER_MS,
            "residual_to_65_h4": RESIDUAL_MS,
        },
    }
    cp = None
    pinned_handle = None
    try:
        import cupy as cp_module

        cp = cp_module
        phase70 = json.loads(PHASE70.read_text("utf-8"))
        counts_by_policy = {
            "lru52": _warm_counts(phase70, "lru52"),
            "belady52": _warm_counts(phase70, "belady52"),
        }
        max_groups = max(
            value for blocks in counts_by_policy.values() for row in blocks for value in row
        )
        max_copy_bytes = max_groups * EXPERT_BYTES
        props = cp.cuda.runtime.getDeviceProperties(0)
        l2_bytes = int(props.get("l2CacheSize", 0))
        clock_khz = int(props.get("clockRate", 0))
        rotations = max(2, math.ceil((4 * l2_bytes) / max_copy_bytes) + 1)
        pinned_handle = cp.cuda.alloc_pinned_memory(max_copy_bytes * rotations)
        pinned = np.frombuffer(
            pinned_handle, dtype=np.uint8, count=max_copy_bytes * rotations
        )
        pinned.fill(0xA5)
        source_base = int(pinned.ctypes.data)
        working_set_bytes = int(pinned.nbytes)

        module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++17",),
            name_expressions=("p71_wait_cycles",),
        )
        wait_kernel = module.get_function("p71_wait_cycles")
        sink = cp.zeros(1, dtype=cp.uint64)
        layer_cycles, layer_calibration = _calibrate_wait(
            cp, wait_kernel, sink, LAYER_MS, clock_khz
        )
        head_cycles, head_calibration = _calibrate_wait(
            cp, wait_kernel, sink, HEAD_MS, clock_khz
        )

        def wait_layer() -> None:
            wait_kernel((1,), (1,), (np.uint64(layer_cycles), sink))

        def wait_head() -> None:
            wait_kernel((1,), (1,), (np.uint64(head_cycles), sink))

        n_blocks = 28

        def run_compute_epoch() -> None:
            for _block in range(n_blocks):
                for _layer in range(40):
                    wait_layer()
                wait_head()

        compute_timing = _measure(cp, run_compute_epoch, args.warmup, args.reps)
        compute_per_h4 = float(compute_timing["p50"]) / n_blocks

        def source_ptr(block: int, layer: int) -> int:
            rotation = (block * 40 + layer) % rotations
            return source_base + rotation * max_copy_bytes

        def enqueue_copy(stream, destination, block: int, layer: int, groups: int) -> None:
            nbytes = int(groups * EXPERT_BYTES)
            if nbytes:
                cp.cuda.runtime.memcpyAsync(
                    int(destination.data.ptr), source_ptr(block, layer), nbytes,
                    cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
                )

        records = {}
        for policy, trace_counts in counts_by_policy.items():
            policy_record: dict[str, Any] = {
                "trace": {
                    "blocks": len(trace_counts),
                    "sum_groups": int(sum(sum(row) for row in trace_counts)),
                    "mean_groups_per_h4": float(np.mean([sum(row) for row in trace_counts])),
                    "max_groups_one_layer": int(max(max(row) for row in trace_counts)),
                    "mean_bytes_per_h4": float(np.mean([
                        sum(row) * EXPERT_BYTES for row in trace_counts
                    ])),
                },
                "arms": {},
            }

            serial_destination = cp.empty(max_copy_bytes, dtype=cp.uint8)

            def run_serial_epoch() -> None:
                stream = cp.cuda.get_current_stream()
                for block, row in enumerate(trace_counts):
                    for layer, groups in enumerate(row):
                        enqueue_copy(stream, serial_destination, block, layer, groups)
                        wait_layer()
                    wait_head()

            serial = _measure(cp, run_serial_epoch, args.warmup, args.reps)
            policy_record["arms"]["serial"] = serial
            del serial_destination
            cp.get_default_memory_pool().free_all_blocks()

            for lead in LEADS:
                destinations = [cp.empty(max_copy_bytes, dtype=cp.uint8) for _ in range(lead)]
                copy_stream = cp.cuda.Stream(non_blocking=True)
                copied = [cp.cuda.Event() for _ in range(40)]
                consumed = [cp.cuda.Event() for _ in range(lead)]
                fork = cp.cuda.Event()

                def run_overlap_epoch(
                    lead_value=lead,
                    destination_ring=destinations,
                    auxiliary=copy_stream,
                    copied_events=copied,
                    consumed_events=consumed,
                    fork_event=fork,
                ) -> None:
                    main_stream = cp.cuda.get_current_stream()
                    for block, row in enumerate(trace_counts):
                        fork_event.record(main_stream)
                        with auxiliary:
                            auxiliary.wait_event(fork_event)
                            for layer in range(min(lead_value, 40)):
                                enqueue_copy(
                                    auxiliary, destination_ring[layer % lead_value],
                                    block, layer, row[layer],
                                )
                                copied_events[layer].record(auxiliary)
                        for layer in range(40):
                            slot = layer % lead_value
                            main_stream.wait_event(copied_events[layer])
                            wait_layer()
                            consumed_events[slot].record(main_stream)
                            future = layer + lead_value
                            if future < 40:
                                with auxiliary:
                                    auxiliary.wait_event(consumed_events[slot])
                                    enqueue_copy(
                                        auxiliary, destination_ring[slot],
                                        block, future, row[future],
                                    )
                                    copied_events[future].record(auxiliary)
                        wait_head()

                timing = _measure(cp, run_overlap_epoch, args.warmup, args.reps)
                policy_record["arms"][f"lead{lead}"] = timing
                del run_overlap_epoch, destinations, copied, consumed, fork, copy_stream
                cp.get_default_memory_pool().free_all_blocks()

            serial_per_h4 = float(serial["p50"]) / n_blocks
            policy_record["summary"] = {
                "compute_only_ms_h4": compute_per_h4,
                "serial_ms_h4": serial_per_h4,
                "serial_increment_ms_h4": serial_per_h4 - compute_per_h4,
                "arms": {},
            }
            for lead in LEADS:
                timing = policy_record["arms"][f"lead{lead}"]
                arm_ms = float(timing["p50"]) / n_blocks
                policy_record["summary"]["arms"][f"lead{lead}"] = {
                    "ms_h4": arm_ms,
                    "exposed_tail_ms_h4": arm_ms - compute_per_h4,
                    "speedup_vs_serial": serial_per_h4 / arm_ms,
                    "effective_tok_s": 4000.0 / arm_ms,
                }
            records[policy] = policy_record

        calibration_ok = all((
            abs(float(layer_calibration["p50"]) - LAYER_MS) / LAYER_MS <= 0.05,
            abs(float(head_calibration["p50"]) - HEAD_MS) / HEAD_MS <= 0.05,
        ))
        gates = {
            "P71_G1_pinned_working_set_and_ring_bounds": (
                working_set_bytes >= 4 * l2_bytes
                and max_groups * EXPERT_BYTES <= max_copy_bytes
            ),
            "P71_G2_wait_calibration_within_5pct": calibration_ok,
            "P71_G3_lookahead_no_slower_than_serial": all(
                arm["ms_h4"] <= 1.02 * record["summary"]["serial_ms_h4"]
                for record in records.values()
                for arm in record["summary"]["arms"].values()
            ),
            "P71_G4_lru_exposed_tail_within_residual": any(
                arm["exposed_tail_ms_h4"] <= RESIDUAL_MS
                for arm in records["lru52"]["summary"]["arms"].values()
            ),
            "P71_G5_belady_exposed_tail_within_residual": any(
                arm["exposed_tail_ms_h4"] <= RESIDUAL_MS
                for arm in records["belady52"]["summary"]["arms"].values()
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "gpu": {
                "name": props.get("name", b"").decode(errors="replace")
                if isinstance(props.get("name"), bytes) else str(props.get("name")),
                "l2_bytes": l2_bytes,
                "clock_khz": clock_khz,
            },
            "source": {
                "expert_bytes": EXPERT_BYTES,
                "max_groups": max_groups,
                "max_copy_bytes": max_copy_bytes,
                "rotations": rotations,
                "working_set_bytes": working_set_bytes,
                "working_set_over_l2": working_set_bytes / max(l2_bytes, 1),
            },
            "calibration": {
                "layer_cycles": layer_cycles,
                "layer": layer_calibration,
                "head_cycles": head_cycles,
                "head": head_calibration,
                "compute_epoch": compute_timing,
                "compute_only_ms_h4": compute_per_h4,
            },
            "records": records,
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as error:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    finally:
        if cp is not None:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, PHASE70))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "source": payload.get("source"),
        "calibration": {
            "layer_p50": ((payload.get("calibration") or {}).get("layer") or {}).get("p50"),
            "head_p50": ((payload.get("calibration") or {}).get("head") or {}).get("p50"),
            "compute_only_ms_h4": (payload.get("calibration") or {}).get("compute_only_ms_h4"),
        },
        "summaries": {
            policy: record["summary"] for policy, record in (payload.get("records") or {}).items()
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
