"""Phase72 continuous cross-H4 prefetch oracle on the real Ornith trace."""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase71_ornith_trace_prefetch_oracle import (
    CUDA_SOURCE,
    EXPERT_BYTES,
    FLOOR_MS,
    HEAD_MS,
    LAYER_MS,
    PHASE70,
    RESIDUAL_MS,
    _calibrate_wait,
    _measure,
    _warm_counts,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase72"
PREREG = REPO / "pro_research" / "S100_PHASE72_ORNITH_ROLLING_PREFETCH_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase72_ornith_rolling_prefetch.py"
PHASE71 = REPO / "pro_research" / "results" / "s100_phase71" / "S100_PHASE71_ORNITH_TRACE_PREFETCH_ORACLE.json"
LEADS = (2, 4, 8, 16)
TARGET_H4_MS = 4000.0 / 65.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE72_ORNITH_ROLLING_PREFETCH.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase72_ornith_rolling_prefetch",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    cp = None
    pinned_handle = None
    try:
        import cupy as cp_module

        cp = cp_module
        phase70 = json.loads(PHASE70.read_text("utf-8"))
        phase71 = json.loads(PHASE71.read_text("utf-8"))
        counts_by_policy = {
            policy: _warm_counts(phase70, policy) for policy in ("lru52", "belady52")
        }
        max_groups = max(
            count for blocks in counts_by_policy.values() for row in blocks for count in row
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
        pinned.fill(0x5A)
        source_base = int(pinned.ctypes.data)

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
        compute_ms_h4 = float(compute_timing["p50"]) / n_blocks

        def source_ptr(task: int) -> int:
            return source_base + (task % rotations) * max_copy_bytes

        def enqueue_copy(stream, destination, task: int, groups: int) -> None:
            nbytes = int(groups * EXPERT_BYTES)
            if nbytes:
                cp.cuda.runtime.memcpyAsync(
                    int(destination.data.ptr), source_ptr(task), nbytes,
                    cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
                )

        records = {}
        for policy, block_counts in counts_by_policy.items():
            flat_counts = [count for row in block_counts for count in row]
            n_tasks = len(flat_counts)
            arms = {}
            for lead in LEADS:
                destinations = [cp.empty(max_copy_bytes, dtype=cp.uint8) for _ in range(lead)]
                copy_stream = cp.cuda.Stream(non_blocking=True)
                copied = [cp.cuda.Event() for _ in range(lead)]
                consumed = [cp.cuda.Event() for _ in range(lead)]
                fork = cp.cuda.Event()

                def run_epoch(
                    lead_value=lead,
                    destination_ring=destinations,
                    auxiliary=copy_stream,
                    copied_events=copied,
                    consumed_events=consumed,
                    fork_event=fork,
                ) -> None:
                    main_stream = cp.cuda.get_current_stream()
                    fork_event.record(main_stream)
                    with auxiliary:
                        auxiliary.wait_event(fork_event)
                        for task in range(min(lead_value, n_tasks)):
                            slot = task % lead_value
                            enqueue_copy(
                                auxiliary, destination_ring[slot], task, flat_counts[task]
                            )
                            copied_events[slot].record(auxiliary)

                    for task, groups in enumerate(flat_counts):
                        del groups
                        slot = task % lead_value
                        main_stream.wait_event(copied_events[slot])
                        wait_layer()
                        consumed_events[slot].record(main_stream)
                        future = task + lead_value
                        if future < n_tasks:
                            with auxiliary:
                                auxiliary.wait_event(consumed_events[slot])
                                enqueue_copy(
                                    auxiliary, destination_ring[slot], future,
                                    flat_counts[future],
                                )
                                copied_events[slot].record(auxiliary)
                        if task % 40 == 39:
                            wait_head()

                timing = _measure(cp, run_epoch, args.warmup, args.reps)
                raw_ms_h4 = float(timing["p50"]) / n_blocks
                exposed = raw_ms_h4 - compute_ms_h4
                normalized = FLOOR_MS + exposed
                arms[f"lead{lead}"] = {
                    "timing_epoch_ms": timing,
                    "raw_ms_h4": raw_ms_h4,
                    "exposed_tail_ms_h4": exposed,
                    "floor_normalized_ms_h4": normalized,
                    "floor_normalized_tok_s": 4000.0 / normalized,
                }
                del run_epoch, destinations, copied, consumed, fork, copy_stream
                cp.get_default_memory_pool().free_all_blocks()

            selected_name, selected = min(
                arms.items(), key=lambda item: item[1]["floor_normalized_ms_h4"]
            )
            phase71_serial = float(
                phase71["records"][policy]["summary"]["serial_ms_h4"]
            )
            records[policy] = {
                "trace": {
                    "blocks": n_blocks,
                    "layer_tasks": n_tasks,
                    "mean_groups_per_h4": float(np.mean([sum(row) for row in block_counts])),
                },
                "phase71_serial_ms_h4": phase71_serial,
                "arms": arms,
                "selected": {"arm": selected_name, **selected},
            }

        lru_selected = records["lru52"]["selected"]
        belady_selected = records["belady52"]["selected"]
        calibration_ok = all((
            abs(float(layer_calibration["p50"]) - LAYER_MS) / LAYER_MS <= 0.05,
            abs(float(head_calibration["p50"]) - HEAD_MS) / HEAD_MS <= 0.05,
        ))
        gates = {
            "P72_G1_working_set_and_calibration": (
                int(pinned.nbytes) >= 4 * l2_bytes and calibration_ok
            ),
            "P72_G2_all_rolling_arms_2pct_faster_than_serial": all(
                arm["raw_ms_h4"] <= 0.98 * record["phase71_serial_ms_h4"]
                for record in records.values() for arm in record["arms"].values()
            ),
            "P72_G3_belady_normalized_below_65_boundary": (
                belady_selected["floor_normalized_ms_h4"] <= TARGET_H4_MS
            ),
            "P72_G4_lru_normalized_below_65_boundary": (
                lru_selected["floor_normalized_ms_h4"] <= TARGET_H4_MS
            ),
            "P72_G5_lru_belady_exposed_gap_le_0_75ms": (
                lru_selected["exposed_tail_ms_h4"]
                - belady_selected["exposed_tail_ms_h4"] <= 0.75
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "targets_ms": {
                "floor": FLOOR_MS,
                "residual": RESIDUAL_MS,
                "boundary_65": TARGET_H4_MS,
            },
            "source": {
                "expert_bytes": EXPERT_BYTES,
                "max_groups": max_groups,
                "max_copy_bytes": max_copy_bytes,
                "rotations": rotations,
                "working_set_bytes": int(pinned.nbytes),
                "working_set_over_l2": int(pinned.nbytes) / max(l2_bytes, 1),
            },
            "calibration": {
                "layer": layer_calibration,
                "head": head_calibration,
                "compute_epoch": compute_timing,
                "compute_ms_h4": compute_ms_h4,
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, PHASE70, PHASE71))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "compute_ms_h4": (payload.get("calibration") or {}).get("compute_ms_h4"),
        "selected": {
            policy: record["selected"] for policy, record in (payload.get("records") or {}).items()
        },
        "arms": {
            policy: {
                arm: {
                    key: row[key] for key in (
                        "raw_ms_h4", "exposed_tail_ms_h4",
                        "floor_normalized_ms_h4", "floor_normalized_tok_s",
                    )
                }
                for arm, row in record["arms"].items()
            }
            for policy, record in (payload.get("records") or {}).items()
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
