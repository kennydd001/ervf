"""Phase73 segmented rolling H2D under real Ornith expert compute."""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase59_ornith_bulk_expert import _load_experts, _stack
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1
from s100_phase71_ornith_trace_prefetch_oracle import (
    CUDA_SOURCE,
    FLOOR_MS,
    HEAD_MS,
    LAYER_MS,
    PHASE70,
    _calibrate_wait,
    _measure,
    _warm_counts,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase73"
PREREG = REPO / "pro_research" / "S100_PHASE73_ORNITH_SEGMENTED_REALCOMPUTE_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase73_ornith_segmented_realcompute.py"
TARGET_H4_MS = 4000.0 / 65.0
PHASE59_HOT_MS = 0.5609599948
LEADS = (2, 4, 8)
SEGMENTS = (
    ("gate_codes", 524_288),
    ("gate_scales", 65_536),
    ("up_codes", 524_288),
    ("up_scales", 65_536),
    ("down_codes", 524_288),
    ("down_scales", 65_536),
)
EXPERT_BYTES = sum(size for _name, size in SEGMENTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE73_ORNITH_SEGMENTED_REALCOMPUTE.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase73_ornith_segmented_realcompute",
        "status": "started",
        "started_utc": utc_now(),
        "snapshot": str(args.snapshot.resolve()),
        "layer": int(args.layer),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    cp = None
    pinned_handles = []
    try:
        import cupy as cp_module
        import sys

        src = REPO / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        cp = cp_module
        phase70 = json.loads(PHASE70.read_text("utf-8"))
        counts_by_policy = {
            policy: _warm_counts(phase70, policy) for policy in ("lru52", "belady52")
        }
        max_groups = max(
            count for blocks in counts_by_policy.values() for row in blocks for count in row
        )
        props = cp.cuda.runtime.getDeviceProperties(0)
        l2_bytes = int(props.get("l2CacheSize", 0))
        clock_khz = int(props.get("clockRate", 0))
        max_copy_bytes = max_groups * EXPERT_BYTES
        rotations = max(2, math.ceil((4 * l2_bytes) / max_copy_bytes) + 1)

        pinned_sources = {}
        for segment_index, (name, bytes_per_expert) in enumerate(SEGMENTS):
            stride = max_groups * bytes_per_expert
            handle = cp.cuda.alloc_pinned_memory(stride * rotations)
            pinned_handles.append(handle)
            view = np.frombuffer(handle, dtype=np.uint8, count=stride * rotations)
            view.fill((0x31 + 17 * segment_index) & 0xFF)
            pinned_sources[name] = {
                "base": int(view.ctypes.data),
                "stride": stride,
                "bytes_per_expert": bytes_per_expert,
            }
        working_set_bytes = sum(source["stride"] * rotations for source in pinned_sources.values())

        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        experts, _ = _load_experts(snapshot, index["weight_map"], args.layer, 32)
        lookup = FusedNVFP4()
        bulk = OrnithNVFP4BulkM1()
        routed = {}
        for projection in ("gate", "up", "down"):
            routed[projection] = {
                "codes": cp.asarray(_stack(experts, projection, "codes")),
                "scales": cp.asarray(_stack(experts, projection, "scales")),
                "global": cp.asarray(np.asarray([
                    row[projection]["global_scale"] for row in experts
                ], dtype=np.float32)),
            }
        rng = np.random.default_rng(73000000 + args.layer)
        x = cp.asarray(rng.standard_normal((32, 2048), dtype=np.float32))
        gate = cp.empty((32, 512), dtype=cp.float32)
        up = cp.empty_like(gate)
        activation = cp.empty_like(gate)
        output = cp.empty((32, 2048), dtype=cp.float32)

        def run_hot() -> None:
            for projection, target in (("gate", gate), ("up", up)):
                row = routed[projection]
                bulk.nvfp4(
                    row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                    x, target, row["global"], 32, 512, 2048,
                )
            bulk.swiglu(gate, up, activation, 32)
            row = routed["down"]
            bulk.nvfp4(
                row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                activation, output, row["global"], 32, 2048, 512,
            )

        hot_timing = _measure(cp, run_hot, 10, 31)
        hot_ms = float(hot_timing["p50"])
        remaining_ms = LAYER_MS - hot_ms
        if remaining_ms <= 0:
            raise RuntimeError(f"real hot kernel {hot_ms:.6f} ms exceeds layer envelope")

        module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++17",),
            name_expressions=("p71_wait_cycles",),
        )
        wait_kernel = module.get_function("p71_wait_cycles")
        sink = cp.zeros(1, dtype=cp.uint64)
        remaining_cycles, remaining_calibration = _calibrate_wait(
            cp, wait_kernel, sink, remaining_ms, clock_khz
        )
        head_cycles, head_calibration = _calibrate_wait(
            cp, wait_kernel, sink, HEAD_MS, clock_khz
        )

        def wait_remaining() -> None:
            wait_kernel((1,), (1,), (np.uint64(remaining_cycles), sink))

        def wait_head() -> None:
            wait_kernel((1,), (1,), (np.uint64(head_cycles), sink))

        def run_layer() -> None:
            run_hot()
            wait_remaining()

        layer_timing = _measure(cp, run_layer, 5, 21)
        n_blocks = 28

        def run_compute_epoch() -> None:
            for _block in range(n_blocks):
                for _layer in range(40):
                    run_layer()
                wait_head()

        run_hot()
        cp.cuda.get_current_stream().synchronize()
        reference_output = cp.asnumpy(output)
        compute_timing = _measure(cp, run_compute_epoch, args.warmup, args.reps)
        compute_ms_h4 = float(compute_timing["p50"]) / n_blocks

        def enqueue_segmented(stream, destination, task: int, groups: int) -> None:
            rotation = task % rotations
            for name, _bytes_per_expert in SEGMENTS:
                source = pinned_sources[name]
                nbytes = int(groups * source["bytes_per_expert"])
                if nbytes:
                    cp.cuda.runtime.memcpyAsync(
                        int(destination[name].data.ptr),
                        int(source["base"] + rotation * source["stride"]),
                        nbytes,
                        cp.cuda.runtime.memcpyHostToDevice,
                        stream.ptr,
                    )

        records = {}
        for policy, block_counts in counts_by_policy.items():
            flat_counts = [count for row in block_counts for count in row]
            n_tasks = len(flat_counts)
            arms = {}
            for lead in LEADS:
                destinations = [
                    {
                        name: cp.empty(max_groups * bytes_per_expert, dtype=cp.uint8)
                        for name, bytes_per_expert in SEGMENTS
                    }
                    for _ in range(lead)
                ]
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
                            enqueue_segmented(
                                auxiliary, destination_ring[slot], task, flat_counts[task]
                            )
                            copied_events[slot].record(auxiliary)
                    for task in range(n_tasks):
                        slot = task % lead_value
                        main_stream.wait_event(copied_events[slot])
                        run_layer()
                        consumed_events[slot].record(main_stream)
                        future = task + lead_value
                        if future < n_tasks:
                            with auxiliary:
                                auxiliary.wait_event(consumed_events[slot])
                                enqueue_segmented(
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
            records[policy] = {
                "arms": arms,
                "selected": {"arm": selected_name, **selected},
            }

        cp.cuda.get_current_stream().synchronize()
        overlap_output = cp.asnumpy(output)
        run_hot()
        cp.cuda.get_current_stream().synchronize()
        repeat_output = cp.asnumpy(output)
        quality = {
            "overlap_vs_reference_bit_exact": bool(np.array_equal(
                overlap_output.view(np.uint32), reference_output.view(np.uint32)
            )),
            "repeat_bit_exact": bool(np.array_equal(
                repeat_output.view(np.uint32), reference_output.view(np.uint32)
            )),
            "finite": bool(np.isfinite(overlap_output).all()),
        }
        lru_selected = records["lru52"]["selected"]
        belady_selected = records["belady52"]["selected"]
        gates = {
            "P73_G1_exact_segments_working_set_and_bounds": (
                EXPERT_BYTES == 1_769_472
                and working_set_bytes >= 4 * l2_bytes
                and max_groups * EXPERT_BYTES == max_copy_bytes
            ),
            "P73_G2_hot_output_exact_repeat_finite": all(quality.values()),
            "P73_G3_hot_latency_and_compute_envelope": (
                abs(hot_ms - PHASE59_HOT_MS) / PHASE59_HOT_MS <= 0.20
                and abs(compute_ms_h4 - FLOOR_MS) / FLOOR_MS <= 0.05
            ),
            "P73_G4_belady_normalized_below_65_boundary": (
                belady_selected["floor_normalized_ms_h4"] <= TARGET_H4_MS
            ),
            "P73_G5_lru_normalized_below_65_boundary": (
                lru_selected["floor_normalized_ms_h4"] <= TARGET_H4_MS
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "source": {
                "segments": [{"name": name, "bytes_per_expert": size} for name, size in SEGMENTS],
                "expert_bytes": EXPERT_BYTES,
                "max_groups": max_groups,
                "max_copy_bytes": max_copy_bytes,
                "rotations": rotations,
                "working_set_bytes": working_set_bytes,
                "working_set_over_l2": working_set_bytes / max(l2_bytes, 1),
            },
            "compute": {
                "hot_bulk32": hot_timing,
                "remaining_target_ms": remaining_ms,
                "remaining_wait": remaining_calibration,
                "layer": layer_timing,
                "head": head_calibration,
                "epoch": compute_timing,
                "compute_ms_h4": compute_ms_h4,
            },
            "quality": quality,
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
        "compute": {
            "hot_bulk32_p50": ((payload.get("compute") or {}).get("hot_bulk32") or {}).get("p50"),
            "layer_p50": ((payload.get("compute") or {}).get("layer") or {}).get("p50"),
            "compute_ms_h4": (payload.get("compute") or {}).get("compute_ms_h4"),
        },
        "quality": payload.get("quality"),
        "selected": {
            policy: record["selected"] for policy, record in (payload.get("records") or {}).items()
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
