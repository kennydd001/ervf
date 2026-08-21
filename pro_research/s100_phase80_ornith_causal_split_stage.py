"""Phase80 real-weight causal gate/up -> down transport pipeline."""
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


RESULTS = REPO / "pro_research" / "results" / "s100_phase80"
PREREG = REPO / "pro_research" / "S100_PHASE80_ORNITH_CAUSAL_SPLIT_STAGE_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase80_ornith_causal_split_stage.py"
TARGET_MS = 4000.0 / 65.0
PHASE59_HOT_MS = 0.5609599948
SEGMENT_NAMES = (
    ("gate", "codes", "gate_codes"),
    ("gate", "scales", "gate_scales"),
    ("up", "codes", "up_codes"),
    ("up", "scales", "up_scales"),
    ("down", "codes", "down_codes"),
    ("down", "scales", "down_scales"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE80_ORNITH_CAUSAL_SPLIT_STAGE.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase80_ornith_causal_split_stage",
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

        source_path = REPO / "src"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        cp = cp_module
        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        experts, contracts = _load_experts(
            snapshot, index["weight_map"], args.layer, 64
        )
        hot_experts = experts[:32]
        miss_experts = experts[32:64]
        phase70 = json.loads(PHASE70.read_text("utf-8"))
        trace_counts = _warm_counts(phase70, "lru52")
        flat_counts = [count for block in trace_counts for count in block]
        max_groups = max(flat_counts)

        props = cp.cuda.runtime.getDeviceProperties(0)
        l2_bytes = int(props.get("l2CacheSize", 0))
        clock_khz = int(props.get("clockRate", 0))
        host_arrays = {}
        segment_bytes = {}
        for projection, key, name in SEGMENT_NAMES:
            value = np.ascontiguousarray(_stack(miss_experts, projection, key))
            host_arrays[name] = value
            segment_bytes[name] = value[0].nbytes
        expert_bytes = sum(segment_bytes.values())
        max_copy_bytes = max_groups * expert_bytes
        rotations = max(2, math.ceil((4 * l2_bytes) / max_copy_bytes) + 1)

        pinned = {}
        for name, value in host_arrays.items():
            bytes_per_expert = segment_bytes[name]
            stride = max_groups * bytes_per_expert
            handle = cp.cuda.alloc_pinned_memory(stride * rotations)
            pinned_handles.append(handle)
            view = np.frombuffer(handle, dtype=np.uint8, count=stride * rotations)
            source = value[:max_groups].view(np.uint8).reshape(-1)
            for rotation in range(rotations):
                view[rotation * stride:(rotation + 1) * stride] = source
            pinned[name] = {
                "base": int(view.ctypes.data),
                "stride": stride,
                "bytes_per_expert": bytes_per_expert,
            }
        working_set_bytes = sum(row["stride"] * rotations for row in pinned.values())

        lookup = FusedNVFP4()
        bulk = OrnithNVFP4BulkM1()

        def device_bank(selected):
            result = {}
            for projection in ("gate", "up", "down"):
                result[projection] = {
                    "codes": cp.asarray(_stack(selected, projection, "codes")),
                    "scales": cp.asarray(_stack(selected, projection, "scales")),
                    "global": cp.asarray(np.asarray([
                        row[projection]["global_scale"] for row in selected
                    ], dtype=np.float32)),
                }
            return result

        hot = device_bank(hot_experts)
        miss_global = {
            projection: cp.asarray(np.asarray([
                row[projection]["global_scale"] for row in miss_experts
            ], dtype=np.float32))
            for projection in ("gate", "up", "down")
        }
        staged = {
            name: cp.empty_like(cp.asarray(value)) for name, value in host_arrays.items()
        }
        # Free the temporary copies created only to establish shape/dtype.
        cp.cuda.get_current_stream().synchronize()

        rng = np.random.default_rng(80000000 + args.layer)
        x_hot = cp.asarray(rng.standard_normal((32, 2048), dtype=np.float32))
        x_miss = cp.asarray(rng.standard_normal((32, 2048), dtype=np.float32))
        hot_gate = cp.empty((32, 512), dtype=cp.float32)
        hot_up = cp.empty_like(hot_gate)
        hot_act = cp.empty_like(hot_gate)
        hot_out = cp.empty((32, 2048), dtype=cp.float32)
        miss_gate = cp.empty((32, 512), dtype=cp.float32)
        miss_up = cp.empty_like(miss_gate)
        miss_act = cp.empty_like(miss_gate)
        miss_out = cp.empty((32, 2048), dtype=cp.float32)

        def projection(bank, name, inp, target, groups, rows, cols):
            if groups <= 0:
                return
            row = bank[name]
            bulk.nvfp4(
                row["codes"][:groups], row["scales"][:groups],
                lookup.e2m1, lookup.e4m3, inp[:groups], target[:groups],
                row["global"][:groups], groups, rows, cols,
            )

        miss_bank = {
            projection: {
                "codes": staged[f"{projection}_codes"],
                "scales": staged[f"{projection}_scales"],
                "global": miss_global[projection],
            }
            for projection in ("gate", "up", "down")
        }

        def run_hot(groups):
            if groups <= 0:
                return
            projection(hot, "gate", x_hot, hot_gate, groups, 512, 2048)
            projection(hot, "up", x_hot, hot_up, groups, 512, 2048)
            bulk.swiglu(hot_gate[:groups], hot_up[:groups], hot_act[:groups], groups)
            projection(hot, "down", hot_act, hot_out, groups, 2048, 512)

        def run_miss_gate_up(groups):
            if groups <= 0:
                return
            projection(miss_bank, "gate", x_miss, miss_gate, groups, 512, 2048)
            projection(miss_bank, "up", x_miss, miss_up, groups, 512, 2048)
            bulk.swiglu(miss_gate[:groups], miss_up[:groups], miss_act[:groups], groups)

        def run_miss_down(groups):
            if groups > 0:
                projection(miss_bank, "down", miss_act, miss_out, groups, 2048, 512)

        hot32_timing = _measure(cp, lambda: run_hot(32), 10, 31)
        hot32_ms = float(hot32_timing["p50"])
        remaining_ms = LAYER_MS - hot32_ms
        if remaining_ms <= 0:
            raise RuntimeError("bulk32 exceeds frozen layer envelope")
        module = cp.RawModule(
            code=CUDA_SOURCE, options=("-std=c++17",),
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

        def wait_remaining():
            wait_kernel((1,), (1,), (np.uint64(remaining_cycles), sink))

        def wait_head():
            wait_kernel((1,), (1,), (np.uint64(head_cycles), sink))

        copy_stream = cp.cuda.Stream(non_blocking=True)
        fork = cp.cuda.Event()
        gate_up_ready = cp.cuda.Event()
        down_ready = cp.cuda.Event()
        consumed = cp.cuda.Event()
        consumed.record(cp.cuda.get_current_stream())

        def enqueue(names, groups, task):
            rotation = task % rotations
            for name in names:
                row = pinned[name]
                cp.cuda.runtime.memcpyAsync(
                    int(staged[name].data.ptr),
                    int(row["base"] + rotation * row["stride"]),
                    int(groups * row["bytes_per_expert"]),
                    cp.cuda.runtime.memcpyHostToDevice,
                    copy_stream.ptr,
                )

        gate_up_names = ("gate_codes", "gate_scales", "up_codes", "up_scales")
        down_names = ("down_codes", "down_scales")

        def run_causal_layer(groups, task):
            groups = int(groups)
            if groups <= 0:
                run_hot(32)
                wait_remaining()
                return
            main = cp.cuda.get_current_stream()
            fork.record(main)
            with copy_stream:
                copy_stream.wait_event(fork)
                copy_stream.wait_event(consumed)
                enqueue(gate_up_names, groups, task)
                gate_up_ready.record(copy_stream)
                enqueue(down_names, groups, task)
                down_ready.record(copy_stream)
            run_hot(32 - groups)
            main.wait_event(gate_up_ready)
            run_miss_gate_up(groups)
            main.wait_event(down_ready)
            run_miss_down(groups)
            consumed.record(main)
            wait_remaining()

        def run_compute_epoch():
            for _block in trace_counts:
                for _layer in range(40):
                    run_hot(32)
                    wait_remaining()
                wait_head()

        def run_causal_epoch():
            task = 0
            for block in trace_counts:
                for groups in block:
                    run_causal_layer(groups, task)
                    task += 1
                wait_head()

        compute_a = _measure(cp, run_compute_epoch, args.warmup, args.reps)
        # Establish exact reference from the same staged real weights.
        enqueue(gate_up_names + down_names, max_groups, 0)
        copy_stream.synchronize()
        run_hot(32 - max_groups)
        run_miss_gate_up(max_groups)
        run_miss_down(max_groups)
        cp.cuda.get_current_stream().synchronize()
        reference_hot = cp.asnumpy(hot_out[:32 - max_groups])
        reference_miss = cp.asnumpy(miss_out[:max_groups])

        causal = _measure(cp, run_causal_epoch, args.warmup, args.reps)
        cp.cuda.get_current_stream().synchronize()
        run_causal_layer(max_groups, 0)
        cp.cuda.get_current_stream().synchronize()
        overlap_hot = cp.asnumpy(hot_out[:32 - max_groups])
        overlap_miss = cp.asnumpy(miss_out[:max_groups])
        run_causal_layer(max_groups, 0)
        cp.cuda.get_current_stream().synchronize()
        repeat_hot = cp.asnumpy(hot_out[:32 - max_groups])
        repeat_miss = cp.asnumpy(miss_out[:max_groups])
        compute_b = _measure(cp, run_compute_epoch, args.warmup, args.reps)

        n_blocks = len(trace_counts)
        compute_a_h4 = float(compute_a["p50"]) / n_blocks
        compute_b_h4 = float(compute_b["p50"]) / n_blocks
        compute_bracket_h4 = 0.5 * (compute_a_h4 + compute_b_h4)
        causal_h4 = float(causal["p50"]) / n_blocks
        exposed = causal_h4 - compute_bracket_h4
        normalized = FLOOR_MS + exposed
        quality = {
            "hot_vs_reference_bit_exact": bool(np.array_equal(
                overlap_hot.view(np.uint32), reference_hot.view(np.uint32)
            )),
            "miss_vs_reference_bit_exact": bool(np.array_equal(
                overlap_miss.view(np.uint32), reference_miss.view(np.uint32)
            )),
            "hot_repeat_bit_exact": bool(np.array_equal(
                repeat_hot.view(np.uint32), overlap_hot.view(np.uint32)
            )),
            "miss_repeat_bit_exact": bool(np.array_equal(
                repeat_miss.view(np.uint32), overlap_miss.view(np.uint32)
            )),
            "finite": bool(np.isfinite(overlap_hot).all() and np.isfinite(overlap_miss).all()),
        }
        contract_ok = all(
            row[name]["contract_match"]
            for row in contracts for name in ("gate", "up", "down")
        )
        gates = {
            "P80_G1_real_segments_working_set_contract": (
                contract_ok and expert_bytes == 1_769_472
                and working_set_bytes >= 4 * l2_bytes
            ),
            "P80_G2_overlap_output_exact_repeat_finite": all(quality.values()),
            "P80_G3_hot_and_compute_envelope": (
                abs(hot32_ms - PHASE59_HOT_MS) / PHASE59_HOT_MS <= 0.20
                and abs(compute_bracket_h4 - FLOOR_MS) / FLOOR_MS <= 0.05
            ),
            "P80_G4_floor_normalized_below_65": normalized <= TARGET_MS,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "source": {
                "segment_bytes": segment_bytes,
                "expert_bytes": expert_bytes,
                "max_groups": max_groups,
                "rotations": rotations,
                "working_set_bytes": working_set_bytes,
                "working_set_over_l2": working_set_bytes / max(l2_bytes, 1),
                "trace_blocks": n_blocks,
                "trace_layer_tasks": len(flat_counts),
                "mean_miss_groups_h4": float(np.mean([sum(row) for row in trace_counts])),
            },
            "compute": {
                "hot32": hot32_timing,
                "remaining_target_ms": remaining_ms,
                "remaining_calibration": remaining_calibration,
                "head_calibration": head_calibration,
                "baseline_a": compute_a,
                "baseline_b": compute_b,
                "baseline_a_ms_h4": compute_a_h4,
                "baseline_b_ms_h4": compute_b_h4,
                "baseline_bracket_ms_h4": compute_bracket_h4,
            },
            "causal": {
                "timing": causal,
                "raw_ms_h4": causal_h4,
                "paired_exposed_tail_ms_h4": exposed,
                "floor_normalized_ms_h4": normalized,
                "floor_normalized_tok_s": 4000.0 / normalized,
            },
            "quality": quality,
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
        "hot32_p50": ((payload.get("compute") or {}).get("hot32") or {}).get("p50"),
        "compute_bracket_ms_h4": (payload.get("compute") or {}).get("baseline_bracket_ms_h4"),
        "causal": payload.get("causal"),
        "quality": payload.get("quality"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
