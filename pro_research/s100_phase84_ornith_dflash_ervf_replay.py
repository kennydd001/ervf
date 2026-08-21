"""Phase84D: stress the ERVF MoE/transport path with DFlash-candidate rows."""
from __future__ import annotations

import argparse
import gzip
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import REPO, environment_snapshot, percentiles, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
from s100_phase48_ornith_swiglu_h8 import _load_projection
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1
from s100_phase60_ornith_route_adaptive_kernels import OrnithNVFP4RouteAdaptive
from s100_phase69_ornith_support_h4_kernels import OrnithSupportH4Kernels
from s100_phase78_ornith_dflash_route_signal import _rows, _target_groups


SRC = REPO / "src"
RESULTS = REPO / "pro_research" / "results" / "s100_phase84"
PREREG = REPO / "pro_research" / "S100_PHASE84_ORNITH_DFLASH_ERVF_REPLAY_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase84_ornith_dflash_ervf_replay.py"
TRACE_DEFAULT = (
    REPO / "pro_research" / "results" / "s100_phase76" /
    "dflash_spec_trace_v2.json.gz"
)
CACHE_SLOTS = 52
STAGE_SLOTS = 32
TOTAL_SLOTS = CACHE_SLOTS + STAGE_SLOTS
EXPERT_BYTES = 1_769_472

SEGMENTS = (
    ("gate", "codes", "gate_codes", (512, 1024), np.uint8),
    ("gate", "scales", "gate_scales", (512, 128), np.uint8),
    ("up", "codes", "up_codes", (512, 1024), np.uint8),
    ("up", "scales", "up_scales", (512, 128), np.uint8),
    ("down", "codes", "down_codes", (2048, 256), np.uint8),
    ("down", "scales", "down_scales", (2048, 32), np.uint8),
)


@dataclass(frozen=True)
class PlannedTask:
    event: int
    chunk: int
    valid_rows: int
    hidden: np.ndarray
    routes: tuple[tuple[int, ...], ...]
    before: Any
    after: Any
    misses: tuple[int, ...]
    plan: Any
    payload_offset: int


class PinnedArrays:
    def __init__(self, cp) -> None:
        self.cp = cp
        self.handles: list[Any] = []
        self.arrays: dict[str, np.ndarray] = {}

    def allocate(self, name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        count = int(np.prod(shape, dtype=np.int64))
        nbytes = count * np.dtype(dtype).itemsize
        handle = self.cp.cuda.alloc_pinned_memory(nbytes)
        self.handles.append(handle)
        value = np.frombuffer(handle, dtype=dtype, count=count).reshape(shape)
        self.arrays[name] = value
        return value


def _load_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text("utf-8"))


def _weight_map(snapshot: Path) -> dict[str, str]:
    return json.loads(
        (snapshot / "model.safetensors.index.json").read_text("utf-8")
    )["weight_map"]


def _load_router_support(
    snapshot: Path,
    weight_map: dict[str, str],
    layer: int,
) -> dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open

    names = {
        "router": f"model.layers.{layer}.mlp.gate.weight",
        "shared_gate": f"model.layers.{layer}.mlp.shared_expert_gate.weight",
    }
    expected = {"router": (256, 2048), "shared_gate": (1, 2048)}
    result = {}
    for label, name in names.items():
        with safe_open(
            str(snapshot / weight_map[name]), framework="pt", device="cpu"
        ) as handle:
            tensor = handle.get_tensor(name).contiguous()
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected[label]:
            raise TypeError(
                f"{name}: expected BF16{expected[label]}, got {tensor.dtype}{tensor.shape}"
            )
        result[label] = tensor.view(torch.uint16).numpy().copy().reshape(expected[label])
    return result


def _warm_rows(groups: Iterable[dict[str, Any]], layer: int) -> Iterable[tuple[int, ...]]:
    for group in groups:
        for row in _rows(group["routes"][layer], 8).astype(np.int64):
            yield tuple(int(value) for value in row)


def _plan_layer(
    all_groups: list[dict[str, Any]],
    measured_groups: list[dict[str, Any]],
    layer: int,
):
    from moe_lab.ornith.rolling_prefetch import _LayerCache, build_execution_layer_plan

    cache = _LayerCache(CACHE_SLOTS)
    prefill_count = len(all_groups) - len(measured_groups)
    for row in _warm_rows(all_groups[:prefill_count], layer):
        cache.process_rows((row,))
    warm = cache.snapshot()
    tasks: list[PlannedTask] = []
    payload_offset = 0
    for event, group in enumerate(measured_groups):
        hidden_rows = _rows(group["hidden"][layer], 2048).astype(np.float32)
        route_rows = _rows(group["routes"][layer], 8).astype(np.int64)
        if len(hidden_rows) != len(route_rows):
            raise ValueError(f"layer {layer} event {event}: hidden/route length mismatch")
        for begin in range(0, len(hidden_rows), 4):
            valid_rows = min(4, len(hidden_rows) - begin)
            real_routes = tuple(
                tuple(int(value) for value in row)
                for row in route_rows[begin:begin + 4]
            )
            padded_routes = list(real_routes)
            padded_hidden = list(hidden_rows[begin:begin + 4])
            while len(padded_routes) < 4:
                # H4 kernels have fixed geometry. Repeating the last real row is
                # arithmetically inert for the committed prefix: it introduces no
                # new expert and is excluded from the cache transition below.
                padded_routes.append(padded_routes[-1])
                padded_hidden.append(padded_hidden[-1])
            routes = tuple(padded_routes)
            before = cache.snapshot()
            replay = _LayerCache.from_snapshot(before)
            misses = replay.process_rows(real_routes)
            after = replay.snapshot()
            plan = build_execution_layer_plan(routes, before, misses, layer=layer)
            if plan.uncovered_experts or plan.false_prefetch_experts:
                raise AssertionError("authoritative miss staging must cover the complete H4")
            tasks.append(PlannedTask(
                event=event,
                chunk=begin // 4,
                valid_rows=valid_rows,
                hidden=np.ascontiguousarray(np.stack(padded_hidden)),
                routes=routes,
                before=before,
                after=after,
                misses=misses,
                plan=plan,
                payload_offset=payload_offset,
            ))
            payload_offset += len(misses)
            cache = replay
    return warm, tasks


def _selected_experts(warm, tasks: list[PlannedTask]) -> tuple[int, ...]:
    values = list(warm.expert_to_slot)
    for task in tasks:
        values.extend(task.misses)
    return tuple(dict.fromkeys(values))


def _load_host_bank(
    cp,
    snapshot: Path,
    weight_map: dict[str, str],
    layer: int,
    expert_ids: tuple[int, ...],
):
    pinned = PinnedArrays(cp)
    count = len(expert_ids)
    for _projection, _key, name, shape, dtype in SEGMENTS:
        pinned.allocate(name, (count, *shape), dtype)
    for projection in ("gate", "up", "down"):
        pinned.allocate(f"{projection}_global", (count,), np.float32)
    index_of = {expert: index for index, expert in enumerate(expert_ids)}
    for expert, index in index_of.items():
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        for projection in ("gate", "up", "down"):
            row = _load_projection(snapshot, weight_map, f"{base}.{projection}_proj")
            pinned.arrays[f"{projection}_codes"][index] = row["codes"].reshape(
                pinned.arrays[f"{projection}_codes"].shape[1:]
            )
            pinned.arrays[f"{projection}_scales"][index] = row["scales"].reshape(
                pinned.arrays[f"{projection}_scales"].shape[1:]
            )
            pinned.arrays[f"{projection}_global"][index] = np.float32(
                row["global_scale"]
            )
    return pinned, index_of


def _pack_tasks(cp, host_bank: PinnedArrays, index_of, tasks: list[PlannedTask]):
    total = sum(len(task.misses) for task in tasks)
    packed = PinnedArrays(cp)
    for _projection, _key, name, shape, dtype in SEGMENTS:
        packed.allocate(name, (total, *shape), dtype)
    for projection in ("gate", "up", "down"):
        packed.allocate(f"{projection}_global", (total,), np.float32)
    for task in tasks:
        for local, expert in enumerate(task.misses):
            target = task.payload_offset + local
            source = index_of[expert]
            for _projection, _key, name, _shape, _dtype in SEGMENTS:
                packed.arrays[name][target] = host_bank.arrays[name][source]
            for projection in ("gate", "up", "down"):
                packed.arrays[f"{projection}_global"][target] = (
                    host_bank.arrays[f"{projection}_global"][source]
                )
    return packed


def _device_bank(cp):
    values = {}
    for _projection, _key, name, shape, dtype in SEGMENTS:
        values[name] = cp.empty((TOTAL_SLOTS, *shape), dtype=dtype)
    for projection in ("gate", "up", "down"):
        values[f"{projection}_global"] = cp.empty(TOTAL_SLOTS, dtype=cp.float32)
    return values


def _shared_bank(cp, snapshot: Path, weight_map: dict[str, str], layer: int):
    result = {}
    base = f"model.layers.{layer}.mlp.shared_expert"
    for projection in ("gate", "up", "down"):
        row = _load_projection(snapshot, weight_map, f"{base}.{projection}_proj")
        result[f"{projection}_codes"] = cp.asarray(row["codes"].reshape(
            (1, 512, 1024) if projection != "down" else (1, 2048, 256)
        ))
        result[f"{projection}_scales"] = cp.asarray(row["scales"].reshape(
            (1, 512, 128) if projection != "down" else (1, 2048, 32)
        ))
        result[f"{projection}_global"] = cp.asarray(
            np.asarray([row["global_scale"]], dtype=np.float32)
        )
    return result


def _copy_row(cp, destination, dst_row: int, source, src_row: int, kind: int) -> None:
    row_bytes = int(destination[dst_row].nbytes)
    cp.cuda.runtime.memcpyAsync(
        int(destination[dst_row].data.ptr),
        int(source.ctypes.data + src_row * row_bytes),
        row_bytes,
        kind,
        cp.cuda.get_current_stream().ptr,
    )


def _copy_device_row(cp, destination, dst_row: int, src_row: int) -> None:
    cp.cuda.runtime.memcpyAsync(
        int(destination[dst_row].data.ptr),
        int(destination[src_row].data.ptr),
        int(destination[dst_row].nbytes),
        cp.cuda.runtime.memcpyDeviceToDevice,
        cp.cuda.get_current_stream().ptr,
    )


def _bench_layer(
    cp,
    snapshot: Path,
    weight_map: dict[str, str],
    all_groups: list[dict[str, Any]],
    measured_groups: list[dict[str, Any]],
    layer: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

    warm, tasks = _plan_layer(all_groups, measured_groups, layer)
    selected = _selected_experts(warm, tasks)
    host_bank, index_of = _load_host_bank(cp, snapshot, weight_map, layer, selected)
    packed = _pack_tasks(cp, host_bank, index_of, tasks)
    bank = _device_bank(cp)
    shared = _shared_bank(cp, snapshot, weight_map, layer)
    support_host = _load_router_support(snapshot, weight_map, layer)
    support = {name: cp.asarray(row) for name, row in support_host.items()}
    del support_host

    lookup = FusedNVFP4()
    bulk = OrnithNVFP4BulkM1()
    adaptive = OrnithNVFP4RouteAdaptive()
    support_kernels = OrnithSupportH4Kernels()
    slot_of = cp.full(256, -1, dtype=cp.int32)
    router_logits = cp.empty((4, 256), dtype=cp.float32)
    shared_logits = cp.empty(4, dtype=cp.float32)
    route_ids = cp.empty((4, 8), dtype=cp.int32)
    route_weights = cp.empty((4, 8), dtype=cp.float32)
    route_slots = cp.empty((4, 8), dtype=cp.int32)
    route_need = cp.empty((4, 8), dtype=cp.int32)
    expert_outputs = cp.empty((32, 2048), dtype=cp.float32)
    shared_gate = cp.empty((1, 4, 512), dtype=cp.float32)
    shared_up = cp.empty_like(shared_gate)
    shared_act = cp.empty_like(shared_gate)
    shared_out = cp.empty((1, 4, 2048), dtype=cp.float32)
    shared_slots = cp.zeros(1, dtype=cp.int32)
    shared_inputs = cp.arange(4, dtype=cp.int32)
    shared_down_inputs = cp.arange(4, dtype=cp.int32)
    hidden_device = [cp.asarray(task.hidden) for task in tasks]

    bucket_device = []
    for task in tasks:
        rows = []
        for bucket in task.plan.combined_plan.hot_buckets:
            rows.append({
                "bucket": bucket,
                "slots": cp.asarray(bucket.cache_slots, dtype=cp.int32),
                "inputs": cp.asarray(bucket.input_ids, dtype=cp.int32),
                "down_inputs": cp.arange(bucket.assignments, dtype=cp.int32),
                "route_indices": cp.asarray(bucket.route_indices, dtype=cp.int32),
                "gate": cp.empty((bucket.groups, bucket.multiplicity, 512), dtype=cp.float32),
                "up": cp.empty((bucket.groups, bucket.multiplicity, 512), dtype=cp.float32),
                "act": cp.empty((bucket.groups, bucket.multiplicity, 512), dtype=cp.float32),
                "out": cp.empty((bucket.groups, bucket.multiplicity, 2048), dtype=cp.float32),
            })
        if task.plan.combined_plan.miss_buckets:
            raise AssertionError("combined authoritative plan still has misses")
        bucket_device.append(rows)

    def copy_source_to_slot(source_row: int, slot: int) -> None:
        for _projection, _key, name, _shape, _dtype in SEGMENTS:
            _copy_row(
                cp, bank[name], slot, host_bank.arrays[name], source_row,
                cp.cuda.runtime.memcpyHostToDevice,
            )
        for projection in ("gate", "up", "down"):
            _copy_row(
                cp, bank[f"{projection}_global"], slot,
                host_bank.arrays[f"{projection}_global"], source_row,
                cp.cuda.runtime.memcpyHostToDevice,
            )

    def initialize_warm_cache() -> None:
        for expert, slot in warm.expert_to_slot.items():
            copy_source_to_slot(index_of[expert], slot)

    def stage_task(task: PlannedTask) -> None:
        count = len(task.misses)
        if not count:
            return
        for _projection, _key, name, _shape, _dtype in SEGMENTS:
            row_bytes = int(bank[name][0].nbytes)
            cp.cuda.runtime.memcpyAsync(
                int(bank[name][CACHE_SLOTS].data.ptr),
                int(packed.arrays[name].ctypes.data + task.payload_offset * row_bytes),
                count * row_bytes,
                cp.cuda.runtime.memcpyHostToDevice,
                cp.cuda.get_current_stream().ptr,
            )
        for projection in ("gate", "up", "down"):
            name = f"{projection}_global"
            row_bytes = np.dtype(np.float32).itemsize
            cp.cuda.runtime.memcpyAsync(
                int(bank[name][CACHE_SLOTS].data.ptr),
                int(packed.arrays[name].ctypes.data + task.payload_offset * row_bytes),
                count * row_bytes,
                cp.cuda.runtime.memcpyHostToDevice,
                cp.cuda.get_current_stream().ptr,
            )

    def run_projection(bank_value, projection, x, target, multiplicity, slots, inputs, groups, rows, cols):
        adaptive.nvfp4(
            multiplicity,
            bank_value[f"{projection}_codes"],
            bank_value[f"{projection}_scales"],
            lookup.e2m1,
            lookup.e4m3,
            x,
            target,
            bank_value[f"{projection}_global"],
            slots,
            inputs,
            groups,
            rows,
            cols,
        )

    def execute_task(task_index: int):
        task = tasks[task_index]
        hidden = hidden_device[task_index]
        stage_task(task)
        support_kernels.router_shared(
            support["router"], support["shared_gate"], hidden,
            router_logits, shared_logits,
        )
        support_kernels.top8_cache(
            router_logits, slot_of, route_ids, route_weights, route_slots, route_need
        )
        for row in bucket_device[task_index]:
            bucket = row["bucket"]
            for name, target in (("gate", row["gate"]), ("up", row["up"])):
                run_projection(
                    bank, name, hidden, target, bucket.multiplicity,
                    row["slots"], row["inputs"], bucket.groups, 512, 2048,
                )
            # The tiny activation launch is intentionally shared with the frozen
            # Phase59 implementation so arithmetic/order stays identical.
            bulk.swiglu(
                row["gate"], row["up"], row["act"], bucket.assignments
            )
            run_projection(
                bank, "down", row["act"].reshape(bucket.assignments, 512),
                row["out"], bucket.multiplicity, row["slots"],
                row["down_inputs"], bucket.groups, 2048, 512,
            )
            expert_outputs[row["route_indices"]] = row["out"].reshape(
                bucket.assignments, 2048
            )

        for name, target in (("gate", shared_gate), ("up", shared_up)):
            run_projection(
                shared, name, hidden, target, 4, shared_slots,
                shared_inputs, 1, 512, 2048,
            )
        bulk.swiglu(shared_gate, shared_up, shared_act, 4)
        run_projection(
            shared, "down", shared_act.reshape(4, 512), shared_out,
            4, shared_slots, shared_down_inputs, 1, 2048, 512,
        )
        routed = cp.sum(
            expert_outputs.reshape(4, 8, 2048) * route_weights[:, :, None], axis=1
        )
        branch = routed + shared_out[0] / (1.0 + cp.exp(-shared_logits[:, None]))

        staged = {expert: CACHE_SLOTS + index for index, expert in enumerate(task.misses)}
        before_slots = task.before.slot_to_expert
        for slot, expert in enumerate(task.after.slot_to_expert):
            if expert is None or expert == before_slots[slot]:
                continue
            source_slot = staged[expert]
            for _projection, _key, name, _shape, _dtype in SEGMENTS:
                _copy_device_row(cp, bank[name], slot, source_slot)
            for projection in ("gate", "up", "down"):
                _copy_device_row(cp, bank[f"{projection}_global"], slot, source_slot)
        return branch

    # Route parity is checked before timing and against the custom BF16 router.
    route_exact = []
    for index, task in enumerate(tasks):
        hidden = hidden_device[index]
        support_kernels.router_shared(
            support["router"], support["shared_gate"], hidden,
            router_logits, shared_logits,
        )
        support_kernels.top8_cache(
            router_logits, slot_of, route_ids, route_weights, route_slots, route_need
        )
        cp.cuda.get_current_stream().synchronize()
        route_exact.append(bool(np.array_equal(
            cp.asnumpy(route_ids), np.asarray(task.routes, dtype=np.int32)
        )))

    def correctness_pass():
        initialize_warm_cache()
        cp.cuda.get_current_stream().synchronize()
        outputs = []
        for index in range(len(tasks)):
            branch = execute_task(index)
            cp.cuda.get_current_stream().synchronize()
            outputs.append(cp.asnumpy(branch))
        return outputs

    first = correctness_pass()
    second = correctness_pass()
    repeat_exact = all(
        np.array_equal(left.view(np.uint32), right.view(np.uint32))
        for left, right in zip(first, second)
    )
    finite = all(np.isfinite(value).all() for value in first)

    def run_epoch() -> None:
        for index in range(len(tasks)):
            execute_task(index)

    for _ in range(warmup):
        initialize_warm_cache()
        cp.cuda.get_current_stream().synchronize()
        run_epoch()
        cp.cuda.get_current_stream().synchronize()
    samples = []
    for _ in range(reps):
        initialize_warm_cache()
        cp.cuda.get_current_stream().synchronize()
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        begin.record()
        run_epoch()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(begin, end)))

    miss_counts = [len(task.misses) for task in tasks]
    transition_exact = all(
        task.plan.combined_plan.hot_assignments == 32
        and task.plan.combined_plan.miss_assignments == 0
        for task in tasks
    )
    timing = percentiles(samples)
    timing["samples_ms"] = samples
    result = {
        "layer": layer,
        "tasks": len(tasks),
        "warm_residents": len(warm.expert_to_slot),
        "selected_source_experts": len(selected),
        "misses": {
            "total": int(sum(miss_counts)),
            "mean": float(np.mean(miss_counts)),
            "max": int(max(miss_counts, default=0)),
            "counts": miss_counts,
            "transport_bytes": int(sum(miss_counts) * EXPERT_BYTES),
        },
        "timing_epoch_ms": timing,
        "route_rows_exact": int(sum(route_exact)),
        "route_rows_total": len(route_exact),
        "route_exact": all(route_exact),
        "transition_exact": transition_exact,
        "repeat_bit_exact": repeat_exact,
        "finite": finite,
    }
    del bank, shared, support, hidden_device, bucket_device
    cp.get_default_memory_pool().free_all_blocks()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=TRACE_DEFAULT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--layers", type=int, default=40)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE84_ORNITH_DFLASH_ERVF_REPLAY.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase84d_dflash_candidate_moe_transport_stress",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "DFlash-candidate-workload MoE/transport stress test; not a complete "
            "verifier, not a complete decoder, and not output tok/s"
        ),
    }
    cp = None
    try:
        import cupy as cp_module
        import sys

        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        cp = cp_module
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        snapshot = args.snapshot.expanduser().resolve()
        trace_path = args.trace.expanduser().resolve()
        trace = _load_trace(trace_path)
        all_groups = _target_groups(trace["target"])
        n_events = len(trace["target_batches"])
        measured_groups = all_groups[-n_events:]
        expected_lengths = [len(tokens) for tokens in trace["target_batches"]]
        observed_lengths = [int(group["routes"][0]["shape"][1]) for group in measured_groups]
        if expected_lengths != observed_lengths:
            raise ValueError("target batch/callback alignment mismatch")
        if args.layers < 1 or args.layers > 40:
            raise ValueError("layers must be in [1, 40]")
        weight_map = _weight_map(snapshot)
        records = []
        for layer in range(args.layers):
            record = _bench_layer(
                cp, snapshot, weight_map, all_groups, measured_groups,
                layer, args.warmup, args.reps,
            )
            records.append(record)
            print(json.dumps({
                "layer": layer,
                "p50_ms": record["timing_epoch_ms"]["p50"],
                "misses": record["misses"]["total"],
                "route_exact": record["route_exact"],
                "repeat_exact": record["repeat_bit_exact"],
            }), flush=True)

        h4_blocks = sum((length + 3) // 4 for length in expected_lengths)
        total_epoch_ms = sum(float(row["timing_epoch_ms"]["p50"]) for row in records)
        total_misses = sum(int(row["misses"]["total"]) for row in records)
        expert_bytes_ok = sum(
            int(np.prod(shape)) for _projection, _key, _name, shape, _dtype in SEGMENTS
        ) == EXPERT_BYTES
        gates = {
            "P84_G1_callback_alignment_and_h4_geometry": (
                len(measured_groups) == n_events
                and all(length > 0 for length in expected_lengths)
            ),
            "P84_G2_custom_router_exact_all_rows": all(row["route_exact"] for row in records),
            "P84_G3_authoritative_staging_covers_every_assignment": all(
                row["transition_exact"] for row in records
            ),
            "P84_G4_fresh_cache_repeat_exact_and_finite": all(
                row["repeat_bit_exact"] and row["finite"] for row in records
            ),
            "P84_G5_real_expert_segment_contract": expert_bytes_ok,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "completed_utc": utc_now(),
            "inputs": {
                "snapshot": str(snapshot),
                "trace": str(trace_path),
                "callback_groups": len(all_groups),
                "prefill_groups": len(all_groups) - n_events,
                "measured_events": n_events,
                "target_batch_lengths": expected_lengths,
                "h4_blocks": h4_blocks,
            },
            "summary": {
                "layers_executed": args.layers,
                "total_layer_epoch_p50_ms": total_epoch_ms,
                "stress_test_ms_per_h4": total_epoch_ms / h4_blocks,
                "first_full_sweep_observed_ms_per_h4": 67.37636498843922,
                "previous_three_repeat_median_ms_per_h4": 66.95565190034755,
                "total_unique_miss_copies": total_misses,
                "total_miss_transport_bytes": total_misses * EXPERT_BYTES,
                "mean_unique_misses_per_layer_h4": total_misses / (args.layers * h4_blocks),
                "explicit_no_complete_verifier_or_output_tok_s_claim": True,
            },
            "records": records,
            "gates": gates,
        })
    except Exception as error:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        })
    finally:
        if cp is not None:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, args.trace))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "inputs": payload.get("inputs"),
        "summary": payload.get("summary"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2), flush=True)
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
