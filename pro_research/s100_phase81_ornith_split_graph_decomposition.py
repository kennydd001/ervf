"""Phase81 isolate eager versus CUDA-Graph Ornith split dispatch overhead."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import REPO, environment_snapshot, percentiles, utc_now, write_json_atomic
from s100_phase59_ornith_bulk_expert import _load_experts, _stack
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1
from s100_phase71_ornith_trace_prefetch_oracle import (
    FLOOR_MS,
    PHASE70,
    _measure,
    _warm_counts,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase81"
PREREG = REPO / "pro_research" / "S100_PHASE81_ORNITH_SPLIT_GRAPH_DECOMPOSITION_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase81_ornith_split_graph_decomposition.py"
PHASE80 = REPO / "pro_research" / "results" / "s100_phase80" / "S100_PHASE80_ORNITH_CAUSAL_SPLIT_STAGE.json"
TARGET_MS = 4000.0 / 65.0


def _relative_delta(left: float, right: float) -> float:
    return abs(left - right) / max(0.5 * (left + right), 1.0e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=9)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE81_ORNITH_SPLIT_GRAPH_DECOMPOSITION.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase81_ornith_split_graph_decomposition",
        "status": "started",
        "started_utc": utc_now(),
        "snapshot": str(args.snapshot.resolve()),
        "layer": int(args.layer),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    cp = None
    try:
        import cupy as cp_module
        import sys

        cp = cp_module
        source_path = REPO / "src"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        experts, contracts = _load_experts(snapshot, index["weight_map"], args.layer, 64)
        resident_experts = experts[:32]
        miss_experts = experts[32:64]
        phase70 = json.loads(PHASE70.read_text("utf-8"))
        trace_counts = _warm_counts(phase70, "lru52")
        flat_counts = [count for block in trace_counts for count in block]
        observed_counts = sorted(set(flat_counts))
        max_groups = max(observed_counts)

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

        resident = device_bank(resident_experts)
        miss = device_bank(miss_experts)
        rng = np.random.default_rng(81000000 + args.layer)
        x_resident = cp.asarray(rng.standard_normal((32, 2048), dtype=np.float32))
        x_miss = cp.asarray(rng.standard_normal((32, 2048), dtype=np.float32))
        resident_gate = cp.empty((32, 512), dtype=cp.float32)
        resident_up = cp.empty_like(resident_gate)
        resident_act = cp.empty_like(resident_gate)
        resident_out = cp.empty((32, 2048), dtype=cp.float32)
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

        def run_bank(bank, inp, gate, up, act, output, groups):
            if groups <= 0:
                return
            projection(bank, "gate", inp, gate, groups, 512, 2048)
            projection(bank, "up", inp, up, groups, 512, 2048)
            bulk.swiglu(gate[:groups], up[:groups], act[:groups], groups)
            projection(bank, "down", act, output, groups, 2048, 512)

        def run_baseline():
            run_bank(
                resident, x_resident, resident_gate, resident_up,
                resident_act, resident_out, 32,
            )

        def run_split(groups):
            groups = int(groups)
            run_bank(
                resident, x_resident, resident_gate, resident_up,
                resident_act, resident_out, 32 - groups,
            )
            run_bank(miss, x_miss, miss_gate, miss_up, miss_act, miss_out, groups)

        # Compile every launch shape before graph capture.
        run_baseline()
        for groups in observed_counts:
            run_split(groups)
        cp.cuda.get_current_stream().synchronize()

        capture_stream = cp.cuda.Stream(non_blocking=True)

        def capture(function: Callable[[], None]):
            capture_stream.begin_capture()
            with capture_stream:
                function()
            graph = capture_stream.end_capture()
            capture_stream.synchronize()
            return graph

        baseline_graph = capture(run_baseline)
        split_graphs = {
            groups: capture(lambda groups=groups: run_split(groups))
            for groups in observed_counts
        }
        main_stream = cp.cuda.get_current_stream()

        def eager_baseline_epoch():
            for _ in flat_counts:
                run_baseline()

        def eager_split_epoch():
            for groups in flat_counts:
                run_split(groups)

        def graph_baseline_epoch():
            for _ in flat_counts:
                baseline_graph.launch(main_stream)

        def graph_split_epoch():
            for groups in flat_counts:
                split_graphs[groups].launch(main_stream)

        arms = {
            "eager_baseline": eager_baseline_epoch,
            "eager_split": eager_split_epoch,
            "graph_baseline": graph_baseline_epoch,
            "graph_split": graph_split_epoch,
        }
        for _ in range(args.warmup):
            for function in arms.values():
                function()
        main_stream.synchronize()

        def time_once(function: Callable[[], None]) -> float:
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            begin.record(main_stream)
            function()
            end.record(main_stream)
            end.synchronize()
            return float(cp.cuda.get_elapsed_time(begin, end))

        forward = (
            "eager_baseline", "eager_split", "graph_baseline", "graph_split",
            "graph_split", "graph_baseline", "eager_split", "eager_baseline",
        )
        rounds = []
        for round_index in range(args.reps):
            order = forward if round_index % 2 == 0 else tuple(reversed(forward))
            observations = {name: [] for name in arms}
            for name in order:
                observations[name].append(time_once(arms[name]))
            means = {name: float(np.mean(values)) for name, values in observations.items()}
            rounds.append({
                "round": round_index,
                "order": list(order),
                "observations_ms": observations,
                "means_ms": means,
                "eager_delta_ms": means["eager_split"] - means["eager_baseline"],
                "graph_delta_ms": means["graph_split"] - means["graph_baseline"],
                "eager_baseline_mirror_relative_delta": _relative_delta(
                    observations["eager_baseline"][0],
                    observations["eager_baseline"][1],
                ),
            })

        # Exactness is checked after timing so it cannot warm only the candidate.
        run_split(max_groups)
        main_stream.synchronize()
        eager_hot = cp.asnumpy(resident_out[:32 - max_groups])
        eager_miss = cp.asnumpy(miss_out[:max_groups])
        split_graphs[max_groups].launch(main_stream)
        main_stream.synchronize()
        graph_hot = cp.asnumpy(resident_out[:32 - max_groups])
        graph_miss = cp.asnumpy(miss_out[:max_groups])
        split_graphs[max_groups].launch(main_stream)
        main_stream.synchronize()
        repeat_hot = cp.asnumpy(resident_out[:32 - max_groups])
        repeat_miss = cp.asnumpy(miss_out[:max_groups])
        n_blocks = len(trace_counts)
        arm_timings = {
            name: percentiles([row["means_ms"][name] for row in rounds])
            for name in arms
        }
        eager_delta_timing = percentiles([row["eager_delta_ms"] for row in rounds])
        graph_delta_timing = percentiles([row["graph_delta_ms"] for row in rounds])
        mirror_timing = percentiles([
            row["eager_baseline_mirror_relative_delta"] for row in rounds
        ])
        eager_delta_h4 = float(eager_delta_timing["p50"]) / n_blocks
        graph_delta_h4 = float(graph_delta_timing["p50"]) / n_blocks
        reduction = 1.0 - graph_delta_h4 / max(eager_delta_h4, 1.0e-9)
        phase80 = json.loads(PHASE80.read_text("utf-8"))
        phase80_tail = float(phase80["causal"]["paired_exposed_tail_ms_h4"])
        # Only dispatch is substituted; the residual retains every other Phase80 cost.
        residual_non_dispatch = max(0.0, phase80_tail - eager_delta_h4)
        projected_tail = residual_non_dispatch + max(0.0, graph_delta_h4)
        projected_ms = FLOOR_MS + projected_tail
        quality = {
            "hot_graph_vs_eager_bit_exact": bool(np.array_equal(
                graph_hot.view(np.uint32), eager_hot.view(np.uint32)
            )),
            "miss_graph_vs_eager_bit_exact": bool(np.array_equal(
                graph_miss.view(np.uint32), eager_miss.view(np.uint32)
            )),
            "hot_graph_repeat_bit_exact": bool(np.array_equal(
                repeat_hot.view(np.uint32), graph_hot.view(np.uint32)
            )),
            "miss_graph_repeat_bit_exact": bool(np.array_equal(
                repeat_miss.view(np.uint32), graph_miss.view(np.uint32)
            )),
            "finite": bool(np.isfinite(graph_hot).all() and np.isfinite(graph_miss).all()),
        }
        contract_ok = all(
            row[name]["contract_match"]
            for row in contracts for name in ("gate", "up", "down")
        )
        gates = {
            "P81_G1_real_contract_and_trace_shape": (
                contract_ok and len(trace_counts) == 28 and len(flat_counts) == 1120
            ),
            "P81_G2_graph_exact_repeat_finite": all(quality.values()),
            "P81_G3_eager_baseline_bracket_stable": float(mirror_timing["p50"]) <= 0.05,
            "P81_G4_graph_reduces_split_overhead_30pct": (
                eager_delta_h4 > 0.0 and graph_delta_h4 <= 0.70 * eager_delta_h4
            ),
            "P81_G5_dispatch_substitution_below_65": projected_ms <= TARGET_MS,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "source": {
                "trace_blocks": n_blocks,
                "trace_layer_tasks": len(flat_counts),
                "observed_miss_counts": observed_counts,
                "max_groups": max_groups,
                "mean_miss_groups_h4": float(np.mean([sum(row) for row in trace_counts])),
            },
            "timings": {
                "mirrored_rounds": rounds,
                "arms": arm_timings,
                "eager_delta_ms_epoch": eager_delta_timing,
                "graph_delta_ms_epoch": graph_delta_timing,
                "eager_baseline_mirror_relative_delta": mirror_timing,
            },
            "decomposition": {
                "eager_split_extra_ms_h4": eager_delta_h4,
                "graph_split_extra_ms_h4": graph_delta_h4,
                "graph_reduction_fraction": reduction,
                "phase80_total_tail_ms_h4": phase80_tail,
                "phase80_residual_non_dispatch_ms_h4": residual_non_dispatch,
                "dispatch_substituted_tail_ms_h4": projected_tail,
                "dispatch_substituted_floor_ms_h4": projected_ms,
                "dispatch_substituted_floor_tok_s": 4000.0 / projected_ms,
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, PHASE70, PHASE80))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "source": payload.get("source"),
        "decomposition": payload.get("decomposition"),
        "quality": payload.get("quality"),
        "gates": payload.get("gates"),
        "error": payload.get("error"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
