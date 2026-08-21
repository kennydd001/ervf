"""Phase84 target-only H4 MoE/router/cache/transport discrepancy experiment."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
from s100_phase84_ornith_dflash_ervf_replay import (
    EXPERT_BYTES,
    SEGMENTS,
    _bench_layer,
    _weight_map,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase84_target_h4"
PREREG = (
    REPO / "pro_research" /
    "S100_PHASE84_ORNITH_TARGET_H4_MOE_TRANSPORT_PREREGISTRATION.md"
)
SCRIPT = REPO / "pro_research" / "s100_phase84_ornith_target_h4_moe_transport.py"
TRACE_DEFAULT = (
    REPO / "pro_research" / "results" / "s100_phase76" /
    "ornith_64_hidden_trace.json.gz"
)
DFLASH_RESULT = (
    REPO / "pro_research" / "results" / "s100_phase84" /
    "S100_PHASE84_ORNITH_DFLASH_ERVF_REPLAY.json"
)
PHASE66_RESULT = (
    REPO / "pro_research" / "results" / "s100_phase66" /
    "S100_PHASE66_ORNITH_65TPS_BUDGET.json"
)
PHASE69_RESULT = (
    REPO / "pro_research" / "results" / "s100_phase69" /
    "S100_PHASE69_ORNITH_SUPPORT_H4.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text("utf-8"))


def _slice_tensor(tensor: dict[str, Any], width: int, begin: int, end: int):
    values = np.asarray(tensor["values"])
    rows = values.reshape(-1, width)[begin:end]
    return {
        "name": tensor["name"],
        "type": tensor.get("type"),
        "shape": [width, end - begin, 1, 1],
        "values": rows.reshape(-1).tolist(),
    }


def _reference_groups(
    trace: dict[str, Any], warm_rows: int, measured_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokens = tuple(int(value) for value in trace["tokens"])
    end = warm_rows + measured_rows
    if warm_rows <= 0 or measured_rows <= 0 or end > len(tokens):
        raise ValueError("warm/measured rows must form a non-empty trace prefix")
    routes: dict[int, dict[str, Any]] = {}
    hidden: dict[int, dict[str, Any]] = {}
    for tensor in trace["tensors"]:
        name = tensor["name"]
        if name.startswith("ffn_moe_topk-"):
            routes[int(name.rsplit("-", 1)[1])] = tensor
        elif name.startswith("attn_post_norm-"):
            hidden[int(name.rsplit("-", 1)[1])] = tensor
    if sorted(routes) != list(range(40)) or sorted(hidden) != list(range(40)):
        raise ValueError("target trace lacks the 40 authoritative route/hidden pairs")

    def group(begin: int, stop: int) -> dict[str, Any]:
        return {
            "routes": {
                layer: _slice_tensor(routes[layer], 8, begin, stop)
                for layer in range(40)
            },
            "hidden": {
                layer: _slice_tensor(hidden[layer], 2048, begin, stop)
                for layer in range(40)
            },
        }

    warm = group(0, warm_rows)
    measured = group(warm_rows, end)
    return [warm, measured], [measured]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=TRACE_DEFAULT)
    parser.add_argument("--warm-rows", type=int, default=32)
    parser.add_argument("--measured-rows", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--layers", type=int, default=40)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE84_ORNITH_TARGET_H4_MOE_TRANSPORT.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase84_authoritative_target_h4_moe_transport",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "authoritative target/reference H4 MoE/router/cache/transport only; "
            "not a complete verifier and not output tok/s"
        ),
    }
    cp = None
    try:
        import cupy as cp_module
        import sys

        src = REPO / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        cp = cp_module
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        snapshot = args.snapshot.expanduser().resolve()
        trace_path = args.trace.expanduser().resolve()
        trace = _load_json(trace_path)
        all_groups, measured_groups = _reference_groups(
            trace, args.warm_rows, args.measured_rows,
        )
        if args.layers < 1 or args.layers > 40:
            raise ValueError("layers must be in [1, 40]")
        h4_blocks = math.ceil(args.measured_rows / 4)
        records = []
        weight_map = _weight_map(snapshot)
        for layer in range(args.layers):
            record = _bench_layer(
                cp, snapshot, weight_map, all_groups, measured_groups,
                layer, args.warmup, args.reps, profile_breakdown=True,
            )
            records.append(record)
            print(json.dumps({
                "layer": layer,
                "p50_ms": record["timing_epoch_ms"]["p50"],
                "misses": record["misses"]["total"],
                "route_exact": record["route_exact"],
                "repeat_exact": record["repeat_bit_exact"],
            }), flush=True)

        total_epoch_ms = sum(float(row["timing_epoch_ms"]["p50"]) for row in records)
        total_misses = sum(int(row["misses"]["total"]) for row in records)
        breakdown = {
            name: sum(float(row["timing_breakdown"][name]) for row in records)
            for name in (
                "stage_h2d_ms", "router_ms", "expert_and_combine_ms",
                "cache_commit_d2d_ms", "total_ms",
            )
        }
        target_ms_per_h4 = total_epoch_ms / h4_blocks
        target_misses_per_layer_h4 = total_misses / (args.layers * h4_blocks)
        dflash = _load_json(DFLASH_RESULT)
        dflash_summary = dflash["summary"]
        phase66 = _load_json(PHASE66_RESULT)
        phase69 = _load_json(PHASE69_RESULT)
        dflash_ms_per_h4 = float(dflash_summary["stress_test_ms_per_h4"])
        dflash_misses_per_layer_h4 = float(
            dflash_summary["mean_unique_misses_per_layer_h4"]
        )
        phase69_floor = float(phase69["budget"]["combined_known_floor_ms_h4"])
        hot_moe_allowance = sum((
            float(phase66["components_ms_h4"]["routed_hot_40_layers"]),
            float(phase66["components_ms_h4"]["shared_serial_40_layers"]),
            float(phase69["budget"]["conservative_indirect_correction_40_layers_ms_h4"]),
            float(phase69["budget"]["worse_support_40_ms_h4"]),
        ))
        expert_bytes_ok = sum(
            int(np.prod(shape)) for _projection, _key, _name, shape, _dtype in SEGMENTS
        ) == EXPERT_BYTES
        gates = {
            "P84_G1_authoritative_target_only_input": (
                trace.get("schema") == "ervf.ornith.llama_trace.v1"
                and "dflash" not in trace_path.name.lower()
            ),
            "P84_G2_custom_router_exact_all_rows": all(
                row["route_exact"] for row in records
            ),
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
                "target_trace": str(trace_path),
                "target_trace_schema": trace.get("schema"),
                "warm_rows": args.warm_rows,
                "measured_rows": args.measured_rows,
                "h4_blocks": h4_blocks,
                "layers": args.layers,
                "dflash_result_used_post_measurement_for_comparison_only": str(
                    DFLASH_RESULT
                ),
            },
            "summary": {
                "target_total_layer_epoch_p50_ms": total_epoch_ms,
                "target_stress_ms_per_h4": target_ms_per_h4,
                "previous_target_stress_ms_per_h4": 74.10151553153992,
                "target_total_unique_miss_copies": total_misses,
                "target_total_miss_transport_bytes": total_misses * EXPERT_BYTES,
                "target_mean_unique_misses_per_layer_h4": target_misses_per_layer_h4,
                "target_layer0_misses_per_h4": (
                    records[0]["misses"]["total"] / h4_blocks
                ),
                "instrumented_breakdown_ms_per_h4": {
                    name: value / h4_blocks for name, value in breakdown.items()
                },
                "dflash_latest_stress_ms_per_h4": dflash_ms_per_h4,
                "dflash_first_observed_ms_per_h4": float(
                    dflash_summary["first_full_sweep_observed_ms_per_h4"]
                ),
                "dflash_mean_unique_misses_per_layer_h4": (
                    dflash_misses_per_layer_h4
                ),
                "target_minus_dflash_latest_ms_per_h4": (
                    target_ms_per_h4 - dflash_ms_per_h4
                ),
                "target_minus_dflash_misses_per_layer_h4": (
                    target_misses_per_layer_h4 - dflash_misses_per_layer_h4
                ),
                "phase69_all_hot_component_floor_ms_per_h4": phase69_floor,
                "phase69_conservative_hot_moe_allowance_ms_per_h4": hot_moe_allowance,
                "target_real_moe_minus_hot_moe_allowance_ms_per_h4": (
                    target_ms_per_h4 - hot_moe_allowance
                ),
                "diagnostic_substitution_floor_ms_per_h4": (
                    phase69_floor - hot_moe_allowance + target_ms_per_h4
                ),
                "diagnosis": (
                    "Phase69 is optimistic/incomplete for real execution because its "
                    "all-hot MoE allowance excludes H2D staging and D2D LRU promotion; "
                    "the target-only route workload is miss-heavier than Phase84D"
                ),
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
        payload["environment"] = environment_snapshot(
            (
                SCRIPT, PREREG, args.trace, DFLASH_RESULT,
                PHASE66_RESULT, PHASE69_RESULT,
            )
        )
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
