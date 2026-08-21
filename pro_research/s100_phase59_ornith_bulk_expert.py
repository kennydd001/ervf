"""Phase59 benchmark of 1..32 real Ornith expert assignments in one dispatch."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase48_ornith_swiglu_h8 import (
    _decode,
    _load_projection,
    _measure,
    _metrics,
    _projection_contract,
)
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1


RESULTS = REPO / "pro_research" / "results" / "s100_phase59"
PREREG = REPO / "pro_research" / "S100_PHASE59_ORNITH_BULK_EXPERT_H4_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase59_ornith_bulk_expert.py"
KERNELS = REPO / "pro_research" / "s100_phase59_ornith_bulk_expert_kernels.py"
GROUP_COUNTS = (1, 4, 8, 16, 32)


def _load_experts(snapshot: Path, weight_map: dict[str, str], layer: int, count: int):
    experts = []
    contracts = []
    for expert in range(count):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        row = {
            name: _load_projection(snapshot, weight_map, base + f".{name}_proj")
            for name in ("gate", "up", "down")
        }
        experts.append(row)
        contracts.append({
            "expert": expert,
            "gate": _projection_contract(row["gate"], 512, 2048),
            "up": _projection_contract(row["up"], 512, 2048),
            "down": _projection_contract(row["down"], 2048, 512),
        })
    return experts, contracts


def _stack(experts, projection: str, key: str) -> np.ndarray:
    return np.stack([row[projection][key] for row in experts], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=61)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE59_ORNITH_BULK_EXPERT_H4.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase59_ornith_bulk_expert_h4",
        "status": "started",
        "started_utc": utc_now(),
        "snapshot": str(args.snapshot.resolve()),
        "layer": int(args.layer),
        "group_counts": list(GROUP_COUNTS),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "hot GPU-resident routed experts only; no host misses, router, shared expert, "
            "attention/recurrent state, LM head, DFlash orchestration, or tok/s claim"
        ),
    }
    cp = None
    try:
        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        experts, contracts = _load_experts(snapshot, index["weight_map"], args.layer, 32)

        rng = np.random.default_rng(59000000 + args.layer)
        x_host = rng.standard_normal((32, 2048), dtype=np.float32)
        reference = []
        for expert in range(2):
            gate = x_host[expert] @ _decode(experts[expert]["gate"], 512, 2048).T
            up = x_host[expert] @ _decode(experts[expert]["up"], 512, 2048).T
            act = (gate / (1.0 + np.exp(-gate))) * up
            reference.append(act @ _decode(experts[expert]["down"], 2048, 512).T)
        reference_host = np.stack(reference)

        import cupy as cp_module
        import sys

        src = REPO / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        cp = cp_module
        lookup = FusedNVFP4()
        kernels = OrnithNVFP4BulkM1()
        x = cp.asarray(x_host)
        device = {}
        payload_bytes = 0
        for name in ("gate", "up", "down"):
            codes_host = _stack(experts, name, "codes")
            scales_host = _stack(experts, name, "scales")
            global_host = np.asarray(
                [row[name]["global_scale"] for row in experts], dtype=np.float32
            )
            payload_bytes += int(codes_host.nbytes + scales_host.nbytes + global_host.nbytes)
            device[name] = {
                "codes": cp.asarray(codes_host),
                "scales": cp.asarray(scales_host),
                "global": cp.asarray(global_host),
            }

        gate = cp.empty((32, 512), dtype=cp.float32)
        up = cp.empty((32, 512), dtype=cp.float32)
        act = cp.empty((32, 512), dtype=cp.float32)
        bulk_out = cp.empty((32, 2048), dtype=cp.float32)
        serial_out = cp.empty_like(bulk_out)
        repeat_out = cp.empty_like(bulk_out)

        def launch_projection(name: str, inp, target, groups: int, rows: int, cols: int) -> None:
            row = device[name]
            kernels.nvfp4(
                row["codes"][:groups], row["scales"][:groups],
                lookup.e2m1, lookup.e4m3, inp[:groups], target[:groups],
                row["global"][:groups], groups, rows, cols,
            )

        def run_bulk(groups: int, target=bulk_out) -> None:
            launch_projection("gate", x, gate, groups, 512, 2048)
            launch_projection("up", x, up, groups, 512, 2048)
            kernels.swiglu(gate, up, act, groups)
            launch_projection("down", act, target, groups, 2048, 512)

        def run_serial(groups: int) -> None:
            for group in range(groups):
                for name, inp, target, rows, cols in (
                    ("gate", x, gate, 512, 2048),
                    ("up", x, up, 512, 2048),
                ):
                    row = device[name]
                    kernels.nvfp4(
                        row["codes"][group:group + 1],
                        row["scales"][group:group + 1],
                        lookup.e2m1, lookup.e4m3,
                        inp[group:group + 1], target[group:group + 1],
                        row["global"][group:group + 1], 1, rows, cols,
                    )
                kernels.swiglu(
                    gate[group:group + 1], up[group:group + 1],
                    act[group:group + 1], 1,
                )
                row = device["down"]
                kernels.nvfp4(
                    row["codes"][group:group + 1], row["scales"][group:group + 1],
                    lookup.e2m1, lookup.e4m3,
                    act[group:group + 1], serial_out[group:group + 1],
                    row["global"][group:group + 1], 1, 2048, 512,
                )

        records = []
        for groups in GROUP_COUNTS:
            run_bulk(groups, bulk_out)
            run_serial(groups)
            cp.cuda.get_current_stream().synchronize()
            bulk_host = cp.asnumpy(bulk_out[:groups])
            serial_host = cp.asnumpy(serial_out[:groups])
            run_bulk(groups, repeat_out)
            cp.cuda.get_current_stream().synchronize()
            repeat_host = cp.asnumpy(repeat_out[:groups])
            bulk_timing = _measure(cp, lambda g=groups: run_bulk(g, bulk_out), args.warmup, args.reps)
            serial_timing = _measure(cp, lambda g=groups: run_serial(g), args.warmup, args.reps)
            bulk_ms = float(bulk_timing["p50"])
            serial_ms = float(serial_timing["p50"])
            records.append({
                "groups": groups,
                "bulk_timing_ms": bulk_timing,
                "serial_timing_ms": serial_timing,
                "speedup": serial_ms / bulk_ms,
                "bulk_assignments_per_ms": groups / bulk_ms,
                "bulk_vs_serial_bit_exact": bool(
                    np.array_equal(bulk_host.view(np.uint32), serial_host.view(np.uint32))
                ),
                "bulk_repeat_bit_exact": bool(
                    np.array_equal(bulk_host.view(np.uint32), repeat_host.view(np.uint32))
                ),
                "finite": bool(np.isfinite(bulk_host).all()),
                "reference_first_two": (
                    _metrics(bulk_host[:min(2, groups)], reference_host[:min(2, groups)])
                ),
            })

        resources = kernels.resource_audit()
        final = records[-1]
        reference_green = all(
            row["reference_first_two"]["normalized_rmse"] <= 0.005
            and row["reference_first_two"]["cosine"] >= 0.9999
            and row["reference_first_two"]["normalized_max_abs_error"] <= 0.020
            for row in records
        )
        gates = {
            "P59_G1_all_checkpoint_contracts": all(
                row[name]["contract_match"]
                for row in contracts for name in ("gate", "up", "down")
            ),
            "P59_G2_bulk_serial_and_repeat_bit_exact": all(
                row["bulk_vs_serial_bit_exact"] and row["bulk_repeat_bit_exact"]
                for row in records
            ),
            "P59_G3_all_finite": all(row["finite"] for row in records),
            "P59_G4_independent_reference_green": reference_green,
            "P59_G5_resource_budget": all(
                (row.get("local_size_bytes") or 0) == 0
                and (row.get("num_regs") or 10_000) <= 64
                for row in resources.values()
            ),
            "P59_G6_bulk32_speedup_ge_2": final["speedup"] >= 2.0,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "payload_bytes_32_experts": payload_bytes,
            "payload_MiB_32_experts": payload_bytes / 2**20,
            "checkpoint_contracts": contracts,
            "records": records,
            "resource_audit": resources,
            "summary": {
                "bulk32_median_ms": final["bulk_timing_ms"]["p50"],
                "serial32_median_ms": final["serial_timing_ms"]["p50"],
                "bulk32_speedup": final["speedup"],
                "bulk32_assignments_per_ms": final["bulk_assignments_per_ms"],
            },
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNELS))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "records": [
            {
                "groups": row["groups"],
                "bulk_ms": row["bulk_timing_ms"]["p50"],
                "serial_ms": row["serial_timing_ms"]["p50"],
                "speedup": row["speedup"],
                "exact": row["bulk_vs_serial_bit_exact"],
                "reference_nrmse": row["reference_first_two"]["normalized_rmse"],
            }
            for row in payload.get("records", [])
        ],
        "resources": payload.get("resource_audit"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
