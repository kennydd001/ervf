"""Phase60 benchmark of cache-indirect, exact-multiplicity Ornith H4 routes."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase48_ornith_swiglu_h8 import _measure
from s100_phase59_ornith_bulk_expert import _load_experts, _stack
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1
from s100_phase60_ornith_route_adaptive_kernels import OrnithNVFP4RouteAdaptive


RESULTS = REPO / "pro_research" / "results" / "s100_phase60"
PREREG = REPO / "pro_research" / "S100_PHASE60_ORNITH_ROUTE_ADAPTIVE_BULK_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase60_ornith_route_adaptive.py"
KERNELS = REPO / "pro_research" / "s100_phase60_ornith_route_adaptive_kernels.py"


def _token_ids(groups: int, multiplicity: int) -> np.ndarray:
    rows = []
    for group in range(groups):
        start = group % 4
        rows.extend((start + offset) % 4 for offset in range(multiplicity))
    return np.asarray(rows, dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=61)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE60_ORNITH_ROUTE_ADAPTIVE_BULK.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase60_ornith_route_adaptive_bulk",
        "status": "started",
        "started_utc": utc_now(),
        "snapshot": str(args.snapshot.resolve()),
        "layer": int(args.layer),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    cp = None
    try:
        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        experts, _ = _load_experts(snapshot, index["weight_map"], args.layer, 32)

        import cupy as cp_module
        import sys

        src = REPO / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        cp = cp_module
        lookup = FusedNVFP4()
        assignment = OrnithNVFP4BulkM1()
        adaptive = OrnithNVFP4RouteAdaptive()
        rng = np.random.default_rng(60000000 + args.layer)
        x4 = cp.asarray(rng.standard_normal((4, 2048), dtype=np.float32))
        bank = {}
        for name in ("gate", "up", "down"):
            bank[name] = {
                "codes": cp.asarray(_stack(experts, name, "codes")),
                "scales": cp.asarray(_stack(experts, name, "scales")),
                "global": cp.asarray(np.asarray(
                    [row[name]["global_scale"] for row in experts], dtype=np.float32
                )),
            }

        records = []
        for multiplicity in range(1, 5):
            groups = 32 // multiplicity
            assignments = groups * multiplicity
            slots_host = np.arange(groups, dtype=np.int32)
            tokens_host = _token_ids(groups, multiplicity)
            assignment_slots_host = np.repeat(slots_host, multiplicity)
            slots = cp.asarray(slots_host)
            token_ids = cp.asarray(tokens_host)
            down_ids = cp.arange(assignments, dtype=cp.int32)
            assignment_x = x4[token_ids]

            adaptive_device = {}
            assignment_device = {}
            for name in ("gate", "up", "down"):
                row = bank[name]
                adaptive_device[name] = row
                assignment_device[name] = {
                    "codes": row["codes"][assignment_slots_host],
                    "scales": row["scales"][assignment_slots_host],
                    "global": row["global"][assignment_slots_host],
                }

            ag = cp.empty((groups, multiplicity, 512), dtype=cp.float32)
            au = cp.empty_like(ag)
            aa = cp.empty_like(ag)
            ao = cp.empty((groups, multiplicity, 2048), dtype=cp.float32)
            ao_repeat = cp.empty_like(ao)
            bg = cp.empty((assignments, 512), dtype=cp.float32)
            bu = cp.empty_like(bg)
            ba = cp.empty_like(bg)
            bo = cp.empty((assignments, 2048), dtype=cp.float32)

            def run_adaptive(target=ao) -> None:
                for name, inp, output, ids, rows, cols in (
                    ("gate", x4, ag, token_ids, 512, 2048),
                    ("up", x4, au, token_ids, 512, 2048),
                ):
                    row = adaptive_device[name]
                    adaptive.nvfp4(
                        multiplicity, row["codes"], row["scales"],
                        lookup.e2m1, lookup.e4m3, inp, output, row["global"],
                        slots, ids, groups, rows, cols,
                    )
                assignment.swiglu(ag, au, aa, assignments)
                row = adaptive_device["down"]
                adaptive.nvfp4(
                    multiplicity, row["codes"], row["scales"],
                    lookup.e2m1, lookup.e4m3, aa.reshape(assignments, 512), target,
                    row["global"], slots, down_ids, groups, 2048, 512,
                )

            def run_assignment() -> None:
                for name, output in (("gate", bg), ("up", bu)):
                    row = assignment_device[name]
                    assignment.nvfp4(
                        row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                        assignment_x, output, row["global"], assignments, 512, 2048,
                    )
                assignment.swiglu(bg, bu, ba, assignments)
                row = assignment_device["down"]
                assignment.nvfp4(
                    row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                    ba, bo, row["global"], assignments, 2048, 512,
                )

            run_adaptive(ao)
            run_assignment()
            cp.cuda.get_current_stream().synchronize()
            adaptive_host = cp.asnumpy(ao).reshape(assignments, 2048)
            assignment_host = cp.asnumpy(bo)
            run_adaptive(ao_repeat)
            cp.cuda.get_current_stream().synchronize()
            repeat_host = cp.asnumpy(ao_repeat).reshape(assignments, 2048)
            adaptive_timing = _measure(cp, run_adaptive, args.warmup, args.reps)
            assignment_timing = _measure(cp, run_assignment, args.warmup, args.reps)
            candidate_ms = float(adaptive_timing["p50"])
            control_ms = float(assignment_timing["p50"])
            records.append({
                "multiplicity": multiplicity,
                "expert_groups": groups,
                "assignments": assignments,
                "adaptive_timing_ms": adaptive_timing,
                "assignment_m1_timing_ms": assignment_timing,
                "speedup": control_ms / candidate_ms,
                "candidate_vs_assignment_bit_exact": bool(np.array_equal(
                    adaptive_host.view(np.uint32), assignment_host.view(np.uint32)
                )),
                "candidate_repeat_bit_exact": bool(np.array_equal(
                    adaptive_host.view(np.uint32), repeat_host.view(np.uint32)
                )),
                "finite": bool(np.isfinite(adaptive_host).all()),
            })

        resources = adaptive.resource_audit()
        by_m = {row["multiplicity"]: row for row in records}
        gates = {
            "P60_G1_all_exact_and_repeat_exact": all(
                row["candidate_vs_assignment_bit_exact"]
                and row["candidate_repeat_bit_exact"] for row in records
            ),
            "P60_G2_all_finite": all(row["finite"] for row in records),
            "P60_G3_m1_indirect_overhead_le_15pct": by_m[1]["speedup"] >= 1 / 1.15,
            "P60_G4_m2_m3_m4_all_faster": all(by_m[m]["speedup"] > 1 for m in (2, 3, 4)),
            "P60_G5_m4_speedup_ge_1_15": by_m[4]["speedup"] >= 1.15,
            "P60_G6_resource_budget": all(
                (row.get("local_size_bytes") or 0) == 0
                and (row.get("num_regs") or 10_000) <= 64
                for row in resources.values()
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "records": records,
            "resource_audit": resources,
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
        "records": [{
            "M": row["multiplicity"],
            "groups": row["expert_groups"],
            "assignments": row["assignments"],
            "adaptive_ms": row["adaptive_timing_ms"]["p50"],
            "assignment_ms": row["assignment_m1_timing_ms"]["p50"],
            "speedup": row["speedup"],
            "exact": row["candidate_vs_assignment_bit_exact"],
        } for row in payload.get("records", [])],
        "resources": payload.get("resource_audit"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
