"""Phase61 matched benchmark of one-warp and two-warp Ornith route buckets."""
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
from s100_phase60_ornith_route_adaptive import _token_ids
from s100_phase60_ornith_route_adaptive_kernels import OrnithNVFP4RouteAdaptive
from s100_phase61_ornith_pairwarp_kernels import OrnithNVFP4RoutePairWarp


RESULTS = REPO / "pro_research" / "results" / "s100_phase61"
PREREG = REPO / "pro_research" / "S100_PHASE61_ORNITH_PAIRWARP_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase61_ornith_pairwarp.py"
KERNELS = REPO / "pro_research" / "s100_phase61_ornith_pairwarp_kernels.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=61)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE61_ORNITH_PAIRWARP.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase61_ornith_pairwarp",
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
        one = OrnithNVFP4RouteAdaptive()
        pair = OrnithNVFP4RoutePairWarp()
        assignment = OrnithNVFP4BulkM1()
        rng = np.random.default_rng(61000000 + args.layer)
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
            assignment_slots = np.repeat(slots_host, multiplicity)
            slots = cp.asarray(slots_host)
            tokens = cp.asarray(tokens_host)
            down_ids = cp.arange(assignments, dtype=cp.int32)
            assignment_x = x4[tokens]

            buffers = {}
            for label in ("one", "pair"):
                buffers[label] = {
                    "gate": cp.empty((groups, multiplicity, 512), dtype=cp.float32),
                    "up": cp.empty((groups, multiplicity, 512), dtype=cp.float32),
                    "act": cp.empty((groups, multiplicity, 512), dtype=cp.float32),
                    "out": cp.empty((groups, multiplicity, 2048), dtype=cp.float32),
                }
            assignment_proj = {
                name: {
                    "codes": bank[name]["codes"][assignment_slots],
                    "scales": bank[name]["scales"][assignment_slots],
                    "global": bank[name]["global"][assignment_slots],
                } for name in ("gate", "up", "down")
            }
            bg = cp.empty((assignments, 512), dtype=cp.float32)
            bu = cp.empty_like(bg)
            ba = cp.empty_like(bg)
            bo = cp.empty((assignments, 2048), dtype=cp.float32)

            def run_family(family, label: str) -> None:
                b = buffers[label]
                for name in ("gate", "up"):
                    row = bank[name]
                    family.nvfp4(
                        multiplicity, row["codes"], row["scales"],
                        lookup.e2m1, lookup.e4m3, x4, b[name], row["global"],
                        slots, tokens, groups, 512, 2048,
                    )
                assignment.swiglu(b["gate"], b["up"], b["act"], assignments)
                row = bank["down"]
                family.nvfp4(
                    multiplicity, row["codes"], row["scales"],
                    lookup.e2m1, lookup.e4m3,
                    b["act"].reshape(assignments, 512), b["out"], row["global"],
                    slots, down_ids, groups, 2048, 512,
                )

            def run_assignment() -> None:
                for name, target in (("gate", bg), ("up", bu)):
                    row = assignment_proj[name]
                    assignment.nvfp4(
                        row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                        assignment_x, target, row["global"], assignments, 512, 2048,
                    )
                assignment.swiglu(bg, bu, ba, assignments)
                row = assignment_proj["down"]
                assignment.nvfp4(
                    row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                    ba, bo, row["global"], assignments, 2048, 512,
                )

            run_family(one, "one")
            run_family(pair, "pair")
            run_assignment()
            cp.cuda.get_current_stream().synchronize()
            one_host = cp.asnumpy(buffers["one"]["out"])
            pair_host = cp.asnumpy(buffers["pair"]["out"])
            run_family(pair, "pair")
            cp.cuda.get_current_stream().synchronize()
            repeat_host = cp.asnumpy(buffers["pair"]["out"])
            one_timing = _measure(cp, lambda: run_family(one, "one"), args.warmup, args.reps)
            pair_timing = _measure(cp, lambda: run_family(pair, "pair"), args.warmup, args.reps)
            assignment_timing = _measure(cp, run_assignment, args.warmup, args.reps)
            records.append({
                "multiplicity": multiplicity,
                "groups": groups,
                "assignments": assignments,
                "onewarp_timing_ms": one_timing,
                "pairwarp_timing_ms": pair_timing,
                "assignment_timing_ms": assignment_timing,
                "pair_speedup_vs_one": one_timing["p50"] / pair_timing["p50"],
                "pair_speedup_vs_assignment": assignment_timing["p50"] / pair_timing["p50"],
                "pair_vs_one_bit_exact": bool(np.array_equal(
                    pair_host.view(np.uint32), one_host.view(np.uint32)
                )),
                "pair_repeat_bit_exact": bool(np.array_equal(
                    pair_host.view(np.uint32), repeat_host.view(np.uint32)
                )),
                "finite": bool(np.isfinite(pair_host).all()),
            })

        resources = pair.resource_audit()
        m4 = records[-1]
        gates = {
            "P61_G1_all_exact_and_repeat_exact": all(
                row["pair_vs_one_bit_exact"] and row["pair_repeat_bit_exact"]
                for row in records
            ),
            "P61_G2_all_finite": all(row["finite"] for row in records),
            "P61_G3_resource_budget": all(
                (row.get("local_size_bytes") or 0) == 0
                and (row.get("num_regs") or 10_000) <= 56
                for row in resources.values()
            ),
            "P61_G4_pair_m4_ge_1_05_vs_one": m4["pair_speedup_vs_one"] >= 1.05,
            "P61_G5_pair_m4_ge_1_15_vs_assignment": m4["pair_speedup_vs_assignment"] >= 1.15,
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
            "one_ms": row["onewarp_timing_ms"]["p50"],
            "pair_ms": row["pairwarp_timing_ms"]["p50"],
            "assignment_ms": row["assignment_timing_ms"]["p50"],
            "pair_vs_one": row["pair_speedup_vs_one"],
            "pair_vs_assignment": row["pair_speedup_vs_assignment"],
            "exact": row["pair_vs_one_bit_exact"],
        } for row in payload.get("records", [])],
        "resources": payload.get("resource_audit"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
