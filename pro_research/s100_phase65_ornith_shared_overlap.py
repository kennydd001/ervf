"""Phase65 overlap real Ornith shared M4 with routed bulk H4."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase48_ornith_swiglu_h8 import _load_projection, _measure
from s100_phase49_nvfp4_mfamily import NVFP4MFamilyWarp32
from s100_phase59_ornith_bulk_expert import _load_experts, _stack
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1


RESULTS = REPO / "pro_research" / "results" / "s100_phase65"
PREREG = REPO / "pro_research" / "S100_PHASE65_ORNITH_SHARED_OVERLAP_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase65_ornith_shared_overlap.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=61)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE65_ORNITH_SHARED_OVERLAP.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase65_ornith_shared_overlap",
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
        weight_map = index["weight_map"]
        experts, _ = _load_experts(snapshot, weight_map, args.layer, 32)
        shared_base = f"model.layers.{args.layer}.mlp.shared_expert"
        shared_host = {
            name: _load_projection(snapshot, weight_map, shared_base + f".{name}_proj")
            for name in ("gate", "up", "down")
        }

        import cupy as cp_module
        import sys

        src = REPO / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        cp = cp_module
        lookup = FusedNVFP4()
        bulk = OrnithNVFP4BulkM1()
        m4 = NVFP4MFamilyWarp32()
        rng = np.random.default_rng(65000000 + args.layer)
        route_x = cp.asarray(rng.standard_normal((32, 2048), dtype=np.float32))
        shared_x = cp.asarray(rng.standard_normal((4, 2048), dtype=np.float32))
        routed = {}
        shared = {}
        for name in ("gate", "up", "down"):
            routed[name] = {
                "codes": cp.asarray(_stack(experts, name, "codes")),
                "scales": cp.asarray(_stack(experts, name, "scales")),
                "global": cp.asarray(np.asarray(
                    [row[name]["global_scale"] for row in experts], dtype=np.float32
                )),
            }
            shared[name] = {
                "codes": cp.asarray(shared_host[name]["codes"]),
                "scales": cp.asarray(shared_host[name]["scales"]),
                "global": shared_host[name]["global_scale"],
            }

        route_gate = cp.empty((32, 512), dtype=cp.float32)
        route_up = cp.empty_like(route_gate)
        route_act = cp.empty_like(route_gate)
        route_out = cp.empty((32, 2048), dtype=cp.float32)
        shared_gate = cp.empty((4, 512), dtype=cp.float32)
        shared_up = cp.empty_like(shared_gate)
        shared_act = cp.empty_like(shared_gate)
        shared_out = cp.empty((4, 2048), dtype=cp.float32)

        def run_routed() -> None:
            for name, target in (("gate", route_gate), ("up", route_up)):
                row = routed[name]
                bulk.nvfp4(
                    row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                    route_x, target, row["global"], 32, 512, 2048,
                )
            bulk.swiglu(route_gate, route_up, route_act, 32)
            row = routed["down"]
            bulk.nvfp4(
                row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                route_act, route_out, row["global"], 32, 2048, 512,
            )

        def run_shared() -> None:
            for name, target in (("gate", shared_gate), ("up", shared_up)):
                row = shared[name]
                m4.nvfp4(
                    4, row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                    shared_x, target, row["global"], 512, 2048,
                )
            bulk.swiglu(shared_gate, shared_up, shared_act, 4)
            row = shared["down"]
            m4.nvfp4(
                4, row["codes"], row["scales"], lookup.e2m1, lookup.e4m3,
                shared_act, shared_out, row["global"], 2048, 512,
            )

        shared_stream = cp.cuda.Stream(non_blocking=True)
        fork = cp.cuda.Event()
        shared_done = cp.cuda.Event()

        def run_serial() -> None:
            run_routed()
            run_shared()

        def run_overlap() -> None:
            main_stream = cp.cuda.get_current_stream()
            fork.record(main_stream)
            with shared_stream:
                shared_stream.wait_event(fork)
                run_shared()
                shared_done.record(shared_stream)
            run_routed()
            main_stream.wait_event(shared_done)

        run_serial()
        cp.cuda.get_current_stream().synchronize()
        serial_route = cp.asnumpy(route_out)
        serial_shared = cp.asnumpy(shared_out)
        run_overlap()
        cp.cuda.get_current_stream().synchronize()
        overlap_route = cp.asnumpy(route_out)
        overlap_shared = cp.asnumpy(shared_out)
        run_overlap()
        cp.cuda.get_current_stream().synchronize()
        repeat_route = cp.asnumpy(route_out)
        repeat_shared = cp.asnumpy(shared_out)
        timings = {
            "routed_isolated": _measure(cp, run_routed, args.warmup, args.reps),
            "shared_isolated": _measure(cp, run_shared, args.warmup, args.reps),
            "serial": _measure(cp, run_serial, args.warmup, args.reps),
            "overlap": _measure(cp, run_overlap, args.warmup, args.reps),
        }
        serial_ms = float(timings["serial"]["p50"])
        overlap_ms = float(timings["overlap"]["p50"])
        slow_branch = max(
            float(timings["routed_isolated"]["p50"]),
            float(timings["shared_isolated"]["p50"]),
        )
        gates = {
            "P65_G1_outputs_and_repeat_bit_exact": all((
                np.array_equal(serial_route.view(np.uint32), overlap_route.view(np.uint32)),
                np.array_equal(serial_shared.view(np.uint32), overlap_shared.view(np.uint32)),
                np.array_equal(overlap_route.view(np.uint32), repeat_route.view(np.uint32)),
                np.array_equal(overlap_shared.view(np.uint32), repeat_shared.view(np.uint32)),
            )),
            "P65_G2_all_finite": bool(
                np.isfinite(overlap_route).all() and np.isfinite(overlap_shared).all()
            ),
            "P65_G3_overlap_speedup_ge_1_05": serial_ms / overlap_ms >= 1.05,
            "P65_G4_overlap_within_5pct_of_slow_branch": overlap_ms <= 1.05 * slow_branch,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "timings_ms": timings,
            "summary": {
                "serial_ms": serial_ms,
                "overlap_ms": overlap_ms,
                "speedup": serial_ms / overlap_ms,
                "saved_ms_per_layer": serial_ms - overlap_ms,
                "projected_saved_ms_h4_40_layers": 40 * (serial_ms - overlap_ms),
                "overlap_over_slow_branch": overlap_ms / slow_branch,
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "isolated": {
            name: row["p50"] for name, row in (payload.get("timings_ms") or {}).items()
            if name.endswith("isolated")
        },
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
