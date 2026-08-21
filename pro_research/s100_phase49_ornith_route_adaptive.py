"""Phase49: route-multiplicity-specific complete Ornith expert benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic


SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from s100_phase49_nvfp4_mfamily import NVFP4MFamilyWarp32  # noqa: E402
from s100_phase48_ornith_swiglu_h8 import (  # noqa: E402
    SWIGLU_SOURCE,
    _decode,
    _load_projection,
    _measure,
    _metrics,
    _projection_contract,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase49"
PREREG = REPO / "pro_research" / "S100_PHASE49_ORNITH_ROUTE_ADAPTIVE_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase49_ornith_route_adaptive.py"
KERNEL = REPO / "pro_research" / "s100_phase49_nvfp4_mfamily.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--layer-root", default="model.layers")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--tag", default="pottokao_layer20_expert0")
    args = parser.parse_args()
    out = RESULTS / f"S100_PHASE49_{args.tag.upper()}.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase49_ornith_route_adaptive",
        "status": "started",
        "snapshot": str(args.snapshot.resolve()),
        "layer": int(args.layer),
        "expert": int(args.expert),
        "layer_root": args.layer_root,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "hot expert multiplicity curve only; no real route census or tok/s claim",
    }
    cp = None
    try:
        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        weight_map = index["weight_map"]
        base = f"{args.layer_root}.{args.layer}.mlp.experts.{args.expert}"
        gate = _load_projection(snapshot, weight_map, base + ".gate_proj")
        up = _load_projection(snapshot, weight_map, base + ".up_proj")
        down = _load_projection(snapshot, weight_map, base + ".down_proj")
        contracts = {
            "gate": _projection_contract(gate, 512, 2048),
            "up": _projection_contract(up, 512, 2048),
            "down": _projection_contract(down, 2048, 512),
        }
        rng = np.random.default_rng(49002000 + args.layer * 256 + args.expert)
        x_host = rng.standard_normal((8, 2048), dtype=np.float32)
        gate_ref = x_host @ _decode(gate, 512, 2048).T
        up_ref = x_host @ _decode(up, 512, 2048).T
        act_ref = (gate_ref / (1.0 + np.exp(-gate_ref))) * up_ref
        reference = act_ref @ _decode(down, 2048, 512).T

        import cupy as cp_module

        cp = cp_module
        fused = FusedNVFP4()
        family = NVFP4MFamilyWarp32()
        swiglu_module = cp.RawModule(
            code=SWIGLU_SOURCE,
            options=("-std=c++14",),
            name_expressions=("swiglu_f32",),
        )
        swiglu = swiglu_module.get_function("swiglu_f32")
        device_proj = {
            name: {
                "codes": cp.asarray(proj["codes"]),
                "scales": cp.asarray(proj["scales"]),
                "global_scale": proj["global_scale"],
            }
            for name, proj in (("gate", gate), ("up", up), ("down", down))
        }
        x8 = cp.asarray(x_host)
        gate1 = cp.empty(512, dtype=cp.float32)
        up1 = cp.empty(512, dtype=cp.float32)
        act1 = cp.empty(512, dtype=cp.float32)
        out1 = cp.empty(2048, dtype=cp.float32)

        def launch_swiglu(g, u, target, n: int) -> None:
            swiglu(((n + 255) // 256,), (256,), (g, u, target, np.int32(n)))

        def run_h1(row: int, target) -> None:
            fused.gemv_into(
                gate1, device_proj["gate"]["codes"], device_proj["gate"]["scales"],
                x8[row], device_proj["gate"]["global_scale"], 512, 2048,
            )
            fused.gemv_into(
                up1, device_proj["up"]["codes"], device_proj["up"]["scales"],
                x8[row], device_proj["up"]["global_scale"], 512, 2048,
            )
            launch_swiglu(gate1, up1, act1, 512)
            fused.gemv_into(
                target, device_proj["down"]["codes"], device_proj["down"]["scales"],
                act1, device_proj["down"]["global_scale"], 2048, 512,
            )

        records: list[dict[str, Any]] = []
        for batch in range(2, 9):
            xb = x8[:batch]
            gate_b = cp.empty((batch, 512), dtype=cp.float32)
            up_b = cp.empty((batch, 512), dtype=cp.float32)
            act_b = cp.empty((batch, 512), dtype=cp.float32)
            out_b = cp.empty((batch, 2048), dtype=cp.float32)
            repeat_b = cp.empty_like(out_b)
            h1_b = cp.empty_like(out_b)

            def run_group(target=out_b, b=batch, x=xb) -> None:
                family.nvfp4(
                    b, device_proj["gate"]["codes"], device_proj["gate"]["scales"],
                    fused.e2m1, fused.e4m3, x, gate_b,
                    device_proj["gate"]["global_scale"], 512, 2048,
                )
                family.nvfp4(
                    b, device_proj["up"]["codes"], device_proj["up"]["scales"],
                    fused.e2m1, fused.e4m3, x, up_b,
                    device_proj["up"]["global_scale"], 512, 2048,
                )
                launch_swiglu(gate_b, up_b, act_b, b * 512)
                family.nvfp4(
                    b, device_proj["down"]["codes"], device_proj["down"]["scales"],
                    fused.e2m1, fused.e4m3, act_b, target,
                    device_proj["down"]["global_scale"], 2048, 512,
                )

            def run_sequential() -> None:
                for row in range(batch):
                    run_h1(row, h1_b[row])

            run_group(out_b)
            run_sequential()
            cp.cuda.get_current_stream().synchronize()
            group_host = cp.asnumpy(out_b)
            h1_host = cp.asnumpy(h1_b)
            run_group(repeat_b)
            cp.cuda.get_current_stream().synchronize()
            repeat_host = cp.asnumpy(repeat_b)
            group_timing = _measure(cp, run_group, args.warmup, args.reps)
            h1_timing = _measure(cp, run_sequential, args.warmup, args.reps)
            group_ms = float(group_timing["p50"])
            h1_ms = float(h1_timing["p50"])
            records.append({
                "multiplicity": batch,
                "candidate_timing_ms": group_timing,
                "sequential_h1_timing_ms": h1_timing,
                "speedup": h1_ms / group_ms,
                "candidate_over_h1_per_row": group_ms / (h1_ms / batch),
                "vs_independent_reference": _metrics(group_host, reference[:batch]),
                "vs_sequential_h1": _metrics(group_host, h1_host),
                "bitwise_repeat": bool(np.array_equal(group_host, repeat_host)),
                "finite": bool(np.isfinite(group_host).all()),
            })

        all_reference = all(
            row["finite"]
            and row["vs_independent_reference"]["normalized_rmse"] <= 0.005
            and row["vs_independent_reference"]["cosine"] >= 0.9999
            and row["vs_independent_reference"]["normalized_max_abs_error"] <= 0.020
            for row in records
        )
        all_h1 = all(
            row["vs_sequential_h1"]["normalized_rmse"] <= 0.001
            and row["vs_sequential_h1"]["normalized_max_abs_error"] <= 0.005
            for row in records
        )
        beneficial = [row["multiplicity"] for row in records if row["speedup"] > 1.0]
        gates = {
            "P49_G1_checkpoint_contract": all(row["contract_match"] for row in contracts.values()),
            "P49_G2_all_independent_references_green": all_reference,
            "P49_G3_all_h1_controls_green": all_h1,
            "P49_G4_all_bitwise_repeat": all(row["bitwise_repeat"] for row in records),
            "P49_G5_all_m2_through_m8_faster_than_sequential_h1": len(beneficial) == 7,
            "P49_G6_first_beneficial_multiplicity_is_2": beneficial[:1] == [2],
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "checkpoint_contract": contracts,
            "records": records,
            "resource_audit": family.resource_audit(),
            "beneficial_multiplicities": beneficial,
            "dispatch_policy": {"M1": "production_ERVF_H1", "M2_to_M8": "exact_size_family"},
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNEL))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "gates": payload.get("gates"),
        "records": [
            {
                "multiplicity": row["multiplicity"],
                "candidate_p50_ms": row["candidate_timing_ms"]["p50"],
                "h1_control_p50_ms": row["sequential_h1_timing_ms"]["p50"],
                "speedup": row["speedup"],
                "reference_nrmse": row["vs_independent_reference"]["normalized_rmse"],
            }
            for row in payload.get("records", [])
        ],
        "resource_audit": payload.get("resource_audit"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
