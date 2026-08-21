"""Phase63 real Ornith NVFP4 LM-head H4 benchmark."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase30e_shared_kernels import SharedOccupancyKernels
from s100_phase48_ornith_swiglu_h8 import _load_projection, _measure, _metrics
from s100_phase49_nvfp4_mfamily import NVFP4MFamilyWarp32


RESULTS = REPO / "pro_research" / "results" / "s100_phase63"
PREREG = REPO / "pro_research" / "S100_PHASE63_ORNITH_LM_HEAD_H4_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase63_ornith_lm_head_h4.py"
KERNEL_M4 = REPO / "pro_research" / "s100_phase49_nvfp4_mfamily.py"
KERNEL_R16 = REPO / "pro_research" / "s100_phase30e_shared_kernels.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--reps", type=int, default=31)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE63_ORNITH_LM_HEAD_H4.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase63_ornith_lm_head_h4",
        "status": "started",
        "started_utc": utc_now(),
        "snapshot": str(args.snapshot.resolve()),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    cp = None
    try:
        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        head = _load_projection(snapshot, index["weight_map"], "lm_head")
        rows, cols = 248320, 2048
        if head["codes_meta"]["shape"] != [rows, cols // 2]:
            raise RuntimeError(f"unexpected LM-head shape {head['codes_meta']['shape']}")

        import cupy as cp_module
        import sys

        src = REPO / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        cp = cp_module
        fused = FusedNVFP4()
        warp32 = NVFP4MFamilyWarp32()
        r16 = SharedOccupancyKernels()
        codes = cp.asarray(head["codes"])
        scales = cp.asarray(head["scales"])
        rng = np.random.default_rng(63000000)
        x = cp.asarray(rng.standard_normal((4, cols), dtype=np.float32))
        control = cp.empty((4, rows), dtype=cp.float32)
        warp_out = cp.empty_like(control)
        r16_out = cp.empty_like(control)
        repeat = cp.empty_like(control)

        def run_control() -> None:
            for row in range(4):
                fused.gemv_into(
                    control[row], codes, scales, x[row], head["global_scale"], rows, cols
                )

        def run_warp(target=warp_out) -> None:
            warp32.nvfp4(
                4, codes, scales, fused.e2m1, fused.e4m3, x, target,
                head["global_scale"], rows, cols,
            )

        def run_r16(target=r16_out) -> None:
            r16.nvfp4(
                codes, scales, fused.e2m1, fused.e4m3, x, target,
                head["global_scale"], rows, cols, 4, False, False,
            )

        run_control()
        run_warp()
        run_r16()
        cp.cuda.get_current_stream().synchronize()
        control_host = cp.asnumpy(control)
        candidate_hosts = {
            "warp32_m4": cp.asnumpy(warp_out),
            "r16_m4": cp.asnumpy(r16_out),
        }
        timings = {
            "h1_x4": _measure(cp, run_control, args.warmup, args.reps),
            "warp32_m4": _measure(cp, run_warp, args.warmup, args.reps),
            "r16_m4": _measure(cp, run_r16, args.warmup, args.reps),
        }
        controls = np.argmax(control_host, axis=1).astype(np.int64)
        candidates = {}
        for name, host in candidate_hosts.items():
            if name == "warp32_m4":
                run_warp(repeat)
            else:
                run_r16(repeat)
            cp.cuda.get_current_stream().synchronize()
            repeat_host = cp.asnumpy(repeat)
            candidates[name] = {
                "timing_ms": timings[name],
                "speedup_vs_h1_x4": timings["h1_x4"]["p50"] / timings[name]["p50"],
                "vs_control": _metrics(host, control_host),
                "control_argmax": controls.tolist(),
                "candidate_argmax": np.argmax(host, axis=1).astype(np.int64).tolist(),
                "argmax_exact": bool(np.array_equal(np.argmax(host, axis=1), controls)),
                "repeat_bit_exact": bool(np.array_equal(
                    host.view(np.uint32), repeat_host.view(np.uint32)
                )),
                "finite": bool(np.isfinite(host).all()),
            }
        eligible = [
            name for name, row in candidates.items()
            if row["finite"] and row["repeat_bit_exact"] and row["argmax_exact"]
            and row["vs_control"]["normalized_rmse"] <= 0.001
            and row["vs_control"]["cosine"] >= 0.999999
        ]
        selected = min(eligible, key=lambda name: timings[name]["p50"]) if eligible else None
        r16_resource = r16.resource_audit().get("m4_direct", {})
        warp_resource = warp32.resource_audit().get("M4", {})
        resources = {"warp32_m4": warp_resource, "r16_m4": r16_resource}
        selected_resource = resources.get(selected or "", {})
        gates = {
            "P63_G1_both_finite_deterministic": all(
                row["finite"] and row["repeat_bit_exact"] for row in candidates.values()
            ),
            "P63_G2_both_argmax_exact": all(row["argmax_exact"] for row in candidates.values()),
            "P63_G3_both_numerically_green": len(eligible) == 2,
            "P63_G4_selected_speedup_ge_2_5": bool(
                selected and candidates[selected]["speedup_vs_h1_x4"] >= 2.5
            ),
            "P63_G5_selected_resource_budget": bool(
                selected
                and (selected_resource.get("local_size_bytes") or 0) == 0
                and (selected_resource.get("num_regs") or 10_000) <= 64
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "shape": [rows, cols],
            "payload_bytes": int(head["codes"].nbytes + head["scales"].nbytes + 4),
            "control_timing_ms": timings["h1_x4"],
            "candidates": candidates,
            "resources": resources,
            "selected": selected,
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNEL_M4, KERNEL_R16))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "control_ms": (payload.get("control_timing_ms") or {}).get("p50"),
        "candidates": {
            name: {
                "ms": row["timing_ms"]["p50"],
                "speedup": row["speedup_vs_h1_x4"],
                "nrmse": row["vs_control"]["normalized_rmse"],
                "argmax_exact": row["argmax_exact"],
                "repeat": row["repeat_bit_exact"],
            } for name, row in payload.get("candidates", {}).items()
        },
        "selected": payload.get("selected"),
        "resources": payload.get("resources"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
