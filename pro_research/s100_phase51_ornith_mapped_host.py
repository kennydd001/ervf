"""Phase51: direct-UVA versus staged transport for complete Ornith experts."""
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
    _load_projection,
    _measure,
    _metrics,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase51"
PREREG = REPO / "pro_research" / "S100_PHASE51_ORNITH_MAPPED_HOST_PREREGISTRATION.md"
RESULTS52 = REPO / "pro_research" / "results" / "s100_phase52"
PREREG52 = REPO / "pro_research" / "S100_PHASE52_ORNITH_COLD_MAPPED_HOST_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase51_ornith_mapped_host.py"
KERNEL = REPO / "pro_research" / "s100_phase49_nvfp4_mfamily.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--phase", type=int, choices=(51, 52), default=51)
    parser.add_argument("--rotation", type=int, default=1)
    args = parser.parse_args()
    if args.rotation < 1:
        parser.error("--rotation must be >= 1")
    prereg = PREREG52 if args.phase == 52 else PREREG
    results = RESULTS52 if args.phase == 52 else RESULTS
    out = results / f"S100_PHASE{args.phase}_POTTOKAO_LAYER20_EXPERT0.json"
    payload: dict[str, Any] = {
        "kind": f"s100_phase{args.phase}_ornith_mapped_host",
        "status": "started",
        "started_utc": utc_now(),
        "snapshot": str(args.snapshot.resolve()),
        "preregistration": str(prereg.relative_to(REPO)),
        "host_record_rotation": int(args.rotation),
        "claim_boundary": "single complete expert transport only; no route-union or tok/s claim",
    }
    cp = None
    try:
        snapshot = args.snapshot.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
        weight_map = index["weight_map"]
        base = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
        host_proj = {
            name: _load_projection(snapshot, weight_map, base + suffix)
            for name, suffix in (
                ("gate", ".gate_proj"),
                ("up", ".up_proj"),
                ("down", ".down_proj"),
            )
        }
        rng = np.random.default_rng(51002000 + args.layer * 256 + args.expert)
        x_host = rng.standard_normal((8, 2048), dtype=np.float32)

        import cupy as cp_module

        cp = cp_module
        properties = cp.cuda.runtime.getDeviceProperties(0)
        l2_bytes = int(properties.get("l2CacheSize", 0))
        fused = FusedNVFP4()
        family = NVFP4MFamilyWarp32()
        swiglu_module = cp.RawModule(
            code=SWIGLU_SOURCE,
            options=("-std=c++14",),
            name_expressions=("swiglu_f32",),
        )
        swiglu = swiglu_module.get_function("swiglu_f32")
        x8 = cp.asarray(x_host)
        pinned_handles: list[Any] = []
        records: dict[str, dict[str, Any]] = {}

        def pinned_bank(source: np.ndarray) -> tuple[Any, np.ndarray]:
            memory = cp.cuda.alloc_pinned_memory(int(source.nbytes * args.rotation))
            view = np.frombuffer(
                memory, dtype=np.uint8, count=source.nbytes * args.rotation
            ).reshape(args.rotation, source.nbytes)
            for slot in range(args.rotation):
                view[slot] = source.reshape(-1)
            return memory, view

        for name, proj in host_proj.items():
            code_mem, code_view = pinned_bank(proj["codes"])
            scale_mem, scale_view = pinned_bank(proj["scales"])
            pinned_handles.extend((code_mem, scale_mem))
            records[name] = {
                "code_host": code_view,
                "scale_host": scale_view,
                "code_device": cp.asarray(proj["codes"]),
                "scale_device": cp.asarray(proj["scales"]),
                "code_stage": cp.empty(proj["codes"].size, dtype=cp.uint8),
                "scale_stage": cp.empty(proj["scales"].size, dtype=cp.uint8),
                "global_scale": proj["global_scale"],
            }

        def copy_to_stage(slot: int) -> None:
            stream = cp.cuda.get_current_stream()
            for row in records.values():
                code_host = row["code_host"][slot]
                scale_host = row["scale_host"][slot]
                cp.cuda.runtime.memcpyAsync(
                    row["code_stage"].data.ptr, code_host.ctypes.data,
                    code_host.nbytes, cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
                cp.cuda.runtime.memcpyAsync(
                    row["scale_stage"].data.ptr, scale_host.ctypes.data,
                    scale_host.nbytes, cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )

        def launch_swiglu(g, u, target, n: int) -> None:
            swiglu(((n + 255) // 256,), (256,), (g, u, target, np.int32(n)))

        rows: list[dict[str, Any]] = []
        for batch in (1, 2, 4, 8):
            xb = x8[:batch]
            gate_b = cp.empty((batch, 512), dtype=cp.float32)
            up_b = cp.empty((batch, 512), dtype=cp.float32)
            act_b = cp.empty((batch, 512), dtype=cp.float32)
            out_hot = cp.empty((batch, 2048), dtype=cp.float32)
            out_stage = cp.empty_like(out_hot)
            out_direct = cp.empty_like(out_hot)
            out_repeat = cp.empty_like(out_hot)

            def run_with(kind: str, target, slot: int = 0) -> None:
                def launch_projection(name: str, x, output, nrows: int, ncols: int) -> None:
                    row = records[name]
                    if batch == 1:
                        codes = (
                            np.uint64(row["code_host"][slot].ctypes.data)
                            if kind == "direct" else row[f"code_{kind}"]
                        )
                        scales = (
                            np.uint64(row["scale_host"][slot].ctypes.data)
                            if kind == "direct" else row[f"scale_{kind}"]
                        )
                        fused.gemv_into(
                            output.reshape(-1), codes, scales, x.reshape(-1),
                            row["global_scale"], nrows, ncols,
                        )
                    elif kind == "direct":
                        family.nvfp4_ptr(
                            batch, row["code_host"][slot].ctypes.data,
                            row["scale_host"][slot].ctypes.data, fused.e2m1, fused.e4m3,
                            x, output, row["global_scale"], nrows, ncols,
                        )
                    else:
                        family.nvfp4(
                            batch, row[f"code_{kind}"], row[f"scale_{kind}"],
                            fused.e2m1, fused.e4m3, x, output,
                            row["global_scale"], nrows, ncols,
                        )

                launch_projection("gate", xb, gate_b, 512, 2048)
                launch_projection("up", xb, up_b, 512, 2048)
                launch_swiglu(gate_b, up_b, act_b, batch * 512)
                launch_projection("down", act_b, target, 2048, 512)

            def run_hot() -> None:
                run_with("device", out_hot)

            stage_cursor = [0]
            direct_cursor = [0]

            def run_stage() -> None:
                slot = stage_cursor[0] % args.rotation
                stage_cursor[0] += 1
                copy_to_stage(slot)
                run_with("stage", out_stage)

            def run_direct(target=out_direct) -> None:
                slot = direct_cursor[0] % args.rotation
                direct_cursor[0] += 1
                run_with("direct", target, slot)

            run_hot()
            run_stage()
            run_direct(out_direct)
            cp.cuda.get_current_stream().synchronize()
            hot_host = cp.asnumpy(out_hot)
            stage_host = cp.asnumpy(out_stage)
            direct_host = cp.asnumpy(out_direct)
            run_direct(out_repeat)
            cp.cuda.get_current_stream().synchronize()
            repeat_host = cp.asnumpy(out_repeat)
            timings = {
                "hot": _measure(cp, run_hot, args.warmup, args.reps),
                "stage": _measure(cp, run_stage, args.warmup, args.reps),
                "direct": _measure(cp, run_direct, args.warmup, args.reps),
            }
            direct_ms = float(timings["direct"]["p50"])
            stage_ms = float(timings["stage"]["p50"])
            rows.append({
                "multiplicity": batch,
                "timings_ms": timings,
                "direct_over_stage": direct_ms / stage_ms,
                "direct_vs_hot": _metrics(direct_host, hot_host),
                "stage_vs_hot": _metrics(stage_host, hot_host),
                "direct_bitwise_repeat": bool(np.array_equal(direct_host, repeat_host)),
                "direct_finite": bool(np.isfinite(direct_host).all()),
            })

        by_m = {row["multiplicity"]: row for row in rows}
        correctness = all(
            row["direct_finite"]
            and row["direct_bitwise_repeat"]
            and row["direct_vs_hot"]["normalized_rmse"] <= 0.001
            and row["direct_vs_hot"]["normalized_max_abs_error"] <= 0.005
            for row in rows
        )
        complete_bytes = int(sum(
            proj["codes"].nbytes + proj["scales"].nbytes + proj["global_raw"].nbytes
            for proj in host_proj.values()
        ))
        working_set_bytes = complete_bytes * int(args.rotation)
        for row in rows:
            hot_ms = float(row["timings_ms"]["hot"]["p50"])
            stage_ms = float(row["timings_ms"]["stage"]["p50"])
            direct_ms = float(row["timings_ms"]["direct"]["p50"])
            direct_transport_ms = max(direct_ms - hot_ms, 1e-12)
            stage_transport_ms = max(stage_ms - hot_ms, 1e-12)
            row["derived"] = {
                "direct_speedup_over_stage": stage_ms / direct_ms,
                "direct_transport_increment_ms": direct_transport_ms,
                "stage_transport_increment_ms": stage_transport_ms,
                "direct_effective_record_GB_s": complete_bytes / direct_transport_ms / 1e6,
                "stage_effective_record_GB_s": complete_bytes / stage_transport_ms / 1e6,
            }
        if args.phase == 52:
            gates = {
                "P52_G1_rotation_working_set_ge_4x_l2": working_set_bytes >= 4 * l2_bytes,
                "P52_G2_direct_correct_finite_deterministic": correctness,
                "P52_G3_direct_m8_le_0_75_ms": float(by_m[8]["timings_ms"]["direct"]["p50"]) <= 0.75,
                "P52_G4_direct_m8_no_slower_than_stage": by_m[8]["direct_over_stage"] <= 1.0,
                "P52_G5_direct_m1_no_slower_than_stage": by_m[1]["direct_over_stage"] <= 1.0,
            }
        else:
            gates = {
                "P51_G1_direct_correct_finite_deterministic": correctness,
                "P51_G2_direct_m8_le_0_50_ms": float(by_m[8]["timings_ms"]["direct"]["p50"]) <= 0.50,
                "P51_G3_direct_m8_no_slower_than_stage": by_m[8]["direct_over_stage"] <= 1.0,
                "P51_G4_direct_m1_no_slower_than_stage": by_m[1]["direct_over_stage"] <= 1.0,
            }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "l2_bytes": l2_bytes,
            "complete_expert_payload_bytes": complete_bytes,
            "rotation_working_set_bytes": working_set_bytes,
            "rotation_working_set_over_l2": working_set_bytes / max(l2_bytes, 1),
            "records": rows,
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
        payload["environment"] = environment_snapshot((SCRIPT, prereg, KERNEL))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "gates": payload.get("gates"),
        "records": [
            {
                "M": row["multiplicity"],
                "hot_ms": row["timings_ms"]["hot"]["p50"],
                "stage_ms": row["timings_ms"]["stage"]["p50"],
                "direct_ms": row["timings_ms"]["direct"]["p50"],
                "direct_over_stage": row["direct_over_stage"],
            }
            for row in payload.get("records", [])
        ],
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
