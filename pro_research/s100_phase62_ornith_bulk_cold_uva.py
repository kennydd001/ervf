"""Phase62 cold rotating direct-UVA versus stage for bulk Ornith expert misses."""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase48_ornith_swiglu_h8 import _measure
from s100_phase59_ornith_bulk_expert import _load_experts, _stack
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1


RESULTS = REPO / "pro_research" / "results" / "s100_phase62"
PREREG = REPO / "pro_research" / "S100_PHASE62_ORNITH_BULK_COLD_UVA_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase62_ornith_bulk_cold_uva.py"
KERNELS = REPO / "pro_research" / "s100_phase59_ornith_bulk_expert_kernels.py"
GROUP_COUNTS = (1, 4, 8, 16, 32)


def _pinned_rotating(cp, source: np.ndarray, rotations: int):
    memory = cp.cuda.alloc_pinned_memory(int(source.nbytes * rotations))
    view = np.frombuffer(
        memory, dtype=np.uint8, count=source.nbytes * rotations
    ).reshape(rotations, source.nbytes)
    flat = source.reshape(-1)
    for rotation in range(rotations):
        view[rotation] = flat
    return memory, view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--reps", type=int, default=41)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE62_ORNITH_BULK_COLD_UVA.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase62_ornith_bulk_cold_uva",
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
        kernels = OrnithNVFP4BulkM1()
        props = cp.cuda.runtime.getDeviceProperties(0)
        l2_bytes = int(props.get("l2CacheSize", 0))
        rng = np.random.default_rng(62000000 + args.layer)
        x_all = cp.asarray(rng.standard_normal((32, 2048), dtype=np.float32))
        host = {}
        hot = {}
        globals_device = {}
        for name in ("gate", "up", "down"):
            host[name] = {
                "codes": _stack(experts, name, "codes"),
                "scales": _stack(experts, name, "scales"),
            }
            hot[name] = {
                "codes": cp.asarray(host[name]["codes"]),
                "scales": cp.asarray(host[name]["scales"]),
            }
            globals_device[name] = cp.asarray(np.asarray(
                [row[name]["global_scale"] for row in experts], dtype=np.float32
            ))

        records = []
        pinned_handles = []
        for groups in GROUP_COUNTS:
            one_set_bytes = int(sum(
                row[key][:groups].nbytes
                for row in host.values() for key in ("codes", "scales")
            ))
            rotations = max(2, math.ceil((4 * l2_bytes) / one_set_bytes) + 1)
            direct = {}
            staged = {}
            local_handles = []
            for name in ("gate", "up", "down"):
                direct[name] = {}
                staged[name] = {}
                for key in ("codes", "scales"):
                    source = host[name][key][:groups]
                    handle, view = _pinned_rotating(cp, source, rotations)
                    local_handles.append(handle)
                    direct[name][key] = view
                    staged[name][key] = cp.empty(source.shape, dtype=cp.uint8)
            pinned_handles.extend(local_handles)

            x = x_all[:groups]
            buffers = {
                label: {
                    "gate": cp.empty((groups, 512), dtype=cp.float32),
                    "up": cp.empty((groups, 512), dtype=cp.float32),
                    "act": cp.empty((groups, 512), dtype=cp.float32),
                    "out": cp.empty((groups, 2048), dtype=cp.float32),
                } for label in ("hot", "direct", "stage", "repeat")
            }

            def run_compute(kind: str, rotation: int = 0, target_label: str | None = None) -> None:
                label = target_label or kind
                b = buffers[label]
                for name in ("gate", "up"):
                    if kind == "direct":
                        kernels.nvfp4_ptr(
                            direct[name]["codes"][rotation].ctypes.data,
                            direct[name]["scales"][rotation].ctypes.data,
                            lookup.e2m1, lookup.e4m3, x, b[name],
                            globals_device[name][:groups], groups, 512, 2048,
                        )
                    else:
                        source = hot[name] if kind == "hot" else staged[name]
                        kernels.nvfp4(
                            source["codes"], source["scales"], lookup.e2m1, lookup.e4m3,
                            x, b[name], globals_device[name][:groups], groups, 512, 2048,
                        )
                kernels.swiglu(b["gate"], b["up"], b["act"], groups)
                name = "down"
                if kind == "direct":
                    kernels.nvfp4_ptr(
                        direct[name]["codes"][rotation].ctypes.data,
                        direct[name]["scales"][rotation].ctypes.data,
                        lookup.e2m1, lookup.e4m3, b["act"], b["out"],
                        globals_device[name][:groups], groups, 2048, 512,
                    )
                else:
                    source = hot[name] if kind == "hot" else staged[name]
                    kernels.nvfp4(
                        source["codes"], source["scales"], lookup.e2m1, lookup.e4m3,
                        b["act"], b["out"], globals_device[name][:groups],
                        groups, 2048, 512,
                    )

            direct_cursor = [0]
            stage_cursor = [0]

            def run_direct(target_label="direct") -> None:
                rotation = direct_cursor[0] % rotations
                direct_cursor[0] += 1
                run_compute("direct", rotation, target_label)

            def run_stage() -> None:
                rotation = stage_cursor[0] % rotations
                stage_cursor[0] += 1
                stream = cp.cuda.get_current_stream()
                for name in ("gate", "up", "down"):
                    for key in ("codes", "scales"):
                        source = direct[name][key][rotation]
                        cp.cuda.runtime.memcpyAsync(
                            staged[name][key].data.ptr, source.ctypes.data, source.nbytes,
                            cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
                        )
                run_compute("stage")

            run_compute("hot")
            run_direct()
            run_stage()
            cp.cuda.get_current_stream().synchronize()
            hot_host = cp.asnumpy(buffers["hot"]["out"])
            direct_host = cp.asnumpy(buffers["direct"]["out"])
            stage_host = cp.asnumpy(buffers["stage"]["out"])
            run_direct("repeat")
            cp.cuda.get_current_stream().synchronize()
            repeat_host = cp.asnumpy(buffers["repeat"]["out"])
            timings = {
                "hot": _measure(cp, lambda: run_compute("hot"), args.warmup, args.reps),
                "direct": _measure(cp, run_direct, args.warmup, args.reps),
                "stage": _measure(cp, run_stage, args.warmup, args.reps),
            }
            direct_ms = float(timings["direct"]["p50"])
            stage_ms = float(timings["stage"]["p50"])
            hot_ms = float(timings["hot"]["p50"])
            records.append({
                "groups": groups,
                "one_set_bytes": one_set_bytes,
                "rotations": rotations,
                "working_set_bytes": one_set_bytes * rotations,
                "working_set_over_l2": one_set_bytes * rotations / max(l2_bytes, 1),
                "timings_ms": timings,
                "direct_over_stage": direct_ms / stage_ms,
                "direct_speedup_over_stage": stage_ms / direct_ms,
                "direct_transport_increment_ms": max(0.0, direct_ms - hot_ms),
                "stage_transport_increment_ms": max(0.0, stage_ms - hot_ms),
                "direct_vs_hot_bit_exact": bool(np.array_equal(
                    direct_host.view(np.uint32), hot_host.view(np.uint32)
                )),
                "stage_vs_hot_bit_exact": bool(np.array_equal(
                    stage_host.view(np.uint32), hot_host.view(np.uint32)
                )),
                "direct_repeat_bit_exact": bool(np.array_equal(
                    direct_host.view(np.uint32), repeat_host.view(np.uint32)
                )),
                "finite": bool(np.isfinite(direct_host).all() and np.isfinite(stage_host).all()),
            })
            del direct, staged, buffers
            pinned_handles = pinned_handles[:-len(local_handles)]
            del local_handles
            cp.get_default_memory_pool().free_all_blocks()

        by_groups = {row["groups"]: row for row in records}
        gates = {
            "P62_G1_all_working_sets_ge_4x_l2": all(
                row["working_set_over_l2"] >= 4 for row in records
            ),
            "P62_G2_all_exact_repeat_finite": all(
                row["direct_vs_hot_bit_exact"] and row["stage_vs_hot_bit_exact"]
                and row["direct_repeat_bit_exact"] and row["finite"] for row in records
            ),
            "P62_G3_direct_no_slower_at_1_and_32": all(
                by_groups[g]["direct_over_stage"] <= 1 for g in (1, 32)
            ),
            "P62_G4_direct32_below_5ms": by_groups[32]["timings_ms"]["direct"]["p50"] < 5,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "l2_bytes": l2_bytes,
            "records": records,
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
            "groups": row["groups"],
            "rotation": row["rotations"],
            "ws_l2": row["working_set_over_l2"],
            "hot_ms": row["timings_ms"]["hot"]["p50"],
            "direct_ms": row["timings_ms"]["direct"]["p50"],
            "stage_ms": row["timings_ms"]["stage"]["p50"],
            "direct_vs_stage": row["direct_speedup_over_stage"],
            "exact": row["direct_vs_hot_bit_exact"],
        } for row in payload.get("records", [])],
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
