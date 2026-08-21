"""Phase48: execute one complete real Ornith NVFP4 SwiGLU expert at H1/H8."""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import REPO, environment_snapshot, percentiles, utc_now, write_json_atomic


SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_lab.lightningstream_nemotron import nvfp4  # noqa: E402
from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from s100_phase33_nvfp4_m8 import NVFP4M8Warp32  # noqa: E402


RESULTS = REPO / "pro_research" / "results" / "s100_phase48"
PREREG = REPO / "pro_research" / "S100_PHASE48_ORNITH_SWIGLU_H8_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase48_ornith_swiglu_h8.py"
PHASE33 = REPO / "pro_research" / "s100_phase33_nvfp4_m8.py"
FUSED = SRC / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py"
REVISION = "d60d98b0b2feeabca19196005f4ac378279e2f25"

SWIGLU_SOURCE = r"""
extern "C" __global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
"""


@lru_cache(maxsize=None)
def _header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise IOError(f"short safetensors header length: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        raw = handle.read(header_len)
    if len(raw) != header_len:
        raise IOError(f"short safetensors header: {path}")
    return header_len, json.loads(raw.decode("utf-8"))


def _read_tensor(
    snapshot: Path,
    weight_map: dict[str, str],
    name: str,
) -> tuple[np.ndarray, dict[str, Any], str]:
    shard = weight_map[name]
    path = snapshot / shard
    header_len, header = _header(path)
    meta = header[name]
    begin, end = (int(value) for value in meta["data_offsets"])
    nbytes = end - begin
    with path.open("rb") as handle:
        handle.seek(8 + header_len + begin)
        raw = handle.read(nbytes)
    if len(raw) != nbytes:
        raise IOError(f"short tensor read {name}: {len(raw)} != {nbytes}")
    return np.frombuffer(raw, dtype=np.uint8).copy(), meta, shard


def _load_projection(
    snapshot: Path,
    weight_map: dict[str, str],
    prefix: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {"prefix": prefix}
    for key, suffix in (
        ("codes", ".weight"),
        ("scales", ".weight_scale"),
        ("global_raw", ".weight_scale_2"),
    ):
        raw, meta, shard = _read_tensor(snapshot, weight_map, prefix + suffix)
        values[key] = raw
        values[key + "_meta"] = meta
        values[key + "_shard"] = shard
    values["global_scale"] = float(values["global_raw"].view("<f4")[0])
    return values


def _projection_contract(proj: dict[str, Any], rows: int, cols: int) -> dict[str, Any]:
    observed = {
        "weight_dtype": proj["codes_meta"]["dtype"],
        "weight_shape": proj["codes_meta"]["shape"],
        "scale_dtype": proj["scales_meta"]["dtype"],
        "scale_shape": proj["scales_meta"]["shape"],
        "global_dtype": proj["global_raw_meta"]["dtype"],
        "global_shape": proj["global_raw_meta"]["shape"],
        "shard": proj["codes_shard"],
        "payload_bytes": int(
            proj["codes"].nbytes + proj["scales"].nbytes + proj["global_raw"].nbytes
        ),
        "global_scale": proj["global_scale"],
    }
    expected = {
        "weight_dtype": "U8",
        "weight_shape": [rows, cols // 2],
        "scale_dtype": "F8_E4M3",
        "scale_shape": [rows, cols // 16],
        "global_dtype": "F32",
        "global_shape": [],
    }
    observed["contract_match"] = all(observed[key] == value for key, value in expected.items())
    observed["expected"] = expected
    return observed


def _decode(proj: dict[str, Any], rows: int, cols: int) -> np.ndarray:
    values = nvfp4.dequantize(
        proj["codes"], proj["scales"], proj["global_scale"], implementation="bits"
    )
    return values.astype(np.float32).reshape(rows, cols)


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    a = np.asarray(candidate, dtype=np.float64).reshape(-1)
    b = np.asarray(reference, dtype=np.float64).reshape(-1)
    delta = a - b
    rmse = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(b * b)))
    max_abs = float(np.max(np.abs(delta)))
    reference_max = float(np.max(np.abs(b)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "rmse": rmse,
        "reference_rms": reference_rms,
        "normalized_rmse": rmse / max(reference_rms, 1e-30),
        "cosine": float(np.dot(a, b) / denom) if denom else 1.0,
        "max_abs_error": max_abs,
        "reference_max_abs": reference_max,
        "normalized_max_abs_error": max_abs / max(reference_max, 1e-30),
    }


def _measure(cp, fn: Callable[[], None], warmup: int, reps: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()
    samples: list[float] = []
    for _ in range(reps):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)))
    summary = percentiles(samples)
    summary["samples_ms"] = samples
    summary["min"] = float(min(samples))
    return summary


def _pinned_copy(cp, source: np.ndarray) -> tuple[Any, np.ndarray]:
    memory = cp.cuda.alloc_pinned_memory(int(source.nbytes))
    view = np.frombuffer(memory, dtype=np.uint8, count=source.nbytes)
    view[:] = source.reshape(-1)
    return memory, view


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
    out = RESULTS / f"S100_PHASE48_{args.tag.upper()}.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase48_ornith_swiglu_h8",
        "status": "started",
        "tag": args.tag,
        "snapshot": str(args.snapshot.resolve()),
        "revision": REVISION,
        "layer": int(args.layer),
        "expert": int(args.expert),
        "layer_root": args.layer_root,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "one real maximum-reuse routed expert only; no route census, full decoder, "
            "DFlash acceptance or whole-model tok/s claim"
        ),
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
        payload_bytes = int(sum(row["payload_bytes"] for row in contracts.values()))

        rng = np.random.default_rng(48002000 + args.layer * 256 + args.expert)
        x_host = rng.standard_normal((8, 2048), dtype=np.float32)
        gate_ref = x_host @ _decode(gate, 512, 2048).T
        up_ref = x_host @ _decode(up, 512, 2048).T
        act_ref = (gate_ref / (1.0 + np.exp(-gate_ref))) * up_ref
        reference = act_ref @ _decode(down, 2048, 512).T

        import cupy as cp_module

        cp = cp_module
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        capability = [int(props["major"]), int(props["minor"])]
        fused = FusedNVFP4()
        m8 = NVFP4M8Warp32()
        swiglu_module = cp.RawModule(
            code=SWIGLU_SOURCE,
            options=("-std=c++14",),
            name_expressions=("swiglu_f32",),
        )
        swiglu = swiglu_module.get_function("swiglu_f32")

        device_proj: dict[str, dict[str, Any]] = {}
        for name, proj in (("gate", gate), ("up", up), ("down", down)):
            device_proj[name] = {
                "codes": cp.asarray(proj["codes"]),
                "scales": cp.asarray(proj["scales"]),
                "global_scale": proj["global_scale"],
            }
        x8 = cp.asarray(x_host)
        gate8 = cp.empty((8, 512), dtype=cp.float32)
        up8 = cp.empty((8, 512), dtype=cp.float32)
        act8 = cp.empty((8, 512), dtype=cp.float32)
        out8 = cp.empty((8, 2048), dtype=cp.float32)
        out8_repeat = cp.empty_like(out8)
        gate1 = cp.empty(512, dtype=cp.float32)
        up1 = cp.empty(512, dtype=cp.float32)
        act1 = cp.empty(512, dtype=cp.float32)
        out1 = cp.empty(2048, dtype=cp.float32)
        out_h1 = cp.empty((8, 2048), dtype=cp.float32)

        def launch_swiglu(g, u, target, n: int) -> None:
            swiglu(((n + 255) // 256,), (256,), (g, u, target, np.int32(n)))

        def run_m8(target=out8, projections=device_proj) -> None:
            m8.nvfp4(
                projections["gate"]["codes"], projections["gate"]["scales"],
                fused.e2m1, fused.e4m3, x8, gate8,
                projections["gate"]["global_scale"], 512, 2048,
            )
            m8.nvfp4(
                projections["up"]["codes"], projections["up"]["scales"],
                fused.e2m1, fused.e4m3, x8, up8,
                projections["up"]["global_scale"], 512, 2048,
            )
            launch_swiglu(gate8, up8, act8, 8 * 512)
            m8.nvfp4(
                projections["down"]["codes"], projections["down"]["scales"],
                fused.e2m1, fused.e4m3, act8, target,
                projections["down"]["global_scale"], 2048, 512,
            )

        def run_h1_row(row: int, target) -> None:
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

        def run_h1x8() -> None:
            for row in range(8):
                run_h1_row(row, out_h1[row])

        run_h1x8()
        run_m8(out8)
        cp.cuda.get_current_stream().synchronize()
        h1_host = cp.asnumpy(out_h1)
        m8_host = cp.asnumpy(out8)
        run_m8(out8_repeat)
        cp.cuda.get_current_stream().synchronize()
        repeat_host = cp.asnumpy(out8_repeat)

        pinned: list[Any] = []
        staged_proj: dict[str, dict[str, Any]] = {}
        for name, proj in (("gate", gate), ("up", up), ("down", down)):
            code_mem, code_host = _pinned_copy(cp, proj["codes"])
            scale_mem, scale_host = _pinned_copy(cp, proj["scales"])
            pinned.extend((code_mem, scale_mem))
            staged_proj[name] = {
                "codes": cp.empty(proj["codes"].size, dtype=cp.uint8),
                "scales": cp.empty(proj["scales"].size, dtype=cp.uint8),
                "global_scale": proj["global_scale"],
                "codes_host": code_host,
                "scales_host": scale_host,
            }

        def run_staged_m8() -> None:
            stream = cp.cuda.get_current_stream()
            for name in ("gate", "up", "down"):
                row = staged_proj[name]
                cp.cuda.runtime.memcpyAsync(
                    row["codes"].data.ptr, row["codes_host"].ctypes.data,
                    row["codes_host"].nbytes, cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
                cp.cuda.runtime.memcpyAsync(
                    row["scales"].data.ptr, row["scales_host"].ctypes.data,
                    row["scales_host"].nbytes, cp.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
            run_m8(out8, staged_proj)

        timings = {
            "hot_h1_one": _measure(cp, lambda: run_h1_row(0, out1), args.warmup, args.reps),
            "hot_h1_sequential_x8": _measure(cp, run_h1x8, args.warmup, args.reps),
            "hot_h8": _measure(cp, run_m8, args.warmup, args.reps),
            "pinned_stage_plus_h8": _measure(cp, run_staged_m8, args.warmup, args.reps),
        }
        h1_ms = float(timings["hot_h1_one"]["p50"])
        h1x8_ms = float(timings["hot_h1_sequential_x8"]["p50"])
        h8_ms = float(timings["hot_h8"]["p50"])
        staged_ms = float(timings["pinned_stage_plus_h8"]["p50"])
        break_even = next((m for m in range(1, 9) if m * h1_ms >= h8_ms), None)
        reference_metrics = _metrics(m8_host, reference)
        h1_metrics = _metrics(m8_host, h1_host)
        gates = {
            "P48_G1_checkpoint_contract": all(row["contract_match"] for row in contracts.values()),
            "P48_G2_all_paths_finite": bool(
                np.isfinite(reference).all()
                and np.isfinite(h1_host).all()
                and np.isfinite(m8_host).all()
            ),
            "P48_G3_independent_reference": bool(
                reference_metrics["normalized_rmse"] <= 0.005
                and reference_metrics["cosine"] >= 0.9999
                and reference_metrics["normalized_max_abs_error"] <= 0.020
            ),
            "P48_G4_h1_agreement": bool(
                h1_metrics["normalized_rmse"] <= 0.001
                and h1_metrics["normalized_max_abs_error"] <= 0.005
            ),
            "P48_G5_h8_bitwise_repeat": bool(np.array_equal(m8_host, repeat_host)),
            "P48_G6_hot_h8_le_0_50_ms": h8_ms <= 0.50,
            "P48_G7_speedup_ge_4x_vs_h1x8": (h1x8_ms / h8_ms) >= 4.0,
            "P48_G8_stage_plus_h8_le_1_50_ms": staged_ms <= 1.50,
            "P48_G9_break_even_multiplicity_le_4": break_even is not None and break_even <= 4,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "gpu": {
                "name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
                "capability": capability,
                "cupy": cp.__version__,
            },
            "checkpoint_contract": contracts,
            "complete_expert_payload_bytes": payload_bytes,
            "complete_expert_payload_MiB": payload_bytes / 2**20,
            "input": {
                "seed": 48002000 + args.layer * 256 + args.expert,
                "shape": list(x_host.shape),
                "distinct_rows": bool(len({row.tobytes() for row in x_host}) == 8),
            },
            "correctness": {
                "h8_vs_independent_decode": reference_metrics,
                "h8_vs_eight_h1": h1_metrics,
                "h8_bitwise_repeat": bool(np.array_equal(m8_host, repeat_host)),
            },
            "kernel_resources": {
                "phase33_m8": m8.resource_audit(),
                "h1_ervf_width": fused.ervf_width,
            },
            "timings_ms": timings,
            "derived": {
                "h1x8_over_h8_speedup": h1x8_ms / h8_ms,
                "h8_over_h1": h8_ms / h1_ms,
                "stage_plus_h8_over_hot_h8": staged_ms / h8_ms,
                "padded_h8_break_even_route_multiplicity": break_even,
                "maximum_reuse_expert_output_rows_per_second": 8000.0 / h8_ms,
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, PHASE33, FUSED))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "gates": payload.get("gates"),
        "timings_ms": payload.get("timings_ms"),
        "derived": payload.get("derived"),
        "correctness": payload.get("correctness"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
