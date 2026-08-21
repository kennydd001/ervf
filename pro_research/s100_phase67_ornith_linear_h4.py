"""Phase67 fused H4 benchmark for Ornith/Qwen3.5 linear attention."""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, percentiles, utc_now, write_json_atomic
from s100_phase67_ornith_linear_h4_kernels import OrnithLinearH4Kernels


RESULTS = REPO / "pro_research" / "results" / "s100_phase67"
PREREG = REPO / "pro_research" / "S100_PHASE67_ORNITH_LINEAR_H4_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase67_ornith_linear_h4.py"
KERNELS = REPO / "pro_research" / "s100_phase67_ornith_linear_h4_kernels.py"
PHASE66 = (
    REPO
    / "pro_research"
    / "results"
    / "s100_phase66"
    / "S100_PHASE66_ORNITH_65TPS_BUDGET.json"
)


def _weight_map(snapshot: Path) -> dict[str, str]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text("utf-8"))
    return index["weight_map"]


def _load_auxiliary(snapshot: Path, prefix: str) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    weight_map = _weight_map(snapshot)
    suffixes = (
        "in_proj_a.weight",
        "in_proj_b.weight",
        "A_log",
        "dt_bias",
        "conv1d.weight",
        "norm.weight",
    )
    result: dict[str, Any] = {}
    shards: dict[str, Any] = {}
    for suffix in suffixes:
        name = f"{prefix}.{suffix}"
        shard_name = weight_map[name]
        handle = shards.get(shard_name)
        if handle is None:
            handle = safe_open(snapshot / shard_name, framework="pt", device="cpu")
            shards[shard_name] = handle
        tensor = handle.get_tensor(name).contiguous()
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name}: expected BF16, got {tensor.dtype}")
        result[suffix] = {
            "raw": tensor.view(torch.uint16).numpy().copy().reshape(tensor.shape),
            "float": tensor.float().numpy().copy().reshape(tensor.shape),
            "shape": list(tensor.shape),
            "shard": shard_name,
        }
    result["conv1d.weight"]["raw"] = result["conv1d.weight"]["raw"].reshape(8192, 4)
    result["conv1d.weight"]["float"] = result["conv1d.weight"]["float"].reshape(8192, 4)
    return result


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (np.float32(1.0) + np.exp(-value))


def _reference(
    auxiliary: dict[str, Any],
    hidden4: np.ndarray,
    mixed4: np.ndarray,
    z4: np.ndarray,
    initial_conv_state: np.ndarray,
    initial_recurrent_state: np.ndarray,
) -> dict[str, np.ndarray]:
    wa = auxiliary["in_proj_a.weight"]["float"]
    wb = auxiliary["in_proj_b.weight"]["float"]
    a_log = auxiliary["A_log"]["float"]
    dt_bias = auxiliary["dt_bias"]["float"]
    conv_weight = auxiliary["conv1d.weight"]["float"]
    norm_weight = auxiliary["norm.weight"]["float"]

    a = np.asarray(hidden4 @ wa.T, dtype=np.float32)
    b = np.asarray(hidden4 @ wb.T, dtype=np.float32)
    beta = np.asarray(1.0 / (1.0 + np.exp(-b)), dtype=np.float32)
    gate_input = np.asarray(a + dt_bias[None, :], dtype=np.float32)
    softplus = np.where(
        gate_input > np.float32(20.0),
        gate_input,
        np.log1p(np.exp(gate_input)),
    ).astype(np.float32)
    g = np.asarray(-np.exp(a_log)[None, :] * softplus, dtype=np.float32)

    history = np.concatenate((initial_conv_state, mixed4.T), axis=1)
    convolved = np.empty_like(mixed4)
    for token in range(4):
        value = np.sum(
            history[:, token + 1 : token + 5] * conv_weight,
            axis=1,
            dtype=np.float32,
        )
        convolved[token] = _silu(value).astype(np.float32)
    final_conv_state = mixed4.T.copy()

    query = convolved[:, :2048].reshape(4, 16, 128)
    key = convolved[:, 2048:4096].reshape(4, 16, 128)
    value = convolved[:, 4096:].reshape(4, 32, 128)
    query = np.repeat(query, 2, axis=1)
    key = np.repeat(key, 2, axis=1)
    query = query * np.reciprocal(
        np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + np.float32(1.0e-6))
    )
    key = key * np.reciprocal(
        np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + np.float32(1.0e-6))
    )
    query = np.asarray(query * np.float32(1.0 / math.sqrt(128.0)), dtype=np.float32)
    state = initial_recurrent_state.copy()
    output = np.empty((4, 32, 128), dtype=np.float32)
    z = z4.reshape(4, 32, 128)
    for token in range(4):
        for head in range(32):
            state[head] *= np.exp(g[token, head]).astype(np.float32)
            kv_memory = np.asarray(key[token, head] @ state[head], dtype=np.float32)
            delta = np.asarray(
                (value[token, head] - kv_memory) * beta[token, head], dtype=np.float32
            )
            state[head] += np.outer(key[token, head], delta).astype(np.float32)
            core = np.asarray(query[token, head] @ state[head], dtype=np.float32)
            inv_rms = np.float32(
                1.0 / math.sqrt(float(np.mean(core * core, dtype=np.float32)) + 1.0e-6)
            )
            output[token, head] = np.asarray(
                core * inv_rms * norm_weight * _silu(z[token, head]), dtype=np.float32
            )
    return {
        "beta": beta,
        "g": g,
        "convolved": convolved,
        "conv_state": final_conv_state,
        "recurrent_state": state,
        "output": output.reshape(4, 4096),
    }


def _nrmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    error = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    denominator = max(float(np.sqrt(np.mean(np.square(reference, dtype=np.float64)))), 1.0e-12)
    return float(np.sqrt(np.mean(np.square(error))) / denominator)


def _quality(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    return {
        "nrmse": _nrmse(candidate, reference),
        "max_abs": float(np.max(np.abs(candidate - reference))),
        "candidate_finite": bool(np.isfinite(candidate).all()),
        "reference_finite": bool(np.isfinite(reference).all()),
    }


def _measure(cp, function, warmup: int, repeats: int) -> dict[str, float | int | None]:
    for _ in range(warmup):
        function()
    cp.cuda.get_current_stream().synchronize()
    samples = []
    for _ in range(repeats):
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        begin.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(begin, end)))
    return percentiles(samples)


def _bench_repository(
    kernels: OrnithLinearH4Kernels,
    label: str,
    snapshot: Path,
    prefix: str,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import cupy as cp

    auxiliary = _load_auxiliary(snapshot, prefix)
    rng = np.random.default_rng(seed)
    hidden4 = rng.normal(0.0, 0.45, size=(4, 2048)).astype(np.float32)
    mixed4 = rng.normal(0.0, 0.50, size=(4, 8192)).astype(np.float32)
    z4 = rng.normal(0.0, 0.50, size=(4, 4096)).astype(np.float32)
    conv_state = rng.normal(0.0, 0.20, size=(8192, 4)).astype(np.float32)
    recurrent_state = rng.normal(0.0, 0.005, size=(32, 128, 128)).astype(np.float32)
    reference = _reference(
        auxiliary, hidden4, mixed4, z4, conv_state, recurrent_state
    )

    gpu = {
        "weight_a": cp.asarray(auxiliary["in_proj_a.weight"]["raw"]),
        "weight_b": cp.asarray(auxiliary["in_proj_b.weight"]["raw"]),
        "a_log": cp.asarray(auxiliary["A_log"]["raw"]),
        "dt_bias": cp.asarray(auxiliary["dt_bias"]["raw"]),
        "conv_weight": cp.asarray(auxiliary["conv1d.weight"]["raw"]),
        "norm_weight": cp.asarray(auxiliary["norm.weight"]["raw"]),
        "hidden4": cp.asarray(hidden4),
        "mixed4": cp.asarray(mixed4),
        "z4": cp.asarray(z4),
        "conv_state": cp.asarray(conv_state),
        "state": cp.asarray(recurrent_state),
        "beta": cp.empty((4, 32), dtype=cp.float32),
        "g": cp.empty((4, 32), dtype=cp.float32),
        "convolved": cp.empty((4, 8192), dtype=cp.float32),
        "output": cp.empty((4, 4096), dtype=cp.float32),
    }

    def gates() -> None:
        kernels.gates(
            gpu["weight_a"], gpu["weight_b"], gpu["hidden4"],
            gpu["a_log"], gpu["dt_bias"], gpu["beta"], gpu["g"],
        )

    def convolution() -> None:
        kernels.convolution(
            gpu["mixed4"], gpu["conv_weight"], gpu["conv_state"], gpu["convolved"]
        )

    def delta_norm() -> None:
        kernels.delta_norm(
            gpu["convolved"], gpu["z4"], gpu["beta"], gpu["g"],
            gpu["norm_weight"], gpu["state"], gpu["output"],
        )

    def complete() -> None:
        gates()
        convolution()
        delta_norm()

    complete()
    cp.cuda.get_current_stream().synchronize()
    candidate = {
        "beta": cp.asnumpy(gpu["beta"]),
        "g": cp.asnumpy(gpu["g"]),
        "convolved": cp.asnumpy(gpu["convolved"]),
        "conv_state": cp.asnumpy(gpu["conv_state"]),
        "recurrent_state": cp.asnumpy(gpu["state"]),
        "output": cp.asnumpy(gpu["output"]),
    }
    quality = {name: _quality(candidate[name], reference[name]) for name in reference}

    # Fresh-state repeat proves determinism independently from the timing state.
    repeat_outputs = []
    repeat_states = []
    for _ in range(2):
        gpu["conv_state"].set(conv_state)
        gpu["state"].set(recurrent_state)
        complete()
        cp.cuda.get_current_stream().synchronize()
        repeat_outputs.append(cp.asnumpy(gpu["output"]))
        repeat_states.append(cp.asnumpy(gpu["state"]))
    deterministic = bool(
        np.array_equal(repeat_outputs[0].view(np.uint32), repeat_outputs[1].view(np.uint32))
        and np.array_equal(repeat_states[0].view(np.uint32), repeat_states[1].view(np.uint32))
    )

    timings = {
        "gates": _measure(cp, gates, warmup, repeats),
        "convolution": _measure(cp, convolution, warmup, repeats),
        "delta_norm": _measure(cp, delta_norm, warmup, repeats),
        "complete": _measure(cp, complete, warmup, repeats),
    }
    cp.cuda.get_current_stream().synchronize()
    record = {
        "repository": label,
        "snapshot": str(snapshot),
        "prefix": prefix,
        "auxiliary": {
            name: {key: value for key, value in row.items() if key not in ("raw", "float")}
            for name, row in auxiliary.items()
        },
        "quality": quality,
        "fresh_state_bit_deterministic": deterministic,
        "timings_ms": timings,
        "projected_30_linear_layers_ms_h4": 30.0 * float(timings["complete"]["p50"]),
    }
    del gpu
    cp.get_default_memory_pool().free_all_blocks()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--pottokao", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=51)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE67_ORNITH_LINEAR_H4.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase67_ornith_linear_h4",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    cp = None
    try:
        for path in (args.official, args.pottokao):
            if not path.is_dir():
                raise FileNotFoundError(path)
        import cupy as cp_module

        cp = cp_module
        kernels = OrnithLinearH4Kernels()
        resources = kernels.resource_audit()
        records = [
            _bench_repository(
                kernels,
                "official",
                args.official.resolve(),
                "model.language_model.layers.20.linear_attn",
                67001,
                args.warmup,
                args.repeats,
            ),
            _bench_repository(
                kernels,
                "pottokao",
                args.pottokao.resolve(),
                "model.layers.20.linear_attn",
                67002,
                args.warmup,
                args.repeats,
            ),
        ]
        phase66 = json.loads(PHASE66.read_text("utf-8"))
        known_floor = float(phase66["hot_budget"]["known_floor_ms_h4"])
        worst_layer = max(float(row["timings_ms"]["complete"]["p50"]) for row in records)
        worst_30 = worst_layer * 30.0
        combined = known_floor + worst_30
        quality_ok = all(
            metric["nrmse"] <= 5.0e-5
            and metric["candidate_finite"]
            and metric["reference_finite"]
            for row in records
            for name, metric in row["quality"].items()
            if name in ("output", "conv_state", "recurrent_state")
        )
        recurrent_resource = resources["ornith_delta_norm_h4"]
        gates = {
            "P67_G1_reference_quality": quality_ok,
            "P67_G2_each_layer_le_0_20ms_and_30_le_6ms": all(
                float(row["timings_ms"]["complete"]["p50"]) <= 0.20
                and row["projected_30_linear_layers_ms_h4"] <= 6.0
                for row in records
            ),
            "P67_G3_phase66_plus_linear_core_below_65_boundary": combined < 4000.0 / 65.0,
            "P67_G4_resource_budget": (
                all((row.get("local_size_bytes") or 0) == 0 for row in resources.values())
                and (recurrent_resource.get("num_regs") or 10_000) <= 96
                and (recurrent_resource.get("max_threads_per_block") or 0) >= 128
            ),
            "P67_G5_fresh_state_deterministic": all(
                row["fresh_state_bit_deterministic"] for row in records
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "records": records,
            "resources": resources,
            "budget": {
                "phase66_known_hot_floor_ms_h4": known_floor,
                "worse_linear_core_ms_per_layer": worst_layer,
                "worse_30_linear_core_ms_h4": worst_30,
                "combined_known_floor_ms_h4": combined,
                "combined_equivalent_tok_s": 4000.0 / combined,
                "remaining_to_65_ms": 4000.0 / 65.0 - combined,
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNELS, PHASE66))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "records": [
            {
                "repository": row["repository"],
                "complete_ms": row["timings_ms"]["complete"]["p50"],
                "components_ms": {
                    name: timing["p50"] for name, timing in row["timings_ms"].items()
                    if name != "complete"
                },
                "projected_30_ms": row["projected_30_linear_layers_ms_h4"],
                "output_nrmse": row["quality"]["output"]["nrmse"],
                "state_nrmse": row["quality"]["recurrent_state"]["nrmse"],
                "deterministic": row["fresh_state_bit_deterministic"],
            }
            for row in payload.get("records", [])
        ],
        "resources": payload.get("resources"),
        "budget": payload.get("budget"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
