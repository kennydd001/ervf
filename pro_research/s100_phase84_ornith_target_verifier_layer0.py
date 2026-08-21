"""Phase84 first integrated target-verifier layer gate."""
from __future__ import annotations

import argparse
import gzip
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
from s100_phase58_ornith_fp8_h4_kernels import OrnithFP8H4Kernels
from s100_phase67_ornith_linear_h4 import _load_auxiliary, _reference as _linear_reference
from s100_phase67_ornith_linear_h4_kernels import OrnithLinearH4Kernels
from s100_phase69_ornith_support_h4_kernels import OrnithSupportH4Kernels


RESULTS = REPO / "pro_research" / "results" / "s100_phase84_target_verifier"
PREREG = (
    REPO / "pro_research" /
    "S100_PHASE84_ORNITH_TARGET_VERIFIER_LAYER0_PREREGISTRATION.md"
)
SCRIPT = REPO / "pro_research" / "s100_phase84_ornith_target_verifier_layer0.py"
TRACE_DEFAULT = (
    REPO / "pro_research" / "results" / "s100_phase76" /
    "ornith_64_hidden_trace.json.gz"
)


def _load_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text("utf-8"))


def _weight_map(snapshot: Path) -> dict[str, str]:
    return json.loads(
        (snapshot / "model.safetensors.index.json").read_text("utf-8")
    )["weight_map"]


def _load_tensor(snapshot: Path, weight_map: dict[str, str], name: str):
    from safetensors import safe_open

    with safe_open(
        str(snapshot / weight_map[name]), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(name).contiguous()


def _load_fp8(snapshot: Path, weight_map: dict[str, str], name: str):
    import torch
    from safetensors import safe_open

    with safe_open(
        str(snapshot / weight_map[name]), framework="pt", device="cpu"
    ) as handle:
        weight = handle.get_tensor(name).contiguous()
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name}: expected E4M3, got {weight.dtype}")
        prefix = name.removesuffix(".weight")
        weight_scale = float(handle.get_tensor(f"{prefix}.weight_scale").item())
        input_scale = float(handle.get_tensor(f"{prefix}.input_scale").item())
    return weight.view(torch.uint8).numpy().copy(), weight_scale, input_scale


def _trace_layer0(trace: dict[str, Any]) -> np.ndarray:
    count = len(trace["tokens"])
    for tensor in trace["tensors"]:
        if tensor["name"] == "attn_post_norm-0":
            return np.asarray(tensor["values"], dtype=np.float32).reshape(
                count, 2048
            )[:4]
    raise KeyError("attn_post_norm-0")


def _nrmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = max(float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))), 1e-12)
    return float(np.sqrt(np.mean(delta ** 2)) / denominator)


def _rmsnorm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    inverse = np.reciprocal(
        np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + np.float32(1e-6))
    )
    return np.asarray(value * inverse * (np.float32(1.0) + weight), dtype=np.float32)


def _fp8_reference(
    torch, x: np.ndarray, projection: tuple[np.ndarray, float, float],
) -> np.ndarray:
    weight_raw, weight_scale, input_scale = projection
    x_quantized = torch.from_numpy(
        np.asarray(x / np.float32(input_scale), dtype=np.float32)
    ).to(torch.float8_e4m3fn).float().numpy()
    weight = torch.from_numpy(weight_raw.copy()).view(
        torch.float8_e4m3fn
    ).float().numpy()
    return np.asarray(
        (x_quantized * np.float32(input_scale))
        @ (weight * np.float32(weight_scale)).T,
        dtype=np.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=TRACE_DEFAULT)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE84_ORNITH_TARGET_VERIFIER_LAYER0.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase84_ornith_target_verifier_layer0",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "embedding through layer-0 attention/state only; not output tok/s",
    }
    cp = None
    try:
        import cupy as cp_module
        import sys
        import torch

        src = REPO / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from moe_lab.ornith.static_fp8 import StaticFP8H4Quantizer

        cp = cp_module
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        snapshot = args.snapshot.expanduser().resolve()
        trace_path = args.trace.expanduser().resolve()
        trace = _load_trace(trace_path)
        tokens = tuple(int(value) for value in trace["tokens"][:4])
        gguf_reference = _trace_layer0(trace)
        weight_map = _weight_map(snapshot)

        embedding = _load_tensor(snapshot, weight_map, "model.embed_tokens.weight")
        residual_host = embedding[list(tokens)].float().numpy().copy()
        del embedding
        input_norm_tensor = _load_tensor(
            snapshot, weight_map, "model.layers.0.input_layernorm.weight"
        )
        post_norm_tensor = _load_tensor(
            snapshot, weight_map, "model.layers.0.post_attention_layernorm.weight"
        )
        input_norm = input_norm_tensor.view(torch.uint16).numpy().copy()
        post_norm = post_norm_tensor.view(torch.uint16).numpy().copy()
        projections = {
            label: _load_fp8(snapshot, weight_map, name)
            for label, name in {
                "qkv": "model.layers.0.linear_attn.in_proj_qkv.weight",
                "z": "model.layers.0.linear_attn.in_proj_z.weight",
                "out": "model.layers.0.linear_attn.out_proj.weight",
            }.items()
        }
        auxiliary = _load_auxiliary(snapshot, "model.layers.0.linear_attn")
        cpu_normed = _rmsnorm(
            residual_host, input_norm_tensor.float().numpy().copy()
        )
        cpu_mixed = _fp8_reference(torch, cpu_normed, projections["qkv"])
        cpu_z = _fp8_reference(torch, cpu_normed, projections["z"])
        cpu_linear = _linear_reference(
            auxiliary, cpu_normed, cpu_mixed, cpu_z,
            np.zeros((8192, 4), dtype=np.float32),
            np.zeros((32, 128, 128), dtype=np.float32),
        )
        cpu_branch = _fp8_reference(torch, cpu_linear["output"], projections["out"])
        hf_reference = _rmsnorm(
            residual_host + cpu_branch, post_norm_tensor.float().numpy().copy()
        )

        fp8 = OrnithFP8H4Kernels()
        quantizer = StaticFP8H4Quantizer()
        linear = OrnithLinearH4Kernels()
        support = OrnithSupportH4Kernels()
        residual = cp.asarray(residual_host)
        normed = cp.empty_like(residual)
        input_norm_gpu = cp.asarray(input_norm)
        post_norm_gpu = cp.asarray(post_norm)
        weight_gpu = {label: cp.asarray(row[0]) for label, row in projections.items()}
        quantized_2048 = cp.empty((4, 2048), dtype=cp.uint8)
        quantized_4096 = cp.empty((4, 4096), dtype=cp.uint8)
        mixed = cp.empty((4, 8192), dtype=cp.float32)
        z = cp.empty((4, 4096), dtype=cp.float32)
        core = cp.empty((4, 4096), dtype=cp.float32)
        branch = cp.empty((4, 2048), dtype=cp.float32)
        result = cp.empty((4, 2048), dtype=cp.float32)
        aux_gpu = {
            name: cp.asarray(row["raw"])
            for name, row in auxiliary.items()
        }
        beta = cp.empty((4, 32), dtype=cp.float32)
        g = cp.empty((4, 32), dtype=cp.float32)
        convolved = cp.empty((4, 8192), dtype=cp.float32)
        conv_state = cp.zeros((8192, 4), dtype=cp.float32)
        recurrent_state = cp.zeros((32, 128, 128), dtype=cp.float32)

        def projection(label: str, x, q, target) -> None:
            raw, weight_scale, input_scale = projections[label]
            quantizer.quantize(x, q, input_scale)
            fp8.m4(
                weight_gpu[label], q, target, raw.shape[0], raw.shape[1],
                weight_scale * input_scale,
            )

        def execute_fresh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            residual.set(residual_host)
            conv_state.fill(0)
            recurrent_state.fill(0)
            support.norm(residual, input_norm_gpu, normed)
            projection("qkv", normed, quantized_2048, mixed)
            projection("z", normed, quantized_2048, z)
            linear.gates(
                aux_gpu["in_proj_a.weight"], aux_gpu["in_proj_b.weight"], normed,
                aux_gpu["A_log"], aux_gpu["dt_bias"], beta, g,
            )
            linear.convolution(
                mixed, aux_gpu["conv1d.weight"], conv_state, convolved
            )
            linear.delta_norm(
                convolved, z, beta, g, aux_gpu["norm.weight"],
                recurrent_state, core,
            )
            projection("out", core, quantized_4096, branch)
            support.add_norm(residual, branch, post_norm_gpu, result)
            cp.cuda.get_current_stream().synchronize()
            return cp.asnumpy(result), cp.asnumpy(conv_state), cp.asnumpy(recurrent_state)

        first, first_conv, first_recurrent = execute_fresh()
        second, second_conv, second_recurrent = execute_fresh()
        probe = np.asarray([
            -448.0, -15.5, -1.0, -0.09375, -0.0, 0.0, 0.09375, 1.0,
            15.5, 448.0,
        ], dtype=np.float32)
        probe_gpu = cp.asarray(probe.reshape(1, -1))
        probe_raw = cp.empty(probe_gpu.shape, dtype=cp.uint8)
        quantizer.quantize(probe_gpu, probe_raw, 1.0)
        cp.cuda.get_current_stream().synchronize()
        ours_raw = cp.asnumpy(probe_raw).reshape(-1)
        torch_raw = torch.from_numpy(probe.copy()).to(torch.float8_e4m3fn).view(
            torch.uint8
        ).numpy()
        nrmse = _nrmse(first, hf_reference)
        max_abs = float(np.max(np.abs(first - hf_reference)))
        gguf_nrmse = _nrmse(first, gguf_reference)
        gguf_max_abs = float(np.max(np.abs(first - gguf_reference)))
        gates = {
            "P84_L0_G1_cuda_fp8_bytes_match_torch": bool(np.array_equal(ours_raw, torch_raw)),
            "P84_L0_G2_output_and_state_repeat_exact": bool(
                np.array_equal(first.view(np.uint32), second.view(np.uint32))
                and np.array_equal(first_conv.view(np.uint32), second_conv.view(np.uint32))
                and np.array_equal(
                    first_recurrent.view(np.uint32), second_recurrent.view(np.uint32)
                )
            ),
            "P84_L0_G3_authoritative_layer0_parity": nrmse <= 1e-4 and max_abs <= 1e-2,
            "P84_L0_G4_real_state_mutation": bool(
                np.any(first_conv != 0.0) and np.any(first_recurrent != 0.0)
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "completed_utc": utc_now(),
            "inputs": {
                "snapshot": str(snapshot),
                "trace": str(trace_path),
                "tokens": list(tokens),
                "fresh_zero_state": True,
            },
            "quality": {
                "hf_layer0_attn_post_norm_nrmse": nrmse,
                "hf_layer0_attn_post_norm_max_abs": max_abs,
                "gguf_q8_cross_format_nrmse": gguf_nrmse,
                "gguf_q8_cross_format_max_abs": gguf_max_abs,
                "candidate_finite": bool(np.isfinite(first).all()),
                "hf_reference_finite": bool(np.isfinite(hf_reference).all()),
                "gguf_reference_finite": bool(np.isfinite(gguf_reference).all()),
                "reference_boundary": (
                    "HF FP8 is the exact activation reference; GGUF Q8_0 is "
                    "cross-format evidence only"
                ),
                "fp8_probe_candidate_bytes": ours_raw.tolist(),
                "fp8_probe_torch_bytes": torch_raw.tolist(),
            },
            "state": {
                "conv_nonzero": int(np.count_nonzero(first_conv)),
                "recurrent_nonzero": int(np.count_nonzero(first_recurrent)),
            },
            "resources": {"static_fp8_quantizer": quantizer.resource_audit()},
            "gates": gates,
        })
    except Exception as error:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        })
    finally:
        if cp is not None:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, args.trace))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "quality": payload.get("quality"),
        "state": payload.get("state"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2), flush=True)
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
