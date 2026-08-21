"""Phase47: execute pottokao's embedded Ornith-compatible DFlash body on SM120."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, sha256_file, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm


RESULTS = REPO / "pro_research" / "results" / "s100_phase47"
PREREG = REPO / "pro_research" / "S100_PHASE47_ORNITH_DFLASH_SMOKE_PREREGISTRATION.md"
EXPECTED_PARAMETERS = 385_906_176
EXPECTED_TENSORS = 69


def _rms_norm(torch, value, weight, eps: float):
    dtype = value.dtype
    work = value.float()
    normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    return weight * normalized.to(dtype)


def _rotate_half(torch, value):
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope(torch, value, positions, *, head_dim: int, theta: float):
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2, device=value.device, dtype=torch.float32)
            / head_dim
        )
    )
    frequencies = torch.outer(positions.float(), inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().to(dtype=value.dtype)[None, :, None, :]
    sine = embedding.sin().to(dtype=value.dtype)[None, :, None, :]
    return value * cosine + _rotate_half(torch, value) * sine


def _load_checkpoint(torch, safe_open, snapshot: Path, config: dict[str, Any]):
    device = torch.device("cuda")
    model_path = snapshot / "model.safetensors"
    layers = []
    tensor_rows = []
    parameter_count = 0
    resident_bytes = 0

    def load(handle, name: str):
        nonlocal parameter_count, resident_bytes
        value = handle.get_tensor(name)
        if value.dtype != torch.bfloat16:
            raise TypeError(f"{name}: expected BF16, got {value.dtype}")
        parameter_count += value.numel()
        resident_bytes += value.numel() * value.element_size()
        tensor_rows.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        return value.to(device=device)

    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        all_names = list(handle.keys())
        fc = load(handle, "fc.weight")
        hidden_norm = load(handle, "hidden_norm.weight")
        final_norm = load(handle, "norm.weight")
        for index in range(int(config["num_hidden_layers"])):
            prefix = f"layers.{index}"
            attention = f"{prefix}.self_attn"
            mlp = f"{prefix}.mlp"
            layers.append({
                "input_norm": load(handle, f"{prefix}.input_layernorm.weight"),
                "post_norm": load(handle, f"{prefix}.post_attention_layernorm.weight"),
                "q_norm": load(handle, f"{attention}.q_norm.weight"),
                "k_norm": load(handle, f"{attention}.k_norm.weight"),
                "q_proj": load(handle, f"{attention}.q_proj.weight"),
                "k_proj": load(handle, f"{attention}.k_proj.weight"),
                "v_proj": load(handle, f"{attention}.v_proj.weight"),
                "o_proj": load(handle, f"{attention}.o_proj.weight"),
                "gate_proj": load(handle, f"{mlp}.gate_proj.weight"),
                "up_proj": load(handle, f"{mlp}.up_proj.weight"),
                "down_proj": load(handle, f"{mlp}.down_proj.weight"),
                "kind": config["layer_types"][index],
            })
    if sorted(row["name"] for row in tensor_rows) != sorted(all_names):
        loaded = {row["name"] for row in tensor_rows}
        raise RuntimeError(f"unconsumed checkpoint tensors: {sorted(set(all_names) - loaded)}")
    return {
        "fc": fc,
        "hidden_norm": hidden_norm,
        "final_norm": final_norm,
        "layers": layers,
        "tensor_rows": tensor_rows,
        "parameter_count": parameter_count,
        "resident_bytes": resident_bytes,
    }


def _project_context(torch, functional, weights, target_hidden, eps: float):
    flattened = target_hidden.reshape(target_hidden.shape[0], -1)
    projected = functional.linear(flattened, weights["fc"])
    return _rms_norm(torch, projected, weights["hidden_norm"], eps)


def _precompute_context_kv(torch, functional, weights, projected, config):
    hidden = int(config["hidden_size"])
    del hidden
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    eps = float(config["rms_norm_eps"])
    theta = float((config.get("rope_parameters") or {}).get("rope_theta", 10_000.0))
    positions = torch.arange(projected.shape[0], device=projected.device, dtype=torch.long)
    result = []
    for layer in weights["layers"]:
        key = functional.linear(projected, layer["k_proj"]).view(1, -1, kv_heads, head_dim)
        value = functional.linear(projected, layer["v_proj"]).view(1, -1, kv_heads, head_dim)
        key = _rope(
            torch,
            _rms_norm(torch, key, layer["k_norm"], eps),
            positions,
            head_dim=head_dim,
            theta=theta,
        )
        result.append((key, value))
    return result


def _attention(
    torch,
    functional,
    hidden,
    layer,
    context_kv,
    *,
    context_length: int,
    config: dict[str, Any],
):
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    groups = heads // kv_heads
    eps = float(config["rms_norm_eps"])
    theta = float((config.get("rope_parameters") or {}).get("rope_theta", 10_000.0))
    block = hidden.shape[0]
    positions = torch.arange(
        context_length, context_length + block, device=hidden.device, dtype=torch.long
    )
    query = functional.linear(hidden, layer["q_proj"]).view(1, block, heads, head_dim)
    block_key = functional.linear(hidden, layer["k_proj"]).view(1, block, kv_heads, head_dim)
    block_value = functional.linear(hidden, layer["v_proj"]).view(1, block, kv_heads, head_dim)
    query = _rope(
        torch,
        _rms_norm(torch, query, layer["q_norm"], eps),
        positions,
        head_dim=head_dim,
        theta=theta,
    )
    block_key = _rope(
        torch,
        _rms_norm(torch, block_key, layer["k_norm"], eps),
        positions,
        head_dim=head_dim,
        theta=theta,
    )
    context_key, context_value = context_kv
    key = torch.cat((context_key, block_key), dim=1)
    value = torch.cat((context_value, block_value), dim=1)
    key = torch.repeat_interleave(key, groups, dim=2).transpose(1, 2)
    value = torch.repeat_interleave(value, groups, dim=2).transpose(1, 2)
    query = query.transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-1, -2)) * (head_dim ** -0.5)
    if layer["kind"] == "sliding_attention":
        window = int(config["sliding_window"])
        query_positions = positions[:, None]
        key_positions = torch.arange(key.shape[-2], device=hidden.device)[None, :]
        visible = (key_positions <= query_positions) & (query_positions - key_positions < window)
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
    elif layer["kind"] != "full_attention":
        raise ValueError(f"unsupported DFlash layer kind: {layer['kind']}")
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attended = torch.matmul(probabilities, value).transpose(1, 2).reshape(
        block, heads * head_dim
    )
    return functional.linear(attended, layer["o_proj"])


def _forward_block(torch, functional, weights, context_kv, noise_embedding, config):
    eps = float(config["rms_norm_eps"])
    hidden = noise_embedding
    context_length = int(context_kv[0][0].shape[1])
    for index, layer in enumerate(weights["layers"]):
        residual = hidden
        normed = _rms_norm(torch, hidden, layer["input_norm"], eps)
        hidden = residual + _attention(
            torch,
            functional,
            normed,
            layer,
            context_kv[index],
            context_length=context_length,
            config=config,
        )
        residual = hidden
        normed = _rms_norm(torch, hidden, layer["post_norm"], eps)
        mlp = functional.silu(functional.linear(normed, layer["gate_proj"]))
        mlp = mlp * functional.linear(normed, layer["up_proj"])
        hidden = residual + functional.linear(mlp, layer["down_proj"])
    return _rms_norm(torch, hidden, weights["final_norm"], eps)


def _percentiles(numpy, values):
    array = numpy.asarray(values, dtype=numpy.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(numpy.median(array)),
        "p10_ms": float(numpy.percentile(array, 10)),
        "p90_ms": float(numpy.percentile(array, 90)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ornith DFlash real-checkpoint body smoke")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    output = RESULTS / "S100_PHASE47_ORNITH_DFLASH_SMOKE.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase47_ornith_dflash_real_checkpoint_smoke",
        "status": "started",
        "snapshot": str(snapshot),
        "context": int(args.context),
        "warmup": int(args.warmup),
        "reps": int(args.reps),
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "real DFlash body with synthetic target residuals/embeddings; "
            "no acceptance, quality, verifier or end-to-end throughput claim"
        ),
    }
    try:
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        import numpy as np
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        config_path = snapshot / "config.json"
        model_path = snapshot / "model.safetensors"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dflash = config.get("dflash_config") or {}
        expected_config = {
            "architectures": ["DFlashDraftModel"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 6,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
            "sliding_window": 4096,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
            "block_size": 16,
            "mask_token_id": 248077,
        }
        observed_config = {
            key: dflash.get(key) if key in ("target_layer_ids", "block_size", "mask_token_id") else config.get(key)
            for key in expected_config
        }
        # block_size and mask_token_id are nested in this checkpoint.
        observed_config["block_size"] = dflash.get("block_size", config.get("block_size"))
        observed_config["mask_token_id"] = dflash.get("mask_token_id", config.get("mask_token_id"))

        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        weights = _load_checkpoint(torch, safe_open, snapshot, config)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_start

        generator = torch.Generator(device="cuda")
        generator.manual_seed(4701)
        features = len(dflash["target_layer_ids"])
        target_hidden = torch.randn(
            (int(args.context), features, int(config["hidden_size"])),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ) * 0.02
        torch.cuda.synchronize()
        context_start = time.perf_counter_ns()
        projected = _project_context(torch, F, weights, target_hidden, float(config["rms_norm_eps"]))
        context_kv = _precompute_context_kv(torch, F, weights, projected, config)
        torch.cuda.synchronize()
        context_ms = (time.perf_counter_ns() - context_start) / 1e6

        blocks = {}
        outputs = {}
        for block_size in (8, 16):
            embedding = torch.randn(
                (block_size, int(config["hidden_size"])),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            ) * 0.02
            for _ in range(int(args.warmup)):
                _forward_block(torch, F, weights, context_kv, embedding, config)
            torch.cuda.synchronize()
            samples = []
            output_value = None
            for _ in range(int(args.reps)):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output_value = _forward_block(torch, F, weights, context_kv, embedding, config)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            assert output_value is not None
            replay = _forward_block(torch, F, weights, context_kv, embedding, config)
            torch.cuda.synchronize()
            blocks[f"K{block_size}"] = {
                "output_shape": list(output_value.shape),
                "finite": bool(torch.isfinite(output_value.float()).all().item()),
                "bitwise_repeat": bool(torch.equal(output_value, replay)),
                "timing": _percentiles(np, samples),
                "output_norm_mean": float(output_value.float().norm(dim=-1).mean().item()),
            }
            outputs[block_size] = output_value

        tensor_contract = bool(
            len(weights["tensor_rows"]) == EXPECTED_TENSORS
            and weights["parameter_count"] == EXPECTED_PARAMETERS
            and all(row["dtype"] == "torch.bfloat16" for row in weights["tensor_rows"])
        )
        gates = {
            "P47_G1_config_contract": observed_config == expected_config,
            "P47_G2_all_69_bf16_tensors": tensor_contract,
            "P47_G3_context_projection_finite": bool(torch.isfinite(projected.float()).all().item()),
            "P47_G4_K8_finite_shape": blocks["K8"]["finite"] and blocks["K8"]["output_shape"] == [8, 2048],
            "P47_G5_K16_finite_shape": blocks["K16"]["finite"] and blocks["K16"]["output_shape"] == [16, 2048],
            "P47_G6_K8_K16_bitwise_repeat": blocks["K8"]["bitwise_repeat"] and blocks["K16"]["bitwise_repeat"],
            "P47_G7_peak_torch_allocation_lt_3GiB": int(torch.cuda.max_memory_allocated()) < 3 * 2**30,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "completed_utc": utc_now(),
            "checkpoint": {
                "config_sha256": sha256_file(config_path),
                "model_sha256": sha256_file(model_path),
                "model_bytes": int(model_path.stat().st_size),
                "parameter_count": weights["parameter_count"],
                "tensor_count": len(weights["tensor_rows"]),
                "resident_weight_bytes": weights["resident_bytes"],
                "all_tensor_dtype": "bfloat16",
                "load_seconds": load_seconds,
            },
            "config": {"expected": expected_config, "observed": observed_config},
            "context_projection_and_kv_ms": context_ms,
            "blocks": blocks,
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "free_bytes_after": int(torch.cuda.mem_get_info()[0]),
            },
            "gates": gates,
            "environment": environment_snapshot((Path(__file__), PREREG, config_path)),
        })
        del outputs
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })
    write_json_atomic(output, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "checkpoint": payload.get("checkpoint"),
        "context_projection_and_kv_ms": payload.get("context_projection_and_kv_ms"),
        "blocks": payload.get("blocks"),
        "memory": payload.get("memory"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(output),
    }, indent=2), flush=True)
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
