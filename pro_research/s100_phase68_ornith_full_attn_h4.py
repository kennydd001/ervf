"""Phase68 H4 full-attention benchmark for Official and Pottokao Ornith."""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, percentiles, utc_now, write_json_atomic
from s100_phase68_ornith_full_attn_h4_kernels import OrnithFullAttentionH4Kernels


RESULTS = REPO / "pro_research" / "results" / "s100_phase68"
PREREG = REPO / "pro_research" / "S100_PHASE68_ORNITH_FULL_ATTN_H4_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase68_ornith_full_attn_h4.py"
KERNELS = REPO / "pro_research" / "s100_phase68_ornith_full_attn_h4_kernels.py"
PHASE67 = (
    REPO / "pro_research" / "results" / "s100_phase67"
    / "S100_PHASE67_ORNITH_LINEAR_H4.json"
)
CONTEXTS = (128, 1024)


def _weight_map(snapshot: Path) -> dict[str, str]:
    return json.loads(
        (snapshot / "model.safetensors.index.json").read_text("utf-8")
    )["weight_map"]


def _load_norms(snapshot: Path, prefix: str) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    weight_map = _weight_map(snapshot)
    result = {}
    for label in ("q_norm.weight", "k_norm.weight"):
        name = f"{prefix}.{label}"
        with safe_open(snapshot / weight_map[name], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name).contiguous()
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (256,):
                raise TypeError(f"{name}: expected BF16[256], got {tensor.dtype}{tensor.shape}")
            result[label] = {
                "raw": tensor.view(torch.uint16).numpy().copy(),
                "float": tensor.float().numpy().copy(),
                "shard": weight_map[name],
            }
    return result


def _rope(base_context: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(base_context, base_context + 4, dtype=np.float32)
    frequencies = np.float32(1.0) / np.power(
        np.float32(10_000_000.0),
        np.arange(0, 64, 2, dtype=np.float32) / np.float32(64.0),
    )
    phase = positions[:, None] * frequencies[None, :]
    embedding = np.concatenate((phase, phase), axis=1)
    return np.cos(embedding).astype(np.float32), np.sin(embedding).astype(np.float32)


def _apply_rope(values: np.ndarray, cos4: np.ndarray, sin4: np.ndarray) -> np.ndarray:
    output = values.copy()
    rotary = values[..., :64]
    rotated = np.concatenate((-rotary[..., 32:], rotary[..., :32]), axis=-1)
    output[..., :64] = rotary * cos4[:, None, :] + rotated * sin4[:, None, :]
    return output


def _reference(
    norms: dict[str, Any],
    q_gate4: np.ndarray,
    key4: np.ndarray,
    value4: np.ndarray,
    cos4: np.ndarray,
    sin4: np.ndarray,
    initial_key_cache: np.ndarray,
    initial_value_cache: np.ndarray,
    base_context: int,
) -> dict[str, np.ndarray]:
    q_pairs = q_gate4.reshape(4, 16, 512)
    query = q_pairs[..., :256]
    gate = q_pairs[..., 256:]
    key = key4.reshape(4, 2, 256)
    value = value4.reshape(4, 2, 256)
    query = query * np.reciprocal(
        np.sqrt(np.mean(query * query, axis=-1, keepdims=True) + np.float32(1.0e-6))
    )
    key = key * np.reciprocal(
        np.sqrt(np.mean(key * key, axis=-1, keepdims=True) + np.float32(1.0e-6))
    )
    query = np.asarray(
        query * (np.float32(1.0) + norms["q_norm.weight"]["float"]),
        dtype=np.float32,
    )
    key = np.asarray(
        key * (np.float32(1.0) + norms["k_norm.weight"]["float"]),
        dtype=np.float32,
    )
    query = _apply_rope(query, cos4, sin4)
    key = _apply_rope(key, cos4, sin4)
    key_cache = initial_key_cache.copy()
    value_cache = initial_value_cache.copy()
    key_cache[:, base_context : base_context + 4] = key.transpose(1, 0, 2)
    value_cache[:, base_context : base_context + 4] = value.transpose(1, 0, 2)
    output = np.empty((4, 16, 256), dtype=np.float32)
    for token in range(4):
        length = base_context + token + 1
        for head in range(16):
            kv_head = head // 8
            scores = np.asarray(
                key_cache[kv_head, :length] @ query[token, head] * np.float32(0.0625),
                dtype=np.float32,
            )
            probabilities = np.exp(scores - np.max(scores)).astype(np.float32)
            probabilities /= np.sum(probabilities, dtype=np.float32)
            attended = np.asarray(
                probabilities @ value_cache[kv_head, :length], dtype=np.float32
            )
            sigmoid_gate = np.asarray(
                1.0 / (1.0 + np.exp(-gate[token, head])), dtype=np.float32
            )
            output[token, head] = attended * sigmoid_gate
    return {
        "prepared_q": query.reshape(4, 4096),
        "appended_key": key.transpose(1, 0, 2),
        "appended_value": value.transpose(1, 0, 2),
        "output": output.reshape(4, 4096),
    }


def _nrmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    error = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = max(float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))), 1.0e-12)
    return float(np.sqrt(np.mean(error ** 2)) / denominator)


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


def _bench_case(
    kernels: OrnithFullAttentionH4Kernels,
    repository: str,
    snapshot: Path,
    prefix: str,
    base_context: int,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import cupy as cp

    norms = _load_norms(snapshot, prefix)
    rng = np.random.default_rng(seed)
    max_context = base_context + 4
    q_gate4 = rng.normal(0.0, 0.55, size=(4, 8192)).astype(np.float32)
    key4 = rng.normal(0.0, 0.55, size=(4, 512)).astype(np.float32)
    value4 = rng.normal(0.0, 0.40, size=(4, 512)).astype(np.float32)
    key_cache = rng.normal(0.0, 0.15, size=(2, max_context, 256)).astype(np.float32)
    value_cache = rng.normal(0.0, 0.40, size=(2, max_context, 256)).astype(np.float32)
    cos4, sin4 = _rope(base_context)
    reference = _reference(
        norms, q_gate4, key4, value4, cos4, sin4,
        key_cache, value_cache, base_context,
    )
    gpu = {
        "q_gate4": cp.asarray(q_gate4),
        "key4": cp.asarray(key4),
        "value4": cp.asarray(value4),
        "q_norm": cp.asarray(norms["q_norm.weight"]["raw"]),
        "k_norm": cp.asarray(norms["k_norm.weight"]["raw"]),
        "cos4": cp.asarray(cos4),
        "sin4": cp.asarray(sin4),
        "prepared_q": cp.empty((4, 4096), dtype=cp.float32),
        "key_cache": cp.asarray(key_cache),
        "value_cache": cp.asarray(value_cache),
        "outputs": {
            arm: cp.empty((4, 4096), dtype=cp.float32) for arm in kernels.ARMS
        },
    }

    def prepare() -> None:
        kernels.prepare(
            gpu["q_gate4"], gpu["key4"], gpu["value4"],
            gpu["q_norm"], gpu["k_norm"], gpu["cos4"], gpu["sin4"],
            gpu["prepared_q"], gpu["key_cache"], gpu["value_cache"],
            base_context, max_context,
        )

    def attention(arm: str) -> None:
        kernels.attention(
            arm, gpu["prepared_q"], gpu["q_gate4"], gpu["key_cache"],
            gpu["value_cache"], gpu["outputs"][arm], base_context, max_context,
        )

    prepare()
    for arm in kernels.ARMS:
        attention(arm)
    cp.cuda.get_current_stream().synchronize()
    preparation_quality = {
        "prepared_q": _quality(cp.asnumpy(gpu["prepared_q"]), reference["prepared_q"]),
        "appended_key": _quality(
            cp.asnumpy(gpu["key_cache"][:, base_context : base_context + 4]),
            reference["appended_key"],
        ),
        "appended_value": _quality(
            cp.asnumpy(gpu["value_cache"][:, base_context : base_context + 4]),
            reference["appended_value"],
        ),
    }
    arms = {}
    for arm in kernels.ARMS:
        first = cp.asnumpy(gpu["outputs"][arm])
        attention(arm)
        cp.cuda.get_current_stream().synchronize()
        second = cp.asnumpy(gpu["outputs"][arm])
        arms[arm] = {
            "quality": _quality(first, reference["output"]),
            "fresh_input_bit_deterministic": bool(
                np.array_equal(first.view(np.uint32), second.view(np.uint32))
            ),
        }

    prepare_timing = _measure(cp, prepare, warmup, repeats)
    for arm in kernels.ARMS:
        def run_attention(selected=arm) -> None:
            attention(selected)

        def run_complete(selected=arm) -> None:
            prepare()
            attention(selected)

        arms[arm]["attention_timing_ms"] = _measure(cp, run_attention, warmup, repeats)
        arms[arm]["complete_timing_ms"] = _measure(cp, run_complete, warmup, repeats)

    valid = [
        arm for arm, row in arms.items()
        if row["quality"]["nrmse"] <= 5.0e-5
        and row["quality"]["candidate_finite"]
        and row["fresh_input_bit_deterministic"]
    ]
    if not valid:
        raise RuntimeError(f"{repository} ctx{base_context}: no correct attention arm")
    selected = min(valid, key=lambda arm: float(arms[arm]["complete_timing_ms"]["p50"]))
    record = {
        "repository": repository,
        "snapshot": str(snapshot),
        "prefix": prefix,
        "base_context": base_context,
        "max_context": max_context,
        "norm_shards": {name: row["shard"] for name, row in norms.items()},
        "preparation_quality": preparation_quality,
        "preparation_timing_ms": prepare_timing,
        "arms": arms,
        "selected_arm": selected,
        "selected_complete_ms": float(arms[selected]["complete_timing_ms"]["p50"]),
        "projected_10_full_layers_ms_h4": 10.0 * float(
            arms[selected]["complete_timing_ms"]["p50"]
        ),
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
    out = RESULTS / "S100_PHASE68_ORNITH_FULL_ATTN_H4.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase68_ornith_full_attn_h4",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "contexts": list(CONTEXTS),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    cp = None
    try:
        import cupy as cp_module

        cp = cp_module
        kernels = OrnithFullAttentionH4Kernels()
        resources = kernels.resource_audit()
        repositories = (
            ("official", args.official.resolve(), "model.language_model.layers.23.self_attn"),
            ("pottokao", args.pottokao.resolve(), "model.layers.23.self_attn"),
        )
        records = []
        for repository_index, (repository, snapshot, prefix) in enumerate(repositories):
            if not snapshot.is_dir():
                raise FileNotFoundError(snapshot)
            for context_index, context in enumerate(CONTEXTS):
                records.append(_bench_case(
                    kernels, repository, snapshot, prefix, context,
                    68000 + repository_index * 10 + context_index,
                    args.warmup, args.repeats,
                ))
        phase67 = json.loads(PHASE67.read_text("utf-8"))
        known_floor = float(phase67["budget"]["combined_known_floor_ms_h4"])
        worse_ten = max(row["projected_10_full_layers_ms_h4"] for row in records)
        combined = known_floor + worse_ten
        preparation_ok = all(
            metric["nrmse"] <= 5.0e-5
            and metric["candidate_finite"]
            and metric["reference_finite"]
            for row in records for metric in row["preparation_quality"].values()
        )
        arms_ok = all(
            arm["quality"]["nrmse"] <= 5.0e-5
            and arm["quality"]["candidate_finite"]
            and arm["quality"]["reference_finite"]
            for row in records for arm in row["arms"].values()
        )
        resource_ok = all(
            (row.get("local_size_bytes") or 0) == 0
            and (row.get("num_regs") or 10_000) <= 96
            and (row.get("max_threads_per_block") or 0) >= 256
            for row in resources.values()
        )
        gates = {
            "P68_G1_reference_quality": preparation_ok and arms_ok,
            "P68_G2_fresh_input_deterministic": all(
                arm["fresh_input_bit_deterministic"]
                for row in records for arm in row["arms"].values()
            ),
            "P68_G3_selected_le_0_40ms_and_ten_le_4ms": all(
                row["selected_complete_ms"] <= 0.40
                and row["projected_10_full_layers_ms_h4"] <= 4.0
                for row in records
            ),
            "P68_G4_phase67_plus_full_attention_below_65_boundary": combined < 4000.0 / 65.0,
            "P68_G5_resource_budget": resource_ok,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "records": records,
            "resources": resources,
            "budget": {
                "phase67_known_floor_ms_h4": known_floor,
                "worse_10_full_layers_ms_h4": worse_ten,
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
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, KERNELS, PHASE67))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "records": [
            {
                "repository": row["repository"],
                "context": row["base_context"],
                "selected": row["selected_arm"],
                "selected_complete_ms": row["selected_complete_ms"],
                "projected_10_ms": row["projected_10_full_layers_ms_h4"],
                "arms_ms": {
                    arm: data["complete_timing_ms"]["p50"]
                    for arm, data in row["arms"].items()
                },
                "arms_nrmse": {
                    arm: data["quality"]["nrmse"] for arm, data in row["arms"].items()
                },
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
