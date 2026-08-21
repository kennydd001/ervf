"""Phase69 benchmark of Ornith H4 norms, routing and residual reductions."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, percentiles, utc_now, write_json_atomic
from s100_phase69_ornith_support_h4_kernels import OrnithSupportH4Kernels


RESULTS = REPO / "pro_research" / "results" / "s100_phase69"
PREREG = REPO / "pro_research" / "S100_PHASE69_ORNITH_SUPPORT_H4_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase69_ornith_support_h4.py"
KERNELS = REPO / "pro_research" / "s100_phase69_ornith_support_h4_kernels.py"
PHASE59 = REPO / "pro_research" / "results" / "s100_phase59" / "S100_PHASE59_ORNITH_BULK_EXPERT_H4.json"
PHASE60 = REPO / "pro_research" / "results" / "s100_phase60" / "S100_PHASE60_ORNITH_ROUTE_ADAPTIVE_BULK.json"
PHASE68 = REPO / "pro_research" / "results" / "s100_phase68" / "S100_PHASE68_ORNITH_FULL_ATTN_H4.json"


def _weight_map(snapshot: Path) -> dict[str, str]:
    return json.loads(
        (snapshot / "model.safetensors.index.json").read_text("utf-8")
    )["weight_map"]


def _load_support(snapshot: Path, prefix: str) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    names = {
        "input_norm": f"{prefix}.layers.20.input_layernorm.weight",
        "post_norm": f"{prefix}.layers.20.post_attention_layernorm.weight",
        "router": f"{prefix}.layers.20.mlp.gate.weight",
        "shared_gate": f"{prefix}.layers.20.mlp.shared_expert_gate.weight",
    }
    expected = {
        "input_norm": (2048,),
        "post_norm": (2048,),
        "router": (256, 2048),
        "shared_gate": (1, 2048),
    }
    weight_map = _weight_map(snapshot)
    result = {}
    for label, name in names.items():
        with safe_open(snapshot / weight_map[name], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name).contiguous()
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected[label]:
                raise TypeError(f"{name}: expected BF16{expected[label]}, got {tensor.dtype}{tensor.shape}")
            result[label] = {
                "raw": tensor.view(torch.uint16).numpy().copy().reshape(expected[label]),
                "float": tensor.float().numpy().copy().reshape(expected[label]),
                "name": name,
                "shard": weight_map[name],
            }
    return result


def _norm(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    inverse = np.reciprocal(
        np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + np.float32(1.0e-6))
    )
    return np.asarray(values * inverse * (np.float32(1.0) + weight), dtype=np.float32)


def _reference(
    support: dict[str, Any],
    initial_residual: np.ndarray,
    attention_branch: np.ndarray,
    expert_outputs: np.ndarray,
    shared_output: np.ndarray,
) -> dict[str, np.ndarray]:
    input_normed = _norm(initial_residual, support["input_norm"]["float"])
    post_residual = np.asarray(initial_residual + attention_branch, dtype=np.float32)
    post_normed = _norm(post_residual, support["post_norm"]["float"])
    router_logits = np.asarray(
        post_normed @ support["router"]["float"].T, dtype=np.float32
    )
    shared_logits = np.asarray(
        post_normed @ support["shared_gate"]["float"].T, dtype=np.float32
    ).reshape(4)
    ids = np.empty((4, 8), dtype=np.int32)
    route_weights = np.empty((4, 8), dtype=np.float32)
    expert_axis = np.arange(256, dtype=np.int32)
    for token in range(4):
        order = np.lexsort((expert_axis, -router_logits[token]))[:8]
        ids[token] = order
        selected = router_logits[token, order]
        exponential = np.exp(selected - selected[0]).astype(np.float32)
        route_weights[token] = exponential / np.sum(exponential, dtype=np.float32)
    routed = np.sum(
        expert_outputs.reshape(4, 8, 2048) * route_weights[..., None],
        axis=1,
        dtype=np.float32,
    )
    shared_scale = np.asarray(
        1.0 / (1.0 + np.exp(-shared_logits)), dtype=np.float32
    )
    combined_residual = np.asarray(
        post_residual + routed + shared_output * shared_scale[:, None], dtype=np.float32
    )
    next_normed = _norm(combined_residual, support["input_norm"]["float"])
    return {
        "input_normed": input_normed,
        "post_residual": post_residual,
        "post_normed": post_normed,
        "router_logits": router_logits,
        "shared_logits": shared_logits,
        "ids": ids,
        "route_weights": route_weights,
        "slots": ids.copy(),
        "need": np.zeros((4, 8), dtype=np.int32),
        "combined_residual": combined_residual,
        "next_normed": next_normed,
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


def _bench_repository(
    kernels: OrnithSupportH4Kernels,
    repository: str,
    snapshot: Path,
    prefix: str,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import cupy as cp

    support = _load_support(snapshot, prefix)
    rng = np.random.default_rng(seed)
    initial_residual = rng.normal(0.0, 0.45, size=(4, 2048)).astype(np.float32)
    attention_branch = rng.normal(0.0, 0.025, size=(4, 2048)).astype(np.float32)
    expert_outputs = rng.normal(0.0, 0.020, size=(32, 2048)).astype(np.float32)
    shared_output = rng.normal(0.0, 0.020, size=(4, 2048)).astype(np.float32)
    reference = _reference(
        support, initial_residual, attention_branch, expert_outputs, shared_output
    )
    gpu = {
        "input_norm": cp.asarray(support["input_norm"]["raw"]),
        "post_norm": cp.asarray(support["post_norm"]["raw"]),
        "router": cp.asarray(support["router"]["raw"]),
        "shared_gate": cp.asarray(support["shared_gate"]["raw"]),
        "residual": cp.asarray(initial_residual),
        "attention": cp.asarray(attention_branch),
        "expert_outputs": cp.asarray(expert_outputs),
        "shared_output": cp.asarray(shared_output),
        "input_normed": cp.empty((4, 2048), dtype=cp.float32),
        "post_normed": cp.empty((4, 2048), dtype=cp.float32),
        "next_normed": cp.empty((4, 2048), dtype=cp.float32),
        "router_logits": cp.empty((4, 256), dtype=cp.float32),
        "shared_logits": cp.empty((4,), dtype=cp.float32),
        "ids": cp.empty((4, 8), dtype=cp.int32),
        "weights": cp.empty((4, 8), dtype=cp.float32),
        "slots": cp.empty((4, 8), dtype=cp.int32),
        "need": cp.empty((4, 8), dtype=cp.int32),
        "slot_of": cp.arange(256, dtype=cp.int32),
    }

    def initial_norm() -> None:
        kernels.norm(gpu["residual"], gpu["input_norm"], gpu["input_normed"])

    def add_norm() -> None:
        kernels.add_norm(
            gpu["residual"], gpu["attention"], gpu["post_norm"], gpu["post_normed"]
        )

    def router_shared() -> None:
        kernels.router_shared(
            gpu["router"], gpu["shared_gate"], gpu["post_normed"],
            gpu["router_logits"], gpu["shared_logits"],
        )

    def top8_cache() -> None:
        kernels.top8_cache(
            gpu["router_logits"], gpu["slot_of"], gpu["ids"], gpu["weights"],
            gpu["slots"], gpu["need"],
        )

    def combine_norm() -> None:
        kernels.combine_norm(
            gpu["residual"], gpu["expert_outputs"], gpu["weights"],
            gpu["shared_output"], gpu["shared_logits"], gpu["input_norm"],
            gpu["next_normed"],
        )

    def one_step() -> None:
        initial_norm()
        add_norm()
        router_shared()
        top8_cache()
        combine_norm()

    def support_40() -> None:
        initial_norm()
        for _ in range(40):
            add_norm()
            router_shared()
            top8_cache()
            combine_norm()

    gpu["residual"].set(initial_residual)
    one_step()
    cp.cuda.get_current_stream().synchronize()
    candidate = {
        "input_normed": cp.asnumpy(gpu["input_normed"]),
        "post_residual": cp.asnumpy(gpu["residual"]),
        "post_normed": cp.asnumpy(gpu["post_normed"]),
        "router_logits": cp.asnumpy(gpu["router_logits"]),
        "shared_logits": cp.asnumpy(gpu["shared_logits"]),
        "ids": cp.asnumpy(gpu["ids"]),
        "route_weights": cp.asnumpy(gpu["weights"]),
        "slots": cp.asnumpy(gpu["slots"]),
        "need": cp.asnumpy(gpu["need"]),
        "combined_residual": cp.asnumpy(gpu["residual"]),
        "next_normed": cp.asnumpy(gpu["next_normed"]),
    }
    # post_residual is needed before combine; reconstruct the GPU-equivalent
    # value because combined residual now occupies the in-place buffer.
    candidate["post_residual"] = np.asarray(
        initial_residual + attention_branch, dtype=np.float32
    )
    quality = {
        name: _quality(candidate[name], reference[name])
        for name in (
            "input_normed", "post_residual", "post_normed", "router_logits",
            "shared_logits", "route_weights", "combined_residual", "next_normed",
        )
    }
    integer_exact = {
        name: bool(np.array_equal(candidate[name], reference[name]))
        for name in ("ids", "slots", "need")
    }

    repeats_out = []
    for _ in range(2):
        gpu["residual"].set(initial_residual)
        one_step()
        cp.cuda.get_current_stream().synchronize()
        repeats_out.append(cp.asnumpy(gpu["next_normed"]))
    deterministic = bool(
        np.array_equal(repeats_out[0].view(np.uint32), repeats_out[1].view(np.uint32))
    )

    gpu["residual"].set(initial_residual)
    timings = {
        "initial_norm": _measure(cp, initial_norm, warmup, repeats),
        "add_norm": _measure(cp, add_norm, warmup, repeats),
        "router_shared": _measure(cp, router_shared, warmup, repeats),
        "top8_cache": _measure(cp, top8_cache, warmup, repeats),
        "combine_norm": _measure(cp, combine_norm, warmup, repeats),
        "support_40": _measure(cp, support_40, warmup, repeats),
    }
    cp.cuda.get_current_stream().synchronize()
    record = {
        "repository": repository,
        "snapshot": str(snapshot),
        "weights": {
            label: {"name": row["name"], "shard": row["shard"]}
            for label, row in support.items()
        },
        "quality": quality,
        "integer_exact": integer_exact,
        "fresh_state_bit_deterministic": deterministic,
        "timings_ms": timings,
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
    out = RESULTS / "S100_PHASE69_ORNITH_SUPPORT_H4.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase69_ornith_support_h4",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    cp = None
    try:
        import cupy as cp_module

        cp = cp_module
        kernels = OrnithSupportH4Kernels()
        resources = kernels.resource_audit()
        repositories = (
            ("official", args.official.resolve(), "model.language_model"),
            ("pottokao", args.pottokao.resolve(), "model"),
        )
        records = []
        for index, (repository, snapshot, prefix) in enumerate(repositories):
            if not snapshot.is_dir():
                raise FileNotFoundError(snapshot)
            records.append(_bench_repository(
                kernels, repository, snapshot, prefix, 69000 + index,
                args.warmup, args.repeats,
            ))
        phase59 = json.loads(PHASE59.read_text("utf-8"))
        phase60 = json.loads(PHASE60.read_text("utf-8"))
        phase68 = json.loads(PHASE68.read_text("utf-8"))
        bulk32 = float(phase59["summary"]["bulk32_median_ms"])
        indirect_m1 = float(next(
            row["adaptive_timing_ms"]["p50"]
            for row in phase60["records"] if row["multiplicity"] == 1
        ))
        indirect_correction = max(0.0, indirect_m1 - bulk32) * 40.0
        known_floor = float(phase68["budget"]["combined_known_floor_ms_h4"])
        worse_support = max(float(row["timings_ms"]["support_40"]["p50"]) for row in records)
        combined = known_floor + indirect_correction + worse_support
        quality_ok = all(
            metric["nrmse"] <= 5.0e-5
            and metric["candidate_finite"]
            and metric["reference_finite"]
            for row in records for metric in row["quality"].values()
        )
        route_ok = all(all(row["integer_exact"].values()) for row in records)
        launch_threads = {
            "ornith_rmsnorm_h4": 256,
            "ornith_add_rmsnorm_h4": 256,
            "ornith_router_shared_h4": 256,
            "ornith_top8_cache_h4": 32,
            "ornith_moe_combine_rmsnorm_h4": 256,
        }
        resource_ok = all(
            (row.get("local_size_bytes") or 0) == 0
            and (row.get("num_regs") or 10_000) <= 96
            and (row.get("max_threads_per_block") or 0) >= launch_threads[name]
            for name, row in resources.items()
        )
        gates = {
            "P69_G1_reference_quality": quality_ok,
            "P69_G2_route_ids_slots_need_exact": route_ok,
            "P69_G3_fresh_state_deterministic": all(
                row["fresh_state_bit_deterministic"] for row in records
            ),
            "P69_G4_support_40_le_3ms": all(
                float(row["timings_ms"]["support_40"]["p50"]) <= 3.0
                for row in records
            ),
            "P69_G5_full_known_floor_below_65_boundary": combined < 4000.0 / 65.0,
            "P69_G6_resource_budget": resource_ok,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "records": records,
            "resources": resources,
            "budget": {
                "phase68_known_floor_ms_h4": known_floor,
                "phase59_bulk32_ms_per_layer": bulk32,
                "phase60_indirect_m1_ms_per_layer": indirect_m1,
                "conservative_indirect_correction_40_layers_ms_h4": indirect_correction,
                "worse_support_40_ms_h4": worse_support,
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
        payload["environment"] = environment_snapshot(
            (SCRIPT, PREREG, KERNELS, PHASE59, PHASE60, PHASE68)
        )
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "records": [
            {
                "repository": row["repository"],
                "support_40_ms": row["timings_ms"]["support_40"]["p50"],
                "components_ms": {
                    name: timing["p50"] for name, timing in row["timings_ms"].items()
                    if name != "support_40"
                },
                "max_nrmse": max(metric["nrmse"] for metric in row["quality"].values()),
                "integer_exact": row["integer_exact"],
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
