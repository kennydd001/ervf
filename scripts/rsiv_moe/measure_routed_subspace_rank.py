from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import safetensors
import torch
import torch.nn.functional as F
from safetensors import safe_open

from moe_lab.moe_layer import load_moe_layer
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.subspace import (
    append_residual_direction,
    cold_byte_fraction,
    fit_origin_subspace,
    image_storage_elements,
    relative_residual_ratio,
    select_validation_candidate,
)


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LAYERS = (1, 13, 26)
RANKS = (4, 8, 16, 32, 64, 128)
PRIMARY_RANKS = (4, 8, 16, 32)
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
EXPERTS = 64
TOP_K = 6
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1408
BLOCK_SIZE = 128
PREFIX_TOKENS = 96
FULL_RANK_RELATIVE_TOLERANCE = 2e-5
FULL_RANK_MAXIMUM_TOLERANCE = 2e-4
CAPTURE = ROOT / "reports/runs/rsiv_moe/routed_subspace_pilot.safetensors"
CAPTURE_REPORT = ROOT / "reports/rsiv_moe/routed_subspace_capture.json"
PREREGISTRATION = ROOT / "reports/rsiv_moe/RSIV_MOE_PREREGISTRATION.md"
SELECTION = ROOT / "reports/rsiv_moe/p1_validation_selection_v2.json"
INVALID_SELECTION_V1 = ROOT / "reports/rsiv_moe/p1_validation_selection.json"
CONTROL_ADDENDUM = ROOT / "reports/rsiv_moe/RSIV_MOE_P1_CONTROL_ADDENDUM_001.md"
RESULT = ROOT / "reports/rsiv_moe/routed_subspace_rank.json"
REPORT = ROOT / "reports/rsiv_moe/ROUTED_SUBSPACE_RANK.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage preregistered RSIV P1 routed-subspace census."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select-validation", action="store_true")
    group.add_argument("--open-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def capture_metadata() -> dict[str, Any]:
    with safe_open(CAPTURE, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("capture model revision mismatch")
    if metadata.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError("capture dataset revision mismatch")
    return metadata


def split_offsets() -> dict[str, tuple[int, int]]:
    metadata = capture_metadata()
    raw = json.loads(metadata["split_offsets"])
    return {name: (int(value[0]), int(value[1])) for name, value in raw.items()}


def load_split(layer: int, split: str) -> dict[str, torch.Tensor]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("unknown split")
    start, stop = split_offsets()[split]
    prefix = f"layer_{layer:02d}"
    with safe_open(CAPTURE, framework="pt", device="cpu") as handle:
        return {
            "x": handle.get_slice(f"{prefix}_moe_input")[start:stop],
            "ids": handle.get_slice(f"{prefix}_router_ids")[start:stop].long(),
            "weights": handle.get_slice(f"{prefix}_router_weights")[start:stop].float(),
            "z": handle.get_slice(f"{prefix}_intermediate_z")[start:stop],
        }


def expert_positions(ids: torch.Tensor, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = (ids == expert).nonzero(as_tuple=False)
    return positions[:, 0], positions[:, 1]


def expert_matrices(
    data: dict[str, torch.Tensor], expert: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids, slots = expert_positions(data["ids"], expert)
    return (
        data["x"][token_ids],
        data["z"][token_ids, slots],
        data["weights"][token_ids, slots],
        token_ids * TOP_K + slots,
    )


def finite_quantile(values: torch.Tensor, q: float) -> float | None:
    values = values.double().reshape(-1)
    finite = values[torch.isfinite(values)]
    return float(torch.quantile(finite, q).item()) if finite.numel() else None


def metric_summary(
    x_residual: torch.Tensor,
    z_residual: torch.Tensor,
    router_weights: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    if x_residual.shape != z_residual.shape or x_residual.shape != router_weights.shape:
        raise ValueError("aligned invocation vectors are required")
    x_hit = x_residual <= threshold
    z_hit = z_residual <= threshold
    double_hit = x_hit & z_hit
    cold = cold_byte_fraction(~x_hit, ~z_hit)
    mean_cold = float(cold.mean().item()) if cold.numel() else 1.0
    mass = router_weights.double()
    total_mass = float(mass.sum().item())
    mass_cold = (
        float((mass * cold).sum().item()) / total_mass if total_mass > 0.0 else 1.0
    )
    return {
        "invocations": int(x_residual.numel()),
        "threshold": threshold,
        "x_fast_fraction": float(x_hit.double().mean().item()),
        "z_fast_fraction": float(z_hit.double().mean().item()),
        "double_gate_fast_fraction": float(double_hit.double().mean().item()),
        "router_mass_double_gate_fast_fraction": (
            float(mass[double_hit].sum().item()) / total_mass if total_mass > 0.0 else 0.0
        ),
        "projected_cold_byte_fraction": mean_cold,
        "projected_routed_cold_byte_reduction": (
            1.0 / mean_cold if mean_cold > 0.0 else None
        ),
        "router_mass_projected_cold_byte_fraction": mass_cold,
        "router_mass_projected_cold_byte_reduction": (
            1.0 / mass_cold if mass_cold > 0.0 else None
        ),
        "x_residual_p50": finite_quantile(x_residual, 0.50),
        "x_residual_p95": finite_quantile(x_residual, 0.95),
        "x_residual_p99": finite_quantile(x_residual, 0.99),
        "z_residual_p50": finite_quantile(z_residual, 0.50),
        "z_residual_p95": finite_quantile(z_residual, 0.95),
        "z_residual_p99": finite_quantile(z_residual, 0.99),
    }


def rank_census(layer: int, split: str, data: dict[str, torch.Tensor]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    input_ranks: list[int] = []
    intermediate_ranks: list[int] = []
    counts: list[int] = []
    for expert in range(EXPERTS):
        x, z, weights, _flat = expert_matrices(data, expert)
        x_fit = fit_origin_subspace(x)
        z_fit = fit_origin_subspace(z)
        count = int(x.shape[0])
        if z.shape[0] != count:
            raise RuntimeError("x/z routed counts differ")
        if x_fit.stored_rank > count or z_fit.stored_rank > count:
            raise RuntimeError("rank(X_e) <= n_e control failed")
        rows.append(
            {
                "expert": expert,
                "count": count,
                "router_mass": float(weights.double().sum().item()),
                "input_stored_rank": x_fit.stored_rank,
                "input_effective_rank": x_fit.effective_rank,
                "input_energy_ranks": x_fit.energy_ranks,
                "intermediate_stored_rank": z_fit.stored_rank,
                "intermediate_effective_rank": z_fit.effective_rank,
                "intermediate_energy_ranks": z_fit.energy_ranks,
            }
        )
        counts.append(count)
        input_ranks.append(x_fit.stored_rank)
        intermediate_ranks.append(z_fit.stored_rank)
    tokens = int(data["x"].shape[0])
    count_sum = sum(counts)
    expected = TOP_K * tokens
    if count_sum != expected:
        raise RuntimeError("sum_e n_e == top_k*T control failed")
    elements = image_storage_elements(
        HIDDEN_SIZE, INTERMEDIATE_SIZE, input_ranks, intermediate_ranks
    )
    bound = (2 * HIDDEN_SIZE + 3 * INTERMEDIATE_SIZE) * TOP_K * tokens
    if elements > bound:
        raise RuntimeError("expert-count cancellation bound failed")

    def weighted_rank_summary(key: str) -> dict[str, float]:
        expanded = torch.tensor(
            [row[key] for row in rows for _ in range(row["count"])],
            dtype=torch.float64,
        )
        return {
            "mean": float(expanded.mean().item()),
            "p50": float(torch.quantile(expanded, 0.50).item()),
            "p95": float(torch.quantile(expanded, 0.95).item()),
            "p99": float(torch.quantile(expanded, 0.99).item()),
            "maximum": float(expanded.max().item()),
        }

    return {
        "layer": layer,
        "split": split,
        "tokens": tokens,
        "expert_invocations": count_sum,
        "expected_expert_invocations": expected,
        "rare_experts_count_lt_4": sum(count < 4 for count in counts),
        "unseen_experts": sum(count == 0 for count in counts),
        "minimum_expert_count": min(counts),
        "maximum_expert_count": max(counts),
        "input_rank_invocation_weighted": weighted_rank_summary("input_stored_rank"),
        "intermediate_rank_invocation_weighted": weighted_rank_summary(
            "intermediate_stored_rank"
        ),
        "full_rank_image_elements": elements,
        "expert_count_cancellation_bound_elements": bound,
        "bound_utilization": elements / bound,
        "controls": {
            "sum_counts_exact": count_sum == expected,
            "all_input_ranks_le_counts": all(
                rank <= count for rank, count in zip(input_ranks, counts)
            ),
            "all_intermediate_ranks_le_counts": all(
                rank <= count for rank, count in zip(intermediate_ranks, counts)
            ),
            "storage_bound_pass": elements <= bound,
        },
        "experts": rows,
    }


def offline_residuals(
    train: dict[str, torch.Tensor],
    evaluation: dict[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    invocations = evaluation["ids"].numel()
    result = {
        rank: {
            "x": torch.full((invocations,), float("inf"), dtype=torch.float64),
            "z": torch.full((invocations,), float("inf"), dtype=torch.float64),
            "weights": evaluation["weights"].reshape(-1).double(),
        }
        for rank in RANKS
    }
    for expert in range(EXPERTS):
        train_x, train_z, _train_weights, _train_flat = expert_matrices(train, expert)
        eval_x, eval_z, _eval_weights, eval_flat = expert_matrices(evaluation, expert)
        x_fit = fit_origin_subspace(train_x)
        z_fit = fit_origin_subspace(train_z)
        for rank in RANKS:
            x_basis = x_fit.basis[:, : min(rank, x_fit.basis.shape[1])]
            z_basis = z_fit.basis[:, : min(rank, z_fit.basis.shape[1])]
            result[rank]["x"][eval_flat] = relative_residual_ratio(eval_x, x_basis)
            result[rank]["z"][eval_flat] = relative_residual_ratio(eval_z, z_basis)
    return result


def causal_prefix_residuals(
    data: dict[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    if data["x"].shape[0] % BLOCK_SIZE:
        raise ValueError("split token count must be divisible by block size")
    block_count = data["x"].shape[0] // BLOCK_SIZE
    future_invocations = block_count * (BLOCK_SIZE - PREFIX_TOKENS) * TOP_K
    result = {
        rank: {
            "x": torch.full((future_invocations,), float("inf"), dtype=torch.float64),
            "z": torch.full((future_invocations,), float("inf"), dtype=torch.float64),
            "weights": torch.empty(future_invocations, dtype=torch.float64),
        }
        for rank in RANKS
    }
    output_offset = 0
    for block in range(block_count):
        start = block * BLOCK_SIZE
        prefix_slice = slice(start, start + PREFIX_TOKENS)
        future_slice = slice(start + PREFIX_TOKENS, start + BLOCK_SIZE)
        prefix = {key: value[prefix_slice] for key, value in data.items()}
        future = {key: value[future_slice] for key, value in data.items()}
        block_invocations = future["ids"].numel()
        for rank in RANKS:
            result[rank]["weights"][output_offset : output_offset + block_invocations] = (
                future["weights"].reshape(-1).double()
            )
        for expert in range(EXPERTS):
            prefix_x, prefix_z, _prefix_weights, _prefix_flat = expert_matrices(
                prefix, expert
            )
            future_x, future_z, _future_weights, future_flat = expert_matrices(
                future, expert
            )
            x_fit = fit_origin_subspace(prefix_x)
            z_fit = fit_origin_subspace(prefix_z)
            destination = future_flat + output_offset
            for rank in RANKS:
                q = x_fit.basis[:, : min(rank, x_fit.basis.shape[1])]
                p = z_fit.basis[:, : min(rank, z_fit.basis.shape[1])]
                result[rank]["x"][destination] = relative_residual_ratio(future_x, q)
                result[rank]["z"][destination] = relative_residual_ratio(future_z, p)
        output_offset += block_invocations
    if output_offset != future_invocations:
        raise RuntimeError("causal-prefix output alignment failed")
    return result


def aggregate_metric(
    sources: dict[int, dict[int, dict[str, torch.Tensor]]],
    rank: int,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_layer = {
        str(layer): metric_summary(
            values[rank]["x"], values[rank]["z"], values[rank]["weights"], threshold
        )
        for layer, values in sources.items()
    }
    x = torch.cat([sources[layer][rank]["x"] for layer in LAYERS])
    z = torch.cat([sources[layer][rank]["z"] for layer in LAYERS])
    weights = torch.cat([sources[layer][rank]["weights"] for layer in LAYERS])
    return metric_summary(x, z, weights, threshold), per_layer


def build_grid(
    offline: dict[int, dict[int, dict[str, torch.Tensor]]],
    causal: dict[int, dict[int, dict[str, torch.Tensor]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in RANKS:
        for threshold in THRESHOLDS:
            offline_aggregate, offline_layers = aggregate_metric(
                offline, rank, threshold
            )
            causal_aggregate, causal_layers = aggregate_metric(causal, rank, threshold)
            rows.append(
                {
                    "rank_cap": rank,
                    "threshold": threshold,
                    "offline_double_fast_fraction": offline_aggregate[
                        "double_gate_fast_fraction"
                    ],
                    "offline_cold_byte_reduction": offline_aggregate[
                        "projected_routed_cold_byte_reduction"
                    ],
                    "causal_double_fast_fraction": causal_aggregate[
                        "double_gate_fast_fraction"
                    ],
                    "causal_cold_byte_reduction": causal_aggregate[
                        "projected_routed_cold_byte_reduction"
                    ],
                    "offline": {
                        "aggregate": offline_aggregate,
                        "layers": offline_layers,
                    },
                    "causal_prefix_future": {
                        "aggregate": causal_aggregate,
                        "layers": causal_layers,
                    },
                }
            )
    return rows


def regression_accumulator() -> dict[str, dict[str, float]]:
    return {
        name: {"delta_square_sum": 0.0, "reference_square_sum": 0.0, "maximum_absolute_error": 0.0}
        for name in ("x", "g", "u", "z", "y")
    }


def update_regression(
    accumulator: dict[str, dict[str, float]],
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> None:
    delta = candidate.double() - reference.double()
    accumulator[name]["delta_square_sum"] += float(delta.square().sum().item())
    accumulator[name]["reference_square_sum"] += float(
        reference.double().square().sum().item()
    )
    accumulator[name]["maximum_absolute_error"] = max(
        accumulator[name]["maximum_absolute_error"],
        float(delta.abs().max().item()) if delta.numel() else 0.0,
    )


def full_rank_operator_controls(
    train_by_layer: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("full-rank operator controls require CUDA")
    device = torch.device("cuda")
    per_layer: dict[str, Any] = {}
    global_accumulator = regression_accumulator()
    raw_z_bit_exact = True
    raw_z_maximum_absolute_error = 0.0
    for layer_index in LAYERS:
        moe = load_moe_layer(ROOT / "models/deepseek-v2-lite", layer_index, device)
        data = train_by_layer[layer_index]
        layer_accumulator = regression_accumulator()
        layer_raw_exact = True
        layer_raw_max = 0.0
        for expert_index, expert in enumerate(moe.experts):
            x_cpu, raw_z_cpu, _weights, _flat = expert_matrices(data, expert_index)
            if x_cpu.shape[0] == 0:
                continue
            x64 = x_cpu.double()
            q64 = fit_origin_subspace(x64).basis
            x = x_cpu.to(device=device, dtype=torch.float32)
            q = q64.to(device=device, dtype=torch.float32)
            gate = expert.gate.float()
            up = expert.up.float()
            down = expert.down.float()
            direct_g = F.linear(x, gate)
            direct_u = F.linear(x, up)
            direct_z = F.silu(direct_g) * direct_u
            direct_y = F.linear(direct_z, down)
            a = gate @ q
            b = up @ q
            coordinates = x @ q
            cached_x = coordinates @ q.T
            cached_g = coordinates @ a.T
            cached_u = coordinates @ b.T
            cached_z = F.silu(cached_g) * cached_u
            p64 = fit_origin_subspace(direct_z.detach().cpu().double()).basis
            p = p64.to(device=device, dtype=torch.float32)
            c = down @ p
            cached_y = (cached_z @ p) @ c.T
            for name, reference, candidate in (
                ("x", x, cached_x),
                ("g", direct_g, cached_g),
                ("u", direct_u, cached_u),
                ("z", direct_z, cached_z),
                ("y", direct_y, cached_y),
            ):
                update_regression(layer_accumulator, name, reference, candidate)
                update_regression(global_accumulator, name, reference, candidate)

            x_bf16 = x_cpu.to(device=device, dtype=expert.gate.dtype)
            bf16_z = F.silu(F.linear(x_bf16, expert.gate)) * F.linear(
                x_bf16, expert.up
            )
            raw_z = raw_z_cpu.to(device)
            layer_raw_exact = layer_raw_exact and bool(torch.equal(bf16_z, raw_z))
            raw_delta = (bf16_z.float() - raw_z.float()).abs()
            local_raw_max = float(raw_delta.max().item()) if raw_delta.numel() else 0.0
            layer_raw_max = max(layer_raw_max, local_raw_max)
            raw_z_bit_exact = raw_z_bit_exact and bool(torch.equal(bf16_z, raw_z))
            raw_z_maximum_absolute_error = max(raw_z_maximum_absolute_error, local_raw_max)
            del x, q, gate, up, down, direct_g, direct_u, direct_z, direct_y
            del a, b, coordinates, cached_x, cached_g, cached_u, cached_z, p, c, cached_y
            del x_bf16, bf16_z, raw_z

        def finish(accumulator: dict[str, dict[str, float]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name, values in accumulator.items():
                denominator = values["reference_square_sum"]
                relative = (
                    math.sqrt(values["delta_square_sum"] / denominator)
                    if denominator > 0.0
                    else 0.0
                )
                maximum = values["maximum_absolute_error"]
                result[name] = {
                    "relative_l2": relative,
                    "maximum_absolute_error": maximum,
                    "pass": (
                        relative <= FULL_RANK_RELATIVE_TOLERANCE
                        and maximum <= FULL_RANK_MAXIMUM_TOLERANCE
                    ),
                }
            return result

        layer_finished = finish(layer_accumulator)
        per_layer[str(layer_index)] = {
            "operator_images": layer_finished,
            "all_operator_image_controls_pass": all(
                value["pass"] for value in layer_finished.values()
            ),
            "stored_bf16_z_bit_exact": layer_raw_exact,
            "stored_bf16_z_maximum_absolute_error": layer_raw_max,
        }
        del moe
        gc.collect()
        torch.cuda.empty_cache()
        print(f"full-rank operator control layer={layer_index} complete", flush=True)

    global_finished: dict[str, Any] = {}
    for name, values in global_accumulator.items():
        denominator = values["reference_square_sum"]
        relative = (
            math.sqrt(values["delta_square_sum"] / denominator)
            if denominator > 0.0
            else 0.0
        )
        maximum = values["maximum_absolute_error"]
        global_finished[name] = {
            "relative_l2": relative,
            "maximum_absolute_error": maximum,
            "pass": (
                relative <= FULL_RANK_RELATIVE_TOLERANCE
                and maximum <= FULL_RANK_MAXIMUM_TOLERANCE
            ),
        }
    all_pass = (
        all(value["pass"] for value in global_finished.values())
        and all(value["all_operator_image_controls_pass"] for value in per_layer.values())
    )
    return {
        "relative_l2_tolerance": FULL_RANK_RELATIVE_TOLERANCE,
        "maximum_absolute_tolerance": FULL_RANK_MAXIMUM_TOLERANCE,
        "global": global_finished,
        "layers": per_layer,
        "stored_bf16_z_batch_shape_bit_exact_diagnostic": raw_z_bit_exact,
        "stored_bf16_z_batch_shape_maximum_absolute_error_diagnostic": raw_z_maximum_absolute_error,
        "stored_bf16_z_diagnostic_claim_boundary": (
            "Not a required control: BF16 GEMM can change accumulation order when "
            "the same expert rows are regrouped into different batch shapes."
        ),
        "full_weight_fallback_control": {
            "bit_exact": True,
            "definition": "A miss calls the unchanged original expert projection; P1 introduces no fallback approximation.",
        },
        "all_required_controls_pass": all_pass,
    }


def online_curves(
    data_by_layer: dict[int, dict[str, torch.Tensor]],
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in LAYERS:
        data = data_by_layer[layer]
        block_count = data["x"].shape[0] // BLOCK_SIZE
        layer_rows: dict[str, Any] = {}
        for rank_cap in RANKS:
            token_x_additions = torch.zeros(data["x"].shape[0], dtype=torch.int64)
            token_z_additions = torch.zeros_like(token_x_additions)
            x_misses: list[bool] = []
            z_misses: list[bool] = []
            weights: list[float] = []
            final_x_ranks: list[int] = []
            final_z_ranks: list[int] = []
            for block in range(block_count):
                q = [torch.empty(HIDDEN_SIZE, 0, dtype=torch.float64) for _ in range(EXPERTS)]
                p = [torch.empty(INTERMEDIATE_SIZE, 0, dtype=torch.float64) for _ in range(EXPERTS)]
                for local_token in range(BLOCK_SIZE):
                    token = block * BLOCK_SIZE + local_token
                    for slot in range(TOP_K):
                        expert = int(data["ids"][token, slot].item())
                        x = data["x"][token].double()
                        z = data["z"][token, slot].double()
                        x_ratio = float(relative_residual_ratio(x.unsqueeze(0), q[expert])[0].item())
                        z_ratio = float(relative_residual_ratio(z.unsqueeze(0), p[expert])[0].item())
                        x_miss = x_ratio > threshold
                        z_miss = z_ratio > threshold
                        if x_miss and q[expert].shape[1] < rank_cap:
                            q[expert], _ratio, added = append_residual_direction(q[expert], x)
                            token_x_additions[token] += int(added)
                        if z_miss and p[expert].shape[1] < rank_cap:
                            p[expert], _ratio, added = append_residual_direction(p[expert], z)
                            token_z_additions[token] += int(added)
                        x_misses.append(x_miss)
                        z_misses.append(z_miss)
                        weights.append(float(data["weights"][token, slot].item()))
                final_x_ranks.extend(int(value.shape[1]) for value in q)
                final_z_ranks.extend(int(value.shape[1]) for value in p)
            x_miss_tensor = torch.tensor(x_misses, dtype=torch.bool)
            z_miss_tensor = torch.tensor(z_misses, dtype=torch.bool)
            cold = cold_byte_fraction(x_miss_tensor, z_miss_tensor)
            layer_rows[str(rank_cap)] = {
                "threshold": threshold,
                "invocations": len(x_misses),
                "double_gate_fast_fraction": float(
                    ((~x_miss_tensor) & (~z_miss_tensor)).double().mean().item()
                ),
                "projected_cold_byte_fraction": float(cold.mean().item()),
                "projected_routed_cold_byte_reduction": (
                    1.0 / float(cold.mean().item())
                    if float(cold.mean().item()) > 0.0
                    else None
                ),
                "x_rank_additions": int(token_x_additions.sum().item()),
                "z_rank_additions": int(token_z_additions.sum().item()),
                "x_rank_additions_per_token": token_x_additions.tolist(),
                "z_rank_additions_per_token": token_z_additions.tolist(),
                "final_x_rank_mean_across_block_experts": float(
                    torch.tensor(final_x_ranks, dtype=torch.float64).mean().item()
                ),
                "final_z_rank_mean_across_block_experts": float(
                    torch.tensor(final_z_ranks, dtype=torch.float64).mean().item()
                ),
            }
        result[str(layer)] = layer_rows
    return result


def environment() -> dict[str, Any]:
    process = psutil.Process()
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "safetensors": safetensors.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "process_rss_bytes_at_report": process.memory_info().rss,
        "declared_process_budget_bytes": 32 * 1024**3,
    }


def render_report(result: dict[str, Any]) -> str:
    selected = result["selection"]["selected_candidate"]
    test = result["test_confirmation"]["selected_candidate_metrics"]
    validation = result["selection"]["selected_candidate"]
    lines = [
        "# RSIV-MoE P1 routed-subspace-rankcensus",
        "",
        "## Uitkomst",
        "",
        f"**P1-verdict: `{result['verdict']}`.**",
        "",
        result["verdict_explanation"],
        "",
        "## Bevroren kandidaat",
        "",
        f"- Rankcap: `{selected['rank_cap']}`.",
        f"- Gedeelde residualthreshold: `{selected['threshold']}`.",
        f"- Selectietype: `{selected['selection_kind']}`.",
        f"- Capture-SHA-256: `{result['capture_sha256']}`.",
        f"- Selection-lock-SHA-256: `{result['selection_sha256']}`.",
        "",
        "## Validation en eenmalig geopende test",
        "",
        "| Evaluatie | Split | Double-fast | Koude-bytereductie |",
        "|---|---|---:|---:|",
        (
            f"| Offline trainbasis | validation | "
            f"{validation['offline_double_fast_fraction']:.3%} | "
            f"{validation['offline_cold_byte_reduction']:.3f}× |"
        ),
        (
            f"| Causale 96→32-prefix | validation | "
            f"{validation['causal_double_fast_fraction']:.3%} | "
            f"{validation['causal_cold_byte_reduction']:.3f}× |"
        ),
        (
            f"| Offline trainbasis | test | "
            f"{test['offline']['aggregate']['double_gate_fast_fraction']:.3%} | "
            f"{test['offline']['aggregate']['projected_routed_cold_byte_reduction']:.3f}× |"
        ),
        (
            f"| Causale 96→32-prefix | test | "
            f"{test['causal_prefix_future']['aggregate']['double_gate_fast_fraction']:.3%} | "
            f"{test['causal_prefix_future']['aggregate']['projected_routed_cold_byte_reduction']:.3f}× |"
        ),
        "",
        "Primaire eis voor beide evaluaties: minstens 92% double-fast en minstens 10× minder geprojecteerde koude expertbytes bij rank maximaal 32.",
        "",
        "## Exacte controles",
        "",
        f"- Capture-routes en routergewichten: `{result['controls']['capture_controls_pass']}`.",
        f"- Rank/count- en expert-count-cancellationcontroles: `{result['controls']['rank_and_bound_controls_pass']}`.",
        f"- Full-rank operatorimages (`x/g/u/z/y`): `{result['controls']['operator_image_controls']['all_required_controls_pass']}`.",
        (
            "- Opgeslagen BF16-`z`, opnieuw gegroepeerd met andere GEMM-batchvorm, "
            f"bit-exact (diagnostisch, geen gate): `"
            f"{result['controls']['operator_image_controls']['stored_bf16_z_batch_shape_bit_exact_diagnostic']}`."
        ),
        "",
        "## Claimgrens",
        "",
        "Dit is een rank- en page-faultscreen. De koude bytes zijn analytische packed-int4-boekhouding; atlasreads, projectiecompute, kwaliteit, latency en SSD-stalls zijn nog niet gemeten. Een positief P1-resultaat is daarom geen Eureka en geen runtimeclaim.",
        "",
        "## Volgende actie",
        "",
        result["next_action"],
        "",
    ]
    return "\n".join(lines)


def validation_phase() -> None:
    if SELECTION.exists():
        raise FileExistsError(f"validation selection already exists: {SELECTION}")
    if (
        not CAPTURE.is_file()
        or not CAPTURE_REPORT.is_file()
        or not PREREGISTRATION.is_file()
        or not CONTROL_ADDENDUM.is_file()
        or not INVALID_SELECTION_V1.is_file()
    ):
        raise FileNotFoundError(
            "capture, reports, preregistration, addendum and preserved v1 lock are required"
        )
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    capture_hash = sha256_file(CAPTURE)
    capture_report = json.loads(CAPTURE_REPORT.read_text(encoding="utf-8"))
    if capture_report["capture_sha256"] != capture_hash:
        raise RuntimeError("capture report hash mismatch")
    capture_controls_pass = all(
        control["route_ids_exact"]
        and control["router_weight_maximum_absolute_error"] <= 1e-6
        and control["sum_expert_invocations"] == control["expected_expert_invocations"]
        for control in capture_report["controls"].values()
    )
    if not capture_controls_pass:
        raise RuntimeError("raw capture controls failed")

    train_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    validation_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    census: dict[str, Any] = {"train": {}, "validation": {}}
    offline_sources: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    causal_sources: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    for layer in LAYERS:
        train = load_split(layer, "train")
        validation = load_split(layer, "validation")
        train_by_layer[layer] = train
        validation_by_layer[layer] = validation
        census["train"][str(layer)] = rank_census(layer, "train", train)
        census["validation"][str(layer)] = rank_census(
            layer, "validation", validation
        )
        offline_sources[layer] = offline_residuals(train, validation)
        causal_sources[layer] = causal_prefix_residuals(validation)
        print(f"validation census layer={layer} complete", flush=True)
    grid = build_grid(offline_sources, causal_sources)
    selected = select_validation_candidate(grid)
    operator_controls = full_rank_operator_controls(train_by_layer)
    rank_controls_pass = all(
        all(layer["controls"].values())
        for split in census.values()
        for layer in split.values()
    )
    valid = capture_controls_pass and rank_controls_pass and operator_controls[
        "all_required_controls_pass"
    ]
    payload = {
        "kind": "rsiv_moe_p1_validation_selection_lock",
        "status": "validation_selected" if valid else "invalid_controls_failed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "capture": str(CAPTURE.relative_to(ROOT)).replace("\\", "/"),
        "capture_sha256": capture_hash,
        "capture_bytes": CAPTURE.stat().st_size,
        "preregistration": str(PREREGISTRATION.relative_to(ROOT)).replace("\\", "/"),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "control_addendum": str(CONTROL_ADDENDUM.relative_to(ROOT)).replace("\\", "/"),
        "control_addendum_sha256": sha256_file(CONTROL_ADDENDUM),
        "preserved_invalid_selection_v1": {
            "path": str(INVALID_SELECTION_V1.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(INVALID_SELECTION_V1),
            "status": "invalid_controls_failed",
        },
        "implementation_sha256": sha256_file(Path(__file__)),
        "subspace_module_sha256": sha256_file(
            ROOT / "src/moe_lab/rsiv_moe/subspace.py"
        ),
        "test_data_opened": False,
        "selection_rule": "preregistered validation-only global rank/threshold selection",
        "selected_candidate": selected,
        "validation_grid": grid,
        "rank_census": census,
        "controls": {
            "capture_controls_pass": capture_controls_pass,
            "rank_and_bound_controls_pass": rank_controls_pass,
            "operator_image_controls": operator_controls,
        },
        "environment": environment(),
    }
    write_json_once(SELECTION, payload)
    print(
        json.dumps(
            {
                "selection": str(SELECTION),
                "selection_sha256": sha256_file(SELECTION),
                "status": payload["status"],
                "selected_candidate": selected,
            },
            indent=2,
        )
    )


def test_phase() -> None:
    if RESULT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P1 result or report")
    if not SELECTION.is_file():
        raise FileNotFoundError("validation selection lock is required")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["status"] != "validation_selected":
        raise RuntimeError("validation controls were invalid; test remains closed")
    capture_hash = sha256_file(CAPTURE)
    if selection["capture_sha256"] != capture_hash:
        raise RuntimeError("capture changed after validation selection")
    selection_hash = sha256_file(SELECTION)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    rank = int(selection["selected_candidate"]["rank_cap"])
    threshold = float(selection["selected_candidate"]["threshold"])

    train_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    test_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    census: dict[str, Any] = {}
    offline_sources: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    causal_sources: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    for layer in LAYERS:
        train = load_split(layer, "train")
        test = load_split(layer, "test")
        train_by_layer[layer] = train
        test_by_layer[layer] = test
        census[str(layer)] = rank_census(layer, "test", test)
        offline_sources[layer] = offline_residuals(train, test)
        causal_sources[layer] = causal_prefix_residuals(test)
        print(f"test census layer={layer} complete", flush=True)
    test_grid = build_grid(offline_sources, causal_sources)
    selected_row = next(
        row
        for row in test_grid
        if int(row["rank_cap"]) == rank and float(row["threshold"]) == threshold
    )
    test_pass = (
        selected_row["offline_double_fast_fraction"] >= 0.92
        and selected_row["offline_cold_byte_reduction"] is not None
        and selected_row["offline_cold_byte_reduction"] >= 10.0
        and selected_row["causal_double_fast_fraction"] >= 0.92
        and selected_row["causal_cold_byte_reduction"] is not None
        and selected_row["causal_cold_byte_reduction"] >= 10.0
    )
    validation_pass = selection["selected_candidate"]["selection_kind"] == "primary_gate_pass"
    controls = selection["controls"]
    all_controls = (
        controls["capture_controls_pass"]
        and controls["rank_and_bound_controls_pass"]
        and controls["operator_image_controls"]["all_required_controls_pass"]
        and all(all(layer["controls"].values()) for layer in census.values())
    )
    verdict = (
        "screen_positive"
        if all_controls and validation_pass and test_pass
        else "screen_negative_v2"
        if all_controls
        else "invalid"
    )
    if verdict == "screen_positive":
        explanation = (
            "Dezelfde validation-gekozen rank/threshold haalt op test zowel de offline "
            "als causale prefix→future P1-gates. Dit opent uitsluitend P2; het is geen Eureka."
        )
        next_action = (
            "Preregistreer P2 en bouw echte A/B/C-operatorimages voor één laag; "
            "claim nog geen kwaliteit of snelheid."
        )
    elif verdict == "screen_negative_v2":
        explanation = (
            "De vooraf vastgelegde P1-screen faalt op validation of op de eenmalig geopende "
            "test voor dezelfde kandidaat. P2 op V2 wordt niet geopend."
        )
        next_action = (
            "Sluit P1 op V2. Een hogere-E-proef vereist een afzonderlijke preregistratie; "
            "dit resultaat mag niet post-hoc met een andere threshold worden gered."
        )
    else:
        explanation = "Minstens één verplichte controle faalde; er volgt geen empirisch verdict."
        next_action = "Diagnosticeer uitsluitend de controlefout en open test niet opnieuw."

    payload = {
        "kind": "rsiv_moe_p1_routed_subspace_rank_census",
        "status": "complete",
        "verdict": verdict,
        "verdict_explanation": explanation,
        "next_action": next_action,
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "capture_sha256": capture_hash,
        "selection_lock": str(SELECTION.relative_to(ROOT)).replace("\\", "/"),
        "selection_sha256": selection_hash,
        "selection": selection,
        "test_opened_once": True,
        "test_confirmation": {
            "selected_candidate_metrics": selected_row,
            "selected_candidate_pass": test_pass,
            "full_post_lock_grid_for_diagnostics_only": test_grid,
            "rank_census": census,
            "causal_online_curves_at_locked_threshold": online_curves(
                test_by_layer, threshold
            ),
        },
        "controls": {
            **controls,
            "test_rank_and_bound_controls_pass": all(
                all(layer["controls"].values()) for layer in census.values()
            ),
            "all_required_controls_pass": all_controls,
        },
        "claim_boundaries": {
            "runtime": "not measured",
            "quality": "not measured in P1",
            "cold_bytes": "optimistic packed-int4 projection excluding atlas and non-expert costs",
            "generalization": "V2-Lite only; no cross-architecture conclusion",
            "eureka": False,
        },
        "environment": environment(),
    }
    write_json_once(RESULT, payload)
    REPORT.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "report": str(REPORT),
                "verdict": verdict,
                "selection_sha256": selection_hash,
                "result_sha256": sha256_file(RESULT),
                "report_sha256": sha256_file(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.select_validation:
        validation_phase()
    else:
        test_phase()
