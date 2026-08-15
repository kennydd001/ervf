from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow
import safetensors
import tokenizers
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from evaluate_crcq_oracle import (
    BLOCK_SIZE,
    COMPONENT_RELATIVE,
    DATASET_REVISION,
    MODEL_REVISION,
    SEED,
    TRACE_TOKENS_PER_SPLIT,
    corpus_tokens,
    evaluate_hidden,
    git_state,
    hardware_state,
    make_teacher_reference,
    metadata,
    nullable,
    quantized_copy,
    regression_summary,
    sequence_blocks,
    sha256_file,
    write_json_once,
)
from moe_lab.behavioral import rmsnorm
from moe_lab.craft_moe.sketchgate import (
    choose_validation_configuration,
    delta_patch,
    exact_schedule_mean_kl,
    false_negative_rate,
    high_damage_mask,
    mask_indices,
    mix_selected_outputs,
    nested_quantized_sketch_scores,
    oracle_recovery,
    probe_bank,
    sketch_metadata_accounting,
    stable_top_fraction_mask,
)
from moe_lab.dynamic_precision import (
    best_mask_per_cardinality,
    binary_upgrade_masks,
    discrete_rate_distortion,
    recover_cost_schedule,
)
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT


FULL_TOKENS_PER_SPLIT = 256
SMOKE_TOKENS = 32
BOOTSTRAP_RESAMPLES = 10_000
CANDIDATE_BATCH = 128
MASK_TOKEN_BATCH = 2
LAYER = 26
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1408
ACTIVE_EXPERTS = 6
EXPERTS = 64
UPGRADE_FRACTION = 0.25
HIGH_DAMAGE_FRACTION = 0.10
COMPONENT_NRMSE_TOLERANCE = 1e-3
COMPONENT_MAX_ABS_TOLERANCE = 0.25
RANKS = (4, 8, 16, 32, 64)
PROBE_DISTRIBUTIONS = ("gaussian", "rademacher")
PROBE_SEEDS = tuple(range(SEED, SEED + 5))
PRIMARY_SEED = SEED
FAMILIES = ("gate_only", "up_only", "down_only", "gate_up", "all_matrices")
PREREGISTRATION = Path(
    "reports/craft_moe/H4_SKETCHGATE_LAYER26_PREREGISTRATION.md"
)
REPLICATION_PREREGISTRATION = Path(
    "reports/craft_moe/H4_SKETCHGATE_TRACE_ANCHORED_REPLICATION_PREREGISTRATION.md"
)
REPLICATION_SPLIT_OFFSET = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered H4 layer-26 residual-syndrome SketchGate screen."
    )
    parser.add_argument(
        "--stage", choices=("smoke", "full", "replication"), default="full"
    )
    parser.add_argument(
        "--tokens-per-split", type=int, default=FULL_TOKENS_PER_SPLIT
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--candidate-batch", type=int, default=CANDIDATE_BATCH)
    parser.add_argument("--mask-token-batch", type=int, default=MASK_TOKEN_BATCH)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if not 1 <= args.tokens_per_split <= TRACE_TOKENS_PER_SPLIT:
        raise ValueError("tokens-per-split is outside the existing trace")
    if args.stage == "smoke":
        if args.splits != ("validation",) or args.tokens_per_split > SMOKE_TOKENS:
            raise ValueError("smoke is limited to at most 32 validation tokens")
    elif args.stage in ("full", "replication") and (
        args.tokens_per_split != FULL_TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.candidate_batch != CANDIDATE_BATCH
        or args.mask_token_batch != MASK_TOKEN_BATCH
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
    ):
        raise ValueError(
            "full/replication is fixed at 256 validation + 256 test, candidate batch 128, "
            "mask-token batch 2, 10k bootstrap, and seed 20260810"
        )
    if min(args.candidate_batch, args.mask_token_batch, args.bootstrap_resamples) < 1:
        raise ValueError("batch sizes and bootstrap count must be positive")
    if args.output_json is None:
        if args.stage == "full":
            relative = Path("reports/craft_moe/sketchgate.json")
        elif args.stage == "replication":
            relative = Path(
                "reports/craft_moe/sketchgate_trace_anchored_replication.json"
            )
        else:
            relative = Path("reports/runs/craft_moe/sketchgate_smoke.json")
        args.output_json = ROOT / relative
    elif not args.output_json.is_absolute():
        args.output_json = ROOT / args.output_json
    args.output_json = args.output_json.resolve()
    if (ROOT / "reports").resolve() not in args.output_json.parents:
        raise ValueError("output-json must be inside reports/")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output_json}")
    return args


def numeric_summary(values: list[float] | torch.Tensor) -> dict[str, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("summary values must be non-empty and finite")
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def packed_bool_record(mask: torch.Tensor) -> dict[str, Any]:
    if mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean")
    array = mask.detach().cpu().numpy().reshape(-1)
    packed = np.packbits(array, bitorder="little")
    payload = packed.tobytes()
    return {
        "encoding": "base64(numpy.packbits(bitorder=little))",
        "logical_shape": list(mask.shape),
        "selected_count": int(mask.sum().item()),
        "sha256_packed": hashlib.sha256(payload).hexdigest(),
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


@torch.inference_mode()
def selected_matrix_outputs_and_sketches(
    moe: LoadedMoELayer,
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    probe_banks: dict[tuple[str, int], torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    if inputs.shape != (expert_ids.shape[0], HIDDEN_SIZE):
        raise ValueError("inputs must be [tokens, 2048]")
    if expert_ids.shape[1] != ACTIVE_EXPERTS or router_weights.shape != expert_ids.shape:
        raise ValueError("routes and weights must be [tokens, 6]")
    tokens, slots = expert_ids.shape
    variants = {
        name: torch.empty(tokens, slots, HIDDEN_SIZE, dtype=inputs.dtype)
        for name in (
            "bf16",
            "all_q3",
            "gate_only",
            "up_only",
            "down_only",
            "gate_up",
            "all_q4",
        )
    }
    sketch = torch.empty(
        len(PROBE_DISTRIBUTIONS),
        len(PROBE_SEEDS),
        len(RANKS),
        tokens,
        slots,
        dtype=torch.float32,
    )
    quantization_diagnostics: list[dict[str, Any]] = []
    touched = 0
    for expert_id, expert in enumerate(moe.experts):
        positions = (expert_ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            touched += 1
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            x = inputs[token_indices].to(moe.device)
            q3 = quantized_copy(expert, 3)
            q4 = quantized_copy(expert, 4)

            gate3 = F.linear(x, q3.gate)
            up3 = F.linear(x, q3.up)
            activation3 = F.silu(gate3) * up3
            gate4 = F.linear(x, q4.gate)
            up4 = F.linear(x, q4.up)
            activation4 = F.silu(gate4) * up4
            gate_only_activation = F.silu(gate4) * up3
            up_only_activation = F.silu(gate3) * up4
            local = {
                "bf16": moe.expert_forward(x, expert),
                "all_q3": F.linear(activation3, q3.down),
                "gate_only": F.linear(gate_only_activation, q3.down),
                "up_only": F.linear(up_only_activation, q3.down),
                "down_only": F.linear(activation3, q4.down),
                "gate_up": F.linear(activation4, q3.down),
                "all_q4": F.linear(activation4, q4.down),
            }
            for name, values in local.items():
                variants[name][token_indices, slot_indices] = values.cpu()

            residual_down = q4.down.float() - q3.down.float()
            local_router = router_weights[token_indices, slot_indices].to(moe.device)
            for distribution_index, distribution in enumerate(PROBE_DISTRIBUTIONS):
                for seed_index, seed in enumerate(PROBE_SEEDS):
                    scores, diagnostic = nested_quantized_sketch_scores(
                        activation3,
                        residual_down,
                        probe_banks[(distribution, seed)],
                        local_router,
                        RANKS,
                    )
                    for rank_index, rank in enumerate(RANKS):
                        sketch[
                            distribution_index,
                            seed_index,
                            rank_index,
                            token_indices,
                            slot_indices,
                        ] = scores[rank].cpu()
                    quantization_diagnostics.append(
                        {
                            "expert": expert_id,
                            "distribution": distribution,
                            "seed": seed,
                            **diagnostic,
                        }
                    )
            del q3, q4, residual_down, local, gate3, gate4, up3, up4
            del activation3, activation4, gate_only_activation, up_only_activation
        if expert_id % 8 == 7:
            print(f"sketchgate_experts={expert_id + 1}/64", flush=True)
    for name, output in variants.items():
        if not torch.isfinite(output.float()).all():
            raise RuntimeError(f"non-finite {name} selected output")
    if not torch.isfinite(sketch).all():
        raise RuntimeError("non-finite sketch score")
    return variants, sketch, {
        "experts_touched": touched,
        "records": quantization_diagnostics,
        "int8_syndrome_nrmse": numeric_summary(
            [row["syndrome_int8_nrmse"] for row in quantization_diagnostics]
        ),
        "int8_syndrome_maximum_absolute_error": numeric_summary(
            [
                row["syndrome_int8_maximum_absolute_error"]
                for row in quantization_diagnostics
            ]
        ),
        "total_zero_syndrome_rows": int(
            sum(row["scale_zero_rows"] for row in quantization_diagnostics)
        ),
    }


def combine_selected(
    selected: torch.Tensor, router_weights: torch.Tensor
) -> torch.Tensor:
    if selected.shape[:2] != router_weights.shape:
        raise ValueError("selected outputs and router weights do not align")
    return (
        selected.float() * router_weights.float().unsqueeze(-1)
    ).sum(dim=1).to(selected.dtype)


@torch.inference_mode()
def exact_kl_all_masks(
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
    base: torch.Tensor,
    upgraded: torch.Tensor,
    router_weights: torch.Tensor,
    masks: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    token_batch: int,
    post_attention: torch.Tensor | None = None,
    exact_shared: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-vocabulary teacher KL for every six-invocation upgrade mask."""

    tokens = teacher.shape[0]
    result = torch.empty(tokens, masks.shape[0], dtype=torch.float32)
    device = lm_head.device
    masks_device = masks.to(device)
    for start in range(0, tokens, token_batch):
        stop = min(start + token_batch, tokens)
        batch = stop - start
        base_batch = base[start:stop].to(device)
        upgraded_batch = upgraded[start:stop].to(device)
        mixed = torch.where(
            masks_device.view(1, masks.shape[0], masks.shape[1], 1),
            upgraded_batch.unsqueeze(1),
            base_batch.unsqueeze(1),
        )
        weights = router_weights[start:stop].to(device).view(
            batch, 1, masks.shape[1], 1
        )
        routed = (mixed.float() * weights).sum(dim=2).to(base_batch.dtype)
        if post_attention is not None and exact_shared is not None:
            candidates = post_attention[start:stop].to(device).unsqueeze(1) + (
                routed + exact_shared[start:stop].to(device).unsqueeze(1)
            )
        elif post_attention is None and exact_shared is None:
            candidates = (
                teacher[start:stop].to(device).float().unsqueeze(1)
                + routed.float()
                - natural_routed[start:stop].to(device).float().unsqueeze(1)
            ).to(teacher.dtype)
        else:
            raise ValueError("post_attention and exact_shared must be supplied together")
        logits = F.linear(
            rmsnorm(candidates.reshape(-1, HIDDEN_SIZE), norm_weight), lm_head
        ).float()
        candidate_log_probs = F.log_softmax(logits, dim=-1).view(
            batch, masks.shape[0], -1
        )
        teacher_lp = teacher_log_probs[start:stop].to(device)
        result[start:stop] = (
            teacher_lp.exp().unsqueeze(1)
            * (teacher_lp.unsqueeze(1) - candidate_log_probs)
        ).sum(dim=-1).clamp_min(0.0).cpu()
        if stop % 32 == 0 or stop == tokens:
            print(f"mask_kl_tokens={stop}/{tokens}", flush=True)
        del mixed, routed, candidates, logits, candidate_log_probs
    return result


def candidate_hidden_for_mask(
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
    base: torch.Tensor,
    upgraded: torch.Tensor,
    router_weights: torch.Tensor,
    mask: torch.Tensor,
    post_attention: torch.Tensor | None = None,
    exact_shared: torch.Tensor | None = None,
) -> torch.Tensor:
    selected = mix_selected_outputs(base, upgraded, mask)
    routed = combine_selected(selected, router_weights)
    if post_attention is not None and exact_shared is not None:
        return post_attention + (routed + exact_shared)
    if post_attention is not None or exact_shared is not None:
        raise ValueError("post_attention and exact_shared must be supplied together")
    return delta_patch(teacher, natural_routed, routed)


def evaluate_mask(
    mask: torch.Tensor,
    *,
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
    base: torch.Tensor,
    upgraded: torch.Tensor,
    router_weights: torch.Tensor,
    reference: Any,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    post_attention: torch.Tensor | None = None,
    exact_shared: torch.Tensor | None = None,
) -> dict[str, Any]:
    hidden = candidate_hidden_for_mask(
        teacher,
        natural_routed,
        base,
        upgraded,
        router_weights,
        mask,
        post_attention,
        exact_shared,
    )
    return evaluate_hidden(
        hidden,
        reference,
        norm_weight,
        lm_head,
        candidate_batch,
        bootstrap_resamples,
        bootstrap_seed,
    )


def exact_oracle_25(
    damage: torch.Tensor,
    masks: torch.Tensor,
    *,
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
    base: torch.Tensor,
    upgraded: torch.Tensor,
    router_weights: torch.Tensor,
    reference: Any,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    post_attention: torch.Tensor | None = None,
    exact_shared: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    best_damage, best_masks = best_mask_per_cardinality(damage, masks)
    curve, backpointers = discrete_rate_distortion(best_damage.double().numpy())
    budget = int(damage.shape[0] * ACTIVE_EXPERTS * UPGRADE_FRACTION)
    token_cost = recover_cost_schedule(backpointers, budget)
    token_indices = torch.arange(damage.shape[0])
    selected_indices = best_masks[
        token_indices, torch.from_numpy(token_cost).long()
    ]
    selected_mask = masks[selected_indices]
    direct_lookup = exact_schedule_mean_kl(selected_mask, damage)
    metrics = evaluate_mask(
        selected_mask,
        teacher=teacher,
        natural_routed=natural_routed,
        base=base,
        upgraded=upgraded,
        router_weights=router_weights,
        reference=reference,
        norm_weight=norm_weight,
        lm_head=lm_head,
        candidate_batch=candidate_batch,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        post_attention=post_attention,
        exact_shared=exact_shared,
    )
    direct_metric = metrics["aggregate"]["teacher_to_candidate_kl"]
    return {
        "budget": budget,
        "upgrade_fraction": budget / (damage.shape[0] * ACTIVE_EXPERTS),
        "dynamic_program_mean_kl": float(curve[budget] / damage.shape[0]),
        "lookup_mean_kl": direct_lookup,
        "direct_metric_mean_kl": direct_metric,
        "dp_vs_lookup_absolute_error": abs(
            float(curve[budget] / damage.shape[0]) - direct_lookup
        ),
        "lookup_vs_direct_metric_absolute_error": abs(direct_lookup - direct_metric),
        "per_token_upgrade_count": token_cost.tolist(),
        "selected_schedule": packed_bool_record(selected_mask),
        "full_model": metrics,
        "exact_cost_curve_total_kl": curve.tolist(),
    }, selected_mask


def schedule_record(
    mask: torch.Tensor,
    damage: torch.Tensor,
    high_damage: torch.Tensor,
    base_kl: float,
    oracle_kl: float,
) -> dict[str, Any]:
    mean_kl = exact_schedule_mean_kl(mask, damage)
    return {
        "upgrade_fraction": float(mask.double().mean().item()),
        "mean_kl_from_exact_mask_table": mean_kl,
        "oracle_recovery": oracle_recovery(base_kl, mean_kl, oracle_kl),
        "high_damage_false_negative_rate": false_negative_rate(mask, high_damage),
        "selected_schedule": packed_bool_record(mask),
    }


def router_rank_scores(router_weights: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(router_weights, dim=1, descending=True, stable=True)
    scores = torch.empty_like(router_weights, dtype=torch.float32)
    ranks = torch.arange(
        router_weights.shape[1], 0, -1, dtype=torch.float32
    ).view(1, -1).expand_as(scores)
    scores.scatter_(1, order, ranks)
    return scores


def random_mask(shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    return stable_top_fraction_mask(
        torch.from_numpy(generator.random(shape)), UPGRADE_FRACTION
    )


def historical_predictor_comparison() -> dict[str, Any]:
    files = {
        "ridge": ROOT / "reports/baseline/layer26_dynamic_precision_predictors.json",
        "progressive": ROOT
        / "reports/baseline/layer26_progressive_bitplane_predictor.json",
        "quadratic": ROOT / "reports/baseline/layer26_quadratic_mask_predictor.json",
    }
    loaded = {
        name: json.loads(path.read_text(encoding="utf-8"))["payload"]
        for name, path in files.items()
    }
    base_kl = float(loaded["progressive"]["all_3bit_test_kl"])
    oracle_rows = loaded["progressive"]["test_rate_distortion"][
        "direct_teacher_kl_oracle"
    ]
    oracle_kl = next(
        float(row["actual_kl_mean"])
        for row in oracle_rows
        if math.isclose(float(row["requested_fraction"]), UPGRADE_FRACTION)
    )

    def row_at(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        return next(
            row
            for row in rows
            if math.isclose(float(row[key]), UPGRADE_FRACTION)
        )

    ridge_row = row_at(
        loaded["ridge"]["test_predictor_rate_distortion"][
            "calibrated_ridge_available_features"
        ],
        "upgrade_fraction",
    )
    progressive_row = row_at(
        loaded["progressive"]["test_rate_distortion"]["expert_channel_diagonal"],
        "requested_fraction",
    )
    quadratic_row = row_at(
        loaded["quadratic"]["test_predicted_rate_distortion"],
        "requested_fraction",
    )
    methods = {
        "calibrated_ridge_available_features": float(
            ridge_row["teacher_to_candidate_kl"]
        ),
        "progressive_expert_channel_diagonal": float(
            progressive_row["actual_kl_mean"]
        ),
        "quadratic_mask_predictor": float(quadratic_row["actual_kl_mean"]),
    }
    return {
        "comparability": (
            "historical untouched WikiText test, 1024 tokens; not the H4 same-window "
            "gate and not used for configuration selection"
        ),
        "historical_all_q3_kl": base_kl,
        "historical_perfect_oracle25_kl": oracle_kl,
        "methods": {
            name: {
                "mean_kl": value,
                "oracle_recovery": oracle_recovery(base_kl, value, oracle_kl),
            }
            for name, value in methods.items()
        },
        "sources": {
            str(path.resolve()): {
                "sha256": sha256_file(path), "bytes": path.stat().st_size
            }
            for path in files.values()
        },
    }


@torch.inference_mode()
def measured_hardware_model(rank: int) -> dict[str, Any]:
    """Microbenchmark the explicit batch-1 out-of-core timing model."""

    weights_per_invocation = 3 * INTERMEDIATE_SIZE * HIDDEN_SIZE
    avoided_bytes = int(
        ACTIVE_EXPERTS * (1.0 - UPGRADE_FRACTION) * weights_per_invocation / 8
    )
    device = torch.device("cuda")
    host = torch.empty(avoided_bytes, dtype=torch.uint8, pin_memory=True)
    destination = torch.empty_like(host, device=device)
    activations = torch.randn(
        ACTIVE_EXPERTS, INTERMEDIATE_SIZE, device=device, dtype=torch.float16
    )
    syndrome = torch.randn(
        ACTIVE_EXPERTS,
        rank,
        INTERMEDIATE_SIZE,
        device=device,
        dtype=torch.float16,
    )

    for _ in range(20):
        destination.copy_(host, non_blocking=True)
        torch.bmm(syndrome, activations.unsqueeze(2)).squeeze(2).square().mean(dim=1)
    torch.cuda.synchronize()

    repetitions = 7
    inner = 200
    transfer_ms = []
    compute_ms = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            destination.copy_(host, non_blocking=True)
        stop.record()
        stop.synchronize()
        transfer_ms.append(float(start.elapsed_time(stop) / inner))

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            torch.bmm(syndrome, activations.unsqueeze(2)).squeeze(2).square().mean(dim=1)
        stop.record()
        stop.synchronize()
        compute_ms.append(float(start.elapsed_time(stop) / inner))

    transfer_median = float(np.median(transfer_ms))
    compute_median = float(np.median(compute_ms))
    ratio = compute_median / transfer_median if transfer_median > 0 else float("inf")
    del host, destination, activations, syndrome
    torch.cuda.empty_cache()
    return {
        "model": (
            "batch-1 out-of-core: six active experts; Q3 resident/transferred; "
            "only selected Q4 residual bitplanes transferred; syndrome already decoded "
            "to FP16 on device"
        ),
        "not_a_runtime_claim": True,
        "weights_per_expert_invocation": weights_per_invocation,
        "avoided_fourth_bit_bytes_per_token": avoided_bytes,
        "rank": rank,
        "repetitions": repetitions,
        "inner_iterations": inner,
        "transfer_milliseconds_raw": transfer_ms,
        "sketch_compute_milliseconds_raw": compute_ms,
        "transfer_milliseconds_median": transfer_median,
        "sketch_compute_milliseconds_median": compute_median,
        "compute_over_avoided_transfer_time": ratio,
        "passes_lt_0_10": ratio < 0.10,
    }


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("SketchGate exact attribution requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    model_dir = ROOT / "models/deepseek-v2-lite"
    component_path = ROOT / COMPONENT_RELATIVE
    preregistration_path = ROOT / (
        REPLICATION_PREREGISTRATION
        if args.stage == "replication"
        else PREREGISTRATION
    )
    split_window_offset = (
        REPLICATION_SPLIT_OFFSET if args.stage == "replication" else 0
    )
    for path in (model_dir, component_path, preregistration_path):
        if not path.exists():
            raise FileNotFoundError(path)

    phase = time.perf_counter()
    input_hashes = {
        str(component_path.resolve()): sha256_file(component_path),
        str(preregistration_path.resolve()): sha256_file(preregistration_path),
        str((model_dir / "config.json").resolve()): sha256_file(
            model_dir / "config.json"
        ),
        str((model_dir / "model.safetensors.index.json").resolve()): sha256_file(
            model_dir / "model.safetensors.index.json"
        ),
    }
    repository = git_state()
    initial_hardware = hardware_state()
    timings["input_sha256_and_environment_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    component_all = load_file(component_path, device="cpu")
    trace_indices: dict[str, list[int]] = {}
    indices: list[int] = []
    for split in args.splits:
        base_index = (
            0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        ) + split_window_offset
        chosen = list(range(base_index, base_index + args.tokens_per_split))
        trace_indices[split] = chosen
        indices.extend(chosen)
    index = torch.tensor(indices, dtype=torch.long)
    component_keys = (
        "teacher",
        "moe_input",
        "router_ids",
        "router_weights",
        "post_attention",
        "shared",
        "selected_quant3",
        "selected_quant4",
    )
    components = {
        key: component_all[key].index_select(0, index) for key in component_keys
    }
    del component_all
    token_ids = {
        split: corpus_tokens(
            model_dir, split, split_window_offset + args.tokens_per_split
        )[split_window_offset:]
        for split in args.splits
    }
    timings["load_fixed_inputs_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    moe = load_moe_layer(model_dir, LAYER, device)
    route_ids, route_weights = moe.route(components["moe_input"].to(device))
    route_control = {
        "slot_order_ids_exact": bool(
            torch.equal(route_ids.cpu(), components["router_ids"].long())
        ),
        "set_ids_exact": bool(
            torch.equal(
                route_ids.cpu().sort(dim=1).values,
                components["router_ids"].long().sort(dim=1).values,
            )
        ),
        "router_weight_maximum_absolute_error": float(
            (
                route_weights.cpu().float()
                - components["router_weights"].float()
            ).abs().max().item()
        ),
    }
    probe_banks = {
        (distribution, seed): probe_bank(
            distribution, seed, max(RANKS), HIDDEN_SIZE
        ).to(device)
        for distribution in PROBE_DISTRIBUTIONS
        for seed in PROBE_SEEDS
    }
    variants, sketch_scores, sketch_quantization = selected_matrix_outputs_and_sketches(
        moe,
        components["moe_input"],
        components["router_ids"].long(),
        components["router_weights"].float(),
        probe_banks,
    )
    component_regression = {
        "all_q3": regression_summary(
            components["selected_quant3"], variants["all_q3"]
        ),
        "all_q4": regression_summary(
            components["selected_quant4"], variants["all_q4"]
        ),
        "all_q3_bit_exact": bool(
            torch.equal(components["selected_quant3"], variants["all_q3"])
        ),
        "all_q4_bit_exact": bool(
            torch.equal(components["selected_quant4"], variants["all_q4"])
        ),
    }
    component_regression["fixed_tolerance"] = {
        "nrmse_le": COMPONENT_NRMSE_TOLERANCE,
        "maximum_absolute_error_le": COMPONENT_MAX_ABS_TOLERANCE,
    }
    component_regression["all_q3_passes_fixed_tolerance"] = bool(
        component_regression["all_q3"]["nrmse"] <= COMPONENT_NRMSE_TOLERANCE
        and component_regression["all_q3"]["maximum_absolute_error"]
        <= COMPONENT_MAX_ABS_TOLERANCE
    )
    component_regression["all_q4_passes_fixed_tolerance"] = bool(
        component_regression["all_q4"]["nrmse"] <= COMPONENT_NRMSE_TOLERANCE
        and component_regression["all_q4"]["maximum_absolute_error"]
        <= COMPONENT_MAX_ABS_TOLERANCE
    )
    trace_anchor = {
        "enabled": args.stage == "replication",
        "selected_quant3_source": "recomputed output",
        "selected_quant4_source": "recomputed output",
        "candidate_hidden": "official teacher delta patch",
    }
    if args.stage == "replication":
        # The replication is intentionally anchored to the immutable component
        # outputs rather than relaxing a failed recomputation tolerance.
        variants["all_q3"] = components["selected_quant3"]
        variants["all_q4"] = components["selected_quant4"]
        trace_anchor = {
            "enabled": True,
            "selected_quant3_source": "immutable component trace, used bit exactly",
            "selected_quant4_source": "immutable component trace, used bit exactly",
            "candidate_hidden": (
                "BF16(post_attention + BF16(candidate_routed + exact_shared))"
            ),
            "stored_sources_used_bit_exact": True,
            "recomputed_output_regression_is_diagnostic_only": True,
        }
    natural_routed = combine_selected(
        variants["bf16"], components["router_weights"]
    )
    exact_delta_control = torch.equal(
        delta_patch(components["teacher"], natural_routed, natural_routed),
        components["teacher"],
    )
    del moe, probe_banks, route_ids, route_weights
    gc.collect()
    torch.cuda.empty_cache()
    timings["matrix_variants_and_sketch_scores_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    timings["load_final_projection_seconds"] = time.perf_counter() - phase

    split_slices: dict[str, slice] = {}
    references: dict[str, Any] = {}
    offset = 0
    for split in args.splits:
        selected = slice(offset, offset + args.tokens_per_split)
        split_slices[split] = selected
        references[split] = make_teacher_reference(
            components["teacher"][selected],
            token_ids[split],
            sequence_blocks(args.tokens_per_split),
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        offset += args.tokens_per_split

    masks = binary_upgrade_masks(ACTIVE_EXPERTS)
    family_variant = {
        "gate_only": "gate_only",
        "up_only": "up_only",
        "down_only": "down_only",
        "gate_up": "gate_up",
        "all_matrices": "all_q4",
    }
    results: dict[str, Any] = {}
    masks_by_split: dict[str, dict[str, torch.Tensor]] = {}
    phase = time.perf_counter()
    for split_index, split in enumerate(args.splits):
        selected = split_slices[split]
        teacher = components["teacher"][selected]
        weights = components["router_weights"][selected]
        natural = natural_routed[selected]
        base = variants["all_q3"][selected]
        reference = references[split]
        post_attention = (
            components["post_attention"][selected]
            if args.stage == "replication"
            else None
        )
        exact_shared = (
            components["shared"][selected]
            if args.stage == "replication"
            else None
        )
        exact_control_hidden = delta_patch(teacher, natural, natural)
        exact_control_metrics = evaluate_hidden(
            exact_control_hidden,
            reference,
            norm_weight,
            lm_head,
            args.candidate_batch,
            args.bootstrap_resamples,
            args.seed + split_index,
        )
        if (
            not torch.equal(exact_control_hidden, teacher)
            or max(exact_control_metrics["raw"]["teacher_to_candidate_kl"]) != 0.0
            or exact_control_metrics["aggregate"]["cross_entropy_delta"] != 0.0
            or not all(exact_control_metrics["raw"]["top1_agreement"])
        ):
            raise RuntimeError(f"official exact BF16 delta control failed on {split}")

        all_q3_mask = torch.zeros(
            args.tokens_per_split, ACTIVE_EXPERTS, dtype=torch.bool
        )
        all_q4_mask = torch.ones_like(all_q3_mask)
        all_q3_metrics = evaluate_mask(
            all_q3_mask,
            teacher=teacher,
            natural_routed=natural,
            base=base,
            upgraded=variants["all_q4"][selected],
            router_weights=weights,
            reference=reference,
            norm_weight=norm_weight,
            lm_head=lm_head,
            candidate_batch=args.candidate_batch,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed + split_index,
            post_attention=post_attention,
            exact_shared=exact_shared,
        )
        all_q4_metrics = evaluate_mask(
            all_q4_mask,
            teacher=teacher,
            natural_routed=natural,
            base=base,
            upgraded=variants["all_q4"][selected],
            router_weights=weights,
            reference=reference,
            norm_weight=norm_weight,
            lm_head=lm_head,
            candidate_batch=args.candidate_batch,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed + split_index,
            post_attention=post_attention,
            exact_shared=exact_shared,
        )
        split_result: dict[str, Any] = {
            "teacher_reference": {
                "token_ids": token_ids[split].tolist(),
                "true_token_nll": nullable(reference.true_token_nll),
                "sequence_blocks": [list(block) for block in reference.blocks],
            },
            "controls": {
                "official_bf16_delta": exact_control_metrics,
                "all_q3": all_q3_metrics,
                "all_q4": all_q4_metrics,
            },
            "attribution": {},
        }
        masks_by_split[split] = {}
        for family in FAMILIES:
            print(f"attribution[{split}][{family}]", flush=True)
            upgraded = variants[family_variant[family]][selected]
            damage = exact_kl_all_masks(
                teacher,
                natural,
                base,
                upgraded,
                weights,
                masks,
                reference.log_probs,
                norm_weight,
                lm_head,
                args.mask_token_batch,
                post_attention,
                exact_shared,
            )
            oracle, oracle_mask = exact_oracle_25(
                damage,
                masks,
                teacher=teacher,
                natural_routed=natural,
                base=base,
                upgraded=upgraded,
                router_weights=weights,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
                post_attention=post_attention,
                exact_shared=exact_shared,
            )
            uniform_metrics = evaluate_mask(
                all_q4_mask,
                teacher=teacher,
                natural_routed=natural,
                base=base,
                upgraded=upgraded,
                router_weights=weights,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
                post_attention=post_attention,
                exact_shared=exact_shared,
            )
            split_result["attribution"][family] = {
                "uniform_upgrade": uniform_metrics,
                "perfect_oracle25": oracle,
                "exact_mask_kl": damage.tolist(),
            }
            masks_by_split[split][f"oracle_{family}"] = oracle_mask

        base_kl = all_q3_metrics["aggregate"]["teacher_to_candidate_kl"]
        all_oracle_kl = split_result["attribution"]["all_matrices"][
            "perfect_oracle25"
        ]["lookup_mean_kl"]
        denominator = base_kl - all_oracle_kl
        attribution_gain = {}
        for family in FAMILIES:
            family_kl = split_result["attribution"][family]["perfect_oracle25"][
                "lookup_mean_kl"
            ]
            attribution_gain[family] = {
                "oracle25_mean_kl": family_kl,
                "absolute_kl_improvement_from_all_q3": base_kl - family_kl,
                "fraction_of_all_matrix_oracle_improvement": (
                    (base_kl - family_kl) / denominator if denominator > 0 else None
                ),
            }
        split_result["attribution_gain"] = attribution_gain
        split_result["gate_up_dominates_down_only"] = bool(
            attribution_gain["gate_up"]["absolute_kl_improvement_from_all_q3"]
            > attribution_gain["down_only"]["absolute_kl_improvement_from_all_q3"]
        )

        all_damage = torch.tensor(
            split_result["attribution"]["all_matrices"]["exact_mask_kl"],
            dtype=torch.float32,
        )
        singleton_indices = torch.tensor(
            [1 << slot for slot in range(ACTIVE_EXPERTS)], dtype=torch.long
        )
        singleton_benefit = (
            all_damage[:, 0].unsqueeze(1) - all_damage[:, singleton_indices]
        )
        high_damage, high_description = high_damage_mask(
            singleton_benefit, HIGH_DAMAGE_FRACTION
        )
        split_result["high_damage"] = {
            **high_description,
            "benefit_summary": numeric_summary(singleton_benefit.reshape(-1)),
            "singleton_kl_benefit": singleton_benefit.tolist(),
            "mask": packed_bool_record(high_damage),
        }
        masks_by_split[split]["high_damage"] = high_damage

        schedule_records: list[dict[str, Any]] = []
        for distribution_index, distribution in enumerate(PROBE_DISTRIBUTIONS):
            for seed_index, probe_seed in enumerate(PROBE_SEEDS):
                for rank_index, rank in enumerate(RANKS):
                    score = sketch_scores[
                        distribution_index, seed_index, rank_index, selected
                    ]
                    chosen_mask = stable_top_fraction_mask(score, UPGRADE_FRACTION)
                    record = {
                        "distribution": distribution,
                        "seed": probe_seed,
                        "rank": rank,
                        **schedule_record(
                            chosen_mask,
                            all_damage,
                            high_damage,
                            base_kl,
                            all_oracle_kl,
                        ),
                    }
                    schedule_records.append(record)
        split_result["sketchgate_configurations"] = schedule_records

        baselines: dict[str, Any] = {}
        baseline_masks: dict[str, torch.Tensor] = {
            "router_weight": stable_top_fraction_mask(weights, UPGRADE_FRACTION),
            "router_rank": stable_top_fraction_mask(
                router_rank_scores(weights), UPGRADE_FRACTION
            ),
            "q4_delta_energy_non_deployable": stable_top_fraction_mask(
                (variants["all_q4"][selected].float() - base.float())
                .square()
                .sum(dim=-1)
                .mul(weights.float().square()),
                UPGRADE_FRACTION,
            ),
        }
        for name, baseline_mask in baseline_masks.items():
            baselines[name] = schedule_record(
                baseline_mask, all_damage, high_damage, base_kl, all_oracle_kl
            )
            masks_by_split[split][name] = baseline_mask
        random_records = []
        for random_seed in PROBE_SEEDS:
            selected_random = random_mask(
                (args.tokens_per_split, ACTIVE_EXPERTS), random_seed
            )
            random_records.append(
                {
                    "seed": random_seed,
                    **schedule_record(
                        selected_random,
                        all_damage,
                        high_damage,
                        base_kl,
                        all_oracle_kl,
                    ),
                }
            )
            if random_seed == PRIMARY_SEED:
                masks_by_split[split]["random_primary"] = selected_random
        baselines["random"] = random_records
        split_result["same_window_baselines"] = baselines
        results[split] = split_result
    timings["exact_attribution_and_schedule_lookup_seconds"] = (
        time.perf_counter() - phase
    )

    phase = time.perf_counter()
    validation_selection = choose_validation_configuration(
        results["validation"]["sketchgate_configurations"]
    )
    selected_distribution = validation_selection["selected_distribution"]
    selected_rank = validation_selection["selected_rank"]
    selected_distribution_index = PROBE_DISTRIBUTIONS.index(selected_distribution)
    selected_rank_index = RANKS.index(selected_rank)
    primary_seed_index = PROBE_SEEDS.index(PRIMARY_SEED)
    for split_index, split in enumerate(args.splits):
        selected = split_slices[split]
        score = sketch_scores[
            selected_distribution_index,
            primary_seed_index,
            selected_rank_index,
            selected,
        ]
        primary_mask = stable_top_fraction_mask(score, UPGRADE_FRACTION)
        masks_by_split[split]["sketchgate_primary"] = primary_mask
        detailed_masks = {
            "sketchgate_primary": primary_mask,
            "router_weight": masks_by_split[split]["router_weight"],
            "router_rank": masks_by_split[split]["router_rank"],
            "q4_delta_energy_non_deployable": masks_by_split[split][
                "q4_delta_energy_non_deployable"
            ],
            "random_primary": masks_by_split[split]["random_primary"],
        }
        exact_damage = torch.tensor(
            results[split]["attribution"]["all_matrices"]["exact_mask_kl"],
            dtype=torch.float32,
        )
        detailed: dict[str, Any] = {}
        detail_post_attention = (
            components["post_attention"][selected]
            if args.stage == "replication"
            else None
        )
        detail_exact_shared = (
            components["shared"][selected]
            if args.stage == "replication"
            else None
        )
        for method_index, (name, method_mask) in enumerate(detailed_masks.items()):
            metrics = evaluate_mask(
                method_mask,
                teacher=components["teacher"][selected],
                natural_routed=natural_routed[selected],
                base=variants["all_q3"][selected],
                upgraded=variants["all_q4"][selected],
                router_weights=components["router_weights"][selected],
                reference=references[split],
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index + method_index,
                post_attention=detail_post_attention,
                exact_shared=detail_exact_shared,
            )
            lookup_kl = exact_schedule_mean_kl(method_mask, exact_damage)
            direct_kl = metrics["aggregate"]["teacher_to_candidate_kl"]
            detailed[name] = {
                "exact_lookup_mean_kl": lookup_kl,
                "lookup_vs_direct_absolute_error": abs(lookup_kl - direct_kl),
                "full_model": metrics,
                "schedule": packed_bool_record(method_mask),
            }
        results[split]["selected_and_baseline_full_metrics"] = detailed
    timings["selected_schedule_full_metrics_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    metadata_accounting = sketch_metadata_accounting(selected_rank)
    hardware_model = measured_hardware_model(selected_rank)
    historical = historical_predictor_comparison()
    timings["hardware_model_and_historical_load_seconds"] = time.perf_counter() - phase

    if args.stage == "smoke":
        verdict = "smoke_passed_not_adjudicated"
        gates: dict[str, Any] = {
            "adjudicated": False,
            "reason": "fixed validation and test windows are required",
        }
    else:
        down_by_split = {
            split: results[split]["attribution_gain"]["down_only"][
                "fraction_of_all_matrix_oracle_improvement"
            ]
            for split in ("validation", "test")
        }
        down_gate = all(value is not None and value >= 0.70 for value in down_by_split.values())
        selected_records_by_split = {
            split: [
                row
                for row in results[split]["sketchgate_configurations"]
                if row["distribution"] == selected_distribution
                and row["rank"] == selected_rank
            ]
            for split in ("validation", "test")
        }
        primary_by_split = {
            split: next(
                row
                for row in selected_records_by_split[split]
                if row["seed"] == PRIMARY_SEED
            )
            for split in ("validation", "test")
        }
        primary_gate = all(
            row["oracle_recovery"] >= 0.80
            and row["high_damage_false_negative_rate"] <= 0.01
            for row in primary_by_split.values()
        )
        stability_by_split = {
            split: {
                "minimum_recovery": min(
                    row["oracle_recovery"]
                    for row in selected_records_by_split[split]
                ),
                "maximum_false_negative_rate": max(
                    row["high_damage_false_negative_rate"]
                    for row in selected_records_by_split[split]
                ),
                "all_five_seeds_pass": all(
                    row["oracle_recovery"] >= 0.80
                    and row["high_damage_false_negative_rate"] <= 0.01
                    for row in selected_records_by_split[split]
                ),
            }
            for split in ("validation", "test")
        }
        stability_gate = all(
            row["all_five_seeds_pass"] for row in stability_by_split.values()
        )
        if args.stage == "replication":
            controls_gate = bool(
                exact_delta_control
                and route_control["slot_order_ids_exact"]
                and trace_anchor.get("stored_sources_used_bit_exact", False)
            )
            component_recompute_is_gate = False
        else:
            controls_gate = bool(
                exact_delta_control
                and route_control["slot_order_ids_exact"]
                and component_regression["all_q3_passes_fixed_tolerance"]
                and component_regression["all_q4_passes_fixed_tolerance"]
            )
            component_recompute_is_gate = True
        gates = {
            "adjudicated": True,
            "down_attribution_ge_0_70_both_splits": down_gate,
            "down_attribution_fraction_by_split": down_by_split,
            "validation_selection": validation_selection,
            "primary_seed": PRIMARY_SEED,
            "primary_seed_recovery_ge_0_80_and_fn_le_0_01_both_splits": primary_gate,
            "primary_by_split": primary_by_split,
            "all_five_seeds_stable_both_splits": stability_gate,
            "stability_by_split": stability_by_split,
            "metadata_lt_0_1_effective_bit": metadata_accounting[
                "passes_lt_0_1_bit"
            ],
            "hardware_model_compute_lt_0_10_avoided_transfer": hardware_model[
                "passes_lt_0_10"
            ],
            "exact_controls_pass": controls_gate,
            "component_recompute_is_gate": component_recompute_is_gate,
        }
        positive = all(
            (
                down_gate,
                primary_gate,
                stability_gate,
                metadata_accounting["passes_lt_0_1_bit"],
                hardware_model["passes_lt_0_10"],
                controls_gate,
            )
        )
        gate_up_dominates_validation = results["validation"][
            "gate_up_dominates_down_only"
        ]
        sketch_misses = not primary_gate or not stability_gate
        if positive:
            verdict = "layer26_positive_opens_spread_preregistration"
        elif gate_up_dominates_validation and sketch_misses:
            verdict = "falsified_gate_up_dominates_and_down_sketch_misses"
        else:
            verdict = "falsified_layer26_gate_no_spread"

    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report = {
        "schema_version": 1,
        "kind": "craft_moe_h4_sketchgate_layer26",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H4_SKETCHGATE",
        "verdict": verdict,
        "preregistration": str(preregistration_path.resolve()),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "local_path": str(model_dir.resolve()),
            "layer": LAYER,
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "window": {
                split: (
                    f"split-relative trace tokens {split_window_offset}.."
                    f"{split_window_offset + args.tokens_per_split - 1}"
                )
                for split in args.splits
            },
            "block_size": BLOCK_SIZE,
        },
        "configuration": {
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
            "upgrade_fraction": UPGRADE_FRACTION,
            "high_damage_fraction": HIGH_DAMAGE_FRACTION,
            "probe_distributions": list(PROBE_DISTRIBUTIONS),
            "probe_seeds": list(PROBE_SEEDS),
            "primary_seed": PRIMARY_SEED,
            "ranks": list(RANKS),
            "families": list(FAMILIES),
            "candidate_batch": args.candidate_batch,
            "mask_token_batch": args.mask_token_batch,
            "bootstrap_resamples": args.bootstrap_resamples,
            "router_weights_renormalized": False,
            "sketch_score": "p_router^2 * mean((int8_dequant(DeltaD^T z)^T a_q3)^2)",
            "probe_nesting": "all r use the first r rows of a fixed 64-row bank",
            "calibration": "none",
            "counterfactual_patch": (
                trace_anchor["candidate_hidden"]
            ),
            "trace_anchor": trace_anchor,
        },
        "selection": validation_selection,
        "gates": gates,
        "metadata_accounting": metadata_accounting,
        "hardware_model": hardware_model,
        "historical_predictor_comparison": historical,
        "results": results,
        "controls": {
            "route_recomputation": route_control,
            "component_recomputation": component_regression,
            "official_teacher_exact_delta_bit_exact": exact_delta_control,
            "trace_anchor": trace_anchor,
            "sketch_int8_quantization": sketch_quantization,
        },
        "raw_sketch_scores": {
            "shape": list(sketch_scores.shape),
            "axis_order": "distribution, seed, rank, concatenated_split_token, expert_slot",
            "values": sketch_scores.tolist(),
        },
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "safetensors": safetensors.__version__,
                "tokenizers": tokenizers.__version__,
                "pyarrow": pyarrow.__version__,
                "psutil": psutil.__version__,
            },
            "inputs": {
                "sha256": input_hashes,
                "component_metadata": metadata(component_path),
                "trace_indices": trace_indices,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "late-layer exploratory screen; stability across layers 13/23 and OOD is not tested unless this layer-26 gate passes",
            "the 256-token windows are exploratory and are not a 1024-token candidate validation or fresh confirmation",
            "full-vocabulary exact mask KL is used for adjudication, but no task accuracy or autoregressive rollout is measured",
            "the hardware result is a microbenchmarked batch-1 transfer/compute model, not a packed end-to-end runtime",
            "historical learned-predictor comparisons use a separate 1024-token test window and are not same-window gates",
            "int8 syndrome rows are recomputed offline from Q3/Q4 weights; lifecycle and cache management are not implemented",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    print(f"gates={json.dumps(gates, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
