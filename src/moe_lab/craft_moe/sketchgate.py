from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def probe_bank(
    distribution: str,
    seed: int,
    rows: int,
    hidden_size: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create one deterministic CPU probe bank.

    A distribution-specific seed offset prevents Gaussian and Rademacher banks
    from accidentally sharing the same pseudo-random stream. Prefixes are used
    for all smaller sketch dimensions, so comparisons across ``r`` are nested.
    """

    if rows < 1 or hidden_size < 1:
        raise ValueError("rows and hidden_size must be positive")
    offsets = {"gaussian": 0, "rademacher": 1_000_003}
    if distribution not in offsets:
        raise ValueError("distribution must be gaussian or rademacher")
    generator = torch.Generator(device="cpu").manual_seed(seed + offsets[distribution])
    if distribution == "gaussian":
        # Row-wise draws keep the first r rows identical even when callers ask
        # for a shorter bank; some normal kernels consume shape-dependent tails.
        return torch.stack(
            [
                torch.randn(hidden_size, generator=generator, dtype=dtype)
                for _ in range(rows)
            ]
        )
    values = torch.stack(
        [
            torch.randint(
                0, 2, (hidden_size,), generator=generator, dtype=torch.int8
            )
            for _ in range(rows)
        ]
    )
    return values.to(dtype).mul_(2).sub_(1)


def quantize_rows_int8(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8 quantization with a finite zero-row control."""

    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("rows must be a non-empty matrix")
    maximum = rows.float().abs().amax(dim=1)
    scale = maximum.div(127.0)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = (
        rows.float().div(safe_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    )
    scale = torch.where(scale > 0, scale, torch.zeros_like(scale)).to(torch.float16)
    return quantized, scale


def dequantize_rows_int8(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if quantized.ndim != 2 or quantized.dtype is not torch.int8:
        raise ValueError("quantized must be an int8 matrix")
    if scale.ndim != 1 or scale.shape[0] != quantized.shape[0]:
        raise ValueError("scale must contain one value per row")
    return quantized.float() * scale.float().unsqueeze(1)


def nested_quantized_sketch_scores(
    activations: torch.Tensor,
    residual_down: torch.Tensor,
    probes: torch.Tensor,
    router_weights: torch.Tensor,
    ranks: Iterable[int],
) -> tuple[dict[int, torch.Tensor], dict[str, float]]:
    """Score invocations from int8-quantized residual-syndrome rows.

    ``residual_down`` has output-by-intermediate orientation and each probe is
    in output space. Returned scores include squared natural router weight.
    """

    ranks = tuple(int(rank) for rank in ranks)
    if activations.ndim != 2 or residual_down.ndim != 2 or probes.ndim != 2:
        raise ValueError("activations, residual_down, and probes must be matrices")
    if residual_down.shape[1] != activations.shape[1]:
        raise ValueError("residual_down input dimension must match activations")
    if probes.shape[1] != residual_down.shape[0]:
        raise ValueError("probe dimension must match residual_down output")
    if router_weights.shape != (activations.shape[0],):
        raise ValueError("router_weights must have one value per activation")
    if not ranks or min(ranks) < 1 or max(ranks) > probes.shape[0]:
        raise ValueError("ranks must be non-empty prefixes of the probe bank")

    syndrome = probes.float() @ residual_down.float()
    quantized, scale = quantize_rows_int8(syndrome)
    reconstructed = dequantize_rows_int8(quantized, scale)
    projections = activations.float() @ reconstructed.T
    cumulative = projections.square().cumsum(dim=1)
    router_square = router_weights.float().square()
    scores = {
        rank: cumulative[:, rank - 1].div(float(rank)).mul(router_square)
        for rank in ranks
    }
    error = reconstructed - syndrome
    reference_rms = syndrome.square().mean().sqrt().clamp_min(1e-30)
    diagnostics = {
        "syndrome_int8_nrmse": float(
            (error.square().mean().sqrt() / reference_rms).item()
        ),
        "syndrome_int8_maximum_absolute_error": float(error.abs().max().item()),
        "scale_zero_rows": int(scale.eq(0).sum().item()),
    }
    return scores, diagnostics


def stable_top_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Select an exact global fraction with stable flattened tie-breaking."""

    if scores.numel() == 0 or not torch.isfinite(scores.float()).all():
        raise ValueError("scores must be non-empty and finite")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    count = int(scores.numel() * fraction)
    flattened = scores.detach().cpu().double().reshape(-1).numpy()
    order = np.argsort(-flattened, kind="stable")
    selected = np.zeros(flattened.shape[0], dtype=np.bool_)
    selected[order[:count]] = True
    return torch.from_numpy(selected.reshape(tuple(scores.shape)))


def mix_selected_outputs(
    base: torch.Tensor, upgraded: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if base.shape != upgraded.shape:
        raise ValueError("base and upgraded outputs must have equal shape")
    if mask.shape != base.shape[:-1] or mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean and match all non-hidden dimensions")
    return torch.where(mask.unsqueeze(-1), upgraded, base)


def delta_patch(
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
    candidate_routed: torch.Tensor,
) -> torch.Tensor:
    if teacher.shape != natural_routed.shape or teacher.shape != candidate_routed.shape:
        raise ValueError("all hidden-state tensors must have equal shape")
    return (
        teacher.float() + candidate_routed.float() - natural_routed.float()
    ).to(teacher.dtype)


def mask_indices(mask: torch.Tensor) -> torch.Tensor:
    """Convert a boolean [tokens, slots] schedule to integer bit patterns."""

    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean [tokens, slots]")
    bits = (1 << torch.arange(mask.shape[1], dtype=torch.long)).view(1, -1)
    return (mask.long() * bits).sum(dim=1)


def exact_schedule_mean_kl(mask: torch.Tensor, damage: torch.Tensor) -> float:
    if damage.ndim != 2 or damage.shape[0] != mask.shape[0]:
        raise ValueError("damage must be [tokens, all masks]")
    indices = mask_indices(mask)
    if int(indices.max().item()) >= damage.shape[1]:
        raise ValueError("damage does not contain all requested mask indices")
    values = damage[torch.arange(mask.shape[0]), indices]
    return float(values.double().mean().item())


def high_damage_mask(
    singleton_benefit: torch.Tensor, fraction: float = 0.10
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the highest fixed fraction of positive singleton KL benefits."""

    if singleton_benefit.ndim != 2 or singleton_benefit.numel() == 0:
        raise ValueError("singleton_benefit must be [tokens, slots]")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    flat = singleton_benefit.detach().cpu().double().reshape(-1).numpy()
    positive = np.flatnonzero(flat > 0)
    nominal_count = max(1, int(np.ceil(flat.size * fraction)))
    count = min(nominal_count, positive.size)
    selected = np.zeros(flat.size, dtype=np.bool_)
    if count:
        order = positive[np.argsort(-flat[positive], kind="stable")]
        selected[order[:count]] = True
    return torch.from_numpy(selected.reshape(tuple(singleton_benefit.shape))), {
        "definition_fraction": fraction,
        "total_invocations": int(flat.size),
        "positive_benefit_invocations": int(positive.size),
        "nominal_top_count": int(nominal_count),
        "selected_high_damage_count": int(count),
        "used_all_positive_because_fewer_than_nominal": bool(positive.size < nominal_count),
    }


def false_negative_rate(selected: torch.Tensor, high_damage: torch.Tensor) -> float:
    if selected.shape != high_damage.shape or selected.dtype is not torch.bool:
        raise ValueError("selected and high_damage must be equally shaped boolean masks")
    if high_damage.dtype is not torch.bool:
        raise ValueError("selected and high_damage must be equally shaped boolean masks")
    positives = int(high_damage.sum().item())
    if positives == 0:
        return 0.0
    missed = high_damage & ~selected
    return float(missed.sum().item() / positives)


def oracle_recovery(base_kl: float, method_kl: float, oracle_kl: float) -> float:
    denominator = base_kl - oracle_kl
    if denominator <= 0:
        raise ValueError("oracle must improve on the base KL")
    return float((base_kl - method_kl) / denominator)


def sketch_metadata_accounting(
    rank: int,
    *,
    experts: int = 64,
    intermediate_size: int = 1408,
    hidden_size: int = 2048,
    matrices_per_expert: int = 3,
    value_bytes: int = 1,
    scale_bytes: int = 2,
) -> dict[str, Any]:
    if min(rank, experts, intermediate_size, hidden_size, matrices_per_expert) < 1:
        raise ValueError("all dimensions must be positive")
    syndrome_values = experts * rank * intermediate_size
    scales = experts * rank
    metadata_bytes = syndrome_values * value_bytes + scales * scale_bytes
    original_weights = experts * matrices_per_expert * intermediate_size * hidden_size
    effective_bits = metadata_bytes * 8.0 / original_weights
    return {
        "rank": rank,
        "syndrome_values": syndrome_values,
        "scale_values": scales,
        "metadata_bytes": metadata_bytes,
        "original_routed_expert_weights": original_weights,
        "effective_bits_per_original_weight": effective_bits,
        "passes_lt_0_1_bit": effective_bits < 0.1,
        "representation": "int8 syndrome values plus FP16 per-row scales",
    }


def choose_validation_configuration(
    records: list[dict[str, Any]],
    *,
    recovery_gate: float = 0.80,
    false_negative_gate: float = 0.01,
) -> dict[str, Any]:
    """Choose distribution/r from validation only; never choose a seed."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["distribution"]), int(record["rank"]))
        grouped.setdefault(key, []).append(record)
    if not grouped:
        raise ValueError("at least one validation record is required")
    summaries = []
    for (distribution, rank), rows in grouped.items():
        recoveries = np.asarray([float(row["oracle_recovery"]) for row in rows])
        false_negatives = np.asarray([float(row["high_damage_false_negative_rate"]) for row in rows])
        summaries.append(
            {
                "distribution": distribution,
                "rank": rank,
                "seeds": sorted(int(row["seed"]) for row in rows),
                "seed_count": len(rows),
                "minimum_recovery": float(recoveries.min()),
                "median_recovery": float(np.median(recoveries)),
                "maximum_false_negative_rate": float(false_negatives.max()),
                "median_false_negative_rate": float(np.median(false_negatives)),
                "all_seeds_pass": bool(
                    np.all(recoveries >= recovery_gate)
                    and np.all(false_negatives <= false_negative_gate)
                ),
            }
        )
    qualifying = [summary for summary in summaries if summary["all_seeds_pass"]]
    distribution_priority = {"rademacher": 0, "gaussian": 1}
    if qualifying:
        selected = min(
            qualifying,
            key=lambda row: (
                row["rank"],
                distribution_priority.get(row["distribution"], 99),
            ),
        )
        rule = "smallest qualifying rank; Rademacher wins an equal-rank tie"
        qualified = True
    else:
        selected = min(
            summaries,
            key=lambda row: (
                -row["median_recovery"],
                row["maximum_false_negative_rate"],
                row["rank"],
                distribution_priority.get(row["distribution"], 99),
            ),
        )
        rule = (
            "diagnostic fallback: highest median recovery, lowest maximum FN, "
            "smaller rank, then Rademacher"
        )
        qualified = False
    return {
        "qualified_on_validation": qualified,
        "selection_rule": rule,
        "selected_distribution": selected["distribution"],
        "selected_rank": selected["rank"],
        "selected_summary": selected,
        "configuration_summaries": sorted(
            summaries,
            key=lambda row: (
                row["rank"], distribution_priority.get(row["distribution"], 99)
            ),
        ),
    }
