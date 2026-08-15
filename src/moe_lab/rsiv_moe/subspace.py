from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class BasisFit:
    """An uncentred origin-subspace fit to row-wise activation samples."""

    basis: torch.Tensor
    singular_values: torch.Tensor
    stored_rank: int
    effective_rank: float
    energy_ranks: dict[str, int]
    reconstruction_relative_l2: float
    reconstruction_max_abs: float


def _matrix(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError("samples must have shape [observations, features]")
    if not torch.isfinite(samples).all():
        raise ValueError("samples must be finite")
    return samples


def energy_rank(singular_values: torch.Tensor, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    values = singular_values.double().square()
    if values.numel() == 0 or float(values.sum().item()) == 0.0:
        return 0
    target = fraction * values.sum()
    return int(torch.searchsorted(values.cumsum(0), target).item()) + 1


def _effective_rank(singular_values: torch.Tensor) -> float:
    energies = singular_values.double().square()
    total = energies.sum()
    if energies.numel() == 0 or float(total.item()) == 0.0:
        return 0.0
    probabilities = energies / total
    positive = probabilities > 0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return float(entropy.exp().item())


def fit_origin_subspace(
    samples: torch.Tensor,
    rank_cap: int | None = None,
) -> BasisFit:
    """Fit an FP64, uncentred right-singular-vector basis.

    The rank refers to the finite stored matrix under the standard LAPACK-style
    tolerance ``max(n, d) * eps(float64) * sigma_max``. No mean is removed.
    """

    samples = _matrix(samples)
    observations, features = samples.shape
    if rank_cap is not None and rank_cap < 0:
        raise ValueError("rank_cap must be non-negative")
    if observations == 0 or features == 0:
        empty = torch.empty(features, 0, dtype=torch.float64, device=samples.device)
        return BasisFit(empty, torch.empty(0, dtype=torch.float64), 0, 0.0,
                        {"r90": 0, "r95": 0, "r99": 0, "r999": 0}, 0.0, 0.0)

    matrix = samples.to(torch.float64)
    _u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = (
        max(observations, features)
        * torch.finfo(torch.float64).eps
        * singular_values[0]
    )
    stored_rank = int((singular_values > tolerance).sum().item())
    retained = stored_rank if rank_cap is None else min(stored_rank, rank_cap)
    basis = vh[:retained].transpose(0, 1).contiguous()
    reconstructed = (matrix @ basis) @ basis.transpose(0, 1)
    delta = reconstructed - matrix
    denominator = float(torch.linalg.vector_norm(matrix).item())
    relative = (
        float(torch.linalg.vector_norm(delta).item()) / denominator
        if denominator > 0.0
        else 0.0
    )
    maximum = float(delta.abs().max().item()) if delta.numel() else 0.0
    active_singular = singular_values[:stored_rank]
    return BasisFit(
        basis=basis,
        singular_values=singular_values,
        stored_rank=stored_rank,
        effective_rank=_effective_rank(active_singular),
        energy_ranks={
            "r90": energy_rank(active_singular, 0.90),
            "r95": energy_rank(active_singular, 0.95),
            "r99": energy_rank(active_singular, 0.99),
            "r999": energy_rank(active_singular, 0.999),
        },
        reconstruction_relative_l2=relative,
        reconstruction_max_abs=maximum,
    )


def relative_residual_ratio(
    samples: torch.Tensor,
    basis: torch.Tensor,
    epsilon: float = 1e-30,
) -> torch.Tensor:
    samples = _matrix(samples)
    if basis.ndim != 2 or basis.shape[0] != samples.shape[1]:
        raise ValueError("basis must have shape [features, rank]")
    matrix = samples.to(torch.float64)
    q = basis.to(device=matrix.device, dtype=torch.float64)
    residual = matrix - (matrix @ q) @ q.transpose(0, 1)
    numerator = torch.linalg.vector_norm(residual, dim=1)
    denominator = torch.linalg.vector_norm(matrix, dim=1)
    ratios = numerator / denominator.clamp_min(epsilon)
    zero = denominator == 0
    if zero.any():
        ratios = ratios.clone()
        ratios[zero] = torch.where(
            numerator[zero] == 0,
            torch.zeros_like(numerator[zero]),
            torch.full_like(numerator[zero], float("inf")),
        )
    return ratios


def append_residual_direction(
    basis: torch.Tensor,
    vector: torch.Tensor,
    tolerance: float = 1e-12,
) -> tuple[torch.Tensor, float, bool]:
    """Append one direction using two-pass DGKS reorthogonalisation."""

    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if basis.ndim != 2 or basis.shape[0] != vector.numel():
        raise ValueError("basis and vector feature dimensions must match")
    q = basis.to(dtype=torch.float64, device=vector.device)
    residual = vector.to(torch.float64)
    original_norm = float(torch.linalg.vector_norm(residual).item())
    if original_norm == 0.0:
        return q, 0.0, False
    for _ in range(2):
        if q.shape[1]:
            residual = residual - q @ (q.transpose(0, 1) @ residual)
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    ratio = residual_norm / original_norm
    if residual_norm <= tolerance * original_norm:
        return q, ratio, False
    direction = (residual / residual_norm).unsqueeze(1)
    return torch.cat((q, direction), dim=1), ratio, True


def online_fault_curve(
    samples: torch.Tensor,
    threshold: float,
    rank_cap: int,
) -> dict[str, Any]:
    samples = _matrix(samples)
    if threshold < 0.0 or rank_cap < 0:
        raise ValueError("threshold and rank_cap must be non-negative")
    basis = torch.empty(samples.shape[1], 0, dtype=torch.float64, device=samples.device)
    ratios: list[float] = []
    misses: list[bool] = []
    ranks: list[int] = []
    additions = 0
    for row in samples:
        ratio = float(relative_residual_ratio(row.unsqueeze(0), basis)[0].item())
        miss = ratio > threshold
        if miss and basis.shape[1] < rank_cap:
            basis, _append_ratio, added = append_residual_direction(basis, row)
            additions += int(added)
        ratios.append(ratio)
        misses.append(miss)
        ranks.append(int(basis.shape[1]))
    return {
        "residual_ratios": ratios,
        "misses": misses,
        "ranks_after": ranks,
        "rank_additions": additions,
        "final_rank": int(basis.shape[1]),
    }


def cold_byte_fraction(x_miss: torch.Tensor, z_miss: torch.Tensor) -> torch.Tensor:
    if x_miss.shape != z_miss.shape:
        raise ValueError("x_miss and z_miss must have equal shape")
    return (2.0 * x_miss.to(torch.float64) + z_miss.to(torch.float64)) / 3.0


def image_storage_elements(
    d: int,
    m: int,
    input_ranks: Iterable[int],
    intermediate_ranks: Iterable[int],
) -> int:
    if d <= 0 or m <= 0:
        raise ValueError("dimensions must be positive")
    r = list(input_ranks)
    s = list(intermediate_ranks)
    if len(r) != len(s) or any(value < 0 for value in (*r, *s)):
        raise ValueError("rank lists must align and contain non-negative values")
    return int(sum((d + 2 * m) * ri + (m + d) * si for ri, si in zip(r, s)))


def select_validation_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered global P1 selection without seeing test data."""

    if not rows:
        raise ValueError("at least one validation row is required")
    required = {
        "rank_cap",
        "threshold",
        "offline_double_fast_fraction",
        "offline_cold_byte_reduction",
        "causal_double_fast_fraction",
        "causal_cold_byte_reduction",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"candidate is missing fields: {sorted(missing)}")
        if "test" in " ".join(row).lower():
            raise ValueError("validation selection must not contain test fields")

    primary = [
        row
        for row in rows
        if int(row["rank_cap"]) <= 32
        and float(row["offline_double_fast_fraction"]) >= 0.92
        and float(row["offline_cold_byte_reduction"]) >= 10.0
        and float(row["causal_double_fast_fraction"]) >= 0.92
        and float(row["causal_cold_byte_reduction"]) >= 10.0
    ]
    if primary:
        selected = min(primary, key=lambda row: (float(row["threshold"]), int(row["rank_cap"])))
        return {**selected, "selection_kind": "primary_gate_pass"}

    def score(row: dict[str, Any]) -> float:
        return min(
            float(row["offline_double_fast_fraction"]) / 0.92,
            float(row["offline_cold_byte_reduction"]) / 10.0,
            float(row["causal_double_fast_fraction"]) / 0.92,
            float(row["causal_cold_byte_reduction"]) / 10.0,
        )

    eligible = [row for row in rows if int(row["rank_cap"]) <= 32]
    if not eligible:
        raise ValueError("validation grid contains no rank_cap <= 32")
    selected = min(
        eligible,
        key=lambda row: (-score(row), float(row["threshold"]), int(row["rank_cap"])),
    )
    return {
        **selected,
        "selection_kind": "diagnostic_validation_failure",
        "normalized_bottleneck_score": score(selected),
    }


def select_single_evaluation_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the preregistered P1B long-prefix validation candidate."""

    if not rows:
        raise ValueError("at least one validation row is required")
    required = {"rank_cap", "threshold", "double_gate_fast_fraction", "cold_byte_reduction"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"candidate is missing fields: {sorted(missing)}")
        if "test" in " ".join(row).lower():
            raise ValueError("validation selection must not contain test fields")
    primary = [
        row
        for row in rows
        if int(row["rank_cap"]) <= 32
        and float(row["double_gate_fast_fraction"]) >= 0.92
        and float(row["cold_byte_reduction"]) >= 10.0
    ]
    if primary:
        selected = min(primary, key=lambda row: (float(row["threshold"]), int(row["rank_cap"])))
        return {**selected, "selection_kind": "primary_gate_pass"}

    def score(row: dict[str, Any]) -> float:
        return min(
            float(row["double_gate_fast_fraction"]) / 0.92,
            float(row["cold_byte_reduction"]) / 10.0,
        )

    eligible = [row for row in rows if int(row["rank_cap"]) <= 32]
    if not eligible:
        raise ValueError("validation grid contains no rank_cap <= 32")
    selected = min(
        eligible,
        key=lambda row: (-score(row), float(row["threshold"]), int(row["rank_cap"])),
    )
    return {
        **selected,
        "selection_kind": "diagnostic_validation_failure",
        "normalized_bottleneck_score": score(selected),
    }
