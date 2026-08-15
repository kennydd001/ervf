from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import lsq_linear, nnls

from moe_lab.cache_routing import CacheRoutingPolicy, select_route, touch_route


@dataclass(frozen=True)
class SpanFit:
    method: str
    coefficients: torch.Tensor
    prediction: torch.Tensor
    squared_error: float
    normalized_squared_error: float
    cosine: float


def simulate_mass_budget_trace(
    ranked_ids: torch.Tensor,
    ranked_probabilities: torch.Tensor,
    raw_logits: torch.Tensor,
    *,
    capacity: int,
    delta_average: float,
    block_size: int,
    top_k: int = 6,
    top_j: int = 2,
    delta: float = 0.004,
) -> dict[str, object]:
    """Reproduce the baseline Mass-Budget LRU and retain pre-touch cache state."""

    if ranked_ids.ndim != 2 or ranked_ids.shape != ranked_probabilities.shape:
        raise ValueError("ranked ids/probabilities must be aligned [tokens, experts]")
    if raw_logits.shape != ranked_ids.shape:
        raise ValueError("raw logits must align with ranked tensors")
    if ranked_ids.shape[0] % block_size:
        raise ValueError("token count must be divisible by block_size")
    if capacity < top_k or not 0 < block_size:
        raise ValueError("invalid capacity or block size")

    ids = ranked_ids.cpu()
    probabilities = ranked_probabilities.cpu()
    logits = raw_logits.cpu()
    tokens, experts = ids.shape
    policy = CacheRoutingPolicy("mass_budget", top_j=top_j, parameter=delta)
    routes = torch.empty(tokens, top_k, dtype=torch.long)
    cache_before = torch.zeros(tokens, experts, dtype=torch.bool)
    miss_mask = torch.empty(tokens, top_k, dtype=torch.bool)
    per_block: list[dict[str, object]] = []
    total_misses = 0
    substitutions = 0

    for block_start in range(0, tokens, block_size):
        cache: list[int] = []
        block_misses = 0
        block_substitutions = 0
        for token in range(block_start, block_start + block_size):
            cache_before[token, cache] = True
            token_ids = ids[token].tolist()
            route = select_route(
                token_ids,
                probabilities[token].tolist(),
                logits[token].tolist(),
                set(cache),
                policy,
                delta_average,
                top_k,
            )
            routes[token] = torch.tensor(route, dtype=torch.long)
            changed = route != token_ids[:top_k]
            substitutions += int(changed)
            block_substitutions += int(changed)
            original_rank = {expert: rank for rank, expert in enumerate(token_ids)}
            touch_order = sorted(route, key=original_rank.__getitem__)
            audit_cache = list(cache)
            miss_by_expert: dict[int, bool] = {}
            for expert in touch_order:
                missed = expert not in audit_cache
                miss_by_expert[expert] = missed
                if not missed:
                    audit_cache.remove(expert)
                audit_cache.append(expert)
                if len(audit_cache) > capacity:
                    audit_cache.pop(0)
            misses = torch.tensor(
                [miss_by_expert[expert] for expert in route], dtype=torch.bool
            )
            miss_mask[token] = misses
            miss_count = int(misses.sum().item())
            total_misses += miss_count
            block_misses += miss_count
            observed = touch_route(cache, touch_order, capacity)
            if observed != miss_count:
                raise RuntimeError("sequential miss mask and LRU update disagree")
            if cache != audit_cache:
                raise RuntimeError("audited and primary LRU states disagree")
        per_block.append(
            {
                "block_start": block_start,
                "expert_loads": block_misses,
                "cache_miss_fraction": block_misses / (block_size * top_k),
                "substituted_token_fraction": block_substitutions / block_size,
            }
        )

    return {
        "routes": routes,
        "cache_before": cache_before,
        "miss_mask": miss_mask,
        "expert_loads": total_misses,
        "cache_miss_fraction": total_misses / (tokens * top_k),
        "substituted_token_fraction": substitutions / tokens,
        "per_block": per_block,
    }


def nonempty_subsets(indices: Iterable[int]) -> list[tuple[int, ...]]:
    values = tuple(int(index) for index in indices)
    return [
        subset
        for size in range(1, len(values) + 1)
        for subset in combinations(values, size)
    ]


def _fit_metrics(target: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float, float]:
    target64 = target.detach().double().cpu()
    prediction64 = prediction.detach().double().cpu()
    squared_error = float((prediction64 - target64).square().sum().item())
    energy = float(target64.square().sum().item())
    normalized = squared_error / max(energy, 1e-30)
    denominator = float(target64.norm().item() * prediction64.norm().item())
    cosine = (
        float(torch.dot(target64, prediction64).item()) / denominator
        if denominator > 0
        else float(target64.norm().item() == 0 and prediction64.norm().item() == 0)
    )
    return squared_error, normalized, cosine


def fit_span(
    basis: torch.Tensor,
    target: torch.Tensor,
    method: str,
    *,
    ridge_relative: float = 1e-4,
    coefficient_bound: float = 1.0,
) -> SpanFit:
    """Fit target from basis columns using one preregistered coefficient family."""

    if basis.ndim != 2 or target.ndim != 1 or basis.shape[0] != target.shape[0]:
        raise ValueError("basis must be [hidden, columns] and target [hidden]")
    if method not in {"ridge", "nnls", "bounded"}:
        raise ValueError(f"unsupported span method: {method}")
    if ridge_relative <= 0 or coefficient_bound <= 0:
        raise ValueError("ridge and coefficient bounds must be positive")
    if not torch.isfinite(basis).all() or not torch.isfinite(target).all():
        raise ValueError("basis and target must be finite")

    output_device = target.device
    output_dtype = target.dtype
    if basis.shape[1] == 0:
        coefficients = torch.empty(0, dtype=torch.float64)
        prediction = torch.zeros_like(target, dtype=torch.float64, device="cpu")
    else:
        matrix = np.asarray(basis.detach().double().cpu().numpy(), dtype=np.float64)
        vector = np.asarray(target.detach().double().cpu().numpy(), dtype=np.float64)
        if method == "ridge":
            gram = matrix.T @ matrix
            scale = max(float(np.trace(gram) / gram.shape[0]), 1e-30)
            system = gram + ridge_relative * scale * np.eye(gram.shape[0])
            rhs = matrix.T @ vector
            try:
                coefficients_np = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                coefficients_np = np.linalg.lstsq(system, rhs, rcond=None)[0]
        elif method == "nnls":
            coefficients_np, _ = nnls(matrix, vector)
        else:
            result = lsq_linear(
                matrix,
                vector,
                bounds=(-coefficient_bound, coefficient_bound),
                method="trf",
                lsq_solver="exact",
                tol=1e-10,
                max_iter=200,
            )
            if not result.success:
                raise RuntimeError(f"bounded least squares failed: {result.message}")
            coefficients_np = result.x
        coefficients = torch.from_numpy(np.asarray(coefficients_np, dtype=np.float64))
        prediction = torch.from_numpy(matrix @ coefficients_np)

    squared_error, normalized, cosine = _fit_metrics(
        target.detach().cpu(), prediction
    )
    return SpanFit(
        method=method,
        coefficients=coefficients.to(device=output_device, dtype=output_dtype),
        prediction=prediction.to(device=output_device, dtype=output_dtype),
        squared_error=squared_error,
        normalized_squared_error=normalized,
        cosine=cosine,
    )


def omp_cached_order(
    base_basis: torch.Tensor,
    cached_basis: torch.Tensor,
    target: torch.Tensor,
    *,
    max_extra: int = 4,
) -> list[int]:
    """Return a deterministic normalized-correlation OMP order."""

    if base_basis.ndim != 2 or cached_basis.ndim != 2 or target.ndim != 1:
        raise ValueError("basis matrices and target have invalid ranks")
    if base_basis.shape[0] != target.shape[0] or cached_basis.shape[0] != target.shape[0]:
        raise ValueError("basis matrices must share target hidden size")
    if max_extra < 0:
        raise ValueError("max_extra must be non-negative")
    selected: list[int] = []
    remaining = list(range(cached_basis.shape[1]))
    steps = min(max_extra, len(remaining))
    for _ in range(steps):
        if selected:
            basis = torch.cat((base_basis, cached_basis[:, selected]), dim=1)
        else:
            basis = base_basis
        residual = target.float() - fit_span(
            basis.float(), target.float(), "ridge", ridge_relative=1e-8
        ).prediction.float()
        vectors = cached_basis[:, remaining].float()
        norms = vectors.square().sum(0).sqrt().clamp_min(1e-30)
        scores = (vectors.T @ residual).abs() / norms
        best_local = int(torch.argmax(scores).item())
        selected.append(remaining.pop(best_local))
    return selected


def anchored_candidate(
    teacher: torch.Tensor,
    candidate_routed: torch.Tensor,
    natural_routed: torch.Tensor,
) -> torch.Tensor:
    if teacher.shape != candidate_routed.shape or teacher.shape != natural_routed.shape:
        raise ValueError("teacher and routed tensors must have equal shape")
    delta = (candidate_routed.float() - natural_routed.float()).to(teacher.dtype)
    return teacher + delta


def choose_lowest_mse_candidate(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    return min(
        candidates,
        key=lambda row: (
            float(row["target_squared_error"]),
            int(row["extra_computations"]),
            tuple(int(value) for value in row["reconstructed_expert_ids"]),
        ),
    )


def paired_load_bootstrap(
    baseline_misses: Iterable[int],
    primary_avoided: Iterable[int],
    zero_avoided: Iterable[int],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, object]:
    baseline = np.asarray(list(baseline_misses), dtype=np.float64)
    primary = np.asarray(list(primary_avoided), dtype=np.float64)
    zero = np.asarray(list(zero_avoided), dtype=np.float64)
    if baseline.ndim != 1 or not (
        baseline.size == primary.size == zero.size and baseline.size > 0
    ):
        raise ValueError("paired block series must be non-empty and aligned")
    if np.any(baseline <= 0) or np.any(primary < 0) or np.any(zero < 0):
        raise ValueError("block load counts are outside their valid range")
    if np.any(primary > baseline) or np.any(zero > baseline):
        raise ValueError("avoided loads cannot exceed baseline misses")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(resamples, baseline.size))
    sampled_baseline = baseline[indices].sum(axis=1)
    primary_reduction = primary[indices].sum(axis=1) / sampled_baseline
    zero_reduction = zero[indices].sum(axis=1) / sampled_baseline
    uplift = primary_reduction - zero_reduction

    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, (0.025, 0.975), method="linear")
        return {"low": float(low), "high": float(high)}

    return {
        "method": "paired sequence-block percentile bootstrap with replacement",
        "seed": seed,
        "resamples": resamples,
        "sampling_units": int(baseline.size),
        "point_estimates": {
            "primary_miss_reduction_fraction": float(primary.sum() / baseline.sum()),
            "zero_fill_miss_reduction_fraction": float(zero.sum() / baseline.sum()),
            "span_uplift_fraction": float((primary.sum() - zero.sum()) / baseline.sum()),
        },
        "intervals_95": {
            "primary_miss_reduction_fraction": interval(primary_reduction),
            "zero_fill_miss_reduction_fraction": interval(zero_reduction),
            "span_uplift_fraction": interval(uplift),
        },
        "raw": {
            "primary_miss_reduction_fraction": primary_reduction.tolist(),
            "zero_fill_miss_reduction_fraction": zero_reduction.tolist(),
            "span_uplift_fraction": uplift.tolist(),
        },
    }
