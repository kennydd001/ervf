from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch


@dataclass
class CandidateFit:
    """Best per-token fit for one candidate family and cardinality."""

    coefficients: torch.Tensor
    subset_mask: torch.Tensor
    squared_error: torch.Tensor


def routed_target(
    selected_outputs: torch.Tensor, router_weights: torch.Tensor
) -> torch.Tensor:
    if selected_outputs.ndim != 3:
        raise ValueError("selected_outputs must have shape [tokens, experts, hidden]")
    if router_weights.shape != selected_outputs.shape[:2]:
        raise ValueError("router_weights must match [tokens, experts]")
    return torch.einsum(
        "teh,te->th", selected_outputs.double(), router_weights.double()
    )


def reconstruct_routed(
    selected_outputs: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    if selected_outputs.ndim != 3:
        raise ValueError("selected_outputs must have shape [tokens, experts, hidden]")
    if coefficients.shape != selected_outputs.shape[:2]:
        raise ValueError("coefficients must match [tokens, experts]")
    return torch.einsum(
        "teh,te->th", selected_outputs.double(), coefficients.double()
    )


def delta_patched_hidden(
    official_teacher: torch.Tensor,
    original_routed: torch.Tensor,
    candidate_routed: torch.Tensor,
) -> torch.Tensor:
    if official_teacher.shape != original_routed.shape:
        raise ValueError("official_teacher and original_routed must have equal shape")
    if candidate_routed.shape != original_routed.shape:
        raise ValueError("candidate_routed and original_routed must have equal shape")
    return (
        official_teacher.float()
        + (candidate_routed.float() - original_routed.float())
    ).to(official_teacher.dtype)


def _quadratic_error(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    target_norm_squared: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    linear = torch.einsum("te,te->t", coefficients, rhs)
    quadratic = torch.einsum("te,tef,tf->t", coefficients, gram, coefficients)
    return (target_norm_squared - 2.0 * linear + quadratic).clamp_min(0.0)


def _least_squares(
    gram: torch.Tensor, rhs: torch.Tensor, rtol: float = 1e-10
) -> torch.Tensor:
    inverse = torch.linalg.pinv(gram, hermitian=True, rtol=rtol)
    return torch.einsum("tij,tj->ti", inverse, rhs)


def box_coordinate_descent(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    upper: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    iterations: int = 256,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    """Solve batched box-constrained positive least squares by coordinate descent."""

    if gram.ndim != 3 or gram.shape[1] != gram.shape[2]:
        raise ValueError("gram must have shape [tokens, k, k]")
    if rhs.shape != gram.shape[:2] or upper.shape != rhs.shape:
        raise ValueError("rhs and upper must have shape [tokens, k]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if (upper < 0).any():
        raise ValueError("upper bounds must be non-negative")
    diagonal = gram.diagonal(dim1=1, dim2=2).clamp_min(1e-30)
    if initial is None:
        values = (rhs / diagonal).clamp_min(0.0)
    else:
        if initial.shape != rhs.shape:
            raise ValueError("initial must match rhs")
        values = initial.clone()
    values = torch.minimum(values, upper).clamp_min(0.0)
    for _ in range(iterations):
        previous = values.clone()
        for coordinate in range(values.shape[1]):
            interaction = torch.einsum(
                "tj,tj->t", gram[:, coordinate, :], values
            ) - gram[:, coordinate, coordinate] * values[:, coordinate]
            updated = (rhs[:, coordinate] - interaction) / diagonal[:, coordinate]
            values[:, coordinate] = torch.minimum(
                updated.clamp_min(0.0), upper[:, coordinate]
            )
        if float((values - previous).abs().max().item()) <= tolerance:
            break
    return values


def _empty_fit(tokens: int, experts: int, device: torch.device) -> CandidateFit:
    return CandidateFit(
        coefficients=torch.zeros(tokens, experts, dtype=torch.float64, device=device),
        subset_mask=torch.zeros(tokens, experts, dtype=torch.bool, device=device),
        squared_error=torch.full(
            (tokens,), float("inf"), dtype=torch.float64, device=device
        ),
    )


def _update_fit(
    current: CandidateFit,
    coefficients: torch.Tensor,
    subset_mask: torch.Tensor,
    squared_error: torch.Tensor,
    eligible: torch.Tensor | None = None,
) -> None:
    better = squared_error < current.squared_error
    if eligible is not None:
        better &= eligible
    if not better.any():
        return
    current.squared_error[better] = squared_error[better]
    current.coefficients[better] = coefficients[better]
    if subset_mask.ndim == 1:
        current.subset_mask[better] = subset_mask
    else:
        current.subset_mask[better] = subset_mask[better]


def _copy_fit(fit: CandidateFit) -> CandidateFit:
    return CandidateFit(
        coefficients=fit.coefficients.clone(),
        subset_mask=fit.subset_mask.clone(),
        squared_error=fit.squared_error.clone(),
    )


def enumerate_best_fits(
    selected_outputs: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    max_k: int = 5,
    box_multiplier: float = 2.0,
    nnls_tolerance: float = 1e-8,
    box_iterations: int = 256,
) -> dict[str, dict[int, CandidateFit]]:
    """Exhaust all selected-expert subsets and return per-token best fits.

    `nnls` is exact for the cardinality constraint: every possible positive
    least-squares active set is enumerated. The box family uses deterministic
    coordinate descent for 0 <= alpha_i <= box_multiplier * p_i.
    """

    if selected_outputs.ndim != 3:
        raise ValueError("selected_outputs must have shape [tokens, experts, hidden]")
    tokens, experts, _ = selected_outputs.shape
    if router_weights.shape != (tokens, experts):
        raise ValueError("router_weights must match selected_outputs")
    if not 1 <= max_k < experts:
        raise ValueError("max_k must be in [1, experts - 1]")
    if box_multiplier <= 0:
        raise ValueError("box_multiplier must be positive")

    outputs = selected_outputs.detach().cpu().double()
    weights = router_weights.detach().cpu().double()
    target = routed_target(outputs, weights)
    gram = torch.einsum("teh,tfh->tef", outputs, outputs)
    rhs = torch.einsum("teh,th->te", outputs, target)
    target_norm_squared = target.square().sum(-1).clamp_min(1e-30)
    device = outputs.device
    methods = ("original_drop", "free_ls", "nnls", "bounded_2x")
    exact: dict[str, dict[int, CandidateFit]] = {
        method: {
            k: _empty_fit(tokens, experts, device) for k in range(1, max_k + 1)
        }
        for method in methods
    }

    for k in range(1, max_k + 1):
        for subset in itertools.combinations(range(experts), k):
            index = torch.tensor(subset, dtype=torch.long)
            selected_gram = gram.index_select(1, index).index_select(2, index)
            selected_rhs = rhs.index_select(1, index)
            mask = torch.zeros(experts, dtype=torch.bool)
            mask[index] = True

            original_coefficients = torch.zeros(tokens, experts, dtype=torch.float64)
            original_coefficients[:, index] = weights[:, index]
            original_error = _quadratic_error(
                gram, rhs, target_norm_squared, original_coefficients
            )
            _update_fit(
                exact["original_drop"][k],
                original_coefficients,
                mask,
                original_error,
            )

            free_selected = _least_squares(selected_gram, selected_rhs)
            free_coefficients = torch.zeros(tokens, experts, dtype=torch.float64)
            free_coefficients[:, index] = free_selected
            free_error = _quadratic_error(
                gram, rhs, target_norm_squared, free_coefficients
            )
            _update_fit(
                exact["free_ls"][k],
                free_coefficients,
                mask,
                free_error,
            )

            nonnegative = (free_selected >= -nnls_tolerance).all(dim=1)
            nnls_coefficients = free_coefficients.clamp_min(0.0)
            nnls_error = _quadratic_error(
                gram, rhs, target_norm_squared, nnls_coefficients
            )
            _update_fit(
                exact["nnls"][k],
                nnls_coefficients,
                mask,
                nnls_error,
                eligible=nonnegative,
            )

            upper = box_multiplier * weights.index_select(1, index)
            bounded_selected = box_coordinate_descent(
                selected_gram,
                selected_rhs,
                upper,
                initial=torch.minimum(free_selected.clamp_min(0.0), upper),
                iterations=box_iterations,
            )
            bounded_coefficients = torch.zeros(tokens, experts, dtype=torch.float64)
            bounded_coefficients[:, index] = bounded_selected
            bounded_error = _quadratic_error(
                gram, rhs, target_norm_squared, bounded_coefficients
            )
            _update_fit(
                exact["bounded_2x"][k],
                bounded_coefficients,
                mask,
                bounded_error,
            )

    nnls_at_most: dict[int, CandidateFit] = {}
    for k in range(1, max_k + 1):
        candidate = _copy_fit(exact["nnls"][k])
        if k > 1:
            previous = nnls_at_most[k - 1]
            _update_fit(
                candidate,
                previous.coefficients,
                previous.subset_mask,
                previous.squared_error,
            )
        nnls_at_most[k] = candidate
    exact["nnls"] = nnls_at_most
    return exact


def ranked_original_coefficients(
    router_weights: torch.Tensor, k: int
) -> torch.Tensor:
    if router_weights.ndim != 2:
        raise ValueError("router_weights must have shape [tokens, experts]")
    if not 1 <= k <= router_weights.shape[1]:
        raise ValueError("k is outside the expert count")
    weights = router_weights.double()
    ranked = weights.argsort(dim=1, descending=True, stable=True)
    selected = ranked[:, :k]
    coefficients = torch.zeros_like(weights)
    coefficients.scatter_(1, selected, weights.gather(1, selected))
    return coefficients


def rescaled_rank1_coefficients(
    selected_outputs: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    nonnegative: bool = True,
) -> torch.Tensor:
    weights = router_weights.detach().cpu().double()
    outputs = selected_outputs.detach().cpu().double()
    target = routed_target(outputs, weights)
    rank1 = weights.argsort(dim=1, descending=True, stable=True)[:, :1]
    vectors = outputs.gather(
        1, rank1.unsqueeze(-1).expand(-1, -1, outputs.shape[-1])
    ).squeeze(1)
    scalar = (vectors * target).sum(-1) / vectors.square().sum(-1).clamp_min(1e-30)
    if nonnegative:
        scalar = scalar.clamp_min(0.0)
    coefficients = torch.zeros_like(weights)
    coefficients.scatter_(1, rank1, scalar.unsqueeze(1))
    return coefficients


def relative_routed_error(
    selected_outputs: torch.Tensor,
    router_weights: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    target = routed_target(selected_outputs, router_weights)
    candidate = reconstruct_routed(selected_outputs, coefficients)
    return (
        (candidate - target).square().sum(-1)
        / target.square().sum(-1).clamp_min(1e-30)
    ).clamp_min(0.0).sqrt()


def minimum_k_from_kl(
    kl_by_k: dict[int, torch.Tensor], threshold: float
) -> torch.Tensor:
    if threshold < 0 or not kl_by_k:
        raise ValueError("threshold must be non-negative and kl_by_k non-empty")
    sizes = sorted(kl_by_k)
    tokens = kl_by_k[sizes[0]].numel()
    result = torch.full((tokens,), max(sizes) + 1, dtype=torch.long)
    for k in sizes:
        values = kl_by_k[k].reshape(-1)
        if values.numel() != tokens:
            raise ValueError("all KL tensors must contain the same token count")
        result[(result > max(sizes)) & (values <= threshold)] = k
    return result


def higher_quantile(values: torch.Tensor, probability: float) -> float:
    if values.numel() == 0 or not 0 <= probability <= 1:
        raise ValueError("values must be non-empty and probability in [0, 1]")
    ordered = values.double().sort().values
    index = max(0, math.ceil(probability * ordered.numel()) - 1)
    return float(ordered[index].item())
