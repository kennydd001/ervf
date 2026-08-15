from __future__ import annotations

from typing import Any

import torch


def routed_output(selected: torch.Tensor, router_weights: torch.Tensor) -> torch.Tensor:
    if selected.ndim != 3 or router_weights.shape != selected.shape[:2]:
        raise ValueError("selected must be [tokens, slots, hidden] with aligned weights")
    return (
        selected.float() * router_weights.float().unsqueeze(-1)
    ).sum(dim=1).to(selected.dtype)


def weighted_error_decomposition(
    bf16_selected: torch.Tensor,
    q3_selected: torch.Tensor,
    router_weights: torch.Tensor,
) -> dict[str, Any]:
    if bf16_selected.shape != q3_selected.shape:
        raise ValueError("BF16 and Q3 selected outputs must have equal shape")
    if router_weights.shape != bf16_selected.shape[:2]:
        raise ValueError("router weights must match token and slot dimensions")
    errors = (
        (q3_selected.float() - bf16_selected.float())
        * router_weights.float().unsqueeze(-1)
    )
    diagonal = errors.square().sum(dim=(1, 2))
    aggregate = errors.sum(dim=1).square().sum(dim=1)
    cross = aggregate - diagonal
    cancellation = torch.where(
        diagonal > 0,
        (diagonal - aggregate) / diagonal,
        torch.zeros_like(diagonal),
    )
    diagonal_sum = diagonal.double().sum()
    aggregate_sum = aggregate.double().sum()
    cross_sum = cross.double().sum()
    global_cancellation = (
        (diagonal_sum - aggregate_sum) / diagonal_sum
        if diagonal_sum > 0
        else torch.zeros((), dtype=torch.float64)
    )
    return {
        "diagonal_energy": diagonal,
        "aggregate_energy": aggregate,
        "cross_term": cross,
        "token_cancellation_fraction": cancellation,
        "diagonal_energy_sum": float(diagonal_sum.item()),
        "aggregate_energy_sum": float(aggregate_sum.item()),
        "cross_term_sum": float(cross_sum.item()),
        "global_cancellation_fraction": float(global_cancellation.item()),
    }


def routed_design_matrix(
    q3_selected: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    experts: int = 64,
) -> torch.Tensor:
    """Return A[t,h,e] = routed contribution of expert e before its gain."""

    if q3_selected.ndim != 3:
        raise ValueError("q3_selected must be [tokens, slots, hidden]")
    if expert_ids.shape != q3_selected.shape[:2]:
        raise ValueError("expert_ids must match token and slot dimensions")
    if router_weights.shape != expert_ids.shape:
        raise ValueError("router_weights must match expert_ids")
    if experts < 1 or int(expert_ids.min()) < 0 or int(expert_ids.max()) >= experts:
        raise ValueError("expert ids are outside the configured expert count")
    tokens, slots, hidden = q3_selected.shape
    design = torch.zeros(
        tokens,
        hidden,
        experts,
        dtype=torch.float32,
        device=q3_selected.device,
    )
    for slot in range(slots):
        contribution = (
            q3_selected[:, slot].float()
            * router_weights[:, slot].float().unsqueeze(1)
        )
        indices = expert_ids[:, slot].long().view(tokens, 1, 1).expand(
            tokens, hidden, 1
        )
        design.scatter_add_(2, indices, contribution.unsqueeze(2))
    return design


def _ridge_system(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    experts = gram.shape[-1]
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    regularization = alpha * scale.clamp_min(torch.finfo(gram.dtype).tiny)
    eye = torch.eye(experts, dtype=gram.dtype, device=gram.device)
    matrix = gram + regularization[..., None, None] * eye
    target = rhs + regularization[..., None]
    return matrix, target, regularization


def fit_corouted_row_gains(
    design: torch.Tensor,
    target_routed: torch.Tensor,
    *,
    alpha: float,
    lower: float = 0.75,
    upper: float = 1.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit one bounded gain per expert/output row using co-routed loss."""

    if design.ndim != 3 or target_routed.shape != design.shape[:2]:
        raise ValueError("design must be [tokens, hidden, experts] with target [tokens, hidden]")
    if not lower < upper or not lower <= 1.0 <= upper:
        raise ValueError("bounds must contain one")
    channel_design = design.permute(1, 0, 2).contiguous()
    target = target_routed.float().T.contiguous()
    gram = torch.bmm(channel_design.transpose(1, 2), channel_design)
    rhs = torch.bmm(
        channel_design.transpose(1, 2), target.unsqueeze(2)
    ).squeeze(2)
    matrix, regularized_rhs, regularization = _ridge_system(gram, rhs, alpha)
    gains_by_channel = torch.linalg.solve(
        matrix, regularized_rhs.unsqueeze(2)
    ).squeeze(2)
    finite_before_clamp = bool(torch.isfinite(gains_by_channel).all())
    if not finite_before_clamp:
        raise RuntimeError("non-finite co-routed row gain")
    clamped = gains_by_channel.clamp(lower, upper)
    gains = clamped.T.contiguous()
    return gains, {
        "alpha": alpha,
        "bounds": [lower, upper],
        "finite_before_clamp": finite_before_clamp,
        "lower_clamped_values": int(gains_by_channel.lt(lower).sum().item()),
        "upper_clamped_values": int(gains_by_channel.gt(upper).sum().item()),
        "total_gain_values": gains_by_channel.numel(),
        "regularization_minimum": float(regularization.min().item()),
        "regularization_maximum": float(regularization.max().item()),
    }


def fit_corouted_scalar_gains(
    design: torch.Tensor,
    target_routed: torch.Tensor,
    *,
    alpha: float,
    lower: float = 0.75,
    upper: float = 1.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if design.ndim != 3 or target_routed.shape != design.shape[:2]:
        raise ValueError("design and target shapes do not align")
    flattened = design.reshape(-1, design.shape[-1])
    target = target_routed.float().reshape(-1)
    gram = flattened.T @ flattened
    rhs = flattened.T @ target
    matrix, regularized_rhs, regularization = _ridge_system(gram, rhs, alpha)
    gains = torch.linalg.solve(matrix, regularized_rhs).clamp(lower, upper)
    if not torch.isfinite(gains).all():
        raise RuntimeError("non-finite scalar gain")
    return gains, {
        "alpha": alpha,
        "bounds": [lower, upper],
        "lower_clamped_values": int(gains.eq(lower).sum().item()),
        "upper_clamped_values": int(gains.eq(upper).sum().item()),
        "total_gain_values": gains.numel(),
        "regularization": float(regularization.item()),
    }


def fit_independent_row_gains(
    bf16_selected: torch.Tensor,
    q3_selected: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    experts: int = 64,
    lower: float = 0.75,
    upper: float = 1.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if bf16_selected.shape != q3_selected.shape:
        raise ValueError("BF16 and Q3 outputs must match")
    tokens, slots, hidden = q3_selected.shape
    numerator = torch.zeros(
        experts, hidden, dtype=torch.float64, device=q3_selected.device
    )
    denominator = torch.zeros_like(numerator)
    counts = torch.zeros(experts, dtype=torch.long, device=q3_selected.device)
    for slot in range(slots):
        ids = expert_ids[:, slot].long()
        weight_square = router_weights[:, slot].double().square().unsqueeze(1)
        q = q3_selected[:, slot].double()
        b = bf16_selected[:, slot].double()
        numerator.index_add_(0, ids, weight_square * q * b)
        denominator.index_add_(0, ids, weight_square * q.square())
        counts.index_add_(0, ids, torch.ones(tokens, dtype=torch.long, device=ids.device))
    gains = torch.where(
        denominator > 0, numerator / denominator, torch.ones_like(denominator)
    )
    unclamped = gains
    gains = gains.clamp(lower, upper).float()
    return gains, {
        "bounds": [lower, upper],
        "lower_clamped_values": int(unclamped.lt(lower).sum().item()),
        "upper_clamped_values": int(unclamped.gt(upper).sum().item()),
        "total_gain_values": gains.numel(),
        "expert_invocation_counts": counts.cpu().tolist(),
        "unseen_experts": int(counts.eq(0).sum().item()),
    }


def apply_output_gains(
    q3_selected: torch.Tensor,
    expert_ids: torch.Tensor,
    gains: torch.Tensor,
) -> torch.Tensor:
    if expert_ids.shape != q3_selected.shape[:2]:
        raise ValueError("expert ids must match selected outputs")
    ids = expert_ids.long().to(gains.device)
    q3 = q3_selected.to(gains.device)
    if gains.ndim == 1:
        selected_gains = gains[ids].unsqueeze(-1)
    elif gains.ndim == 2 and gains.shape[1] == q3.shape[-1]:
        selected_gains = gains[ids]
    else:
        raise ValueError("gains must be [experts] or [experts, hidden]")
    return (q3.float() * selected_gains.float()).to(q3.dtype)


def aggregate_error_reduction(
    reference_routed: torch.Tensor,
    baseline_routed: torch.Tensor,
    candidate_routed: torch.Tensor,
) -> dict[str, float]:
    if reference_routed.shape != baseline_routed.shape or reference_routed.shape != candidate_routed.shape:
        raise ValueError("routed tensors must have equal shape")
    baseline_error = (baseline_routed.float() - reference_routed.float()).square()
    candidate_error = (candidate_routed.float() - reference_routed.float()).square()
    baseline_mse = float(baseline_error.double().mean().item())
    candidate_mse = float(candidate_error.double().mean().item())
    if baseline_mse <= 0:
        reduction = 0.0 if candidate_mse == 0 else float("-inf")
    else:
        reduction = 1.0 - candidate_mse / baseline_mse
    return {
        "baseline_mse": baseline_mse,
        "candidate_mse": candidate_mse,
        "error_reduction": reduction,
    }


def scale_layout_accounting(
    *, experts: int = 64, hidden_size: int = 2048, scale_bytes: int = 2
) -> dict[str, Any]:
    values = experts * hidden_size
    return {
        "experts": experts,
        "output_rows_per_expert": hidden_size,
        "existing_down_scale_values": values,
        "candidate_down_scale_values": values,
        "existing_scale_bytes": values * scale_bytes,
        "candidate_scale_bytes": values * scale_bytes,
        "additional_values": 0,
        "additional_bytes": 0,
        "integer_codes_changed": False,
        "tensor_shape_changed": False,
        "new_kernel_operand": False,
        "interpretation": "gain is folded into each existing down-row dequantization scale",
    }
