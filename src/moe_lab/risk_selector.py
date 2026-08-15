from __future__ import annotations

import math

import torch
from torch import nn


def subset_mask(subsets: torch.Tensor, candidates: int = 12) -> torch.Tensor:
    if subsets.ndim != 2:
        raise ValueError("subsets must have shape [routes, selected]")
    mask = torch.zeros(
        subsets.shape[0], candidates, dtype=torch.bool, device=subsets.device
    )
    mask.scatter_(1, subsets.to(torch.long), True)
    return mask


def top_j_slate(subsets: torch.Tensor, top_j: int) -> torch.Tensor:
    if not 0 <= top_j <= subsets.shape[1]:
        raise ValueError("top_j must be between zero and the route width")
    mask = subset_mask(subsets)
    return torch.arange(top_j, device=subsets.device).view(1, -1).eq(
        subsets.unsqueeze(-1)
    ).any(dim=1).all(dim=1)


def route_risk_features(
    router_weights: torch.Tensor, subsets: torch.Tensor, original_k: int = 6
) -> torch.Tensor:
    """Build teacher-free features for every token/route pair.

    The features expose router scores and rank substitutions, but no expert
    outputs, teacher logits, exact KL values, or cache outcomes.
    """
    if router_weights.ndim != 2:
        raise ValueError("router_weights must have shape [tokens, candidates]")
    if subsets.ndim != 2:
        raise ValueError("subsets must have shape [routes, selected]")
    tokens, candidates = router_weights.shape
    if subsets.max().item() >= candidates:
        raise ValueError("subset rank exceeds router candidate count")
    if not 0 < original_k <= candidates:
        raise ValueError("original_k must be in [1, candidates]")

    route_mask = subset_mask(subsets, candidates).to(router_weights.dtype)
    original_mask = torch.zeros(candidates, dtype=torch.bool, device=subsets.device)
    original_mask[:original_k] = True
    dropped = original_mask.unsqueeze(0) & ~route_mask.bool()
    added = ~original_mask.unsqueeze(0) & route_mask.bool()

    weights = router_weights.float().clamp_min(1e-12)
    normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_relative = torch.log(weights / weights[:, :1]).clamp(min=-20.0, max=0.0)
    expanded_weights = normalized.unsqueeze(1).expand(tokens, subsets.shape[0], candidates)
    expanded_log = log_relative.unsqueeze(1).expand_as(expanded_weights)
    expanded_mask = route_mask.unsqueeze(0).expand_as(expanded_weights)
    selected_weights = expanded_weights * expanded_mask
    signed_change = expanded_weights * (
        expanded_mask - original_mask.to(expanded_mask.dtype).view(1, 1, -1)
    )

    dropped_f = dropped.to(expanded_weights.dtype).unsqueeze(0)
    added_f = added.to(expanded_weights.dtype).unsqueeze(0)
    dropped_weight = (expanded_weights * dropped_f).sum(-1, keepdim=True)
    added_weight = (expanded_weights * added_f).sum(-1, keepdim=True)
    selected_sum = selected_weights.sum(-1, keepdim=True)
    original_sum = expanded_weights[..., :original_k].sum(-1, keepdim=True)
    changed = added_f.sum(-1, keepdim=True).expand(tokens, -1, -1)
    ranks = torch.arange(candidates, device=subsets.device, dtype=torch.float32)
    selected_rank_mean = (
        expanded_mask * ranks.view(1, 1, -1)
    ).sum(-1, keepdim=True) / subsets.shape[1]
    selected_rank_max = (
        expanded_mask * ranks.view(1, 1, -1)
    ).amax(-1, keepdim=True)
    selected_min = expanded_weights.masked_fill(
        ~expanded_mask.bool(), float("inf")
    ).amin(-1, keepdim=True)
    excluded_max = expanded_weights.masked_fill(
        expanded_mask.bool(), float("-inf")
    ).amax(-1, keepdim=True)
    entropy = -(
        normalized * normalized.clamp_min(1e-12).log()
    ).sum(-1, keepdim=True).unsqueeze(1).expand(tokens, subsets.shape[0], 1)
    top6_boundary = (
        normalized[:, original_k - 1 : original_k]
        - normalized[:, original_k : original_k + 1]
    ).unsqueeze(1).expand(tokens, subsets.shape[0], 1)

    summaries = torch.cat(
        (
            selected_sum,
            original_sum,
            dropped_weight,
            added_weight,
            added_weight - dropped_weight,
            changed / max(1, subsets.shape[1] - 2),
            selected_rank_mean / max(1, candidates - 1),
            selected_rank_max / max(1, candidates - 1),
            selected_min,
            excluded_max,
            entropy,
            top6_boundary,
        ),
        dim=-1,
    )
    return torch.cat(
        (
            expanded_log,
            expanded_weights,
            expanded_mask,
            selected_weights,
            signed_change,
            summaries,
        ),
        dim=-1,
    )


def expert_identity_features(
    router_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    subsets: torch.Tensor,
    total_experts: int,
    original_k: int = 6,
) -> torch.Tensor:
    """Encode which concrete experts a ranked candidate keeps or swaps."""
    if router_weights.shape != expert_ids.shape or router_weights.ndim != 2:
        raise ValueError("router_weights and expert_ids must share [tokens, ranks]")
    if expert_ids.min().item() < 0 or expert_ids.max().item() >= total_experts:
        raise ValueError("expert id outside total_experts")
    route_mask = subset_mask(subsets, router_weights.shape[1]).float()
    identity = torch.nn.functional.one_hot(
        expert_ids.to(torch.long), num_classes=total_experts
    ).float()
    normalized = router_weights.float() / router_weights.float().sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    selected_binary = torch.einsum("mr,tre->tme", route_mask, identity)
    selected_weighted = torch.einsum(
        "mr,tr,tre->tme", route_mask, normalized, identity
    )
    original_weighted = (
        identity[:, :original_k]
        * normalized[:, :original_k].unsqueeze(-1)
    ).sum(1)
    signed_weighted = selected_weighted - original_weighted.unsqueeze(1)
    return torch.cat(
        (
            selected_binary / subsets.shape[1],
            selected_weighted,
            signed_weighted,
        ),
        dim=-1,
    )


def finite_sample_upper_quantile(scores: torch.Tensor, alpha: float) -> float:
    """Split-conformal one-sided quantile with the finite-sample correction."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = min(scores.numel(), math.ceil((scores.numel() + 1) * (1 - alpha)))
    return float(torch.kthvalue(scores.float(), rank).values.item())


class RouteRiskMLP(nn.Module):
    def __init__(self, features: int, hidden: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
