from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from moe_lab.dynamic_precision import discrete_rate_distortion, recover_cost_schedule


@dataclass
class BudgetSolution:
    target_total_damage: float
    total_cost: int | None
    upgrade_fraction: float | None
    average_active_bits: float | None
    per_token_cost: torch.Tensor | None
    exact_cost_curve: np.ndarray
    backpointers: np.ndarray


def six_of_twelve_subsets() -> torch.Tensor:
    return torch.tensor(
        list(itertools.combinations(range(12), 6)), dtype=torch.long
    )


def natural_subset_index(subsets: torch.Tensor) -> int:
    if subsets.ndim != 2 or subsets.shape[1] != 6:
        raise ValueError("subsets must have shape [routes, 6]")
    natural = torch.arange(6, dtype=subsets.dtype, device=subsets.device)
    matches = (subsets == natural).all(dim=1).nonzero(as_tuple=False).squeeze(1)
    if matches.numel() != 1:
        raise ValueError("subsets must contain the natural [0,1,2,3,4,5] route once")
    return int(matches.item())


def force_natural_shortlist(
    damage: torch.Tensor, natural_index: int, shortlist_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select lowest-damage routes and force natural into the final slot if absent."""

    if damage.ndim != 2:
        raise ValueError("damage must have shape [tokens, routes]")
    if not 1 <= shortlist_size <= damage.shape[1]:
        raise ValueError("shortlist_size is outside the route count")
    if not 0 <= natural_index < damage.shape[1]:
        raise ValueError("natural_index is outside the route count")
    shortlist = damage.argsort(dim=1, stable=True)[:, :shortlist_size].clone()
    present = (shortlist == natural_index).any(dim=1)
    shortlist[~present, -1] = natural_index
    return shortlist, ~present


def routed_for_routes(
    selected_outputs: torch.Tensor,
    router_weights: torch.Tensor,
    route_subsets: torch.Tensor,
) -> torch.Tensor:
    """Construct routed outputs for one token and an arbitrary route slate."""

    if selected_outputs.ndim != 2:
        raise ValueError("selected_outputs must have shape [experts, hidden]")
    if router_weights.shape != (selected_outputs.shape[0],):
        raise ValueError("router_weights must match the expert dimension")
    if route_subsets.ndim != 2:
        raise ValueError("route_subsets must have shape [routes, active_experts]")
    outputs = selected_outputs[route_subsets]
    weights = router_weights[route_subsets]
    routed = (outputs.float() * weights.float().unsqueeze(-1)).sum(dim=1)
    return routed.to(selected_outputs.dtype)


def mixed_precision_routed(
    selected_q3: torch.Tensor,
    selected_q4: torch.Tensor,
    router_weights: torch.Tensor,
    route_subsets: torch.Tensor,
    upgrade_masks: torch.Tensor,
) -> torch.Tensor:
    """Return [routes, masks, hidden] Q3/Q4 routed outputs for one token."""

    if selected_q3.shape != selected_q4.shape or selected_q3.ndim != 2:
        raise ValueError("selected_q3 and selected_q4 must match [experts, hidden]")
    if router_weights.shape != selected_q3.shape[:1]:
        raise ValueError("router_weights must match the expert dimension")
    if route_subsets.ndim != 2 or upgrade_masks.ndim != 2:
        raise ValueError("routes and masks must be two-dimensional")
    if route_subsets.shape[1] != upgrade_masks.shape[1]:
        raise ValueError("routes and masks must use the same active expert count")
    q3 = selected_q3[route_subsets].float()
    q4 = selected_q4[route_subsets].float()
    weights = router_weights[route_subsets].float().unsqueeze(-1)
    base = (q3 * weights).sum(dim=1)
    deltas = (q4 - q3) * weights
    routed = base.unsqueeze(1) + torch.einsum(
        "me,reh->rmh", upgrade_masks.float(), deltas
    )
    return routed.to(selected_q3.dtype)


def best_by_upgrade_count(
    damage: torch.Tensor, upgrade_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimize [tokens, routes, masks] damage at each exact upgrade count."""

    if damage.ndim != 3 or damage.shape[2] != upgrade_masks.shape[0]:
        raise ValueError("damage must have shape [tokens, routes, masks]")
    if upgrade_masks.ndim != 2:
        raise ValueError("upgrade_masks must have shape [masks, slots]")
    tokens, routes, _ = damage.shape
    slots = upgrade_masks.shape[1]
    cardinalities = upgrade_masks.sum(dim=1)
    best_damage = torch.empty(tokens, slots + 1, dtype=damage.dtype)
    best_route = torch.empty(tokens, slots + 1, dtype=torch.long)
    best_mask = torch.empty(tokens, slots + 1, dtype=torch.long)
    for count in range(slots + 1):
        mask_indices = (cardinalities == count).nonzero(as_tuple=False).squeeze(1)
        selected = damage[:, :, mask_indices].reshape(tokens, -1)
        values, flat_indices = selected.min(dim=1)
        masks_at_count = mask_indices.numel()
        best_damage[:, count] = values
        best_route[:, count] = flat_indices // masks_at_count
        best_mask[:, count] = mask_indices[flat_indices % masks_at_count]
    return best_damage, best_route, best_mask


def solve_minimum_budget(
    damage_by_cost: torch.Tensor,
    reference_mean_damage: float,
    *,
    tolerance_multiplier: float = 1.01,
) -> BudgetSolution:
    if damage_by_cost.ndim != 2 or damage_by_cost.shape[1] != 7:
        raise ValueError("damage_by_cost must have shape [tokens, 7]")
    if reference_mean_damage < 0 or tolerance_multiplier <= 0:
        raise ValueError("reference damage and tolerance must be non-negative")
    curve, backpointers = discrete_rate_distortion(
        damage_by_cost.detach().cpu().double().numpy()
    )
    target = reference_mean_damage * tolerance_multiplier * damage_by_cost.shape[0]
    qualifying = np.flatnonzero(curve <= target)
    if qualifying.size == 0:
        return BudgetSolution(
            target,
            None,
            None,
            None,
            None,
            curve,
            backpointers,
        )
    cost = int(qualifying[0])
    schedule = torch.from_numpy(recover_cost_schedule(backpointers, cost)).long()
    fraction = cost / (damage_by_cost.shape[0] * 6)
    return BudgetSolution(
        target,
        cost,
        fraction,
        3.0 + fraction,
        schedule,
        curve,
        backpointers,
    )


def best_schedule_within_fraction(
    solution: BudgetSolution, tokens: int, requested_fraction: float
) -> tuple[int, torch.Tensor]:
    if not 0 <= requested_fraction <= 1:
        raise ValueError("requested_fraction must be in [0, 1]")
    budget = min(int(requested_fraction * tokens * 6), tokens * 6)
    cost = int(np.argmin(solution.exact_cost_curve[: budget + 1]))
    schedule = torch.from_numpy(
        recover_cost_schedule(solution.backpointers, cost)
    ).long()
    return cost, schedule


def routed_from_choices(
    selected_q3: torch.Tensor,
    selected_q4: torch.Tensor,
    router_weights: torch.Tensor,
    all_subsets: torch.Tensor,
    route_indices: torch.Tensor,
    upgrade_masks: torch.Tensor,
    mask_indices: torch.Tensor,
) -> torch.Tensor:
    if selected_q3.shape != selected_q4.shape or selected_q3.ndim != 3:
        raise ValueError("selected outputs must match [tokens, experts, hidden]")
    tokens = selected_q3.shape[0]
    if router_weights.shape != selected_q3.shape[:2]:
        raise ValueError("router_weights must match [tokens, experts]")
    if route_indices.shape != (tokens,) or mask_indices.shape != (tokens,):
        raise ValueError("route_indices and mask_indices must match tokens")
    routes = all_subsets[route_indices]
    gather_index = routes.unsqueeze(-1).expand(-1, -1, selected_q3.shape[-1])
    q3 = selected_q3.gather(1, gather_index)
    q4 = selected_q4.gather(1, gather_index)
    weights = router_weights.gather(1, routes)
    masks = upgrade_masks[mask_indices].unsqueeze(-1)
    selected = torch.where(masks, q4, q3)
    routed = (selected.float() * weights.float().unsqueeze(-1)).sum(dim=1)
    return routed.to(selected_q3.dtype)


def mean_gap_closure(
    natural_q3_damage: torch.Tensor,
    natural_q4_damage: torch.Tensor,
    alternative_q3_damage: torch.Tensor,
) -> float:
    if not (
        natural_q3_damage.shape
        == natural_q4_damage.shape
        == alternative_q3_damage.shape
    ):
        raise ValueError("all damage tensors must have the same shape")
    denominator = (
        natural_q3_damage.double().mean() - natural_q4_damage.double().mean()
    )
    if denominator <= 0:
        return float("nan")
    numerator = (
        natural_q3_damage.double().mean()
        - alternative_q3_damage.double().mean()
    )
    return float((numerator / denominator).item())


def local_routed_mean_squared_error(
    candidate_routed: torch.Tensor, target_routed: torch.Tensor
) -> torch.Tensor:
    """Per-candidate local routed-output MSE with broadcasting over candidates."""

    if candidate_routed.ndim < 2 or target_routed.ndim != 1:
        raise ValueError("candidates must end in hidden and target must be [hidden]")
    if candidate_routed.shape[-1] != target_routed.shape[0]:
        raise ValueError("candidate and target hidden dimensions must match")
    return (
        candidate_routed.float() - target_routed.float()
    ).square().mean(dim=-1)
