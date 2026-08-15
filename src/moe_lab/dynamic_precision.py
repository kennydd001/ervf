from __future__ import annotations

import numpy as np
import torch


def binary_upgrade_masks(slots: int) -> torch.Tensor:
    """Return all 2**slots masks, ordered by their integer bit pattern."""
    values = torch.arange(1 << slots, dtype=torch.long)
    bits = torch.arange(slots, dtype=torch.long)
    return values.unsqueeze(1).bitwise_and(1 << bits).ne(0)


def best_mask_per_cardinality(
    damage: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the least-damaging mask for every token and upgrade count."""
    if damage.ndim != 2 or damage.shape[1] != masks.shape[0]:
        raise ValueError("damage must have shape [tokens, masks]")
    cardinality = masks.sum(dim=1)
    slots = masks.shape[1]
    best_damage = torch.empty(damage.shape[0], slots + 1, dtype=damage.dtype)
    best_mask = torch.empty(damage.shape[0], slots + 1, dtype=torch.long)
    for count in range(slots + 1):
        candidates = (cardinality == count).nonzero(as_tuple=False).squeeze(1)
        selected_damage = damage[:, candidates]
        values, local_indices = selected_damage.min(dim=1)
        best_damage[:, count] = values
        best_mask[:, count] = candidates[local_indices]
    return best_damage, best_mask


def discrete_rate_distortion(
    damage_by_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact dynamic program over independent token choices.

    ``damage_by_cost[i, k]`` is the best distortion for token ``i`` when
    exactly ``k`` of its expert invocations are upgraded. The returned first
    array contains minimum total distortion at every exact global cost. The
    uint8 backpointer matrix can be passed to ``recover_cost_schedule``.
    """
    if damage_by_cost.ndim != 2 or damage_by_cost.shape[1] < 1:
        raise ValueError("damage_by_cost must be a nonempty 2D array")
    tokens, choices = damage_by_cost.shape
    max_token_cost = choices - 1
    max_cost = tokens * max_token_cost
    previous = np.array([0.0], dtype=np.float64)
    backpointers = np.full((tokens, max_cost + 1), 255, dtype=np.uint8)
    for token in range(tokens):
        next_values = np.full(
            previous.shape[0] + max_token_cost, np.inf, dtype=np.float64
        )
        next_choice = np.full(next_values.shape[0], 255, dtype=np.uint8)
        for cost in range(choices):
            candidate = previous + float(damage_by_cost[token, cost])
            target = slice(cost, cost + previous.shape[0])
            better = candidate < next_values[target]
            next_values[target][better] = candidate[better]
            next_choice[target][better] = cost
        previous = next_values
        backpointers[token, : next_choice.shape[0]] = next_choice
    return previous, backpointers


def recover_cost_schedule(backpointers: np.ndarray, total_cost: int) -> np.ndarray:
    """Recover per-token costs for one exact global cost."""
    if total_cost < 0 or total_cost >= backpointers.shape[1]:
        raise ValueError("total_cost is outside the dynamic-program range")
    schedule = np.empty(backpointers.shape[0], dtype=np.uint8)
    remaining = int(total_cost)
    for token in range(backpointers.shape[0] - 1, -1, -1):
        choice = int(backpointers[token, remaining])
        if choice == 255:
            raise ValueError("requested total_cost is unreachable")
        schedule[token] = choice
        remaining -= choice
    if remaining != 0:
        raise RuntimeError("invalid backpointer chain")
    return schedule
