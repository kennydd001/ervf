import itertools

import numpy as np
import torch

from moe_lab.dynamic_precision import (
    best_mask_per_cardinality,
    binary_upgrade_masks,
    discrete_rate_distortion,
    recover_cost_schedule,
)


def test_binary_upgrade_masks_cover_all_patterns() -> None:
    masks = binary_upgrade_masks(3)
    assert masks.shape == (8, 3)
    assert masks.sum(dim=1).tolist() == [0, 1, 1, 2, 1, 2, 2, 3]


def test_best_mask_per_cardinality() -> None:
    masks = binary_upgrade_masks(2)
    damage = torch.tensor([[9.0, 4.0, 2.0, 1.0]])
    best, indices = best_mask_per_cardinality(damage, masks)
    assert best.tolist() == [[9.0, 2.0, 1.0]]
    assert indices.tolist() == [[0, 2, 3]]


def test_rate_distortion_matches_brute_force() -> None:
    damage = np.array([[4.0, 2.0, 1.0], [3.0, 2.5, 0.0], [5.0, 1.0, 0.5]])
    curve, backpointers = discrete_rate_distortion(damage)
    brute = np.full(7, np.inf)
    for choices in itertools.product(range(3), repeat=3):
        cost = sum(choices)
        value = sum(damage[token, choice] for token, choice in enumerate(choices))
        brute[cost] = min(brute[cost], value)
    assert np.allclose(curve, brute)
    for cost in range(7):
        schedule = recover_cost_schedule(backpointers, cost)
        assert int(schedule.sum()) == cost
        assert np.isclose(
            sum(damage[token, choice] for token, choice in enumerate(schedule)),
            curve[cost],
        )
