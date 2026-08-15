from __future__ import annotations

import itertools

import torch

from moe_lab.craft_moe.crcq import (
    best_by_upgrade_count,
    force_natural_shortlist,
    local_routed_mean_squared_error,
    mean_gap_closure,
    mixed_precision_routed,
    natural_subset_index,
    routed_for_routes,
    routed_from_choices,
    six_of_twelve_subsets,
    solve_minimum_budget,
)
from moe_lab.craft_moe.route_coreset import delta_patched_hidden
from moe_lab.dynamic_precision import binary_upgrade_masks


def test_six_of_twelve_contains_all_routes_and_unique_natural_route() -> None:
    subsets = six_of_twelve_subsets()

    assert subsets.shape == (924, 6)
    assert len({tuple(row) for row in subsets.tolist()}) == 924
    assert natural_subset_index(subsets) == 0


def test_shortlist_forces_natural_without_duplicates() -> None:
    damage = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0]])

    shortlist, forced = force_natural_shortlist(damage, natural_index=0, shortlist_size=2)

    assert torch.equal(shortlist, torch.tensor([[0, 1], [3, 0]]))
    assert torch.equal(forced, torch.tensor([False, True]))
    assert all(len(set(row)) == 2 for row in shortlist.tolist())


def test_vectorized_route_and_bit_mixture_match_brute_force() -> None:
    q3 = torch.arange(5 * 4, dtype=torch.float32).view(5, 4) / 10
    q4 = q3 + torch.tensor([0.1, -0.2, 0.3, -0.4])
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    routes = torch.tensor([[0, 1, 2], [1, 3, 4]])
    masks = torch.tensor(list(itertools.product((False, True), repeat=3)))

    all_q3 = routed_for_routes(q3, weights, routes)
    mixed = mixed_precision_routed(q3, q4, weights, routes, masks)

    assert torch.allclose(mixed[:, 0], all_q3)
    for route_index, route in enumerate(routes):
        for mask_index, mask in enumerate(masks):
            expected = torch.zeros(4)
            for slot, expert in enumerate(route):
                expected += weights[expert] * (q4[expert] if mask[slot] else q3[expert])
            assert torch.allclose(mixed[route_index, mask_index], expected, atol=1e-6)


def test_best_route_mask_is_exact_at_each_upgrade_count() -> None:
    masks = binary_upgrade_masks(2)
    damage = torch.tensor(
        [[[9.0, 6.0, 7.0, 5.0], [8.0, 4.0, 3.0, 2.0]]]
    )

    best, routes, selected_masks = best_by_upgrade_count(damage, masks)

    assert torch.equal(best, torch.tensor([[8.0, 3.0, 2.0]]))
    assert torch.equal(routes, torch.tensor([[1, 1, 1]]))
    assert masks[selected_masks[0, 1]].sum() == 1


def test_global_budget_solver_matches_small_brute_force() -> None:
    damage = torch.tensor(
        [
            [4.0, 2.0, 1.5, 1.0, 0.9, 0.8, 0.7],
            [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0],
        ]
    )

    solution = solve_minimum_budget(
        damage, reference_mean_damage=1.25, tolerance_multiplier=1.0
    )

    assert solution.total_cost == 6
    assert solution.per_token_cost is not None
    assert int(solution.per_token_cost.sum()) == 6
    assert solution.upgrade_fraction == 0.5
    assert solution.average_active_bits == 3.5


def test_choice_reconstruction_and_official_delta_control() -> None:
    generator = torch.Generator().manual_seed(11)
    q3 = torch.randn(2, 12, 5, generator=generator)
    q4 = q3 + 0.1
    weights = torch.softmax(torch.randn(2, 12, generator=generator), dim=1)
    subsets = six_of_twelve_subsets()
    masks = binary_upgrade_masks(6)
    natural = natural_subset_index(subsets)
    routes = torch.full((2,), natural)
    mask_indices = torch.zeros(2, dtype=torch.long)
    natural_q3 = routed_from_choices(
        q3, q4, weights, subsets, routes, masks, mask_indices
    )
    teacher = torch.randn(2, 5, generator=generator, dtype=torch.bfloat16)

    patched = delta_patched_hidden(teacher, natural_q3, natural_q3)

    assert torch.equal(patched, teacher)


def test_gap_closure_uses_aggregate_natural_q3_to_q4_gap() -> None:
    natural_q3 = torch.tensor([6.0, 4.0])
    natural_q4 = torch.tensor([2.0, 2.0])
    alternative = torch.tensor([4.0, 3.0])

    assert mean_gap_closure(natural_q3, natural_q4, alternative) == 0.5


def test_local_routed_damage_broadcasts_over_routes_and_masks() -> None:
    target = torch.tensor([1.0, 2.0])
    candidates = torch.tensor(
        [[[1.0, 2.0], [3.0, 2.0]], [[0.0, 1.0], [2.0, 3.0]]]
    )

    damage = local_routed_mean_squared_error(candidates, target)

    assert torch.equal(damage, torch.tensor([[0.0, 2.0], [1.0, 1.0]]))
