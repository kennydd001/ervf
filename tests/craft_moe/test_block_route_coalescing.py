from __future__ import annotations

import itertools

import torch

from moe_lab.craft_moe.block_coalescing import (
    RouteCandidate,
    beam_union_solution,
    brute_force_solution,
    build_candidate_slates,
    eligible_set_pruning,
    exact_ilp_solution,
    experts_to_mask,
    fixed_cache_prior_solution,
    highs_optimal_control,
    marginal_union_greedy,
    mass_budget_solution,
    natural_solution,
    solution_metrics,
)


def candidate(index: int, experts: tuple[int, ...], kl: float, natural=False):
    return RouteCandidate(index, experts, kl, 0.0 if natural else 0.002, natural)


def toy_slates() -> list[list[RouteCandidate]]:
    return [
        [
            candidate(0, (0, 1, 2, 3, 4, 5), 0.0, True),
            candidate(1, (6, 7, 8, 9, 10, 11), 0.0002),
        ],
        [
            candidate(0, (12, 13, 14, 15, 16, 17), 0.0, True),
            candidate(1, (6, 7, 8, 9, 10, 11), 0.0003),
        ],
        [
            candidate(0, (18, 19, 20, 21, 22, 23), 0.0, True),
            candidate(1, (6, 7, 8, 9, 10, 11), 0.0004),
        ],
    ]


def test_exact_ilp_matches_brute_force_and_objective() -> None:
    slates = toy_slates()
    exact = exact_ilp_solution(slates)
    brute = brute_force_solution(slates)

    assert exact.union_count == brute.union_count == 6
    assert exact.diagnostics is not None
    assert exact.diagnostics["status"] == 0
    assert exact.diagnostics["objective"] == 6


def test_beam_and_greedy_find_shared_route_on_toy_problem() -> None:
    slates = toy_slates()
    assert beam_union_solution(slates, width=16).union_count == 6
    assert marginal_union_greedy(slates).union_count == 18
    assert eligible_set_pruning(slates).union_count == 6


def test_exact_original_control_for_singleton_slates() -> None:
    slates = [[rows[0]] for rows in toy_slates()]
    methods = (
        natural_solution(slates),
        marginal_union_greedy(slates),
        mass_budget_solution(slates, 0.004),
        fixed_cache_prior_solution(slates, 0),
        eligible_set_pruning(slates),
        beam_union_solution(slates, 8),
        exact_ilp_solution(slates),
    )
    natural = methods[0]
    for result in methods[1:]:
        assert result.route_indices == natural.route_indices
        assert result.union_mask == natural.union_mask
        assert result.total_local_kl == natural.total_local_kl


def test_fixed_cache_ilp_objective_excludes_cached_experts() -> None:
    slates = toy_slates()
    cache = experts_to_mask((6, 7, 8, 9, 10, 11))
    exact = exact_ilp_solution(slates, cache_mask=cache)
    assert exact.union_count == 6
    assert exact.diagnostics is not None
    assert exact.diagnostics["objective"] == 0


def test_slate_cap_is_stable_and_never_drops_natural() -> None:
    subsets = torch.tensor(list(itertools.combinations(range(12), 6)))
    ids = torch.arange(12).view(1, 12)
    weights = torch.linspace(1.0, 0.1, 12).view(1, 12)
    damage = torch.arange(subsets.shape[0], dtype=torch.float32).view(1, -1)
    original_index = int(
        (subsets == torch.arange(6)).all(dim=1).nonzero(as_tuple=False).item()
    )
    damage[0, original_index] = 100.0

    slates = build_candidate_slates(
        ids, weights, damage, subsets, threshold=10.0, cap=4
    )

    assert len(slates[0]) == 4
    assert sum(route.natural for route in slates[0]) == 1
    assert [route.subset_index for route in slates[0] if not route.natural] == [1, 2, 3]


def test_solution_metrics_preserve_exact_union_and_route_change() -> None:
    slates = toy_slates()
    exact = exact_ilp_solution(slates)
    metrics = solution_metrics(slates, exact)
    assert metrics["union_count"] == 6
    assert metrics["changed_token_fraction"] == 1.0
    assert metrics["mean_local_kl"] == (0.0002 + 0.0003 + 0.0004) / 3


def test_highs_control_accepts_machine_epsilon_not_real_gap() -> None:
    diagnostics = {
        "status": 0,
        "success": True,
        "mip_gap": 4.1e-16,
        "objective": 13.000000000000016,
    }
    assert highs_optimal_control(diagnostics, 13)
    assert not highs_optimal_control({**diagnostics, "mip_gap": 1e-5}, 13)
