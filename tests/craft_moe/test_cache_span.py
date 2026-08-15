from __future__ import annotations

import torch

from moe_lab.craft_moe.cache_span import (
    anchored_candidate,
    choose_lowest_mse_candidate,
    fit_span,
    nonempty_subsets,
    omp_cached_order,
    paired_load_bootstrap,
    simulate_mass_budget_trace,
)


def test_nonempty_subsets_are_complete_and_ordered() -> None:
    assert nonempty_subsets([1, 3, 5]) == [
        (1,),
        (3,),
        (5,),
        (1, 3),
        (1, 5),
        (3, 5),
        (1, 3, 5),
    ]


def test_mass_budget_trace_retains_exact_pre_touch_cache() -> None:
    ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    probabilities = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).repeat(2, 1)
    logits = probabilities.log()
    trace = simulate_mass_budget_trace(
        ids,
        probabilities,
        logits,
        capacity=2,
        delta_average=1.0,
        block_size=2,
        top_k=2,
        top_j=2,
        delta=0.0,
    )
    assert torch.equal(trace["routes"], torch.tensor([[0, 1], [0, 1]]))
    assert not trace["cache_before"][0].any()
    assert torch.equal(trace["cache_before"][1], torch.tensor([True, True, False, False]))
    assert torch.equal(trace["miss_mask"], torch.tensor([[True, True], [False, False]]))
    assert trace["expert_loads"] == 2


def test_miss_mask_observes_intra_token_lru_eviction_order() -> None:
    ids = torch.tensor([[0, 1, 2, 3], [2, 0, 1, 3]])
    probabilities = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]]
    )
    logits = probabilities.log()
    trace = simulate_mass_budget_trace(
        ids,
        probabilities,
        logits,
        capacity=2,
        delta_average=1.0,
        block_size=2,
        top_k=2,
        top_j=2,
        delta=0.0,
    )
    assert torch.equal(trace["routes"], torch.tensor([[0, 1], [2, 0]]))
    assert bool(trace["cache_before"][1, 0])
    assert torch.equal(trace["miss_mask"][1], torch.tensor([True, True]))
    assert trace["expert_loads"] == 4


def test_ridge_reconstructs_a_well_conditioned_span() -> None:
    basis = torch.eye(4, 2, dtype=torch.float64)
    target = basis @ torch.tensor([0.25, -0.5], dtype=torch.float64)
    fit = fit_span(basis, target, "ridge", ridge_relative=1e-12)
    assert torch.allclose(fit.prediction, target, atol=1e-10, rtol=0)
    assert fit.normalized_squared_error < 1e-18


def test_nnls_enforces_nonnegative_coefficients() -> None:
    basis = torch.eye(2, dtype=torch.float64)
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)
    fit = fit_span(basis, target, "nnls")
    assert torch.equal(fit.coefficients, torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.equal(fit.prediction, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_bounded_fit_respects_registered_bounds() -> None:
    basis = torch.eye(2, dtype=torch.float64)
    target = torch.tensor([3.0, -4.0], dtype=torch.float64)
    fit = fit_span(basis, target, "bounded", coefficient_bound=1.0)
    assert torch.all(fit.coefficients <= 1.0 + 1e-9)
    assert torch.all(fit.coefficients >= -1.0 - 1e-9)
    assert torch.allclose(fit.coefficients, torch.tensor([1.0, -1.0], dtype=torch.float64), atol=1e-8)


def test_omp_selects_the_correlated_cached_output_first() -> None:
    base = torch.empty(3, 0)
    cached = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    target = torch.tensor([2.0, 0.1, 0.0])
    assert omp_cached_order(base, cached, target, max_extra=2)[0] == 1


def test_original_anchor_is_bit_exact() -> None:
    teacher = torch.randn(7, 13, dtype=torch.bfloat16)
    routed = torch.randn(7, 13, dtype=torch.bfloat16)
    control = anchored_candidate(teacher, routed, routed)
    assert torch.equal(control, teacher)


def test_candidate_tie_breaks_on_compute_then_ids() -> None:
    candidates = [
        {
            "target_squared_error": 1.0,
            "extra_computations": 2,
            "reconstructed_expert_ids": [1],
        },
        {
            "target_squared_error": 1.0,
            "extra_computations": 1,
            "reconstructed_expert_ids": [3],
        },
        {
            "target_squared_error": 1.0,
            "extra_computations": 1,
            "reconstructed_expert_ids": [2],
        },
    ]
    assert choose_lowest_mse_candidate(candidates)["reconstructed_expert_ids"] == [2]


def test_paired_load_bootstrap_reconciles_point_estimates() -> None:
    result = paired_load_bootstrap(
        [10, 20], [4, 8], [3, 6], seed=7, resamples=100
    )
    assert result["point_estimates"] == {
        "primary_miss_reduction_fraction": 0.4,
        "zero_fill_miss_reduction_fraction": 0.3,
        "span_uplift_fraction": 0.1,
    }
    assert result["sampling_units"] == 2
    assert len(result["raw"]["span_uplift_fraction"]) == 100
