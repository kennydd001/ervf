from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from moe_lab.rsiv_moe.subspace import (
    append_residual_direction,
    cold_byte_fraction,
    energy_rank,
    fit_origin_subspace,
    image_storage_elements,
    online_fault_curve,
    relative_residual_ratio,
    select_single_evaluation_candidate,
    select_validation_candidate,
)


def test_operator_image_identity_is_exact_on_span() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(9, 6, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(6, 3, generator=generator, dtype=torch.float64))
    coefficients = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    x = coefficients @ q.T
    direct = x @ weight.T
    image = weight @ q
    cached = (x @ q) @ image.T
    torch.testing.assert_close(cached, direct, atol=1e-12, rtol=1e-12)


def test_full_rank_swiglu_images_reconstruct_direct_operator() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    gate = torch.randn(8, 5, generator=generator, dtype=torch.float64)
    up = torch.randn(8, 5, generator=generator, dtype=torch.float64)
    down = torch.randn(5, 8, generator=generator, dtype=torch.float64)
    q = fit_origin_subspace(x).basis
    aq, bq = gate @ q, up @ q
    coordinates = x @ q
    z_cached = F.silu(coordinates @ aq.T) * (coordinates @ bq.T)
    z_direct = F.silu(x @ gate.T) * (x @ up.T)
    p = fit_origin_subspace(z_direct).basis
    cp = down @ p
    y_cached = (z_cached @ p) @ cp.T
    y_direct = z_direct @ down.T
    torch.testing.assert_close(z_cached, z_direct, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(y_cached, y_direct, atol=1e-10, rtol=1e-10)


def test_fit_is_uncentred_and_reports_energy_rank() -> None:
    samples = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    fit = fit_origin_subspace(samples, rank_cap=1)
    assert fit.stored_rank == 2
    assert fit.basis.shape == (2, 1)
    assert energy_rank(fit.singular_values, 0.79) == 1
    assert energy_rank(fit.singular_values, 0.81) == 2
    assert fit.reconstruction_relative_l2 == pytest.approx(1 / math.sqrt(5))


def test_residual_ratio_and_dgks_online_growth() -> None:
    samples = torch.eye(3, dtype=torch.float64)
    empty = torch.empty(3, 0, dtype=torch.float64)
    torch.testing.assert_close(
        relative_residual_ratio(samples, empty),
        torch.ones(3, dtype=torch.float64),
    )
    basis, ratio, added = append_residual_direction(empty, samples[0])
    assert added and ratio == pytest.approx(1.0)
    basis, ratio, added = append_residual_direction(basis, samples[0])
    assert not added and ratio == pytest.approx(0.0, abs=1e-12)
    curve = online_fault_curve(samples, threshold=1e-9, rank_cap=2)
    assert curve["misses"] == [True, True, True]
    assert curve["ranks_after"] == [1, 2, 2]
    assert curve["rank_additions"] == 2


def test_partial_cold_byte_accounting() -> None:
    x_miss = torch.tensor([False, True, False, True])
    z_miss = torch.tensor([False, False, True, True])
    fractions = cold_byte_fraction(x_miss, z_miss)
    torch.testing.assert_close(
        fractions,
        torch.tensor([0.0, 2 / 3, 1 / 3, 1.0], dtype=torch.float64),
    )
    assert float(fractions.mean()) == pytest.approx(0.5)


def test_expert_count_cancellation_bound() -> None:
    counts = [3, 0, 2, 1]
    input_ranks = [3, 0, 2, 1]
    intermediate_ranks = [2, 0, 2, 1]
    d, m = 5, 7
    elements = image_storage_elements(d, m, input_ranks, intermediate_ranks)
    bound = (2 * d + 3 * m) * sum(counts)
    assert elements <= bound
    assert sum(input_ranks) <= sum(counts)
    assert sum(intermediate_ranks) <= sum(counts)


def _candidate(rank: int, threshold: float, fast: float, reduction: float) -> dict:
    return {
        "rank_cap": rank,
        "threshold": threshold,
        "offline_double_fast_fraction": fast,
        "offline_cold_byte_reduction": reduction,
        "causal_double_fast_fraction": fast,
        "causal_cold_byte_reduction": reduction,
    }


def test_validation_selection_prefers_safest_then_smallest_primary() -> None:
    selected = select_validation_candidate(
        [
            _candidate(8, 0.02, 0.93, 11.0),
            _candidate(4, 0.02, 0.93, 11.0),
            _candidate(32, 0.01, 0.92, 10.0),
        ]
    )
    assert selected["selection_kind"] == "primary_gate_pass"
    assert selected["threshold"] == 0.01
    assert selected["rank_cap"] == 32


def test_validation_selection_locks_diagnostic_without_test_fields() -> None:
    selected = select_validation_candidate(
        [_candidate(4, 0.001, 0.5, 2.0), _candidate(32, 0.1, 0.9, 8.0)]
    )
    assert selected["selection_kind"] == "diagnostic_validation_failure"
    assert selected["rank_cap"] == 32
    contaminated = _candidate(32, 0.1, 0.9, 8.0)
    contaminated["test_fast_fraction"] = 1.0
    with pytest.raises(ValueError, match="test"):
        select_validation_candidate([contaminated])


def test_long_prefix_selection_uses_same_validation_discipline() -> None:
    rows = [
        {
            "rank_cap": 32,
            "threshold": 0.01,
            "double_gate_fast_fraction": 0.92,
            "cold_byte_reduction": 10.0,
        },
        {
            "rank_cap": 8,
            "threshold": 0.02,
            "double_gate_fast_fraction": 0.99,
            "cold_byte_reduction": 20.0,
        },
    ]
    selected = select_single_evaluation_candidate(rows)
    assert selected["selection_kind"] == "primary_gate_pass"
    assert selected["rank_cap"] == 32
    assert selected["threshold"] == 0.01
