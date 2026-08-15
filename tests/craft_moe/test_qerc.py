from __future__ import annotations

import torch

from moe_lab.craft_moe.qerc import (
    aggregate_error_reduction,
    apply_output_gains,
    fit_corouted_row_gains,
    routed_design_matrix,
    routed_output,
    scale_layout_accounting,
    weighted_error_decomposition,
)


def test_exact_original_and_identity_gain_controls() -> None:
    generator = torch.Generator().manual_seed(5)
    selected = torch.randn(4, 2, 3, generator=generator, dtype=torch.bfloat16)
    weights = torch.rand(4, 2, generator=generator)
    ids = torch.tensor([[0, 1]] * 4)
    gains = torch.ones(2, 3)

    gained = apply_output_gains(selected, ids, gains)
    assert torch.equal(gained, selected)
    assert torch.equal(routed_output(selected, weights), routed_output(gained, weights))
    decomposition = weighted_error_decomposition(selected, selected, weights)
    assert decomposition["diagonal_energy_sum"] == 0.0
    assert decomposition["aggregate_energy_sum"] == 0.0


def test_cross_term_detects_perfect_error_cancellation() -> None:
    bf16 = torch.zeros(1, 2, 2)
    q3 = torch.tensor([[[1.0, -2.0], [-1.0, 2.0]]])
    weights = torch.ones(1, 2)

    result = weighted_error_decomposition(bf16, q3, weights)

    assert result["diagonal_energy_sum"] == 10.0
    assert result["aggregate_energy_sum"] == 0.0
    assert result["cross_term_sum"] == -10.0
    assert result["global_cancellation_fraction"] == 1.0


def test_corouted_row_fit_recovers_known_output_scales() -> None:
    generator = torch.Generator().manual_seed(9)
    tokens, slots, hidden = 32, 2, 4
    ids = torch.tensor([[0, 1]] * tokens)
    weights = torch.ones(tokens, slots)
    q3 = torch.randn(tokens, slots, hidden, generator=generator)
    true_gains = torch.tensor(
        [[0.8, 0.9, 1.1, 1.2], [1.2, 1.1, 0.9, 0.8]]
    )
    target_selected = q3 * true_gains[ids]
    target = routed_output(target_selected, weights).float()
    design = routed_design_matrix(q3, ids, weights, experts=2)

    fitted, diagnostics = fit_corouted_row_gains(
        design, target, alpha=1e-6, lower=0.5, upper=1.5
    )
    candidate = routed_output(apply_output_gains(q3, ids, fitted), weights)

    assert torch.allclose(candidate.float(), target, atol=2e-3)
    assert diagnostics["finite_before_clamp"]


def test_error_reduction_reports_fraction_not_percent() -> None:
    reference = torch.zeros(2, 3)
    baseline = torch.ones(2, 3)
    candidate = torch.full((2, 3), 0.5)
    result = aggregate_error_reduction(reference, baseline, candidate)
    assert result["error_reduction"] == 0.75


def test_scale_gain_replaces_existing_values_without_layout_cost() -> None:
    accounting = scale_layout_accounting(experts=3, hidden_size=5)
    assert accounting["existing_down_scale_values"] == 15
    assert accounting["candidate_down_scale_values"] == 15
    assert accounting["additional_bytes"] == 0
    assert not accounting["integer_codes_changed"]
    assert not accounting["tensor_shape_changed"]
