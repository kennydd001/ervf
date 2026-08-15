from __future__ import annotations

import math

import torch

from moe_lab.craft_moe.sketchgate import (
    choose_validation_configuration,
    delta_patch,
    dequantize_rows_int8,
    exact_schedule_mean_kl,
    false_negative_rate,
    high_damage_mask,
    mask_indices,
    mix_selected_outputs,
    nested_quantized_sketch_scores,
    oracle_recovery,
    probe_bank,
    quantize_rows_int8,
    sketch_metadata_accounting,
    stable_top_fraction_mask,
)


def test_exact_original_delta_and_output_controls() -> None:
    generator = torch.Generator().manual_seed(3)
    base = torch.randn(4, 2, 5, generator=generator, dtype=torch.bfloat16)
    upgraded = torch.randn(4, 2, 5, generator=generator, dtype=torch.bfloat16)
    teacher = torch.randn(4, 5, generator=generator, dtype=torch.bfloat16)
    routed = torch.randn(4, 5, generator=generator, dtype=torch.bfloat16)

    assert torch.equal(
        mix_selected_outputs(base, upgraded, torch.zeros(4, 2, dtype=torch.bool)),
        base,
    )
    assert torch.equal(
        mix_selected_outputs(base, upgraded, torch.ones(4, 2, dtype=torch.bool)),
        upgraded,
    )
    assert torch.equal(delta_patch(teacher, routed, routed), teacher)


def test_probe_banks_are_deterministic_nested_and_distribution_specific() -> None:
    gaussian = probe_bank("gaussian", 17, 8, 11)
    repeated = probe_bank("gaussian", 17, 8, 11)
    shorter = probe_bank("gaussian", 17, 4, 11)
    rademacher = probe_bank("rademacher", 17, 8, 11)

    assert torch.equal(gaussian, repeated)
    assert torch.equal(gaussian[:4], shorter)
    assert set(rademacher.unique().tolist()) == {-1.0, 1.0}
    assert not torch.equal(gaussian, rademacher)


def test_int8_rows_handle_zero_and_respect_half_step_error() -> None:
    rows = torch.tensor([[0.0, 0.0, 0.0], [-2.0, -0.2, 1.0]])
    quantized, scale = quantize_rows_int8(rows)
    restored = dequantize_rows_int8(quantized, scale)

    assert quantized.dtype is torch.int8
    assert scale.dtype is torch.float16
    assert torch.equal(restored[0], rows[0])
    assert (restored[1] - rows[1]).abs().max() <= scale[1].float() / 2 + 1e-6


def test_quantized_sketch_matches_manual_prefix_formula() -> None:
    activations = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    residual = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    probes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    weights = torch.tensor([0.5, 2.0])

    scores, diagnostics = nested_quantized_sketch_scores(
        activations, residual, probes, weights, (1, 2)
    )
    syndrome = probes @ residual
    q, scale = quantize_rows_int8(syndrome)
    projected = activations @ dequantize_rows_int8(q, scale).T
    expected_one = projected[:, 0].square() * weights.square()
    expected_two = projected.square().mean(dim=1) * weights.square()

    assert torch.allclose(scores[1], expected_one)
    assert torch.allclose(scores[2], expected_two)
    assert diagnostics["syndrome_int8_nrmse"] >= 0


def test_stable_schedule_masks_and_exact_damage_lookup() -> None:
    scores = torch.ones(2, 4)
    selected = stable_top_fraction_mask(scores, 0.25)
    assert torch.equal(
        selected,
        torch.tensor(
            [[True, True, False, False], [False, False, False, False]]
        ),
    )
    assert torch.equal(mask_indices(selected), torch.tensor([3, 0]))
    damage = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    assert exact_schedule_mean_kl(selected, damage) == (3 + 16) / 2


def test_high_damage_and_false_negative_are_fixed_global_events() -> None:
    benefit = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    high, description = high_damage_mask(benefit, 0.10)
    assert description["selected_high_damage_count"] == 2
    assert high.reshape(-1)[-2:].all()
    selected = high.clone()
    selected.reshape(-1)[-1] = False
    assert false_negative_rate(selected, high) == 0.5


def test_recovery_and_metadata_gates() -> None:
    assert oracle_recovery(0.02, 0.006, 0.004) == 0.875
    accounting = sketch_metadata_accounting(64)
    assert accounting["passes_lt_0_1_bit"]
    assert accounting["effective_bits_per_original_weight"] < 0.084


def test_validation_selection_never_selects_a_seed() -> None:
    rows = []
    for distribution in ("gaussian", "rademacher"):
        for rank in (4, 8):
            for seed in range(5):
                rows.append(
                    {
                        "distribution": distribution,
                        "rank": rank,
                        "seed": seed,
                        "oracle_recovery": 0.81 if rank == 8 else 0.79,
                        "high_damage_false_negative_rate": 0.0,
                    }
                )
    choice = choose_validation_configuration(rows)
    assert choice["qualified_on_validation"]
    assert choice["selected_rank"] == 8
    assert choice["selected_distribution"] == "rademacher"
    assert "selected_seed" not in choice
    assert math.isclose(choice["selected_summary"]["minimum_recovery"], 0.81)
