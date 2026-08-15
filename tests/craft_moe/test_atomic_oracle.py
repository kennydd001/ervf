from __future__ import annotations

import math

import torch

from moe_lab.craft_moe.atomic import (
    atomic_selector_masks,
    delta_patched_hidden,
    global_tile_topk_mask,
    global_topk_mask,
    per_expert_topk_mask,
    reconstruct_weighted_atoms,
    relative_routed_l2,
    retained_count,
    support_known_accounting,
)


def test_exact_full_atom_control_and_delta_patch() -> None:
    generator = torch.Generator().manual_seed(11)
    activations = torch.randn(3, 2, 4, generator=generator)
    weights = torch.rand(3, 2, generator=generator)
    columns = torch.randn(3, 2, 4, 5, generator=generator)
    full = torch.ones_like(activations, dtype=torch.bool)
    teacher = torch.randn(3, 5, generator=generator, dtype=torch.bfloat16)

    reference = reconstruct_weighted_atoms(activations, weights, columns, full)
    direct = torch.zeros(3, 5, dtype=torch.float64)
    for token in range(3):
        for expert in range(2):
            for atom in range(4):
                direct[token] += (
                    weights[token, expert]
                    * activations[token, expert, atom]
                    * columns[token, expert, atom]
                ).double()

    assert torch.allclose(reference, direct, atol=1e-6)
    routed = reference.to(torch.bfloat16)
    assert torch.equal(delta_patched_hidden(teacher, routed, routed), teacher)
    assert torch.equal(relative_routed_l2(routed, routed), torch.zeros(3))


def test_stable_per_expert_and_global_topk_counts() -> None:
    scores = torch.ones(2, 3, 4)

    per_expert = per_expert_topk_mask(scores, 0.25)
    globally = global_topk_mask(scores, 0.25)

    assert retained_count(12, 0.25) == 3
    assert torch.equal(per_expert.sum(dim=2), torch.ones(2, 3, dtype=torch.long))
    assert per_expert[:, :, 0].all()
    assert torch.equal(globally.sum(dim=(1, 2)), torch.full((2,), 3))
    assert globally[:, 0, :3].all()
    assert not globally[:, 1:].any()


def test_tile_selector_keeps_whole_tiles_and_uses_squared_norm_score() -> None:
    score = torch.zeros(1, 2, 8)
    score[0, 1, 4:8] = 2.0
    score[0, 0, 0:4] = 1.0

    mask = global_tile_topk_mask(score, 0.25, tile_size=4)

    assert mask.sum().item() == 4
    assert mask[0, 1, 4:8].all()
    assert not mask[0, 1, :4].any()


def test_all_fixed_selectors_have_preregistered_counts() -> None:
    generator = torch.Generator().manual_seed(17)
    activations = torch.randn(2, 6, 64, generator=generator)
    router_weights = torch.rand(2, 6, generator=generator)
    down_norms = torch.rand(2, 6, 64, generator=generator)

    masks = atomic_selector_masks(
        activations, router_weights, down_norms, fraction=0.25
    )

    for name in ("per_expert_activation", "per_expert_contribution"):
        assert torch.equal(
            masks[name].sum(dim=2), torch.full((2, 6), 16, dtype=torch.long)
        )
    assert torch.equal(
        masks["global_contribution"].sum(dim=(1, 2)),
        torch.full((2,), 96, dtype=torch.long),
    )
    expected = {
        "tile16_contribution": math.ceil(6 * 4 * 0.25) * 16,
        "tile32_contribution": math.ceil(6 * 2 * 0.25) * 32,
        "tile64_contribution": math.ceil(6 * 1 * 0.25) * 64,
    }
    for name, count in expected.items():
        assert torch.equal(
            masks[name].sum(dim=(1, 2)),
            torch.full((2,), count, dtype=torch.long),
        )


def test_accounting_exposes_down_page_floor_without_calling_it_runtime() -> None:
    counts = torch.full((2, 6), 352, dtype=torch.long)

    accounting = support_known_accounting(
        counts, atoms_per_expert=1408, hidden_size=2048
    )

    assert accounting["retained_atoms"] == [2112, 2112]
    assert accounting["ideal_weight_byte_fraction"] == [0.25, 0.25]
    expected_page_fraction = (2 * 352 + 1408) / (3 * 1408)
    assert math.isclose(
        accounting["tensor_local_page_byte_fraction"][0],
        expected_page_fraction,
    )
    assert accounting["assumption"].startswith("support known")


def test_shape_validation_rejects_mismatched_router_weights() -> None:
    activations = torch.zeros(2, 6, 64)
    try:
        atomic_selector_masks(
            activations,
            torch.zeros(2, 5),
            torch.zeros_like(activations),
            0.25,
        )
    except ValueError as error:
        assert "router_weights" in str(error)
    else:
        raise AssertionError("expected a router-weight shape validation error")
