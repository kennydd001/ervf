from __future__ import annotations

import torch

from moe_lab.craft_moe.reduction_order import (
    SCHEMES,
    anchored_reduction_candidate,
    paired_gap_closure_bootstrap,
    q3_q4_gap_closure,
    reduce_permutation_batch,
    routed_mse_by_order,
    six_term_permutations,
)


def test_six_term_permutations_are_complete_and_lexicographic() -> None:
    permutations = six_term_permutations()
    assert permutations.shape == (720, 6)
    assert torch.equal(permutations[0], torch.arange(6))
    assert torch.equal(permutations[-1], torch.arange(5, -1, -1))
    assert len({tuple(row) for row in permutations.tolist()}) == 720


def test_fp32_sequential_matches_literal_addition() -> None:
    terms = torch.randn(3, 6, 11)
    order = torch.tensor([[5, 2, 4, 1, 3, 0]])
    reduced = reduce_permutation_batch(terms, order, "fp32_sequential")[0]
    expected = torch.zeros(3, 11)
    for slot in order[0].tolist():
        expected = expected + terms[:, slot]
    assert torch.equal(reduced, expected)


def test_tree_matches_registered_parenthesization() -> None:
    terms = torch.randn(2, 6, 7)
    order = torch.tensor([[3, 1, 4, 0, 5, 2]])
    reduced = reduce_permutation_batch(terms, order, "bf16_tree")[0]
    operands = terms.to(torch.bfloat16)
    expected = (
        (operands[:, 3] + operands[:, 1])
        + (operands[:, 4] + operands[:, 0])
    ) + (operands[:, 5] + operands[:, 2])
    assert torch.equal(reduced, expected)


def test_bf16_operands_promoted_to_fp32_can_be_order_invariant() -> None:
    base = torch.tensor([1.0, -0.5, 0.25, 0.125, -0.0625, 0.03125])
    terms = base.view(1, 6, 1).expand(2, 6, 9).contiguous()
    orders = torch.stack((torch.arange(6), torch.arange(5, -1, -1)))
    reduced = reduce_permutation_batch(
        terms, orders, "bf16_operands_fp32_sequential"
    )
    assert torch.equal(reduced[0], reduced[1])


def test_bf16_accumulation_exposes_order_effect() -> None:
    values = torch.tensor([256.0, 1.0, -256.0, 1.0, 1.0, 1.0])
    terms = values.view(1, 6, 1)
    orders = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 2, 1, 3, 4, 5]])
    reduced = reduce_permutation_batch(terms, orders, "bf16_sequential")
    assert not torch.equal(reduced[0], reduced[1])


def test_routed_mse_has_order_and_token_axes() -> None:
    reduced = torch.tensor([[[1.0, 2.0]], [[2.0, 2.0]]])
    target = torch.tensor([[1.0, 1.0]])
    mse = routed_mse_by_order(reduced, target)
    assert torch.allclose(mse, torch.tensor([[0.5], [1.0]]))


def test_gap_closure_definition() -> None:
    assert q3_q4_gap_closure(0.5, 0.1, 0.3) == 0.5


def test_original_reduction_anchor_is_bit_exact() -> None:
    teacher = torch.randn(5, 17, dtype=torch.bfloat16)
    natural = torch.randn_like(teacher)
    assert torch.equal(
        anchored_reduction_candidate(teacher, natural, natural), teacher
    )


def test_scheme_names_and_tie_order_are_fixed() -> None:
    assert [scheme.name for scheme in SCHEMES] == [
        "fp32_sequential",
        "fp32_tree",
        "bf16_operands_fp32_sequential",
        "bf16_operands_fp32_tree",
        "bf16_sequential",
        "bf16_tree",
        "fp16_sequential",
        "fp16_tree",
    ]


def test_paired_gap_bootstrap_reconciles_point_closure() -> None:
    result = paired_gap_closure_bootstrap(
        [0.5, 0.5, 0.7, 0.7],
        [0.1, 0.1, 0.2, 0.2],
        {"fixed": [0.3, 0.3, 0.45, 0.45]},
        block_size=2,
        seed=9,
        resamples=100,
    )
    assert abs(result["point_closure"]["fixed"] - 0.5) < 1e-15
    assert result["sampling_units"] == 2
    assert len(result["raw"]["fixed"]) == 100
