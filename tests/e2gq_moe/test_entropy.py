import math

import torch

from moe_lab.e2gq_moe.entropy import (
    code_histogram,
    ideal_total_bpp,
    projection_codes,
    shannon_entropy,
)


def test_entropy_uniform_four_symbols_is_two_bits() -> None:
    assert shannon_entropy({-2: 1, -1: 1, 0: 1, 1: 1}) == 2.0


def test_projection_codes_reconstructs_bf16_exactly() -> None:
    scales = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
    codes = torch.tensor([[-2, -1, 0, 1, -2, -1, 0, 1]], dtype=torch.bfloat16)
    expanded = scales[:, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])]
    weight = codes * expanded
    recovered = projection_codes(weight, scales, group_size=4)
    assert code_histogram(recovered) == {-2: 2, -1: 2, 0: 2, 1: 2}


def test_scale_overhead_is_counted() -> None:
    assert math.isclose(ideal_total_bpp({-2: 1, -1: 1, 0: 1, 1: 1}, 1, 128), 2.125)
