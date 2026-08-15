from __future__ import annotations

import math
from collections.abc import Mapping

import torch


CODEBOOK = (-2, -1, 0, 1)


def shannon_entropy(counts: Mapping[int, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("entropy requires at least one symbol")
    return -sum(
        (int(value) / total) * math.log2(int(value) / total)
        for value in counts.values()
        if int(value) > 0
    )


def projection_codes(
    weight: torch.Tensor, scales: torch.Tensor, *, group_size: int = 128
) -> torch.Tensor:
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight and scales must be matrices")
    expected_groups = math.ceil(weight.shape[1] / group_size)
    if scales.shape != (weight.shape[0], expected_groups):
        raise ValueError(
            f"scale shape {tuple(scales.shape)} != {(weight.shape[0], expected_groups)}"
        )
    group_ids = torch.arange(weight.shape[1]) // group_size
    expanded = scales[:, group_ids].to(dtype=torch.float32)
    codes = torch.round(weight.float() / expanded).to(torch.int8)
    if not bool(torch.isin(codes, torch.tensor(CODEBOOK, dtype=torch.int8)).all()):
        raise ValueError("projection contains codes outside the pinned GPTQ codebook")
    reconstructed = (codes.to(torch.float32) * expanded).to(weight.dtype)
    if not torch.equal(reconstructed, weight):
        maximum = float((reconstructed.float() - weight.float()).abs().max())
        raise ValueError(f"code/scale reconstruction is not bit-exact; max error {maximum}")
    return codes


def code_histogram(codes: torch.Tensor) -> dict[int, int]:
    return {symbol: int((codes == symbol).sum()) for symbol in CODEBOOK}


def ideal_total_bpp(counts: Mapping[int, int], scale_count: int, weight_count: int) -> float:
    return shannon_entropy(counts) + 16 * scale_count / weight_count


def multinomial_bits(counts: Mapping[int, int]) -> int:
    total = sum(int(value) for value in counts.values())
    return math.ceil(
        (math.lgamma(total + 1) - sum(math.lgamma(int(value) + 1) for value in counts.values()))
        / math.log(2)
    )

