from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReductionScheme:
    name: str
    operand_dtype: torch.dtype
    accumulator_dtype: torch.dtype
    topology: str


SCHEMES = (
    ReductionScheme("fp32_sequential", torch.float32, torch.float32, "sequential"),
    ReductionScheme("fp32_tree", torch.float32, torch.float32, "tree"),
    ReductionScheme(
        "bf16_operands_fp32_sequential",
        torch.bfloat16,
        torch.float32,
        "sequential",
    ),
    ReductionScheme(
        "bf16_operands_fp32_tree", torch.bfloat16, torch.float32, "tree"
    ),
    ReductionScheme("bf16_sequential", torch.bfloat16, torch.bfloat16, "sequential"),
    ReductionScheme("bf16_tree", torch.bfloat16, torch.bfloat16, "tree"),
    ReductionScheme("fp16_sequential", torch.float16, torch.float16, "sequential"),
    ReductionScheme("fp16_tree", torch.float16, torch.float16, "tree"),
)


def six_term_permutations() -> torch.Tensor:
    return torch.tensor(list(itertools.permutations(range(6))), dtype=torch.long)


def scheme_by_name(name: str) -> ReductionScheme:
    try:
        return next(scheme for scheme in SCHEMES if scheme.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown reduction scheme: {name}") from error


def _operands(weighted_terms: torch.Tensor, scheme: ReductionScheme) -> torch.Tensor:
    if weighted_terms.ndim != 3 or weighted_terms.shape[1] != 6:
        raise ValueError("weighted terms must have shape [tokens, 6, hidden]")
    if not torch.isfinite(weighted_terms).all():
        raise ValueError("weighted terms must be finite")
    converted = weighted_terms.to(scheme.operand_dtype)
    return converted.to(scheme.accumulator_dtype)


def reduce_permutation_batch(
    weighted_terms: torch.Tensor,
    permutations: torch.Tensor,
    scheme: ReductionScheme | str,
) -> torch.Tensor:
    """Reduce a batch of slot permutations, returning [orders, tokens, hidden]."""

    if isinstance(scheme, str):
        scheme = scheme_by_name(scheme)
    if permutations.ndim != 2 or permutations.shape[1] != 6:
        raise ValueError("permutations must have shape [orders, 6]")
    if permutations.numel() and (
        int(permutations.min()) < 0 or int(permutations.max()) > 5
    ):
        raise ValueError("permutation slots must be in [0, 5]")
    if not all(
        len(set(row)) == 6 for row in permutations.detach().cpu().tolist()
    ):
        raise ValueError("every order must be a six-slot permutation")
    operands = _operands(weighted_terms, scheme)
    order = permutations.to(weighted_terms.device)

    def leaf(position: int) -> torch.Tensor:
        return operands[:, order[:, position], :]

    if scheme.topology == "sequential":
        accumulator = torch.zeros(
            weighted_terms.shape[0],
            permutations.shape[0],
            weighted_terms.shape[2],
            dtype=scheme.accumulator_dtype,
            device=weighted_terms.device,
        )
        for position in range(6):
            accumulator = accumulator + leaf(position)
    elif scheme.topology == "tree":
        left = leaf(0) + leaf(1)
        middle = leaf(2) + leaf(3)
        right = leaf(4) + leaf(5)
        accumulator = (left + middle) + right
    else:
        raise ValueError(f"unknown topology: {scheme.topology}")
    return accumulator.permute(1, 0, 2).contiguous()


def routed_mse_by_order(
    reduced: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if reduced.ndim != 3 or target.ndim != 2 or reduced.shape[1:] != target.shape:
        raise ValueError("reduced [orders,tokens,hidden] must align with target")
    return (reduced.float() - target.float().unsqueeze(0)).square().mean(-1)


def q3_q4_gap_closure(q3_kl: float, q4_kl: float, candidate_kl: float) -> float:
    gap = q3_kl - q4_kl
    if not all(torch.isfinite(torch.tensor([q3_kl, q4_kl, candidate_kl]))):
        raise ValueError("KL values must be finite")
    if gap <= 0:
        return float("nan")
    return (q3_kl - candidate_kl) / gap


def anchored_reduction_candidate(
    teacher: torch.Tensor,
    candidate_routed: torch.Tensor,
    natural_routed: torch.Tensor,
) -> torch.Tensor:
    if teacher.shape != candidate_routed.shape or teacher.shape != natural_routed.shape:
        raise ValueError("teacher and routed states must align")
    delta = (candidate_routed.float() - natural_routed.float()).to(teacher.dtype)
    return teacher + delta


def paired_gap_closure_bootstrap(
    q3_kl: list[float] | np.ndarray,
    q4_kl: list[float] | np.ndarray,
    candidates: dict[str, list[float] | np.ndarray],
    *,
    block_size: int,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, object]:
    q3 = np.asarray(q3_kl, dtype=np.float64)
    q4 = np.asarray(q4_kl, dtype=np.float64)
    candidate_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in candidates.items()
    }
    if q3.ndim != 1 or q3.size == 0 or q3.size % block_size:
        raise ValueError("KL series must form complete sequence blocks")
    if q4.shape != q3.shape or any(values.shape != q3.shape for values in candidate_arrays.values()):
        raise ValueError("all paired KL series must align")
    if not (
        np.isfinite(q3).all()
        and np.isfinite(q4).all()
        and all(np.isfinite(values).all() for values in candidate_arrays.values())
    ):
        raise ValueError("KL series must be finite")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    blocks = q3.size // block_size

    def block_sums(values: np.ndarray) -> np.ndarray:
        return values.reshape(blocks, block_size).sum(axis=1)

    q3_sums = block_sums(q3)
    q4_sums = block_sums(q4)
    candidate_sums = {
        name: block_sums(values) for name, values in candidate_arrays.items()
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, blocks, size=(resamples, blocks))
    q3_sample = q3_sums[indices].sum(axis=1) / (blocks * block_size)
    q4_sample = q4_sums[indices].sum(axis=1) / (blocks * block_size)
    denominator = q3_sample - q4_sample
    positive = denominator > 0
    if not positive.all():
        raise ValueError("bootstrap encountered non-positive Q3-to-Q4 gaps")

    def closure(candidate: np.ndarray) -> np.ndarray:
        sampled = candidate[indices].sum(axis=1) / (blocks * block_size)
        return (q3_sample - sampled) / denominator

    closures = {name: closure(values) for name, values in candidate_sums.items()}

    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, (0.025, 0.975), method="linear")
        return {"low": float(low), "high": float(high)}

    q3_mean = float(q3.mean())
    q4_mean = float(q4.mean())
    point_gap = q3_mean - q4_mean
    point = {
        name: float((q3_mean - values.mean()) / point_gap)
        for name, values in candidate_arrays.items()
    }
    return {
        "method": "paired sequence-block percentile bootstrap with replacement",
        "seed": seed,
        "resamples": resamples,
        "sampling_units": blocks,
        "block_size": block_size,
        "point_gap": point_gap,
        "point_closure": point,
        "intervals_95": {name: interval(values) for name, values in closures.items()},
        "probability_ge_0_10": {
            name: float((values >= 0.10).mean()) for name, values in closures.items()
        },
        "probability_ge_0_20": {
            name: float((values >= 0.20).mean()) for name, values in closures.items()
        },
        "raw": {name: values.tolist() for name, values in closures.items()},
    }
