from __future__ import annotations

import torch


def normalized_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    error = torch.mean((prediction.float() - target.float()) ** 2).sqrt()
    scale = torch.mean(target.float() ** 2).sqrt().clamp_min(1e-12)
    return float((error / scale).item())


def mean_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            prediction.float(), target.float(), dim=-1, eps=1e-12
        )
        .mean()
        .item()
    )


def topk_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("top-k tensors must have identical [tokens, k] shapes")
    overlaps = []
    for left, right in zip(a.tolist(), b.tolist(), strict=True):
        overlaps.append(len(set(left) & set(right)) / len(left))
    return float(sum(overlaps) / len(overlaps))


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction_f = prediction.float()
    target_f = target.float()
    difference = prediction_f - target_f
    return {
        "nrmse": normalized_rmse(prediction_f, target_f),
        "cosine": mean_cosine_similarity(prediction_f, target_f),
        "mae": float(difference.abs().mean().item()),
        "max_abs_error": float(difference.abs().max().item()),
    }
