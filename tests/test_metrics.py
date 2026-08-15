import pytest
import torch

from moe_lab.metrics import (
    mean_cosine_similarity,
    normalized_rmse,
    regression_metrics,
    topk_overlap,
)


def test_normalized_rmse_identity() -> None:
    x = torch.tensor([[1.0, -2.0]])
    assert normalized_rmse(x, x) == pytest.approx(0.0)


def test_cosine_identity() -> None:
    x = torch.tensor([[1.0, -2.0]])
    assert mean_cosine_similarity(x, x) == pytest.approx(1.0)


def test_topk_overlap() -> None:
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[2, 5], [3, 4]])
    assert topk_overlap(a, b) == pytest.approx(0.75)


def test_topk_overlap_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        topk_overlap(torch.tensor([[1, 2]]), torch.tensor([[1]]))


def test_regression_metrics_identity() -> None:
    x = torch.tensor([[1.0, 2.0]])
    result = regression_metrics(x, x)
    assert result == pytest.approx(
        {"nrmse": 0.0, "cosine": 1.0, "mae": 0.0, "max_abs_error": 0.0}
    )
