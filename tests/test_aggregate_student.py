import torch

from moe_lab.aggregate_student import (
    AggregateStudent,
    ResidualBasisStudent,
    dense_router_features,
)


def test_dense_router_features_preserve_sparse_weights() -> None:
    ids = torch.tensor([[1, 3], [0, 2]])
    weights = torch.tensor([[0.2, 0.4], [0.1, 0.7]])
    dense = dense_router_features(ids, weights, 4)
    assert torch.allclose(dense, torch.tensor([[0.0, 0.2, 0.0, 0.4], [0.1, 0.0, 0.7, 0.0]]))


def test_conditioned_student_shape_and_parameter_count() -> None:
    model = AggregateStudent(8, 4, 3, route_conditioned=True)
    output = model(
        torch.randn(2, 8),
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[0.4, 0.3], [0.5, 0.2]]),
    )
    assert output.shape == (2, 8)
    assert model.parameter_count == (11 * 4 * 2) + (4 * 8)


def test_residual_basis_student_shape() -> None:
    model = ResidualBasisStudent(8, 4, 3, adapter_rank=2)
    output = model(
        torch.randn(2, 8),
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[0.4, 0.3], [0.5, 0.2]]),
    )
    assert output.shape == (2, 8)


def test_bf16_residual_student_accepts_float_router_weights() -> None:
    model = ResidualBasisStudent(8, 4, 3, adapter_rank=2).to(torch.bfloat16)
    output = model(
        torch.randn(2, 8, dtype=torch.bfloat16),
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[0.4, 0.3], [0.5, 0.2]], dtype=torch.float32),
    )
    assert output.dtype == torch.bfloat16
