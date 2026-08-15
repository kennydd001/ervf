import torch

from moe_lab.quantization import fake_quantize_symmetric_per_row_, packed_quantized_bytes


def test_fake_quantization_is_per_row() -> None:
    weight = torch.tensor([[1.0, -0.5], [10.0, -5.0]])
    scales = fake_quantize_symmetric_per_row_(weight, 2)
    assert scales == 2
    assert torch.equal(weight, torch.tensor([[1.0, -0.0], [10.0, -0.0]]))


def test_packed_bytes_include_bf16_scales() -> None:
    assert packed_quantized_bytes(9, 4, 2) == 9


def test_binary_quantization_uses_mean_absolute_scale() -> None:
    weight = torch.tensor([[1.0, -3.0]])
    fake_quantize_symmetric_per_row_(weight, 1)
    assert torch.equal(weight, torch.tensor([[2.0, -2.0]]))
