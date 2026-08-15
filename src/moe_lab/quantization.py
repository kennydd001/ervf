from __future__ import annotations

import torch


@torch.inference_mode()
def fake_quantize_symmetric_per_row_(weight: torch.Tensor, bits: int) -> int:
    """In-place fake quantization; returns the number of stored row scales."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be between 1 and 8")
    if bits == 1:
        rows = weight.reshape(weight.shape[0], -1)
        scale = rows.float().abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
        signs = torch.where(rows.float() >= 0, 1.0, -1.0)
        rows.copy_((signs * scale).to(rows.dtype))
        return rows.shape[0]
    qmax = (1 << (bits - 1)) - 1
    original_shape = weight.shape
    rows = weight.reshape(original_shape[0], -1)
    scale = rows.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    quantized = torch.round(rows.float() / scale).clamp(-qmax, qmax)
    rows.copy_((quantized * scale).to(rows.dtype))
    return rows.shape[0]


def packed_quantized_bytes(parameter_count: int, bits: int, scale_count: int) -> int:
    weight_bytes = (parameter_count * bits + 7) // 8
    return weight_bytes + scale_count * 2
