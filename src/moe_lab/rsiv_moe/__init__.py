"""RSIV-MoE / GhostWeights research primitives."""

from .subspace import (
    BasisFit,
    append_residual_direction,
    cold_byte_fraction,
    energy_rank,
    fit_origin_subspace,
    image_storage_elements,
    online_fault_curve,
    relative_residual_ratio,
    select_validation_candidate,
)
from .qwen_capture import CapturedQwenMoeInvocations, QwenMoeInvocationCapture
from .qwen_stream import load_qwen_decoder_layer, load_token_embeddings

__all__ = [
    "BasisFit",
    "append_residual_direction",
    "cold_byte_fraction",
    "energy_rank",
    "fit_origin_subspace",
    "image_storage_elements",
    "online_fault_curve",
    "relative_residual_ratio",
    "select_validation_candidate",
    "CapturedQwenMoeInvocations",
    "QwenMoeInvocationCapture",
    "load_qwen_decoder_layer",
    "load_token_embeddings",
]
