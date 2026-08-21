"""Static-scale FP8 activation quantization for Ornith projection kernels."""
from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
#include <cuda_fp8.h>

extern "C" __global__ void ornith_static_e4m3_quantize(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    const int count,
    const float inverse_scale)
{
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (index >= count) return;
    const __nv_fp8_e4m3 value(input[index] * inverse_scale);
    output[index] = *reinterpret_cast<const unsigned char*>(&value);
}
"""


class StaticFP8H4Quantizer:
    """Quantize FP32 H4 activations with the checkpoint's static input scale."""

    def __init__(self) -> None:
        import cupy as cp

        self.cp = cp
        self.module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++14",),
            name_expressions=("ornith_static_e4m3_quantize",),
        )
        self.function = self.module.get_function("ornith_static_e4m3_quantize")

    def quantize(self, input4, output4, input_scale: float) -> None:
        if input4.dtype != self.cp.float32 or output4.dtype != self.cp.uint8:
            raise TypeError("expected FP32 input and uint8 E4M3 output")
        if input4.shape != output4.shape:
            raise ValueError(f"shape mismatch: {input4.shape} != {output4.shape}")
        scale = float(input_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"input_scale must be finite and positive, got {scale}")
        count = int(input4.size)
        self.function(
            ((count + 255) // 256,),
            (256,),
            (input4, output4, np.int32(count), np.float32(1.0 / scale)),
        )

    def resource_audit(self) -> dict[str, int | None]:
        self.function.compile()
        attributes = getattr(self.function, "attributes", {}) or {}
        return {
            "num_regs": attributes.get("num_regs"),
            "shared_size_bytes_static": attributes.get("shared_size_bytes"),
            "local_size_bytes": attributes.get("local_size_bytes"),
            "max_threads_per_block": attributes.get("max_threads_per_block"),
        }
