from __future__ import annotations

import json

import numpy as np

from common import REPO
from diag_native_nvfp4_c3b_realact import native_call
import native_nvfp4_c3a_layout_v2 as c3v2
import native_nvfp4_c3a_lib as c3lib
from s100_phase35_c3c_quantizer import FusedStaticNVFP4Quantizer


CONVERT_SOURCE = r"""
#include <cuda_bf16.h>
extern "C" __global__ void bf16_to_f32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    const int count)
{
    int i=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
    if(i<count)dst[i]=__bfloat162float(src[i]);
}
"""


class NativeFP4HeadH8:
    def __init__(self, rt, stream):
        import cupy as cp
        import torch
        import torch.nn.functional as F

        self.cp = cp
        self.torch = torch
        self.F = F
        self.ST = F.ScalingType
        self.SW = F.SwizzleType
        self.rt = rt
        self.stream = stream
        self.external_stream = torch.cuda.ExternalStream(stream.ptr)
        self.vocab = int(rt.vocab)
        self.hidden = int(rt.hidden)
        self.m = 8

        c3v2.install(c3lib)
        c3b = json.loads(
            (
                REPO
                / "pro_research"
                / "results"
                / "native_nvfp4"
                / "C3B_REAL_ACTIVATION.json"
            ).read_text(encoding="utf-8")
        )
        if c3b.get("status") != "real_activation_native_candidate":
            raise RuntimeError("C4 requires green C3B")
        family = next(f for f in c3b["families"] if f["label"] == "lm_head")
        self.tensor_scale_value = float(
            family["activation"]["static_tensor_scale"]
        )

        self.quantizer = FusedStaticNVFP4Quantizer(self.hidden, self.m)
        self.a_global = torch.tensor(
            [self.tensor_scale_value], dtype=torch.float32, device="cuda"
        )
        self.a = {
            "fp4": torch.as_tensor(
                self.quantizer.packed, device="cuda"
            ).view(torch.float4_e2m1fn_x2),
            "block": torch.as_tensor(
                self.quantizer.blocked_scales, device="cuda"
            ).view(torch.float8_e4m3fn),
            "global": self.a_global,
        }

        codes = torch.as_tensor(rt.lm_head_codes, device="cuda").reshape(
            self.vocab, self.hidden // 2
        )
        scale_raw = cp.asnumpy(rt.lm_head_scales).view(np.uint8).tobytes()
        blocked_b = c3v2.repack_b_scale(
            torch, scale_raw, self.vocab, self.hidden
        )
        self.b = {
            "u8": codes,
            "fp4": codes.view(torch.float4_e2m1fn_x2).t(),
            "block": blocked_b,
            "global": torch.tensor(
                [float(rt.lm_head_g)], dtype=torch.float32, device="cuda"
            ),
        }
        self.convert_mod = cp.RawModule(
            code=CONVERT_SOURCE,
            options=("-std=c++14",),
            name_expressions=("bf16_to_f32",),
        )
        self.convert = self.convert_mod.get_function("bf16_to_f32")
        self.last_output = None

    def __call__(self, x8, logits8) -> None:
        self.quantizer.quantize(x8, self.tensor_scale_value)
        with self.torch.cuda.stream(self.external_stream):
            output = native_call(
                self.torch, self.F, self.ST, self.SW, self.a, self.b
            )
        self.last_output = output
        count = self.m * self.vocab
        self.convert(
            ((count + 255) // 256,),
            (256,),
            (np.uint64(output.data_ptr()), logits8, np.int32(count)),
        )

    def resource_audit(self) -> dict:
        self.convert.compile()
        attrs = getattr(self.convert, "attributes", {}) or {}
        return {
            "quantizer": self.quantizer.resource_audit(),
            "bf16_to_f32": {
                "num_regs": attrs.get("num_regs"),
                "shared_size_bytes_static": attrs.get("shared_size_bytes"),
                "local_size_bytes": attrs.get("local_size_bytes"),
                "max_threads_per_block": attrs.get("max_threads_per_block"),
            },
        }

    @property
    def extra_device_bytes(self) -> int:
        return int(
            self.b["block"].numel()
            + self.a_global.numel() * self.a_global.element_size()
            + self.quantizer.packed.nbytes
            + self.quantizer.blocked_scales.nbytes
        )
