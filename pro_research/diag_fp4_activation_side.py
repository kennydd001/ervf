"""Is native FP4 really "format-preserving"? Test the ACTIVATION side.

I have been writing that native FP4 on the already-NVFP4 shapes is
"format-preserving -- only the accumulation order changes, not the
quantisation". That claim covers the WEIGHTS. It says nothing about the
activation, and our production path is W4A32: NVFP4 weights times an FP32
activation vector.

`F.scaled_mm` on Blackwell is a tensor-core path. If it requires BOTH operands
in FP4, then adopting it means quantising the activation to FP4 every token,
which is:
  * extra work per call (a dynamic per-16-block quantisation kernel), and
  * a genuine quality change, not a reordering.

Kimi's C3A claim boundary already flagged this -- "exact +1 A ... no
real-activation quantization" -- and the vLLM SM120 NVFP4 path is documented as
dynamic FP4 activation quantisation followed by a CUTLASS FP4 GEMM, which points
the same way. But neither is a measurement on this machine.

So: does scaled_mm accept a higher-precision A against FP4 B?

  A  FP4 x FP4      the C2b/C2c/C2d configuration, control, must work
  B  BF16 x FP4     would make the adoption genuinely format-preserving
  C  FP8E4M3 x FP4  a middle option if B is rejected

Each arm records whether it executes at all and, if so, whether the value is
right. A rejection is as informative as a pass here -- it tells us the adoption
needs a quality gate on the activation, and how big the extra kernel is.

Run in .venv-fp4-c2b.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "FP4_ACTIVATION_SIDE.json"

E4M3_ONE = 0x38
M, N, K = 1, 128, 256


def _ceil(x, q):
    return ((x + q - 1) // q) * q


def main() -> int:
    import torch
    import torch.nn.functional as F

    ScalingType = getattr(F, "ScalingType")
    SwizzleType = getattr(F, "SwizzleType")
    sfp = _ceil(K // 16, 4)
    rec: dict = {"kind": "diag_fp4_activation_side", "torch": torch.__version__,
                 "shape": {"M": M, "N": N, "K": K}, "arms": {}}

    b = torch.full((N, K // 2), 0x22, dtype=torch.uint8,
                   device="cuda").view(torch.float4_e2m1fn_x2).t()
    sa = torch.full((_ceil(M, 128), sfp), E4M3_ONE, dtype=torch.uint8,
                    device="cuda").view(torch.float8_e4m3fn)
    sb = torch.full((sfp, _ceil(N, 128)), E4M3_ONE, dtype=torch.uint8,
                    device="cuda").view(torch.float8_e4m3fn)

    def build_a(kind):
        if kind == "fp4":
            return torch.full((M, K // 2), 0x22, dtype=torch.uint8,
                              device="cuda").view(torch.float4_e2m1fn_x2)
        # B is stored PACKED as (N, K//2) and scaled_mm reads that packed shape,
        # so a K-wide high-precision A is a shape mismatch before the dtype is
        # ever considered. Offer BOTH widths: K (the honest logical width) and
        # K//2 (shape-matching the packed operand). If the K//2 form is rejected
        # on DTYPE rather than shape, that is clean evidence the path is W4A4.
        if kind == "bf16":
            return torch.ones((M, K), dtype=torch.bfloat16, device="cuda")
        if kind == "bf16_packedshape":
            return torch.ones((M, K // 2), dtype=torch.bfloat16, device="cuda")
        if kind == "fp8":
            return torch.ones((M, K), dtype=torch.float8_e4m3fn, device="cuda")
        if kind == "fp8_packedshape":
            return torch.ones((M, K // 2), dtype=torch.float8_e4m3fn, device="cuda")
        raise ValueError(kind)

    for label, kind, recipe in (
        ("A_fp4_x_fp4", "fp4", ScalingType.BlockWise1x16),
        ("B_bf16_x_fp4", "bf16", ScalingType.BlockWise1x16),
        ("C_fp8_x_fp4", "fp8", ScalingType.BlockWise1x16),
        ("B2_bf16_packedshape_x_fp4", "bf16_packedshape", ScalingType.BlockWise1x16),
        ("C2_fp8_packedshape_x_fp4", "fp8_packedshape", ScalingType.BlockWise1x16),
    ):
        try:
            a = build_a(kind)
            out = F.scaled_mm(a, b,
                              scale_a=sa, scale_recipe_a=recipe,
                              scale_b=sb, scale_recipe_b=ScalingType.BlockWise1x16,
                              swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                              swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                              output_dtype=torch.bfloat16, use_fast_accum=False)
            torch.cuda.synchronize()
            v = float(out.flatten()[0].item())
            rec["arms"][label] = {"executes": True, "value": v,
                                  "expected": float(K),
                                  "value_correct": v == float(K)}
        except Exception as exc:
            rec["arms"][label] = {"executes": False,
                                  "error": f"{type(exc).__name__}: {exc}"[:400]}

    mixed_ok = any(rec["arms"].get(k, {}).get("executes")
                   for k in ("B_bf16_x_fp4", "C_fp8_x_fp4",
                             "B2_bf16_packedshape_x_fp4", "C2_fp8_packedshape_x_fp4"))
    # Distinguish "rejected for shape" from "rejected for dtype" -- only the
    # latter is clean evidence that the path is W4A4 by design.
    rec["rejection_kinds"] = {
        k: ("shape" if "shapes cannot be multiplied" in v.get("error", "")
            else "dtype_or_other" if not v.get("executes") else "executed")
        for k, v in rec["arms"].items()}
    rec["verdict"] = ("mixed_precision_A_supported" if mixed_ok
                      else "W4A4_required_activation_must_be_quantised")
    rec["consequence"] = (
        "adoption is genuinely format-preserving: the FP32/BF16 activation can "
        "stay, only the weight path and accumulation order change"
        if mixed_ok else
        "adoption is NOT format-preserving. The activation must be quantised to "
        "FP4 every call, which adds a dynamic per-16-block quantisation kernel "
        "AND is a real quality change. The measured 2.52x/1.68x kernel speedups "
        "do not include that quantisation cost, and no quality evidence exists "
        "for it yet. My earlier 'format-preserving' framing was wrong on the "
        "activation side and is corrected here.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
