"""Does W4A8 exist? Enumerate the scaling recipes for an FP8 A against an FP4 B.

FP4_ACTIVATION_SIDE.json established that `scaled_mm` refuses a BF16 activation
outright -- "Expected mat_a to be Float8 or Float4_x2 matrix got BFloat16" --
so adopting native FP4 is NOT format-preserving: the activation has to be
quantised too. That is a real quality change and it carries an unmeasured
per-call quantisation kernel.

But the FP8 arm did not fail on dtype. It failed on "Invalid scaling
configuration", which means **FP8 is an accepted activation type** and only the
recipe pairing was wrong. That matters a lot:

  W4A4  activation to 4 bits   large quality change, no evidence
  W4A8  activation to 8 bits   far milder, and this project already runs an
                               FP8 KV cache, so FP8 activations are not new
                               ground here

So: which (scale_recipe_a, scale_recipe_b) pairing, if any, accepts FP8 x FP4?
This enumerates every ScalingType the build exposes, for both the FP8 and FP4
activation, and records for each pairing whether it executes and whether the
value is right.

A pairing that "executes" but returns the wrong number is worse than a
rejection, so the value is checked against the known answer (all +1 codes and
unit scales give K), and pairings are only reported as usable when both hold.

Read-only, synthetic, no model load. Run in .venv-fp4-c2b.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "FP4_W4A8_RECIPES.json"

E4M3_ONE = 0x38
M, N, K = 1, 128, 256


def _ceil(x, q):
    return ((x + q - 1) // q) * q


def main() -> int:
    import torch
    import torch.nn.functional as F

    ScalingType = getattr(F, "ScalingType")
    SwizzleType = getattr(F, "SwizzleType")

    recipes = [n for n in dir(ScalingType) if not n.startswith("_")]
    swizzles = [n for n in dir(SwizzleType) if not n.startswith("_")]
    rec: dict = {
        "kind": "diag_fp4_w4a8_recipes",
        "torch": torch.__version__,
        "available_scaling_types": recipes,
        "available_swizzle_types": swizzles,
        "shape": {"M": M, "N": N, "K": K},
        "expected_value_fp4_arm": float(K),
        "expected_value_fp8_arm_packed_width": float(K // 2),
        "results": [],
    }

    sfp = _ceil(K // 16, 4)
    b = torch.full((N, K // 2), 0x22, dtype=torch.uint8,
                   device="cuda").view(torch.float4_e2m1fn_x2).t()
    sb_block = torch.full((sfp, _ceil(N, 128)), E4M3_ONE, dtype=torch.uint8,
                          device="cuda").view(torch.float8_e4m3fn)

    def make_a(kind):
        if kind == "fp4":
            return torch.full((M, K // 2), 0x22, dtype=torch.uint8,
                              device="cuda").view(torch.float4_e2m1fn_x2)
        # B is stored PACKED as (N, K//2) and scaled_mm compares those stored
        # shapes, so a K-wide FP8 A dies on shape before any recipe is checked
        # -- which is exactly how the first run of this sweep wasted all six
        # FP8 arms. Match the packed width so the recipe check is actually
        # reached. The contraction is then 128 wide, not 256, so the expected
        # value for the FP8 arms is K//2, not K.
        return torch.ones((M, K // 2), dtype=torch.float8_e4m3fn, device="cuda")

    def make_sa(kind, recipe_name):
        # Scale shape depends on the recipe: block-scaled wants one scale per
        # 16 elements, tensor/row-wise want a scalar or one per row.
        if "Block" in recipe_name:
            return torch.full((_ceil(M, 128), sfp), E4M3_ONE, dtype=torch.uint8,
                              device="cuda").view(torch.float8_e4m3fn)
        if "Row" in recipe_name:
            return torch.ones((M, 1), dtype=torch.float32, device="cuda")
        return torch.ones((), dtype=torch.float32, device="cuda")

    for a_kind in ("fp8", "fp4"):
        for ra in recipes:
            for rb in ("BlockWise1x16",) if "BlockWise1x16" in recipes else recipes:
                entry = {"a_dtype": a_kind, "scale_recipe_a": ra,
                         "scale_recipe_b": rb}
                try:
                    a = make_a(a_kind)
                    sa = make_sa(a_kind, ra)
                    out = F.scaled_mm(
                        a, b,
                        scale_a=sa, scale_recipe_a=getattr(ScalingType, ra),
                        scale_b=sb_block, scale_recipe_b=getattr(ScalingType, rb),
                        swizzle_a=(SwizzleType.SWIZZLE_32_4_4 if "Block" in ra
                                   else SwizzleType.NO_SWIZZLE
                                   if "NO_SWIZZLE" in swizzles else None),
                        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                        output_dtype=torch.bfloat16, use_fast_accum=False)
                    torch.cuda.synchronize()
                    v = float(out.flatten()[0].item())
                    exp = float(K) if a_kind == "fp4" else float(K // 2)
                    entry.update({"executes": True, "value": v,
                                  "expected": exp,
                                  "value_correct": v == exp})
                except Exception as exc:
                    entry.update({"executes": False,
                                  "error": f"{type(exc).__name__}: {exc}"[:220]})
                rec["results"].append(entry)

    usable = [r for r in rec["results"]
              if r.get("executes") and r.get("value_correct")]
    w4a8 = [r for r in usable if r["a_dtype"] == "fp8"]
    rec["usable_pairings"] = usable
    rec["w4a8_pairings"] = w4a8
    rec["verdict"] = "w4a8_available" if w4a8 else "w4a4_only"
    rec["consequence"] = (
        "an FP8 activation is accepted with a working recipe, so the adoption's "
        "quality change is 8-bit not 4-bit on the activation side -- far milder, "
        "and this project already runs an FP8 KV cache. The quantisation kernel "
        "cost is still unmeasured."
        if w4a8 else
        "no FP8-activation pairing produced a correct result, so the path really "
        "is W4A4: the activation must go to 4 bits, which is a substantial "
        "quality change with no evidence behind it yet.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in
                      ("available_scaling_types", "verdict", "consequence",
                       "w4a8_pairings")}, indent=2))
    print("\nper-pairing:")
    for r in rec["results"]:
        if r.get("executes"):
            print(f"  {r['a_dtype']:4s} A={r['scale_recipe_a']:<16s} -> "
                  f"value={r.get('value')} correct={r.get('value_correct')}")
        else:
            print(f"  {r['a_dtype']:4s} A={r['scale_recipe_a']:<16s} -> "
                  f"{r['error'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
