"""C2c: native FP4 on the already-NVFP4 shapes, with a COLD working set.

C2b answered "does native Blackwell FP4 execute, and is M=2 free?" -- yes to
both, all 17 gates green. It cannot answer "is it faster than our ERVF kernel?"
for two reasons, and both have to be fixed before the comparison means anything:

1. **Wrong shapes.** C2b timed Q-like (4096x2688), Mamba-in (10304x2688) and
   LM-head. Of those only LM-head is genuinely NVFP4 in the Lightning
   checkpoint. Mamba is FP8 and attention is BF16, so putting them on FP4 is a
   quantisation change with an unmeasured quality cost -- a different claim
   entirely. The shapes where native FP4 is *format-preserving* (accumulation
   order changes, quantisation does not) are lm_head, shared_up, shared_down
   and routed_up.

2. **L2.** C2b re-reads ONE matrix per shape (300 reps for Q-like). Q-like's FP4
   weights are 5.51 MB against a 32.0 MiB L2, so that number is an L2-resident
   rate, not a DRAM rate. This exact artifact already cost this project a 1.46x
   phantom today: one matrix re-read measured 336 GB/s where the cold rate was
   230. The small NVFP4 shapes here are 2.8-5.6 MB, i.e. squarely inside L2.

So this measures native FP4 under the SAME protocol as
`diag_nvfp4_ervf_reference_rates.py`: a rotation over enough distinct matrices
that the working set is >= 4x L2, CUDA event timing, p50 over rounds, at M=1
and M=2.

Only then is `native_ms / ervf_ms` a real number.

Runs in .venv-fp4-c2b (Torch 2.12.1+cu132). That venv has no numpy, so nothing
here imports it. Synthetic +1 codes and +1 scales, as in C2b: this is a
bandwidth/geometry measurement, not a numerical one.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from common import REPO, require_gpu_free, utc_now, write_json_atomic

OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C2C_COLD_NVFP4_SHAPES.json"

L2_TARGET_MULTIPLE = 4.0
ROUNDS = 7

# name, N (rows of W), K (cols), reps, calls_per_token -- every one of these is
# NVFP4 in the checkpoint, so native FP4 here is format-preserving.
SHAPES = [
    ("lm_head",     131072, 2688,  40,   1),
    ("shared_up",     3712, 2688, 300,  23),
    ("shared_down",   2688, 3712, 300,  23),
    ("routed_up",     1856, 2688, 300, 138),
]

# From diag_nvfp4_ervf_reference_rates.json, same cold protocol.
ERVF_MS = {
    "lm_head": 1.5122630310058593,
    "shared_up": 0.03913887977600097,
    "shared_down": 0.043616957664489746,
    "routed_up": 0.020918560028076173,
}


def _ceil(x: int, q: int) -> int:
    return ((int(x) + q - 1) // q) * q


def _scale_shapes(m: int, n: int, k: int):
    if k % 16:
        raise ValueError("K must be divisible by 16")
    sfp = _ceil(k // 16, 4)
    return (_ceil(m, 128), sfp), (sfp, _ceil(n, 128))


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_native_nvfp4_c2c_cold",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": "native FP4 kernel rate on the already-NVFP4 shapes under a cold (>=4x L2) rotation; synthetic values; not a tok/s claim and not bit-exact against the ERVF reduction tree",
    }
    try:
        require_gpu_free()
        import torch
        import torch.nn.functional as F

        # Public names, matching how C2b resolves them (the leading-underscore
        # forms are the type ANNOTATIONS in the signature, not the enums).
        ScalingType = getattr(F, "ScalingType", None)
        SwizzleType = getattr(F, "SwizzleType", None)
        if ScalingType is None or SwizzleType is None:
            raise RuntimeError("F.ScalingType / F.SwizzleType missing")
        props = torch.cuda.get_device_properties(0)
        l2 = int(getattr(props, "L2_cache_size", 0) or 32 * 1024 * 1024)
        payload["device"] = {"name": props.name,
                             "capability": list(torch.cuda.get_device_capability(0)),
                             "l2_bytes": l2,
                             "torch": torch.__version__}

        arms = {}
        for name, n, k, reps, calls in SHAPES:
            fp4_bytes = n * k // 2
            cycle = max(2, int((L2_TARGET_MULTIPLE * l2 + fp4_bytes - 1) // fp4_bytes))
            cycle = min(cycle, 24)

            bs_list = []
            for _ in range(cycle):
                bu8 = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
                bs_list.append(bu8.view(torch.float4_e2m1fn_x2).t())

            rec: dict[str, Any] = {
                "N": n, "K": k, "reps": reps, "calls_per_token": calls,
                "fp4_weight_bytes": fp4_bytes,
                "matrices_in_rotation": cycle,
                "working_set_bytes": fp4_bytes * cycle,
                "working_set_over_l2": (fp4_bytes * cycle) / l2,
            }

            for m in (1, 2):
                au8 = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
                a = au8.view(torch.float4_e2m1fn_x2)
                ash, bsh = _scale_shapes(m, n, k)
                # Both scales contiguous -- the ABI requirement that failed the
                # first C2b run (scale_b was a transposed view).
                scale_a = torch.ones(ash, dtype=torch.float8_e4m3fn, device="cuda")
                scale_b = torch.ones(bsh, dtype=torch.float8_e4m3fn, device="cuda")

                def call(i):
                    return F.scaled_mm(
                        a, bs_list[i % cycle],
                        scale_a=scale_a, scale_recipe_a=ScalingType.BlockWise1x16,
                        scale_b=scale_b, scale_recipe_b=ScalingType.BlockWise1x16,
                        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                        output_dtype=torch.bfloat16, use_fast_accum=False)

                for i in range(8):
                    call(i)
                torch.cuda.synchronize()
                samples = []
                for _ in range(ROUNDS):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for i in range(reps):
                        call(i)
                    e1.record()
                    e1.synchronize()
                    samples.append(float(e0.elapsed_time(e1)) / reps)
                samples.sort()
                p50 = samples[len(samples) // 2]
                rec[f"M{m}_p50_ms"] = p50
                rec[f"M{m}_gb_s"] = fp4_bytes / (p50 * 1e-3) / 1e9
                rec[f"M{m}_samples_ms"] = samples
                del au8, a, scale_a, scale_b

            e = ERVF_MS[name]
            rec["ervf_p50_ms_same_protocol"] = e
            rec["native_speedup_M1"] = e / rec["M1_p50_ms"]
            rec["native_speedup_per_token_at_M2"] = e / (rec["M2_p50_ms"] / 2.0)
            rec["M2_over_M1"] = rec["M2_p50_ms"] / rec["M1_p50_ms"]
            arms[name] = rec
            del bs_list
            torch.cuda.empty_cache()

        ervf_tok = sum(ERVF_MS[n] * c for n, _, _, _, c in SHAPES)
        nat1_tok = sum(arms[n]["M1_p50_ms"] * c for n, _, _, _, c in SHAPES)
        nat2_tok = sum(arms[n]["M2_p50_ms"] / 2.0 * c for n, _, _, _, c in SHAPES)

        payload.update({
            "arms": arms,
            "per_token_ms_over_nvfp4_shapes": {
                "ervf": ervf_tok,
                "native_M1": nat1_tok,
                "native_M2_per_token": nat2_tok,
                "saving_M1_ms": ervf_tok - nat1_tok,
                "saving_M2_ms": ervf_tok - nat2_tok,
            },
            "status": "measured",
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=False)
    print(json.dumps({k: v for k, v in payload.items() if k != "arms"}, indent=2))
    if payload.get("arms"):
        for n, r in payload["arms"].items():
            print(f"  {n:12s} {r['N']:>6}x{r['K']:<5} ws={r['working_set_over_l2']:.1f}xL2  "
                  f"ervf={r['ervf_p50_ms_same_protocol']*1000:8.2f}us  "
                  f"nativeM1={r['M1_p50_ms']*1000:8.2f}us ({r['M1_gb_s']:6.1f} GB/s)  "
                  f"M2/M1={r['M2_over_M1']:.3f}  "
                  f"speedup M1={r['native_speedup_M1']:.2f}x  perTok@M2={r['native_speedup_per_token_at_M2']:.2f}x")
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
