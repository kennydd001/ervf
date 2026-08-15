"""Derive the NVFP4 record layout implied by the pinned config, and test it
against the independently measured N1 header inventory.

This is arithmetic on already-published numbers.  It downloads nothing and
proves nothing on its own: agreement between a layout hypothesis and the frozen
byte buckets is strong evidence, not a layout proof.  N2 must confirm the real
layout by reading actual tensor-index entries.

Hypothesis under test (ModelOpt NVFP4 convention):
  per quantized matrix of N weights
    - NVFP4 codes  : N/2 bytes  (two 4-bit codes per byte, dtype U8)
    - block scales : N/group    bytes (FP8 E4M3, group_size from hf_quant_config)
    - global scales: 2 x FP32   (weight_scale_2 and input_scale) = 8 bytes
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "reports" / "lightningstream_nemotron" / "n0r_layout_consistency.json"

# From the pinned public config.json (N0R).
HIDDEN = 2688
MOE_INTERMEDIATE = 1856
SHARED_INTERMEDIATE = 3712
MOE_LAYERS = 23
ROUTED_EXPERTS_PER_LAYER = 128
GROUP_SIZE = 16
FP32 = 4

# From the frozen N1 header inventory (protected, read-only).
N1 = {
    "routed_record_bytes": 5_612_560,
    "routed_records": 2_944,
    "routed_bucket_bytes": 16_523_376_640,
    "shared_bucket_bytes": 258_177_392,
    "trunk_other_bytes": 2_558_227_600,
    "tensor_bytes": 19_339_781_632,
    "u8_bytes": 15_245_905_920,
    "f8_e4m3_bytes": 1_905_738_240,
    "bf16_bytes": 2_156_424_064,
    "f32_bytes": 31_713_408,
}


def matrix_cost(n_weights: int) -> dict:
    codes = n_weights // 2
    scales = n_weights // GROUP_SIZE
    globals_ = 2 * FP32
    return {"weights": n_weights, "code_bytes": codes,
            "block_scale_bytes": scales, "global_scale_bytes": globals_,
            "total": codes + scales + globals_}


def main() -> int:
    # A routed expert under relu2 has two matrices: up [I,H] and down [H,I].
    up = matrix_cost(MOE_INTERMEDIATE * HIDDEN)
    down = matrix_cost(HIDDEN * MOE_INTERMEDIATE)
    routed_record = up["total"] + down["total"]

    # A shared expert has the same two-matrix shape at 2x intermediate.
    s_up = matrix_cost(SHARED_INTERMEDIATE * HIDDEN)
    s_down = matrix_cost(HIDDEN * SHARED_INTERMEDIATE)
    shared_record = s_up["total"] + s_down["total"]

    routed_total = routed_record * MOE_LAYERS * ROUTED_EXPERTS_PER_LAYER
    shared_total = shared_record * MOE_LAYERS

    routed_u8 = (up["code_bytes"] + down["code_bytes"]) * MOE_LAYERS * ROUTED_EXPERTS_PER_LAYER
    routed_f8 = (up["block_scale_bytes"] + down["block_scale_bytes"]) * MOE_LAYERS * ROUTED_EXPERTS_PER_LAYER
    shared_u8 = (s_up["code_bytes"] + s_down["code_bytes"]) * MOE_LAYERS
    shared_f8 = (s_up["block_scale_bytes"] + s_down["block_scale_bytes"]) * MOE_LAYERS

    checks = {
        "routed_record_bytes_match": routed_record == N1["routed_record_bytes"],
        "routed_record_count_match": MOE_LAYERS * ROUTED_EXPERTS_PER_LAYER == N1["routed_records"],
        "routed_bucket_match": routed_total == N1["routed_bucket_bytes"],
        "shared_bucket_match": shared_total == N1["shared_bucket_bytes"],
        "routed_plus_shared_u8_within_total": routed_u8 + shared_u8 <= N1["u8_bytes"],
        "routed_plus_shared_f8_within_total": routed_f8 + shared_f8 <= N1["f8_e4m3_bytes"],
    }

    residual_u8 = N1["u8_bytes"] - routed_u8 - shared_u8
    residual_f8 = N1["f8_e4m3_bytes"] - routed_f8 - shared_f8

    result = {
        "kind": "lightningstream_nemotron_n0r_layout_consistency",
        "status": "derived_hypothesis_not_layout_proof",
        "inputs": {
            "config": {"hidden_size": HIDDEN, "moe_intermediate_size": MOE_INTERMEDIATE,
                       "moe_shared_expert_intermediate_size": SHARED_INTERMEDIATE,
                       "moe_layers": MOE_LAYERS,
                       "n_routed_experts": ROUTED_EXPERTS_PER_LAYER,
                       "group_size": GROUP_SIZE, "mlp_hidden_act": "relu2"},
            "n1_frozen": N1,
        },
        "per_routed_expert": {"up": up, "down": down, "record_bytes": routed_record},
        "per_shared_expert": {"up": s_up, "down": s_down, "record_bytes": shared_record},
        "totals": {
            "routed_bucket_bytes": routed_total,
            "shared_bucket_bytes": shared_total,
            "routed_u8": routed_u8, "routed_f8": routed_f8,
            "shared_u8": shared_u8, "shared_f8": shared_f8,
            "residual_u8_for_trunk": residual_u8,
            "residual_f8_for_trunk": residual_f8,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "interpretation": (
            "Every frozen N1 byte bucket is reproduced exactly by the ModelOpt "
            "NVFP4 convention applied to the pinned config shapes. This makes the "
            "layout hypothesis very likely, but N2 must confirm it against real "
            "tensor-index entries and dtypes before any decoder relies on it."
        ),
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"routed record bytes derived : {routed_record:,}   N1: {N1['routed_record_bytes']:,}")
    print(f"routed bucket derived       : {routed_total:,}   N1: {N1['routed_bucket_bytes']:,}")
    print(f"shared bucket derived       : {shared_total:,}   N1: {N1['shared_bucket_bytes']:,}")
    print(f"residual U8 for trunk       : {residual_u8:,}")
    print(f"residual F8 for trunk       : {residual_f8:,}")
    for key, value in checks.items():
        print(f"  {key:<38}: {value}")
    print(f"all checks pass             : {result['all_checks_pass']}")
    return 0 if result["all_checks_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
