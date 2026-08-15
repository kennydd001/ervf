"""Derive per-sequence state sizes from real tensor shapes and the pinned config.

These are ARITHMETIC PROJECTIONS from shapes, not measured allocations.  They
inform the H4/H8 plans; they are not evidence that anything fits.  Nothing here
may be quoted as a measured VRAM figure.

Shapes are the ones read from the checkpoint headers in N2:
  attention (6 layers): k_proj/v_proj [256, 2688] -> 2 KV heads x 128 head_dim
  mamba (23 layers)   : conv1d [6144, 1, 4]; A_log/D/dt_bias [64]; norm [4096]
                        in_proj [10304, 2688] = 4096 (z) + 6144 (conv) + 64 (dt)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "reports" / "lightningstream_nemotron" / "n3_state_budget.json"

ATTENTION_LAYERS = 6
MAMBA_LAYERS = 23
MOE_LAYERS = 23

NUM_KV_HEADS = 2
HEAD_DIM = 128

MAMBA_NUM_HEADS = 64
MAMBA_HEAD_DIM = 64
SSM_STATE_SIZE = 128
CONV_KERNEL = 4
CONV_DIM = 6144            # from conv1d weight shape [6144, 1, 4]
D_INNER = MAMBA_NUM_HEADS * MAMBA_HEAD_DIM  # 4096, matches norm [4096]

CONTEXTS = [4_096, 32_768, 131_072, 262_144]
MAX_POSITION_EMBEDDINGS = 262_144

MIB = 1024 ** 2


def main() -> int:
    # ---- KV cache: only the 6 attention layers hold one, and there is no RoPE
    kv_elems_per_token = ATTENTION_LAYERS * 2 * NUM_KV_HEADS * HEAD_DIM
    kv = {}
    for ctx in CONTEXTS:
        elems = kv_elems_per_token * ctx
        kv[str(ctx)] = {
            "elements": elems,
            "fp8_bytes": elems,
            "bf16_bytes": elems * 2,
            "fp8_mib": round(elems / MIB, 3),
            "bf16_mib": round(elems * 2 / MIB, 3),
        }

    # ---- Mamba state: constant in context length
    ssm_elems = MAMBA_LAYERS * MAMBA_NUM_HEADS * MAMBA_HEAD_DIM * SSM_STATE_SIZE
    conv_elems = MAMBA_LAYERS * CONV_DIM * CONV_KERNEL
    mamba = {
        "ssm_state_elements": ssm_elems,
        "ssm_state_fp32_bytes": ssm_elems * 4,
        "conv_state_elements": conv_elems,
        "conv_state_bf16_bytes": conv_elems * 2,
        "conv_state_fp32_bytes": conv_elems * 4,
        "total_bytes_fp32_ssm_bf16_conv": ssm_elems * 4 + conv_elems * 2,
        "total_mib_fp32_ssm_bf16_conv": round((ssm_elems * 4 + conv_elems * 2) / MIB, 3),
        "context_dependent": False,
    }

    combined = {}
    for ctx in CONTEXTS:
        total = kv[str(ctx)]["fp8_bytes"] + mamba["total_bytes_fp32_ssm_bf16_conv"]
        combined[str(ctx)] = {
            "kv_fp8_plus_mamba_bytes": total,
            "mib": round(total / MIB, 3),
        }

    result = {
        "kind": "lightningstream_nemotron_n3_state_budget",
        "status": "arithmetic_projection_from_shapes_not_measured",
        "inputs": {
            "attention_layers": ATTENTION_LAYERS,
            "mamba_layers": MAMBA_LAYERS,
            "moe_layers": MOE_LAYERS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "mamba_num_heads": MAMBA_NUM_HEADS,
            "mamba_head_dim": MAMBA_HEAD_DIM,
            "ssm_state_size": SSM_STATE_SIZE,
            "conv_kernel": CONV_KERNEL,
            "conv_dim": CONV_DIM,
            "d_inner": D_INNER,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "rope_present_in_modeling_code": False,
        },
        "kv_elements_per_token": kv_elems_per_token,
        "kv_bytes_per_token_fp8": kv_elems_per_token,
        "kv_by_context": kv,
        "mamba_state": mamba,
        "combined_state_by_context": combined,
        "notes": [
            "The six attention layers carry no rotary embedding; positional "
            "information comes from the Mamba layers. Long-context work here is "
            "therefore not a RoPE-scaling problem.",
            "Mamba state is constant in context length, so the only "
            "context-dependent term is the KV cache of six layers with two KV "
            "heads each.",
            "These are projections from tensor shapes and the pinned config. "
            "They are not measured allocations and must not be quoted as VRAM "
            "figures.",
        ],
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"KV elements/token (6 attn layers) : {kv_elems_per_token:,}")
    print(f"KV bytes/token FP8                : {kv_elems_per_token:,}")
    print()
    for ctx in CONTEXTS:
        row = kv[str(ctx)]
        tot = combined[str(ctx)]
        print(f"  ctx {ctx:>7,} : KV fp8 {row['fp8_mib']:>9.3f} MiB | "
              f"KV bf16 {row['bf16_mib']:>9.3f} MiB | +mamba {tot['mib']:>9.3f} MiB")
    print()
    print(f"Mamba state (context-independent) : {mamba['total_mib_fp32_ssm_bf16_conv']} MiB")
    print(f"  ssm fp32  : {mamba['ssm_state_fp32_bytes']:,} B")
    print(f"  conv bf16 : {mamba['conv_state_bf16_bytes']:,} B")
    print(f"written : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
