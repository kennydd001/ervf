"""Redo the MTP speculative-decode economics with C2d's free-M finding -- and
correct my own 99 tok/s projection from the previous turn.

## Why this needs redoing

The MTP route was closed on 2026-08-16 with a clean measurement
(`diag_mtp_route_union.json`): one speculative round costs the MTP chain plus a
verification sweep over D+1 positions, and the verification sweep was priced at
3.313x a single token's MoE cost because five consecutive tokens route to
**19.88 of 128 experts** per layer against 6 for one token. Conclusion then:
~6.0% SLOWER than plain decode.

That closure rested on one assumption: verifying K positions costs K times the
weight reads. C2d refuted exactly that for dense GEMMs -- native FP4 on tensor
cores does M=8 for the price of M=1 (M8/M1 = 0.989 / 0.814 / 1.003 / 0.891).
So the closure has to be recomputed rather than left standing.

## The error in my own follow-up, which this also corrects

Last turn I projected ~99 tok/s by dividing "all divisible weight work" by M and
listing `up_proj` (2.253 ms) among the divisible terms. That is wrong, and I
flagged the reason myself in the same message without applying it: `up_proj` is
**routed**. Its cost does not divide by M, it follows the expert UNION, which
grows 3.313x over five positions. The same applies to the gather, down_masked
and panel_scan/reduce/accumulate -- every per-expert term.

Only genuinely shared weights divide by M: Mamba's projections, the attention
projections, the shared expert, and lm_head.

## What divides and what does not

  M-free across all D+1 positions (one weight pass serves every position):
      Mamba in/out GEMV, attention Q/K/V/O projections, shared expert, lm_head
  Follows the routed union (measured 3.313x for 5 positions, not 5x and not 1x):
      MoE gather (PCIe), up_proj, down_masked, panel_scan/reduce/accumulate
  Strictly per-position, no sharing possible at all:
      ssm_step (Mamba recurrence is sequential in position), conv/dt,
      gated_norm, norms/adds, and attention's KV/flash part

That third class is the one nobody has been costing, and ssm_step alone is
1.095 ms/token.

Read-only arithmetic over already-measured quantities. Writes a JSON so the
result is auditable rather than asserted in prose. No GPU needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import utc_now, write_json_atomic

# In-graph token map, measured marginals on a 21.24 ms basis
# (agents/STATE_OF_THE_WORK.md). Scaled to the 19.60 ms V18 record.
MAP_BASIS_MS = 21.24
V18_MS = 19.60
SCALE = V18_MS / MAP_BASIS_MS

TOKEN_MAP = {
    "mamba_gemv":        4.187,
    "moe_gather_pcie":   3.849,
    "attention":         2.479,
    "moe_up_proj":       2.253,
    "moe_shared_expert": 1.810,
    "moe_down_masked":   1.372,
    "moe_panel_reduce":  1.119,
    "lm_head":           1.107,
    "mamba_ssm_step":    1.095,
    "norms_adds":        0.370,
    "mamba_gated_norm":  0.273,
    "mamba_conv_dt":     0.197,
}
# attention splits: projections are weight-bound and shareable; the KV/flash
# part is per-position. The token map does not split it, so this is an
# assumption and is flagged as one.
ATTENTION_PROJ_FRACTION = 0.60

M_FREE = ["mamba_gemv", "moe_shared_expert", "lm_head"]
ROUTED = ["moe_gather_pcie", "moe_up_proj", "moe_down_masked", "moe_panel_reduce"]
PER_POSITION = ["mamba_ssm_step", "norms_adds", "mamba_gated_norm", "mamba_conv_dt"]

# Measured (diag_mtp_route_union.json): union over 5 consecutive positions is
# 19.88 of 128 experts per layer vs 6 for one token.
UNION_5 = 19.88 / 6.0          # 3.313x
UNION_2 = (12 - 2.011) / 6.0   # N7-A pairwise overlap 2.011 of 6 -> 1.665x

# S10-A: acceptance A = 2.114 over 360 steps, so A+1 = 3.114 tokens per round
# at D=4. MTP chain cost measured at 19.10 ms on a 54.28 ms/token stack.
ACCEPT_D4 = 3.114
MTP_CHAIN_FRACTION_OF_TOKEN = 19.10 / 54.28


def main() -> int:
    m = {k: v * SCALE for k, v in TOKEN_MAP.items()}
    attn_proj = m["attention"] * ATTENTION_PROJ_FRACTION
    attn_kv = m["attention"] - attn_proj

    dense_free = sum(m[k] for k in M_FREE) + attn_proj
    routed = sum(m[k] for k in ROUTED)
    per_pos = sum(m[k] for k in PER_POSITION) + attn_kv
    accounted = dense_free + routed + per_pos
    unattributed = V18_MS - accounted

    scenarios = {}
    for label, D1, union, accepted in (
        ("D+1=5 (D=4 drafts)", 5, UNION_5, ACCEPT_D4),
        ("D+1=2 (checkpoint MTP, num_nextn_predict_layers=1)", 2, UNION_2, 1.6),
    ):
        mtp_cost = MTP_CHAIN_FRACTION_OF_TOKEN * V18_MS * ((D1 - 1) / 4.0)
        sweep = (
            dense_free                      # one weight pass for all positions
            + routed * union                # follows the measured expert union
            + (per_pos + unattributed) * D1  # strictly sequential per position
            + mtp_cost
        )
        ms_per_token = sweep / accepted
        scenarios[label] = {
            "positions_verified": D1,
            "routed_union_multiplier": union,
            "accepted_tokens_per_round": accepted,
            "dense_m_free_ms": dense_free,
            "routed_ms": routed * union,
            "per_position_ms": (per_pos + unattributed) * D1,
            "mtp_draft_ms": mtp_cost,
            "round_total_ms": sweep,
            "ms_per_accepted_token": ms_per_token,
            "tok_s": 1000.0 / ms_per_token,
            "vs_v18_speedup": V18_MS / ms_per_token,
            "routed_share_of_round": (routed * union) / sweep,
        }

    payload = {
        "kind": "diag_mtp_native_fp4_economics",
        "created_utc": utc_now(),
        "note": "recomputes the MTP speculative-decode economics under C2d's measured free-M property, and corrects the assistant's own 99 tok/s projection from the previous turn, which wrongly counted the ROUTED up_proj among the terms that divide by M",
        "inputs": {
            "v18_ms_per_token": V18_MS,
            "token_map_basis_ms": MAP_BASIS_MS,
            "routed_union_5_positions": "19.88 of 128 experts vs 6 for one token = 3.313x (diag_mtp_route_union.json, measured)",
            "acceptance_A_plus_1": ACCEPT_D4,
            "acceptance_source": "S10A_MTP_ACCEPTANCE_REPORT_2026-08-15.md, A=2.114 over 360 steps, gate G-S10-1 >=1.5 PASSED",
            "free_M_source": "C2D_M_SCALING.json, M8/M1 = 0.989 / 0.814 / 1.003 / 0.891",
            "attention_projection_fraction_ASSUMED": ATTENTION_PROJ_FRACTION,
        },
        "classification": {
            "m_free_ms": dense_free,
            "routed_follows_union_ms": routed,
            "strictly_per_position_ms": per_pos,
            "unattributed_ms": unattributed,
        },
        "scenarios": scenarios,
        "correction": {
            "previous_claim_tok_s": 99.0,
            "why_wrong": "it divided moe_up_proj (2.253 ms) by M. up_proj is routed: its cost follows the expert union, which grows 3.313x over 5 positions, not 1x. The same applies to gather, down_masked and panel_scan/reduce/accumulate. Only Mamba/attention projections, the shared expert and lm_head are genuinely shared across positions.",
        },
        "caveats": [
            "the Mamba and attention projections only become M-free if they are moved to native FP4, which is a real quantisation change with an unmeasured quality cost",
            "the attention projection/KV split is assumed at 0.60, not measured",
            "the MTP chain cost is carried over as a fraction of a token from a 54.28 ms/token stack",
            "no end-to-end run; this is arithmetic over measured components, exactly like the closure it revisits",
        ],
    }
    write_json_atomic(REPO / "pro_research" / "diag_mtp_native_fp4_economics.json",
                      payload, archive=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
