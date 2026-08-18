"""S100 phase 12B route-union census (preregistered).

Reuses the frozen phase-9 routing trace (S100_PHASE9_TRACE.npz): 9376 decode
steps (8192 counted/measured) x 23 MoE layers x top-6 expert ids, with
session boundaries. No new GPU work; pure analysis.

Per B in {2,4,8}, non-overlapping blocks of B consecutive counted tokens
inside one session, per MoE layer and pooled:

- unique experts per block (the expert union);
- rows-per-unique-expert histogram = the M distribution for grouped
  routed-up/down;
- device weight-read bytes per token: current M=1 path reads 6 experts per
  token whether the fetch hit or missed; grouped reads each unique expert
  once per block. reduction = 1 - |union|/(6B);
- PCIe fetch slots: current = sum(need) under the production LRU; grouped =
  unique experts whose first in-block occurrence missed (the LRU already
  deduplicates temporal repeats, so this isolates the residual).

Preregistered gate: grouped MoE opens only if median routed-weight bytes per
token fall by >=20% at B=4.

Deviation note: the trace holds 8192 counted tokens, below the 10,000 in the
preregistration. If a gate-relevant median lands within 2 points of 20%, the
census must be re-run on a >=10k capture before deciding.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "pro_research" / "results"
NPZ = RES / "s100_phase9" / "S100_PHASE9_TRACE.npz"
OUT = RES / "s100_phase12" / "S100_PHASE12B_CENSUS.json"
TOP_K = 6
BLOCKS = (2, 4, 8)
GATE_B = 4
GATE_REDUCTION = 0.20


def main() -> int:
    p: dict = {"kind": "s100_phase12b_census", "status": "started"}
    try:
        d = np.load(NPZ)
        ids = d["ids"].astype(np.int64)          # (N, L, K)
        need = d["need"].astype(bool)            # (N, L, K)
        counted = d["counted"].astype(bool)      # (N,)
        session = d["session"].astype(np.int64)  # (N,)
        layers = [int(x) for x in d["layers"]]
        n, L, K = ids.shape
        assert K == TOP_K, f"trace top_k {K} != {TOP_K}"

        # Counted steps only, keeping session contiguity.
        keep = np.nonzero(counted)[0]
        n_counted = int(keep.size)

        report_blocks: dict[str, dict] = {}
        for B in BLOCKS:
            # Per session, form non-overlapping blocks of B counted steps.
            blocks: list[np.ndarray] = []
            for sid in np.unique(session[keep]):
                rows = keep[session[keep] == sid]
                for off in range(0, rows.size - B + 1, B):
                    blocks.append(rows[off : off + B])
            nb = len(blocks)
            if nb == 0:
                raise RuntimeError(f"no blocks at B={B}")

            union_sizes = np.zeros((nb, L), np.int64)
            grouped_miss = np.zeros((nb, L), np.int64)
            current_miss = np.zeros((nb, L), np.int64)
            row_hist = np.zeros(B + 1, np.int64)  # rows per unique expert, pooled over layers

            for bi, rows in enumerate(blocks):
                bids = ids[rows]      # (B, L, K)
                bneed = need[rows]    # (B, L, K)
                for li in range(L):
                    flat = bids[:, li, :].ravel()
                    uniq, first_idx, counts = np.unique(
                        flat, return_index=True, return_counts=True
                    )
                    union_sizes[bi, li] = uniq.size
                    for c in counts:
                        row_hist[c] += 1
                    current_miss[bi, li] = int(bneed[:, li, :].sum())
                    # Grouped fetch: expert fetched once; it misses iff its
                    # first in-block occurrence missed under the LRU.
                    first_need = bneed[:, li, :].ravel()[np.sort(first_idx)]
                    grouped_miss[bi, li] = int(first_need.sum())

            # Per-token views.
            union_per_tok = union_sizes / B                        # unique experts read per token
            slot_ratio = union_sizes / (B * TOP_K)                 # device-read fraction remaining
            reduction = 1.0 - slot_ratio                           # device-read byte reduction
            per_tok_reduction = 1.0 - (union_sizes / B) / TOP_K

            cur = current_miss.sum(axis=1) / B                     # PCIe miss slots per token
            grp = grouped_miss.sum(axis=1) / B
            pcie_ratio = float(grp.sum() / cur.sum()) if cur.sum() else float("nan")

            report_blocks[str(B)] = {
                "blocks": nb,
                "tokens": nb * B,
                "union_per_block_median": float(np.median(union_sizes)),
                "union_per_block_p90": float(np.percentile(union_sizes, 90)),
                "union_slots_ratio_median": float(np.median(slot_ratio)),
                "device_read_reduction_mean": float(per_tok_reduction.mean()),
                "device_read_reduction_median": float(np.median(per_tok_reduction)),
                "device_read_reduction_p10": float(np.percentile(per_tok_reduction, 10)),
                "rows_per_expert_hist": [int(x) for x in row_hist[1:]],
                "rows_per_expert_mean": float(
                    sum((i + 1) * row_hist[i + 1] for i in range(B)) / max(1, row_hist[1:].sum())
                ),
                "pcie_miss_slots_per_token_current": float(cur.mean()),
                "pcie_miss_slots_per_token_grouped": float(grp.mean()),
                "pcie_fetch_ratio_grouped_over_current": pcie_ratio,
            }

        gate = report_blocks[str(GATE_B)]["device_read_reduction_median"]
        p.update(
            {
                "status": "measured",
                "source_trace": str(NPZ),
                "counted_tokens": n_counted,
                "sessions": int(np.unique(session[keep]).size),
                "moe_layers": layers,
                "top_k": TOP_K,
                "deviation_counted_tokens_lt_10000": n_counted < 10000,
                "per_B": report_blocks,
                "gate": {
                    "metric": "median routed device-read bytes per token reduction at B=4",
                    "value": gate,
                    "threshold": GATE_REDUCTION,
                    "grouped_moe_opens": bool(gate >= GATE_REDUCTION),
                },
            }
        )
    except Exception as e:  # noqa: BLE001
        p.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(p, indent=2, allow_nan=False))
    return 0 if p.get("status") == "measured" else 2


if __name__ == "__main__":
    sys.exit(main())
