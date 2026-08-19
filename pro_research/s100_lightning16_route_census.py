from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from common import write_json_atomic, utc_now
from s100_lightning16_common import (
    RESULTS, assert_lightning, ensure_results,
)

BLOCKS = (2, 4, 8)
TOP_K = 6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()
    ensure_results()
    trace_path = Path(args.trace)
    output = RESULTS / "S100_LIGHTNING16_ROUTE_CENSUS.json"
    payload = {
        "kind": "s100_lightning16_route_census",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        ident = assert_lightning()
        with np.load(trace_path) as data:
            ids = data["ids"].astype(np.int64)
            need = data["need"].astype(bool)
            counted = data["counted"].astype(bool)
            session = data["session"].astype(np.int64)
            layers = [int(x) for x in data["layers"]]

        keep = np.nonzero(counted)[0]
        _, layer_count, top_k = ids.shape
        if top_k != TOP_K:
            raise RuntimeError(f"top_k={top_k}")
        report = {}

        for block in BLOCKS:
            blocks = []
            for sid in np.unique(session[keep]):
                rows = keep[session[keep] == sid]
                for offset in range(
                    0, rows.size - block + 1, block
                ):
                    blocks.append(rows[offset:offset + block])
            union_sizes = np.zeros(
                (len(blocks), layer_count), np.int32
            )
            grouped_miss = np.zeros_like(union_sizes)
            current_miss = np.zeros_like(union_sizes)
            histogram = np.zeros(block + 1, np.int64)

            for bi, rows in enumerate(blocks):
                block_ids = ids[rows]
                block_need = need[rows]
                for li in range(layer_count):
                    flat = block_ids[:, li].reshape(-1)
                    unique, first, counts = np.unique(
                        flat,
                        return_index=True,
                        return_counts=True,
                    )
                    union_sizes[bi, li] = len(unique)
                    for count in counts:
                        histogram[count] += 1
                    current_miss[bi, li] = int(
                        block_need[:, li].sum()
                    )
                    grouped_miss[bi, li] = int(
                        block_need[:, li].reshape(-1)[first].sum()
                    )

            reduction = 1.0 - (
                union_sizes.astype(np.float64)
                / (block * TOP_K)
            )
            current = current_miss.sum(axis=1) / block
            grouped = grouped_miss.sum(axis=1) / block
            report[str(block)] = {
                "blocks": len(blocks),
                "tokens": len(blocks) * block,
                "device_read_reduction_mean": float(
                    reduction.mean()
                ),
                "device_read_reduction_median": float(
                    np.median(reduction)
                ),
                "device_read_reduction_p10": float(
                    np.percentile(reduction, 10)
                ),
                "rows_per_expert_hist": [
                    int(x) for x in histogram[1:]
                ],
                "rows_per_expert_mean": float(
                    sum(
                        index * histogram[index]
                        for index in range(1, block + 1)
                    ) / max(histogram[1:].sum(), 1)
                ),
                "pcie_miss_slots_per_token_current": float(
                    current.mean()
                ),
                "pcie_miss_slots_per_token_grouped": float(
                    grouped.mean()
                ),
            }

        payload.update({
            "status": "measured",
            "identity": ident,
            "source_trace": str(trace_path),
            "counted_tokens": int(counted.sum()),
            "sessions": int(np.unique(session[keep]).size),
            "moe_layers": layers,
            "top_k": top_k,
            "per_B": report,
            "LIGHTNING_GROUPED_MOE_REUSE_GATE": bool(
                report["4"]["device_read_reduction_median"]
                >= 0.20
            ),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    write_json_atomic(output, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
