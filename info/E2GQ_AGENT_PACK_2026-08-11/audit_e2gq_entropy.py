#!/usr/bin/env python3
"""Reproduce the E2GQ entropy audit from the supplied fleq_moe.zip.

This script does not assess model quality. It audits the actual symbol
histograms emitted by the locked 2-bit GPTQ P1 artifacts and calculates
lossless coding bounds plus explicit metadata projections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import zipfile
from pathlib import Path


SYMBOLS = (-2, -1, 0, 1)


def entropy(counts: dict[int, int]) -> float:
    n = sum(counts.values())
    return -sum((v / n) * math.log2(v / n) for v in counts.values() if v)


def multinomial_bits(counts: dict[int, int]) -> int:
    n = sum(counts.values())
    return math.ceil(
        (math.lgamma(n + 1) - sum(math.lgamma(v + 1) for v in counts.values()))
        / math.log(2)
    )


def align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    raw = args.zip_path.read_bytes()
    aggregate = {k: 0 for k in SYMBOLS}
    experts = []

    with zipfile.ZipFile(args.zip_path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.startswith("fleq_moe/p1_experts/layer_")
            and n.endswith(".json")
            and "attempt_" not in Path(n).name
        )
        if len(names) != 16:
            raise RuntimeError(f"Expected 16 locked experts, found {len(names)}")

        for name in names:
            d = json.loads(zf.read(name))
            counts = {k: 0 for k in SYMBOLS}
            weights = scales = enum_bits = conservative_bytes = 0

            for matrix in ("gate", "up", "down"):
                x = d["code_summaries"]["gptq_2bit"][matrix]
                c = {int(k): int(v) for k, v in x["histogram"].items()}
                n = int(x["weights"])
                s = int(x["scales"])
                H = entropy(c)

                enum_bits += multinomial_bits(c) + 16 * 8 + s * 16
                code_bytes = math.ceil(n * (H + 0.01) / 8) + 32
                conservative_bytes += align_up(code_bytes, 4096) + s * 2

                for k in SYMBOLS:
                    counts[k] += c[k]
                    aggregate[k] += c[k]
                weights += n
                scales += s

            H = entropy(counts)
            experts.append({
                "layer": d["layer"],
                "expert": d["expert"],
                "ideal_bpp": H + 16 * scales / weights,
                "enumerative_bpp": enum_bits / weights,
                "conservative_projected_bpp": conservative_bytes * 8 / weights,
            })

    n = sum(aggregate.values())
    H = entropy(aggregate)
    result = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "histogram": {str(k): aggregate[k] for k in SYMBOLS},
        "probabilities": {str(k): aggregate[k] / n for k in SYMBOLS},
        "code_entropy_bpp": H,
        "scale_overhead_bpp": 16 / 128,
        "ideal_total_bpp": H + 16 / 128,
        "all_experts_below_2bpp": all(x["ideal_bpp"] < 2 for x in experts),
        "expert_ideal_range": [
            min(x["ideal_bpp"] for x in experts),
            max(x["ideal_bpp"] for x in experts),
        ],
        "enumerative_mean_bpp": statistics.mean(
            x["enumerative_bpp"] for x in experts
        ),
        "conservative_projected_mean_bpp": statistics.mean(
            x["conservative_projected_bpp"] for x in experts
        ),
        "experts": experts,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
