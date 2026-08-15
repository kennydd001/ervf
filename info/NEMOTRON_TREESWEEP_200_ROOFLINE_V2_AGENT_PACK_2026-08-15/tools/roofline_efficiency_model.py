#!/usr/bin/env python3
"""Transparent arithmetic for the N1-N5 exact-efficiency track.

This tool computes byte floors and labeled first-order projections only.
It never converts component timings into an integrated speed claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--target-tok-s", type=float, nargs="*", default=[50, 75, 100, 200])
    args = ap.parse_args()
    d = json.loads(args.measurements.read_text())
    bw = float(d["roofline"]["streaming_gb_s"])
    print(f"Measured streaming roofline: {bw:.3f} GB/s")
    for name, row in d["contexts"].items():
        b = float(row["compulsory_bytes"])
        floor_ms = b / (bw * 1e9) * 1000.0
        ceiling = 1000.0 / floor_ms
        print(f"\n{name}: bytes={b/2**20:.3f} MiB floor={floor_ms:.3f} ms ceiling={ceiling:.3f} tok/s")
        for t in args.target_tok_s:
            budget = 1000.0 / t
            required = b / (budget / 1000.0) / 1e9
            print(f"  {t:7.1f} tok/s -> {budget:7.3f} ms, {required:8.3f} GB/s ({required/bw*100:6.2f}% roofline)")
    if "n1" in d and "n2" in d:
        graph_ms = float(d["n1"]["graph_ms"])
        gather_ms = float(d["n2"]["gather_ms"])
        projected = graph_ms - gather_ms
        print("\nProjection only: graph - gather")
        print(f"  {graph_ms:.3f} - {gather_ms:.3f} = {projected:.3f} ms = {1000.0/projected:.3f} tok/s")
        print("  WARNING: integrated physical measurement required; costs may overlap.")


if __name__ == "__main__":
    main()
