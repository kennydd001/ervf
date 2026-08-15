#!/usr/bin/env python3
"""Reproduce HERA tier counts and memory projections from e2gq_moe.zip."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path

LAYERS = 48
EXPERTS = 128
TOP_K = 8
THRESHOLD = 128
ROUTED_PARAMS = 28_991_029_248
NONEXPERT_PARAMS = 1_541_093_376
ENTROPY_BPP = 1.930709

def gib(params: int, bpp: float) -> float:
    return params * bpp / 8 / 2**30

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("zip_path", type=Path)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    counts = []
    with zipfile.ZipFile(args.zip_path) as zf:
        for layer in range(LAYERS):
            d = json.loads(zf.read(
                f"e2gq_moe/p0_capture_layers/layer_{layer:02d}.json"
            ))
            counts.extend(int(x) for x in d["router_counts"])

    hot = sum(x >= THRESHOLD for x in counts)
    cold = len(counts) - hot
    total_inv = sum(counts)
    cold_inv = sum(x for x in counts if x < THRESHOLD)
    params_per_expert = ROUTED_PARAMS // len(counts)

    hot_params = hot * params_per_expert
    cold_params = cold * params_per_expert
    out = {
        "source_sha256": hashlib.sha256(args.zip_path.read_bytes()).hexdigest(),
        "hot_experts": hot,
        "cold_experts": cold,
        "zero_experts": sum(x == 0 for x in counts),
        "cold_parameter_fraction": cold / len(counts),
        "cold_invocation_fraction": cold_inv / total_inv,
        "projected_hot_entropy_gib": gib(hot_params, ENTROPY_BPP),
        "projected_nonexpert_int4_gib": gib(NONEXPERT_PARAMS, 4),
        "projected_cold_bf16_gib": gib(cold_params, 16),
    }
    print(json.dumps(out, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
