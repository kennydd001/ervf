#!/usr/bin/env python3
"""Transparent TreeSweep-200 throughput calculator.

V2 accepts either one verifier time (`verify_ms`) or explicit
`baseline_verify_ms` and `optimized_verify_ms`. The 250 tok/s hard stop is
applied only to the optimized verifier when supplied.

Example:
{
  "target_tok_s": 200,
  "draft_ms": 4.0,
  "points": [
    {"nodes": 32, "oracle_output_tokens": 8.2,
     "baseline_verify_ms": 52.0, "optimized_verify_ms": 34.0,
     "real_output_tokens": 7.1}
  ]
}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def tps(tokens: float, ms: float) -> float:
    return tokens / (ms / 1000.0) if tokens > 0 and ms > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", type=Path)
    args = ap.parse_args()
    d = json.loads(args.json_file.read_text())
    target = float(d.get("target_tok_s", 200))
    draft = float(d.get("draft_ms", 0.0))

    print(f"target: {target:.3f} tok/s; draft: {draft:.3f} ms")
    best_base = (0.0, None)
    best_opt = (0.0, None)
    best_real = (0.0, None)
    has_opt = False

    for p in d["points"]:
        nodes = int(p["nodes"])
        oracle_a = float(p["oracle_output_tokens"])
        real_a = float(p.get("real_output_tokens", 0.0))
        base_ms = float(p.get("baseline_verify_ms", p.get("verify_ms", 0.0)))
        opt_raw = p.get("optimized_verify_ms")
        opt_ms = float(opt_raw) if opt_raw is not None else base_ms
        has_opt = has_opt or opt_raw is not None

        base_tps = tps(oracle_a, base_ms)
        opt_tps = tps(oracle_a, opt_ms)
        real_tps = tps(real_a, draft + opt_ms)
        print(
            f"nodes={nodes:3d} A_oracle={oracle_a:7.3f} "
            f"base={base_ms:8.3f}ms/{base_tps:8.2f}tps "
            f"opt={opt_ms:8.3f}ms/{opt_tps:8.2f}tps "
            f"A_real={real_a:7.3f} integrated={real_tps:8.2f}tps"
        )
        if base_tps > best_base[0]:
            best_base = (base_tps, nodes)
        if opt_tps > best_opt[0]:
            best_opt = (opt_tps, nodes)
        if real_tps > best_real[0]:
            best_real = (real_tps, nodes)

    print(f"\nbest baseline target-only oracle: {best_base[0]:.2f} tok/s at nodes={best_base[1]}")
    print(f"best optimized target-only oracle: {best_opt[0]:.2f} tok/s at nodes={best_opt[1]}")
    print(f"best integrated real:               {best_real[0]:.2f} tok/s at nodes={best_real[1]}")
    if not has_opt:
        print("optimized verifier absent: baseline result is diagnostic only")
    print("optimized oracle 250 tok/s gate:", "PASS" if has_opt and best_opt[0] >= 250 else "NOT PASSED")
    print("integrated 200 tok/s gate:       ", "PASS" if best_real[0] >= 200 else "FAIL")


if __name__ == "__main__":
    main()
