from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, default=Path("pro_research/results/s100_phase13d/S100_PHASE13D_NATIVE_BLOCK.json"))
    args = ap.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = []
    if result.get("status") != "measured":
        failures.append("status is not measured")
    if result.get("rotation_over_l2", 0) <= 4:
        failures.append("weight rotation is not >4x L2")
    blocks = result.get("per_block", {})
    for block in ("2", "4", "8"):
        if block not in blocks:
            failures.append(f"missing block {block}")
            continue
        speed = float(blocks[block].get("speedup_vs_custom_rowwise", float("nan")))
        if not math.isfinite(speed) or speed <= 0:
            failures.append(f"invalid speedup at B={block}")
        cases = blocks[block].get("case_results", [])
        if not cases or not all(case.get("error_vs_fp32_same_bf16_input") for case in cases):
            failures.append(f"missing errors at B={block}")
    if result.get("gates", {}).get("promotion_open") is not False:
        failures.append("native component must not promote without official quality")
    payload = {"kind": "verify_s100_phase13d_native_block", "status": "PASS" if not failures else "FAIL", "failures": failures, "promotion_open": False}
    output = args.result.with_name("S100_PHASE13D_NATIVE_BLOCK_VERIFY.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
