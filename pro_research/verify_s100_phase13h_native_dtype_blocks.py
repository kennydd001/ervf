from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, default=Path("pro_research/results/s100_phase13h/S100_PHASE13H_NATIVE_DTYPE_BLOCKS.json")); args = ap.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8")); failures = []
    if result.get("status") != "measured": failures.append("status is not measured")
    cases = result.get("cases", [])
    if len(cases) < 3: failures.append("expected at least three available datatype cases")
    if not any(row.get("kind") == "bf16" for row in cases): failures.append("missing BF16 attention case")
    if not any(row.get("kind") == "nvfp4" for row in cases): failures.append("missing NVFP4 case")
    if result.get("gates", {}).get("promotion_open") is not False: failures.append("promotion must remain closed")
    payload = {"kind": "verify_s100_phase13h_native_dtype_blocks", "status": "PASS" if not failures else "FAIL", "failures": failures, "promotion_open": False}
    args.result.with_name("S100_PHASE13H_NATIVE_DTYPE_BLOCKS_VERIFY.json").write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2)); return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
