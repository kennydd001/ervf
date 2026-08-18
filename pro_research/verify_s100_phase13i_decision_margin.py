from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, default=Path("pro_research/results/s100_phase13i/S100_PHASE13I_DECISION_MARGIN.json")); args = ap.parse_args(); result = json.loads(args.result.read_text(encoding="utf-8")); failures = []
    if result.get("status") != "measured": failures.append("status is not measured")
    if result.get("tokens", 0) < 100: failures.append("too few measured tokens")
    if len(result.get("aggregates", [])) != 3: failures.append("missing margin thresholds")
    if result.get("gates", {}).get("promotion_open") is not False: failures.append("promotion must remain closed")
    payload = {"kind": "verify_s100_phase13i_decision_margin", "status": "PASS" if not failures else "FAIL", "failures": failures, "promotion_open": False}; args.result.with_name("S100_PHASE13I_DECISION_MARGIN_VERIFY.json").write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2)); return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
