from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, default=Path("pro_research/results/s100_phase13g/S100_PHASE13G_ENTROPY_CODEC.json")); args = ap.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8")); failures = []
    if result.get("status") != "measured": failures.append("status is not measured")
    records = result.get("records", [])
    if len(records) != 18: failures.append(f"expected 18 codec records, got {len(records)}")
    if not records or not all(r.get("roundtrip_exact") for r in records): failures.append("not all roundtrips are exact")
    if result.get("gates", {}).get("promotion_open") is not False: failures.append("promotion must remain closed")
    payload = {"kind": "verify_s100_phase13g_entropy_codec", "status": "PASS" if not failures else "FAIL", "failures": failures, "promotion_open": False}
    args.result.with_name("S100_PHASE13G_ENTROPY_CODEC_VERIFY.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2)); return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
