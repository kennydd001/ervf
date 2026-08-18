from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, default=Path("pro_research/results/s100_phase13f/S100_PHASE13F_SUBSPACE_RESIDUAL.json"))
    args = ap.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures = []
    records = result.get("records", [])
    if result.get("status") != "measured":
        failures.append("status is not measured")
    if len(records) != 36:
        failures.append(f"expected 36 layer/rank records, got {len(records)}")
    for row in records:
        for key in ("validation_residual_energy_mean", "output_nrmse_mean_no_fallback", "projected_weight_read_fraction"):
            if not math.isfinite(float(row.get(key, float("nan")))):
                failures.append(f"non-finite {key}")
        if len(row.get("gates", [])) != 4:
            failures.append("missing fallback gates")
    if result.get("gates", {}).get("promotion_open") is not False:
        failures.append("promotion must remain closed")
    payload = {"kind": "verify_s100_phase13f_subspace_residual", "status": "PASS" if not failures else "FAIL", "failures": failures, "promotion_open": False}
    output = args.result.with_name("S100_PHASE13F_SUBSPACE_RESIDUAL_VERIFY.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
