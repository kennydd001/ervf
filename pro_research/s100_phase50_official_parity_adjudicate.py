"""Adjudicate the official Ornith Phase49 confirmation against Pottokao."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REPO, environment_snapshot, utc_now, write_json_atomic


RESULTS = REPO / "pro_research" / "results" / "s100_phase50"
PREREG = REPO / "pro_research" / "S100_PHASE50_OFFICIAL_ORNITH_PARITY_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase50_official_parity_adjudicate.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    official = json.loads(args.official.read_text("utf-8"))
    control = json.loads(args.control.read_text("utf-8"))
    official_rows = {int(row["multiplicity"]): row for row in official.get("records", [])}
    control_rows = {int(row["multiplicity"]): row for row in control.get("records", [])}
    ratios = {}
    for multiplicity in range(2, 9):
        a = float(official_rows[multiplicity]["candidate_timing_ms"]["p50"])
        b = float(control_rows[multiplicity]["candidate_timing_ms"]["p50"])
        ratios[f"M{multiplicity}"] = a / b
    gates = {
        "P50_G1_official_phase49_all_green": bool(
            official.get("status") == "measured_pass"
            and official.get("gates")
            and all(official["gates"].values())
        ),
        "P50_G2_shape_latency_parity": all(0.65 <= ratio <= 1.35 for ratio in ratios.values()),
        "P50_G3_first_beneficial_multiplicity_is_2": (
            official.get("beneficial_multiplicities", [])[:1] == [2]
        ),
    }
    payload = {
        "kind": "s100_phase50_official_ornith_parity",
        "status": "measured_pass" if all(gates.values()) else "measured_fail",
        "started_utc": utc_now(),
        "completed_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "official_result": str(args.official.resolve()),
        "control_result": str(args.control.resolve()),
        "official_status": official.get("status"),
        "official_gates": official.get("gates"),
        "candidate_median_ratio_official_over_pottokao": ratios,
        "gates": gates,
        "claim_boundary": (
            "official routed expert only; excludes vision, MTP, target decoder and DFlash acceptance"
        ),
        "environment": environment_snapshot((SCRIPT, PREREG)),
    }
    out = RESULTS / "S100_PHASE50_OFFICIAL_ORNITH_PARITY.json"
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({"status": payload["status"], "ratios": ratios, "gates": gates, "output": str(out)}, indent=2))
    return 0 if payload["status"] == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

