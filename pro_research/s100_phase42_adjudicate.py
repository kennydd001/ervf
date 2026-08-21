"""Adjudicate the frozen Phase42 B3 screen."""
from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase42"


def load(arm: str) -> dict:
    return json.loads(
        (RESULTS / f"S100_PHASE42_{arm}_CTX1024.json").read_text(encoding="utf-8")
    )


def main() -> int:
    compile_result = load("COMPILE")
    serial = load("SERIAL_SMOKE")
    overlap = load("OVERLAP_SMOKE")
    arms = {name: load(name) for name in ("BASE_A", "GLOBAL_PIPELINE_B3", "BASE_B")}
    medians = {name: float(row["summary"]["median_ms"]) for name, row in arms.items()}
    midpoint = (medians["BASE_A"] + medians["BASE_B"]) / 2.0
    drift = abs(medians["BASE_A"] - medians["BASE_B"]) / midpoint
    gain = (midpoint - medians["GLOBAL_PIPELINE_B3"]) / midpoint
    resource_rows = [
        row
        for family in compile_result["kernel_resources"].values()
        for row in family.values()
    ]
    zero_local = all(int(row.get("local_size_bytes") or 0) == 0 for row in resource_rows)
    exact = all(
        row.get("status") == "measured" and row.get("tokens_exact") is True
        for row in (*arms.values(), serial, overlap)
    )
    gates = {
        "G42_C1_smokes_and_arms_exact": exact,
        "G42_R1_zero_local_memory": zero_local,
        "G42_D1_baseline_drift_le_5pct": drift <= 0.05,
        "G42_P1_gain_ge_3pct": gain >= 0.03,
        "G42_P2_candidate_le_120ms": medians["GLOBAL_PIPELINE_B3"] <= 120.0,
    }
    promoted = all((
        gates["G42_C1_smokes_and_arms_exact"],
        gates["G42_R1_zero_local_memory"],
        gates["G42_D1_baseline_drift_le_5pct"],
        gates["G42_P1_gain_ge_3pct"],
    ))
    payload = {
        "kind": "s100_phase42_adjudication",
        "status": "promotion_candidate" if promoted else "gate_failed",
        "created_utc": utc_now(),
        "median_ms": medians,
        "tok_s": {name: 8000.0 / value for name, value in medians.items()},
        "baseline_midpoint_ms": midpoint,
        "baseline_drift_fraction": drift,
        "candidate_gain_ms": midpoint - medians["GLOBAL_PIPELINE_B3"],
        "candidate_gain_fraction": gain,
        "gates": gates,
        "claim_boundary": "exact synchronous target-only H8 screen",
    }
    write_json_atomic(RESULTS / "S100_PHASE42_ADJUDICATION.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())

