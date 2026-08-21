"""Adjudicate the frozen Phase40 screen."""
from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase40"


def load(arm: str) -> dict:
    path = RESULTS / f"S100_PHASE40_{arm}_CTX1024.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    compile_result = load("COMPILE")
    arms = {name: load(name) for name in ("BASE_A", "PIPELINE_B3", "BASE_B")}
    medians = {name: float(row["summary"]["median_ms"]) for name, row in arms.items()}
    midpoint = (medians["BASE_A"] + medians["BASE_B"]) / 2.0
    drift_fraction = abs(medians["BASE_A"] - medians["BASE_B"]) / midpoint
    gain_fraction = (midpoint - medians["PIPELINE_B3"]) / midpoint
    resources = compile_result["kernel_resources"]
    zero_local = all(int(row.get("local_size_bytes") or 0) == 0 for row in resources.values())
    exact = all(
        row.get("status") == "measured" and row.get("tokens_exact") is True
        for row in arms.values()
    )
    gates = {
        "G40_C1_all_tokens_exact": exact,
        "G40_R1_zero_local_memory": zero_local,
        "G40_D1_baseline_drift_le_5pct": drift_fraction <= 0.05,
        "G40_P1_gain_ge_3pct": gain_fraction >= 0.03,
        "G40_P2_candidate_le_120ms": medians["PIPELINE_B3"] <= 120.0,
    }
    required = (
        gates["G40_C1_all_tokens_exact"]
        and gates["G40_R1_zero_local_memory"]
        and gates["G40_D1_baseline_drift_le_5pct"]
        and gates["G40_P1_gain_ge_3pct"]
    )
    payload = {
        "kind": "s100_phase40_adjudication",
        "status": "promotion_candidate" if required else "gate_failed",
        "created_utc": utc_now(),
        "median_ms": medians,
        "tok_s": {name: 8000.0 / value for name, value in medians.items()},
        "baseline_midpoint_ms": midpoint,
        "baseline_drift_fraction": drift_fraction,
        "candidate_gain_ms": midpoint - medians["PIPELINE_B3"],
        "candidate_gain_fraction": gain_fraction,
        "kernel_resources": resources,
        "gates": gates,
        "claim_boundary": "exact synchronous target-only H8 screen",
    }
    write_json_atomic(RESULTS / "S100_PHASE40_ADJUDICATION.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if required else 2


if __name__ == "__main__":
    raise SystemExit(main())

