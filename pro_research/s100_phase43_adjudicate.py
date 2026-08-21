"""Adjudicate Phase43's frozen B2/B4 geometry screen."""
from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase43"


def load(arm: str) -> dict:
    return json.loads((RESULTS / f"S100_PHASE43_{arm}_CTX1024.json").read_text(encoding="utf-8"))


def main() -> int:
    arms = {name: load(name) for name in ("BASE_A", "GLOBAL_B2", "GLOBAL_B4", "BASE_B")}
    med = {name: float(row["summary"]["median_ms"]) for name, row in arms.items()}
    midpoint = (med["BASE_A"] + med["BASE_B"]) / 2.0
    drift = abs(med["BASE_A"] - med["BASE_B"]) / midpoint
    candidates = {
        name: {
            "median_ms": med[name],
            "tok_s": 8000.0 / med[name],
            "gain_fraction": (midpoint - med[name]) / midpoint,
        }
        for name in ("GLOBAL_B2", "GLOBAL_B4")
    }
    selected = min(candidates, key=lambda name: candidates[name]["median_ms"])
    exact = all(row.get("status") == "measured" and row.get("tokens_exact") is True for row in arms.values())
    gates = {
        "G43_C1_all_tokens_exact": exact,
        "G43_D1_baseline_drift_le_5pct": drift <= 0.05,
        "G43_P1_selected_gain_ge_3pct": candidates[selected]["gain_fraction"] >= 0.03,
        "G43_P2_selected_le_120ms": candidates[selected]["median_ms"] <= 120.0,
    }
    promoted = gates["G43_C1_all_tokens_exact"] and gates["G43_D1_baseline_drift_le_5pct"] and gates["G43_P1_selected_gain_ge_3pct"]
    payload = {
        "kind": "s100_phase43_adjudication",
        "status": "promotion_candidate" if promoted else "gate_failed",
        "created_utc": utc_now(),
        "baseline_midpoint_ms": midpoint,
        "baseline_drift_fraction": drift,
        "candidates": candidates,
        "selected": selected,
        "gates": gates,
        "claim_boundary": "exact synchronous target-only H8 geometry screen",
    }
    write_json_atomic(RESULTS / "S100_PHASE43_ADJUDICATION.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())

