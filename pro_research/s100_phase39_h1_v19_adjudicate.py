"""Adjudicate the three frozen Phase39 H1 arms."""
from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase39"


def _load(arm: str) -> dict:
    path = RESULTS / f"S100_PHASE39_{arm}_CTX1024.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    arms = {name: _load(name) for name in ("BASE_A", "V19", "BASE_B")}
    med = {
        name: float(payload["timing"]["median_ms"])
        for name, payload in arms.items()
    }
    exact = all(
        payload.get("status") == "measured" and payload.get("tokens_exact") is True
        for payload in arms.values()
    )
    fits = (
        arms["V19"].get("status") == "measured"
        and int((arms["V19"].get("capture") or {}).get("free_after_capture_bytes", -1)) >= 0
    )
    drift = abs(med["BASE_A"] - med["BASE_B"])
    midpoint = (med["BASE_A"] + med["BASE_B"]) / 2.0
    gain_fraction = (midpoint - med["V19"]) / midpoint
    gates = {
        "G39_C1_all_tokens_exact": exact,
        "G39_D1_baseline_drift_le_1ms": drift <= 1.0,
        "G39_P1_v19_gain_ge_5pct": gain_fraction >= 0.05,
        "G39_P2_v19_fits_at_cache72": fits,
    }
    if not exact:
        status = "correctness_failed"
    elif not fits:
        status = "infeasible_vram"
    elif not gates["G39_D1_baseline_drift_le_1ms"]:
        status = "measurement_unstable"
    elif gates["G39_P1_v19_gain_ge_5pct"]:
        status = "adoption_candidate"
    else:
        status = "gate_failed"

    payload = {
        "kind": "s100_phase39_h1_v19_adjudication",
        "status": status,
        "created_utc": utc_now(),
        "preregistration": "pro_research/S100_PHASE39_H1_V19_PREREGISTRATION.md",
        "median_ms": med,
        "tok_s": {name: 1000.0 / value for name, value in med.items()},
        "baseline_midpoint_ms": midpoint,
        "baseline_drift_ms": drift,
        "v19_gain_ms": midpoint - med["V19"],
        "v19_gain_fraction": gain_fraction,
        "gates": gates,
        "claim_boundary": (
            "Exact synchronous H1 target-model decode at context 1024; this does not "
            "supersede a faster H4 verifier or imply speculative-decoding throughput."
        ),
    }
    out = RESULTS / "S100_PHASE39_H1_V19_ADJUDICATION.json"
    write_json_atomic(out, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if status == "adoption_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())

