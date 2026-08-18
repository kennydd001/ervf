
"""Compare fresh candidate timing and exact backend parity."""
from __future__ import annotations
import argparse
import json
import traceback

from common import REPO, first_divergence, utc_now, write_json_atomic

ROLES = (
    "base_a",
    "legacy_cand",
    "cand_a",
    "cand_b",
    "base_b",
)


def path(candidate, role):
    return (
        REPO / "pro_research" / "results"
        / f"S100_PHASE7_TIMING_{candidate.upper()}_{role.upper()}.json"
    )


def load(candidate, role):
    p = path(candidate, role)
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "measured":
        raise RuntimeError(f"{p}: {d.get('status')}")
    return d


def divergence(a, b):
    return {
        key: first_divergence(a[key], b[key])
        for key in a
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    out = (
        REPO / "pro_research" / "results"
        / f"S100_PHASE7_TIMING_COMPARE_{args.candidate.upper()}.json"
    )
    payload = {
        "kind": "s100_phase7_candidate_compare",
        "status": "started",
        "candidate": args.candidate,
        "created_utc": utc_now(),
    }

    try:
        d = {r: load(args.candidate, r) for r in ROLES}
        v = {r: float(d[r]["timing"]["p50"]) for r in ROLES}
        base = 0.5 * (v["base_a"] + v["base_b"])
        cand = 0.5 * (v["cand_a"] + v["cand_b"])
        base_drift = abs(v["base_a"] - v["base_b"])
        cand_drift = abs(v["cand_a"] - v["cand_b"])

        base_div = divergence(
            d["base_a"]["ids"], d["base_b"]["ids"]
        )
        repeat_div = divergence(
            d["cand_a"]["ids"], d["cand_b"]["ids"]
        )
        backend_div = divergence(
            d["legacy_cand"]["ids"], d["cand_a"]["ids"]
        )
        gates = {
            "G1_base_repeat": all(
                x is None for x in base_div.values()
            ),
            "G2_candidate_repeat": all(
                x is None for x in repeat_div.values()
            ),
            "G3_backend_exact_for_candidate": all(
                x is None for x in backend_div.values()
            ),
            "G4_finite": bool(
                d["cand_a"]["finite"]
                and d["cand_b"]["finite"]
                and d["legacy_cand"]["finite"]
            ),
            "M1_base_drift": base_drift <= 1.0,
            "M2_candidate_drift": cand_drift <= 1.0,
            "M3_samples": all(
                int(d[r]["timing"]["count"]) >= 765 for r in ROLES
            ),
            "M4_vram": all(
                int(d[r]["vram_mib"]) <= 7987 for r in ROLES
            ),
        }
        status = (
            "fresh_timing_candidate"
            if all(gates.values())
            else "measurement_failed"
        )
        payload.update(
            {
                "status": status,
                "selected_backend": d["cand_a"].get(
                    "selected_backend", "legacy"
                ),
                "summary": {
                    "legacy_qfast_midpoint_ms": base,
                    "legacy_candidate_ms": v["legacy_cand"],
                    "candidate_midpoint_ms": cand,
                    "saving_vs_qfast_ms": base - cand,
                    "candidate_tok_s": 1000.0 / cand,
                    "remaining_ms_to_s100": cand - 10.0,
                    "base_drift_ms": base_drift,
                    "candidate_drift_ms": cand_drift,
                },
                "divergence": {
                    "base": base_div,
                    "candidate_repeat": repeat_div,
                    "selected_backend_vs_legacy_candidate": backend_div,
                },
                "gates": gates,
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )

    write_json_atomic(out, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 2 if payload.get("status") == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
