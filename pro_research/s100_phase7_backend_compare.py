
"""Independent packed-backend comparison."""
from __future__ import annotations
import argparse
import json
import traceback

from common import REPO, first_divergence, utc_now, write_json_atomic


def path(mode, role):
    return (
        REPO / "pro_research" / "results"
        / f"S100_PHASE7_PACKED_{mode.upper()}_{role.upper()}.json"
    )


def load(mode, role):
    p = path(mode, role)
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
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = ap.parse_args()

    out = (
        REPO / "pro_research" / "results"
        / f"S100_PHASE7_PACKED_COMPARE_{args.mode.upper()}.json"
    )
    payload = {
        "kind": "s100_phase7_backend_compare",
        "status": "started",
        "mode": args.mode,
        "created_utc": utc_now(),
    }

    try:
        roles = ("base_a", "cand_a", "cand_b", "base_b")
        d = {role: load(args.mode, role) for role in roles}
        p50 = {r: float(d[r]["timing"]["p50"]) for r in roles}
        base = 0.5 * (p50["base_a"] + p50["base_b"])
        cand = 0.5 * (p50["cand_a"] + p50["cand_b"])
        base_drift = abs(p50["base_a"] - p50["base_b"])
        cand_drift = abs(p50["cand_a"] - p50["cand_b"])

        base_div = divergence(d["base_a"]["ids"], d["base_b"]["ids"])
        repeat_div = divergence(d["cand_a"]["ids"], d["cand_b"]["ids"])
        exact_div = divergence(d["base_a"]["ids"], d["cand_a"]["ids"])

        control_div = {}
        control_ok = True
        if args.mode == "smoke":
            bad = load(args.mode, "bad")
            control_div = divergence(
                d["base_a"]["ids"], bad["ids"]
            )
            control_ok = any(v is not None for v in control_div.values())
        else:
            smoke = json.loads(
                (
                    REPO / "pro_research" / "results"
                    / "S100_PHASE7_PACKED_COMPARE_SMOKE.json"
                ).read_text(encoding="utf-8")
            )
            control_ok = bool(
                smoke.get("gates", {}).get("G5_control_diverges")
            )

        gates = {
            "G1_base_repeat": all(v is None for v in base_div.values()),
            "G2_candidate_repeat": all(
                v is None for v in repeat_div.values()
            ),
            "G3_packed_exact": all(
                v is None for v in exact_div.values()
            ),
            "G4_finite": bool(
                d["cand_a"]["finite"] and d["cand_b"]["finite"]
            ),
            "G5_control_diverges": control_ok,
            "M1_base_drift": base_drift <= 1.0,
            "M2_candidate_drift": cand_drift <= 1.0,
            "M3_samples": (
                True
                if args.mode == "smoke"
                else all(
                    int(d[r]["timing"]["count"]) >= 765
                    for r in roles
                )
            ),
            "M4_vram": all(
                int(d[r]["vram_mib"]) <= 7987 for r in roles
            ),
        }
        instrument = all(
            gates[k]
            for k in (
                "G1_base_repeat",
                "G2_candidate_repeat",
                "G3_packed_exact",
                "G4_finite",
                "G5_control_diverges",
            )
        )
        stable = instrument and all(
            gates[k]
            for k in (
                "M1_base_drift",
                "M2_candidate_drift",
                "M3_samples",
                "M4_vram",
            )
        )
        saving = base - cand
        status = (
            "instrument_failed"
            if not instrument
            else "measurement_failed"
            if not stable
            else "exact_backend_candidate"
            if saving >= 0.15
            else "exact_backend_below_gate"
        )
        payload.update(
            {
                "status": status,
                "summary": {
                    "legacy_midpoint_ms": base,
                    "packed_midpoint_ms": cand,
                    "saving_ms": saving,
                    "packed_tok_s": 1000.0 / cand,
                    "base_drift_ms": base_drift,
                    "candidate_drift_ms": cand_drift,
                },
                "divergence": {
                    "base": base_div,
                    "candidate_repeat": repeat_div,
                    "packed_vs_legacy": exact_div,
                    "control": control_div,
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
    return 2 if payload.get("status") in {
        "technical_failure", "instrument_failed"
    } else 0


if __name__ == "__main__":
    raise SystemExit(main())
