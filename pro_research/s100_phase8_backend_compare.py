
"""Independent comparison for one static-cache budget."""
from __future__ import annotations

import argparse
import json
import traceback

from common import REPO, first_divergence, utc_now, write_json_atomic
from s100_phase8_common import BUDGETS, load_profile


ROLES = ("base_a", "cand_a", "cand_b", "base_b")


def path(budget, mode, role):
    return (
        REPO / "pro_research" / "results"
        / (
            f"S100_PHASE8_STATIC_{budget}_"
            f"{mode.upper()}_{role.upper()}.json"
        )
    )


def load(budget, mode, role):
    p = path(budget, mode, role)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("status") != "measured":
        raise RuntimeError(f"{p}: {data.get('status')}")
    return data


def divergence(a, b):
    return {
        key: first_divergence(a[key], b[key])
        for key in a
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--budget",
        type=int,
        choices=BUDGETS,
        required=True,
    )
    ap.add_argument(
        "--mode",
        choices=("smoke", "full"),
        required=True,
    )
    args = ap.parse_args()

    out = (
        REPO / "pro_research" / "results"
        / (
            f"S100_PHASE8_STATIC_COMPARE_{args.budget}_"
            f"{args.mode.upper()}.json"
        )
    )
    payload = {
        "kind": "s100_phase8_backend_compare",
        "status": "started",
        "budget": args.budget,
        "mode": args.mode,
        "created_utc": utc_now(),
    }

    try:
        d = {
            role: load(args.budget, args.mode, role)
            for role in ROLES
        }
        value = {
            role: float(d[role]["timing"]["p50"])
            for role in ROLES
        }
        base = 0.5 * (
            value["base_a"] + value["base_b"]
        )
        candidate = 0.5 * (
            value["cand_a"] + value["cand_b"]
        )
        base_drift = abs(
            value["base_a"] - value["base_b"]
        )
        candidate_drift = abs(
            value["cand_a"] - value["cand_b"]
        )

        base_div = divergence(
            d["base_a"]["ids"], d["base_b"]["ids"]
        )
        repeat_div = divergence(
            d["cand_a"]["ids"], d["cand_b"]["ids"]
        )
        exact_div = divergence(
            d["base_a"]["ids"], d["cand_a"]["ids"]
        )

        control_ok = True
        control_div = {}
        if args.mode == "smoke":
            bad = load(args.budget, args.mode, "bad")
            control_div = divergence(
                d["base_a"]["ids"], bad["ids"]
            )
            control_ok = any(
                x is not None for x in control_div.values()
            )
        else:
            smoke = json.loads(
                (
                    REPO / "pro_research" / "results"
                    / (
                        "S100_PHASE8_STATIC_COMPARE_"
                        f"{args.budget}_SMOKE.json"
                    )
                ).read_text(encoding="utf-8")
            )
            control_ok = bool(
                smoke.get("gates", {}).get(
                    "G5_control_diverges"
                )
            )

        profile = load_profile()
        expected_hash = profile["selections"][
            str(args.budget)
        ]["selection_sha256"]
        hash_ok = all(
            d[role].get("actual_selection_sha256")
            == expected_hash
            for role in ("cand_a", "cand_b")
        )

        gates = {
            "G1_base_repeat": all(
                x is None for x in base_div.values()
            ),
            "G2_candidate_repeat": all(
                x is None for x in repeat_div.values()
            ),
            "G3_static_exact": all(
                x is None for x in exact_div.values()
            ),
            "G4_finite": bool(
                d["cand_a"]["finite"] and d["cand_b"]["finite"]
            ),
            "G5_control_diverges": control_ok,
            "G6_selection_hash": hash_ok,
            "M1_base_drift": base_drift <= 1.0,
            "M2_candidate_drift": candidate_drift <= 1.0,
            "M3_samples": (
                True
                if args.mode == "smoke"
                else all(
                    int(d[role]["timing"]["count"]) >= 765
                    for role in ROLES
                )
            ),
            "M4_vram": all(
                int(d[role]["vram_mib"]) <= 7987
                for role in ROLES
            ),
        }

        instrument = all(
            gates[key]
            for key in (
                "G1_base_repeat",
                "G2_candidate_repeat",
                "G3_static_exact",
                "G4_finite",
                "G5_control_diverges",
                "G6_selection_hash",
            )
        )
        stable = instrument and all(
            gates[key]
            for key in (
                "M1_base_drift",
                "M2_candidate_drift",
                "M3_samples",
                "M4_vram",
            )
        )
        saving = base - candidate
        status = (
            "instrument_failed"
            if not instrument
            else "measurement_failed"
            if not stable
            else "exact_backend_candidate"
            if saving >= 0.15
            else "exact_backend_below_gate"
        )

        route = profile["selections"][str(args.budget)]
        payload.update(
            {
                "status": status,
                "route_profile": {
                    "physical_mib": route["physical_mib"],
                    "calibration_hit_rate": route[
                        "calibration"
                    ]["hit_rate"],
                    "validation_hit_rate": route[
                        "validation"
                    ]["hit_rate"],
                    "selection_sha256": expected_hash,
                },
                "summary": {
                    "legacy_thr0020_midpoint_ms": base,
                    "static_midpoint_ms": candidate,
                    "saving_ms": saving,
                    "static_tok_s": 1000.0 / candidate,
                    "remaining_ms_to_s100": candidate - 10.0,
                    "base_drift_ms": base_drift,
                    "candidate_drift_ms": candidate_drift,
                },
                "divergence": {
                    "base": base_div,
                    "candidate_repeat": repeat_div,
                    "static_vs_legacy": exact_div,
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
