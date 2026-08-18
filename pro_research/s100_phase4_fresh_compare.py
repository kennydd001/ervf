
"""Independent CPU comparison of four fresh phase-4 timing processes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    REPO,
    first_divergence,
    percentiles,
    utc_now,
    write_json_atomic,
)

PROFILES = (
    "qfast", "mamba", "fast", "k5", "k4", "fast_k5", "fast_k4"
)
ROLES = ("exact_a", "cand_a", "cand_b", "exact_b")
EXPECTED_TOP_K = {
    "qfast": 6,
    "mamba": 6,
    "fast": 6,
    "k5": 5,
    "k4": 4,
    "fast_k5": 5,
    "fast_k4": 4,
}
RULES = {
    "qfast": ("gain", 0.40),
    "mamba": ("gain", 0.70),
    "fast": ("absolute", 17.80),
    "k5": ("gain", 0.30),
    "k4": ("gain", 0.60),
    "fast_k5": ("absolute", 17.60),
    "fast_k4": ("absolute", 17.20),
}


def _path(profile: str, mode: str, role: str) -> Path:
    return (
        REPO / "pro_research" / "results"
        / (
            f"S100_PHASE4_FRESH_{profile.upper()}_"
            f"{mode.upper()}_{role.upper()}.json"
        )
    )


def _load(profile: str, mode: str, role: str) -> dict[str, Any]:
    p = _path(profile, mode, role)
    if not p.exists():
        raise FileNotFoundError(p)
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "measured":
        raise RuntimeError(f"{p}: status={d.get('status')}")
    if d.get("series_profile") != profile:
        raise RuntimeError(f"{p}: wrong series profile")
    if d.get("mode") != mode or d.get("role") != role:
        raise RuntimeError(f"{p}: mode/role mismatch")
    return d


def _ids(d: dict[str, Any]) -> dict[str, list[int]]:
    return {
        str(p["prompt"]): [int(x) for x in p["ids"]]
        for p in d["prompts"]
    }


def _metric(d: dict[str, Any]) -> dict[str, Any]:
    raw = [float(x) for x in d["raw_timing_ms"]]
    return percentiles(raw)


def _rule(profile: str, saving: float, candidate: float):
    kind, threshold = RULES[profile]
    if kind == "gain":
        return (
            saving >= threshold,
            {
                "kind": "gain_ms_at_least",
                "threshold": threshold,
                "observed": saving,
            },
        )
    return (
        candidate <= threshold,
        {
            "kind": "candidate_ms_at_most",
            "threshold": threshold,
            "observed": candidate,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = ap.parse_args()

    out = (
        REPO / "pro_research" / "results"
        / f"S100_PHASE4_FRESH_COMPARE_{args.profile.upper()}_{args.mode.upper()}.json"
    )
    payload: dict[str, Any] = {
        "kind": "s100_phase4_fresh_compare",
        "status": "started",
        "profile": args.profile,
        "mode": args.mode,
        "created_utc": utc_now(),
        "claim_boundary": (
            "fresh-process timing comparison; no quality or S100 claim"
        ),
    }

    try:
        d = {
            role: _load(args.profile, args.mode, role)
            for role in ROLES
        }
        timing = {role: _metric(x) for role, x in d.items()}
        p = {
            role: float(timing[role]["p50"])
            for role in ROLES
        }
        base_mid = 0.5 * (p["exact_a"] + p["exact_b"])
        cand_mid = 0.5 * (p["cand_a"] + p["cand_b"])
        base_drift = abs(p["exact_a"] - p["exact_b"])
        cand_drift = abs(p["cand_a"] - p["cand_b"])
        saving = base_mid - cand_mid

        exact_a_ids = _ids(d["exact_a"])
        exact_b_ids = _ids(d["exact_b"])
        cand_a_ids = _ids(d["cand_a"])
        cand_b_ids = _ids(d["cand_b"])
        base_div = {
            k: first_divergence(exact_a_ids[k], exact_b_ids[k])
            for k in exact_a_ids
        }
        cand_div = {
            k: first_divergence(cand_a_ids[k], cand_b_ids[k])
            for k in cand_a_ids
        }
        cand_vs_base = {
            k: first_divergence(exact_a_ids[k], cand_a_ids[k])
            for k in exact_a_ids
        }

        expected_top_k = EXPECTED_TOP_K[args.profile]
        exact_top_k = [
            int(d[r]["runtime"]["top_k"])
            for r in ("exact_a", "exact_b")
        ]
        cand_top_k = [
            int(d[r]["runtime"]["top_k"])
            for r in ("cand_a", "cand_b")
        ]
        full_count_ok = all(
            int(timing[r]["count"]) >= 765
            for r in ROLES
        ) if args.mode == "full" else True
        all_vram = {
            role: int(d[role].get("vram_mib", 10**9))
            for role in ROLES
        }
        info_pass, info = _rule(args.profile, saving, cand_mid)

        gates = {
            "G1_exact_A_B_token_parity": all(
                x is None for x in base_div.values()
            ),
            "G2_candidate_A_B_token_parity": all(
                x is None for x in cand_div.values()
            ),
            "G3_candidate_finite": bool(
                d["cand_a"].get("finite") and d["cand_b"].get("finite")
            ),
            "G4_exact_top_k_is_6": exact_top_k == [6, 6],
            "G5_candidate_top_k_matches_profile": (
                cand_top_k == [expected_top_k, expected_top_k]
            ),
            "M1_exact_process_drift_le_1ms": base_drift <= 1.0,
            "M2_candidate_process_drift_le_1ms": cand_drift <= 1.0,
            "M3_all_arms_vram_le_7987MiB": all(
                v <= 7987 for v in all_vram.values()
            ),
            "M4_full_samples_ge_765": full_count_ok,
            "P1_confirmatory_information_gate": info_pass,
        }
        instrument_ok = all(
            gates[k]
            for k in (
                "G1_exact_A_B_token_parity",
                "G2_candidate_A_B_token_parity",
                "G3_candidate_finite",
                "G4_exact_top_k_is_6",
                "G5_candidate_top_k_matches_profile",
            )
        )
        measurement_ok = all(
            gates[k]
            for k in (
                "M1_exact_process_drift_le_1ms",
                "M2_candidate_process_drift_le_1ms",
                "M3_all_arms_vram_le_7987MiB",
                "M4_full_samples_ge_765",
            )
        )
        status = (
            "instrument_failed"
            if not instrument_ok
            else "measurement_unstable"
            if not measurement_ok
            else "fresh_timing_candidate"
            if info_pass
            else "fresh_timing_below_gate"
        )

        payload.update(
            {
                "status": status,
                "arms": {
                    role: {
                        "path": str(_path(args.profile, args.mode, role).relative_to(REPO)),
                        "timing": timing[role],
                        "vram_mib": all_vram[role],
                        "smi_before": d[role].get("smi_before"),
                        "smi_after": d[role].get("smi_after"),
                    }
                    for role in ROLES
                },
                "summary": {
                    "exact_a_p50_ms": p["exact_a"],
                    "exact_b_p50_ms": p["exact_b"],
                    "exact_midpoint_ms": base_mid,
                    "candidate_a_p50_ms": p["cand_a"],
                    "candidate_b_p50_ms": p["cand_b"],
                    "candidate_midpoint_ms": cand_mid,
                    "exact_drift_ms": base_drift,
                    "candidate_drift_ms": cand_drift,
                    "saving_ms_per_token": saving,
                    "speedup": base_mid / cand_mid if cand_mid else None,
                    "candidate_tok_s": 1000.0 / cand_mid if cand_mid else None,
                    "remaining_ms_to_s100": cand_mid - 10.0,
                    "information_gate": info,
                },
                "divergence": {
                    "exact_A_B": base_div,
                    "candidate_A_B": cand_div,
                    "candidate_vs_exact_report_only": cand_vs_base,
                },
                "gates": gates,
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        import traceback
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
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 2 if payload.get("status") in {
        "technical_failure", "instrument_failed"
    } else 0


if __name__ == "__main__":
    raise SystemExit(main())
