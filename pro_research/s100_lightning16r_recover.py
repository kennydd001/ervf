from __future__ import annotations

import json
import traceback

from common import write_json_atomic, utc_now
from s100_lightning16r_common import (
    PHASE16_RESULTS,
    RESULTS,
    candidate_key,
    discover_phase16_calibration,
    ensure_results,
    load_json,
)

OUT = RESULTS / "S100_LIGHTNING16R_RECOVERY.json"

def latest(rows):
    return max(
        rows,
        key=lambda row: (
            str(
                row["payload"].get(
                    "completed_utc",
                    row["payload"].get("started_utc", ""),
                )
            ),
            str(row["path"]),
        ),
    )

def main() -> int:
    ensure_results()
    payload = {
        "kind": "s100_lightning16r_recovery",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        screen_path = (
            PHASE16_RESULTS
            / "S100_LIGHTNING16_LAYER_SCREEN.json"
        )
        selection_path = (
            PHASE16_RESULTS
            / "S100_LIGHTNING16_QUALITY_SELECTION.json"
        )
        screen = load_json(screen_path)
        old_selection = load_json(selection_path)
        if not screen or screen.get("status") != "measured":
            raise RuntimeError(
                "measured Phase-16 layer screen unavailable"
            )

        measured = discover_phase16_calibration()
        grouped = {}
        for row in measured:
            grouped.setdefault(row["key"], []).append(row)

        recovered = []
        for candidate in screen.get(
            "selected_for_full_calibration", []
        ):
            canonical = {
                "name": str(candidate["name"]),
                "terms": int(candidate["terms"]),
                "cases": sorted(
                    str(case) for case in candidate["cases"]
                ),
                "handoff": str(
                    screen.get("handoff", "sync_control")
                ),
            }
            key = candidate_key(canonical)
            matches = grouped.get(key, [])
            match = latest(matches) if matches else None
            quality = match["payload"] if match else None
            recovered.append({
                **canonical,
                "matched": bool(match),
                "matched_path": (
                    str(match["path"]) if match else None
                ),
                "measured_name": (
                    quality.get("name") if quality else None
                ),
                "strict_pass": bool(
                    quality and quality.get("strict_pass")
                ),
                "official_pass": bool(
                    quality and quality.get("official_pass")
                ),
                "summary": (
                    quality.get("summary") if quality else None
                ),
                "per_domain": (
                    quality.get("per_domain") if quality else None
                ),
                "name_mismatch": bool(
                    quality
                    and quality.get("name")
                    != canonical["name"]
                ),
            })

        eligible = [
            row for row in recovered
            if row["strict_pass"]
        ]
        eligible.sort(
            key=lambda row: (
                -len(row["cases"]),
                int(row["terms"]),
                float(
                    (row.get("summary") or {}).get(
                        "mean_ce_delta", float("inf")
                    )
                ),
                row["name"],
            )
        )
        selected = eligible[:3]

        old_candidates = (
            (old_selection or {}).get("candidates") or []
        )
        old_null_count = sum(
            row.get("quality") is None
            for row in old_candidates
        )
        old_selected = (
            (old_selection or {}).get(
                "selected_for_validation"
            )
            or []
        )
        bug_confirmed = bool(
            selected
            and not old_selected
            and old_null_count > 0
            and any(row["name_mismatch"] for row in selected)
        )

        payload.update({
            "status": "measured",
            "phase16_screen_path": str(screen_path),
            "phase16_selection_path": str(selection_path),
            "phase16_selection_null_quality_count": (
                old_null_count
            ),
            "phase16_selected_for_validation_count": (
                len(old_selected)
            ),
            "recovered_candidates": recovered,
            "selected_for_validation": selected,
            "PHASE16_SELECTION_ORCHESTRATION_BUG_CONFIRMED": (
                bug_confirmed
            ),
            "RECOVERED_STRICT_CALIBRATION_PASS_COUNT": (
                len(eligible)
            ),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "bug_confirmed": payload.get(
            "PHASE16_SELECTION_ORCHESTRATION_BUG_CONFIRMED"
        ),
        "strict_calibration_pass_count": payload.get(
            "RECOVERED_STRICT_CALIBRATION_PASS_COUNT"
        ),
        "selected_for_validation": [
            {
                "name": row["name"],
                "terms": row["terms"],
                "cases": row["cases"],
            }
            for row in payload.get(
                "selected_for_validation", []
            )
        ],
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
