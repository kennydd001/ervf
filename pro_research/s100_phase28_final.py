from __future__ import annotations

import argparse
import json

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase28_common import RESULTS, robust_cv


def load(tag: str) -> dict:
    path = RESULTS / f"S100_PHASE28_{tag.upper()}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def median_ms(doc: dict):
    try:
        return float(doc["summary"]["median_ms"])
    except Exception:
        return None


def exact(doc: dict) -> bool:
    return bool(
        doc.get("status") == "measured"
        and doc.get("correctness_green")
        and (doc.get("summary") or {}).get(
            "all_token_exact"
        )
    )


def final_screen() -> dict:
    parent_a = load("FINAL_PARENT_A")
    candidate_a = load("FINAL_CANDIDATE_A")
    candidate_b = load("FINAL_CANDIDATE_B")
    parent_b = load("FINAL_PARENT_B")

    values = {
        "parent_a": median_ms(parent_a),
        "candidate_a": median_ms(candidate_a),
        "candidate_b": median_ms(candidate_b),
        "parent_b": median_ms(parent_b),
    }
    complete = all(
        value is not None
        for value in values.values()
    )

    state = {}
    try:
        state = json.loads(
            (
                RESULTS
                / "S100_PHASE28_STATE_CHECK.json"
            ).read_text(encoding="utf-8")
        )
    except Exception:
        pass

    correctness = all(
        exact(doc)
        for doc in (
            parent_a,
            candidate_a,
            candidate_b,
            parent_b,
        )
    )

    if complete:
        parent_midpoint = (
            values["parent_a"]
            + values["parent_b"]
        ) / 2.0
        candidate_midpoint = (
            values["candidate_a"]
            + values["candidate_b"]
        ) / 2.0

        parent_drift = abs(
            values["parent_a"]
            - values["parent_b"]
        ) / parent_midpoint
        candidate_drift = abs(
            values["candidate_a"]
            - values["candidate_b"]
        ) / candidate_midpoint
        gain = 1.0 - (
            candidate_midpoint / parent_midpoint
        )
    else:
        parent_midpoint = None
        candidate_midpoint = None
        parent_drift = None
        candidate_drift = None
        gain = None

    stable = bool(
        complete
        and parent_drift <= 0.07
        and candidate_drift <= 0.07
    )
    open_thermal = bool(
        correctness
        and state.get("SELECTED_STATE_GREEN")
        and stable
        and gain is not None
        and gain >= 0.02
    )

    result = {
        "kind": "s100_phase28_final_screen",
        "created_utc": utc_now(),
        "medians_ms": values,
        "parent_midpoint_ms": parent_midpoint,
        "candidate_midpoint_ms": candidate_midpoint,
        "parent_relative_drift": parent_drift,
        "candidate_relative_drift": candidate_drift,
        "gain_fraction": gain,
        "candidate_ms_per_useful_token": (
            None
            if candidate_midpoint is None
            else candidate_midpoint / 4.0
        ),
        "candidate_target_only_tok_s": (
            None
            if candidate_midpoint is None
            else 4000.0 / candidate_midpoint
        ),
        "correctness_green": correctness,
        "state_green": bool(
            state.get("SELECTED_STATE_GREEN")
        ),
        "stable": stable,
        "RUN_THERMAL_ADOPTION": open_thermal,
    }

    write_json_atomic(
        RESULTS / "S100_PHASE28_FINAL_SCREEN.json",
        result,
        archive=True,
    )
    print(json.dumps(result, indent=2))
    return result


def thermal_adjudication() -> dict:
    screen = {}
    try:
        screen = json.loads(
            (
                RESULTS
                / "S100_PHASE28_FINAL_SCREEN.json"
            ).read_text(encoding="utf-8")
        )
    except Exception:
        pass

    if not screen.get("RUN_THERMAL_ADOPTION"):
        result = {
            "kind": "s100_phase28_thermal_adjudication",
            "status": "not_run_screen_closed",
            "created_utc": utc_now(),
            "PHASE28_MIRRORLESS_DOWN_ADOPTED": False,
        }
        write_json_atomic(
            RESULTS
            / "S100_PHASE28_THERMAL_ADJUDICATION.json",
            result,
            archive=True,
        )
        print(json.dumps(result, indent=2))
        return result

    rounds = []
    paired_gains = []
    parent_medians = []
    candidate_medians = []
    correctness = True
    aligned_all = True

    for round_id in (1, 2, 3, 4):
        parent = load(
            f"THERMAL_R{round_id}_PARENT"
        )
        candidate = load(
            f"THERMAL_R{round_id}_CANDIDATE"
        )

        correctness &= exact(parent) and exact(candidate)
        parent_median = median_ms(parent)
        candidate_median = median_ms(candidate)

        if (
            parent_median is None
            or candidate_median is None
        ):
            rounds.append(
                {
                    "round": round_id,
                    "complete": False,
                    "parent_status": parent.get("status"),
                    "candidate_status": candidate.get(
                        "status"
                    ),
                }
            )
            continue

        parent_medians.append(parent_median)
        candidate_medians.append(candidate_median)

        parent_records = parent.get("records") or []
        candidate_records = candidate.get(
            "records"
        ) or []

        aligned = bool(
            len(parent_records) == 16
            and len(candidate_records) == 16
            and [
                int(row["pos"])
                for row in parent_records
            ]
            == [
                int(row["pos"])
                for row in candidate_records
            ]
        )
        aligned_all &= aligned

        local_gains = []
        if aligned:
            for parent_row, candidate_row in zip(
                parent_records,
                candidate_records,
            ):
                gain = 1.0 - float(
                    candidate_row["ms"]
                ) / float(parent_row["ms"])
                local_gains.append(gain)
                paired_gains.append(gain)

        rounds.append(
            {
                "round": round_id,
                "complete": True,
                "parent_median_ms": parent_median,
                "candidate_median_ms": candidate_median,
                "round_gain_fraction": (
                    1.0
                    - candidate_median
                    / parent_median
                ),
                "positions_aligned": aligned,
                "paired_gain_median": (
                    float(np.median(local_gains))
                    if local_gains
                    else None
                ),
                "parent_telemetry": parent.get(
                    "telemetry"
                ),
                "candidate_telemetry": candidate.get(
                    "telemetry"
                ),
            }
        )

    round_gains = [
        row["round_gain_fraction"]
        for row in rounds
        if row.get("complete")
    ]
    complete = bool(
        len(parent_medians) == 4
        and len(candidate_medians) == 4
        and len(paired_gains) == 64
        and aligned_all
    )
    median_round_gain = (
        float(np.median(round_gains))
        if round_gains
        else None
    )
    median_paired_gain = (
        float(np.median(paired_gains))
        if paired_gains
        else None
    )
    positive_rounds = sum(
        1 for value in round_gains if value > 0
    )
    parent_cv = (
        robust_cv(parent_medians)
        if len(parent_medians) == 4
        else None
    )
    candidate_cv = (
        robust_cv(candidate_medians)
        if len(candidate_medians) == 4
        else None
    )

    gates = {
        "complete": complete,
        "correctness": bool(correctness),
        "positions_aligned": bool(aligned_all),
        "median_round_gain_ge_5pct": bool(
            median_round_gain is not None
            and median_round_gain >= 0.05
        ),
        "median_pair_gain_ge_5pct": bool(
            median_paired_gain is not None
            and median_paired_gain >= 0.05
        ),
        "positive_rounds_ge_3": (
            positive_rounds >= 3
        ),
        "parent_cv_le_5pct": bool(
            parent_cv is not None
            and parent_cv <= 0.05
        ),
        "candidate_cv_le_5pct": bool(
            candidate_cv is not None
            and candidate_cv <= 0.05
        ),
    }
    adopted = all(gates.values())

    result = {
        "kind": "s100_phase28_thermal_adjudication",
        "status": "measured",
        "created_utc": utc_now(),
        "rounds": rounds,
        "parent_medians_ms": parent_medians,
        "candidate_medians_ms": candidate_medians,
        "round_gain_fractions": round_gains,
        "median_round_gain_fraction": median_round_gain,
        "paired_window_count": len(paired_gains),
        "median_paired_window_gain_fraction": (
            median_paired_gain
        ),
        "paired_gain_p10": (
            float(
                np.percentile(
                    paired_gains,
                    10,
                )
            )
            if paired_gains
            else None
        ),
        "paired_gain_p90": (
            float(
                np.percentile(
                    paired_gains,
                    90,
                )
            )
            if paired_gains
            else None
        ),
        "positive_rounds": positive_rounds,
        "parent_robust_cv": parent_cv,
        "candidate_robust_cv": candidate_cv,
        "gates": gates,
        "PHASE28_MIRRORLESS_DOWN_ADOPTED": adopted,
    }

    write_json_atomic(
        RESULTS
        / "S100_PHASE28_THERMAL_ADJUDICATION.json",
        result,
        archive=True,
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("screen", "thermal"),
        required=True,
    )
    args = parser.parse_args()

    if args.stage == "screen":
        final_screen()
    else:
        thermal_adjudication()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
