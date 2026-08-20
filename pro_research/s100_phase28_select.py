from __future__ import annotations

import json

from common import utc_now, write_json_atomic
from s100_phase28_common import RESULTS, Arm

OUT = RESULTS / "S100_PHASE28_SELECTION.json"

TAG_BY_ARM = {
    "p27_control": "SCREEN_P27_CONTROL",
    "direct_route": "SCREEN_DIRECT_ROUTE",
    "group_chunk_v16": "SCREEN_GROUP_CHUNK_V16",
    "group_allchunks_v4": "SCREEN_ALLCHUNKS_V4",
    "group_allchunks_v16": "SCREEN_ALLCHUNKS_V16",
    "group_allchunks_v16_overlap": (
        "SCREEN_ALLCHUNKS_V16_OVERLAP"
    ),
}

PREFERENCE = {
    "group_allchunks_v16": 0,
    "group_allchunks_v4": 1,
    "group_chunk_v16": 2,
    "direct_route": 3,
    "group_allchunks_v16_overlap": 4,
}


def load(tag: str) -> dict:
    path = RESULTS / f"S100_PHASE28_{tag}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def median_ms(doc: dict):
    try:
        return float(doc["summary"]["median_ms"])
    except Exception:
        return None


def exact_measured(doc: dict) -> bool:
    return bool(
        doc.get("status") == "measured"
        and doc.get("correctness_green")
        and (doc.get("summary") or {}).get("all_token_exact")
    )


def main() -> int:
    parent_a = load("SCREEN_PARENT_A")
    parent_b = load("SCREEN_PARENT_B")
    parent_a_ms = median_ms(parent_a)
    parent_b_ms = median_ms(parent_b)

    if parent_a_ms is not None and parent_b_ms is not None:
        parent_midpoint = (
            parent_a_ms + parent_b_ms
        ) / 2.0
        parent_drift = abs(
            parent_a_ms - parent_b_ms
        ) / parent_midpoint
    else:
        parent_midpoint = None
        parent_drift = None

    arms = []
    for name, tag in TAG_BY_ARM.items():
        doc = load(tag)
        ms = median_ms(doc)
        arm = Arm(name)
        arms.append(
            {
                "name": name,
                "tag": tag,
                "arm": arm.as_dict(),
                "status": doc.get("status"),
                "measured_exact": exact_measured(doc),
                "median_ms": ms,
                "ms_per_useful_token": (
                    None if ms is None else ms / 4.0
                ),
                "target_only_tok_s": (
                    None if ms is None else 4000.0 / ms
                ),
                "gain_vs_parent": (
                    None
                    if ms is None
                    or parent_midpoint is None
                    else 1.0 - ms / parent_midpoint
                ),
                "mirror_bytes_removed": doc.get(
                    "mirror_bytes_removed"
                ),
                "telemetry": doc.get("telemetry"),
                "eligible_for_phase28_selection": (
                    arm.eligible_for_phase28_selection
                ),
            }
        )

    eligible = [
        row
        for row in arms
        if row["eligible_for_phase28_selection"]
        and row["measured_exact"]
        and row["median_ms"] is not None
    ]

    selected = None
    if (
        eligible
        and parent_midpoint is not None
        and parent_drift is not None
        and parent_drift <= 0.07
    ):
        fastest_ms = min(row["median_ms"] for row in eligible)

        # Candidates within 1% of the fastest are considered a tie.
        tied = [
            row
            for row in eligible
            if row["median_ms"] <= fastest_ms * 1.01
        ]

        # Do not add the side-stream overlap arm when the ordinary v16
        # all-chunks arm is within 1%. It must be strictly useful.
        ordinary_v16 = next(
            (
                row
                for row in tied
                if row["name"] == "group_allchunks_v16"
            ),
            None,
        )
        if ordinary_v16 is not None:
            tied = [
                row
                for row in tied
                if row["name"]
                != "group_allchunks_v16_overlap"
            ]

        selected = min(
            tied,
            key=lambda row: PREFERENCE[row["name"]],
        )

    selected_gain = (
        None
        if selected is None
        else selected["gain_vs_parent"]
    )
    run_state = bool(
        selected is not None
        and selected_gain is not None
        and selected_gain >= 0.02
    )

    control = next(
        (
            row
            for row in arms
            if row["name"] == "p27_control"
        ),
        None,
    )

    result = {
        "kind": "s100_phase28_selection",
        "created_utc": utc_now(),
        "parent_anchors_ms": [
            parent_a_ms,
            parent_b_ms,
        ],
        "parent_midpoint_ms": parent_midpoint,
        "parent_relative_drift": parent_drift,
        "parent_stable": bool(
            parent_drift is not None
            and parent_drift <= 0.07
        ),
        "phase27_control": control,
        "arms": arms,
        "selected": selected,
        "selected_arm": (
            None if selected is None else selected["name"]
        ),
        "selected_gain_fraction": selected_gain,
        "RUN_STATE_GATE": run_state,
        "selection_rule": (
            "fastest exact mirrorless arm; 1% tie preference; "
            "minimum 2% gain for state/final-screen progression"
        ),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, result, archive=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
