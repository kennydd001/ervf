from __future__ import annotations

import json

import numpy as np

from common import REPO, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase27r"
OUT = RESULTS / "S100_PHASE27R_ADJUDICATION.json"

ROUNDS = {
    1: ("R1_PARENT", "R1_CANDIDATE"),
    2: ("R2_PARENT", "R2_CANDIDATE"),
    3: ("R3_PARENT", "R3_CANDIDATE"),
    4: ("R4_PARENT", "R4_CANDIDATE"),
}


def load(tag: str) -> dict:
    path = RESULTS / f"S100_PHASE27R_{tag}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def robust_cv(values: list[float]) -> float:
    arr = np.asarray(values, np.float64)
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center)))
    return float(1.4826 * mad / max(abs(center), 1e-30))


def compact_telemetry(doc: dict) -> dict:
    source = doc.get("telemetry") or {}
    out = {}
    for moment in ("after_graph_setup", "after_measure"):
        row = source.get(moment) or {}
        out[moment] = {
            key: row.get(key)
            for key in (
                "temperature.gpu",
                "pstate",
                "clocks.sm",
                "clocks.mem",
                "power.draw",
                "utilization.gpu",
                "memory.used",
                "error",
            )
            if key in row
        }
    return out


def main() -> int:
    round_rows = []
    paired_gains: list[float] = []
    parent_medians: list[float] = []
    candidate_medians: list[float] = []
    correctness_green = True
    positions_aligned = True

    for round_id, (parent_tag, candidate_tag) in ROUNDS.items():
        parent = load(parent_tag)
        candidate = load(candidate_tag)

        measured = bool(
            parent.get("status") == "measured"
            and candidate.get("status") == "measured"
            and parent.get("correctness_green")
            and candidate.get("correctness_green")
        )
        correctness_green &= measured

        parent_median = (parent.get("summary") or {}).get("median_ms")
        candidate_median = (
            candidate.get("summary") or {}
        ).get("median_ms")

        if parent_median is None or candidate_median is None:
            round_rows.append(
                {
                    "round": round_id,
                    "complete": False,
                    "parent_status": parent.get("status"),
                    "candidate_status": candidate.get("status"),
                }
            )
            continue

        parent_median = float(parent_median)
        candidate_median = float(candidate_median)
        parent_medians.append(parent_median)
        candidate_medians.append(candidate_median)

        parent_records = parent.get("records") or []
        candidate_records = candidate.get("records") or []
        parent_positions = [
            int(row["pos"]) for row in parent_records
        ]
        candidate_positions = [
            int(row["pos"]) for row in candidate_records
        ]
        aligned = bool(
            len(parent_records) == 16
            and len(candidate_records) == 16
            and parent_positions == candidate_positions
        )
        positions_aligned &= aligned

        local_gains = []
        if aligned:
            for parent_row, candidate_row in zip(
                parent_records, candidate_records
            ):
                gain = 1.0 - float(candidate_row["ms"]) / float(
                    parent_row["ms"]
                )
                local_gains.append(gain)
                paired_gains.append(gain)

        round_rows.append(
            {
                "round": round_id,
                "complete": True,
                "parent_median_ms": parent_median,
                "candidate_median_ms": candidate_median,
                "round_gain_fraction": (
                    1.0 - candidate_median / parent_median
                ),
                "positions_aligned": aligned,
                "paired_block_gain_median": (
                    float(np.median(local_gains))
                    if local_gains
                    else None
                ),
                "parent_telemetry": compact_telemetry(parent),
                "candidate_telemetry": compact_telemetry(candidate),
            }
        )

    round_gains = [
        row["round_gain_fraction"]
        for row in round_rows
        if row.get("complete")
    ]

    complete = bool(
        len(parent_medians) == 4
        and len(candidate_medians) == 4
        and len(paired_gains) == 64
        and positions_aligned
    )
    median_round_gain = (
        float(np.median(round_gains)) if round_gains else None
    )
    median_paired_gain = (
        float(np.median(paired_gains)) if paired_gains else None
    )
    positive_rounds = sum(1 for value in round_gains if value > 0)
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
        "complete_4_rounds_64_pairs": complete,
        "all_correctness_green": bool(correctness_green),
        "positions_aligned": bool(positions_aligned),
        "median_round_gain_ge_5pct": bool(
            median_round_gain is not None
            and median_round_gain >= 0.05
        ),
        "median_paired_gain_ge_5pct": bool(
            median_paired_gain is not None
            and median_paired_gain >= 0.05
        ),
        "positive_rounds_ge_3of4": positive_rounds >= 3,
        "parent_robust_cv_le_5pct": bool(
            parent_cv is not None and parent_cv <= 0.05
        ),
        "candidate_robust_cv_le_5pct": bool(
            candidate_cv is not None and candidate_cv <= 0.05
        ),
    }
    adopted = all(gates.values())

    if adopted:
        next_route = "PROMOTE_PHASE27_PIPELINE_AND_RUN_CONTEXTS"
    elif (
        complete
        and median_round_gain is not None
        and median_round_gain > 0
    ):
        next_route = "FUSE_GATHER_DOWN_AND_ELIMINATE_MIRROR_TRAFFIC"
    elif complete:
        next_route = "KEEP_PHASE24_PARENT_AND_BUILD_ZERO_COPY_GROUPED_DOWN"
    else:
        next_route = "REPAIR_OR_REPEAT_INCOMPLETE_PHASE27R_EVIDENCE"

    result = {
        "kind": "s100_phase27r_adjudication",
        "status": "measured",
        "created_utc": utc_now(),
        "frozen_variant": {
            "gather_y": 4,
            "batches": 3,
            "shared_overlap": True,
        },
        "rounds": round_rows,
        "parent_process_medians_ms": parent_medians,
        "candidate_process_medians_ms": candidate_medians,
        "round_gain_fractions": round_gains,
        "median_round_gain_fraction": median_round_gain,
        "paired_block_count": len(paired_gains),
        "median_paired_block_gain_fraction": median_paired_gain,
        "paired_block_gain_p10": (
            float(np.percentile(paired_gains, 10))
            if paired_gains
            else None
        ),
        "paired_block_gain_p90": (
            float(np.percentile(paired_gains, 90))
            if paired_gains
            else None
        ),
        "positive_rounds": positive_rounds,
        "parent_robust_cv": parent_cv,
        "candidate_robust_cv": candidate_cv,
        "gates": gates,
        "PHASE27_PIPELINE_ADOPTED": adopted,
        "NEXT_ROUTE": next_route,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, result, archive=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
