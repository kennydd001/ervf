from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic

OUTPUT_DIR = REPO / "pro_research" / "results" / "s100_phase9"


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    oracle = load(
        OUTPUT_DIR / "S100_PHASE9_CACHE_ORACLE.json"
    ) or {}
    economics = load(
        OUTPUT_DIR / "S100_PHASE9_MISS_ECONOMICS.json"
    ) or {}

    capacities = []
    for path in sorted(OUTPUT_DIR.glob("CAP_COMPARE_*.json")):
        result = load(path)
        if result:
            capacities.append(
                {
                    "profile": result.get("profile"),
                    "status": result.get("status"),
                    "summary": result.get("summary"),
                    "gates": result.get("gates"),
                    "error": result.get("error"),
                }
            )

    promoted = [
        row for row in capacities
        if row["status"] == "capacity_promote"
    ]
    best_capacity = (
        min(
            promoted,
            key=lambda row: row["summary"]["candidate_ms"],
        )
        if promoted else None
    )

    prefetch = oracle.get("prefetch", [])
    prefetch_best = (
        min(
            prefetch,
            key=lambda row: row.get(
                "demand_miss_fraction", 1.0
            ),
        )
        if prefetch else None
    )
    current_miss = (
        oracle.get("test_current") or {}
    ).get("miss_fraction")
    prefetch_research = bool(
        prefetch_best
        and current_miss not in (None, 0)
        and prefetch_best.get("prefetch_precision", 0) >= 0.40
        and prefetch_best.get("demand_miss_fraction", 1)
            <= 0.70 * current_miss
    )

    instrumentation = {
        "cache_oracle_measured": (
            oracle.get("status") == "measured"
            and bool(oracle.get("simulation_gate"))
        ),
        "capacity_profiles_present": bool(capacities),
        "miss_economics_complete": bool(
            economics.get("instrumentation_complete")
        ),
    }
    instrumentation_complete = all(instrumentation.values())

    payload = {
        "kind": "s100_phase9_summary",
        "created_utc": utc_now(),
        "frozen_quality_green_parent": {
            "profile": "QFAST+thr_0003",
            "heldout_green": True,
            "phase6_candidate_ms": 18.6276,
            "phase6_tok_s": 53.68378105606734,
        },
        "instrumentation": instrumentation,
        "instrumentation_complete": instrumentation_complete,
        "cache_oracle": oracle,
        "capacity_timings": capacities,
        "best_capacity_promote": best_capacity,
        "miss_economics": economics,
        "PREFETCH_RESEARCH": prefetch_research,
        "DIRECTHOST_PROMOTE": bool(
            economics.get("DIRECTHOST_PROMOTE")
        ),
        "ARC_MISS_PROMOTE": bool(
            economics.get("ARC_MISS_PROMOTE")
        ),
        "s100_single_achieved": bool(
            best_capacity
            and best_capacity["summary"]["candidate_ms"] <= 10.0
        ),
    }

    write_json_atomic(
        OUTPUT_DIR / "S100_PHASE9_SUMMARY.json",
        payload,
        archive=True,
    )

    belady = (
        oracle.get("belady_current_map_test") or {}
    ).get("miss_fraction")
    current = (
        oracle.get("test_current") or {}
    ).get("miss_fraction")

    lines = [
        "S100 PHASE 9 REPAIRED SUMMARY",
        f"Instrumentation complete: {instrumentation_complete}",
        f"Best exact capacity promote: {best_capacity}",
        f"DIRECTHOST_PROMOTE: {payload['DIRECTHOST_PROMOTE']}",
        f"ARC_MISS_PROMOTE: {payload['ARC_MISS_PROMOTE']}",
        f"PREFETCH_RESEARCH: {payload['PREFETCH_RESEARCH']}",
        f"Belady current-map miss fraction: {belady}",
        f"Current test miss fraction: {current}",
        f"S100 SINGLE ACHIEVED: {payload['s100_single_achieved']}",
    ]
    if not instrumentation_complete:
        lines.append(
            "NOTE: missing evidence prevents a scientific no-go."
        )

    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "S100_PHASE9_SUMMARY.txt").write_text(
        text, encoding="utf-8"
    )
    print(text)
    return 0 if instrumentation_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
