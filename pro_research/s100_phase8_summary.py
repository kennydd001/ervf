
from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic
from s100_phase8_common import BUDGETS

OUT = (
    REPO / "pro_research" / "results"
    / "S100_PHASE8_SUMMARY.json"
)
TXT = OUT.with_suffix(".txt")


def load(path):
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else None
    )


def main() -> int:
    route = load(
        REPO / "pro_research" / "results"
        / "S100_PHASE8_ROUTE_PROFILE.json"
    ) or {}
    selection = load(
        REPO / "pro_research" / "results"
        / "S100_PHASE8_BACKEND_SELECT.json"
    ) or {}
    phase7 = load(
        REPO / "pro_research" / "results"
        / "S100_PHASE7_SUMMARY.json"
    ) or {}

    rows = []
    for budget in BUDGETS:
        result = load(
            REPO / "pro_research" / "results"
            / (
                "S100_PHASE8_STATIC_COMPARE_"
                f"{budget}_FULL.json"
            )
        )
        if result is None:
            rows.append(
                {
                    "budget": budget,
                    "status": "missing",
                }
            )
            continue
        rows.append(
            {
                "budget": budget,
                "status": result.get("status"),
                "route_profile": result.get("route_profile"),
                "timing": result.get("summary"),
                "gates": result.get("gates"),
            }
        )

    selected = selection.get("selected")
    if selected:
        final_ms = float(
            selected["summary"]["static_midpoint_ms"]
        )
        final_tok_s = 1000.0 / final_ms
        final_name = (
            "thr_0020+static"
            f"{selected['budget']}"
        )
    else:
        parent = phase7.get("fastest_fidelity_green") or {}
        final_ms = float(
            (parent.get("timing") or {}).get(
                "candidate_midpoint_ms", 18.368
            )
        )
        final_tok_s = 1000.0 / final_ms
        final_name = "thr_0020"

    quality = (
        phase7.get("fastest_fidelity_green") or {}
    ).get("quality")
    s100 = final_ms <= 10.0

    payload = {
        "kind": "s100_phase8_summary",
        "created_utc": utc_now(),
        "quality_parent": "phase7/thr_0020",
        "quality": quality,
        "route_profile": {
            "status": route.get("status"),
            "budgets": {
                key: {
                    "physical_mib": value["physical_mib"],
                    "calibration_hit_rate": value[
                        "calibration"
                    ]["hit_rate"],
                    "validation_hit_rate": value[
                        "validation"
                    ]["hit_rate"],
                }
                for key, value in route.get(
                    "selections", {}
                ).items()
            },
        },
        "results": rows,
        "selected_backend": selection.get(
            "selected_backend", "legacy"
        ),
        "selected_budget": selection.get("selected_budget"),
        "fastest_fidelity_green": {
            "candidate": final_name,
            "candidate_midpoint_ms": final_ms,
            "candidate_tok_s": final_tok_s,
            "remaining_ms_to_s100": final_ms - 10.0,
            "quality": quality,
        },
        "s100_single_achieved": s100,
    }
    write_json_atomic(OUT, payload, archive=True)

    lines = [
        "S100 PHASE 8 SUMMARY",
        f"Quality parent: phase7/thr_0020",
        "",
        "budget | status | val_hit | MiB | ms | tok/s | saving",
    ]
    for row in rows:
        route_row = row.get("route_profile") or {}
        timing = row.get("timing") or {}
        lines.append(
            f"{row['budget']} | {row['status']} | "
            f"{route_row.get('validation_hit_rate')} | "
            f"{route_row.get('physical_mib')} | "
            f"{timing.get('static_midpoint_ms')} | "
            f"{timing.get('static_tok_s')} | "
            f"{timing.get('saving_ms')}"
        )
    lines.extend(
        [
            "",
            f"SELECTED: {final_name}",
            f"LATENCY: {final_ms} ms",
            f"THROUGHPUT: {final_tok_s} tok/s",
            f"S100 SINGLE ACHIEVED: {s100}",
        ]
    )
    TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
