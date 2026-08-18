
from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic
from s100_phase8_common import BUDGETS

OUT = (
    REPO / "pro_research" / "results"
    / "S100_PHASE8_BACKEND_SELECT.json"
)


def main() -> int:
    rows = []
    for budget in BUDGETS:
        path = (
            REPO / "pro_research" / "results"
            / (
                "S100_PHASE8_STATIC_COMPARE_"
                f"{budget}_FULL.json"
            )
        )
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "budget": int(budget),
                "status": data.get("status"),
                "route_profile": data.get("route_profile"),
                "summary": data.get("summary"),
                "gates": data.get("gates"),
            }
        )

    good = [
        row
        for row in rows
        if row["status"] == "exact_backend_candidate"
    ]
    selected = (
        min(
            good,
            key=lambda row: float(
                row["summary"]["static_midpoint_ms"]
            ),
        )
        if good
        else None
    )
    payload = {
        "kind": "s100_phase8_backend_select",
        "created_utc": utc_now(),
        "minimum_gain_ms": 0.15,
        "selected_backend": (
            "static" if selected else "legacy"
        ),
        "selected_budget": (
            selected["budget"] if selected else None
        ),
        "selected": selected,
        "results": rows,
    }
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
