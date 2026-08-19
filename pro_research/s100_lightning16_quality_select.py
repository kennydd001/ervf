from __future__ import annotations

import json
from pathlib import Path

from common import write_json_atomic, utc_now
from s100_lightning16_common import RESULTS, ensure_results

def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    ensure_results()
    screen = load(
        RESULTS / "S100_LIGHTNING16_LAYER_SCREEN.json"
    ) or {}
    rows = []
    for candidate in screen.get(
        "selected_for_full_calibration", []
    ):
        name = candidate["name"]
        slug = "".join(
            ch if ch.isalnum() else "_"
            for ch in name
        ).strip("_").upper()
        path = RESULTS / (
            f"S100_LIGHTNING16_QUALITY_{slug}_CALIBRATION.json"
        )
        quality = load(path)
        rows.append({
            "name": name,
            "terms": candidate["terms"],
            "cases": candidate["cases"],
            "quality_path": str(path),
            "quality": quality,
            "strict_pass": bool(
                quality and quality.get("strict_pass")
            ),
        })

    eligible = [row for row in rows if row["strict_pass"]]
    eligible.sort(key=lambda row: (
        -len(row["cases"]),
        row["terms"],
        row["quality"]["summary"]["mean_ce_delta"],
    ))
    selected = eligible[:3]
    payload = {
        "kind": "s100_lightning16_quality_selection",
        "created_utc": utc_now(),
        "candidates": rows,
        "selected_for_validation": selected,
        "ANY_FULL_CALIBRATION_PASS": bool(selected),
    }
    write_json_atomic(
        RESULTS / "S100_LIGHTNING16_QUALITY_SELECTION.json",
        payload,
        archive=True,
    )
    print(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
