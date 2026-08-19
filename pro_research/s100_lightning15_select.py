from __future__ import annotations

import json

from common import write_json_atomic, utc_now
from s100_lightning15_common import RESULTS, ensure_results

CANDIDATES = [
    ("tc1", "kvo"),
    ("tc2", "kvo"),
    ("tc3", "kvo"),
    ("tc1", "k"),
    ("tc1", "v"),
    ("tc1", "o"),
    ("tc2", "k"),
    ("tc2", "v"),
    ("tc2", "o"),
    ("tc2", "kv"),
    ("tc2", "ko"),
    ("tc2", "vo"),
]
COVERAGE = {
    "kvo": 3, "kv": 2, "ko": 2, "vo": 2,
    "k": 1, "v": 1, "o": 1,
}

def load_quality(mode, families, split):
    name = f"{mode}_{families}".upper()
    path = RESULTS / (
        f"S100_LIGHTNING15_QUALITY_{name}_{split.upper()}.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    ensure_results()
    component_path = RESULTS / "S100_LIGHTNING15_COMPONENT.json"
    component = json.loads(component_path.read_text(encoding="utf-8"))

    rows = []
    for mode, families in CANDIDATES:
        quality = load_quality(mode, families, "calibration")
        if not quality:
            continue
        terms = int(mode[2:])
        comp = component["per_block"]["4"]["terms"][str(terms)]
        eligible = bool(
            quality.get("status") == "measured"
            and quality.get("strict_pass")
            and comp.get("finite")
            and comp.get("useful_row_speedup", 0) >= 2.5
        )
        rows.append({
            "mode": mode,
            "families": families,
            "coverage": COVERAGE[families],
            "terms": terms,
            "calibration": quality.get("summary"),
            "strict_pass": quality.get("strict_pass"),
            "component_speedup_b4": comp.get("useful_row_speedup"),
            "component_candidate_ms_b4": comp.get("candidate_mid_ms"),
            "component_max_nrmse": comp.get("max_case_nrmse"),
            "eligible": eligible,
        })

    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (
        -row["coverage"],
        row["terms"],
        row["calibration"]["mean_ce_delta"],
        -row["component_speedup_b4"],
    ))
    selected = eligible[:3]

    baseline = load_quality("baseline", "kvo", "calibration")
    round_control = load_quality(
        "round_ervf", "kvo", "calibration"
    )
    payload = {
        "kind": "s100_lightning15_selection",
        "created_utc": utc_now(),
        "baseline_self_control": baseline,
        "round_input_control": round_control,
        "candidates": rows,
        "selected_for_validation": selected,
        "LIGHTNING_TRACE_PROVENANCE_GREEN": bool(
            baseline
            and baseline.get("status") == "measured"
            and baseline.get("baseline_self_exact")
        ),
        "BF16X2_COLD_STREAM_OPEN": bool(
            component.get("BF16X2_COLD_STREAM_OPEN")
        ),
    }
    write_json_atomic(
        RESULTS / "S100_LIGHTNING15_SELECTION.json",
        payload, archive=True,
    )
    print(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
