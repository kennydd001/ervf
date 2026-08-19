from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common import REPO

PHASE16_RESULTS = (
    REPO / "pro_research" / "results" / "s100_lightning16"
)
RESULTS = (
    REPO / "pro_research" / "results" / "s100_lightning16r"
)

def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS

def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None

def slug(name: str) -> str:
    value = "".join(
        character if character.isalnum() else "_"
        for character in str(name)
    ).strip("_").upper()
    return value or "UNNAMED"

def canonical_cases(cases) -> list[str]:
    if isinstance(cases, str):
        values = [cases]
    elif isinstance(cases, dict):
        raise TypeError("cases must not be a mapping")
    else:
        try:
            values = list(cases)
        except TypeError as exc:
            raise TypeError(
                "cases must be an iterable"
            ) from exc
    return sorted({str(case) for case in values})

def candidate_signature(
    *,
    terms: int,
    cases,
    handoff: str,
) -> str:
    payload = {
        "terms": int(terms),
        "cases": canonical_cases(cases),
        "handoff": str(handoff),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

def candidate_key(payload: dict[str, Any]) -> tuple:
    return (
        int(payload["terms"]),
        tuple(canonical_cases(payload["cases"])),
        str(payload.get("handoff", "sync_control")),
    )

def quality_path(name: str, split: str) -> Path:
    return RESULTS / (
        f"S100_LIGHTNING16R_QUALITY_{slug(name)}_"
        f"{split.upper()}.json"
    )

def discover_phase16_calibration() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        PHASE16_RESULTS.glob(
            "S100_LIGHTNING16_QUALITY_*_CALIBRATION.json"
        )
    ):
        payload = load_json(path)
        if not payload:
            continue
        if (
            payload.get("kind") != "s100_lightning16_quality"
            or payload.get("status") != "measured"
            or payload.get("split") != "calibration"
        ):
            continue
        try:
            key = candidate_key(payload)
        except Exception:
            continue
        rows.append({
            "path": path,
            "payload": payload,
            "key": key,
        })
    return rows
