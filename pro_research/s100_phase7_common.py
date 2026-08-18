
"""Frozen phase-7 continuation definitions."""
from __future__ import annotations
import json
from pathlib import Path
from common import REPO

PHASE6_CANDIDATES = (
    REPO / "pro_research" / "results" / "S100_PHASE6_CANDIDATES.json"
)
EXPECTED = (
    "thr_0003",
    "thr_0010",
    "thr_0015",
    "thr_0020",
    "k1",
    "k2",
    "thr0010_k1",
    "thr0010_k2",
    "thr0015_k1",
)


def load_frozen_candidates() -> dict[str, dict]:
    data = json.loads(PHASE6_CANDIDATES.read_text(encoding="utf-8"))
    selected = data.get("selected") or {}
    if tuple(selected.keys()) != EXPECTED:
        raise RuntimeError(
            "phase-6 candidate set/order drifted: "
            f"expected={EXPECTED}, actual={tuple(selected.keys())}"
        )
    out = {}
    for name, spec in selected.items():
        out[name] = {
            "layer_k": {
                int(k): int(v)
                for k, v in (spec.get("layer_k") or {}).items()
            },
            "alpha": float(spec.get("alpha", 0.0)),
        }
    return out


def public_spec(spec: dict) -> dict:
    return {
        "layer_k": {
            str(k): int(v)
            for k, v in sorted((spec.get("layer_k") or {}).items())
        },
        "alpha": float(spec.get("alpha", 0.0)),
    }
