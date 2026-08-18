
"""Frozen route-cache budgets and profile paths."""
from __future__ import annotations

import hashlib
import json
from common import REPO

BUDGETS = (64, 128, 192, 256, 320)
PROFILE = (
    REPO / "pro_research" / "results"
    / "S100_PHASE8_ROUTE_PROFILE.json"
)


def load_profile() -> dict:
    data = json.loads(PROFILE.read_text(encoding="utf-8"))
    if tuple(int(x) for x in data["budgets"]) != BUDGETS:
        raise RuntimeError("phase-8 route-profile budget drift")
    return data


def selection_for(budget: int) -> dict[int, list[int]]:
    if int(budget) not in BUDGETS:
        raise ValueError(budget)
    data = load_profile()
    raw = data["selections"][str(int(budget))]["by_layer"]
    return {
        int(layer): [int(expert) for expert in experts]
        for layer, experts in raw.items()
    }


def selection_hash(selection: dict[int, list[int]]) -> str:
    canonical = {
        str(layer): [int(x) for x in experts]
        for layer, experts in sorted(selection.items())
    }
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
