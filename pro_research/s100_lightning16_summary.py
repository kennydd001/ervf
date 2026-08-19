from __future__ import annotations

import json
from pathlib import Path

from common import write_json_atomic, utc_now
from s100_lightning16_common import RESULTS, ensure_results

def load(name):
    path = RESULTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def quality_files():
    return sorted(RESULTS.glob(
        "S100_LIGHTNING16_QUALITY_*_*.json"
    ))

def main():
    ensure_results()
    stream = load("S100_LIGHTNING16_STREAM_DIAG.json")
    screen = load("S100_LIGHTNING16_LAYER_SCREEN.json")
    selection = load(
        "S100_LIGHTNING16_QUALITY_SELECTION.json"
    )
    block = load("S100_LIGHTNING16_BLOCK_VERIFIER.json")
    route = load("S100_LIGHTNING16_ROUTE_CENSUS.json")
    proxy = load("S100_LIGHTNING16_DFLASH2_PROXY.json")
    economics = load(
        "S100_LIGHTNING16_DFLASH2_ECONOMICS.json"
    )

    quality = []
    heldout_green = []
    validation_green = []
    for path in quality_files():
        row = load(path.name)
        if not row:
            continue
        quality.append({
            "path": path.name,
            "name": row.get("name"),
            "split": row.get("split"),
            "terms": row.get("terms"),
            "cases": row.get("cases"),
            "summary": row.get("summary"),
            "strict_pass": row.get("strict_pass"),
            "official_pass": row.get("official_pass"),
        })
        if row.get("split") == "validation" and row.get("strict_pass"):
            validation_green.append(row)
        if row.get("split") == "heldout" and row.get("official_pass"):
            heldout_green.append(row)

    stream_confirmed = (
        None if not stream or stream.get("status") != "measured"
        else bool(stream.get("STREAM_HANDSHAKE_BUG_CONFIRMED"))
    )
    shadow_green = (
        None if not stream or stream.get("status") != "measured"
        else bool(stream.get("FIXED_NATIVE_SHADOW_GREEN"))
    )
    family_open = bool(heldout_green)
    perfect_open = (
        None if not block or block.get("status") != "measured"
        else bool(block.get("LIGHTNING_PERFECT_DRAFT_S100_OPEN"))
    )
    dflash_signal = (
        None if not proxy or proxy.get("status") != "measured"
        else bool(
            (proxy.get("gates") or {}).get(
                "DFLASH2_LIGHTNING_SIGNAL_OPEN"
            )
        )
    )
    dflash_training = (
        None if not economics or economics.get("status") != "measured"
        else bool(
            (economics.get("gates") or {}).get(
                "DFLASH2_TRAINING_BUILD_OPEN"
            )
        )
    )

    payload = {
        "kind": "s100_lightning16_summary",
        "created_utc": utc_now(),
        "STREAM_HANDSHAKE_BUG_CONFIRMED": stream_confirmed,
        "FIXED_NATIVE_SHADOW_GREEN": shadow_green,
        "FAMILY_SELECTIVE_NATIVE_QUALITY_OPEN": family_open,
        "LIGHTNING_PERFECT_DRAFT_S100_OPEN": perfect_open,
        "DFLASH2_LIGHTNING_SIGNAL_OPEN": dflash_signal,
        "DFLASH2_TRAINING_BUILD_OPEN": dflash_training,
        "screen_safe_sets": (
            screen.get("selected_for_full_calibration")
            if screen else None
        ),
        "validation_green": [
            {
                "name": row.get("name"),
                "terms": row.get("terms"),
                "cases": row.get("cases"),
            }
            for row in validation_green
        ],
        "heldout_green": [
            {
                "name": row.get("name"),
                "terms": row.get("terms"),
                "cases": row.get("cases"),
            }
            for row in heldout_green
        ],
        "block_verifier": (
            block.get("per_B") if block else None
        ),
        "route_census": (
            route.get("per_B") if route else None
        ),
        "dflash2_gates": (
            economics.get("gates") if economics else None
        ),
        "s100_single_achieved": False,
        "next_action": (
            "BUILD_GRAPH_NATIVE_SAFE_SUBSET_AND_LIGHTNING_BLOCK_VERIFIER"
            if family_open else
            "TRAIN_DFLASH2_LIGHTNING_PILOT"
            if dflash_training else
            "FIXED_STREAM_CONFIRMED_REDO_LAYER_SELECTION"
            if stream_confirmed else
            "NATIVE_NUMERICAL_AMPLIFICATION_CONFIRMED_SEARCH_EXACT_TC_KERNEL"
            if shadow_green else
            "REPAIR_PHASE16_TECHNICAL_EVIDENCE"
        ),
    }
    write_json_atomic(
        RESULTS / "S100_LIGHTNING16_SUMMARY.json",
        payload, archive=True,
    )
    text = (
        "S100 LIGHTNING PHASE 16\n"
        f"STREAM_HANDSHAKE_BUG_CONFIRMED: {stream_confirmed}\n"
        f"FIXED_NATIVE_SHADOW_GREEN: {shadow_green}\n"
        f"FAMILY_SELECTIVE_NATIVE_QUALITY_OPEN: {family_open}\n"
        f"LIGHTNING_PERFECT_DRAFT_S100_OPEN: {perfect_open}\n"
        f"DFLASH2_LIGHTNING_SIGNAL_OPEN: {dflash_signal}\n"
        f"DFLASH2_TRAINING_BUILD_OPEN: {dflash_training}\n"
        f"Next action: {payload['next_action']}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS / "S100_LIGHTNING16_SUMMARY.txt").write_text(
        text, encoding="utf-8"
    )
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
