from __future__ import annotations

import json

from common import write_json_atomic, utc_now
from s100_lightning16r_common import (
    PHASE16_RESULTS,
    RESULTS,
    ensure_results,
    load_json,
)

OUT = RESULTS / "S100_LIGHTNING16R_SUMMARY.json"
TXT = RESULTS / "S100_LIGHTNING16R_SUMMARY.txt"

def quality_rows():
    rows = []
    for path in sorted(
        RESULTS.glob("S100_LIGHTNING16R_QUALITY_*.json")
    ):
        payload = load_json(path)
        if not payload:
            continue
        rows.append({
            "path": str(path),
            "name": payload.get("name"),
            "split": payload.get("split"),
            "terms": payload.get("terms"),
            "handoff": payload.get("handoff"),
            "cases": payload.get("cases"),
            "candidate_signature": payload.get(
                "candidate_signature"
            ),
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "strict_pass": payload.get("strict_pass"),
            "official_pass": payload.get(
                "official_pass"
            ),
        })
    return rows

def main() -> int:
    ensure_results()
    recovery = load_json(
        RESULTS / "S100_LIGHTNING16R_RECOVERY.json"
    )
    subset = load_json(
        RESULTS / "S100_LIGHTNING16R_SUBSET_SEARCH.json"
    )
    throughput = load_json(
        RESULTS / "S100_LIGHTNING16R_THROUGHPUT.json"
    )
    phase16 = load_json(
        PHASE16_RESULTS
        / "S100_LIGHTNING16_SUMMARY.json"
    )
    block = load_json(
        PHASE16_RESULTS
        / "S100_LIGHTNING16_BLOCK_VERIFIER.json"
    )
    dflash = load_json(
        PHASE16_RESULTS
        / "S100_LIGHTNING16_DFLASH2_ECONOMICS.json"
    )

    qualities = quality_rows()
    calibration_green = [
        row for row in qualities
        if (
            row["status"] == "measured"
            and row["split"] == "calibration"
            and row["strict_pass"] is True
        )
    ]
    validation_green = [
        row for row in qualities
        if (
            row["status"] == "measured"
            and row["split"] == "validation"
            and row["strict_pass"] is True
        )
    ]
    heldout_green = [
        row for row in qualities
        if (
            row["status"] == "measured"
            and row["split"] == "heldout"
            and row["official_pass"] is True
            and bool(
                (row.get("summary") or {}).get(
                    "deterministic_anchor_repeat"
                )
            )
        )
    ]

    comparison_by_signature = {
        row["candidate_signature"]: row
        for row in (
            (throughput or {}).get("comparisons") or []
        )
    }
    promotion = []
    for row in heldout_green:
        comparison = comparison_by_signature.get(
            row["candidate_signature"]
        )
        promotion.append({
            **row,
            "throughput": comparison,
            "promotion_open": bool(
                comparison
                and comparison.get(
                    "SELECTIVE_NATIVE_NET_SPEEDUP_OPEN"
                )
            ),
            "s100_open": bool(
                comparison
                and comparison.get(
                    "S100_SINGLE_CANDIDATE_OPEN"
                )
            ),
        })

    bug_confirmed = bool(
        recovery
        and recovery.get("status") == "measured"
        and recovery.get(
            "PHASE16_SELECTION_ORCHESTRATION_BUG_CONFIRMED"
        )
    )
    quality_open = bool(heldout_green)
    speed_open = any(
        row["promotion_open"] for row in promotion
    )
    promotion_open = bool(quality_open and speed_open)
    s100 = any(row["s100_open"] for row in promotion)

    block_open = (
        None
        if not block or block.get("status") != "measured"
        else bool(
            block.get(
                "LIGHTNING_PERFECT_DRAFT_S100_OPEN"
            )
        )
    )
    dflash_open = (
        None
        if not dflash or dflash.get("status") != "measured"
        else bool(
            (dflash.get("gates") or {}).get(
                "DFLASH2_TRAINING_BUILD_OPEN"
            )
        )
    )

    if s100:
        next_action = (
            "INTEGRATE_AND_REPRODUCE_S100_NATIVE_CANDIDATE"
        )
    elif promotion_open:
        next_action = (
            "PORT_GREEN_NATIVE_SET_INTO_CAPTUREABLE_"
            "CUPY_CUBLASLT_OR_CUSTOM_CUDA_PATH"
        )
    elif quality_open:
        next_action = (
            "QUALITY_ROUTE_RECOVERED_BUT_CURRENT_TORCH_"
            "SYNC_BRIDGE_IS_NOT_FASTER_BUILD_CAPTUREABLE_NATIVE_PATH"
        )
    else:
        next_action = (
            "SELECTIVE_NATIVE_ROUTE_CLOSED_AT_FROZEN_"
            "VALIDATION_HELDOUT_GATES"
        )

    payload = {
        "kind": "s100_lightning16r_summary",
        "created_utc": utc_now(),
        "PHASE16_SELECTION_ORCHESTRATION_BUG_CONFIRMED": (
            bug_confirmed
        ),
        "RECOVERED_CALIBRATION_PASS_COUNT": (
            (recovery or {}).get(
                "RECOVERED_STRICT_CALIBRATION_PASS_COUNT"
            )
        ),
        "NOVEL_KV_SUBSET_FOUND": bool(
            subset
            and subset.get("status") == "measured"
            and subset.get("STRICT_NOVEL_SUBSET_FOUND")
        ),
        "CLEAN_CALIBRATION_STRICT_PASS_COUNT": len(
            calibration_green
        ),
        "VALIDATION_STRICT_PASS_COUNT": len(
            validation_green
        ),
        "HELDOUT_OFFICIAL_PASS_COUNT": len(
            heldout_green
        ),
        "SELECTIVE_NATIVE_QUALITY_OPEN": quality_open,
        "SELECTIVE_NATIVE_NET_SPEEDUP_OPEN": speed_open,
        "SELECTIVE_NATIVE_PROMOTION_OPEN": promotion_open,
        "LIGHTNING_PERFECT_DRAFT_S100_OPEN": block_open,
        "DFLASH2_TRAINING_BUILD_OPEN": dflash_open,
        "S100_SINGLE_ACHIEVED": s100,
        "validation_green": validation_green,
        "heldout_green": heldout_green,
        "promotion_candidates": promotion,
        "phase16_original_selective_flag": (
            (phase16 or {}).get(
                "FAMILY_SELECTIVE_NATIVE_QUALITY_OPEN"
            )
        ),
        "throughput_graph_reference": (
            (throughput or {}).get("graph_reference")
        ),
        "next_action": next_action,
        "claim_boundary": (
            "supersedes only Phase 16's selective-native "
            "selection verdict; block-verifier and DFlash2 "
            "measurements are carried forward unchanged"
        ),
    }
    write_json_atomic(OUT, payload, archive=True)

    lines = [
        "S100 LIGHTNING PHASE 16R",
        "",
        "PHASE16_SELECTION_ORCHESTRATION_BUG_CONFIRMED: "
        f"{bug_confirmed}",
        "CLEAN_CALIBRATION_STRICT_PASS_COUNT: "
        f"{len(calibration_green)}",
        "VALIDATION_STRICT_PASS_COUNT: "
        f"{len(validation_green)}",
        "HELDOUT_OFFICIAL_PASS_COUNT: "
        f"{len(heldout_green)}",
        "SELECTIVE_NATIVE_QUALITY_OPEN: "
        f"{quality_open}",
        "SELECTIVE_NATIVE_NET_SPEEDUP_OPEN: "
        f"{speed_open}",
        "SELECTIVE_NATIVE_PROMOTION_OPEN: "
        f"{promotion_open}",
        "LIGHTNING_PERFECT_DRAFT_S100_OPEN: "
        f"{block_open}",
        "DFLASH2_TRAINING_BUILD_OPEN: "
        f"{dflash_open}",
        "S100_SINGLE_ACHIEVED: "
        f"{s100}",
        "",
        f"Next action: {next_action}",
    ]
    if promotion:
        lines.extend(["", "Candidates:"])
        for row in promotion:
            speed = row.get("throughput") or {}
            lines.append(
                "- "
                f"{row['name']}: heldout=True, "
                "aggregate_speedup="
                f"{speed.get('aggregate_speedup_vs_graph')}, "
                "p50_speedup="
                f"{speed.get('p50_speedup_vs_graph')}, "
                f"promotion={row['promotion_open']}"
            )
    TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(TXT.read_text(encoding="utf-8"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
