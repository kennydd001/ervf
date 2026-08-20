from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic
from s100_phase28_common import RESULTS


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    preflight = load(
        RESULTS / "S100_PHASE28_PREFLIGHT.json"
    )
    audit = load(
        RESULTS / "S100_PHASE28_AUDIT.json"
    )
    selection = load(
        RESULTS / "S100_PHASE28_SELECTION.json"
    )
    state = load(
        RESULTS / "S100_PHASE28_STATE_CHECK.json"
    )
    screen = load(
        RESULTS / "S100_PHASE28_FINAL_SCREEN.json"
    )
    thermal = load(
        RESULTS
        / "S100_PHASE28_THERMAL_ADJUDICATION.json"
    )

    adopted = bool(
        thermal.get("PHASE28_MIRRORLESS_DOWN_ADOPTED")
    )
    selected_arm = selection.get("selected_arm")

    promoted = []
    if adopted:
        for context in (128, 1024, 4096):
            doc = load(
                RESULTS
                / f"S100_PHASE28_PROMOTED_CTX{context}.json"
            )
            promoted.append(
                {
                    "context": context,
                    "status": doc.get("status"),
                    "correctness_green": doc.get(
                        "correctness_green"
                    ),
                    "summary": doc.get("summary"),
                    "ms_per_useful_token": doc.get(
                        "ms_per_useful_token"
                    ),
                    "target_only_tok_s": doc.get(
                        "target_only_tok_s"
                    ),
                    "mirror_bytes_removed": doc.get(
                        "mirror_bytes_removed"
                    ),
                    "telemetry": doc.get(
                        "telemetry"
                    ),
                }
            )

    promoted_green = bool(
        adopted
        and len(promoted) == 3
        and all(
            row["status"] == "measured"
            and row["correctness_green"]
            for row in promoted
        )
    )
    ms_per_token = [
        float(row["ms_per_useful_token"])
        for row in promoted
        if row.get("ms_per_useful_token")
        is not None
    ]
    target_100 = bool(
        promoted_green
        and len(ms_per_token) == 3
        and max(ms_per_token) <= 10.0
    )
    drafter_open = bool(
        promoted_green
        and len(ms_per_token) == 3
        and max(ms_per_token) <= 8.0
    )

    screen_gain = screen.get("gain_fraction")
    selected_gain = selection.get(
        "selected_gain_fraction"
    )

    if drafter_open:
        next_route = (
            "OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
        )
    elif target_100:
        next_route = (
            "REDUCE_TARGET_VERIFIER_BELOW_8MS_FOR_DRAFTER_HEADROOM"
        )
    elif adopted:
        next_route = (
            "PROFILE_ADOPTED_MIRRORLESS_DOWN_AND_ATTACK_NEXT_DOMINANT_STAGE"
        )
    elif (
        screen_gain is not None
        and float(screen_gain) > 0
    ):
        next_route = (
            "OPTIMIZE_MIRRORLESS_HOST_LOADS_WITH_CP_ASYNC_OR_ROW_PIPELINING"
        )
    elif (
        selected_gain is not None
        and float(selected_gain) > 0
    ):
        next_route = (
            "MIRRORLESS_POSITIVE_BUT_UNSTABLE_REPEAT_OR_SPLIT_LOAD_COMPUTE"
        )
    else:
        next_route = (
            "KEEP_PHASE24_PARENT_AND_BUILD_DEVICE_TRANSFER_DESCRIPTOR_PATH"
        )

    thermal_gates = thermal.get("gates") or {}
    thermal_complete = bool(
        thermal.get("status") == "measured"
        and thermal_gates.get("complete")
        and thermal_gates.get("correctness")
        and thermal_gates.get("positions_aligned")
    )
    if thermal.get("status") == (
        "not_run_screen_closed"
    ):
        thermal_complete = True

    instrumentation_complete = bool(
        preflight.get("PREFLIGHT_GREEN")
        and audit.get("AUDIT_GREEN")
        and selection
        and (
            not selection.get("RUN_STATE_GATE")
            or state.get("SELECTED_STATE_GREEN")
        )
        and screen
        and thermal_complete
        and (
            not adopted
            or promoted_green
        )
    )

    result = {
        "kind": "s100_phase28_summary",
        "created_utc": utc_now(),
        "instrumentation_complete": (
            instrumentation_complete
        ),
        "PREFLIGHT_GREEN": bool(
            preflight.get("PREFLIGHT_GREEN")
        ),
        "AUDIT_GREEN": bool(
            audit.get("AUDIT_GREEN")
        ),
        "selection": selection,
        "selected_arm": selected_arm,
        "SELECTED_STATE_GREEN": bool(
            state.get("SELECTED_STATE_GREEN")
        ),
        "final_screen": screen,
        "thermal_adjudication": thermal,
        "PHASE28_MIRRORLESS_DOWN_ADOPTED": (
            adopted
        ),
        "promoted_contexts": promoted,
        "TARGET_100_TARGET_ONLY_OPEN": target_100,
        "DRAFTER_SHOOTOUT_OPEN": drafter_open,
        "NEXT_ROUTE": next_route,
        "S100_SINGLE_ACHIEVED": False,
        "claim_boundary": (
            "exact H4 target-only mirrorless verifier; "
            "no drafter/acceptance/fallback cost"
        ),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        RESULTS / "S100_PHASE28_SUMMARY.json",
        result,
        archive=True,
    )

    text = (
        "S100 PHASE 28 — MIRRORLESS GROUPED DOWN\n"
        f"Instrumentation complete: {instrumentation_complete}\n"
        f"Preflight green: {result['PREFLIGHT_GREEN']}\n"
        f"Audit green: {result['AUDIT_GREEN']}\n"
        f"Selected arm: {selected_arm}\n"
        f"Selection gain: {selected_gain}\n"
        f"State green: {result['SELECTED_STATE_GREEN']}\n"
        f"Final screen gain: {screen_gain}\n"
        f"Thermally adopted: {adopted}\n"
        f"TARGET_100_TARGET_ONLY_OPEN: {target_100}\n"
        f"DRAFTER_SHOOTOUT_OPEN: {drafter_open}\n"
        f"NEXT_ROUTE: {next_route}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (
        RESULTS / "S100_PHASE28_SUMMARY.txt"
    ).write_text(text, encoding="utf-8")

    report = (
        REPO
        / "reports"
        / "S100_PHASE28_RUN_REPORT.md"
    )
    report.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# S100 Phase 28 — Mirrorless Grouped Down",
        "",
        "Phase28 removes the temporary device mirror from the sparse routed "
        "down-projection path. Exact route/chunk partials, parent reduction "
        "and slot-order accumulation are retained.",
        "",
        f"- Synthetic partial-buffer preflight: `{result['PREFLIGHT_GREEN']}`",
        f"- Real pointer/resource audit: `{result['AUDIT_GREEN']}`",
        f"- Selected mirrorless arm: `{selected_arm}`",
        f"- Initial selection gain: `{selected_gain}`",
        f"- Full-state gate: `{result['SELECTED_STATE_GREEN']}`",
        f"- Final-screen gain: `{screen_gain}`",
        f"- Thermal adoption: `{adopted}`",
    ]

    if adopted:
        lines += [
            "",
            "## Promoted contexts",
            "",
            "| Context | H4 ms | ms/useful token | target-only tok/s | mirror bytes removed |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in promoted:
            summary = row.get("summary") or {}
            lines.append(
                f"| {row['context']} | "
                f"{summary.get('median_ms')} | "
                f"{row.get('ms_per_useful_token')} | "
                f"{row.get('target_only_tok_s')} | "
                f"{row.get('mirror_bytes_removed')} |"
            )

    lines += [
        "",
        f"- TARGET_100_TARGET_ONLY_OPEN: `{target_100}`",
        f"- DRAFTER_SHOOTOUT_OPEN: `{drafter_open}`",
        f"- NEXT_ROUTE: `{next_route}`",
        "- S100 SINGLE ACHIEVED: `False`",
        "",
    ]
    report.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
