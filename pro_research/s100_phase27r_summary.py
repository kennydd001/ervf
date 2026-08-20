from __future__ import annotations

import json

from common import REPO, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase27r"
OUT = RESULTS / "S100_PHASE27R_SUMMARY.json"


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    adjudication = load(
        RESULTS / "S100_PHASE27R_ADJUDICATION.json"
    )
    adopted = bool(adjudication.get("PHASE27_PIPELINE_ADOPTED"))

    promoted = []
    if adopted:
        for context in (128, 1024, 4096):
            doc = load(
                RESULTS
                / f"S100_PHASE27R_PROMOTED_CTX{context}.json"
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
                    "telemetry": doc.get("telemetry"),
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
        if row.get("ms_per_useful_token") is not None
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

    median_gain = adjudication.get(
        "median_round_gain_fraction"
    )
    if drafter_open:
        next_route = "OPEN_DSPARK_MTP_DFLASH_SHOOTOUT"
    elif target_100:
        next_route = (
            "TARGET_100_CEILING_OPEN_REDUCE_DRAFTER_HEADROOM"
        )
    elif adopted:
        next_route = (
            "PROFILE_ADOPTED_PIPELINE_THEN_FUSE_GATHER_DOWN"
        )
    elif (
        median_gain is not None and float(median_gain) > 0
    ):
        next_route = (
            "FUSE_GATHER_DOWN_AND_ELIMINATE_MIRROR_TRAFFIC"
        )
    else:
        next_route = (
            "KEEP_PHASE24_PARENT_AND_BUILD_ZERO_COPY_GROUPED_DOWN"
        )

    adjudication_gates = adjudication.get("gates") or {}
    adjudication_complete = bool(
        adjudication.get("status") == "measured"
        and adjudication_gates.get("complete_4_rounds_64_pairs")
        and adjudication_gates.get("all_correctness_green")
        and adjudication_gates.get("positions_aligned")
    )
    instrumentation_complete = bool(
        adjudication_complete
        and (
            not adopted
            or promoted_green
        )
    )

    result = {
        "kind": "s100_phase27r_summary",
        "created_utc": utc_now(),
        "instrumentation_complete": instrumentation_complete,
        "frozen_variant": {
            "gather_y": 4,
            "batches": 3,
            "shared_overlap": True,
        },
        "thermal_adjudication": adjudication,
        "PHASE27_PIPELINE_ADOPTED": adopted,
        "promoted_contexts": promoted,
        "TARGET_100_TARGET_ONLY_OPEN": target_100,
        "DRAFTER_SHOOTOUT_OPEN": drafter_open,
        "NEXT_ROUTE": next_route,
        "S100_SINGLE_ACHIEVED": False,
        "claim_boundary": (
            "thermally balanced exact H4 target-only timing; "
            "no drafter/acceptance/fallback cost"
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, result, archive=True)

    text = (
        "S100 PHASE 27R — THERMAL ADJUDICATION\n"
        f"Instrumentation complete: {instrumentation_complete}\n"
        f"Median round gain: {median_gain}\n"
        "Median paired-block gain: "
        f"{adjudication.get('median_paired_block_gain_fraction')}\n"
        f"Positive rounds: {adjudication.get('positive_rounds')}\n"
        f"Parent robust CV: {adjudication.get('parent_robust_cv')}\n"
        "Candidate robust CV: "
        f"{adjudication.get('candidate_robust_cv')}\n"
        f"PHASE27_PIPELINE_ADOPTED: {adopted}\n"
        f"TARGET_100_TARGET_ONLY_OPEN: {target_100}\n"
        f"DRAFTER_SHOOTOUT_OPEN: {drafter_open}\n"
        f"NEXT_ROUTE: {next_route}\n"
        "S100 SINGLE ACHIEVED: False\n"
    )
    (RESULTS / "S100_PHASE27R_SUMMARY.txt").write_text(
        text, encoding="utf-8"
    )

    report = REPO / "reports" / "S100_PHASE27R_RUN_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    table = [
        "| Round | Parent ms | Candidate ms | Gain | Aligned |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in adjudication.get("rounds") or []:
        if not row.get("complete"):
            table.append(
                f"| {row.get('round')} | n/a | n/a | n/a | no |"
            )
            continue
        table.append(
            f"| {row['round']} | "
            f"{row['parent_median_ms']:.4f} | "
            f"{row['candidate_median_ms']:.4f} | "
            f"{100.0 * row['round_gain_fraction']:.3f}% | "
            f"{'yes' if row['positions_aligned'] else 'no'} |"
        )

    lines = [
        "# S100 Phase 27R — Thermally Stable Pipeline Adjudication",
        "",
        "Phase27R changes no kernel or model math. It remeasures the "
        "frozen Phase27 candidate under a balanced thermal/order "
        "protocol because Phase27's candidate A/B was stable while "
        "the parent A/B bracket exceeded the stability gate.",
        "",
        "Frozen candidate:",
        "",
        "```text",
        "gather_y       = 4",
        "batches        = 3",
        "shared_overlap = true",
        "```",
        "",
        *table,
        "",
        f"- Median round gain: `{median_gain}`",
        "- Median 64-position paired gain: "
        f"`{adjudication.get('median_paired_block_gain_fraction')}`",
        f"- Positive rounds: `{adjudication.get('positive_rounds')}/4`",
        f"- Parent robust CV: `{adjudication.get('parent_robust_cv')}`",
        "- Candidate robust CV: "
        f"`{adjudication.get('candidate_robust_cv')}`",
        f"- Adopted: `{adopted}`",
    ]

    if adopted:
        lines += [
            "",
            "## Promoted contexts",
            "",
            "| Context | H4 median ms | ms/useful token | target-only tok/s |",
            "|---:|---:|---:|---:|",
        ]
        for row in promoted:
            summary = row.get("summary") or {}
            lines.append(
                f"| {row['context']} | "
                f"{summary.get('median_ms')} | "
                f"{row.get('ms_per_useful_token')} | "
                f"{row.get('target_only_tok_s')} |"
            )

    lines += [
        "",
        f"- TARGET_100_TARGET_ONLY_OPEN: `{target_100}`",
        f"- DRAFTER_SHOOTOUT_OPEN: `{drafter_open}`",
        f"- NEXT_ROUTE: `{next_route}`",
        "- S100 SINGLE ACHIEVED: `False`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
