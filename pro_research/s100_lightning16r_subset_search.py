from __future__ import annotations

import itertools
import json
import math
import traceback

from common import write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    STRICT,
    assert_lightning,
    case_manifest,
    evaluate_runtime,
    normalize_eager_moe,
)
from s100_lightning16r_common import (
    RESULTS,
    candidate_signature,
    ensure_results,
)
from s100_lightning16r_native import PointerDispatch

OUT = RESULTS / "S100_LIGHTNING16R_SUBSET_SEARCH.json"
TERMS = 1
HANDOFF = "pair_sync_control"
PAIR_SEARCH_WIDTH = 6
MAX_SELECTED = 2

def gate_pressure(result: dict) -> float:
    summary = result["summary"]
    domains = result["per_domain"]
    if not summary.get("all_finite"):
        return float("inf")
    minimum_domain_top1 = min(
        float(row["top1_agreement"])
        for row in domains.values()
    )
    maximum_domain_ce = max(
        float(row["mean_ce_delta"])
        for row in domains.values()
    )

    def lower_pressure(threshold: float, value: float):
        return threshold / max(value, 1e-30)

    def upper_pressure(value: float, threshold: float):
        return value / max(threshold, 1e-30)

    return float(max(
        lower_pressure(
            STRICT["top1"],
            float(summary["top1_agreement"]),
        ),
        lower_pressure(
            STRICT["top5"],
            float(summary["target_in_top5"]),
        ),
        upper_pressure(
            float(summary["mean_ce_delta"]),
            STRICT["mean_ce"],
        ),
        upper_pressure(
            float(summary["mean_coarse_kl"]),
            STRICT["mean_kl"],
        ),
        upper_pressure(
            float(summary["p95_coarse_kl"]),
            STRICT["p95_kl"],
        ),
        lower_pressure(
            STRICT["domain_top1"],
            minimum_domain_top1,
        ),
        upper_pressure(
            maximum_domain_ce,
            STRICT["domain_ce"],
        ),
    ))

def candidate_name(removed: tuple[str, ...]) -> str:
    suffix = "__".join(
        value.replace("attention_", "a")
        for value in removed
    )
    return f"tc1_pair_kv_minus__{suffix}"

def result_record(
    *,
    cases: list[str],
    removed: tuple[str, ...],
    quality: dict,
    bytes_by_case: dict[str, int],
) -> dict:
    return {
        "name": (
            "tc1_pair_kv_all"
            if not removed
            else candidate_name(removed)
        ),
        "terms": TERMS,
        "handoff": HANDOFF,
        "cases": sorted(cases),
        "removed_cases": list(removed),
        "native_weight_bytes": int(sum(
            bytes_by_case[case] for case in cases
        )),
        "candidate_signature": candidate_signature(
            terms=TERMS,
            cases=cases,
            handoff=HANDOFF,
        ),
        "quality": quality,
        "strict_pass": bool(
            quality.get("strict_pass")
        ),
        "official_pass": bool(
            quality.get("official_pass")
        ),
        "gate_pressure": gate_pressure(quality),
    }

def main() -> int:
    ensure_results()
    payload = {
        "kind": "s100_lightning16r_subset_search",
        "status": "started",
        "terms": TERMS,
        "handoff": HANDOFF,
        "started_utc": utc_now(),
        "claim_boundary": (
            "calibration-only deterministic subset search; "
            "validation and heldout are not read"
        ),
    }
    bundle = None
    try:
        identity = assert_lightning()
        bundle = build()
        runtime = bundle.rt
        runtime._graph = None
        runtime.graph_mode = False
        normalize_eager_moe(runtime)

        manifest = case_manifest(runtime)
        kv_rows = [
            row for row in manifest
            if row["family"] in {"k", "v"}
        ]
        union = sorted(row["case"] for row in kv_rows)
        bytes_by_case = {
            row["case"]: int(row["weight_bytes"])
            for row in kv_rows
        }
        if len(union) != 12:
            raise RuntimeError(
                f"expected 12 Lightning K/V matrices, got {len(union)}"
            )

        dispatch = PointerDispatch(
            runtime,
            terms=TERMS,
            handoff=HANDOFF,
            enabled_cases=set(union),
        ).install()

        trials: list[dict] = []

        def evaluate(
            cases: list[str],
            removed: tuple[str, ...],
        ) -> dict:
            dispatch.configure(
                terms=TERMS,
                handoff=HANDOFF,
                enabled_cases=cases,
            )
            dispatch.last_native_case = None
            quality = evaluate_runtime(
                runtime,
                split="calibration",
                deterministic=False,
            )
            quality["dispatch"] = {
                "native_calls": dispatch.native_calls,
                "original_calls": dispatch.original_calls,
                "sync_calls": dispatch.sync_calls,
                "paired_sync_elisions": (
                    dispatch.paired_sync_elisions
                ),
                "torch_mm_style": (
                    dispatch.engine.mm.style
                ),
            }
            record = result_record(
                cases=cases,
                removed=removed,
                quality=quality,
                bytes_by_case=bytes_by_case,
            )
            trials.append(record)
            print(
                f"16R subset {record['name']}: "
                f"strict={record['strict_pass']} "
                f"pressure={record['gate_pressure']:.5f} "
                f"top1={quality['summary']['top1_agreement']:.5f} "
                f"CE={quality['summary']['mean_ce_delta']:.5f} "
                f"sync={quality['dispatch']['sync_calls']}",
                flush=True,
            )
            return record

        union_record = evaluate(union, tuple())

        leave_one = []
        for removed_case in union:
            cases = [
                case for case in union
                if case != removed_case
            ]
            leave_one.append(
                evaluate(cases, (removed_case,))
            )

        strict_novel = (
            [union_record]
            if union_record["strict_pass"]
            else []
        )
        strict_novel.extend(
            row for row in leave_one
            if row["strict_pass"]
        )
        pair_candidates_considered = []

        if not strict_novel:
            ranked_removed = [
                row["removed_cases"][0]
                for row in sorted(
                    leave_one,
                    key=lambda row: (
                        row["gate_pressure"],
                        -row["native_weight_bytes"],
                        row["name"],
                    ),
                )[:PAIR_SEARCH_WIDTH]
            ]
            for removed_pair in itertools.combinations(
                ranked_removed, 2
            ):
                pair_candidates_considered.append(
                    list(removed_pair)
                )
                cases = [
                    case for case in union
                    if case not in set(removed_pair)
                ]
                record = evaluate(
                    cases, tuple(sorted(removed_pair))
                )
                if record["strict_pass"]:
                    strict_novel.append(record)

        # De-duplicate by signature, then maximize replaced bytes.
        unique = {}
        for row in strict_novel:
            unique[row["candidate_signature"]] = row
        selected = sorted(
            unique.values(),
            key=lambda row: (
                -row["native_weight_bytes"],
                row["gate_pressure"],
                float(
                    row["quality"]["summary"][
                        "mean_ce_delta"
                    ]
                ),
                row["name"],
            ),
        )[:MAX_SELECTED]

        payload.update({
            "status": "measured",
            "identity": identity,
            "union_cases": union,
            "union_result": union_record,
            "leave_one_trials": leave_one,
            "pair_candidates_considered": (
                pair_candidates_considered
            ),
            "trials": trials,
            "selected_for_validation": [
                {
                    key: row[key]
                    for key in (
                        "name",
                        "terms",
                        "handoff",
                        "cases",
                        "removed_cases",
                        "native_weight_bytes",
                        "candidate_signature",
                        "gate_pressure",
                    )
                }
                for row in selected
            ],
            "STRICT_NOVEL_SUBSET_FOUND": bool(selected),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })
    finally:
        if bundle is not None:
            try:
                bundle.restore_combined()
                bundle.restore_sel()
            except Exception:
                pass

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "strict_novel_subset_found": payload.get(
            "STRICT_NOVEL_SUBSET_FOUND"
        ),
        "selected_for_validation": payload.get(
            "selected_for_validation"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
