from __future__ import annotations

import json
import traceback

from common import write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    RESULTS, assert_lightning, case_manifest, ensure_results,
    evaluate_runtime, normalize_eager_moe,
)
from s100_lightning16_native import PointerDispatch

OUT = RESULTS / "S100_LIGHTNING16_LAYER_SCREEN.json"
# Measured on hardware: only the FULL calibration split reproduces the
# frozen Lightning trace bit-exactly. Truncated evaluation (4x16 or 10x16)
# perturbs the runtime's numerics (allocation-history sensitive), so the
# screen must evaluate the complete 10x64 calibration split.
SCREEN_PROMPTS = None
SCREEN_TOKENS = None

def screen_pass(result):
    summary = result["summary"]
    return bool(
        summary["top1_agreement"] >= 0.95
        and summary["target_in_top5"] >= 0.99
        and summary["mean_ce_delta"] <= 0.05
        and summary["mean_coarse_kl"] <= 0.03
        and summary["all_finite"]
    )

def evaluate(rt, dispatch, enabled, terms):
    dispatch.enabled_cases = set(enabled)
    dispatch.terms = int(terms)
    result = evaluate_runtime(
        rt,
        split="calibration",
        prompt_limit=SCREEN_PROMPTS,
        token_limit=SCREEN_TOKENS,
    )
    result["screen_pass"] = screen_pass(result)
    result["enabled_cases"] = sorted(enabled)
    result["terms"] = int(terms)
    return result

def main():
    ensure_results()
    payload = {
        "kind": "s100_lightning16_layer_screen",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        ident = assert_lightning()
        diagnostic = json.loads(
            (
                RESULTS / "S100_LIGHTNING16_STREAM_DIAG.json"
            ).read_text(encoding="utf-8")
        )
        handoff = diagnostic.get(
            "recommended_handoff", "context_first"
        )
        bundle = build()
        rt = bundle.rt
        rt._graph = None
        rt.graph_mode = False
        normalize_eager_moe(rt)
        manifest = case_manifest(rt)
        dispatch = PointerDispatch(
            rt, terms=2, handoff=handoff, enabled_cases=set()
        ).install()

        baseline = evaluate(rt, dispatch, set(), 2)
        baseline_exact = bool(
            baseline["summary"]["top1_agreement"] == 1.0
            and abs(baseline["summary"]["mean_ce_delta"]) <= 1e-6
            and baseline["summary"]["mean_coarse_kl"] <= 1e-8
        )
        if not baseline_exact:
            raise RuntimeError(
                "screen baseline is not exact against Lightning trace"
            )

        individual = []
        for terms in (1, 2):
            for row in manifest:
                result = evaluate(
                    rt, dispatch, {row["case"]}, terms
                )
                individual.append({
                    "case": row["case"],
                    "layer": row["layer"],
                    "family": row["family"],
                    "weight_bytes": row["weight_bytes"],
                    **result,
                })
                print(
                    f"16B individual tc{terms} {row['case']}: "
                    f"top1={result['summary']['top1_agreement']:.4f} "
                    f"ce={result['summary']['mean_ce_delta']:.4f} "
                    f"pass={result['screen_pass']}",
                    flush=True,
                )

        family_sets = {}
        for terms in (1, 2):
            for families in (
                "k", "v", "o", "kv", "ko", "vo", "kvo"
            ):
                enabled = {
                    row["case"] for row in manifest
                    if row["family"] in set(families)
                }
                result = evaluate(
                    rt, dispatch, enabled, terms
                )
                family_sets[f"tc{terms}_{families}"] = result

        greedy = {}
        for terms in (1, 2):
            candidates = [
                row for row in individual
                if row["terms"] == terms and row["screen_pass"]
            ]
            candidates.sort(
                key=lambda row: (-row["weight_bytes"], row["case"])
            )
            enabled = set()
            steps = []
            for row in candidates:
                trial = set(enabled)
                trial.add(row["case"])
                result = evaluate(
                    rt, dispatch, trial, terms
                )
                accepted = bool(result["screen_pass"])
                if accepted:
                    enabled = trial
                steps.append({
                    "added": row["case"],
                    "accepted": accepted,
                    "result": result,
                })
            final = evaluate(
                rt, dispatch, enabled, terms
            )
            greedy[f"tc{terms}"] = {
                "selected_cases": sorted(enabled),
                "steps": steps,
                "final": final,
            }

        candidate_sets = []
        for key, record in greedy.items():
            if record["selected_cases"]:
                candidate_sets.append({
                    "name": f"{key}_greedy",
                    "mode": key,
                    "terms": int(key[2:]),
                    "cases": record["selected_cases"],
                    "screen": record["final"],
                })
        for key, result in family_sets.items():
            if result["screen_pass"]:
                candidate_sets.append({
                    "name": key,
                    "mode": key.split("_")[0],
                    "terms": int(key[2]),
                    "cases": result["enabled_cases"],
                    "screen": result,
                })

        # Deduplicate identical term/case sets, then keep at most four.
        unique = {}
        for row in candidate_sets:
            signature = (
                row["terms"], tuple(sorted(row["cases"]))
            )
            previous = unique.get(signature)
            if previous is None or (
                row["screen"]["summary"]["mean_ce_delta"]
                < previous["screen"]["summary"]["mean_ce_delta"]
            ):
                unique[signature] = row
        selected = sorted(
            unique.values(),
            key=lambda row: (
                -len(row["cases"]),
                row["terms"],
                row["screen"]["summary"]["mean_ce_delta"],
            ),
        )[:4]

        payload.update({
            "status": "measured",
            "identity": ident,
            "handoff": handoff,
            "screen_shape": {
                "prompts": SCREEN_PROMPTS,
                "tokens_per_prompt": SCREEN_TOKENS,
            },
            "baseline": baseline,
            "individual": individual,
            "family_sets": family_sets,
            "greedy": greedy,
            "selected_for_full_calibration": selected,
            "ANY_SCREEN_SAFE_NATIVE_SET": bool(selected),
            "completed_utc": utc_now(),
        })
        bundle.restore_combined()
        bundle.restore_sel()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "handoff": payload.get("handoff"),
        "ANY_SCREEN_SAFE_NATIVE_SET": payload.get(
            "ANY_SCREEN_SAFE_NATIVE_SET"
        ),
        "selected": [
            {
                "name": row["name"],
                "terms": row["terms"],
                "cases": row["cases"],
            }
            for row in payload.get(
                "selected_for_full_calibration", []
            )
        ],
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
