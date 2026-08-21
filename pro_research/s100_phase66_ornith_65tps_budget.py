"""Phase66 combine measured Ornith H4 components and cache-miss scenarios."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic

SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_lab.ornith.h4_budget import OrnithH4Budget, interpolate_curve


RESULTS = REPO / "pro_research" / "results" / "s100_phase66"
PREREG = REPO / "pro_research" / "S100_PHASE66_ORNITH_65TPS_BUDGET_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase66_ornith_65tps_budget.py"
SOURCES = {
    "phase49": REPO / "pro_research" / "results" / "s100_phase49" / "S100_PHASE49_POTTOKAO_LAYER20_EXPERT0.json",
    "phase58": REPO / "pro_research" / "results" / "s100_phase58" / "S100_PHASE58_ORNITH_FP8_H4.json",
    "phase59": REPO / "pro_research" / "results" / "s100_phase59" / "S100_PHASE59_ORNITH_BULK_EXPERT_H4.json",
    "phase62": REPO / "pro_research" / "results" / "s100_phase62" / "S100_PHASE62_ORNITH_BULK_COLD_UVA.json",
    "phase64": REPO / "pro_research" / "results" / "s100_phase64" / "S100_PHASE64_ORNITH_NATIVE_SHORTLIST_HEAD.json",
    "phase65": REPO / "pro_research" / "results" / "s100_phase65" / "S100_PHASE65_ORNITH_SHARED_OVERLAP.json",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    out = RESULTS / "S100_PHASE66_ORNITH_65TPS_BUDGET.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase66_ornith_65tps_budget",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        source = {name: _load(path) for name, path in SOURCES.items()}
        p58 = {(row["repository"], row["label"]): row for row in source["phase58"]["records"]}
        linear = sum(p58[("pottokao", label)]["m4_median_ms"] for label in (
            "linear_qkv", "linear_z", "linear_out"
        ))
        full = sum(p58[("pottokao", label)]["m4_median_ms"] for label in (
            "full_q", "full_k", "full_v", "full_out"
        ))
        attention = 30 * linear + 10 * full
        p49_m4 = next(
            row for row in source["phase49"]["records"] if row["multiplicity"] == 4
        )["candidate_timing_ms"]["p50"]
        p59_records = {row["groups"]: row for row in source["phase59"]["records"]}
        hot_curve = {0: 0.0, **{
            groups: float(row["bulk_timing_ms"]["p50"])
            for groups, row in p59_records.items()
        }}
        routed_hot = hot_curve[32]
        head = float(source["phase64"]["summary"]["candidate_h4_ms"])
        budget = OrnithH4Budget(
            attention_projection_ms=attention,
            routed_hot_per_layer_ms=routed_hot,
            shared_per_layer_ms=float(p49_m4),
            head_ms=head,
        )
        p62 = {row["groups"]: row for row in source["phase62"]["records"]}
        miss_cost = {
            1: float(p62[1]["timings_ms"]["direct"]["p50"]),
            **{
                count: float(p62[count]["timings_ms"]["stage"]["p50"])
                for count in (4, 8, 16, 32)
            },
        }
        scenarios = []
        for misses in (0, 1, 4, 8, 16, 32):
            if misses == 0:
                route_layer = routed_hot
                policy = "hot_only"
            else:
                route_layer = interpolate_curve(hot_curve, 32 - misses) + miss_cost[misses]
                policy = "direct_uva" if misses == 1 else "bulk_stage"
            known = attention + 40 * (route_layer + float(p49_m4)) + head
            scenarios.append({
                "uniform_unique_misses_per_layer": misses,
                "route_hit_rate": (32 - misses) / 32,
                "miss_policy": policy,
                "routed_ms_per_layer": route_layer,
                "known_floor_ms_h4": known,
                "known_floor_equivalent_tok_s": 4000.0 / known,
                "remaining_to_65_ms": budget.target_h4_ms - known,
            })
        marginal_one_miss = scenarios[1]["known_floor_ms_h4"] - scenarios[0]["known_floor_ms_h4"]
        max_single_misses_no_other_work = int(
            max(0.0, budget.unmeasured_allowance_ms) // (marginal_one_miss / 40)
        )
        max_single_misses_with_5ms_other = int(
            max(0.0, budget.unmeasured_allowance_ms - 5.0) // (marginal_one_miss / 40)
        )
        gates = {
            "P66_G1_required_sources_present": all(path.is_file() for path in SOURCES.values()),
            "P66_G2_required_component_results_green": all((
                source["phase49"]["status"] == "measured_pass",
                source["phase58"]["status"] == "measured_pass",
                source["phase59"]["status"] == "measured_pass",
                source["phase64"]["status"] == "measured_pass",
            )),
            "P66_G3_hot_known_floor_below_65_boundary": budget.unmeasured_allowance_ms > 0,
            "P66_G4_overlap_correctly_excluded": source["phase65"]["status"] == "measured_fail",
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "components_ms_h4": {
                "fp8_attention_projections": attention,
                "routed_hot_40_layers": 40 * routed_hot,
                "shared_serial_40_layers": 40 * float(p49_m4),
                "native_shortlist_exact_head": head,
            },
            "hot_budget": {
                "known_floor_ms_h4": budget.known_hot_floor_ms,
                "known_floor_equivalent_tok_s": 4000.0 / budget.known_hot_floor_ms,
                "target_ms_h4": budget.target_h4_ms,
                "unmeasured_allowance_ms": budget.unmeasured_allowance_ms,
                "unmeasured_components": [
                    "40 routers and route planning/reduction",
                    "30 linear-attention recurrent cores",
                    "10 causal full-attention cores at serving context",
                    "norms, residuals, final norm, argmax and graph orchestration",
                ],
            },
            "uniform_miss_scenarios": scenarios,
            "cache_ceiling": {
                "marginal_ms_h4_for_one_unique_miss_in_each_layer": marginal_one_miss,
                "max_isolated_single_misses_across_1280_assignments_if_no_other_work": max_single_misses_no_other_work,
                "minimum_hit_rate_if_no_other_work": 1 - max_single_misses_no_other_work / 1280,
                "max_isolated_single_misses_if_5ms_reserved_for_other_work": max_single_misses_with_5ms_other,
                "minimum_hit_rate_if_5ms_reserved_for_other_work": 1 - max_single_misses_with_5ms_other / 1280,
            },
            "claim": (
                "65 tok/s remains physically open on the measured hot components, but is not "
                "proven until the unmeasured attention/router/core work and real cache trace fit "
                "inside the residual budget"
            ),
            "sources": {name: str(path.relative_to(REPO)) for name, path in SOURCES.items()},
            "gates": gates,
            "completed_utc": utc_now(),
        })
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
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, *SOURCES.values()))
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "components_ms_h4": payload.get("components_ms_h4"),
        "hot_budget": payload.get("hot_budget"),
        "cache_ceiling": payload.get("cache_ceiling"),
        "miss_scenarios": payload.get("uniform_miss_scenarios"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
