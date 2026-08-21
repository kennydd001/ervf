"""Phase75 online cross-layer route-association audit."""
from __future__ import annotations

import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic


RESULTS = REPO / "pro_research" / "results" / "s100_phase75"
PREREG = REPO / "pro_research" / "S100_PHASE75_ORNITH_CROSSLAYER_PREFETCH_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase75_ornith_crosslayer_prefetch.py"
TRACE = REPO / "pro_research" / "results" / "s100_phase70" / "ornith_128_trace.json"
PHASE71 = REPO / "pro_research" / "results" / "s100_phase71" / "S100_PHASE71_ORNITH_TRACE_PREFETCH_ORACLE.json"
PHASE73 = REPO / "pro_research" / "results" / "s100_phase73" / "S100_PHASE73_ORNITH_SEGMENTED_REALCOMPUTE.json"
FLOOR_MS = 60.095487602
BOUNDARY_MS = 4000.0 / 65.0
WARMUP_TOKENS = 32
LEADS = (1, 2, 4)
BUDGETS = (8, 16, 24, 32)


def _destination_stats(history, layer: int):
    counts: Counter[int] = Counter()
    last_seen: dict[int, int] = {}
    for token, row in enumerate(history[layer]):
        counts.update(row)
        for expert in row:
            last_seen[expert] = token
    return counts, last_seen


def _frequency_rank(history, layer: int):
    counts, last_seen = _destination_stats(history, layer)
    ranked = sorted(counts, key=lambda expert: (-counts[expert], -last_seen[expert], expert))
    ranked.extend(expert for expert in range(256) if expert not in set(ranked))
    return ranked


def _cross_rank(history, source: int, destination: int, source_rows):
    associations: dict[int, Counter[int]] = defaultdict(Counter)
    for source_row, destination_row in zip(history[source], history[destination]):
        for source_expert in source_row:
            associations[source_expert].update(destination_row)
    scores = Counter()
    for source_row in source_rows:
        for source_expert in source_row:
            scores.update(associations[source_expert])
    counts, last_seen = _destination_stats(history, destination)
    universe = set(scores) | set(counts)
    ranked = sorted(
        universe,
        key=lambda expert: (-scores[expert], -counts[expert], -last_seen[expert], expert),
    )
    ranked.extend(expert for expert in range(256) if expert not in set(ranked))
    return ranked


def _run_arm(name, trace, serial_group_ms: float, overlap_tail_ms: float):
    import sys

    source_path = REPO / "src"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from moe_lab.ornith.rolling_prefetch import (
        RollingPrefetchController,
        build_execution_layer_plan,
    )

    controller = RollingPrefetchController(ring_depth=2)
    controller.reset_request(f"phase75:{name}")
    history = {layer: [] for layer in range(40)}
    totals = Counter()
    blocks = []
    if name == "oracle":
        lead = 40
        budget = 32
    else:
        lead_text, budget_text = name.split("_b")
        lead = int(lead_text.removeprefix("lead"))
        budget = int(budget_text)
    for begin in range(0, len(trace.tokens), 4):
        actual = tuple(trace.routes[layer][begin:begin + 4] for layer in range(40))
        cache_before = controller.cache_snapshot()
        layer_plans = []
        for destination in range(40):
            if name == "oracle":
                ranked = list(dict.fromkeys(
                    expert for row in actual[destination] for expert in row
                ))
            elif destination < lead:
                ranked = _frequency_rank(history, destination)
            else:
                ranked = _cross_rank(
                    history,
                    destination - lead,
                    destination,
                    actual[destination - lead],
                )
            staged = ranked[:budget] if name != "oracle" else ranked
            layer_plans.append(build_execution_layer_plan(
                actual[destination], cache_before[destination], staged, layer=destination
            ))
        # The controller's exact prediction is used only to advance authoritative
        # cache metadata. Predictor accounting above uses the frozen causal set.
        block = controller.prepare_block(actual)
        controller.adjudicate(block.block_id, actual)
        row = {
            "begin_token": begin,
            "history_tokens": begin,
            "staged_candidates": sum(len(plan.staged_experts) for plan in layer_plans),
            "staged_hits": sum(len(plan.staged_hits) for plan in layer_plans),
            "uncovered_misses": sum(len(plan.uncovered_experts) for plan in layer_plans),
            "false_prefetches": sum(len(plan.false_prefetch_experts) for plan in layer_plans),
        }
        if begin >= WARMUP_TOKENS:
            for key, value in row.items():
                if key not in ("begin_token", "history_tokens"):
                    totals[key] += value
            totals["blocks"] += 1
        blocks.append(row)
        for layer in range(40):
            history[layer].extend(actual[layer])
    actual_misses = totals["staged_hits"] + totals["uncovered_misses"]
    recall = totals["staged_hits"] / actual_misses if actual_misses else 1.0
    precision = totals["staged_hits"] / totals["staged_candidates"]
    mean_uncovered = totals["uncovered_misses"] / totals["blocks"]
    projected_ms = FLOOR_MS + overlap_tail_ms + mean_uncovered * serial_group_ms
    return {
        "name": name,
        "lead": lead,
        "budget": budget,
        "evaluation_blocks": totals["blocks"],
        "totals": dict(totals),
        "unique_miss_recall": recall,
        "candidate_precision": precision,
        "mean_uncovered_groups_h4": mean_uncovered,
        "optimistic_projected_ms_h4": projected_ms,
        "optimistic_projected_tok_s": 4000.0 / projected_ms,
        "blocks": blocks,
    }


def main() -> int:
    out = RESULTS / "S100_PHASE75_ORNITH_CROSSLAYER_PREFETCH.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase75_ornith_crosslayer_prefetch",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        import sys

        source = REPO / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from moe_lab.ornith.trace_analysis import parse_llama_trace

        trace = parse_llama_trace(json.loads(TRACE.read_text("utf-8")))
        phase71 = json.loads(PHASE71.read_text("utf-8"))
        phase73 = json.loads(PHASE73.read_text("utf-8"))
        lru71 = phase71["records"]["lru52"]
        serial_group_ms = (
            lru71["summary"]["serial_increment_ms_h4"]
            / lru71["trace"]["mean_groups_per_h4"]
        )
        overlap_tail_ms = phase73["records"]["lru52"]["selected"]["exposed_tail_ms_h4"]
        names = ["oracle"] + [
            f"lead{lead}_b{budget}" for lead in LEADS for budget in BUDGETS
        ]
        arms = {
            name: _run_arm(name, trace, serial_group_ms, overlap_tail_ms)
            for name in names
        }
        physical = [row for row in arms.values() if row["name"] != "oracle" and row["lead"] >= 2]
        winner = max(
            physical,
            key=lambda row: (
                row["unique_miss_recall"], -row["totals"]["staged_candidates"]
            ),
        )
        gates = {
            "P75_G1_trace_and_oracle_contract": (
                len(trace.tokens) == 128
                and arms["oracle"]["evaluation_blocks"] == 24
                and arms["oracle"]["totals"]["uncovered_misses"] == 0
            ),
            "P75_G2_chronological_boundary": all(
                block["history_tokens"] == block["begin_token"]
                for row in arms.values() for block in row["blocks"]
            ),
            "P75_G3_physical_lead_recall_ge_95pct": winner["unique_miss_recall"] >= 0.95,
            "P75_G4_projected_boundary_le_65": winner["optimistic_projected_ms_h4"] <= BOUNDARY_MS,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "inputs": {
                "trace": str(TRACE.relative_to(REPO)),
                "warmup_tokens": WARMUP_TOKENS,
                "serial_group_ms": serial_group_ms,
                "phase73_overlap_tail_ms": overlap_tail_ms,
                "all_hot_floor_ms_h4": FLOOR_MS,
                "boundary_ms_h4": BOUNDARY_MS,
            },
            "arms": arms,
            "winner": winner["name"],
            "gates": gates,
            "completed_utc": utc_now(),
        })
    except Exception as error:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, TRACE, PHASE71, PHASE73))
    write_json_atomic(out, payload, archive=True)
    summary = {
        name: {
            "recall": row["unique_miss_recall"],
            "precision": row["candidate_precision"],
            "uncovered_h4": row["mean_uncovered_groups_h4"],
            "projected_tok_s": row["optimistic_projected_tok_s"],
        }
        for name, row in (payload.get("arms") or {}).items()
    }
    print(json.dumps({
        "status": payload.get("status"),
        "winner": payload.get("winner"),
        "summary": summary,
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
