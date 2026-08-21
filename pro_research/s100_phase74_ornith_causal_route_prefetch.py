"""Phase74 causal route-history prefetch audit on the real Ornith trace."""
from __future__ import annotations

import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from common import REPO, environment_snapshot, utc_now, write_json_atomic


RESULTS = REPO / "pro_research" / "results" / "s100_phase74"
PREREG = REPO / "pro_research" / "S100_PHASE74_ORNITH_CAUSAL_ROUTE_PREFETCH_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase74_ornith_causal_route_prefetch.py"
TRACE = REPO / "pro_research" / "results" / "s100_phase70" / "ornith_128_trace.json"
PHASE71 = REPO / "pro_research" / "results" / "s100_phase71" / "S100_PHASE71_ORNITH_TRACE_PREFETCH_ORACLE.json"
PHASE73 = REPO / "pro_research" / "results" / "s100_phase73" / "S100_PHASE73_ORNITH_SEGMENTED_REALCOMPUTE.json"
FLOOR_MS = 60.095487602
BOUNDARY_MS = 4000.0 / 65.0
WARMUP_TOKENS = 32
BUDGETS = (8, 16, 24, 32)


def _rank_recent(history, layer: int) -> list[int]:
    seen = set()
    ranked = []
    for row in reversed(history[layer]):
        for expert in reversed(row):
            if expert not in seen:
                seen.add(expert)
                ranked.append(expert)
    return ranked


def _statistics(history, layer: int):
    counts: Counter[int] = Counter()
    last_seen: dict[int, int] = {}
    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    rows = history[layer]
    for token, row in enumerate(rows):
        counts.update(row)
        for expert in row:
            last_seen[expert] = token
        if token:
            for source in rows[token - 1]:
                transitions[source].update(row)
    return counts, last_seen, transitions


def _rank_frequency(history, layer: int) -> list[int]:
    counts, last_seen, _transitions = _statistics(history, layer)
    return sorted(counts, key=lambda expert: (-counts[expert], -last_seen[expert], expert))


def _rank_transition(history, layer: int, *, hybrid: bool) -> list[int]:
    counts, last_seen, transitions = _statistics(history, layer)
    if not history[layer]:
        return []
    previous = history[layer][-1]
    scores = Counter()
    for source in previous:
        scores.update(transitions[source])
    universe = set(counts) | set(scores)
    if hybrid:
        return sorted(
            universe,
            key=lambda expert: (
                -scores[expert], -counts[expert], -last_seen[expert], expert
            ),
        )
    return sorted(
        universe,
        key=lambda expert: (-scores[expert], -last_seen[expert], -counts[expert], expert),
    )


def _pack_rows(ranked: list[int], budget: int, experts: int = 256):
    unique = list(dict.fromkeys(ranked))
    unique.extend(expert for expert in range(experts) if expert not in set(unique))
    selected = unique[:budget]
    chunks = [tuple(selected[index:index + 8]) for index in range(0, budget, 8)]
    return tuple(chunks[token % len(chunks)] for token in range(4))


def _predict(
    name: str,
    history: dict[int, list[tuple[int, ...]]],
    previous_h4,
    actual,
):
    if name == "oracle":
        return actual
    if name == "previous_h4":
        return previous_h4
    family, budget_text = name.rsplit("_", 1)
    budget = int(budget_text)
    rankers: dict[str, Callable[[dict[int, list[tuple[int, ...]]], int], list[int]]] = {
        "recent": _rank_recent,
        "frequency": _rank_frequency,
        "transition": lambda rows, layer: _rank_transition(rows, layer, hybrid=False),
        "hybrid": lambda rows, layer: _rank_transition(rows, layer, hybrid=True),
    }
    return tuple(
        _pack_rows(rankers[family](history, layer), budget)
        for layer in range(40)
    )


def _run_arm(name: str, trace, serial_group_ms: float, overlap_tail_ms: float):
    import sys

    source = REPO / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from moe_lab.ornith.rolling_prefetch import RollingPrefetchController

    controller = RollingPrefetchController(ring_depth=2)
    controller.reset_request(f"phase74:{name}")
    history = {layer: [] for layer in range(40)}
    totals = Counter()
    blocks = []
    for begin in range(0, len(trace.tokens), 4):
        actual = tuple(trace.routes[layer][begin:begin + 4] for layer in range(40))
        previous_h4 = tuple(tuple(history[layer][-4:]) for layer in range(40)) if begin else actual
        predicted = _predict(name, history, previous_h4, actual)
        block = controller.prepare_block(predicted)
        layer_plans = [controller.plan_layer(block.block_id, layer, actual[layer]) for layer in range(40)]
        adjudication = controller.adjudicate(block.block_id, actual)
        row = {
            "begin_token": begin,
            "history_tokens": begin,
            "staged_groups": sum(len(plan.staged_experts) for plan in layer_plans),
            "staged_hits": sum(len(plan.staged_hits) for plan in layer_plans),
            "uncovered_misses": sum(len(plan.uncovered_experts) for plan in layer_plans),
            "false_prefetches": sum(len(plan.false_prefetch_experts) for plan in layer_plans),
            "route_accuracy": adjudication.route_accuracy,
        }
        if begin >= WARMUP_TOKENS:
            for key in ("staged_groups", "staged_hits", "uncovered_misses", "false_prefetches"):
                totals[key] += row[key]
            totals["assignments"] += adjudication.compared_assignments
            totals["exact_assignments"] += adjudication.exact_assignments
            totals["blocks"] += 1
        blocks.append(row)
        for layer in range(40):
            history[layer].extend(actual[layer])

    actual_misses = totals["staged_hits"] + totals["uncovered_misses"]
    recall = totals["staged_hits"] / actual_misses if actual_misses else 1.0
    precision = totals["staged_hits"] / totals["staged_groups"] if totals["staged_groups"] else 1.0
    mean_uncovered = totals["uncovered_misses"] / totals["blocks"]
    projected_ms = FLOOR_MS + overlap_tail_ms + mean_uncovered * serial_group_ms
    return {
        "name": name,
        "evaluation_blocks": totals["blocks"],
        "totals": dict(totals),
        "unique_miss_recall": recall,
        "staged_precision": precision,
        "route_assignment_accuracy": totals["exact_assignments"] / totals["assignments"],
        "mean_uncovered_groups_h4": mean_uncovered,
        "optimistic_projected_ms_h4": projected_ms,
        "optimistic_projected_tok_s": 4000.0 / projected_ms,
        "blocks": blocks,
    }


def main() -> int:
    out = RESULTS / "S100_PHASE74_ORNITH_CAUSAL_ROUTE_PREFETCH.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase74_ornith_causal_route_prefetch",
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
        names = ["oracle", "previous_h4"] + [
            f"{family}_{budget}"
            for family in ("recent", "frequency", "transition", "hybrid")
            for budget in BUDGETS
        ]
        arms = {
            name: _run_arm(name, trace, serial_group_ms, overlap_tail_ms)
            for name in names
        }
        causal = [row for name, row in arms.items() if name != "oracle"]
        winner = max(
            causal,
            key=lambda row: (
                row["unique_miss_recall"],
                -row["totals"]["staged_groups"],
                row["route_assignment_accuracy"],
            ),
        )
        gates = {
            "P74_G1_trace_and_oracle_contract": (
                len(trace.tokens) == 128
                and arms["oracle"]["evaluation_blocks"] == 24
                and arms["oracle"]["totals"]["uncovered_misses"] == 0
            ),
            "P74_G2_causal_history_boundary": all(
                block["history_tokens"] == block["begin_token"]
                for row in causal for block in row["blocks"]
            ),
            "P74_G3_causal_recall_ge_95pct": winner["unique_miss_recall"] >= 0.95,
            "P74_G4_projected_boundary_le_65": winner["optimistic_projected_ms_h4"] <= BOUNDARY_MS,
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
            "precision": row["staged_precision"],
            "route_accuracy": row["route_assignment_accuracy"],
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
