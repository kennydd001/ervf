"""Phase77 chronological residual correction for the Ornith future router."""
from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from s100_phase76_ornith_future_router import (
    BOUNDARY_MS,
    FLOOR_MS,
    PHASE70_TRACE,
    PHASE71,
    PHASE73,
    WARMUP_TOKENS,
    _cache_misses,
    _candidate_union,
    _frequency_candidates,
    _load_hidden_trace,
    _load_router_weights,
)


RESULTS = REPO / "pro_research" / "results" / "s100_phase77"
PREREG = REPO / "pro_research" / "S100_PHASE77_ORNITH_ROUTER_RESIDUAL_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase77_ornith_router_residual.py"
PHASE76 = REPO / "pro_research" / "results" / "s100_phase76" / "S100_PHASE76_ORNITH_FUTURE_ROUTER.json"
LEADS = (1, 2)
CORRECTORS = ("last", "ema09", "knn1", "knn4", "knn8", "ridge001", "ridge01", "ridge1")


def _correct(name: str, train_x: np.ndarray, query_x: np.ndarray,
             train_residual: np.ndarray) -> np.ndarray:
    if name == "last":
        return np.repeat(train_residual[-1:], len(query_x), axis=0)
    if name == "ema09":
        ages = np.arange(len(train_residual) - 1, -1, -1, dtype=np.float64)
        weights = np.power(0.9, ages)
        value = np.average(train_residual, axis=0, weights=weights)
        return np.repeat(value[None, :], len(query_x), axis=0)
    train_norm = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1.0e-12)
    query_norm = query_x / np.maximum(np.linalg.norm(query_x, axis=1, keepdims=True), 1.0e-12)
    if name.startswith("knn"):
        k = min(int(name.removeprefix("knn")), len(train_x))
        similarity = query_norm @ train_norm.T
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        return np.stack([train_residual[row].mean(axis=0) for row in nearest])
    lambdas = {"ridge001": 0.01, "ridge01": 0.1, "ridge1": 1.0}
    regularizer = lambdas[name]
    kernel = train_norm @ train_norm.T
    scale = max(float(np.mean(np.diag(kernel))), 1.0e-6)
    alpha = np.linalg.solve(
        kernel + regularizer * scale * np.eye(len(kernel), dtype=np.float32),
        train_residual,
    )
    return (query_norm @ train_norm.T) @ alpha


def _evaluate(name, lead, corrector, hidden, exact, proxy, routes, misses,
              serial_group_ms, overlap_tail_ms):
    totals = Counter()
    blocks = []
    for begin in range(WARMUP_TOKENS, 64, 4):
        row = Counter()
        for destination in range(40):
            if destination < lead:
                candidates = _frequency_candidates(routes, destination, begin, 32)
            else:
                source = destination - lead
                residual = exact[destination][:begin] - proxy[(lead, destination)][:begin]
                correction = _correct(
                    corrector,
                    hidden[source][:begin],
                    hidden[source][begin:begin + 4],
                    residual,
                )
                values = proxy[(lead, destination)][begin:begin + 4] + correction
                candidates = _candidate_union(values, 32)
            true_misses = set(misses[begin][destination])
            predicted = set(candidates)
            row["candidates"] += len(candidates)
            row["hits"] += len(true_misses & predicted)
            row["uncovered"] += len(true_misses - predicted)
            row["false"] += len(predicted - true_misses)
        totals.update(row)
        blocks.append({"begin_token": begin, **dict(row)})
    actual = totals["hits"] + totals["uncovered"]
    mean_uncovered = totals["uncovered"] / len(blocks)
    projected_ms = FLOOR_MS + overlap_tail_ms + mean_uncovered * serial_group_ms
    return {
        "name": name,
        "lead": lead,
        "corrector": corrector,
        "budget": 32,
        "evaluation_blocks": len(blocks),
        "totals": dict(totals),
        "unique_miss_recall": totals["hits"] / actual,
        "candidate_precision": totals["hits"] / totals["candidates"],
        "mean_uncovered_groups_h4": mean_uncovered,
        "optimistic_projected_ms_h4": projected_ms,
        "optimistic_projected_tok_s": 4000.0 / projected_ms,
        "blocks": blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    out = RESULTS / "S100_PHASE77_ORNITH_ROUTER_RESIDUAL.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase77_ornith_router_residual",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        import torch
        from safetensors import safe_open

        _raw, tokens, routes, hidden = _load_hidden_trace(args.trace.resolve())
        routers, _norms = _load_router_weights(torch, safe_open, args.snapshot.resolve())
        torch.backends.cuda.matmul.allow_tf32 = False
        exact = {}
        proxy = {}
        for destination in range(40):
            weight = routers[destination].float().cuda()
            exact[destination] = (
                torch.from_numpy(hidden[destination]).cuda() @ weight.T
            ).cpu().numpy()
            for lead in LEADS:
                if destination >= lead:
                    proxy[(lead, destination)] = (
                        torch.from_numpy(hidden[destination - lead]).cuda() @ weight.T
                    ).cpu().numpy()
        misses = _cache_misses(routes)
        phase71 = json.loads(PHASE71.read_text("utf-8"))
        phase73 = json.loads(PHASE73.read_text("utf-8"))
        phase76 = json.loads(PHASE76.read_text("utf-8"))
        lru71 = phase71["records"]["lru52"]
        serial_group_ms = (
            lru71["summary"]["serial_increment_ms_h4"]
            / lru71["trace"]["mean_groups_per_h4"]
        )
        overlap_tail_ms = phase73["records"]["lru52"]["selected"]["exposed_tail_ms_h4"]
        arms = {}
        for lead in LEADS:
            for corrector in CORRECTORS:
                name = f"lead{lead}_{corrector}"
                arms[name] = _evaluate(
                    name, lead, corrector, hidden, exact, proxy, routes, misses,
                    serial_group_ms, overlap_tail_ms,
                )
        lead2 = [row for row in arms.values() if row["lead"] == 2]
        winner = max(lead2, key=lambda row: row["unique_miss_recall"])
        upstream_green = (
            phase76["gates"]["P76_G1_trace_and_router_parity"]
            and phase76["gates"]["P76_G2_oracle_zero_uncovered"]
            and len(tokens) == 64
        )
        gates = {
            "P77_G1_upstream_exact_router_contract": upstream_green,
            "P77_G2_lead2_recall_ge_95pct": winner["unique_miss_recall"] >= 0.95,
            "P77_G3_projected_boundary_le_65": winner["optimistic_projected_ms_h4"] <= BOUNDARY_MS,
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "inputs": {
                "trace": str(args.trace.resolve()),
                "snapshot": str(args.snapshot.resolve()),
                "warmup_tokens": WARMUP_TOKENS,
                "serial_group_ms": serial_group_ms,
                "phase73_overlap_tail_ms": overlap_tail_ms,
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
    payload["environment"] = environment_snapshot((SCRIPT, PREREG, PHASE76, args.trace, PHASE70_TRACE, PHASE71, PHASE73))
    write_json_atomic(out, payload, archive=True)
    ranked = sorted(
        (payload.get("arms") or {}).values(),
        key=lambda row: row["unique_miss_recall"], reverse=True,
    )
    print(json.dumps({
        "status": payload.get("status"),
        "winner": payload.get("winner"),
        "arms": [{
            "name": row["name"],
            "recall": row["unique_miss_recall"],
            "precision": row["candidate_precision"],
            "uncovered_h4": row["mean_uncovered_groups_h4"],
            "projected_tok_s": row["optimistic_projected_tok_s"],
        } for row in ranked],
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
