#!/usr/bin/env python3
"""TierFlow-F0: CPU-only bounded route-edit traffic oracle.

This deliberately does not train a model or claim quality.  Validation selects
one edit budget; test can only be opened after that selection was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
ROUTES = ROOT / "reports" / "runs" / "streamq5_moe" / "p4d_routes"
CAPTURE = REPORTS / "p4d_route_capture_result.json"
ROUTE_LOCK = REPORTS / "p4d_route_input_lock.json"
PREREG = REPORTS / "TIERFLOW_F0_PREREGISTRATION_2026-08-12.md"
VALIDATION_OUT = REPORTS / "tierflow_f0_validation.json"
RESULT_OUT = REPORTS / "tierflow_f0_result.json"

EXPECTED_CAPTURE_SHA256 = (
    "7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb"
)
EDIT_BUDGETS = (1, 2, 4)
TOP_K = 8
EXPERTS = 128


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def distribution(values: list[int]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": int(a.max()),
    }


def future_tables(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Next use strictly after t and remaining frequency strictly after t."""
    n = observed.shape[0]
    present = np.zeros((n, EXPERTS), dtype=np.uint8)
    rows = np.arange(n)[:, None]
    present[rows, observed] = 1

    suffix = np.zeros((n + 1, EXPERTS), dtype=np.int32)
    for t in range(n - 1, -1, -1):
        suffix[t] = suffix[t + 1] + present[t]

    next_use = np.full((n, EXPERTS), n + 1, dtype=np.int32)
    next_position = np.full(EXPERTS, n + 1, dtype=np.int32)
    for t in range(n - 1, -1, -1):
        next_use[t] = next_position
        next_position[observed[t]] = t
    return next_use, suffix


def simulate_sequence(observed_full: np.ndarray, start: int, end: int, r: int) -> dict[str, Any]:
    if start < 1:
        raise ValueError("warm-started partition must begin after token zero")
    observed = observed_full[start:end].astype(np.int64, copy=False)
    if observed.shape != (end - start, TOP_K):
        raise ValueError(f"unexpected route shape {observed.shape}")
    if np.any(observed < 0) or np.any(observed >= EXPERTS):
        raise ValueError("route ID outside [0, 127]")
    if any(len(set(row.tolist())) != TOP_K for row in observed):
        raise ValueError("top-k route contains duplicate expert IDs")

    next_use, suffix = future_tables(observed)
    previous_observed = set(observed_full[start - 1].astype(np.int64).tolist())
    state = set(previous_observed)

    baseline_new: list[int] = []
    oracle_new: list[int] = []
    overlaps: list[int] = []
    exact_matches = 0

    for local_t, row in enumerate(observed):
        requested = set(row.tolist())
        baseline_new.append(len(requested - previous_observed))

        missing = requested - state
        admit_count = min(r, len(missing))

        def keep_key(expert: int) -> tuple[int, int, int]:
            # Smaller means more valuable to retain/admit.
            return (
                int(next_use[local_t, expert]),
                -int(suffix[local_t + 1, expert]),
                int(expert),
            )

        admissions = sorted(missing, key=keep_key)[:admit_count]
        stale = state - requested
        evictions = sorted(stale, key=keep_key, reverse=True)[:admit_count]
        state.difference_update(evictions)
        state.update(admissions)

        if len(state) != TOP_K:
            raise AssertionError("persistent route set changed cardinality")
        edits = len(state - (state - set(admissions) | set(evictions)))
        # The direct state-delta count is admissions; the expression above is
        # retained as a defensive cross-check of replacement semantics.
        if edits != admit_count or admit_count > r:
            raise AssertionError("route edit budget violated")

        overlap = len(state & requested)
        overlaps.append(overlap)
        oracle_new.append(admit_count)
        exact_matches += int(state == requested)
        previous_observed = requested

    return {
        "baseline_new": baseline_new,
        "oracle_new": oracle_new,
        "overlaps": overlaps,
        "exact_matches": exact_matches,
    }


def summarize(
    baseline_new: list[int],
    oracle_new: list[int],
    overlaps: list[int],
    exact_matches: int,
    expert_record_bytes: int,
) -> dict[str, Any]:
    baseline_total = int(sum(baseline_new))
    oracle_total = int(sum(oracle_new))
    baseline_max = int(max(baseline_new))
    oracle_max = int(max(oracle_new))
    route_slots = len(overlaps) * TOP_K
    overlap_slots = int(sum(overlaps))
    return {
        "transitions": len(overlaps),
        "baseline_new_loads": distribution(baseline_new),
        "oracle_new_loads": distribution(oracle_new),
        "baseline_critical_expert_bytes": baseline_total * expert_record_bytes,
        "oracle_critical_expert_bytes": oracle_total * expert_record_bytes,
        "critical_expert_bytes_reduction_x": (
            float(baseline_total / oracle_total) if oracle_total else math.inf
        ),
        "worst_case_new_load_reduction_x": (
            float(baseline_max / oracle_max) if oracle_max else math.inf
        ),
        "mean_route_set_overlap": float(overlap_slots / route_slots),
        "router_output_substitution_rate": float(1.0 - overlap_slots / route_slots),
        "exact_route_set_match_rate": float(exact_matches / len(overlaps)),
        "overlap_count": distribution(overlaps),
    }


def evaluate_partition(
    route_data: dict[int, dict[str, np.ndarray]],
    domains: list[str],
    start: int,
    end: int,
    r: int,
    expert_record_bytes: int,
) -> dict[str, Any]:
    all_baseline: list[int] = []
    all_oracle: list[int] = []
    all_overlaps: list[int] = []
    all_exact = 0
    per_domain: dict[str, Any] = {}

    for domain in domains:
        d_baseline: list[int] = []
        d_oracle: list[int] = []
        d_overlaps: list[int] = []
        d_exact = 0
        for layer in sorted(route_data):
            result = simulate_sequence(route_data[layer][domain], start, end, r)
            d_baseline.extend(result["baseline_new"])
            d_oracle.extend(result["oracle_new"])
            d_overlaps.extend(result["overlaps"])
            d_exact += result["exact_matches"]

        per_domain[domain] = summarize(
            d_baseline, d_oracle, d_overlaps, d_exact, expert_record_bytes
        )
        all_baseline.extend(d_baseline)
        all_oracle.extend(d_oracle)
        all_overlaps.extend(d_overlaps)
        all_exact += d_exact

    aggregate = summarize(
        all_baseline, all_oracle, all_overlaps, all_exact, expert_record_bytes
    )
    sequences = len(domains) * len(route_data)
    aggregate["cold_start_expert_bytes_separate"] = (
        sequences * TOP_K * expert_record_bytes
    )
    aggregate["stream_all_top8_bytes_without_reuse"] = (
        sequences * (end - start) * TOP_K * expert_record_bytes
    )
    return {"aggregate": aggregate, "per_domain": per_domain}


def load_locked_routes() -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    if sha256(CAPTURE) != EXPECTED_CAPTURE_SHA256:
        raise RuntimeError("P4D route capture hash changed")
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    lock = json.loads(ROUTE_LOCK.read_text(encoding="utf-8"))
    domains = capture["domains"]
    route_data: dict[int, dict[str, np.ndarray]] = {}
    for layer in range(capture["layers"]):
        path = ROUTES / f"layer_{layer:02d}.safetensors"
        expected = capture["manifests"][str(layer)]["artifact_sha256"]
        if sha256(path) != expected:
            raise RuntimeError(f"route artifact hash changed: {path}")
        tensors = load_file(path)
        layer_data: dict[str, np.ndarray] = {}
        for domain in domains:
            key = f"{domain}_router_ids"
            value = tensors[key]
            if value.shape != (1024, TOP_K):
                raise RuntimeError(f"unexpected {key} shape {value.shape}")
            layer_data[domain] = value
        route_data[layer] = layer_data
    return route_data, capture, lock


def gates(metrics: dict[str, Any]) -> dict[str, Any]:
    aggregate = metrics["aggregate"]
    bytes_pass = aggregate["critical_expert_bytes_reduction_x"] >= 4.0
    worst_pass = aggregate["worst_case_new_load_reduction_x"] >= 8.0
    return {
        "critical_expert_bytes_reduction_at_least_4x": bool(bytes_pass),
        "worst_case_new_load_reduction_at_least_8x": bool(worst_pass),
        "traffic_gates_pass": bool(bytes_pass and worst_pass),
        "quality_regression_at_most_1_percent": "UNTESTED_REQUIRES_TRAINING",
        "measured_p95_at_least_2x": "UNTESTED_NO_RUNTIME_EXECUTION",
        "no_p99_collapse": "UNTESTED_NO_RUNTIME_EXECUTION",
        "second_memory_hierarchy": "UNTESTED",
    }


def common_metadata(phase: str, capture: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "tierflow_f0_cpu_trace_feasibility",
        "phase": phase,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Traffic-only optimistic route-edit oracle on frozen real Qwen30 P4D "
            "routes; no training, LM quality, expert execution, latency, or system claim."
        ),
        "inputs": {
            "preregistration": str(PREREG.relative_to(ROOT)).replace("\\", "/"),
            "preregistration_sha256": sha256(PREREG),
            "route_capture_sha256": sha256(CAPTURE),
            "route_input_lock_sha256": sha256(ROUTE_LOCK),
            "model_index_sha256": capture["inputs"]["model_index_sha256"],
        },
        "trace": {
            "model_variant": capture["model_variant"],
            "layers": capture["layers"],
            "domains": capture["domains"],
            "tokens_per_domain": capture["tokens_per_domain"],
            "top_k": TOP_K,
            "experts": EXPERTS,
            "expert_record_bytes": lock["cache"]["expert_record_bytes"],
        },
        "oracle": {
            "edit_semantics": "one replacement equals one newly admitted expert",
            "state_cardinality": TOP_K,
            "tie_break": "within-partition clairvoyant next-use/frequency/expert-id",
            "causal": False,
            "conditional_current_token_overlap_optimal": True,
        },
    }


def run_validation() -> dict[str, Any]:
    route_data, capture, lock = load_locked_routes()
    start, end = lock["partitions"]["validation"]
    expert_record_bytes = int(lock["cache"]["expert_record_bytes"])
    result = common_metadata("validation", capture, lock)
    result["partition"] = [start, end]
    result["candidates"] = {}
    passing: list[int] = []
    for r in EDIT_BUDGETS:
        metrics = evaluate_partition(
            route_data, capture["domains"], start, end, r, expert_record_bytes
        )
        candidate = {"metrics": metrics, "gates": gates(metrics)}
        result["candidates"][str(r)] = candidate
        if candidate["gates"]["traffic_gates_pass"]:
            passing.append(r)

    if passing:
        selected = max(
            passing,
            key=lambda r: (
                result["candidates"][str(r)]["metrics"]["aggregate"][
                    "mean_route_set_overlap"
                ],
                -r,
            ),
        )
        result["selected_edit_budget"] = selected
        result["status"] = "validation_traffic_pass_test_authorized"
    else:
        result["selected_edit_budget"] = None
        result["status"] = "validation_traffic_fail_test_closed"
    result["test_opened"] = False
    VALIDATION_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_test() -> dict[str, Any]:
    if not VALIDATION_OUT.exists():
        raise RuntimeError("validation result missing; test remains closed")
    validation = json.loads(VALIDATION_OUT.read_text(encoding="utf-8"))
    if validation["status"] != "validation_traffic_pass_test_authorized":
        raise RuntimeError("validation did not authorize test")
    if validation["inputs"]["preregistration_sha256"] != sha256(PREREG):
        raise RuntimeError("preregistration changed after validation")

    route_data, capture, lock = load_locked_routes()
    start, end = lock["partitions"]["test"]
    expert_record_bytes = int(lock["cache"]["expert_record_bytes"])
    r = int(validation["selected_edit_budget"])
    metrics = evaluate_partition(
        route_data, capture["domains"], start, end, r, expert_record_bytes
    )
    test_gates = gates(metrics)
    result = common_metadata("test", capture, lock)
    result.update(
        {
            "partition": [start, end],
            "selected_edit_budget": r,
            "validation_result_sha256": sha256(VALIDATION_OUT),
            "validation_summary": validation["candidates"][str(r)],
            "test": {"metrics": metrics, "gates": test_gates},
            "test_opened": True,
            "status": (
                "traffic_feasible_quality_and_runtime_untested"
                if test_gates["traffic_gates_pass"]
                else "heldout_traffic_fail"
            ),
            "conclusion": (
                "The selected bounded-edit process meets only the two trace traffic "
                "targets. Its substitution rate measures the unsupported behavioral "
                "change that training must absorb; quality remains wholly untested."
            ),
        }
    )
    RESULT_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    result = run_validation() if args.phase == "validation" else run_test()
    print(json.dumps({
        "status": result["status"],
        "selected_edit_budget": result.get("selected_edit_budget"),
        "output": str((VALIDATION_OUT if args.phase == "validation" else RESULT_OUT).relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()

