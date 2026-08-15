#!/usr/bin/env python3
"""Independent raw-route verifier for TierFlow-F0.

This implementation intentionally does not import the experiment runner.  It
reloads all frozen P4D route tensors and reimplements the bounded-edit oracle
with occurrence lists and bisect-based future ranking.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
ROUTE_DIR = ROOT / "reports" / "runs" / "streamq5_moe" / "p4d_routes"
CAPTURE = R / "p4d_route_capture_result.json"
LOCK = R / "p4d_route_input_lock.json"
PREREG = R / "TIERFLOW_F0_PREREGISTRATION_2026-08-12.md"
VALIDATION = R / "tierflow_f0_validation.json"
RESULT = R / "tierflow_f0_result.json"
OUT_JSON = R / "tierflow_f0_independent_verification.json"
OUT_REPORT = R / "TIERFLOW_F0_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

EXPECTED_CAPTURE = "7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb"
R_VALUES = (1, 2, 4)
K = 8
N_EXPERTS = 128
ABS_TOL = 1e-12


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def occurrences(routes: np.ndarray) -> list[list[int]]:
    positions: list[list[int]] = [[] for _ in range(N_EXPERTS)]
    for token, row in enumerate(routes.tolist()):
        for expert in row:
            positions[expert].append(token)
    return positions


def priority(positions: list[list[int]], expert: int, token: int, length: int) -> tuple[int, int, int]:
    """Earlier next use, more remaining uses, lower ID is more valuable."""
    expert_positions = positions[expert]
    after = bisect.bisect_right(expert_positions, token)
    next_use = expert_positions[after] if after < len(expert_positions) else length + 1
    remaining = len(expert_positions) - after
    return next_use, -remaining, expert


def independently_simulate(full: np.ndarray, start: int, end: int, budget: int) -> dict[str, Any]:
    routes = full[start:end].astype(np.int64, copy=False)
    pos = occurrences(routes)
    state = set(int(x) for x in full[start - 1].tolist())
    previous_observed = set(state)
    baseline: list[int] = []
    edited: list[int] = []
    overlap: list[int] = []
    exact = 0

    for token, raw in enumerate(routes):
        requested = set(int(x) for x in raw.tolist())
        baseline.append(len(requested - previous_observed))
        old_state = set(state)

        wanted = requested - state
        count = min(budget, len(wanted))
        admit = sorted(
            wanted, key=lambda e: priority(pos, e, token, len(routes))
        )[:count]
        removable = state - requested
        evict = sorted(
            removable,
            key=lambda e: priority(pos, e, token, len(routes)),
            reverse=True,
        )[:count]
        for expert in evict:
            state.remove(expert)
        for expert in admit:
            state.add(expert)

        additions = len(state - old_state)
        if len(state) != K or additions != count or additions > budget:
            raise AssertionError("independent state/edit invariant failed")
        edited.append(additions)
        common = len(state & requested)
        overlap.append(common)
        exact += int(state == requested)
        previous_observed = requested

    return {"baseline": baseline, "edited": edited, "overlap": overlap, "exact": exact}


def dist(values: list[int]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def summary(baseline: list[int], edited: list[int], overlap: list[int], exact: int, record: int) -> dict[str, Any]:
    baseline_total = sum(baseline)
    edited_total = sum(edited)
    return {
        "transitions": len(overlap),
        "baseline_new_loads": dist(baseline),
        "oracle_new_loads": dist(edited),
        "baseline_critical_expert_bytes": baseline_total * record,
        "oracle_critical_expert_bytes": edited_total * record,
        "critical_expert_bytes_reduction_x": baseline_total / edited_total if edited_total else math.inf,
        "worst_case_new_load_reduction_x": max(baseline) / max(edited) if max(edited) else math.inf,
        "mean_route_set_overlap": sum(overlap) / (len(overlap) * K),
        "router_output_substitution_rate": 1.0 - sum(overlap) / (len(overlap) * K),
        "exact_route_set_match_rate": exact / len(overlap),
        "overlap_count": dist(overlap),
    }


def evaluate(raw: dict[int, dict[str, np.ndarray]], domains: list[str], bounds: list[int], budget: int, record: int) -> dict[str, Any]:
    start, end = bounds
    overall_b: list[int] = []
    overall_e: list[int] = []
    overall_o: list[int] = []
    overall_exact = 0
    per_domain: dict[str, Any] = {}
    for domain in domains:
        db: list[int] = []
        de: list[int] = []
        do: list[int] = []
        dx = 0
        for layer in range(48):
            run = independently_simulate(raw[layer][domain], start, end, budget)
            db.extend(run["baseline"])
            de.extend(run["edited"])
            do.extend(run["overlap"])
            dx += run["exact"]
        per_domain[domain] = summary(db, de, do, dx, record)
        overall_b.extend(db)
        overall_e.extend(de)
        overall_o.extend(do)
        overall_exact += dx
    aggregate = summary(overall_b, overall_e, overall_o, overall_exact, record)
    aggregate["cold_start_expert_bytes_separate"] = len(domains) * 48 * K * record
    aggregate["stream_all_top8_bytes_without_reuse"] = len(domains) * 48 * (end - start) * K * record
    return {"aggregate": aggregate, "per_domain": per_domain}


def close(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return all(key in b and close(value, b[key]) for key, value in a.items())
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=ABS_TOL)
    return a == b


def gate_view(metrics: dict[str, Any]) -> dict[str, bool]:
    a = metrics["aggregate"]
    return {
        "critical_expert_bytes_reduction_at_least_4x": a["critical_expert_bytes_reduction_x"] >= 4.0,
        "worst_case_new_load_reduction_at_least_8x": a["worst_case_new_load_reduction_x"] >= 8.0,
        "traffic_gates_pass": a["critical_expert_bytes_reduction_x"] >= 4.0
        and a["worst_case_new_load_reduction_x"] >= 8.0,
    }


def main() -> None:
    checks: list[dict[str, Any]] = []
    capture_hash = digest(CAPTURE)
    check(checks, "capture_hash", capture_hash == EXPECTED_CAPTURE, capture_hash)
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    check(checks, "partition_lock_exact", lock["partitions"] == {
        "calibration": [0, 512], "validation": [512, 768], "test": [768, 1024]
    }, lock["partitions"])
    check(checks, "validation_test_disjoint_adjacent", lock["partitions"]["validation"][1] == lock["partitions"]["test"][0])
    check(checks, "published_partition_match", validation["partition"] == [512, 768] and result["partition"] == [768, 1024])
    prereg_hash = digest(PREREG)
    check(checks, "prereg_hash_locked", validation["inputs"]["preregistration_sha256"] == prereg_hash == result["inputs"]["preregistration_sha256"], prereg_hash)
    check(checks, "validation_hash_locked_by_test", result["validation_result_sha256"] == digest(VALIDATION))

    raw: dict[int, dict[str, np.ndarray]] = {}
    hash_failures: list[int] = []
    shape_failures: list[str] = []
    route_failures: list[str] = []
    for layer in range(48):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        if digest(path) != capture["manifests"][str(layer)]["artifact_sha256"]:
            hash_failures.append(layer)
        tensors = load_file(path)
        raw[layer] = {}
        for domain in capture["domains"]:
            array = tensors[f"{domain}_router_ids"]
            if array.shape != (1024, K):
                shape_failures.append(f"{layer}:{domain}:{array.shape}")
            if np.any(array < 0) or np.any(array >= N_EXPERTS):
                route_failures.append(f"{layer}:{domain}:range")
            if any(len(set(row)) != K for row in array.tolist()):
                route_failures.append(f"{layer}:{domain}:duplicates")
            raw[layer][domain] = array
    check(checks, "all_48_route_hashes", not hash_failures, hash_failures)
    check(checks, "all_route_shapes", not shape_failures, shape_failures)
    check(checks, "all_route_ids_unique_and_valid", not route_failures, route_failures)

    record = int(lock["cache"]["expert_record_bytes"])
    recomputed_validation: dict[str, Any] = {}
    passing: list[int] = []
    for budget in R_VALUES:
        recomputed = evaluate(raw, capture["domains"], [512, 768], budget, record)
        recomputed_validation[str(budget)] = recomputed
        published = validation["candidates"][str(budget)]
        check(checks, f"validation_r{budget}_raw_metrics", close(recomputed, published["metrics"]))
        independent_gates = gate_view(recomputed)
        check(checks, f"validation_r{budget}_gates", all(published["gates"][k] == v for k, v in independent_gates.items()), independent_gates)
        if independent_gates["traffic_gates_pass"]:
            passing.append(budget)

    selected = max(passing, key=lambda r: (recomputed_validation[str(r)]["aggregate"]["mean_route_set_overlap"], -r)) if passing else None
    check(checks, "validation_selection", selected == validation["selected_edit_budget"] == 1, selected)

    recomputed_test = evaluate(raw, capture["domains"], [768, 1024], selected, record)
    check(checks, "test_raw_metrics", close(recomputed_test, result["test"]["metrics"]))
    independent_test_gates = gate_view(recomputed_test)
    check(checks, "test_gates", all(result["test"]["gates"][k] == v for k, v in independent_test_gates.items()), independent_test_gates)

    a = recomputed_test["aggregate"]
    baseline_loads = round(a["baseline_new_loads"]["mean"] * a["transitions"])
    oracle_loads = round(a["oracle_new_loads"]["mean"] * a["transitions"])
    arithmetic_ok = (
        a["baseline_critical_expert_bytes"] == baseline_loads * record
        and a["oracle_critical_expert_bytes"] == oracle_loads * record
        and math.isclose(a["critical_expert_bytes_reduction_x"], baseline_loads / oracle_loads, abs_tol=ABS_TOL)
        and math.isclose(a["mean_route_set_overlap"] + a["router_output_substitution_rate"], 1.0, abs_tol=ABS_TOL)
    )
    check(checks, "test_byte_and_rate_arithmetic", arithmetic_ok)
    check(checks, "claim_boundary_quality_unproven", "quality" in result["status"] and result["test"]["gates"]["quality_regression_at_most_1_percent"].startswith("UNTESTED"))

    failures = [item for item in checks if not item["pass"]]
    verification = {
        "kind": "tierflow_f0_independent_raw_route_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "independence": "No import from run_tierflow_f0_trace_feasibility.py; raw safetensors reloaded and oracle reimplemented with occurrence lists/bisect.",
        "inputs": {
            "capture_sha256": capture_hash,
            "preregistration_sha256": prereg_hash,
            "validation_sha256": digest(VALIDATION),
            "result_sha256": digest(RESULT),
            "raw_route_artifacts_verified": 48,
        },
        "checks": checks,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "status": "independent_verification_pass" if not failures else "independent_verification_fail",
        "recomputed": {
            "validation": {r: {"aggregate": x["aggregate"], "gates": gate_view(x)} for r, x in recomputed_validation.items()},
            "selected_edit_budget": selected,
            "test": {"aggregate": a, "gates": independent_test_gates},
        },
        "failures": failures,
        "claim_boundary": "Verifies route-trace arithmetic only; no model training, LM quality, runtime, latency, or hardware-hierarchy result.",
    }
    OUT_JSON.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    report = f"""# TierFlow-F0 independent verification

## Verdict

**{verification['status']} — {verification['checks_passed']}/{verification['checks_total']} checks passed.**

The verifier independently reloaded all 48 raw P4D safetensors, checked their
locked hashes and route invariants, reimplemented the oracle without importing
the experiment runner, recomputed all validation budgets and the one selected
held-out test, and verified byte/rate arithmetic and gates.

## Independently recomputed held-out result

- selected edit budget: `{selected}`;
- critical-byte reduction: **{a['critical_expert_bytes_reduction_x']:.6f}x**;
- worst-case new-load reduction: **{a['worst_case_new_load_reduction_x']:.1f}x**;
- mean route overlap: **{100*a['mean_route_set_overlap']:.2f}%**;
- router-output substitution: **{100*a['router_output_substitution_rate']:.2f}%**;
- traffic gates: **{'pass' if independent_test_gates['traffic_gates_pass'] else 'fail'}**.

This verification is limited to frozen-route traffic feasibility. LM quality,
causal-controller performance, measured latency and a second memory hierarchy
remain untested.

## Artifacts

- verifier: `scripts/streamq5_moe/verify_tierflow_f0_independent.py`
- machine-readable result: `reports/streamq5_moe/tierflow_f0_independent_verification.json`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": verification["status"], "checks": f"{verification['checks_passed']}/{verification['checks_total']}", "output": str(OUT_JSON.relative_to(ROOT))}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

