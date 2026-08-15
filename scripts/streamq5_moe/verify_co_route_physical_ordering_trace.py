#!/usr/bin/env python3
"""Independent verifier for the co-route physical-ordering trace test.

The verifier does not import the runner.  It rebuilds learned orders from raw
routes and enumerates all 2^7 contiguous partitions of each top-8 route rather
than using the runner's dynamic program.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
ROUTES = ROOT / "reports" / "runs" / "streamq5_moe" / "p4d_routes"
CAPTURE = R / "p4d_route_capture_result.json"
LOCK = R / "p4d_route_input_lock.json"
PREREG = R / "CO_ROUTE_PHYSICAL_ORDERING_TRACE_PREREGISTRATION_2026-08-12.md"
VALIDATION_RESULT = R / "co_route_physical_ordering_trace_validation.json"
TEST_RESULT = R / "co_route_physical_ordering_trace_test.json"
OUT = R / "co_route_physical_ordering_trace_independent_verification.json"
OUT_REPORT = R / "CO_ROUTE_PHYSICAL_ORDERING_TRACE_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

EXPECTED_CAPTURE = "7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb"
N_EXPERTS = 128
K = 8
LAYERS = 48
LEARN = (0, 512)
VALIDATION = (512, 768)
TEST = (768, 1024)
TOL = 1e-12


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def independent_order(layer: dict[str, np.ndarray], domains: list[str]) -> tuple[list[int], list[int], int, int]:
    counts = [0] * N_EXPERTS
    pair = [[0] * N_EXPERTS for _ in range(N_EXPERTS)]
    for domain in domains:
        for raw_row in layer[domain][LEARN[0] : LEARN[1]]:
            row = sorted(int(value) for value in raw_row.tolist())
            for expert in row:
                counts[expert] += 1
            for left, right in itertools.combinations(row, 2):
                pair[left][right] += 1
                pair[right][left] += 1

    schedule = sorted(range(N_EXPERTS), key=lambda expert: (-counts[expert], expert))
    result: list[int] = []
    for expert in schedule:
        scores: list[int] = []
        for location in range(len(result) + 1):
            score = 0
            if location:
                score += pair[result[location - 1]][expert]
            if location < len(result):
                score += pair[expert][result[location]]
            if location and location < len(result):
                score -= pair[result[location - 1]][result[location]]
            scores.append(score)
        chosen = max(range(len(scores)), key=lambda location: (scores[location], -location))
        result.insert(chosen, expert)

    pair_sum = sum(pair[left][right] for left in range(N_EXPERTS) for right in range(left + 1, N_EXPERTS))
    adjacent = sum(pair[result[index]][result[index + 1]] for index in range(N_EXPERTS - 1))
    return result, counts, pair_sum, adjacent


def brute_cover(route: np.ndarray, order: list[int]) -> tuple[int, int, int]:
    inverse = {expert: position for position, expert in enumerate(order)}
    positions = sorted(inverse[int(expert)] for expert in route.tolist())
    best: tuple[int, int, tuple[tuple[int, int], ...]] | None = None
    # A bit after selected position i ends the current interval. Position 7
    # always ends the final interval.
    for cuts in range(1 << (K - 1)):
        groups: list[tuple[int, int]] = []
        begin = 0
        for index in range(K - 1):
            if cuts & (1 << index):
                groups.append((begin, index))
                begin = index + 1
        groups.append((begin, K - 1))
        intervals = tuple((positions[first], positions[last]) for first, last in groups)
        transferred = sum(end - start + 1 for start, end in intervals)
        extra = transferred - K
        if extra > 1:
            continue
        candidate = (len(intervals), transferred, intervals)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise AssertionError("brute-force interval cover missing")
    return best[0], best[1], best[1] - K


def dist(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def summary(counts: list[int], records: list[int]) -> dict[str, Any]:
    required = len(counts) * K
    return {
        "routes": len(counts),
        "interval_count": dist(counts),
        "transferred_records": dist(records),
        "required_records": required,
        "total_transferred_records": sum(records),
        "payload_inflation_x": sum(records) / required,
        "coverage_errors": 0,
    }


def independent_metrics(
    raw: dict[int, dict[str, np.ndarray]],
    domains: list[str],
    bounds: tuple[int, int],
    orders: dict[int, list[int]],
) -> dict[str, Any]:
    overall_counts: list[int] = []
    overall_records: list[int] = []
    per_domain: dict[str, Any] = {}
    per_layer_counts: dict[int, list[int]] = {layer: [] for layer in range(LAYERS)}
    per_layer_records: dict[int, list[int]] = {layer: [] for layer in range(LAYERS)}
    raw_view: dict[str, Any] = {str(layer): {} for layer in range(LAYERS)}

    for domain in domains:
        dc: list[int] = []
        dr: list[int] = []
        for layer in range(LAYERS):
            lc: list[int] = []
            lr: list[int] = []
            le: list[int] = []
            for route in raw[layer][domain][bounds[0] : bounds[1]]:
                count, records, extra = brute_cover(route, orders[layer])
                lc.append(count)
                lr.append(records)
                le.append(extra)
            raw_view[str(layer)][domain] = {
                "interval_counts": lc,
                "transferred_records": lr,
                "extra_records": le,
            }
            dc.extend(lc)
            dr.extend(lr)
            per_layer_counts[layer].extend(lc)
            per_layer_records[layer].extend(lr)
        per_domain[domain] = summary(dc, dr)
        overall_counts.extend(dc)
        overall_records.extend(dr)

    per_layer = {
        str(layer): summary(per_layer_counts[layer], per_layer_records[layer])
        for layer in range(LAYERS)
    }
    return {
        "aggregate": summary(overall_counts, overall_records),
        "per_domain": per_domain,
        "per_layer": per_layer,
        "raw": raw_view,
    }


def independent_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    aggregate = metrics["aggregate"]
    result = {
        "aggregate_p95_intervals_at_most_2": aggregate["interval_count"]["p95"] <= 2.0,
        "aggregate_mean_intervals_at_most_1_5": aggregate["interval_count"]["mean"] <= 1.5,
        "every_domain_p95_intervals_at_most_3": all(
            values["interval_count"]["p95"] <= 3.0
            for values in metrics["per_domain"].values()
        ),
        "aggregate_payload_inflation_at_most_1_10": aggregate["payload_inflation_x"] <= 1.10,
        "exact_coverage": aggregate["coverage_errors"] == 0,
        "valid_permutations": True,
        "learn_only_provenance": True,
    }
    result["trace_gate_pass"] = all(result.values())
    return result


def close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=TOL)
    return left == right


def main() -> None:
    checks: list[dict[str, Any]] = []
    add(checks, "capture_hash", digest(CAPTURE) == EXPECTED_CAPTURE, digest(CAPTURE))
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    add(checks, "locked_capture_dimensions", capture["layers"] == LAYERS and capture["tokens_per_domain"] == 1024)
    add(checks, "expected_expert_record_bytes", int(lock["cache"]["expert_record_bytes"]) == 3_035_136)
    add(checks, "strictly_disjoint_partitions", LEARN[1] == VALIDATION[0] and VALIDATION[1] == TEST[0])

    raw: dict[int, dict[str, np.ndarray]] = {}
    bad_hashes: list[int] = []
    bad_routes: list[str] = []
    for layer in range(LAYERS):
        path = ROUTES / f"layer_{layer:02d}.safetensors"
        if digest(path) != capture["manifests"][str(layer)]["artifact_sha256"]:
            bad_hashes.append(layer)
        tensors = load_file(path)
        raw[layer] = {}
        for domain in capture["domains"]:
            routes = tensors[f"{domain}_router_ids"].astype(np.int64, copy=False)
            if routes.shape != (1024, K) or np.any(routes < 0) or np.any(routes >= N_EXPERTS):
                bad_routes.append(f"{layer}:{domain}:shape_or_range")
            elif any(len(set(row)) != K for row in routes.tolist()):
                bad_routes.append(f"{layer}:{domain}:duplicate")
            raw[layer][domain] = routes
    add(checks, "all_route_hashes", not bad_hashes, bad_hashes)
    add(checks, "all_route_invariants", not bad_routes, bad_routes)

    orders: dict[int, list[int]] = {}
    diagnostics: dict[str, Any] = {}
    for layer in range(LAYERS):
        order, frequency, pair_sum, adjacent = independent_order(raw[layer], capture["domains"])
        orders[layer] = order
        diagnostics[str(layer)] = {
            "frequency": frequency,
            "cooccurrence_sum": pair_sum,
            "adjacent_cooccurrence_objective": adjacent,
        }
    add(checks, "all_orders_are_permutations", all(sorted(order) == list(range(N_EXPERTS)) for order in orders.values()))

    validation = json.loads(VALIDATION_RESULT.read_text(encoding="utf-8"))
    add(checks, "preregistration_hash_locked", validation["inputs"]["preregistration_sha256"] == digest(PREREG), digest(PREREG))
    add(checks, "validation_partition", validation["partition"] == list(VALIDATION))
    add(checks, "learned_orders_rederived", validation["learned_orders"] == {str(layer): order for layer, order in orders.items()})
    add(checks, "learn_diagnostics_rederived", close(validation["learn_diagnostics"], diagnostics))

    recomputed_validation = independent_metrics(raw, capture["domains"], VALIDATION, orders)
    add(checks, "validation_raw_and_summary_recomputed", close(recomputed_validation, validation["learned_ordering"]["metrics"]))
    val_gates = independent_gates(recomputed_validation)
    add(checks, "validation_gates_recomputed", validation["learned_ordering"]["gates"] == val_gates, val_gates)

    validation_pass = val_gates["trace_gate_pass"]
    if validation_pass:
        add(checks, "validation_status_authorizes_test", validation["status"] == "validation_trace_pass_test_authorized")
        add(checks, "test_result_exists_after_authorization", TEST_RESULT.exists())
    else:
        add(checks, "validation_status_closes_test", validation["status"] == "validation_trace_fail_test_closed")
        add(checks, "test_result_absent_when_closed", not TEST_RESULT.exists())

    target = validation
    recomputed_target = recomputed_validation
    target_gates = val_gates
    if validation_pass and TEST_RESULT.exists():
        test = json.loads(TEST_RESULT.read_text(encoding="utf-8"))
        add(checks, "test_locks_validation_hash", test["validation_sha256"] == digest(VALIDATION_RESULT))
        add(checks, "test_partition", test["partition"] == list(TEST))
        add(checks, "test_uses_frozen_orders", test["learned_orders"] == validation["learned_orders"])
        recomputed_test = independent_metrics(raw, capture["domains"], TEST, orders)
        add(checks, "test_raw_and_summary_recomputed", close(recomputed_test, test["learned_ordering"]["metrics"]))
        test_gates = independent_gates(recomputed_test)
        add(checks, "test_gates_recomputed", test["learned_ordering"]["gates"] == test_gates, test_gates)
        target = test
        recomputed_target = recomputed_test
        target_gates = test_gates

    identity = {layer: list(range(N_EXPERTS)) for layer in range(LAYERS)}
    identity_metrics = independent_metrics(
        raw,
        capture["domains"],
        TEST if target["phase"] == "test" else VALIDATION,
        identity,
    )
    published_identity = dict(target["identity_ordering_baseline"]["metrics"])
    published_identity.pop("raw", None)
    identity_metrics_no_raw = dict(identity_metrics)
    identity_metrics_no_raw.pop("raw", None)
    add(checks, "identity_baseline_recomputed", close(identity_metrics_no_raw, published_identity))
    add(checks, "gpu_not_authorized", target["gpu_authorized"] is False)

    failures = [item for item in checks if not item["pass"]]
    verification = {
        "kind": "co_route_physical_ordering_independent_raw_trace_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "independence": (
            "No runner import; raw P4D tensors reloaded, learned insertion order "
            "reimplemented, and all 2^7 route partitions brute-force enumerated."
        ),
        "inputs": {
            "preregistration_sha256": digest(PREREG),
            "validation_sha256": digest(VALIDATION_RESULT),
            "test_sha256": digest(TEST_RESULT) if TEST_RESULT.exists() else None,
            "raw_route_artifacts_verified": LAYERS,
        },
        "checks": checks,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "failures": failures,
        "status": "independent_verification_pass" if not failures else "independent_verification_fail",
        "verified_phase": target["phase"],
        "recomputed": {
            "aggregate": recomputed_target["aggregate"],
            "per_domain": recomputed_target["per_domain"],
            "gates": target_gates,
        },
        "claim_boundary": "Trace locality only; no GPU, physical bank, copied bytes, latency, quality, or 80B result.",
    }
    OUT.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    aggregate = recomputed_target["aggregate"]
    report = f"""# Co-route physical ordering — independent verification

## Verdict

**{verification['status']} — {verification['checks_passed']}/{verification['checks_total']} checks passed.**

The verifier independently reloaded all 48 route tensors, rebuilt every
learned order from learn-only rows, and brute-force enumerated all 128
contiguous partitions of every evaluated top-8 route.

Verified `{target['phase']}` metrics:

- mean intervals: `{aggregate['interval_count']['mean']:.6f}`;
- p95 intervals: `{aggregate['interval_count']['p95']:.3f}`;
- payload inflation: `{aggregate['payload_inflation_x']:.6f}x`;
- trace gate: `{'pass' if target_gates['trace_gate_pass'] else 'fail'}`.

No GPU or physical-bank work is authorized by this verifier.

## Artifacts

- verifier: `scripts/streamq5_moe/verify_co_route_physical_ordering_trace.py`
- machine-readable verification: `reports/streamq5_moe/co_route_physical_ordering_trace_independent_verification.json`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": verification["status"], "checks": f"{verification['checks_passed']}/{verification['checks_total']}", "verified_phase": target["phase"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
