#!/usr/bin/env python3
"""CPU-only co-route-aware physical expert-ordering trace test.

Validation is computed before test.  A failed validation leaves the test
partition unopened and authorizes no physical-bank or GPU work.
"""

from __future__ import annotations

import argparse
import functools
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
ROUTE_DIR = ROOT / "reports" / "runs" / "streamq5_moe" / "p4d_routes"
CAPTURE = REPORTS / "p4d_route_capture_result.json"
LOCK = REPORTS / "p4d_route_input_lock.json"
PREREG = REPORTS / "CO_ROUTE_PHYSICAL_ORDERING_TRACE_PREREGISTRATION_2026-08-12.md"
VALIDATION_OUT = REPORTS / "co_route_physical_ordering_trace_validation.json"
TEST_OUT = REPORTS / "co_route_physical_ordering_trace_test.json"
REPORT_OUT = REPORTS / "CO_ROUTE_PHYSICAL_ORDERING_TRACE_REPORT_2026-08-12.md"

EXPECTED_CAPTURE_SHA256 = (
    "7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb"
)
LEARN = (0, 512)
VALIDATION = (512, 768)
TEST = (768, 1024)
N_LAYERS = 48
N_EXPERTS = 128
TOP_K = 8
EXTRA_RECORD_BUDGET = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def load_locked_routes() -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    capture_hash = sha256(CAPTURE)
    if capture_hash != EXPECTED_CAPTURE_SHA256:
        raise RuntimeError(f"capture hash changed: {capture_hash}")
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if capture["layers"] != N_LAYERS or capture["tokens_per_domain"] != 1024:
        raise RuntimeError("unexpected P4D capture dimensions")
    raw: dict[int, dict[str, np.ndarray]] = {}
    for layer in range(N_LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        expected = capture["manifests"][str(layer)]["artifact_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"route hash changed for layer {layer}: {actual}")
        tensors = load_file(path)
        raw[layer] = {}
        for domain in capture["domains"]:
            routes = tensors[f"{domain}_router_ids"].astype(np.int64, copy=False)
            if routes.shape != (1024, TOP_K):
                raise RuntimeError(f"unexpected route shape {layer}:{domain}:{routes.shape}")
            if np.any(routes < 0) or np.any(routes >= N_EXPERTS):
                raise RuntimeError(f"invalid expert ID {layer}:{domain}")
            if any(len(set(row)) != TOP_K for row in routes.tolist()):
                raise RuntimeError(f"duplicate top-k expert {layer}:{domain}")
            raw[layer][domain] = routes
    return raw, capture, lock


def learn_order(layer_routes: dict[str, np.ndarray], domains: list[str]) -> tuple[list[int], list[int], np.ndarray]:
    frequency = np.zeros(N_EXPERTS, dtype=np.int64)
    cooccurrence = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.int64)
    start, end = LEARN
    for domain in domains:
        for row in layer_routes[domain][start:end]:
            ids = sorted(int(x) for x in row.tolist())
            frequency[ids] += 1
            for i, left in enumerate(ids):
                for right in ids[i + 1 :]:
                    cooccurrence[left, right] += 1
                    cooccurrence[right, left] += 1

    process_order = sorted(range(N_EXPERTS), key=lambda expert: (-int(frequency[expert]), expert))
    physical: list[int] = []
    for expert in process_order:
        if not physical:
            physical.append(expert)
            continue
        best_position = 0
        best_gain: int | None = None
        for position in range(len(physical) + 1):
            gain = 0
            if position > 0:
                gain += int(cooccurrence[physical[position - 1], expert])
            if position < len(physical):
                gain += int(cooccurrence[expert, physical[position]])
            if 0 < position < len(physical):
                gain -= int(cooccurrence[physical[position - 1], physical[position]])
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_position = position
        physical.insert(best_position, expert)

    if sorted(physical) != list(range(N_EXPERTS)):
        raise AssertionError("learned order is not a permutation")
    return physical, frequency.tolist(), cooccurrence


def optimal_interval_cover(route: np.ndarray, physical_order: list[int]) -> dict[str, Any]:
    inverse = [0] * N_EXPERTS
    for position, expert in enumerate(physical_order):
        inverse[expert] = position
    positions = tuple(sorted(inverse[int(expert)] for expert in route.tolist()))

    @functools.lru_cache(maxsize=None)
    def solve(index: int, remaining_extra: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        if index == TOP_K:
            return 0, 0, ()
        best: tuple[int, int, tuple[tuple[int, int], ...]] | None = None
        for last in range(index, TOP_K):
            span = positions[last] - positions[index] + 1
            required = last - index + 1
            extra = span - required
            if extra > remaining_extra:
                continue
            rest_count, rest_records, rest_intervals = solve(last + 1, remaining_extra - extra)
            candidate = (
                1 + rest_count,
                span + rest_records,
                ((positions[index], positions[last]),) + rest_intervals,
            )
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise AssertionError("no interval cover")
        return best

    interval_count, transferred, intervals = solve(0, EXTRA_RECORD_BUDGET)
    covered: set[int] = set()
    for start, end in intervals:
        covered.update(range(start, end + 1))
    required_positions = set(positions)
    coverage_ok = required_positions <= covered
    extra = len(covered - required_positions)
    if transferred != len(covered) or extra > EXTRA_RECORD_BUDGET:
        raise AssertionError("interval accounting mismatch")
    return {
        "interval_count": interval_count,
        "transferred_records": transferred,
        "extra_records": extra,
        "coverage_ok": coverage_ok,
        "intervals": [list(pair) for pair in intervals],
    }


def summarize_counts(intervals: list[int], transferred: list[int], coverage_errors: int) -> dict[str, Any]:
    required = len(intervals) * TOP_K
    return {
        "routes": len(intervals),
        "interval_count": distribution(intervals),
        "transferred_records": distribution(transferred),
        "required_records": required,
        "total_transferred_records": int(sum(transferred)),
        "payload_inflation_x": float(sum(transferred) / required),
        "coverage_errors": coverage_errors,
    }


def evaluate(
    raw: dict[int, dict[str, np.ndarray]],
    domains: list[str],
    bounds: tuple[int, int],
    orders: dict[int, list[int]],
    retain_raw: bool,
) -> dict[str, Any]:
    overall_i: list[int] = []
    overall_t: list[int] = []
    overall_errors = 0
    per_domain: dict[str, Any] = {}
    per_layer: dict[str, Any] = {}
    raw_metrics: dict[str, Any] = {}
    start, end = bounds

    for domain in domains:
        domain_i: list[int] = []
        domain_t: list[int] = []
        domain_errors = 0
        for layer in range(N_LAYERS):
            layer_key = str(layer)
            if layer_key not in raw_metrics:
                raw_metrics[layer_key] = {}
            entries = [optimal_interval_cover(row, orders[layer]) for row in raw[layer][domain][start:end]]
            counts = [entry["interval_count"] for entry in entries]
            records = [entry["transferred_records"] for entry in entries]
            errors = sum(not entry["coverage_ok"] for entry in entries)
            domain_i.extend(counts)
            domain_t.extend(records)
            domain_errors += errors
            if retain_raw:
                raw_metrics[layer_key][domain] = {
                    "interval_counts": counts,
                    "transferred_records": records,
                    "extra_records": [entry["extra_records"] for entry in entries],
                }
        per_domain[domain] = summarize_counts(domain_i, domain_t, domain_errors)
        overall_i.extend(domain_i)
        overall_t.extend(domain_t)
        overall_errors += domain_errors

    for layer in range(N_LAYERS):
        layer_i: list[int] = []
        layer_t: list[int] = []
        for domain in domains:
            if retain_raw:
                layer_i.extend(raw_metrics[str(layer)][domain]["interval_counts"])
                layer_t.extend(raw_metrics[str(layer)][domain]["transferred_records"])
            else:
                for row in raw[layer][domain][start:end]:
                    entry = optimal_interval_cover(row, orders[layer])
                    layer_i.append(entry["interval_count"])
                    layer_t.append(entry["transferred_records"])
        per_layer[str(layer)] = summarize_counts(layer_i, layer_t, 0)

    return {
        "aggregate": summarize_counts(overall_i, overall_t, overall_errors),
        "per_domain": per_domain,
        "per_layer": per_layer,
        "raw": raw_metrics if retain_raw else None,
    }


def gates(metrics: dict[str, Any], permutations_valid: bool, learn_only: bool) -> dict[str, bool]:
    aggregate = metrics["aggregate"]
    domain_p95 = {
        domain: values["interval_count"]["p95"]
        for domain, values in metrics["per_domain"].items()
    }
    result = {
        "aggregate_p95_intervals_at_most_2": aggregate["interval_count"]["p95"] <= 2.0,
        "aggregate_mean_intervals_at_most_1_5": aggregate["interval_count"]["mean"] <= 1.5,
        "every_domain_p95_intervals_at_most_3": all(value <= 3.0 for value in domain_p95.values()),
        "aggregate_payload_inflation_at_most_1_10": aggregate["payload_inflation_x"] <= 1.10,
        "exact_coverage": aggregate["coverage_errors"] == 0,
        "valid_permutations": permutations_valid,
        "learn_only_provenance": learn_only,
    }
    result["trace_gate_pass"] = all(result.values())
    return result


def learn_all(raw: dict[int, dict[str, np.ndarray]], domains: list[str]) -> tuple[dict[int, list[int]], dict[str, Any]]:
    orders: dict[int, list[int]] = {}
    diagnostics: dict[str, Any] = {}
    for layer in range(N_LAYERS):
        order, frequency, cooccurrence = learn_order(raw[layer], domains)
        orders[layer] = order
        diagnostics[str(layer)] = {
            "frequency": frequency,
            "cooccurrence_sum": int(cooccurrence.sum() // 2),
            "adjacent_cooccurrence_objective": int(
                sum(cooccurrence[order[index], order[index + 1]] for index in range(N_EXPERTS - 1))
            ),
        }
    return orders, diagnostics


def metadata(phase: str, capture: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "co_route_physical_ordering_cpu_trace",
        "phase": phase,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "CPU-only exact route-layout locality test on reused P4D captures; "
            "no bank, GPU, bytes copied, latency, model quality, or 80B claim."
        ),
        "inputs": {
            "preregistration": str(PREREG.relative_to(ROOT)).replace("\\", "/"),
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "capture_sha256": sha256(CAPTURE),
            "route_lock_sha256": sha256(LOCK),
            "model_index_sha256": capture["inputs"]["model_index_sha256"],
            "expert_record_bytes": int(lock["cache"]["expert_record_bytes"]),
        },
        "trace": {
            "layers": N_LAYERS,
            "domains": capture["domains"],
            "tokens_per_domain": 1024,
            "experts": N_EXPERTS,
            "top_k": TOP_K,
            "partitions": {
                "learn": list(LEARN),
                "validation": list(VALIDATION),
                "test": list(TEST),
            },
            "partitions_strictly_disjoint": True,
            "globally_fresh": False,
            "reuse_disclosure": "P4D validation/test windows were previously used by TierFlow-F0.",
        },
        "algorithm": {
            "learn": "descending-frequency deterministic maximum-adjacent-cooccurrence insertion",
            "interval_cover": "exact DP: min intervals, then records, then lexicographic intervals",
            "extra_records_per_route_max": EXTRA_RECORD_BUDGET,
            "selection_uses_validation_or_test": False,
        },
    }


def report(result: dict[str, Any]) -> None:
    primary = result["learned_ordering"]["metrics"]["aggregate"]
    identity = result["identity_ordering_baseline"]["metrics"]["aggregate"]
    gate_rows = "\n".join(
        f"| {name} | {'pass' if value else 'fail'} |"
        for name, value in result["learned_ordering"]["gates"].items()
    )
    test_statement = (
        "Test was opened exactly once after validation passed."
        if result.get("test_opened")
        else "Test remained closed because the validation trace gate failed."
    )
    verdict = "PASS" if result["learned_ordering"]["gates"]["trace_gate_pass"] else "FAIL"
    text = f"""# Co-route-aware physical expert ordering — trace report

## Verdict

**{verdict} on {result['phase']}.** {test_statement}

The learned physical ordering changes no route or model value. On the evaluated
partition its exact one-extra-record interval cover measured:

| metric | learned order | identity order |
|---|---:|---:|
| mean intervals / token / layer | {primary['interval_count']['mean']:.6f} | {identity['interval_count']['mean']:.6f} |
| p95 intervals | {primary['interval_count']['p95']:.3f} | {identity['interval_count']['p95']:.3f} |
| p99 intervals | {primary['interval_count']['p99']:.3f} | {identity['interval_count']['p99']:.3f} |
| payload inflation | {primary['payload_inflation_x']:.6f}x | {identity['payload_inflation_x']:.6f}x |
| coverage errors | {primary['coverage_errors']} | {identity['coverage_errors']} |

## Frozen gates

| gate | result |
|---|---|
{gate_rows}

## Split and claim boundary

Learn `[0,512)`, validation `[512,768)` and test `[768,1024)` are strictly
disjoint. The latter windows were previously used by TierFlow-F0, so this is
not a fresh-dataset confirmation. This result contains no GPU execution,
physical relayout, transfer, latency, model-quality, or 80B evidence.

## Artifacts

- preregistration: `reports/streamq5_moe/CO_ROUTE_PHYSICAL_ORDERING_TRACE_PREREGISTRATION_2026-08-12.md`
- runner: `scripts/streamq5_moe/run_co_route_physical_ordering_trace.py`
- raw result: `{result['output_artifact']}`
"""
    REPORT_OUT.write_text(text, encoding="utf-8")


def run_validation() -> dict[str, Any]:
    raw, capture, lock = load_locked_routes()
    orders, diagnostics = learn_all(raw, capture["domains"])
    learned = evaluate(raw, capture["domains"], VALIDATION, orders, retain_raw=True)
    identity_orders = {layer: list(range(N_EXPERTS)) for layer in range(N_LAYERS)}
    identity = evaluate(raw, capture["domains"], VALIDATION, identity_orders, retain_raw=False)
    permutations_valid = all(sorted(order) == list(range(N_EXPERTS)) for order in orders.values())
    learned_gates = gates(learned, permutations_valid, learn_only=True)
    result = metadata("validation", capture, lock)
    result.update(
        {
            "partition": list(VALIDATION),
            "learned_orders": {str(layer): order for layer, order in orders.items()},
            "learn_diagnostics": diagnostics,
            "learned_ordering": {"metrics": learned, "gates": learned_gates},
            "identity_ordering_baseline": {
                "metrics": identity,
                "gates": gates(identity, True, learn_only=True),
            },
            "status": (
                "validation_trace_pass_test_authorized"
                if learned_gates["trace_gate_pass"]
                else "validation_trace_fail_test_closed"
            ),
            "test_opened": False,
            "gpu_authorized": False,
            "output_artifact": str(VALIDATION_OUT.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    VALIDATION_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report(result)
    return result


def run_test() -> dict[str, Any]:
    if not VALIDATION_OUT.exists():
        raise RuntimeError("validation artifact missing")
    validation = json.loads(VALIDATION_OUT.read_text(encoding="utf-8"))
    if validation["status"] != "validation_trace_pass_test_authorized":
        raise RuntimeError("validation trace gate failed; test remains closed")
    if validation["inputs"]["preregistration_sha256"] != sha256(PREREG):
        raise RuntimeError("preregistration changed after validation")

    raw, capture, lock = load_locked_routes()
    recomputed_orders, diagnostics = learn_all(raw, capture["domains"])
    locked_orders = {int(layer): order for layer, order in validation["learned_orders"].items()}
    if recomputed_orders != locked_orders:
        raise RuntimeError("learned orders do not reproduce")
    learned = evaluate(raw, capture["domains"], TEST, locked_orders, retain_raw=True)
    identity_orders = {layer: list(range(N_EXPERTS)) for layer in range(N_LAYERS)}
    identity = evaluate(raw, capture["domains"], TEST, identity_orders, retain_raw=False)
    learned_gates = gates(learned, True, learn_only=True)
    result = metadata("test", capture, lock)
    result.update(
        {
            "partition": list(TEST),
            "validation_sha256": sha256(VALIDATION_OUT),
            "learned_orders": validation["learned_orders"],
            "learn_diagnostics": diagnostics,
            "learned_ordering": {"metrics": learned, "gates": learned_gates},
            "identity_ordering_baseline": {
                "metrics": identity,
                "gates": gates(identity, True, learn_only=True),
            },
            "status": "heldout_trace_pass_gpu_requires_prior_report" if learned_gates["trace_gate_pass"] else "heldout_trace_fail_no_gpu",
            "test_opened": True,
            "gpu_authorized": False,
            "output_artifact": str(TEST_OUT.relative_to(ROOT)).replace("\\", "/"),
        }
    )
    TEST_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("validation", "test"))
    args = parser.parse_args()
    result = run_validation() if args.phase == "validation" else run_test()
    aggregate = result["learned_ordering"]["metrics"]["aggregate"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "mean_intervals": aggregate["interval_count"]["mean"],
                "p95_intervals": aggregate["interval_count"]["p95"],
                "payload_inflation_x": aggregate["payload_inflation_x"],
                "trace_gate_pass": result["learned_ordering"]["gates"]["trace_gate_pass"],
                "output": result["output_artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
