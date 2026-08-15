#!/usr/bin/env python3
"""Independent arithmetic verifier for the TierFlow functional-span run.

This verifier deliberately does not import the experiment runner.  It rebuilds
all published summaries and gate decisions from the raw per-token arrays,
checks the locked local inputs, and confirms that validation failure left the
test partition unopened.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports" / "streamq5_moe"
RUNS = ROOT / "reports" / "runs" / "streamq5_moe"
PREREG = R / "TIERFLOW_PERSISTENT_SET_FUNCTIONAL_SPAN_PREREGISTRATION_2026-08-12.md"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_tierflow_persistent_set_functional_span.py"
VALIDATION = R / "tierflow_persistent_set_functional_span_validation.json"
TEST = R / "tierflow_persistent_set_functional_span_test.json"
INPUT_IDS = RUNS / "p4d_fresh_route_input_ids.safetensors"
INPUT_LOCK = R / "p4d_route_input_lock.json"
CAPTURE = R / "p4d_route_capture_result.json"
MODEL_INDEX = ROOT / "models" / "qwen3-30b-a3b-base" / "model.safetensors.index.json"
F0_REFERENCE = R / "tierflow_f0_validation.json"
OUT = R / "tierflow_persistent_set_functional_span_independent_verification.json"
REPORT = R / "TIERFLOW_PERSISTENT_SET_FUNCTIONAL_SPAN_INDEPENDENT_VERIFICATION_REPORT_2026-08-12.md"

DOMAINS = ("general", "code", "math", "multilingual", "instruction")
SENTINELS = (0, 24, 47)
EXPECTED_INPUT = "32838e94887f8572445159925e815f5353f55a20a954f9adc2f8cef48427af08"
EXPECTED_CAPTURE = "7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb"
# Published per-domain means were reduced in float32 by PyTorch, whereas this
# verifier reduces the serialized float32 samples in float64.  The tolerance
# covers only that deterministic reduction-order difference.
ABS_TOL = 5e-7
NEGATIVE_TOL = 1e-10


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def close(left: Any, right: Any, tolerance: float = ABS_TOL) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def rebuild_oracle(raw: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sentinel in SENTINELS:
        key = str(sentinel)
        metrics = {
            name: [value for domain in DOMAINS for value in raw[key][domain][name]]
            for name in (
                "relative_l2",
                "kkt_violation",
                "simplex_sum_error",
                "minimum_coefficient",
                "support_size",
                "route_overlap",
            )
        }
        result[key] = {name: describe(values) for name, values in metrics.items()}
        result[key]["minimum_coefficient"]["min"] = float(min(metrics["minimum_coefficient"]))
    return result


def rebuild_downstream(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_domains = payload["baseline"]["domains"]
    all_base = [value for domain in DOMAINS for value in baseline_domains[domain]["cross_entropy"]]
    rebuilt_baseline: dict[str, Any] = {
        "domains": {},
        "aggregate_cross_entropy": float(np.mean(all_base)),
    }
    for domain in DOMAINS:
        values = baseline_domains[domain]["cross_entropy"]
        rebuilt_baseline["domains"][domain] = {
            "mean_cross_entropy": float(np.mean(values)),
            "count": len(values),
        }

    rebuilt: dict[str, Any] = {}
    for sentinel in SENTINELS:
        key = str(sentinel)
        domains: dict[str, Any] = {}
        all_ce: list[float] = []
        all_kl: list[float] = []
        all_agreement: list[int] = []
        per_domain_relative: dict[str, float] = {}
        for domain in DOMAINS:
            raw = payload["downstream"][key]["domains"][domain]
            candidate = raw["candidate_cross_entropy"]
            kl = raw["natural_to_candidate_kl"]
            agreement = raw["top1_agreement"]
            candidate_mean = float(np.mean(candidate))
            base_mean = rebuilt_baseline["domains"][domain]["mean_cross_entropy"]
            relative = (candidate_mean - base_mean) / base_mean
            domains[domain] = {
                "count": len(candidate),
                "mean_candidate_cross_entropy": candidate_mean,
                "mean_kl": float(np.mean(kl)),
                "top1_agreement_rate": float(np.mean(agreement)),
                "relative_cross_entropy_regression": relative,
            }
            per_domain_relative[domain] = relative
            all_ce.extend(candidate)
            all_kl.extend(kl)
            all_agreement.extend(agreement)
        base_mean = rebuilt_baseline["aggregate_cross_entropy"]
        candidate_mean = float(np.mean(all_ce))
        rebuilt[key] = {
            "domains": domains,
            "aggregate": {
                "labels": len(all_ce),
                "baseline_cross_entropy": base_mean,
                "candidate_cross_entropy": candidate_mean,
                "relative_cross_entropy_regression": (candidate_mean - base_mean) / base_mean,
                "natural_to_candidate_kl": describe(all_kl),
                "top1_agreement": float(np.mean(all_agreement)),
                "per_domain_relative_cross_entropy": per_domain_relative,
            },
        }
    return rebuilt_baseline, rebuilt


def rebuild_gates(payload: dict[str, Any], oracle: dict[str, Any], downstream: dict[str, Any]) -> dict[str, bool]:
    traffic = payload["traffic"]
    controls = payload["controls"]
    gates: dict[str, bool] = {
        "traffic_reduction_at_least_4x": traffic["critical_bytes_reduction_x"] >= 4.0,
        "worst_case_new_load_reduction_at_least_8x": traffic["worst_case_new_load_reduction_x"] >= 8.0,
        "all_natural_routes_match_capture": controls["route_mismatch_count"] == 0,
        "manual_sentinel_natural_bitexact": controls["manual_natural_mismatch_values"] == 0,
        "all_finite": bool(controls["all_finite"]),
    }
    for sentinel in SENTINELS:
        key = str(sentinel)
        local = oracle[key]
        quality = downstream[key]["aggregate"]
        prefix = f"layer_{sentinel}"
        gates[f"{prefix}_mean_relative_l2_le_0_05"] = local["relative_l2"]["mean"] <= 0.05
        gates[f"{prefix}_p95_relative_l2_le_0_10"] = local["relative_l2"]["p95"] <= 0.10
        gates[f"{prefix}_mean_kl_le_0_001"] = quality["natural_to_candidate_kl"]["mean"] <= 0.001
        gates[f"{prefix}_relative_ce_le_0_01"] = quality["relative_cross_entropy_regression"] <= 0.01
        gates[f"{prefix}_top1_ge_0_99"] = quality["top1_agreement"] >= 0.99
        gates[f"{prefix}_every_domain_relative_ce_le_0_02"] = all(
            value <= 0.02 for value in quality["per_domain_relative_cross_entropy"].values()
        )
        gates[f"{prefix}_simplex_and_kkt"] = (
            local["minimum_coefficient"]["min"] >= -NEGATIVE_TOL
            and local["simplex_sum_error"]["max"] <= 1e-9
            and local["kkt_violation"]["max"] <= 1e-7
        )
    gates["overall_pass"] = all(gates.values())
    return gates


def main() -> None:
    payload = json.loads(VALIDATION.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    locked = {
        "preregistration_sha256": digest(PREREG),
        "runner_sha256": digest(RUNNER),
        "input_ids_sha256": digest(INPUT_IDS),
        "input_lock_sha256": digest(INPUT_LOCK),
        "capture_sha256": digest(CAPTURE),
        "model_index_sha256": digest(MODEL_INDEX),
        "f0_reference_sha256": digest(F0_REFERENCE),
    }
    add(checks, "all_locked_input_hashes", all(payload["inputs"][key] == value for key, value in locked.items()), locked)
    add(checks, "canonical_input_hash", locked["input_ids_sha256"] == EXPECTED_INPUT, locked["input_ids_sha256"])
    add(checks, "canonical_capture_hash", locked["capture_sha256"] == EXPECTED_CAPTURE, locked["capture_sha256"])
    add(checks, "artifact_audit_passed", payload["artifact_audit"]["pass"] and payload["artifact_audit"]["hash_contract_pass"])
    add(checks, "strict_validation_partition", payload["partition"] == [512, 768])
    add(checks, "sentinel_contract", payload["sentinel_layers"] == list(SENTINELS))

    raw_shape_errors: list[str] = []
    raw_value_errors: list[str] = []
    for sentinel in SENTINELS:
        key = str(sentinel)
        for domain in DOMAINS:
            row = payload["oracle_raw"][key][domain]
            for metric in ("relative_l2", "kkt_violation", "simplex_sum_error", "minimum_coefficient", "support_size", "route_overlap"):
                if len(row[metric]) != 256:
                    raw_shape_errors.append(f"{key}:{domain}:{metric}")
                if not np.isfinite(np.asarray(row[metric], dtype=np.float64)).all():
                    raw_value_errors.append(f"{key}:{domain}:{metric}:nonfinite")
            persistent = np.asarray(row["persistent_ids"], dtype=np.int64)
            alpha = np.asarray(row["alpha"], dtype=np.float64)
            if persistent.shape != (256, 8) or alpha.shape != (256, 8):
                raw_shape_errors.append(f"{key}:{domain}:persistent_or_alpha")
                continue
            if np.any(persistent < 0) or np.any(persistent >= 128) or any(len(set(values)) != 8 for values in persistent.tolist()):
                raw_value_errors.append(f"{key}:{domain}:persistent_ids")
            if not np.isfinite(alpha).all() or alpha.min() < -NEGATIVE_TOL or np.max(np.abs(alpha.sum(axis=1) - 1.0)) > 1e-9:
                raw_value_errors.append(f"{key}:{domain}:simplex")
            if not np.array_equal(np.count_nonzero(alpha > 1e-9, axis=1), np.asarray(row["support_size"])):
                raw_value_errors.append(f"{key}:{domain}:support")
            down = payload["downstream"][key]["domains"][domain]
            if any(len(down[name]) != 255 for name in ("candidate_cross_entropy", "natural_to_candidate_kl", "top1_agreement")):
                raw_shape_errors.append(f"{key}:{domain}:downstream")
            if any(value not in (0, 1) for value in down["top1_agreement"]):
                raw_value_errors.append(f"{key}:{domain}:top1")
    add(checks, "raw_shapes_match_preregistration", not raw_shape_errors, raw_shape_errors)
    add(checks, "raw_values_finite_and_feasible", not raw_value_errors, raw_value_errors)

    oracle = rebuild_oracle(payload["oracle_raw"])
    add(checks, "oracle_summaries_recomputed", close(oracle, payload["oracle_summary"]), oracle)

    baseline, downstream = rebuild_downstream(payload)
    baseline_ok = math.isclose(
        baseline["aggregate_cross_entropy"], payload["baseline"]["aggregate_cross_entropy"], rel_tol=0.0, abs_tol=ABS_TOL
    ) and all(
        len(payload["baseline"]["domains"][domain]["cross_entropy"]) == 255
        and math.isclose(
            baseline["domains"][domain]["mean_cross_entropy"],
            payload["baseline"]["domains"][domain]["mean_cross_entropy"],
            rel_tol=0.0,
            abs_tol=ABS_TOL,
        )
        for domain in DOMAINS
    )
    add(checks, "baseline_statistics_recomputed", baseline_ok, baseline)

    downstream_ok = True
    for sentinel in SENTINELS:
        key = str(sentinel)
        published = payload["downstream"][key]
        downstream_ok &= close(downstream[key]["aggregate"], published["aggregate"])
        for domain in DOMAINS:
            for metric in (
                "mean_candidate_cross_entropy",
                "mean_kl",
                "top1_agreement_rate",
                "relative_cross_entropy_regression",
            ):
                downstream_ok &= close(downstream[key]["domains"][domain][metric], published["domains"][domain][metric])
    add(checks, "downstream_statistics_recomputed", downstream_ok, downstream)

    f0 = json.loads(F0_REFERENCE.read_text(encoding="utf-8"))["candidates"]["1"]["metrics"]["aggregate"]
    baseline_loads = int(round(f0["baseline_new_loads"]["mean"] * f0["transitions"]))
    oracle_loads = int(round(f0["oracle_new_loads"]["mean"] * f0["transitions"]))
    traffic = {
        "transitions": f0["transitions"],
        "baseline_new_loads": baseline_loads,
        "oracle_new_loads": oracle_loads,
        "critical_bytes_reduction_x": baseline_loads / oracle_loads,
        "worst_case_new_load_reduction_x": f0["worst_case_new_load_reduction_x"],
        "mean_route_overlap": f0["mean_route_set_overlap"],
        "substitution_rate": f0["router_output_substitution_rate"],
    }
    add(checks, "traffic_recomputed_from_locked_f0", close(traffic, payload["traffic"]), traffic)

    gates = rebuild_gates(payload, oracle, downstream)
    add(checks, "all_gate_decisions_recomputed", gates == payload["gates"], gates)
    add(checks, "validation_failed", not gates["overall_pass"] and payload["status"] == "validation_negative_test_closed")
    add(checks, "test_partition_remained_closed", payload["test_opened"] is False and not TEST.exists(), str(TEST))
    add(checks, "no_training_or_download", payload["training_or_download"] is False)

    result = {
        "kind": "tierflow_persistent_set_functional_span_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_validation_sha256": digest(VALIDATION),
        "checks": checks,
        "passed": all(check["pass"] for check in checks),
        "recomputed_gates": gates,
        "recomputed_oracle_summary": oracle,
        "recomputed_downstream": downstream,
        "claim_boundary": "Arithmetic and provenance verification of the frozen raw validation artifact; no model rerun.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# TierFlow persistent-set functional-span — independent verification",
        "",
        f"- Status: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- Source validation SHA-256: `{result['source_validation_sha256']}`",
        f"- Checks passed: **{sum(check['pass'] for check in checks)}/{len(checks)}**",
        f"- Recomputed validation gate: **{'PASS' if gates['overall_pass'] else 'FAIL'}**",
        f"- Test artifact exists: **{TEST.exists()}**",
        "",
        "## Recomputed sentinel metrics",
        "",
        "| layer | mean rel-L2 | p95 rel-L2 | mean KL | relative CE | top-1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for sentinel in SENTINELS:
        key = str(sentinel)
        local = oracle[key]
        quality = downstream[key]["aggregate"]
        lines.append(
            f"| {sentinel} | {local['relative_l2']['mean']:.6f} | {local['relative_l2']['p95']:.6f} | "
            f"{quality['natural_to_candidate_kl']['mean']:.6f} | "
            f"{quality['relative_cross_entropy_regression']:.3%} | {quality['top1_agreement']:.3%} |"
        )
    lines.extend([
        "",
        "## Check ledger",
        "",
        "| check | pass |",
        "|---|:---:|",
        *[f"| {check['name']} | {'yes' if check['pass'] else 'no'} |" for check in checks],
        "",
        "The verifier does not rerun the model. It independently recomputes every published statistic and hard gate from the frozen per-token arrays, verifies the locked local artefacts, and confirms that validation failure prevented test access.",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": f"{sum(c['pass'] for c in checks)}/{len(checks)}", "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
