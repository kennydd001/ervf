from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = R / "P6B_STRICT_END_TO_END_REPLICATION_PREREGISTRATION.md"
INPUT_LOCK = R / "p6b_strict_end_to_end_input_lock.json"
EVALUATOR_LOCK = R / "p6b_strict_end_to_end_evaluator_lock.json"
SOURCE = ROOT / "scripts/streamq5_moe/run_p6b_strict_end_to_end_decode.py"
P6A_SOURCE = ROOT / "scripts/streamq5_moe/run_p6a_end_to_end_decode.py"
SMOKE = R / "p6b_strict_end_to_end_smoke.json"
VALIDATION = R / "p6b_strict_end_to_end_validation.json"
TEST = R / "p6b_strict_end_to_end_test.json"
P6A_VALIDATION = R / "p6a_end_to_end_validation.json"
P6A_TEST = R / "p6a_end_to_end_test.json"
P0C_VALIDATION = R / "p0c_validation_model_quality.json"
P0C_TEST = R / "p0c_test_model_quality.json"
BANK_VERIFY = R / "p6a_exact_runtime_bank_verification.json"
OUTPUT = R / "p6b_end_to_end_verification.json"
REPORT = R / "P6B_END_TO_END_VERIFICATION.md"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
EXPERT_BYTES = 3_035_136
LAYERS = 48


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def stats_equal(observed, values, tolerance=1e-12):
    expected = statistics(values)
    return all(math.isclose(float(observed[key]), expected[key], rel_tol=tolerance, abs_tol=tolerance) for key in expected)


def verify_quality(result, teacher_path):
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    quality = result["quality"]
    aggregate = quality["aggregate"]
    predictions = []
    weighted_ce = 0.0
    labels = 0
    checks = {}
    for domain in DOMAINS:
        row = quality["per_domain"][domain]
        teacher_ce = teacher["variants"]["bf16_teacher"]["domains"][domain]["next_token_cross_entropy"]
        checks[f"{domain}_labels_254"] = row["labels"] == 254
        checks[f"{domain}_teacher_exact"] = row["teacher_cross_entropy"] == teacher_ce
        checks[f"{domain}_timing_stats"] = stats_equal(row["wall_ms_stats"], row["wall_ms"])
        checks[f"{domain}_miss_stats"] = stats_equal(row["miss_stats"], row["misses"])
        prediction_array = np.asarray(row["predictions"], dtype=np.int32)
        checks[f"{domain}_prediction_hash"] = array_sha(prediction_array) == row["prediction_sha256"]
        predictions.extend(row["predictions"])
        weighted_ce += row["next_token_cross_entropy"] * row["labels"]
        labels += row["labels"]
    calculated_ce = weighted_ce / labels
    teacher_ce = teacher["variants"]["bf16_teacher"]["next_token_cross_entropy"]
    checks.update({
        "aggregate_labels": labels == aggregate["labels"] == 1270,
        "aggregate_ce_weighted": math.isclose(calculated_ce, aggregate["next_token_cross_entropy"], rel_tol=1e-14, abs_tol=1e-14),
        "aggregate_teacher_exact": aggregate["teacher_cross_entropy"] == teacher_ce,
        "aggregate_relative_ce": math.isclose((calculated_ce - teacher_ce) / teacher_ce, aggregate["relative_cross_entropy_increase"], rel_tol=1e-14, abs_tol=1e-14),
        "aggregate_timing_stats": stats_equal(aggregate["wall_ms_stats"], aggregate["wall_ms"]),
        "aggregate_miss_stats": stats_equal(aggregate["miss_stats"], aggregate["misses"]),
        "aggregate_prediction_hash": array_sha(np.asarray(predictions, dtype=np.int32)) == aggregate["predictions_sha256"],
        "aggregate_relative_ce_gate": aggregate["relative_cross_entropy_increase"] <= 0.02,
        "aggregate_mean_gate": aggregate["wall_ms_stats"]["mean"] <= 100.0,
        "aggregate_p95_gate": aggregate["wall_ms_stats"]["p95"] <= 150.0,
        "finite_quality": np.isfinite(aggregate["wall_ms"]).all() and np.isfinite(calculated_ce),
        "kv_digests_10": len(quality["kv_digests"]) == 10,
        "kv_context_127": all(row["context"] == 127 for row in quality["kv_digests"]),
        "kv_nonzero_majority": all(row["nonzero"] > row["elements"] // 2 for row in quality["kv_digests"]),
    })
    return checks


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P6B verification")
    input_lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    evaluator_lock = json.loads(EVALUATOR_LOCK.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    test = json.loads(TEST.read_text(encoding="utf-8"))
    p6a_validation = json.loads(P6A_VALIDATION.read_text(encoding="utf-8"))
    p6a_test = json.loads(P6A_TEST.read_text(encoding="utf-8"))
    bank_verify = json.loads(BANK_VERIFY.read_text(encoding="utf-8"))
    rollout = test["rollout"]
    generated = np.asarray(rollout["generated_ids"], dtype=np.int32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    provenance = {
        "preregistration_hash": sha256(PREREG) == input_lock["preregistration_sha256"],
        "base_input_lock_hash": sha256(R / "p6a_end_to_end_input_lock.json") == input_lock["base_input_lock_sha256"],
        "base_source_hash": sha256(P6A_SOURCE) == input_lock["base_evaluator_source_sha256"],
        "input_lock_hash": sha256(INPUT_LOCK) == evaluator_lock["input_lock_sha256"],
        "evaluator_hash": sha256(SOURCE) == evaluator_lock["evaluator_sha256"],
        "bank_verification_pass": bank_verify["status"] == "p6a_exact_runtime_bank_verification_pass",
        "smoke_status": smoke["status"] == "p6b_smoke_pass",
        "validation_status": validation["status"] == "p6b_validation_pass_test_authorized",
        "test_status": test["status"] == "p6b_strict_end_to_end_eureka_pass",
        "all_phase_gates": all(smoke["gates"].values()) and all(validation["gates"].values()) and all(test["gates"].values()),
        "source_declares_timer_before_embedding": '"        wall_start = time.perf_counter_ns()\\n        state_host = self.embedding(int(token))"' in SOURCE.read_text(encoding="utf-8"),
    }
    validation_checks = verify_quality(validation, P0C_VALIDATION)
    test_checks = verify_quality(test, P0C_TEST)
    replication = {
        "validation_ce_exact": validation["quality"]["aggregate"]["next_token_cross_entropy"] == p6a_validation["quality"]["aggregate"]["next_token_cross_entropy"],
        "validation_prediction_hash_exact": validation["quality"]["aggregate"]["predictions_sha256"] == p6a_validation["quality"]["aggregate"]["predictions_sha256"],
        "validation_misses_exact": validation["quality"]["aggregate"]["misses"] == p6a_validation["quality"]["aggregate"]["misses"],
        "test_ce_exact": test["quality"]["aggregate"]["next_token_cross_entropy"] == p6a_test["quality"]["aggregate"]["next_token_cross_entropy"],
        "test_prediction_hash_exact": test["quality"]["aggregate"]["predictions_sha256"] == p6a_test["quality"]["aggregate"]["predictions_sha256"],
        "test_misses_exact": test["quality"]["aggregate"]["misses"] == p6a_test["quality"]["aggregate"]["misses"],
        "rollout_ids_exact": rollout["generated_ids"] == p6a_test["rollout"]["generated_ids"],
        "rollout_hash_exact": rollout["generated_ids_sha256"] == p6a_test["rollout"]["generated_ids_sha256"],
    }
    prompt_ids = tokenizer.encode(rollout["prompt"], add_special_tokens=False)
    rollout_checks = {
        "tokens_512": generated.size == 512,
        "generated_hash": array_sha(generated) == rollout["generated_ids_sha256"],
        "prompt_ids_exact": prompt_ids == rollout["prompt_ids"],
        "feedback_exact": rollout["feedback_ids"][1:] == rollout["generated_ids"][:-1],
        "decoded_text_exact": tokenizer.decode(rollout["generated_ids"], skip_special_tokens=False) == rollout["text"],
        "timing_stats": stats_equal(rollout["wall_ms_stats"], rollout["wall_ms"]),
        "miss_stats": stats_equal(rollout["miss_stats"], rollout["misses"]),
        "tokens_per_second": math.isclose(1000.0 / np.mean(rollout["wall_ms"]), rollout["tokens_per_second"], rel_tol=1e-14, abs_tol=1e-14),
        "mean_gate": rollout["wall_ms_stats"]["mean"] <= 100.0,
        "p95_gate": rollout["wall_ms_stats"]["p95"] <= 150.0,
        "finite": np.isfinite(rollout["wall_ms"]).all(),
        "rollout_gates": all(rollout["gates"].values()),
    }
    expected_kv_writes = 1270 * LAYERS + (len(prompt_ids) - 1 + 512) * LAYERS
    physical = test["physical"]
    invariants = {
        "expert_miss_bytes": test["runtime_invariants"]["total_miss_bytes"] == test["runtime_invariants"]["total_misses"] * EXPERT_BYTES,
        "kv_write_count": test["runtime_invariants"]["kv_layer_position_writes"] == expected_kv_writes,
        "router_unique": test["runtime_invariants"]["route_unique_failures"] == 0,
        "router_weight_error": test["runtime_invariants"]["route_weight_sum_abs_error_max"] <= 0.02,
        "expert_cache_bytes": physical["expert_cache_bytes"] == 4_977_623_040,
        "trunk_bytes": physical["trunk_device_bytes"] == 1_248_931_840,
        "embedding_bytes": physical["embedding_host_bytes"] == 316_026_880,
        "kv_bytes": physical["kv_bytes"] == 402_653_184,
        "expert_bank_bytes": physical["expert_bank_pinned_bytes"] == 18_647_875_584,
        "q8_bank_bytes": physical["q8_bank_pinned_bytes"] == 1_564_958_720,
        "scratch_gate": physical["free_after_fixed_bytes"] >= 192 * 2**20,
    }
    groups = {
        "provenance": provenance,
        "validation": validation_checks,
        "test": test_checks,
        "replication": replication,
        "rollout": rollout_checks,
        "invariants": invariants,
    }
    groups = {name: {key: bool(value) for key, value in values.items()} for name, values in groups.items()}
    passed = all(all(values.values()) for values in groups.values())
    check_count = sum(len(values) for values in groups.values())
    payload = {
        "kind": "streamq5_moe_p6b_strict_end_to_end_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p6b_end_to_end_verification_pass" if passed else "p6b_end_to_end_verification_fail",
        "inputs": {"smoke_sha256": sha256(SMOKE), "validation_sha256": sha256(VALIDATION), "test_sha256": sha256(TEST), "input_lock_sha256": sha256(INPUT_LOCK), "evaluator_lock_sha256": sha256(EVALUATOR_LOCK)},
        "groups": groups,
        "checks": check_count,
        "passed_checks": sum(int(value) for values in groups.values() for value in values.values()),
        "decision_metrics": {
            "validation_relative_ce": validation["quality"]["aggregate"]["relative_cross_entropy_increase"],
            "validation_mean_ms": validation["quality"]["aggregate"]["wall_ms_stats"]["mean"],
            "validation_p95_ms": validation["quality"]["aggregate"]["wall_ms_stats"]["p95"],
            "test_relative_ce": test["quality"]["aggregate"]["relative_cross_entropy_increase"],
            "test_mean_ms": test["quality"]["aggregate"]["wall_ms_stats"]["mean"],
            "test_p95_ms": test["quality"]["aggregate"]["wall_ms_stats"]["p95"],
            "rollout_tokens": len(rollout["generated_ids"]),
            "rollout_mean_ms": rollout["wall_ms_stats"]["mean"],
            "rollout_p95_ms": rollout["wall_ms_stats"]["p95"],
            "rollout_tokens_per_second": rollout["tokens_per_second"],
            "rollout_sha256": rollout["generated_ids_sha256"],
        },
        "claim_boundary": test["claim_boundary"],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# P6B strikte end-to-end-decode — onafhankelijke verificatie\n\n"
        f"Status: **{payload['status']}**; {payload['passed_checks']}/{check_count} controles. "
        f"Test-CE-relatief {payload['decision_metrics']['test_relative_ce']:.6%}; "
        f"test mean/p95 {payload['decision_metrics']['test_mean_ms']:.3f}/{payload['decision_metrics']['test_p95_ms']:.3f} ms; "
        f"512-tokenrollout {payload['decision_metrics']['rollout_tokens_per_second']:.3f} tok/s.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "checks": f"{payload['passed_checks']}/{check_count}", "decision_metrics": payload["decision_metrics"], "failed": {name: [key for key, value in values.items() if not value] for name, values in groups.items() if not all(values.values())}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
