from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
PREREG = R / "N2D_GREEDY_LM_HEAD_WRITE_ELISION_PREREGISTRATION.md"
EVALUATOR = ROOT / "scripts/streamq5_moe/run_n2d_greedy_lm_head_write_elision.py"
COMPILE = R / "n2d_greedy_lm_head_compile.json"
VALIDATION = R / "n2d_greedy_lm_head_validation.json"
TEST = R / "n2d_greedy_lm_head_test.json"
OUTPUT = R / "n2d_greedy_lm_head_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    compile_result = json.loads(COMPILE.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    rows = validation["correctness"]
    ratios = validation["ratios"]
    checks = {
        "compile_prereg_hash_current": compile_result["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "compile_evaluator_hash_current": compile_result["inputs"]["script_sha256"] == sha256(EVALUATOR),
        "validation_prereg_hash_current": validation["inputs"]["preregistration_sha256"] == sha256(PREREG),
        "validation_evaluator_hash_current": validation["inputs"]["script_sha256"] == sha256(EVALUATOR),
        "validation_compile_hash_current": validation["inputs"]["compile_result_sha256"] == sha256(COMPILE),
        "compile_launched_no_kernel": compile_result["gpu_kernel_launched"] is False,
        "physical_head_bytes_exact": validation["physical"]["head_resident_bytes"] == 316_026_880,
        "correctness_rows_17": len(rows) == 17,
        "all_indices_exact": all(row["all_indices_exact"] for row in rows),
        "all_values_exact": all(row["all_values_exact"] for row in rows),
        "all_values_finite": all(row["finite"] for row in rows),
        "all_zero_tie_exact": rows[0]["input"] == "all_zero_tie" and rows[0]["path_c_argmax"] == 0,
        "output_bytes_exact": validation["output_bytes"] == {
            "a_current": 607_756,
            "b_full_argmax": 607_752,
            "c_fused_candidates": 75_976,
            "c_bytes_saved_vs_a": 531_780,
            "c_fraction_eliminated_vs_a": 531_780 / 607_756,
        },
        "candidate_validation_gate_failed": not validation["overall_pass"],
        "candidate_p50_slower": ratios["c_over_a_p50"] > 1.0 and ratios["c_over_b_p50"] > 1.0,
        "candidate_p95_slower": ratios["c_over_a_p95"] > 1.0 and ratios["c_over_b_p95"] > 1.0,
        "test_not_authorized": validation["test_authorized"] is False,
        "test_partition_remained_sealed": not TEST.exists(),
    }
    result = {
        "kind": "streamq5_moe_n2d_independent_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "evaluator_sha256": sha256(EVALUATOR),
            "compile_sha256": sha256(COMPILE),
            "validation_sha256": sha256(VALIDATION),
        },
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "overall_pass": all(checks.values()),
        "verdict": "verified_exact_but_performance_negative_test_sealed" if all(checks.values()) else "audit_failed",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
