from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT


R = ROOT / "reports/streamq5_moe"
OUTPUT = R / "p7_ervf_independent_verification.json"
REPORT = R / "P7_ERVF_INDEPENDENT_VERIFICATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str):
    return json.loads((R / name).read_text(encoding="utf-8"))


def semantic_equal(left: dict, right: dict) -> bool:
    if left["aggregate"]["next_token_cross_entropy"] != right["aggregate"]["next_token_cross_entropy"]:
        return False
    if left["aggregate"]["predictions_sha256"] != right["aggregate"]["predictions_sha256"]:
        return False
    if left["aggregate"]["misses"] != right["aggregate"]["misses"]:
        return False
    if left["kv_digests"] != right["kv_digests"]:
        return False
    for domain in left["per_domain"]:
        a, b = left["per_domain"][domain], right["per_domain"][domain]
        if a["next_token_cross_entropy"] != b["next_token_cross_entropy"]:
            return False
        if a["predictions"] != b["predictions"] or a["misses"] != b["misses"]:
            return False
    return True


def main() -> None:
    p7a = load("p7a_kernel_roofline.json")
    p7b = load("p7b_ervf_kernel.json")
    p7c_val = load("p7c_ervf_end_to_end_validation.json")
    p7c_test = load("p7c_ervf_end_to_end_test.json")
    p6b_val = load("p6b_strict_end_to_end_validation.json")
    p6b_test = load("p6b_strict_end_to_end_test.json")
    p7d_base = load("p7d_exact_ce_baseline.json")
    p7d_ervf = load("p7d_exact_ce_ervf.json")
    p7c_input = load("p7c_ervf_end_to_end_input_lock.json")
    p7c_eval = load("p7c_ervf_end_to_end_evaluator_lock.json")

    gates = {
        "p7a_prereg_hash": sha256(R / "P7A_KERNEL_ROOFLINE_PREREGISTRATION.md") == p7a["preregistration_sha256"],
        "p7a_script_hash": sha256(ROOT / "scripts/streamq5_moe/run_p7a_kernel_roofline.py") == p7a["script_sha256"],
        "p7a_q8_geometry_diagnosis": p7a["diagnosis"]["q8"]["classification"] == "row_geometry_reduction_or_launch_dominant",
        "p7a_q5_geometry_diagnosis": p7a["diagnosis"]["q5"]["classification"] == "row_geometry_reduction_or_launch_dominant",
        "p7b_prereg_hash": sha256(R / "P7B_ERVF_KERNEL_PREREGISTRATION.md") == p7b["preregistration_sha256"],
        "p7b_script_hash": sha256(ROOT / "scripts/streamq5_moe/run_p7b_ervf_kernel.py") == p7b["script_sha256"],
        "p7b_all_variants_bit_exact": all(p7b["correctness"][bank][str(width)]["bitwise_equal"] for bank in ("q8", "q5") for width in (8, 16, 32)),
        "p7b_selected_16": p7b["selected"] == {"q8": 16, "q5": 16},
        "p7b_isolated_pass": bool(p7b["overall_pass"]),
        "p7c_input_prereg_hash": sha256(R / "P7C_ERVF_END_TO_END_PREREGISTRATION.md") == p7c_input["preregistration_sha256"],
        "p7c_input_base_hash": sha256(R / "p6a_end_to_end_input_lock.json") == p7c_input["base_input_lock_sha256"],
        "p7c_input_p7b_hash": sha256(R / "p7b_ervf_kernel.json") == p7c_input["p7b_result_sha256"],
        "p7c_evaluator_script_hash": sha256(ROOT / "scripts/streamq5_moe/run_p7c_ervf_end_to_end.py") == p7c_eval["evaluator_sha256"],
        "p7c_evaluator_input_hash": sha256(R / "p7c_ervf_end_to_end_input_lock.json") == p7c_eval["input_lock_sha256"],
        "p7c_validation_status": p7c_val["status"] == "p7c_ervf_validation_pass_test_authorized",
        "p7c_test_status": p7c_test["status"] == "p7c_ervf_end_to_end_pass",
        "p7c_all_embedded_gates": all(p7c_val["gates"].values()) and all(p7c_test["gates"].values()),
        "p7c_validation_semantics_equal": semantic_equal(p7c_val["quality"], p6b_val["quality"]),
        "p7c_test_semantics_equal": semantic_equal(p7c_test["quality"], p6b_test["quality"]),
        "p7c_rollout_tokens_equal": p7c_test["rollout"]["generated_ids"] == p6b_test["rollout"]["generated_ids"],
        "p7c_rollout_feedback_equal": p7c_test["rollout"]["feedback_ids"] == p6b_test["rollout"]["feedback_ids"],
        "p7c_rollout_kv_equal": p7c_test["rollout"]["kv_digest"] == p6b_test["rollout"]["kv_digest"],
        "p7c_physical_bytes_equal": all(p7c_test["physical"][key] == p6b_test["physical"][key] for key in ("expert_cache_bytes", "trunk_device_bytes", "embedding_host_bytes", "kv_bytes", "expert_bank_pinned_bytes", "q8_bank_pinned_bytes")),
    }
    for variant, payload in (("baseline", p7d_base), ("ervf", p7d_ervf)):
        gates[f"p7d_{variant}_prereg_hash"] = sha256(R / "P7D_EXACT_CE_REPAIR_PREREGISTRATION.md") == payload["preregistration_sha256"]
        gates[f"p7d_{variant}_script_hash"] = sha256(ROOT / "scripts/streamq5_moe/run_p7d_exact_ce_repair.py") == payload["script_sha256"]
        gates[f"p7d_{variant}_source_hash"] = sha256(ROOT / "scripts/streamq5_moe/run_p6a_end_to_end_decode.py") == payload["source_sha256"]
        gates[f"p7d_{variant}_input_hash"] = sha256(R / "p6a_end_to_end_input_lock.json") == payload["base_input_lock_sha256"]
    for split in ("validation", "test"):
        a, b = p7d_base["phases"][split], p7d_ervf["phases"][split]
        gates[f"p7d_{split}_labels"] = a["aggregate"]["labels"] == b["aggregate"]["labels"] == 1270
        gates[f"p7d_{split}_all_ce_exact"] = a["aggregate"]["cross_entropies"] == b["aggregate"]["cross_entropies"]
        gates[f"p7d_{split}_predictions_exact"] = a["aggregate"]["predictions_sha256"] == b["aggregate"]["predictions_sha256"]
        gates[f"p7d_{split}_misses_exact"] = a["aggregate"]["misses"] == b["aggregate"]["misses"]
        gates[f"p7d_{split}_kv_exact"] = a["kv_digests"] == b["kv_digests"]
        gates[f"p7d_{split}_finite"] = bool(a["aggregate"]["finite"] and b["aggregate"]["finite"])

    ratios = {
        "validation_mean": p7c_val["quality"]["aggregate"]["wall_ms_stats"]["mean"] / p6b_val["quality"]["aggregate"]["wall_ms_stats"]["mean"],
        "validation_p95": p7c_val["quality"]["aggregate"]["wall_ms_stats"]["p95"] / p6b_val["quality"]["aggregate"]["wall_ms_stats"]["p95"],
        "test_mean": p7c_test["quality"]["aggregate"]["wall_ms_stats"]["mean"] / p6b_test["quality"]["aggregate"]["wall_ms_stats"]["mean"],
        "test_p95": p7c_test["quality"]["aggregate"]["wall_ms_stats"]["p95"] / p6b_test["quality"]["aggregate"]["wall_ms_stats"]["p95"],
        "rollout_mean": p7c_test["rollout"]["wall_ms_stats"]["mean"] / p6b_test["rollout"]["wall_ms_stats"]["mean"],
        "rollout_p95": p7c_test["rollout"]["wall_ms_stats"]["p95"] / p6b_test["rollout"]["wall_ms_stats"]["p95"],
    }
    gates["p7c_test_mean_ratio_le_0_80"] = ratios["test_mean"] <= 0.80
    gates["p7c_test_p95_ratio_le_0_80"] = ratios["test_p95"] <= 0.80
    gates["p7c_rollout_mean_ratio_le_0_80"] = ratios["rollout_mean"] <= 0.80
    gates["p7c_rollout_p95_ratio_le_0_80"] = ratios["rollout_p95"] <= 0.80
    gates["all_numbers_finite"] = all(math.isfinite(value) for value in ratios.values())
    passed = all(gates.values())
    metrics = {
        "p7a": {bank: p7a["diagnosis"][bank] for bank in ("q8", "q5")},
        "p7b": {bank: {"baseline_p50_ms": p7b["test"][bank]["baseline"]["stats"]["p50"], "ervf_p50_ms": p7b["test"][bank]["ervf"]["stats"]["p50"], "speedup_p50": p7b["test"][bank]["speedup_p50"]} for bank in ("q8", "q5")},
        "p7c": {
            "test_baseline_mean_ms": p6b_test["quality"]["aggregate"]["wall_ms_stats"]["mean"],
            "test_ervf_mean_ms": p7c_test["quality"]["aggregate"]["wall_ms_stats"]["mean"],
            "test_baseline_tps": 1000.0 / p6b_test["quality"]["aggregate"]["wall_ms_stats"]["mean"],
            "test_ervf_tps": 1000.0 / p7c_test["quality"]["aggregate"]["wall_ms_stats"]["mean"],
            "rollout_baseline_mean_ms": p6b_test["rollout"]["wall_ms_stats"]["mean"],
            "rollout_ervf_mean_ms": p7c_test["rollout"]["wall_ms_stats"]["mean"],
            "rollout_baseline_tps": p6b_test["rollout"]["tokens_per_second"],
            "rollout_ervf_tps": p7c_test["rollout"]["tokens_per_second"],
            "ce": p7c_test["quality"]["aggregate"]["next_token_cross_entropy"],
            "teacher_ce": p7c_test["quality"]["aggregate"]["teacher_cross_entropy"],
            "relative_ce_increase": p7c_test["quality"]["aggregate"]["relative_cross_entropy_increase"],
        },
        "ratios": ratios,
    }
    result = {"kind": "streamq5_moe_p7_ervf_independent_verification", "completed_utc": datetime.now(timezone.utc).isoformat(), "verifier_sha256": sha256(Path(__file__)), "gates": gates, "metrics": metrics, "pass": passed, "claim_boundary": "Verified local, bit-exact P6B-to-P7C acceleration on one GPU/model/runtime. No cross-runtime or world-SOTA claim."}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    status = "PASS" if passed else "FAIL"
    REPORT.write_text(
        f"# P7 ERVF onafhankelijke verificatie — {status}\n\n"
        f"Alle {len(gates)} poorten: **{status}**. P7D bevestigt 0 verschillende CE-waarden over 1.270 validation- en 1.270 testlabels.\n\n"
        f"Test: {metrics['p7c']['test_baseline_mean_ms']:.3f} → {metrics['p7c']['test_ervf_mean_ms']:.3f} ms, "
        f"{metrics['p7c']['test_baseline_tps']:.3f} → {metrics['p7c']['test_ervf_tps']:.3f} tok/s.\n\n"
        f"512-tokenrollout: {metrics['p7c']['rollout_baseline_mean_ms']:.3f} → {metrics['p7c']['rollout_ervf_mean_ms']:.3f} ms, "
        f"{metrics['p7c']['rollout_baseline_tps']:.3f} → {metrics['p7c']['rollout_ervf_tps']:.3f} tok/s.\n",
        encoding="utf-8",
    )
    print(json.dumps({"pass": passed, "gates": len(gates), "failed": [name for name, value in gates.items() if not value], "metrics": metrics["p7c"], "ratios": ratios}, indent=2))


if __name__ == "__main__":
    main()
