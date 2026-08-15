#!/usr/bin/env python3
"""Independent evidence verifier for the S0-R5 + C1-R2A adjudication."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe"
ADJ = R / "PORT80B_T0Q5_S0R5_C1R2A_COMBINED_ADJUDICATION_2026-08-13.json"
OUT = R / "port80b_t0q5_s0r5_c1r2a_combined_independent_verification.json"
R5D = RUNS / "port80b_t0q5s0r5_selected_route_validation"
C1D = RUNS / "port80b_t0q5s0c1r2a_control_only"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_words(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.uint16) != b.contiguous().view(torch.uint16)).sum())


def valid_commit(path: Path, directory: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    return all(
        (directory / name).is_file()
        and row["bytes"] == (directory / name).stat().st_size
        and row["sha256"] == sha(directory / name)
        for name, row in data["files"].items()
    )


def main() -> int:
    a = json.loads(ADJ.read_text(encoding="utf-8"))
    r5 = json.loads((R5D / "s0r5_result.json").read_text(encoding="utf-8"))
    c1 = json.loads((C1D / "s0c1r2a_result.json").read_text(encoding="utf-8"))

    actual_bindings = {}
    bindings_ok = True
    for name, expected in a["bindings"].items():
        path = ROOT / expected["path"]
        actual = {"path": expected["path"], "bytes": path.stat().st_size, "sha256": sha(path)}
        actual_bindings[name] = actual
        bindings_ok &= actual == expected

    rows = [
        (int(prompt), arm, position, float(metric["rel_l2"]))
        for prompt, arms in r5["metrics"].items()
        for arm, metrics in arms.items()
        for position, metric in enumerate(metrics, 8)
    ]
    arm_stats = {
        arm: (len([x for x in rows if x[1] == arm]), max(x[3] for x in rows if x[1] == arm))
        for arm in ("routed", "shared_raw", "shared_gated")
    }

    with safe_open(R5D / "s0r5_raw.safetensors", framework="pt", device="cpu") as raw:
        n8_raw = changed_words(raw.get_tensor("p0_q5_shared_raw")[8:9], raw.get_tensor("p0_n8_mutation_shared_raw"))
        n8_gated = changed_words(raw.get_tensor("p0_q5_shared_gated")[8:9], raw.get_tensor("p0_n8_mutation_shared_gated"))
        n15_raw = changed_words(raw.get_tensor("p0_q5_shared_raw")[15:16], raw.get_tensor("p0_n15_mutation_shared_raw"))
        n15_gated = changed_words(raw.get_tensor("p0_q5_shared_gated")[15:16], raw.get_tensor("p0_n15_mutation_shared_gated"))
    with safe_open(C1D / "s0c1r2a_raw.safetensors", framework="pt", device="cpu") as raw:
        original = raw.get_tensor("original_output")
        mutated = raw.get_tensor("mutated_output")
        c1_delta = changed_words(original, mutated)
        words = [int(original.view(torch.uint16)[0, 0]), int(mutated.view(torch.uint16)[0, 0])]

    checks = {
        "all_bound_artifacts_exact": bindings_ok,
        "r5_commit_exact": valid_commit(R5D / "s0r5_commit.json", R5D),
        "c1_commit_exact": valid_commit(C1D / "s0c1r2a_commit.json", C1D),
        "result_raw_hashes_exact": r5["raw_sha256"] == sha(R5D / "s0r5_raw.safetensors") and c1["raw_sha256"] == sha(C1D / "s0c1r2a_raw.safetensors"),
        "metric_cardinality_3x32": len(rows) == 96 and all(count == 32 for count, _ in arm_stats.values()),
        "all_quality_metrics_finite_and_below_frozen_threshold": all(math.isfinite(value) and value <= 0.08 for *_, value in rows),
        "quality_summary_exact": all(a["r5_numerical_quality_arm"]["by_arm"][arm]["count"] == count and a["r5_numerical_quality_arm"]["by_arm"][arm]["max_rel_l2"] == maximum for arm, (count, maximum) in arm_stats.items()),
        "r5_natural_p0n8_zero_word_change": n8_raw == n8_gated == 0 and a["r5_integrity_control"]["natural_p0n8_changed_bf16_words"] == {"shared_raw": 0, "shared_gated": 0},
        "r5_neighbor_p0n15_one_word_change": n15_raw == n15_gated == 1,
        "r5_formal_status_preserved_negative": a["formal_overall_status"] == "negative" and a["r5_integrity_control"]["status"] == "formal_negative" and a["r5_integrity_control"]["completed_verifier_summary"] == "12/13 checks true; controls=false",
        "c1_exact_single_word_control": c1_delta == 1 and words == [14520, 14489] and c1["record"]["changed_field_count"] == 1 and c1["record"]["bf16_word_xor"] == (14520 ^ 14489),
        "c1_safe_rejection_order": c1["safe_rejection_errors"] == ["codes_scales_digest"] and [x["event"] for x in c1["ledger"]] == ["safe_checker_rejected", "unsafe_decode", "unsafe_linear"],
        "c1_11_of_11_summary_preserved": a["c1_synthetic_integrity_control"]["status"] == "positive" and a["c1_synthetic_integrity_control"]["completed_verifier_summary"] == "11/11 checks true",
        "claim_boundary_exact": a["claim_boundary"] == "Validation-only and non-heldout. No full layer, full model, generation, performance, device, industrial, novelty or breakthrough claim. C1 does not retroactively convert R5 into a formal pass.",
        "no_physical_action_claim": a["physical_actions"] == {"checkpoint_opened": False, "model_run": False, "gpu": False, "device_probe": False},
    }
    result = {
        "kind": "port80b_t0q5_s0r5_c1r2a_combined_independent_verification",
        "pass": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "adjudication_sha256": sha(ADJ),
        "actual_bindings": actual_bindings,
        "formal_overall_status": "negative",
        "positive_subresults": ["R5 numerical-quality arm: 96/96", "C1-R2A synthetic integrity control: 11/11"],
        "claim_boundary": "Evidence-only verification of immutable validation artifacts; no checkpoint/model/GPU/device rerun and no full-layer, model, performance, industrial, novelty or breakthrough claim.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
