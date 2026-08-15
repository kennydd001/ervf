#!/usr/bin/env python3
"""Evidence-only adjudication of immutable S0-R5 and S0-C1-R2A.

This script reads only the two small committed result bundles.  It does not
open the official checkpoint, run a model, or initialize CUDA.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scripts/streamq5_moe"
R = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe"
OUT = R / "PORT80B_T0Q5_S0R5_C1R2A_COMBINED_ADJUDICATION_2026-08-13.json"
REPORT = R / "PORT80B_T0Q5_S0R5_C1R2A_COMBINED_REPORT_2026-08-13.md"

R5D = RUNS / "port80b_t0q5s0r5_selected_route_validation"
C1D = RUNS / "port80b_t0q5s0c1r2a_control_only"

FILES = {
    "r5_raw": R5D / "s0r5_raw.safetensors",
    "r5_result": R5D / "s0r5_result.json",
    "r5_commit": R5D / "s0r5_commit.json",
    "r5_verifier": S / "verify_port80b_t0q5s0r5_selected_route_validation.py",
    "r5_verifier_lock": R / "port80b_t0q5s0r5_verifier_lock.json",
    "r5_runner": S / "run_port80b_t0q5s0r5_selected_route_validation.py",
    "r5_runner_lock": R / "port80b_t0q5s0r5_runner_lock.json",
    "r5_prereg": R / "PORT80B_T0Q5S0R5_SELECTED_ROUTE_VALIDATION_PREREGISTRATION_2026-08-13.md",
    "c1_raw": C1D / "s0c1r2a_raw.safetensors",
    "c1_result": C1D / "s0c1r2a_result.json",
    "c1_commit": C1D / "s0c1r2a_commit.json",
    "c1_verifier": S / "verify_port80b_t0q5s0c1r2a_control_only.py",
    "c1_verifier_lock": R / "port80b_t0q5s0c1r2a_verifier_lock.json",
    "c1_runner": S / "run_port80b_t0q5s0c1r2a_control_only.py",
    "c1_runner_lock": R / "port80b_t0q5s0c1r2a_runner_lock.json",
    "c1_prereg": R / "PORT80B_T0Q5S0C1R2A_CONTROL_ONLY_PREREGISTRATION_2026-08-13.md",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def commit_ok(commit_path: Path, directory: Path) -> bool:
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    return all(
        (directory / name).is_file()
        and row == {"bytes": (directory / name).stat().st_size, "sha256": sha(directory / name)}
        for name, row in commit["files"].items()
    )


def changed_words(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.uint16) != b.contiguous().view(torch.uint16)).sum())


def main() -> int:
    r5 = json.loads(FILES["r5_result"].read_text(encoding="utf-8"))
    c1 = json.loads(FILES["c1_result"].read_text(encoding="utf-8"))
    r5_vlock = json.loads(FILES["r5_verifier_lock"].read_text(encoding="utf-8"))
    c1_vlock = json.loads(FILES["c1_verifier_lock"].read_text(encoding="utf-8"))

    metrics = [
        {"prompt": int(prompt), "arm": arm, "position": position, **row}
        for prompt, arms in r5["metrics"].items()
        for arm, rows in arms.items()
        for position, row in enumerate(rows, start=8)
    ]
    by_arm = {
        arm: {
            "count": sum(row["arm"] == arm for row in metrics),
            "max_rel_l2": max(row["rel_l2"] for row in metrics if row["arm"] == arm),
            "threshold": 0.08,
            "pass": all(row["rel_l2"] <= 0.08 for row in metrics if row["arm"] == arm),
        }
        for arm in ("routed", "shared_raw", "shared_gated")
    }

    with safe_open(FILES["r5_raw"], framework="pt", device="cpu") as raw:
        p0n8_raw = changed_words(raw.get_tensor("p0_q5_shared_raw")[8:9], raw.get_tensor("p0_n8_mutation_shared_raw"))
        p0n8_gated = changed_words(raw.get_tensor("p0_q5_shared_gated")[8:9], raw.get_tensor("p0_n8_mutation_shared_gated"))
        p0n15_raw = changed_words(raw.get_tensor("p0_q5_shared_raw")[15:16], raw.get_tensor("p0_n15_mutation_shared_raw"))
        p0n15_gated = changed_words(raw.get_tensor("p0_q5_shared_gated")[15:16], raw.get_tensor("p0_n15_mutation_shared_gated"))

    with safe_open(FILES["c1_raw"], framework="pt", device="cpu") as raw:
        original = raw.get_tensor("original_output")
        mutated = raw.get_tensor("mutated_output")
        c1_changed = changed_words(original, mutated)
        c1_words = [int(original.view(torch.uint16)[0, 0]), int(mutated.view(torch.uint16)[0, 0])]

    bindings = {name: binding(path) for name, path in FILES.items()}
    evidence_checks = {
        "r5_commit_valid": commit_ok(FILES["r5_commit"], R5D),
        "c1_commit_valid": commit_ok(FILES["c1_commit"], C1D),
        "r5_result_artifact_hashes": r5["raw_sha256"] == sha(FILES["r5_raw"]),
        "c1_result_artifact_hashes": c1["raw_sha256"] == sha(FILES["c1_raw"]),
        "r5_verifier_source_bound": r5["verifier_sha256"] == r5_vlock["verifier_sha256"] == sha(FILES["r5_verifier"]),
        "c1_verifier_source_bound": c1["verifier_sha256"] == c1_vlock["verifier_sha256"] == sha(FILES["c1_verifier"]),
        "r5_metric_layout_3x32": len(metrics) == 96 and all(row["count"] == 32 for row in by_arm.values()),
        "r5_numerical_quality_arm": all(row["pass"] for row in by_arm.values()),
        "r5_natural_p0n8_not_observable": p0n8_raw == p0n8_gated == 0,
        "r5_neighbor_witness_observable": p0n15_raw == p0n15_gated == 1,
        "c1_one_field_metadata": c1["record"]["changed_field_count"] == 1 and c1["record"]["q"] == 6 and c1["record"]["q_prime"] == 5,
        "c1_synthetic_output_observable": c1_changed == 1 and c1_words == [14520, 14489] and c1["record"]["bf16_word_xor"] == (14520 ^ 14489),
        "c1_checker_rejects_before_unsafe": c1["safe_rejection_errors"] == ["codes_scales_digest"] and [x["event"] for x in c1["ledger"]] == ["safe_checker_rejected", "unsafe_decode", "unsafe_linear"],
    }

    outcome = {
        "kind": "port80b_t0q5_s0r5_c1r2a_combined_adjudication",
        "evidence_integrity_pass": all(evidence_checks.values()),
        "formal_overall_status": "negative",
        "formal_reason": "S0-R5 retains its frozen natural p0/n8 observability failure: the real q=6->5 shared-down mutation changes zero BF16 words in both raw and gated outputs.",
        "r5_numerical_quality_arm": {
            "status": "positive",
            "scope": "four frozen validation prompts, token positions 8..15, three output arms",
            "measurements": len(metrics),
            "by_arm": by_arm,
            "overall_max_rel_l2": max(row["rel_l2"] for row in metrics),
        },
        "r5_integrity_control": {
            "status": "formal_negative",
            "completed_verifier_summary": "12/13 checks true; controls=false",
            "natural_p0n8_changed_bf16_words": {"shared_raw": p0n8_raw, "shared_gated": p0n8_gated},
            "natural_p0n15_changed_bf16_words": {"shared_raw": p0n15_raw, "shared_gated": p0n15_gated},
        },
        "c1_synthetic_integrity_control": {
            "status": "positive",
            "completed_verifier_summary": "11/11 checks true",
            "synthetic_changed_bf16_words": c1_changed,
            "selected_output_words": c1_words,
            "scope": "one preregistered real shared-down Q5 record with a deterministic synthetic one-hot activation",
        },
        "evidence_checks": evidence_checks,
        "bindings": bindings,
        "verifier_evidence_boundary": "The completed verifier outcomes are adjudicated, while this evidence-only script independently rechecks committed hashes, metric arithmetic and raw observability. It does not rerun either checkpoint-backed verifier.",
        "claim_boundary": "Validation-only and non-heldout. No full layer, full model, generation, performance, device, industrial, novelty or breakthrough claim. C1 does not retroactively convert R5 into a formal pass.",
        "physical_actions": {"checkpoint_opened": False, "model_run": False, "gpu": False, "device_probe": False},
    }
    OUT.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# PORT80B T0Q5 S0-R5 + C1-R2A combined adjudication

## Outcome

**Formal overall result: negative.** S0-R5 remains verifier-negative because its frozen natural `p0/n8` shared-down `q=6 -> 5` mutation changes **0 BF16 words** in both the raw and gated output. C1-R2A is a separate synthetic sensitivity control and does not repair or reinterpret that frozen result.

Two narrower results are positive:

- **R5 numerical-quality arm:** all **96/96** measurements pass `relL2 <= 0.08`: routed 32/32 (max `{by_arm['routed']['max_rel_l2']:.17g}`), shared-raw 32/32 (max `{by_arm['shared_raw']['max_rel_l2']:.17g}`), and shared-gated 32/32 (max `{by_arm['shared_gated']['max_rel_l2']:.17g}`).
- **C1-R2A synthetic integrity control:** its completed independent verifier is **11/11 positive**. The preregistered one-field mutation changes exactly one BF16 output word (`14520 -> 14489`) and the safe checker rejects the digest mismatch before unsafe decode/linear calls.

For context, the same R5 mutation at frozen natural `p0/n15` changes one BF16 word in both raw and gated outputs. That neighbor witness does not override the failed `p0/n8` conjunct.

## Evidence and provenance

The machine-readable adjudication binds the immutable raw, result, commit, runner, runner-lock, verifier, verifier-lock and preregistration artifacts for both experiments by byte count and SHA-256. Both commits validate, both result-to-raw hashes validate, and both frozen verifier source hashes agree with their locks and result records. The adjudicator opened only the two small committed raw bundles and JSON/source evidence; it did not open the official checkpoint or run a model.

Machine-readable adjudication: `{OUT.relative_to(ROOT).as_posix()}`.

## Claim boundary

This is validation-only and non-heldout. It is not evidence for a complete layer, complete model, generation quality, throughput, heterogeneous-device performance, industrial superiority, novelty, or a breakthrough. R5 remains formally negative; only its numerical-quality sub-arm and C1's separate synthetic integrity-control are positive.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"output": OUT.as_posix(), "report": REPORT.as_posix(), "evidence_integrity_pass": outcome["evidence_integrity_pass"], "formal_overall_status": outcome["formal_overall_status"]}, indent=2))
    return 0 if outcome["evidence_integrity_pass"] and outcome["formal_overall_status"] == "negative" else 2


if __name__ == "__main__":
    raise SystemExit(main())
