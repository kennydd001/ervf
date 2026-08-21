"""Adjudicate Phase38 official NVIDIA DFlash without rerunning GPU work."""
from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, sha256_file, utc_now, write_json_atomic

RESULTS = REPO / "pro_research" / "results" / "s100_phase38"
MAIN_CAPTURE = RESULTS / "S100_PHASE38_TARGET_CAPTURE.json"
MAIN_MEASURE = RESULTS / "S100_PHASE38_DFLASH_MEASURE.json"
PROXY_CAPTURE = RESULTS / "S100_PHASE38_DFLASH_BF16_PROXY_CAPTURE.json"
PROXY_MEASURE = RESULTS / "S100_PHASE38_DFLASH_BF16_PROXY_MEASURE.json"
OUT = RESULTS / "S100_PHASE38_DFLASH_ADJUDICATION.json"
SUMMARY = RESULTS / "S100_PHASE38_DFLASH_SUMMARY.txt"
PREREG = REPO / "pro_research" / "S100_PHASE38_DFLASH_PREREGISTRATION.md"
SOURCES = (
    REPO / "pro_research" / "s100_phase38_dflash_capture.py",
    REPO / "pro_research" / "s100_phase38_dflash_reference.py",
    REPO / "pro_research" / "s100_phase38_dflash_measure.py",
    REPO / "pro_research" / "s100_phase38_dflash_bf16_proxy_capture.py",
    REPO / "pro_research" / "s100_phase38_dflash_bf16_proxy_measure.py",
    Path(__file__),
    PREREG,
)
PHASE31_TOK_S = 62.96114117068372
PHASE31_H4_MS = 63.53125
PHASE32_H8_MS = 122.578525
BLOCK = 8
PUBLISHED_ACCEPTED_DRAFTS = 3.16


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def _load_green(path: Path, expected_status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected_status:
        raise RuntimeError(
            f"{path.name}: status={payload.get('status')!r}, expected={expected_status!r}"
        )
    return payload


def _budget(
    *,
    committed: float,
    gate_tok_s: float,
    draft_ms: float,
    append_ms: float,
) -> dict[str, float]:
    total_budget = committed * 1000.0 / gate_tok_s
    zero_draft_verifier = total_budget
    reference_verifier = total_budget - draft_ms - committed * append_ms
    return {
        "mean_committed_tokens": committed,
        "total_round_budget_ms_at_gate": total_budget,
        "max_verifier_ms_zero_drafter_cost": zero_draft_verifier,
        "max_verifier_ms_with_reference_draft_and_append": reference_verifier,
        "zero_draft_verifier_reduction_vs_current_percent": 100.0
        * (1.0 - zero_draft_verifier / PHASE32_H8_MS),
        "reference_cost_verifier_reduction_vs_current_percent": 100.0
        * (1.0 - reference_verifier / PHASE32_H8_MS),
    }


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_phase38_official_nvidia_dflash_adjudication",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "local greedy DFlash acceptance, activation sensitivity, and acceptance-independent economic upper bounds",
    }
    try:
        main_capture = _load_green(MAIN_CAPTURE, "captured")
        main_measure = _load_green(MAIN_MEASURE, "measured")
        proxy_capture = _load_green(PROXY_CAPTURE, "captured")
        proxy_measure = _load_green(PROXY_MEASURE, "measured")

        main_hash = main_measure["checkpoint"]["dflash_model_sha256"]
        proxy_hash = proxy_measure["checkpoint"]["dflash_model_sha256"]
        if main_hash != proxy_hash:
            raise RuntimeError("main/proxy DFlash checkpoint hash mismatch")
        if not main_capture["capture"]["canonical_replay_exact"]:
            raise RuntimeError("main target capture did not replay canonically")
        if not proxy_capture["trace"]["self_consistent_continuation"]:
            raise RuntimeError("BF16 proxy continuation is not self-consistent")

        main_accept = float(main_measure["acceptance"]["accepted_drafts"]["mean"])
        main_commit = float(main_measure["acceptance"]["committed_length"]["mean"])
        proxy_accept = float(proxy_measure["acceptance"]["accepted_drafts"]["mean"])
        proxy_commit = float(proxy_measure["acceptance"]["committed_length"]["mean"])
        timing = main_measure["timing"]
        draft_ms = float(timing["total_reference_draft_ms_per_round"]["median"])
        append_ms = float(
            timing["incremental_context_projection_kv_ms_per_committed_row"]["median"]
        )
        gate_tok_s = 1.05 * PHASE31_TOK_S
        perfect_zero_cost = BLOCK * 1000.0 / PHASE32_H8_MS
        h4_perfect_zero_cost = 4.0 * 1000.0 / PHASE31_H4_MS

        budgets = {
            "local_fp32_residual_target": _budget(
                committed=main_commit,
                gate_tok_s=gate_tok_s,
                draft_ms=draft_ms,
                append_ms=append_ms,
            ),
            "local_bf16_activation_proxy": _budget(
                committed=proxy_commit,
                gate_tok_s=gate_tok_s,
                draft_ms=draft_ms,
                append_ms=append_ms,
            ),
            "nvidia_published_mean_acceptance_interpretation": _budget(
                committed=1.0 + PUBLISHED_ACCEPTED_DRAFTS,
                gate_tok_s=gate_tok_s,
                draft_ms=draft_ms,
                append_ms=append_ms,
            ),
            "perfect_seven_of_seven": _budget(
                committed=float(BLOCK),
                gate_tok_s=gate_tok_s,
                draft_ms=draft_ms,
                append_ms=append_ms,
            ),
        }

        closure_reasons = [
            (
                f"Local exact-target acceptance is {main_accept:.4f} accepted drafts "
                f"({main_commit:.4f} committed), giving only "
                f"{main_measure['economics']['zero_drafter_cost_ceiling_tok_s']:.3f} tok/s "
                "before any draft cost."
            ),
            (
                f"The BF16 activation proxy raises acceptance to {proxy_accept:.4f} "
                f"({proxy_commit:.4f} committed), but its zero-draft ceiling is only "
                f"{proxy_measure['economics']['proxy_zero_drafter_cost_ceiling_tok_s']:.3f} tok/s."
            ),
            (
                f"Even perfect seven-of-seven acceptance with a zero-cost drafter is "
                f"{perfect_zero_cost:.3f} tok/s, below the frozen 5% gate "
                f"of {gate_tok_s:.3f} tok/s."
            ),
            (
                f"A four-position verifier has a perfect zero-draft ceiling of "
                f"{h4_perfect_zero_cost:.3f} tok/s, exactly the target-only baseline, "
                "so a shorter DFlash block has no positive draft-cost budget."
            ),
        ]

        payload.update({
            "status": "closed",
            "completed_utc": utc_now(),
            "git_head": _git_head(),
            "environment": environment_snapshot(SOURCES),
            "artifact_hashes": {
                path.name: sha256_file(path)
                for path in (MAIN_CAPTURE, MAIN_MEASURE, PROXY_CAPTURE, PROXY_MEASURE)
            },
            "checkpoint": {
                "repository": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
                "model_sha256": main_hash,
                "snapshot": main_measure["checkpoint"]["dflash_snapshot"],
                "dspark_used": False,
            },
            "contract_adjudication": {
                "layer_indexing": "green: checkpoint zero-based post-layer [1,5,19,29,41,51] equals embedding-inclusive hidden indices [2,6,20,30,42,52]",
                "positioning": "green: committed target K/V positions 0..anchor-1; query block anchor..anchor+7",
                "embedding": "green: checkpoint-owned embeddings; only mask row 990 differs from audited target rows",
                "attention": "green: six full non-causal Qwen3 layers, all block queries see committed context plus full block",
                "head": "green: shared exact packed target NVFP4 LM head on block positions 1..7",
                "rope": "green: target has no RoPE-style override, so vLLM default NeoX layout with checkpoint YaRN parameters",
            },
            "acceptance": {
                "local_fp32_residual_target_mean_accepted_drafts": main_accept,
                "local_fp32_residual_target_mean_committed": main_commit,
                "local_bf16_activation_proxy_mean_accepted_drafts": proxy_accept,
                "local_bf16_activation_proxy_mean_committed": proxy_commit,
                "proxy_first_canonical_divergence_position": proxy_capture["trace"]["first_canonical_divergence_position"],
                "nvidia_published_mean_accepted_drafts_context_only": PUBLISHED_ACCEPTED_DRAFTS,
                "published_value_comparability": "different benchmark mix and temperature-1/top-p-0.95 sampling; used only as an optimistic economic scenario",
            },
            "timing": {
                "reference_dflash_body_plus_head_median_ms": draft_ms,
                "incremental_context_append_median_ms_per_committed_row": append_ms,
                "phase31_h4_verifier_median_ms": PHASE31_H4_MS,
                "phase32_h8_verifier_median_ms": PHASE32_H8_MS,
            },
            "economics": {
                "adopted_target_only_tok_s": PHASE31_TOK_S,
                "frozen_5pct_gate_tok_s": gate_tok_s,
                "perfect_h8_zero_draft_ceiling_tok_s": perfect_zero_cost,
                "perfect_h4_zero_draft_ceiling_tok_s": h4_perfect_zero_cost,
                "verifier_budgets": budgets,
            },
            "primary_source_notes": {
                "target_model_card": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                "dflash_model_card": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
                "deployment_note": "current target card's DGX Spark vLLM recipe uses DSpark; its SGLang section allows DFlash and recommends a shorter block on H100/DGX Spark",
            },
            "closure_reasons": closure_reasons,
            "next_breakthrough_requirement": {
                "decision": "Do not implement a native packed DFlash body on this verifier path.",
                "most_relevant_optimistic_scenario": "NVIDIA published 3.16 accepted drafts, interpreted as 4.16 committed tokens",
                "required_exact_verifier_ms_at_5pct_zero_drafter_cost": budgets[
                    "nvidia_published_mean_acceptance_interpretation"
                ]["max_verifier_ms_zero_drafter_cost"],
                "required_exact_verifier_ms_at_5pct_with_reference_cost": budgets[
                    "nvidia_published_mean_acceptance_interpretation"
                ]["max_verifier_ms_with_reference_draft_and_append"],
                "direction": "The next breakthrough must first reduce exact variable-horizon verification cost; DFlash kernel optimization alone cannot open the gate.",
            },
            "PHASE38_DFLASH_INTEGRATION_OPEN": False,
            "PHASE38_DFLASH_CLOSED": True,
        })
        SUMMARY.write_text(
            "\n".join(
                [
                    "S100 Phase38 — official NVIDIA DFlash",
                    "",
                    f"Local FP32-target acceptance: {main_accept:.4f} drafts / {main_commit:.4f} committed",
                    f"BF16-proxy acceptance: {proxy_accept:.4f} drafts / {proxy_commit:.4f} committed",
                    f"Perfect H8 zero-draft ceiling: {perfect_zero_cost:.3f} tok/s",
                    f"Frozen 5% gate: {gate_tok_s:.3f} tok/s",
                    "Decision: CLOSED — no native DFlash integration on the current verifier path.",
                    "Next: reduce exact variable-horizon verifier cost before revisiting any drafter.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "acceptance": payload.get("acceptance"),
        "economics": payload.get("economics"),
        "next_breakthrough_requirement": payload.get("next_breakthrough_requirement"),
        "PHASE38_DFLASH_CLOSED": payload.get("PHASE38_DFLASH_CLOSED"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
