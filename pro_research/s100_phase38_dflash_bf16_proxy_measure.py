"""Measure official DFlash on the post-hoc BF16 target-activation proxy."""
from __future__ import annotations

import json
import subprocess
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, sha256_file, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm

RESULTS = REPO / "pro_research" / "results" / "s100_phase38"
CAPTURE_META = RESULTS / "S100_PHASE38_DFLASH_BF16_PROXY_CAPTURE.json"
MAIN_MEASURE = RESULTS / "S100_PHASE38_DFLASH_MEASURE.json"
OUT = RESULTS / "S100_PHASE38_DFLASH_BF16_PROXY_MEASURE.json"
PREREG = REPO / "pro_research" / "S100_PHASE38_DFLASH_PREREGISTRATION.md"
REFERENCE = REPO / "pro_research" / "s100_phase38_dflash_reference.py"
DFLASH = (
    Path(r"C:\Users\de_do\.cache\huggingface\hub")
    / "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash"
    / "snapshots" / "7fc1f1ff4b82b917efbd0710df0872c2bb89caa5"
)
TARGET = (
    Path(r"C:\Users\de_do\Documents\ChatGPT\New project\.cache\nemotron_3_5_lightning\hub")
    / "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    / "snapshots" / "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"
)
COUNT = 512
START = 128
HIDDEN = 2688
LAYERS = 6
BLOCK = 8
PHASE31_TOK_S = 62.96114117068372
PHASE31_H4_MS = 63.53125
PHASE32_H8_MS = 122.578525


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def _stats(values: list[int]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p90": float(np.percentile(data, 90)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def _load_capture() -> tuple[dict[str, Any], np.memmap, np.memmap]:
    meta = json.loads(CAPTURE_META.read_text(encoding="utf-8"))
    if meta.get("status") != "captured":
        raise RuntimeError("BF16 proxy capture is not green")
    arrays = meta["arrays"]
    hidden_path = REPO / arrays["target_aux_hidden"]["path"]
    token_path = REPO / arrays["tokens"]["path"]
    if sha256_file(hidden_path) != arrays["target_aux_hidden"]["sha256"]:
        raise RuntimeError("BF16 proxy hidden hash mismatch")
    if sha256_file(token_path) != arrays["tokens"]["sha256"]:
        raise RuntimeError("BF16 proxy token hash mismatch")
    hidden = np.memmap(hidden_path, mode="r", dtype="<f4", shape=(COUNT, LAYERS, HIDDEN))
    tokens = np.memmap(token_path, mode="r", dtype="<i4", shape=(COUNT + 1,))
    return meta, hidden, tokens


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_phase38_dflash_bf16_residual_proxy_measure",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "post-hoc target-activation sensitivity; unchanged official DFlash checkpoint/reference forward",
    }
    try:
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        capture_meta, hidden_np, tokens = _load_capture()
        main_measure = json.loads(MAIN_MEASURE.read_text(encoding="utf-8"))
        if main_measure.get("status") != "measured":
            raise RuntimeError("frozen Phase38 reference timing is unavailable")

        import torch
        from s100_phase38_dflash_reference import OfficialDFlashReference, TargetNVFP4Head

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model = OfficialDFlashReference(DFLASH, [int(x) for x in tokens])
        head = TargetNVFP4Head(TARGET)
        aux = torch.from_numpy(np.asarray(hidden_np).copy()).to(
            device="cuda", dtype=torch.bfloat16
        )
        _, context_kv = model.precompute_context(aux)

        # Warm both the DFlash body and packed shared head.
        for _ in range(3):
            warm_hidden = model.forward_block(
                anchor_position=START,
                anchor_token_id=int(tokens[START]),
                context_kv=context_kv,
            )
            head.predict(warm_hidden[1:])

        rounds: list[dict[str, Any]] = []
        anchor = START
        while anchor + (BLOCK - 1) <= COUNT:
            draft_hidden = model.forward_block(
                anchor_position=anchor,
                anchor_token_id=int(tokens[anchor]),
                context_kv=context_kv,
            )
            draft_ids = head.predict(draft_hidden[1:])
            expected = np.asarray(tokens[anchor + 1:anchor + BLOCK], dtype=np.int32)
            accepted = 0
            for got, want in zip(draft_ids, expected):
                if int(got) != int(want):
                    break
                accepted += 1
            committed = accepted + 1
            rounds.append({
                "round": len(rounds),
                "anchor_position": anchor,
                "draft_ids": draft_ids.tolist(),
                "expected_ids": expected.tolist(),
                "accepted_drafts": accepted,
                "committed_length": committed,
            })
            anchor += committed

        accepted_values = [int(r["accepted_drafts"]) for r in rounds]
        committed_values = [int(r["committed_length"]) for r in rounds]
        if not rounds:
            raise RuntimeError("no BF16 proxy rounds measured")
        acceptance = {
            "evaluation_start_position": START,
            "last_anchor_position": rounds[-1]["anchor_position"],
            "rounds": len(rounds),
            "accepted_drafts": _stats(accepted_values),
            "committed_length": _stats(committed_values),
            "accepted_draft_histogram": {
                str(k): int(v) for k, v in sorted(Counter(accepted_values).items())
            },
            "zero_draft_rate": float(np.mean(np.asarray(accepted_values) == 0)),
            "full_seven_draft_rate": float(np.mean(np.asarray(accepted_values) == 7)),
        }

        timing = main_measure["timing"]
        draft_ms = float(timing["total_reference_draft_ms_per_round"]["median"])
        append_ms = float(
            timing["incremental_context_projection_kv_ms_per_committed_row"]["median"]
        )
        mean_commit = float(acceptance["committed_length"]["mean"])
        gate_tok_s = 1.05 * PHASE31_TOK_S
        measured_projection = mean_commit * 1000.0 / (
            PHASE32_H8_MS + draft_ms + mean_commit * append_ms
        )
        perfect_zero_cost = BLOCK * 1000.0 / PHASE32_H8_MS
        perfect_reference = BLOCK * 1000.0 / (
            PHASE32_H8_MS + draft_ms + BLOCK * append_ms
        )
        economics = {
            "adopted_phase31_baseline_tok_s": PHASE31_TOK_S,
            "frozen_gate_5pct_tok_s": gate_tok_s,
            "phase31_h4_median_ms": PHASE31_H4_MS,
            "phase32_h8_median_ms": PHASE32_H8_MS,
            "proxy_mean_committed_tokens": mean_commit,
            "proxy_zero_drafter_cost_ceiling_tok_s": mean_commit * 1000.0 / PHASE32_H8_MS,
            "proxy_reference_projection_tok_s": measured_projection,
            "perfect_7_of_7_zero_drafter_cost_ceiling_tok_s": perfect_zero_cost,
            "perfect_7_of_7_reference_projection_tok_s": perfect_reference,
            "perfect_7_of_7_zero_cost_vs_baseline_percent": 100.0 * (
                perfect_zero_cost / PHASE31_TOK_S - 1.0
            ),
            "perfect_7_of_7_zero_cost_clears_5pct_gate": perfect_zero_cost >= gate_tok_s,
            "h8_verifier_ms_required_for_perfect_acceptance_5pct_zero_draft": BLOCK * 1000.0 / gate_tok_s,
            "h8_verifier_ms_required_for_perfect_acceptance_5pct_with_reference_cost": (
                BLOCK * 1000.0 / gate_tok_s - draft_ms - BLOCK * append_ms
            ),
            "h4_perfect_zero_draft_ceiling_tok_s": 4.0 * 1000.0 / PHASE31_H4_MS,
            "acceptance_independent_conclusion": (
                "current H8 verifier cannot clear the frozen 5% gate even with "
                "perfect seven-of-seven acceptance and a zero-cost drafter"
            ),
        }

        payload.update({
            "status": "measured",
            "completed_utc": utc_now(),
            "git_head": _git_head(),
            "environment": environment_snapshot((Path(__file__), REFERENCE, PREREG)),
            "checkpoint": {
                "dflash_snapshot": DFLASH.name,
                "dflash_model_sha256": sha256_file(DFLASH / "model.safetensors"),
                "target_snapshot": TARGET.name,
                "proxy_capture_meta_sha256": sha256_file(CAPTURE_META),
                "frozen_main_measure_sha256": sha256_file(MAIN_MEASURE),
            },
            "contract_validation": {
                "checkpoint_target_ids_zero_based_post_layer": [1, 5, 19, 29, 41, 51],
                "vllm_aux_hidden_indices_embedding_inclusive": [2, 6, 20, 30, 42, 52],
                "context_positions_at_anchor": "0..anchor-1",
                "query_positions_at_anchor": "anchor..anchor+7",
                "query_embeddings": "official DFlash checkpoint embed_tokens including row 990",
                "sampled_hidden_positions": "block positions 1..7",
            },
            "proxy_capture": {
                "meta": str(CAPTURE_META.relative_to(REPO)),
                "first_canonical_divergence_position": capture_meta["trace"]["first_canonical_divergence_position"],
                "limitations": capture_meta["capture"]["limitations"],
            },
            "acceptance": acceptance,
            "economics": economics,
            "PHASE38_DFLASH_INTEGRATION_OPEN": False,
            "closure_reason": "acceptance-independent perfect-block upper bound misses frozen 5% gate",
            "round_records": rounds,
        })
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
        "PHASE38_DFLASH_INTEGRATION_OPEN": payload.get("PHASE38_DFLASH_INTEGRATION_OPEN"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
