"""Measure official NVIDIA DFlash acceptance and standalone reference cost."""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, sha256_file, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm

RESULTS = REPO / "pro_research" / "results" / "s100_phase38"
CAPTURE_META = RESULTS / "S100_PHASE38_TARGET_CAPTURE.json"
OUT = RESULTS / "S100_PHASE38_DFLASH_MEASURE.json"
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
PHASE32_H8_MS = 122.578525


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def _percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p90": float(np.percentile(data, 90)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def _validate_capture() -> tuple[dict[str, Any], np.memmap, np.memmap]:
    meta = json.loads(CAPTURE_META.read_text(encoding="utf-8"))
    if meta.get("status") != "captured":
        raise RuntimeError("Phase38 target capture is not green")
    arrays = meta["arrays"]
    hidden_path = REPO / arrays["target_aux_hidden"]["path"]
    token_path = REPO / arrays["tokens"]["path"]
    if sha256_file(hidden_path) != arrays["target_aux_hidden"]["sha256"]:
        raise RuntimeError("target hidden capture hash mismatch")
    if sha256_file(token_path) != arrays["tokens"]["sha256"]:
        raise RuntimeError("target token capture hash mismatch")
    hidden = np.memmap(hidden_path, mode="r", dtype="<f4", shape=(COUNT, LAYERS, HIDDEN))
    tokens = np.memmap(token_path, mode="r", dtype="<i4", shape=(COUNT + 1,))
    return meta, hidden, tokens


def _embedding_audit(model, token_ids: np.ndarray) -> dict[str, Any]:
    import torch
    from moe_lab.lightningstream_nemotron.loader import ShardIndex

    index = ShardIndex(TARGET)
    entry = index.entries["backbone.embeddings.weight"]
    base = 8 + entry.header_len + entry.start
    row_bytes = HIDDEN * 2
    unique = sorted({int(x) for x in token_ids} | {990})
    mismatches: list[int] = []
    with (TARGET / entry.shard).open("rb") as handle:
        for token_id in unique:
            handle.seek(base + token_id * row_bytes)
            target_bits = np.frombuffer(handle.read(row_bytes), dtype="<u2")
            draft_bits = (
                model.embedding_rows[token_id].detach().cpu().view(torch.uint16).numpy()
            )
            if not np.array_equal(target_bits, draft_bits):
                mismatches.append(token_id)
    return {
        "unique_rows_checked": len(unique),
        "mismatch_count": len(mismatches),
        "mismatching_token_ids": mismatches,
        "mask_row_is_dflash_specific": mismatches == [990],
        "resident_optimization": "share matching target rows; retain DFlash row 990",
    }


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "s100_phase38_official_nvidia_dflash_measure",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "official checkpoint; BF16-dequant reference body plus exact packed target NVFP4 head",
    }
    try:
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        capture_meta, hidden_np, tokens = _validate_capture()
        if DFLASH.name != "7fc1f1ff4b82b917efbd0710df0872c2bb89caa5":
            raise RuntimeError("wrong DFlash snapshot")
        if TARGET.name != "e8f3c7c4de75ad84fe1bcef95d38eca76214480b":
            raise RuntimeError("wrong target snapshot")

        import torch
        from s100_phase38_dflash_reference import (
            OfficialDFlashReference, TargetNVFP4Head,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.reset_peak_memory_stats()
        required_ids = [int(x) for x in tokens]

        print("Phase38: loading and BF16-decoding official DFlash body...", flush=True)
        load_start = time.perf_counter()
        model = OfficialDFlashReference(DFLASH, required_ids)
        body_load_seconds = time.perf_counter() - load_start
        print(f"Phase38: DFlash body ready in {body_load_seconds:.2f}s", flush=True)
        embedding_audit = _embedding_audit(model, np.asarray(tokens))
        if not embedding_audit["mask_row_is_dflash_specific"]:
            raise RuntimeError(f"unexpected embedding differences: {embedding_audit}")

        print("Phase38: loading exact packed target NVFP4 LM head...", flush=True)
        head_start = time.perf_counter()
        head = TargetNVFP4Head(TARGET)
        head_load_seconds = time.perf_counter() - head_start

        aux = torch.from_numpy(np.asarray(hidden_np)).to(device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        precompute_start = time.perf_counter_ns()
        projected, context_kv = model.precompute_context(aux)
        torch.cuda.synchronize()
        precompute_ms = (time.perf_counter_ns() - precompute_start) / 1e6
        if not torch.isfinite(projected.float()).all():
            raise RuntimeError("projected target context contains non-finite values")

        # True incremental append cost: FC+hidden norm+six K/V projections,
        # K norms and RoPE for one committed target row.
        incremental_ms: list[float] = []
        for index, position in enumerate(range(START, START + 20)):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            one = model.project_and_kv_one(aux[position], position)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            if index >= 4:
                incremental_ms.append(elapsed)
            del one

        # Warm kernels, cuBLAS algorithms and packed-head JIT before timing.
        warm_hidden = None
        warm_ids = None
        for _ in range(4):
            warm_hidden = model.forward_block(
                anchor_position=START,
                anchor_token_id=int(tokens[START]),
                context_kv=context_kv,
            )
            warm_ids = head.predict(warm_hidden[1:])
        assert warm_hidden is not None and warm_ids is not None
        # The explicit second replay below avoids accidentally sampling position 0.
        replay_hidden = model.forward_block(
            anchor_position=START,
            anchor_token_id=int(tokens[START]),
            context_kv=context_kv,
        )
        replay_ids = head.predict(replay_hidden[1:])
        if not np.array_equal(warm_ids, replay_ids):
            raise RuntimeError("DFlash reference is not deterministic at the frozen anchor")

        rounds: list[dict[str, Any]] = []
        body_ms: list[float] = []
        head_ms: list[float] = []
        anchor = START
        while anchor + (BLOCK - 1) <= COUNT:
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            draft_hidden = model.forward_block(
                anchor_position=anchor,
                anchor_token_id=int(tokens[anchor]),
                context_kv=context_kv,
            )
            torch.cuda.synchronize()
            body_elapsed = (time.perf_counter_ns() - t0) / 1e6

            torch.cuda.synchronize()
            h0 = time.perf_counter_ns()
            draft_ids = head.predict(draft_hidden[1:])
            torch.cuda.synchronize()
            head_elapsed = (time.perf_counter_ns() - h0) / 1e6
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
                "anchor_token_id": int(tokens[anchor]),
                "draft_ids": draft_ids.tolist(),
                "expected_ids": expected.tolist(),
                "accepted_drafts": accepted,
                "committed_length": committed,
                "body_ms": body_elapsed,
                "head_ms": head_elapsed,
            })
            body_ms.append(body_elapsed)
            head_ms.append(head_elapsed)
            anchor += committed

        if not rounds:
            raise RuntimeError("no acceptance rounds measured")
        accepted_values = [int(r["accepted_drafts"]) for r in rounds]
        committed_values = [int(r["committed_length"]) for r in rounds]
        total_ms = [a + b for a, b in zip(body_ms, head_ms)]
        acceptance = {
            "evaluation_start_position": START,
            "last_anchor_position": rounds[-1]["anchor_position"],
            "rounds": len(rounds),
            "accepted_drafts": _percentiles([float(x) for x in accepted_values]),
            "committed_length": _percentiles([float(x) for x in committed_values]),
            "accepted_draft_histogram": {
                str(k): int(v) for k, v in sorted(Counter(accepted_values).items())
            },
            "committed_length_histogram": {
                str(k): int(v) for k, v in sorted(Counter(committed_values).items())
            },
            "zero_draft_rate": float(np.mean(np.asarray(accepted_values) == 0)),
            "full_seven_draft_rate": float(np.mean(np.asarray(accepted_values) == 7)),
            "deterministic_anchor_128": True,
        }
        timing = {
            "body_ms_per_seven_draft_block": _percentiles(body_ms),
            "packed_target_head_ms_per_seven_rows": _percentiles(head_ms),
            "total_reference_draft_ms_per_round": _percentiles(total_ms),
            "incremental_context_projection_kv_ms_per_committed_row": _percentiles(incremental_ms),
            "batched_512_context_precompute_ms_not_used_for_economics": precompute_ms,
            "body_load_seconds": body_load_seconds,
            "head_load_seconds": head_load_seconds,
        }
        mean_commit = acceptance["committed_length"]["mean"]
        median_draft = timing["total_reference_draft_ms_per_round"]["median"]
        incremental = timing["incremental_context_projection_kv_ms_per_committed_row"]["median"]
        zero_cost_ceiling = mean_commit * 1000.0 / PHASE32_H8_MS
        measured_reference = mean_commit * 1000.0 / (
            PHASE32_H8_MS + median_draft + mean_commit * incremental
        )
        economics = {
            "adopted_phase31_baseline_tok_s": PHASE31_TOK_S,
            "phase32_exact_h8_verifier_median_ms": PHASE32_H8_MS,
            "mean_committed_tokens_per_h8_verify": mean_commit,
            "zero_drafter_cost_ceiling_tok_s": zero_cost_ceiling,
            "reference_end_to_end_projection_tok_s": measured_reference,
            "zero_cost_ceiling_vs_phase31_percent": 100.0 * (zero_cost_ceiling / PHASE31_TOK_S - 1.0),
            "reference_vs_phase31_percent": 100.0 * (measured_reference / PHASE31_TOK_S - 1.0),
            "zero_cost_gate_above_adopted_baseline": zero_cost_ceiling > PHASE31_TOK_S,
            "measured_margin_gate_ge_5pct": measured_reference >= 1.05 * PHASE31_TOK_S,
        }
        integration_open = bool(
            economics["zero_cost_gate_above_adopted_baseline"]
            and economics["measured_margin_gate_ge_5pct"]
        )

        import cupy as cp
        payload.update({
            "status": "measured",
            "completed_utc": utc_now(),
            "git_head": _git_head(),
            "environment": environment_snapshot((Path(__file__), REFERENCE, PREREG)),
            "checkpoint": {
                "dflash_snapshot": DFLASH.name,
                "dflash_config_sha256": sha256_file(DFLASH / "config.json"),
                "dflash_model_sha256": sha256_file(DFLASH / "model.safetensors"),
                "target_snapshot": TARGET.name,
                "capture_meta_sha256": sha256_file(CAPTURE_META),
            },
            "reference_contract": {
                "body_weights": "official NVFP4 decoded to BF16; attention weights native BF16",
                "target_head": "official target packed NVFP4 through exact LightningStream ERVF GEMV",
                "attention": "six full-sequence non-causal GQA layers; prefix target K/V plus eight-position block",
                "block_input": "one DFlash anchor embedding plus seven DFlash mask-token-990 embeddings",
                "sample_from_anchor": False,
                "target_hidden_dtype": "captured FP32 rounded to BF16 before checkpoint fc",
            },
            "embedding_audit": embedding_audit,
            "acceptance": acceptance,
            "timing": timing,
            "memory": {
                "dflash_reference_resident_weight_bytes": model.resident_weight_bytes,
                "packed_target_head_resident_weight_bytes": head.resident_weight_bytes,
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "cupy_pool_total_bytes": int(cp.get_default_memory_pool().total_bytes()),
            },
            "economics": economics,
            "PHASE38_DFLASH_INTEGRATION_OPEN": integration_open,
            "round_records": rounds,
            "capture": {
                "meta": str(CAPTURE_META.relative_to(REPO)),
                "target_replay_exact": capture_meta["capture"]["canonical_replay_exact"],
            },
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
    brief = {
        "status": payload.get("status"),
        "acceptance": payload.get("acceptance"),
        "timing": payload.get("timing"),
        "economics": payload.get("economics"),
        "PHASE38_DFLASH_INTEGRATION_OPEN": payload.get("PHASE38_DFLASH_INTEGRATION_OPEN"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }
    # Keep stdout compact: round records remain in the result file.
    print(json.dumps(brief, indent=2), flush=True)
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
