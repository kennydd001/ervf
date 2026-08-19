from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

from common import REPO, write_json_atomic, utc_now
from s100_phase14_common import RESULTS, ensure_results

OUT = RESULTS / "S100_PHASE14F_DFLASH2_ECONOMICS.json"
PHASE12A = (
    REPO / "pro_research" / "results" / "s100_phase12"
    / "S100_PHASE12A_BLOCK_VERIFIER.json"
)
PHASE12C = (
    REPO / "pro_research" / "results" / "s100_phase12c"
    / "S100_PHASE12C_ECONOMICS.json"
)
# Reserved for a future measured full-model native block verifier. Phase 14D's
# component ceiling is deliberately not substituted for this file.
PHASE14D_BLOCK = RESULTS / "S100_PHASE14D_NATIVE_BLOCK_RUNTIME.json"

BLOCKS = (2, 4, 8)
DRAFT_LAYERS = (2, 3, 4, 5)
MLP_RATIOS = (2.0, 3.0, 3.75)
WEIGHT_FORMATS = {
    "bf16": 2.0,
    "fp8": 1.0,
    # NVFP4: two 4-bit values per code byte plus one FP8 scale per 16 values.
    "nvfp4": 0.5 + 1.0 / 16.0,
}
CONTEXTS = (4096, 32768, 131072)
RESERVE_BYTES = 512 * 1024**2
WORKSPACE_FRACTION = 0.12
CONV_KERNEL_SIZE = 2
CONV_GROUP_SIZE = 16
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
# The explicit public Qwen3.8 config/architecture count is ~7.7% above the
# simple linear-layer inventory below. Apply an 8% conservative calibration so
# the memory screen does not understate unenumerated parameters/buffers.
PARAMETER_CALIBRATION_FACTOR = 1.08


def load(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(float(x) for x in values)
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def collect_verifiers() -> tuple[list[dict], float | None, dict]:
    sources: list[dict] = []
    provenance: dict = {}
    baseline_samples: list[float] = []

    phase12a = load(PHASE12A)
    provenance["phase12a"] = str(PHASE12A.relative_to(REPO))
    if phase12a and phase12a.get("status") == "measured":
        for block_s, row in (phase12a.get("per_B") or {}).items():
            try:
                block = int(block_s)
                cycle = float(row["cycle_ms_median"])
                seq = float(row["sequential_B_tokens_ms_median"])
            except (KeyError, TypeError, ValueError):
                continue
            if block <= 0 or cycle <= 0:
                continue
            baseline_samples.append(seq / block)
            sources.append({
                "name": f"phase12a_measured_b{block}",
                "kind": "measured",
                "block": block,
                "verify_cycle_ms": cycle,
                "source_path": str(PHASE12A.relative_to(REPO)),
                "claim_boundary": (
                    "bit-exact B-token verifier with ordinary M=1 kernels; "
                    "perfect draft and full acceptance"
                ),
            })

    phase12c = load(PHASE12C)
    provenance["phase12c"] = str(PHASE12C.relative_to(REPO))
    projection = (phase12c or {}).get("projection") or {}
    try:
        projected = float(projection["projected_b4_cycle_ms"])
    except (KeyError, TypeError, ValueError):
        projected = 0.0
    if projected > 0:
        sources.append({
            "name": "phase12c_projected_b4",
            "kind": "projection",
            "block": 4,
            "verify_cycle_ms": projected,
            "source_path": str(PHASE12C.relative_to(REPO)),
            "claim_boundary": str(
                projection.get(
                    "claim_boundary",
                    "component substitution projection, not measured verifier",
                )
            ),
        })

    phase14d = load(PHASE14D_BLOCK)
    provenance["phase14d_future_block"] = str(PHASE14D_BLOCK.relative_to(REPO))
    if phase14d and phase14d.get("status") == "measured":
        for block_s, row in (phase14d.get("per_block") or {}).items():
            try:
                block = int(block_s)
                cycle = float(row["cycle_ms_median"])
            except (KeyError, TypeError, ValueError):
                continue
            if block > 0 and cycle > 0:
                sources.append({
                    "name": f"phase14d_measured_native_block_b{block}",
                    "kind": "measured",
                    "block": block,
                    "verify_cycle_ms": cycle,
                    "source_path": str(PHASE14D_BLOCK.relative_to(REPO)),
                    "claim_boundary": "future measured full-model native block verifier",
                })

    return sources, median(baseline_samples), provenance


def economics_row(source: dict, baseline_ms_per_token: float | None) -> dict:
    block = int(source["block"])
    verify_ms = float(source["verify_cycle_ms"])
    acceptance_rows = []
    for accepted in range(1, block + 1):
        s100_total_budget = 10.0 * accepted
        row = {
            "accepted_output_tokens_including_anchor": accepted,
            "s100_total_cycle_budget_ms": s100_total_budget,
            "draft_plus_selector_budget_for_s100_ms": s100_total_budget - verify_ms,
            "s100_possible_before_draft_cost": verify_ms <= s100_total_budget,
        }
        if baseline_ms_per_token is not None:
            break_even_total = baseline_ms_per_token * accepted
            row.update({
                "autoregressive_break_even_total_budget_ms": break_even_total,
                "draft_plus_selector_break_even_budget_ms": break_even_total - verify_ms,
                "can_beat_current_ar_before_draft_cost": verify_ms < break_even_total,
            })
        acceptance_rows.append(row)

    ceiling = 1000.0 * block / verify_ms
    result = dict(source)
    result.update({
        "perfect_draft_ceiling_tok_s": ceiling,
        "perfect_draft_s100_open": ceiling >= 100.0,
        "acceptance_budgets": acceptance_rows,
    })
    return result


def draft_parameter_estimate(
    *,
    hidden: int,
    q_dim: int,
    kv_dim: int,
    vocab: int,
    layers: int,
    mlp_ratio: float,
) -> dict:
    intermediate = int(math.ceil(hidden * mlp_ratio / 128.0) * 128)
    # DFlash reuses the target embedding and output head. The draft itself has
    # one dense attention+MLP stack and a projection from one selected target
    # hidden state per draft layer.
    attention_per_layer = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    mlp_per_layer = 3 * hidden * intermediate
    norms_per_layer = 2 * hidden + 2 * (q_dim // max(q_dim // hidden, 1))
    context_projection = layers * hidden * hidden
    final_norm = hidden

    groups = hidden // CONV_GROUP_SIZE
    if hidden % CONV_GROUP_SIZE != 0:
        groups = math.ceil(hidden / CONV_GROUP_SIZE)
    # Two GroupedDynamicCausalConv modules per layer. Each contains a static
    # prepare+finish kernel and one projection that predicts both kernels.
    conv_per_module = (
        2 * CONV_KERNEL_SIZE * hidden
        + hidden * 2 * CONV_KERNEL_SIZE * groups
    )
    conv_total = layers * 2 * conv_per_module
    selector = 2 * vocab * SELECTOR_RANK + hidden * SELECTOR_RANK

    base = (
        layers * (attention_per_layer + mlp_per_layer + norms_per_layer)
        + context_projection + final_norm
    )
    raw_total = base + conv_total + selector
    total = int(math.ceil(raw_total * PARAMETER_CALIBRATION_FACTOR))
    return {
        "layers": layers,
        "hidden": hidden,
        "intermediate": intermediate,
        "mlp_ratio": mlp_ratio,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "base_dflash_parameters": int(base),
        "suffix_correction_parameters": int(conv_total),
        "candidate_selector_parameters": int(selector),
        "raw_enumerated_parameters": int(raw_total),
        "parameter_calibration_factor": PARAMETER_CALIBRATION_FACTOR,
        "total_parameters": int(total),
        "dflash2_overhead_fraction_vs_base": float((conv_total + selector) / max(base, 1)),
    }


def memory_rows(rt, free_bytes: int, total_bytes: int) -> list[dict]:
    hidden = int(rt.hidden)
    q_dim = int(rt.n_heads * rt.head_dim)
    kv_dim = int(rt.n_kv * rt.head_dim)
    vocab = int(rt.vocab)
    rows = []
    for layers in DRAFT_LAYERS:
        for ratio in MLP_RATIOS:
            params = draft_parameter_estimate(
                hidden=hidden,
                q_dim=q_dim,
                kv_dim=kv_dim,
                vocab=vocab,
                layers=layers,
                mlp_ratio=ratio,
            )
            for fmt, bytes_per_param in WEIGHT_FORMATS.items():
                weight_bytes = int(math.ceil(params["total_parameters"] * bytes_per_param))
                workspace = int(math.ceil(weight_bytes * WORKSPACE_FRACTION))
                context_rows = []
                for context in CONTEXTS:
                    # A conservative resident draft-KV estimate. DFlash's
                    # attention keeps K and V per layer. BF16 is the guaranteed
                    # compatibility floor for non-causal draft attention; FP8 is
                    # shown only as a hypothetical custom-runtime path.
                    kv_bf16 = int(2 * layers * context * kv_dim * 2)
                    kv_fp8 = int(2 * layers * context * kv_dim)
                    required_bf16 = weight_bytes + workspace + kv_bf16 + RESERVE_BYTES
                    required_fp8 = weight_bytes + workspace + kv_fp8 + RESERVE_BYTES
                    context_rows.append({
                        "context_tokens": context,
                        "draft_kv_bf16_bytes": kv_bf16,
                        "draft_kv_fp8_bytes_hypothetical": kv_fp8,
                        "required_with_bf16_kv_bytes": required_bf16,
                        "required_with_fp8_kv_bytes_hypothetical": required_fp8,
                        "fits_current_free_vram_bf16_kv": required_bf16 <= free_bytes,
                        "fits_current_free_vram_fp8_kv_hypothetical": required_fp8 <= free_bytes,
                    })
                rows.append({
                    **params,
                    "weight_format": fmt,
                    "weight_bytes": weight_bytes,
                    "workspace_bytes": workspace,
                    "reserve_bytes": RESERVE_BYTES,
                    "free_vram_at_measurement_bytes": free_bytes,
                    "total_vram_bytes": total_bytes,
                    "contexts": context_rows,
                })
    return rows


def main() -> int:
    ensure_results()
    payload: dict = {
        "kind": "s100_phase14f_dflash2_economics",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": (
            "hard verifier and resident-memory feasibility screen; no DFlash2 "
            "checkpoint is trained or benchmarked"
        ),
    }
    bundle = None
    try:
        sources, baseline_ms, provenance = collect_verifiers()
        if not sources:
            raise FileNotFoundError(
                "no Phase 12A measured verifier or later block-verifier evidence found"
            )
        economics = [economics_row(row, baseline_ms) for row in sources]
        measured_full = [
            row for row in economics
            if row["kind"] == "measured"
            and (
                row["name"].startswith("phase12a_measured")
                or row["name"].startswith("phase14d_measured_native_block")
            )
        ]
        current_perfect_open = bool(measured_full) and any(
            row["perfect_draft_s100_open"] for row in measured_full
        )

        payload.update({
            "status": "measured",
            "provenance": provenance,
            "autoregressive_baseline_ms_per_token": baseline_ms,
            "verifier_sources": economics,
            "gates": {
                "CURRENT_VERIFIER_PERFECT_DRAFT_S100_OPEN": current_perfect_open,
                "ANY_COMPACT_DRAFTER_MEMORY_OPEN_4K_BF16_KV": None,
                "RESIDENT_DRAFTER_MEMORY_OPEN_4K_BF16_KV": None,
                "RESIDENT_DRAFTER_MEMORY_OPEN_128K_BF16_KV": None,
                "RESIDENT_DRAFTER_MEMORY_OPEN_4K_WITH_CUSTOM_FP8_KV": None,
                "RESIDENT_DRAFTER_MEMORY_OPEN_128K_WITH_CUSTOM_FP8_KV": None,
                "DFLASH2_CURRENT_RUNTIME_TRAINING_PREREQUISITES_OPEN": (
                    False if not current_perfect_open else None
                ),
            },
            "memory_status": "not_started",
        })

        # Verifier economics are valid even when CUDA/memory instrumentation
        # fails. Keep memory as a separately tri-stated subtest so a driver
        # error cannot erase the already measured perfect-draft ceiling.
        try:
            import cupy as cp
            from s100_phase10a_runtime import build

            bundle = build()
            rt = bundle.rt
            free_bytes, total_bytes = cp.cuda.Device(0).mem_info
            free_bytes = int(free_bytes)
            total_bytes = int(total_bytes)
            candidates = memory_rows(rt, free_bytes, total_bytes)

            def fits(
                context: int,
                *,
                allow_hypothetical_fp8: bool = False,
                reference_shape: bool = False,
            ) -> bool:
                for row in candidates:
                    if reference_shape and not (
                        row["layers"] == 5
                        and abs(float(row["mlp_ratio"]) - 3.75) < 1e-9
                        and row["weight_format"] == "nvfp4"
                    ):
                        continue
                    for c in row["contexts"]:
                        if c["context_tokens"] != context:
                            continue
                        if c["fits_current_free_vram_bf16_kv"]:
                            return True
                        if allow_hypothetical_fp8 and c[
                            "fits_current_free_vram_fp8_kv_hypothetical"
                        ]:
                            return True
                return False

            any_compact_memory_4k = fits(4096)
            memory_4k = fits(4096, reference_shape=True)
            memory_128k = fits(131072, reference_shape=True)
            memory_4k_with_custom_fp8 = fits(
                4096, allow_hypothetical_fp8=True, reference_shape=True
            )
            memory_128k_with_custom_fp8 = fits(
                131072, allow_hypothetical_fp8=True, reference_shape=True
            )

            payload.update({
                "memory_status": "measured",
                "runtime_shape": {
                    "hidden": int(rt.hidden),
                    "q_dim": int(rt.n_heads * rt.head_dim),
                    "kv_dim": int(rt.n_kv * rt.head_dim),
                    "vocab": int(rt.vocab),
                    "device_total_vram_bytes": total_bytes,
                    "device_free_vram_after_quality_parent_bytes": free_bytes,
                },
                "draft_estimate_assumptions": {
                    "target_embedding_and_lm_head_reused": True,
                    "draft_layers": list(DRAFT_LAYERS),
                    "mlp_hidden_ratios": list(MLP_RATIOS),
                    "conv_kernel_size": CONV_KERNEL_SIZE,
                    "conv_group_size": CONV_GROUP_SIZE,
                    "selector_rank": SELECTOR_RANK,
                    "selector_top_k": SELECTOR_TOP_K,
                    "parameter_calibration_factor": PARAMETER_CALIBRATION_FACTOR,
                    "workspace_fraction": WORKSPACE_FRACTION,
                    "reserve_bytes": RESERVE_BYTES,
                    "note": (
                        "Qwen3.8 DFlash2 reference uses selector rank 256, "
                        "group 16, kernel 2; parameter and memory envelope only "
                        "because a trained Nemotron draft config may differ"
                    ),
                },
                "draft_memory_candidates": candidates,
            })
            payload["gates"].update({
                "ANY_COMPACT_DRAFTER_MEMORY_OPEN_4K_BF16_KV": (
                    any_compact_memory_4k
                ),
                "RESIDENT_DRAFTER_MEMORY_OPEN_4K_BF16_KV": memory_4k,
                "RESIDENT_DRAFTER_MEMORY_OPEN_128K_BF16_KV": memory_128k,
                "RESIDENT_DRAFTER_MEMORY_OPEN_4K_WITH_CUSTOM_FP8_KV": (
                    memory_4k_with_custom_fp8
                ),
                "RESIDENT_DRAFTER_MEMORY_OPEN_128K_WITH_CUSTOM_FP8_KV": (
                    memory_128k_with_custom_fp8
                ),
                "DFLASH2_CURRENT_RUNTIME_TRAINING_PREREQUISITES_OPEN": bool(
                    current_perfect_open and memory_4k
                ),
            })
        except Exception as memory_exc:
            payload.update({
                "memory_status": "technical_failure",
                "memory_error": {
                    "type": type(memory_exc).__name__,
                    "message": str(memory_exc),
                    "traceback": traceback.format_exc(),
                },
            })

        memory_gate = payload["gates"][
            "RESIDENT_DRAFTER_MEMORY_OPEN_4K_BF16_KV"
        ]
        if not current_perfect_open:
            decision = "BLOCK_FULL_DRAFTER_TRAINING_UNTIL_VERIFIER_BREAKTHROUGH"
        elif memory_gate is None:
            decision = "MEMORY_EVIDENCE_INCOMPLETE"
        elif not memory_gate:
            decision = "MEMORY_REDESIGN_REQUIRED"
        else:
            decision = "RUN_NEMOTRON_DFLASH2_TRAINING_PILOT_IF_TRANSFER_SIGNAL_PASSES"
        payload.update({
            "decision": decision,
            "completed_utc": utc_now(),
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
    finally:
        if bundle is not None:
            try:
                bundle.restore_combined()
                bundle.restore_sel()
            except Exception:
                pass

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "memory_status": payload.get("memory_status"),
        "gates": payload.get("gates"),
        "decision": payload.get("decision"),
        "error": (payload.get("error") or payload.get("memory_error") or {}).get(
            "message"
        ),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
