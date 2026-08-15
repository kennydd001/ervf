from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors, load_qwen_decoder_layer
from scripts.streamq5_moe.run_p0c_model_quality import (
    quantize_experts_q5_, quantize_groupwise_, quantize_trunk_, selected_embeddings, tensor_error,
)
from scripts.streamq5_moe.run_p9b_structured_wanda_pruning import (
    CONTEXT, DOMAINS, EVAL, EVAL_LOCK, EXPERTS, MASKS, MODEL, WIDTH,
    evaluate_head, forward_chunks, sha256,
)

PREREG = ROOT / "reports/streamq5_moe/P9BR_CORRECTED_STRUCTURED_WANDA_PREREGISTRATION.md"
OUT_DIR = ROOT / "reports/streamq5_moe"
EXPECTED_MASK_SHA256 = "a238950c97c52f7b5c12eaae76a1ef21f8c4ad4fabd4724bc13f9ed17801dab1"
KEEP = 384


def validate_masks(mask_tensors: dict[str, torch.Tensor]) -> None:
    if sha256(MASKS) != EXPECTED_MASK_SHA256:
        raise ValueError("frozen P9B mask hash mismatch")
    for layer_index in range(48):
        value = mask_tensors[f"layer_{layer_index:02d}"].long()
        if value.shape != (EXPERTS, KEEP) or int(value.min()) < 0 or int(value.max()) >= WIDTH:
            raise ValueError(f"invalid mask shape/range at layer {layer_index}")
        if bool((value[:, 1:] <= value[:, :-1]).any()):
            raise ValueError(f"mask is not strictly sorted/unique at layer {layer_index}")


@torch.no_grad()
def apply_corrected_pruning_(layer, keeps: torch.Tensor) -> dict[str, int | bool]:
    changed_nonzero = 0
    remaining_masked_nonzero = 0
    for index, expert in enumerate(layer.mlp.experts):
        keep = torch.zeros(WIDTH, dtype=torch.bool, device=expert.down_proj.weight.device)
        keep[keeps[index].long().to(keep.device)] = True
        for projection in (expert.gate_proj.weight, expert.up_proj.weight):
            remove = ~keep[:, None]
            changed_nonzero += int(torch.count_nonzero(projection.masked_select(remove)))
            projection.mul_(keep[:, None])
            remaining_masked_nonzero += int(torch.count_nonzero(projection.masked_select(remove)))
        remove_down = ~keep[None, :]
        changed_nonzero += int(torch.count_nonzero(expert.down_proj.weight.masked_select(remove_down)))
        expert.down_proj.weight.mul_(keep[None, :])
        remaining_masked_nonzero += int(torch.count_nonzero(expert.down_proj.weight.masked_select(remove_down)))
    return {
        "changed_nonzero_elements": changed_nonzero,
        "remaining_masked_nonzero": remaining_masked_nonzero,
        "effective_mutation": changed_nonzero > 0 and remaining_masked_nonzero == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    output = OUT_DIR / f"p9br_corrected_structured_wanda_{args.split}.json"
    if output.exists():
        raise FileExistsError(output)
    validation_path = OUT_DIR / "p9br_corrected_structured_wanda_validation.json"
    if args.split == "test":
        if not validation_path.exists():
            raise RuntimeError("validation required")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation["status"] != "validation_pass_test_authorized":
            raise RuntimeError("test not authorized")
    eval_lock = json.loads(EVAL_LOCK.read_text(encoding="utf-8"))
    if sha256(EVAL) != eval_lock["artifact_sha256"]:
        raise ValueError("evaluation provenance mismatch")
    mask_tensors = load_file(MASKS)
    validate_masks(mask_tensors)
    eval_source = load_file(EVAL)
    ids = torch.cat([eval_source[f"{args.split}_{domain}"] for domain in DOMAINS], 0).long().contiguous()
    if ids.shape != (10, CONTEXT) or not torch.cuda.is_available():
        raise RuntimeError("input shape or CUDA gate failed")

    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    weight_map = checkpoint_weight_map(MODEL)
    started = time.perf_counter()
    teacher = selected_embeddings(MODEL, ids, device, weight_map, None)
    candidate = selected_embeddings(MODEL, ids, device, weight_map, 8)
    layers = []
    for layer_index in range(48):
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        teacher = forward_chunks(layer, rotary, teacher, device)
        mutation = apply_corrected_pruning_(layer, mask_tensors[f"layer_{layer_index:02d}"])
        if not mutation["effective_mutation"]:
            raise RuntimeError(f"pruning mutation gate failed at layer {layer_index}: {mutation}")
        quantize_experts_q5_(layer)
        quantize_trunk_(layer, 8)
        candidate = forward_chunks(layer, rotary, candidate, device)
        row = {"layer": layer_index, "hidden_error": tensor_error(candidate, teacher), "mutation": mutation, "seconds": time.perf_counter() - layer_started}
        layers.append(row)
        print(json.dumps(row), flush=True)
        del layer, rotary
        gc.collect(); torch.cuda.empty_cache()

    final = load_checkpoint_tensors(MODEL, ["model.norm.weight", "lm_head.weight"], weight_map)
    norm, head = final["model.norm.weight"].to(device), final["lm_head.weight"].to(device)
    teacher_ce, teacher_top1, _ = evaluate_head(teacher, ids, norm, head, device)
    quantize_groupwise_(head, 8, row_batch=256)
    candidate_ce, _candidate_top1, agreement = evaluate_head(candidate, ids, norm, head, device, teacher_top1)
    teacher_mean, candidate_mean = float(teacher_ce.mean()), float(candidate_ce.mean())
    relative = (candidate_mean - teacher_mean) / teacher_mean
    finite = bool(np.isfinite(candidate_ce).all())
    mutation_pass = all(row["mutation"]["effective_mutation"] for row in layers)
    validation_relative = relative if args.split == "validation" else validation["relative_cross_entropy_increase"]
    validation_top1 = agreement if args.split == "validation" else validation["top1_agreement"]
    if args.split == "validation":
        opened = finite and mutation_pass and len(layers) == 48 and relative <= 0.025 and agreement >= 0.90
        status = "validation_pass_test_authorized" if opened else "validation_closed"
        overall = False
    else:
        overall = finite and mutation_pass and relative <= 0.02 and validation_relative <= 0.02 and agreement >= 0.90 and validation_top1 >= 0.90
        status = "quality_pass" if overall else "quality_closed"
    gates = {
        "frozen_mask_hash": sha256(MASKS) == EXPECTED_MASK_SHA256,
        "finite": finite,
        "all_48_layers": len(layers) == 48,
        "effective_mutation_every_layer": mutation_pass,
        "relative_ce_gate": relative <= (0.025 if args.split == "validation" else 0.02),
        "top1_ge_90pct": agreement >= 0.90,
    }
    result = {
        "kind": "streamq5_moe_p9br_corrected_structured_wanda",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "split": args.split, "status": status,
        "inputs": {"preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)), "evaluation_sha256": sha256(EVAL), "masks_sha256": sha256(MASKS), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "teacher_context_ce": teacher_ce.tolist(), "candidate_context_ce": candidate_ce.tolist(),
        "teacher_ce": teacher_mean, "candidate_ce": candidate_mean,
        "relative_cross_entropy_increase": relative, "top1_agreement": agreement,
        "expert_keep_fraction": 0.5, "final_hidden_error": tensor_error(candidate, teacher),
        "layers": layers, "gates": gates, "overall_pass": overall,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "Corrected full-depth quality of the frozen P9B structured-Wanda mask; no compact bank, kernel, transfer, or runtime claim.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "relative": relative, "top1": agreement, "gates": gates, "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
