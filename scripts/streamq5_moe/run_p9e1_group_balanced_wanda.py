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
from safetensors.torch import load_file, save_file
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
    CALIBRATION, CALIBRATION_LOCK, CHUNK, CONTEXT, DOMAINS, EVAL, EVAL_LOCK,
    EXPERTS, MODEL, WIDTH, apply_structured_pruning_, evaluate_head,
    forward_chunks, sha256,
)

PREREG = ROOT / "reports/streamq5_moe/P9E1_GROUP_BALANCED_WANDA_PREREGISTRATION.md"
MASKS = ROOT / "reports/runs/streamq5_moe/p9e1_group_balanced_keep.safetensors"
OUT_DIR = ROOT / "reports/streamq5_moe"
GROUP = 128
KEEP_PER_GROUP = 64
KEEP = 6 * KEEP_PER_GROUP


@torch.no_grad()
def fit_group_balanced_indices(layer, rotary, calibration_hidden, device):
    sums = torch.zeros((EXPERTS, WIDTH), dtype=torch.float32, device=device)
    counts = torch.zeros(EXPERTS, dtype=torch.int64, device=device)
    handles = []
    for expert_index, expert in enumerate(layer.mlp.experts):
        def hook(_module, inputs, index=expert_index):
            value = inputs[0].float()
            if value.numel():
                sums[index].add_(value.square().sum(dim=0))
                counts[index].add_(value.shape[0])
        handles.append(expert.down_proj.register_forward_pre_hook(hook))
    try:
        next_hidden = forward_chunks(layer, rotary, calibration_hidden, device)
    finally:
        for handle in handles:
            handle.remove()
    keeps = torch.empty((EXPERTS, KEEP), dtype=torch.int16)
    empty = 0
    tie = torch.arange(WIDTH, device=device, dtype=torch.float32) * (-1e-12)
    for expert_index, expert in enumerate(layer.mlp.experts):
        if counts[expert_index] == 0:
            empty += 1
        rms = torch.sqrt(sums[expert_index] / counts[expert_index].clamp_min(1))
        score = rms * torch.linalg.vector_norm(expert.down_proj.weight.float(), dim=0) + tie
        selected_groups = []
        for group_index in range(WIDTH // GROUP):
            start = group_index * GROUP
            local = torch.topk(score[start:start + GROUP], KEEP_PER_GROUP, largest=True, sorted=False).indices
            selected_groups.append(local + start)
        selected = torch.cat(selected_groups).sort().values
        keeps[expert_index] = selected.to(torch.int16).cpu()
    return next_hidden, keeps, empty


def masks_group_balanced(mask_tensors: dict[str, torch.Tensor]) -> bool:
    for layer_index in range(48):
        value = mask_tensors[f"layer_{layer_index:02d}"].long()
        if value.shape != (EXPERTS, KEEP):
            return False
        if bool((value[:, 1:] <= value[:, :-1]).any()) or int(value.min()) < 0 or int(value.max()) >= WIDTH:
            return False
        groups = torch.div(value, GROUP, rounding_mode="floor")
        counts = torch.stack([(groups == group).sum(dim=1) for group in range(6)], dim=1)
        if not bool((counts == KEEP_PER_GROUP).all()):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    output = OUT_DIR / f"p9e1_group_balanced_wanda_{args.split}.json"
    if output.exists():
        raise FileExistsError(output)
    validation_path = OUT_DIR / "p9e1_group_balanced_wanda_validation.json"
    if args.split == "test":
        if not validation_path.exists() or not MASKS.exists():
            raise RuntimeError("validation and masks required")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation["status"] != "validation_pass_test_authorized":
            raise RuntimeError("test not authorized")

    calibration_lock = json.loads(CALIBRATION_LOCK.read_text(encoding="utf-8"))
    eval_lock = json.loads(EVAL_LOCK.read_text(encoding="utf-8"))
    if sha256(CALIBRATION) != calibration_lock["artifact_sha256"] or sha256(EVAL) != eval_lock["artifact_sha256"]:
        raise ValueError("input provenance mismatch")
    eval_source = load_file(EVAL)
    ids = torch.cat([eval_source[f"{args.split}_{domain}"] for domain in DOMAINS], 0).long().contiguous()
    if ids.shape != (10, CONTEXT):
        raise RuntimeError(tuple(ids.shape))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    weight_map = checkpoint_weight_map(MODEL)
    started = time.perf_counter()
    teacher = selected_embeddings(MODEL, ids, device, weight_map, None)
    candidate = selected_embeddings(MODEL, ids, device, weight_map, 8)
    mask_tensors = {} if args.split == "validation" else load_file(MASKS)
    if args.split == "validation":
        calibration_source = load_file(CALIBRATION)
        calibration_ids = torch.cat([calibration_source[domain][:10, :CONTEXT] for domain in DOMAINS], 0).long()
        calibration_hidden = selected_embeddings(MODEL, calibration_ids, device, weight_map, None)
    else:
        calibration_hidden = None

    layers, empty_experts = [], 0
    for layer_index in range(48):
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        if args.split == "validation":
            calibration_hidden, keeps, empty = fit_group_balanced_indices(layer, rotary, calibration_hidden, device)
            mask_tensors[f"layer_{layer_index:02d}"] = keeps
            empty_experts += empty
        else:
            keeps = mask_tensors[f"layer_{layer_index:02d}"]
            empty = 0
        teacher = forward_chunks(layer, rotary, teacher, device)
        apply_structured_pruning_(layer, keeps)
        quantize_experts_q5_(layer)
        quantize_trunk_(layer, 8)
        candidate = forward_chunks(layer, rotary, candidate, device)
        row = {"layer": layer_index, "hidden_error": tensor_error(candidate, teacher), "empty_calibration_experts": empty, "seconds": time.perf_counter() - layer_started}
        layers.append(row)
        print(json.dumps(row), flush=True)
        del layer, rotary
        gc.collect(); torch.cuda.empty_cache()
    group_balance = masks_group_balanced(mask_tensors)
    if args.split == "validation":
        save_file(mask_tensors, MASKS, metadata={"kind": "p9e1_group_balanced_keep64_per_group", "preregistration_sha256": sha256(PREREG)})

    final = load_checkpoint_tensors(MODEL, ["model.norm.weight", "lm_head.weight"], weight_map)
    norm, head = final["model.norm.weight"].to(device), final["lm_head.weight"].to(device)
    teacher_ce, teacher_top1, _ = evaluate_head(teacher, ids, norm, head, device)
    quantize_groupwise_(head, 8, row_batch=256)
    candidate_ce, _candidate_top1, agreement = evaluate_head(candidate, ids, norm, head, device, teacher_top1)
    teacher_mean, candidate_mean = float(teacher_ce.mean()), float(candidate_ce.mean())
    relative = (candidate_mean - teacher_mean) / teacher_mean
    finite = bool(np.isfinite(candidate_ce).all())
    if args.split == "validation":
        passed = finite and group_balance and relative <= 0.025 and agreement >= 0.90 and len(layers) == 48
        status = "validation_pass_test_authorized" if passed else "validation_closed"
        overall = False
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        overall = finite and group_balance and relative <= 0.02 and validation["relative_cross_entropy_increase"] <= 0.02 and agreement >= 0.90 and validation["top1_agreement"] >= 0.90
        status = "quality_pass" if overall else "quality_closed"
    gates = {
        "finite": finite,
        "all_48_layers": len(layers) == 48,
        "exact_64_of_128_each_group": group_balance,
        "relative_ce_gate": relative <= (0.025 if args.split == "validation" else 0.02),
        "top1_ge_90pct": agreement >= 0.90,
    }
    result = {
        "kind": "streamq5_moe_p9e1_group_balanced_wanda",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "status": status,
        "inputs": {
            "preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)),
            "calibration_sha256": sha256(CALIBRATION), "evaluation_sha256": sha256(EVAL),
            "masks_sha256": sha256(MASKS), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        },
        "calibration_empty_expert_layer_pairs": empty_experts,
        "teacher_context_ce": teacher_ce.tolist(), "candidate_context_ce": candidate_ce.tolist(),
        "teacher_ce": teacher_mean, "candidate_ce": candidate_mean,
        "relative_cross_entropy_increase": relative, "top1_agreement": agreement,
        "expert_keep_fraction": 0.5, "groups": 6, "keep_per_group": KEEP_PER_GROUP,
        "final_hidden_error": tensor_error(candidate, teacher), "layers": layers,
        "gates": gates, "overall_pass": overall,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "Full-depth quality of a fixed 64-of-128-per-original-group structured-Wanda rule only; no physical bytes, kernel, or runtime claim.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "relative": relative, "top1": agreement, "gates": gates, "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
