from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_checkpoint_tensors,
    load_qwen_decoder_layer,
)
from scripts.streamq5_moe.run_p0c_model_quality import (
    quantize_experts_q5_,
    quantize_groupwise_,
    quantize_trunk_,
    selected_embeddings,
    tensor_error,
)


MODEL = ROOT / "models/qwen3-30b-a3b-base"
CALIBRATION = ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_input_ids.safetensors"
CALIBRATION_LOCK = ROOT / "reports/qwen_gptq_bank/p0_input_lock.json"
EVAL = ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors"
EVAL_LOCK = ROOT / "reports/streamq5_moe/p0c_input_lock.json"
PREREG = ROOT / "reports/streamq5_moe/P9B_STRUCTURED_WANDA_PRUNING_PREREGISTRATION.md"
MASKS = ROOT / "reports/runs/streamq5_moe/p9b_structured_wanda_keep.safetensors"
OUT_DIR = ROOT / "reports/streamq5_moe"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
CONTEXT = 128
EXPERTS = 128
WIDTH = 768
KEEP = 384
CHUNK = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forward_chunks(layer, rotary, hidden: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts = []
    for start in range(0, hidden.shape[0], CHUNK):
        batch = hidden[start:start + CHUNK].to(device)
        positions = torch.arange(batch.shape[1], device=device).unsqueeze(0)
        position_embeddings = rotary(batch, positions)
        result = layer(
            batch, attention_mask=None, position_ids=positions, use_cache=False,
            output_attentions=False, output_router_logits=False,
            cache_position=positions.squeeze(0), position_embeddings=position_embeddings,
        )[0]
        parts.append(result.detach().cpu().contiguous())
        del batch, positions, position_embeddings, result
    return torch.cat(parts, dim=0)


@torch.no_grad()
def fit_keep_indices(layer, rotary, calibration_hidden, device):
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
    for index, expert in enumerate(layer.mlp.experts):
        if counts[index] == 0:
            empty += 1
        rms = torch.sqrt(sums[index] / counts[index].clamp_min(1))
        column_norm = torch.linalg.vector_norm(expert.down_proj.weight.float(), dim=0)
        selected = torch.topk(rms * column_norm + tie, KEEP, largest=True, sorted=False).indices
        keeps[index] = selected.sort().values.to(torch.int16).cpu()
    return next_hidden, keeps, empty


@torch.no_grad()
def apply_structured_pruning_(layer, keeps: torch.Tensor) -> None:
    for index, expert in enumerate(layer.mlp.experts):
        mask = torch.zeros(WIDTH, dtype=torch.bool, device=expert.down_proj.weight.device)
        mask[keeps[index].long().to(mask.device)] = True
        expert.gate_proj.weight[~mask].zero_()
        expert.up_proj.weight[~mask].zero_()
        expert.down_proj.weight[:, ~mask].zero_()


@torch.inference_mode()
def evaluate_head(hidden, ids, norm, head, device, reference=None):
    losses, predictions = [], []
    for index in range(hidden.shape[0]):
        state = hidden[index:index + 1].to(device).float()
        state = (state * torch.rsqrt(state.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * norm
        logits = F.linear(state, head)
        labels = ids[index, 1:].to(device)
        losses.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1))))
        predictions.append(logits[:, :-1].argmax(-1).cpu())
        del state, logits, labels
    prediction = torch.cat(predictions, 0)
    agreement = None if reference is None else float((prediction == reference).float().mean())
    return np.asarray(losses, dtype=np.float64), prediction, agreement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    output = OUT_DIR / f"p9b_structured_wanda_{args.split}.json"
    if output.exists():
        raise FileExistsError(output)
    validation_path = OUT_DIR / "p9b_structured_wanda_validation.json"
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
    ids = torch.cat([eval_source[f"{args.split}_{d}"] for d in DOMAINS], 0).long().contiguous()
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
        calibration_ids = torch.cat([calibration_source[d][:10, :CONTEXT] for d in DOMAINS], 0).long()
        calibration_hidden = selected_embeddings(MODEL, calibration_ids, device, weight_map, None)
    else:
        calibration_hidden = None

    layers, empty_experts = [], 0
    for layer_index in range(48):
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        if args.split == "validation":
            calibration_hidden, keeps, empty = fit_keep_indices(layer, rotary, calibration_hidden, device)
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
        layers.append(row); print(json.dumps(row), flush=True)
        del layer, rotary
        gc.collect(); torch.cuda.empty_cache()
    if args.split == "validation":
        save_file(mask_tensors, MASKS, metadata={"kind": "p9b_structured_wanda_keep384", "preregistration_sha256": sha256(PREREG)})

    final = load_checkpoint_tensors(MODEL, ["model.norm.weight", "lm_head.weight"], weight_map)
    norm, head = final["model.norm.weight"].to(device), final["lm_head.weight"].to(device)
    teacher_ce, teacher_top1, _ = evaluate_head(teacher, ids, norm, head, device)
    quantize_groupwise_(head, 8, row_batch=256)
    candidate_ce, _candidate_top1, agreement = evaluate_head(candidate, ids, norm, head, device, teacher_top1)
    teacher_mean, candidate_mean = float(teacher_ce.mean()), float(candidate_ce.mean())
    relative = (candidate_mean - teacher_mean) / teacher_mean
    finite = bool(np.isfinite(candidate_ce).all())
    if args.split == "validation":
        passed = finite and relative <= 0.025 and agreement >= 0.90 and len(layers) == 48
        status = "validation_pass_test_authorized" if passed else "validation_closed"
        overall = False
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        overall = finite and relative <= 0.02 and validation["relative_cross_entropy_increase"] <= 0.02 and agreement >= 0.90 and validation["top1_agreement"] >= 0.90
        status = "quality_pass" if overall else "quality_closed"
    gates = {
        "finite": finite,
        "all_48_layers": len(layers) == 48,
        "exact_keep_fraction_50pct": all(mask_tensors[f"layer_{i:02d}"].shape == (EXPERTS, KEEP) for i in range(48)),
        "relative_ce_gate": relative <= (0.025 if args.split == "validation" else 0.02),
        "top1_ge_90pct": agreement >= 0.90,
    }
    result = {
        "kind": "streamq5_moe_p9b_structured_wanda_pruning",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "status": status,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "calibration_sha256": sha256(CALIBRATION),
            "evaluation_sha256": sha256(EVAL),
            "masks_sha256": sha256(MASKS),
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        },
        "calibration_empty_expert_layer_pairs": empty_experts,
        "teacher_context_ce": teacher_ce.tolist(), "candidate_context_ce": candidate_ce.tolist(),
        "teacher_ce": teacher_mean, "candidate_ce": candidate_mean,
        "relative_cross_entropy_increase": relative, "top1_agreement": agreement,
        "expert_keep_fraction": 0.5, "expert_weight_byte_projection_ratio": 0.5,
        "final_hidden_error": tensor_error(candidate, teacher), "layers": layers,
        "gates": gates, "overall_pass": overall,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "Full-depth quality of one fixed activation-aware structured-neuron pruning rule; compact physical kernel/timing is opened only by a quality pass.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "relative": relative, "top1": agreement, "empty_experts": empty_experts, "gates": gates, "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
