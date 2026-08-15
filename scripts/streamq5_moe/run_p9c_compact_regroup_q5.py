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

import torch
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors, load_qwen_decoder_layer
from scripts.streamq5_moe.run_p0c_model_quality import (
    forward_layer, quantize_groupwise_, quantize_trunk_, selected_embeddings, tensor_error,
)
from scripts.streamq5_moe.run_p9a_mixed_q4_q5 import DOMAINS, head_eval


MODEL = ROOT / "models/qwen3-30b-a3b-base"
DATA = ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors"
DATA_LOCK = ROOT / "reports/streamq5_moe/p0c_input_lock.json"
MASKS = ROOT / "reports/runs/streamq5_moe/p9b_structured_wanda_keep.safetensors"
P9B_VALIDATION = ROOT / "reports/streamq5_moe/p9b_structured_wanda_validation.json"
P9B_TEST = ROOT / "reports/streamq5_moe/p9b_structured_wanda_test.json"
PREREG = ROOT / "reports/streamq5_moe/P9C_COMPACT_REGROUP_Q5_PREREGISTRATION.md"
OUT_DIR = ROOT / "reports/streamq5_moe"
EXPERTS, WIDTH, KEEP = 128, 768, 384


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_masks(mask_tensors: dict[str, torch.Tensor]) -> None:
    for layer in range(48):
        value = mask_tensors[f"layer_{layer:02d}"].long()
        if value.shape != (EXPERTS, KEEP):
            raise RuntimeError(f"mask shape layer {layer}: {tuple(value.shape)}")
        if value.min() < 0 or value.max() >= WIDTH:
            raise RuntimeError(f"mask range layer {layer}")
        if not bool((value.sort(dim=1).values[:, 1:] > value.sort(dim=1).values[:, :-1]).all()):
            raise RuntimeError(f"duplicate mask layer {layer}")


@torch.no_grad()
def compact_quantize_scatter_(layer, keeps: torch.Tensor) -> None:
    device = layer.mlp.experts[0].down_proj.weight.device
    for begin in range(0, EXPERTS, 8):
        experts = layer.mlp.experts[begin:begin + 8]
        index = keeps[begin:begin + len(experts)].long().to(device)
        row_index = index.unsqueeze(-1).expand(-1, -1, 2048)
        for projection in ("gate_proj", "up_proj"):
            original = torch.stack([getattr(expert, projection).weight for expert in experts])
            compact = original.gather(1, row_index).contiguous()
            quantize_groupwise_(compact.reshape(-1, 2048), 5, row_batch=compact.shape[0] * KEEP)
            original.zero_()
            original.scatter_(1, row_index, compact)
            for offset, expert in enumerate(experts):
                getattr(expert, projection).weight.copy_(original[offset])
        original_down = torch.stack([expert.down_proj.weight for expert in experts])
        column_index = index.unsqueeze(1).expand(-1, 2048, -1)
        compact_down = original_down.gather(2, column_index).contiguous()
        quantize_groupwise_(compact_down.reshape(-1, KEEP), 5, row_batch=2048)
        original_down.zero_(); original_down.scatter_(2, column_index, compact_down)
        for offset, expert in enumerate(experts):
            expert.down_proj.weight.copy_(original_down[offset])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    output = OUT_DIR / f"p9c_compact_regroup_q5_{args.split}.json"
    if output.exists():
        raise FileExistsError(output)
    validation_path = OUT_DIR / "p9c_compact_regroup_q5_validation.json"
    if not all(path.exists() for path in (MASKS, P9B_VALIDATION, P9B_TEST)):
        raise RuntimeError("P9B pass artifacts required")
    if not json.loads(P9B_TEST.read_text(encoding="utf-8"))["overall_pass"]:
        raise RuntimeError("P9B quality pass required")
    if args.split == "test":
        if not validation_path.exists() or json.loads(validation_path.read_text(encoding="utf-8"))["status"] != "validation_pass_test_authorized":
            raise RuntimeError("test not authorized")
    lock = json.loads(DATA_LOCK.read_text(encoding="utf-8"))
    if sha256(DATA) != lock["artifact_sha256"]:
        raise ValueError("data provenance mismatch")
    mask_tensors = load_file(MASKS); validate_masks(mask_tensors)
    source = load_file(DATA)
    ids = torch.cat([source[f"{args.split}_{domain}"] for domain in DOMAINS], 0).long()
    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True); config._attn_implementation = "sdpa"
    weight_map = checkpoint_weight_map(MODEL)
    started = time.perf_counter()
    teacher = selected_embeddings(MODEL, ids, device, weight_map, None)
    candidate = selected_embeddings(MODEL, ids, device, weight_map, 8)
    layers = []
    for layer_index in range(48):
        tick = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        teacher = forward_layer(layer, rotary, teacher, device)
        compact_quantize_scatter_(layer, mask_tensors[f"layer_{layer_index:02d}"])
        quantize_trunk_(layer, 8)
        candidate = forward_layer(layer, rotary, candidate, device)
        row = {"layer": layer_index, "hidden_error": tensor_error(candidate, teacher), "seconds": time.perf_counter() - tick}
        layers.append(row); print(json.dumps(row), flush=True)
        del layer, rotary
        gc.collect(); torch.cuda.empty_cache()
    final = load_checkpoint_tensors(MODEL, ["model.norm.weight", "lm_head.weight"], weight_map)
    norm, head = final["model.norm.weight"].to(device), final["lm_head.weight"].to(device)
    teacher_result, teacher_top1, _ = head_eval(teacher, ids, norm, head, device)
    quantize_groupwise_(head, 8, row_batch=256)
    candidate_result, _candidate_top1, agreement = head_eval(candidate, ids, norm, head, device, teacher_top1)
    relative = (candidate_result["ce"] - teacher_result["ce"]) / teacher_result["ce"]
    finite = math.isfinite(relative)
    if args.split == "validation":
        passed = finite and relative <= 0.025 and agreement >= 0.90 and len(layers) == 48
        status = "validation_pass_test_authorized" if passed else "validation_closed"; overall = False
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        overall = finite and relative <= 0.02 and validation["relative_cross_entropy_increase"] <= 0.02 and agreement >= 0.90 and validation["top1_agreement"] >= 0.90
        status = "quality_pass" if overall else "quality_closed"
    gates = {"finite": finite, "all_48_layers": len(layers) == 48, "masks_unique_valid": True,
             "relative_ce_gate": relative <= (0.025 if args.split == "validation" else 0.02), "top1_ge_90pct": agreement >= 0.90}
    result = {
        "kind": "streamq5_moe_p9c_compact_regroup_q5", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "status": status,
        "inputs": {"preregistration_sha256": sha256(PREREG), "data_sha256": sha256(DATA),
                   "masks_sha256": sha256(MASKS), "p9b_validation_sha256": sha256(P9B_VALIDATION),
                   "p9b_test_sha256": sha256(P9B_TEST), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "compact_shapes": {"gate": [384, 2048], "up": [384, 2048], "down": [2048, 384]},
        "expert_weight_ratio": 0.5, "teacher": teacher_result, "candidate": candidate_result,
        "relative_cross_entropy_increase": relative, "top1_agreement": agreement,
        "final_hidden_error": tensor_error(candidate, teacher), "layers": layers,
        "gates": gates, "overall_pass": overall, "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "Full-depth quality of exact compact-regroup Q5 weights; no physical bank or wall-clock claim.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "relative": relative, "top1": agreement, "gates": gates, "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
