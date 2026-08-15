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


MODEL = ROOT / "models/qwen3-30b-a3b-base"
DATA = ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors"
DATA_LOCK = ROOT / "reports/streamq5_moe/p0c_input_lock.json"
PREREG = ROOT / "reports/streamq5_moe/P9A_MIXED_Q4_Q5_PREREGISTRATION.md"
OUT_DIR = ROOT / "reports/streamq5_moe"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
Q4_LAYERS = frozenset((4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def quantize_experts_(layer, bits: int) -> None:
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for start in range(0, len(layer.mlp.experts), 8):
            experts = layer.mlp.experts[start:start + 8]
            weights = [getattr(expert, projection).weight for expert in experts]
            shape = weights[0].shape
            work = torch.stack(weights).reshape(-1, shape[1])
            quantize_groupwise_(work, bits, row_batch=work.shape[0])
            work = work.reshape(len(weights), *shape)
            for offset, weight in enumerate(weights):
                weight.copy_(work[offset])


@torch.inference_mode()
def head_eval(hidden, ids, norm, head, device, reference=None):
    total_loss, labels = 0.0, 0
    predictions = []
    domains = {}
    for domain_index, domain in enumerate(DOMAINS):
        domain_loss, domain_labels = 0.0, 0
        for offset in range(2):
            index = domain_index * 2 + offset
            state = hidden[index:index + 1].to(device).float()
            state = (state * torch.rsqrt(state.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * norm
            logits = F.linear(state, head)
            target = ids[index, 1:].to(device)
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum")
            domain_loss += float(loss); domain_labels += target.numel()
            predictions.append(logits[:, :-1].argmax(-1).cpu())
            del state, logits, target, loss
        domains[domain] = {"ce": domain_loss / domain_labels, "labels": domain_labels}
        total_loss += domain_loss; labels += domain_labels
    predicted = torch.cat(predictions, 0)
    agreement = None if reference is None else float((predicted == reference).float().mean())
    return {"ce": total_loss / labels, "labels": labels, "domains": domains}, predicted, agreement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    output = OUT_DIR / f"p9a_mixed_q4_q5_{args.split}.json"
    if output.exists():
        raise FileExistsError(output)
    validation_path = OUT_DIR / "p9a_mixed_q4_q5_validation.json"
    if args.split == "test":
        if not validation_path.exists() or json.loads(validation_path.read_text(encoding="utf-8"))["status"] != "validation_pass_test_authorized":
            raise RuntimeError("test not authorized")
    lock = json.loads(DATA_LOCK.read_text(encoding="utf-8"))
    if sha256(DATA) != lock["artifact_sha256"]:
        raise ValueError("data provenance mismatch")
    source = load_file(DATA)
    ids = torch.cat([source[f"{args.split}_{d}"] for d in DOMAINS], 0).long()
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
        bits = 4 if layer_index in Q4_LAYERS else 5
        quantize_experts_(layer, bits); quantize_trunk_(layer, 8)
        candidate = forward_layer(layer, rotary, candidate, device)
        row = {"layer": layer_index, "bits": bits, "hidden_error": tensor_error(candidate, teacher), "seconds": time.perf_counter() - tick}
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
    byte_ratio = (len(Q4_LAYERS) * 4 + (48 - len(Q4_LAYERS)) * 5) / (48 * 5)
    gates = {"finite": finite, "all_48_layers": len(layers) == 48, "relative_ce_gate": relative <= (0.025 if args.split == "validation" else 0.02), "top1_ge_90pct": agreement >= 0.90}
    result = {
        "kind": "streamq5_moe_p9a_mixed_q4_q5", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "status": status,
        "inputs": {"preregistration_sha256": sha256(PREREG), "data_sha256": sha256(DATA), "data_lock_sha256": sha256(DATA_LOCK), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "q4_layers": sorted(Q4_LAYERS), "q5_layers": [x for x in range(48) if x not in Q4_LAYERS],
        "expert_code_byte_ratio_vs_uniform_q5": byte_ratio,
        "teacher": teacher_result, "candidate": candidate_result,
        "relative_cross_entropy_increase": relative, "top1_agreement": agreement,
        "final_hidden_error": tensor_error(candidate, teacher), "layers": layers,
        "gates": gates, "overall_pass": overall, "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": "One fixed per-layer Q4/Q5 quality candidate; physical mixed bank and wall-clock remain unopened until quality passes.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": status, "relative": relative, "top1": agreement, "byte_ratio": byte_ratio, "gates": gates, "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
