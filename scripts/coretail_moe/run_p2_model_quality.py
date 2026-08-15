from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.qwen_gptq_bank.batched_gptq import unpack_2bit_codes
from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_checkpoint_tensors,
    load_qwen_decoder_layer,
    load_token_embeddings,
)


MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = ROOT / "reports/coretail_moe/P2_MODEL_QUALITY_PREREGISTRATION.md"
LOCK = ROOT / "reports/coretail_moe/p2_input_lock.json"
P1_VERIFY = ROOT / "reports/coretail_moe/p1_full_benchmark_verification.json"
BANK_VERIFY = ROOT / "reports/qwen_gptq_bank/p0_full_bank_verification.json"
BANK_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_bank"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
VARIANTS = (
    "bf16_teacher",
    "gptq_experts_bf16_trunk",
    "bf16_experts_int4_trunk",
    "gptq_experts_int4_trunk",
    "gptq_experts_int8_trunk",
)
CONTEXT = 128
EXPERT_BATCH = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def quantize_groupwise_(value: torch.Tensor, bits: int, row_batch: int = 512) -> None:
    if value.ndim != 2 or value.shape[1] % 128:
        raise ValueError(f"matrix is not group-128 compatible: {tuple(value.shape)}")
    qmax = 7 if bits == 4 else 127 if bits == 8 else None
    if qmax is None:
        raise ValueError(bits)
    rows, columns = value.shape
    groups = columns // 128
    for start in range(0, rows, row_batch):
        end = min(rows, start + row_batch)
        work = value[start:end].float().reshape(end - start, groups, 128)
        maximum = work.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
        quantized = torch.round(work / scale).clamp(-qmax, qmax)
        value[start:end].copy_((quantized * scale).reshape(end - start, columns).to(value.dtype))


def trunk_parameters(layer):
    return [
        (name, parameter)
        for name, parameter in layer.named_parameters()
        if parameter.ndim == 2 and ".experts." not in name
    ]


@torch.no_grad()
def quantize_trunk_(layer, bits: int) -> None:
    for _name, parameter in trunk_parameters(layer):
        quantize_groupwise_(parameter, bits)


@torch.no_grad()
def restore_trunk_(layer, originals: dict[str, torch.Tensor]) -> None:
    for name, parameter in trunk_parameters(layer):
        parameter.copy_(originals[name])


@torch.no_grad()
def install_gptq_experts(layer, layer_index: int, device: torch.device) -> None:
    path = BANK_DIR / f"layer_{layer_index:02d}.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in ("gate", "up", "down"):
            packed_all = handle.get_tensor(f"{name}_codes_packed")
            scales_all = handle.get_tensor(f"{name}_scales")
            projection = f"{name}_proj"
            columns = packed_all.shape[-1] * 4
            groups = torch.arange(columns, device=device) // 128
            for start in range(0, packed_all.shape[0], EXPERT_BATCH):
                end = min(packed_all.shape[0], start + EXPERT_BATCH)
                packed = packed_all[start:end].to(device, non_blocking=False)
                scales = scales_all[start:end].to(device, non_blocking=False)
                codes = unpack_2bit_codes(packed)
                dequantized = (
                    codes.float() * scales.float().index_select(-1, groups)
                ).to(torch.bfloat16)
                for offset, expert in enumerate(range(start, end)):
                    getattr(layer.mlp.experts[expert], projection).weight.copy_(dequantized[offset])
                del packed, scales, codes, dequantized
            del packed_all, scales_all


@torch.inference_mode()
def forward_layer(layer, rotary, hidden: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = hidden.to(device)
    position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
    position_embeddings = rotary(batch, position_ids)
    output = layer(
        batch,
        attention_mask=None,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
        output_router_logits=False,
        cache_position=position_ids.squeeze(0),
        position_embeddings=position_embeddings,
    )[0]
    result = output.detach().cpu().contiguous()
    del batch, output, position_embeddings
    return result


def tensor_error(candidate: torch.Tensor, teacher: torch.Tensor) -> dict:
    left = candidate.float()
    right = teacher.float()
    delta = left - right
    return {
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(right).clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
        "finite": bool(torch.isfinite(left).all()),
    }


@torch.no_grad()
def selected_embeddings(
    model_dir: Path,
    input_ids: torch.Tensor,
    device: torch.device,
    weight_map: dict[str, str],
    bits: int | None,
) -> torch.Tensor:
    hidden = load_token_embeddings(model_dir, input_ids, device, weight_map)
    if bits is not None:
        flat = hidden.reshape(-1, hidden.shape[-1])
        quantize_groupwise_(flat, bits)
    return hidden.cpu().contiguous()


@torch.inference_mode()
def head_metrics(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    norm_weight: torch.Tensor,
    head_weight: torch.Tensor,
    device: torch.device,
    teacher_top1: torch.Tensor | None,
) -> tuple[dict, torch.Tensor]:
    domain_rows = {}
    all_loss = 0.0
    all_labels = 0
    top1_parts = []
    for domain_index, domain in enumerate(DOMAINS):
        domain_loss = 0.0
        domain_labels = 0
        domain_agree = 0
        domain_total = 0
        for context_offset in range(2):
            index = domain_index * 2 + context_offset
            state = hidden[index : index + 1].to(device)
            normalized = state.float()
            variance = normalized.pow(2).mean(dim=-1, keepdim=True)
            normalized = (normalized * torch.rsqrt(variance + 1e-6)).to(torch.bfloat16) * norm_weight
            logits = F.linear(normalized, head_weight)
            labels = input_ids[index, 1:].to(device)
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1), reduction="sum",
            )
            predicted = logits[:, :-1].argmax(dim=-1).cpu()
            top1_parts.append(predicted)
            domain_loss += float(loss)
            domain_labels += labels.numel()
            if teacher_top1 is not None:
                reference = teacher_top1[index : index + 1]
                domain_agree += int((predicted == reference).sum())
                domain_total += predicted.numel()
            del state, normalized, logits, labels, loss, predicted
        domain_rows[domain] = {
            "next_token_cross_entropy": domain_loss / domain_labels,
            "labels": domain_labels,
            "top1_agreement_vs_teacher": None if teacher_top1 is None else domain_agree / domain_total,
        }
        all_loss += domain_loss
        all_labels += domain_labels
    top1 = torch.cat(top1_parts, dim=0)
    aggregate_agreement = None if teacher_top1 is None else float((top1 == teacher_top1).float().mean())
    return {
        "next_token_cross_entropy": all_loss / all_labels,
        "labels": all_labels,
        "top1_agreement_vs_teacher": aggregate_agreement,
        "domains": domain_rows,
    }, top1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = ROOT / f"reports/coretail_moe/p2_{args.split}_model_quality.json"
    report = ROOT / f"reports/coretail_moe/P2_{args.split.upper()}_MODEL_QUALITY.md"
    if output.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite P2 {args.split}")
    validation_result = ROOT / "reports/coretail_moe/p2_validation_model_quality.json"
    if args.split == "test" and not validation_result.is_file():
        raise RuntimeError("validation must finish before test opens")
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned transformers 4.51.3 and CUDA required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    p1 = json.loads(P1_VERIFY.read_text(encoding="utf-8"))
    bank = json.loads(BANK_VERIFY.read_text(encoding="utf-8"))
    if p1.get("status") != "p1_verification_pass" or bank.get("status") != "full_bank_pass":
        raise ValueError("P1 and bank passes required")
    if sha256(PREREG) != lock["preregistration_sha256"] or sha256(P1_VERIFY) != lock["p1_verification_sha256"]:
        raise ValueError("P2 lock provenance mismatch")
    input_path = ROOT / lock["artifact"]
    if sha256(input_path) != lock["artifact_sha256"] or sha256(MODEL / "model.safetensors.index.json") != lock["model_index_sha256"]:
        raise ValueError("P2 input or model hash mismatch")
    tensors = load_file(input_path)
    input_ids = torch.cat([tensors[f"{args.split}_{domain}"] for domain in DOMAINS], dim=0)
    for domain in DOMAINS:
        value = tensors[f"{args.split}_{domain}"]
        observed = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        if observed != lock["input_ids_sha256"][f"{args.split}_{domain}"]:
            raise ValueError(f"input tensor hash mismatch for {domain}")

    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) != (48, 128, 8):
        raise RuntimeError("unexpected Qwen configuration")
    weight_map = checkpoint_weight_map(MODEL)
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    bf16_embedding = selected_embeddings(MODEL, input_ids, device, weight_map, None)
    int4_embedding = bf16_embedding.to(device).clone(); quantize_groupwise_(int4_embedding.reshape(-1, int4_embedding.shape[-1]), 4); int4_embedding = int4_embedding.cpu()
    int8_embedding = bf16_embedding.to(device).clone(); quantize_groupwise_(int8_embedding.reshape(-1, int8_embedding.shape[-1]), 8); int8_embedding = int8_embedding.cpu()
    hidden = {
        "bf16_teacher": bf16_embedding,
        "gptq_experts_bf16_trunk": bf16_embedding.clone(),
        "bf16_experts_int4_trunk": int4_embedding,
        "gptq_experts_int4_trunk": int4_embedding.clone(),
        "gptq_experts_int8_trunk": int8_embedding,
    }
    del bf16_embedding
    torch.cuda.empty_cache()
    layer_reports = []

    for layer_index in range(48):
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        hidden["bf16_teacher"] = forward_layer(layer, rotary, hidden["bf16_teacher"], device)
        trunk_original = {name: parameter.detach().clone() for name, parameter in trunk_parameters(layer)}

        quantize_trunk_(layer, 4)
        hidden["bf16_experts_int4_trunk"] = forward_layer(layer, rotary, hidden["bf16_experts_int4_trunk"], device)
        restore_trunk_(layer, trunk_original)

        install_gptq_experts(layer, layer_index, device)
        hidden["gptq_experts_bf16_trunk"] = forward_layer(layer, rotary, hidden["gptq_experts_bf16_trunk"], device)
        quantize_trunk_(layer, 8)
        hidden["gptq_experts_int8_trunk"] = forward_layer(layer, rotary, hidden["gptq_experts_int8_trunk"], device)
        restore_trunk_(layer, trunk_original)
        quantize_trunk_(layer, 4)
        hidden["gptq_experts_int4_trunk"] = forward_layer(layer, rotary, hidden["gptq_experts_int4_trunk"], device)

        errors = {variant: tensor_error(hidden[variant], hidden["bf16_teacher"]) for variant in VARIANTS if variant != "bf16_teacher"}
        layer_reports.append({"layer": layer_index, "errors_vs_teacher": errors, "seconds": time.perf_counter() - layer_started})
        print(json.dumps({"layer": layer_index, "seconds": layer_reports[-1]["seconds"], "relative_l2": {key: value["relative_l2"] for key, value in errors.items()}}), flush=True)
        del layer, rotary, trunk_original
        gc.collect(); torch.cuda.empty_cache()

    final_names = ["model.norm.weight", "lm_head.weight"]
    final_tensors = load_checkpoint_tensors(MODEL, final_names, weight_map)
    norm_weight = final_tensors["model.norm.weight"].to(device)
    head = final_tensors["lm_head.weight"].to(device)
    del final_tensors
    teacher_metrics, teacher_top1 = head_metrics(hidden["bf16_teacher"], input_ids, norm_weight, head, device, None)
    variant_metrics = {"bf16_teacher": teacher_metrics}
    gptq_bf16_metrics, _ = head_metrics(hidden["gptq_experts_bf16_trunk"], input_ids, norm_weight, head, device, teacher_top1)
    variant_metrics["gptq_experts_bf16_trunk"] = gptq_bf16_metrics
    head_original = head.detach().clone()
    quantize_groupwise_(head, 8, row_batch=256)
    gptq_int8_metrics, _ = head_metrics(hidden["gptq_experts_int8_trunk"], input_ids, norm_weight, head, device, teacher_top1)
    variant_metrics["gptq_experts_int8_trunk"] = gptq_int8_metrics
    head.copy_(head_original); quantize_groupwise_(head, 4, row_batch=256)
    bf16_int4_metrics, _ = head_metrics(hidden["bf16_experts_int4_trunk"], input_ids, norm_weight, head, device, teacher_top1)
    gptq_int4_metrics, _ = head_metrics(hidden["gptq_experts_int4_trunk"], input_ids, norm_weight, head, device, teacher_top1)
    variant_metrics["bf16_experts_int4_trunk"] = bf16_int4_metrics
    variant_metrics["gptq_experts_int4_trunk"] = gptq_int4_metrics
    teacher_ce = teacher_metrics["next_token_cross_entropy"]
    for variant, metrics in variant_metrics.items():
        metrics["relative_cross_entropy_increase"] = (metrics["next_token_cross_entropy"] - teacher_ce) / teacher_ce
        for domain in DOMAINS:
            domain_teacher = teacher_metrics["domains"][domain]["next_token_cross_entropy"]
            domain_row = metrics["domains"][domain]
            domain_row["relative_cross_entropy_increase"] = (domain_row["next_token_cross_entropy"] - domain_teacher) / domain_teacher
        if variant != "bf16_teacher":
            metrics["final_hidden_error_vs_teacher"] = tensor_error(hidden[variant], hidden["bf16_teacher"])

    primary_relative = variant_metrics["gptq_experts_int4_trunk"]["relative_cross_entropy_increase"]
    finite = all(
        math.isfinite(metrics["next_token_cross_entropy"])
        and math.isfinite(metrics["relative_cross_entropy_increase"])
        for metrics in variant_metrics.values()
    )
    if args.split == "validation":
        status = "validation_complete_test_authorized" if finite else "validation_nonfinite_stop"
        p2_pass = False
    else:
        validation = json.loads(validation_result.read_text(encoding="utf-8"))
        validation_relative = validation["variants"]["gptq_experts_int4_trunk"]["relative_cross_entropy_increase"]
        if finite and validation_relative <= 0.02 and primary_relative <= 0.02:
            status, p2_pass = "p2_pass", True
        elif finite and primary_relative > 0.02 and primary_relative <= 0.10:
            status, p2_pass = "p2_repair_authorized", False
        elif finite and primary_relative > 0.10:
            status, p2_pass = "p2_quality_closed", False
        else:
            status, p2_pass = "p2_validation_gate_fail", False
    payload = {
        "kind": "coretail_moe_p2_full_depth_model_quality",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc,
        "status": status, "split": args.split,
        "inputs": {"preregistration_sha256": sha256(PREREG), "lock_sha256": sha256(LOCK), "input_artifact_sha256": sha256(input_path), "p1_verification_sha256": sha256(P1_VERIFY), "bank_verification_sha256": sha256(BANK_VERIFY), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "data": {"domains": list(DOMAINS), "contexts_per_domain": 2, "context_tokens": CONTEXT, "labels": teacher_metrics["labels"]},
        "variants": variant_metrics,
        "layer_reports": layer_reports,
        "controls": {"all_finite": finite, "all_48_layers": len(layer_reports) == 48, "test_opened_after_validation": args.split == "validation" or validation_result.is_file(), "primary_variant_fixed": lock["primary_gate"]["variant"] == "gptq_experts_int4_trunk"},
        "primary_relative_ce": primary_relative,
        "p2_pass": p2_pass,
        "p3_repair_authorized": status == "p2_repair_authorized",
        "wall_clock_authorized": p2_pass,
        "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_rss_bytes": process.memory_info().rss},
        "claim_boundary": "Full-depth held-out quality for the exact GPTQ representation and fixed fake-quantized trunks; no integrated CORETAIL wall-clock claim.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text("\n".join([
        f"# CORETAIL-MoE P2 — {args.split} full-depth kwaliteit", "",
        f"Uitkomst: **{status}**.", "",
        f"BF16 CE: {teacher_ce:.6f}. GPTQ+BF16 CE: {variant_metrics['gptq_experts_bf16_trunk']['next_token_cross_entropy']:.6f} ({variant_metrics['gptq_experts_bf16_trunk']['relative_cross_entropy_increase']:.3%}).",
        f"BF16+INT4 CE: {variant_metrics['bf16_experts_int4_trunk']['next_token_cross_entropy']:.6f} ({variant_metrics['bf16_experts_int4_trunk']['relative_cross_entropy_increase']:.3%}).",
        f"GPTQ+INT4 CE: {variant_metrics['gptq_experts_int4_trunk']['next_token_cross_entropy']:.6f} ({primary_relative:.3%}); top-1 {variant_metrics['gptq_experts_int4_trunk']['top1_agreement_vs_teacher']:.3%}.",
        f"GPTQ+INT8 CE: {variant_metrics['gptq_experts_int8_trunk']['next_token_cross_entropy']:.6f} ({variant_metrics['gptq_experts_int8_trunk']['relative_cross_entropy_increase']:.3%}).", "",
        "Deze fase is kwaliteitsisolatie. Geïntegreerde tokens per seconde worden alleen bij een P2-pass geopend.", "",
    ]), encoding="utf-8")
    print(json.dumps({"status": status, "split": args.split, "primary_relative_ce": primary_relative, "variants": {key: {"ce": value["next_token_cross_entropy"], "relative": value["relative_cross_entropy_increase"], "top1": value["top1_agreement_vs_teacher"]} for key, value in variant_metrics.items()}, "runtime": payload["runtime"]}, indent=2), flush=True)
