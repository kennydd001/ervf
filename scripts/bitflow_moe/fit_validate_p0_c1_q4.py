from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import safetensors
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_
from moe_lab.reporting import ROOT


MODEL = ROOT / "models/deepseek-v2-lite"
INPUT_LOCK = ROOT / "reports/bitflow_moe/p0_input_lock.json"
INPUTS = ROOT / "reports/runs/bitflow_moe/p0_input_ids.safetensors"
PREREG = ROOT / "reports/bitflow_moe/P0_C1_Q4_PREREGISTRATION.md"
REPAIR_DIR = ROOT / "reports/runs/bitflow_moe/p0_c1_q4_repairs"
RESULT = ROOT / "reports/bitflow_moe/p0_c1_q4_validation.json"
REPORT = ROOT / "reports/bitflow_moe/P0_C1_Q4_VALIDATION.md"
LOGITS = ROOT / "reports/runs/bitflow_moe/p0_c1_q4_validation_logits.safetensors"
LAMBDAS = (1e-4, 1e-2, 1.0)
HIDDEN = 2048
FEATURES = 4096
EPS = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPS)).to(
        torch.bfloat16
    )


@torch.inference_mode()
def forward_capture(layer, hidden_states: torch.Tensor):
    batch, sequence, _ = hidden_states.shape
    positions = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention = layer.self_attn(
        hidden_states=normalized,
        attention_mask=mask,
        position_ids=positions,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )[0]
    post_attention = residual + attention
    moe_input = layer.post_attention_layernorm(post_attention)
    gate_capture, shared_capture = [], []

    def gate_hook(_module, _inputs, output):
        gate_capture.append((output[0].detach(), output[1].detach()))

    def shared_hook(_module, _inputs, output):
        shared_capture.append(output.detach())

    gate_handle = layer.mlp.gate.register_forward_hook(gate_hook)
    shared_handle = layer.mlp.shared_experts.register_forward_hook(shared_hook)
    try:
        moe_total = layer.mlp(moe_input)
    finally:
        gate_handle.remove()
        shared_handle.remove()
    if len(gate_capture) != 1 or len(shared_capture) != 1:
        raise RuntimeError("expected exactly one official gate and shared-expert call")
    post = post_attention + moe_total
    routed = moe_total - shared_capture[0]
    return post, routed, gate_capture[0][0], gate_capture[0][1]


def quantize_q4(layer) -> None:
    moe = loaded_moe_from_official_module(layer.mlp, layer=1)
    for expert in moe.experts:
        for weight in (expert.gate, expert.up, expert.down):
            fake_quantize_symmetric_per_row_(weight, 4)


@torch.inference_mode()
def layer_zero(model_dir: Path, ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    hidden = load_token_embeddings(model_dir, ids, device)
    layer, _ = load_decoder_layer(model_dir, 0, device)
    batch, sequence, _ = hidden.shape
    positions = torch.arange(sequence, device=device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(None, (batch, sequence), hidden, 0)
    output = layer(
        hidden,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        output_attentions=False,
    )[0]
    del layer, hidden
    gc.collect()
    torch.cuda.empty_cache()
    return output


@torch.inference_mode()
def apply_repair(post: torch.Tensor, routed: torch.Tensor, a, b) -> torch.Tensor:
    return post + F.linear(rmsnorm(post), a) + F.linear(rmsnorm(routed), b)


def contraction_summary(
    teacher_pre: torch.Tensor,
    student_pre: torch.Tensor,
    teacher_post: torch.Tensor,
    student_post: torch.Tensor,
) -> dict[str, object]:
    denominator = (student_pre.float() - teacher_pre.float()).norm(dim=-1).reshape(-1)
    numerator = (student_post.float() - teacher_post.float()).norm(dim=-1).reshape(-1)
    zero = denominator <= 1e-12
    finite = numerator[~zero] / denominator[~zero]
    return {
        "zero_denominators": int(zero.sum()),
        "finite_count": int(finite.numel()),
        "p50": float(torch.quantile(finite, 0.50)) if finite.numel() else None,
        "p95": float(torch.quantile(finite, 0.95)) if finite.numel() else None,
        "maximum": float(finite.max()) if finite.numel() else None,
    }


@torch.inference_mode()
def final_logits(hidden_states: torch.Tensor) -> torch.Tensor:
    norm_weight = checkpoint_state_for_prefix(MODEL, "model.norm")["weight"].to(
        hidden_states.device
    )
    head = checkpoint_state_for_prefix(MODEL, "lm_head")["weight"].to(
        hidden_states.device
    )
    normalized = hidden_states.float()
    normalized *= torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + EPS)
    return F.linear(normalized.to(torch.bfloat16) * norm_weight, head)


def final_metrics(logits: torch.Tensor, teacher: torch.Tensor, ids: torch.Tensor):
    labels = ids[:, 1:].to(logits.device).reshape(-1)
    student_ce = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels
    )
    teacher_ce = F.cross_entropy(
        teacher[:, :-1].float().reshape(-1, teacher.shape[-1]), labels
    )
    teacher_logp = F.log_softmax(teacher.float(), dim=-1)
    student_logp = F.log_softmax(logits.float(), dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    return {
        "next_token_cross_entropy": float(student_ce),
        "teacher_next_token_cross_entropy": float(teacher_ce),
        "next_token_cross_entropy_delta": float(student_ce - teacher_ce),
        "relative_cross_entropy_increase": float((student_ce - teacher_ce) / teacher_ce),
        "top1_token_agreement": float(
            (logits.argmax(-1) == teacher.argmax(-1)).float().mean()
        ),
        "teacher_to_student_kl_mean": float(kl.mean()),
        "teacher_to_student_kl_p95": float(torch.quantile(kl, 0.95)),
        "logits": regression_metrics(logits.cpu(), teacher.cpu()),
    }


if __name__ == "__main__":
    if RESULT.exists() or REPORT.exists() or LOGITS.exists() or REPAIR_DIR.exists():
        raise FileExistsError("refusing to overwrite BITFLOW validation artifacts")
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    if sha256(INPUTS) != lock["artifact_sha256"] or sha256(PREREG) != lock["preregistration_sha256"]:
        raise ValueError("BITFLOW lock hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    inputs = load_file(INPUTS)
    device = torch.device("cuda")
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats(device)
    timer = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    REPAIR_DIR.mkdir(parents=True)

    train_ids = inputs["train"].long()
    val_ids = inputs["validation"].long()
    teacher_train = layer_zero(MODEL, train_ids, device)
    teacher_val = layer_zero(MODEL, val_ids, device)
    student_train = teacher_train.clone()
    student_val = teacher_val.clone()
    baseline_val = teacher_val.clone()
    layers = []
    manifests = {}
    all_controls = True

    for layer_index in range(1, 27):
        layer_timer = time.perf_counter()
        teacher_train_pre, student_train_pre = teacher_train, student_train
        layer, _ = load_decoder_layer(MODEL, layer_index, device)
        teacher_train, _, teacher_train_ids, _ = forward_capture(layer, teacher_train)
        teacher_val, _, teacher_val_ids, _ = forward_capture(layer, teacher_val)
        quantize_q4(layer)

        control_input = student_train_pre[:1, :8]
        sequence = control_input.shape[1]
        positions = torch.arange(sequence, device=device).unsqueeze(0)
        mask = _prepare_4d_causal_attention_mask(None, (1, sequence), control_input, 0)
        official = layer(control_input, attention_mask=mask, position_ids=positions, use_cache=False, output_attentions=False)[0]
        decomposed = forward_capture(layer, control_input)[0]
        control_max_abs = float((official - decomposed).abs().max())
        all_controls &= control_max_abs == 0.0
        del official, decomposed, control_input

        baseline_val, _, _, _ = forward_capture(layer, baseline_val)
        provisional_train, routed_train, train_route_ids, _ = forward_capture(layer, student_train)
        provisional_val, routed_val, val_route_ids, _ = forward_capture(layer, student_val)
        target = (teacher_train.float() - provisional_train.float()).reshape(-1, HIDDEN)
        phi_train = torch.cat((rmsnorm(provisional_train), rmsnorm(routed_train)), dim=-1).reshape(-1, FEATURES)
        phi_val = torch.cat((rmsnorm(provisional_val), rmsnorm(routed_val)), dim=-1).reshape(-1, FEATURES)
        scale = math.sqrt(FEATURES)
        x = phi_train.float() / scale
        xv = phi_val.float() / scale
        kernel = x @ x.T
        cross = xv @ x.T
        identity = torch.eye(kernel.shape[0], device=device, dtype=torch.float32)
        lambda_reports = []
        chosen = None
        for ridge_lambda in LAMBDAS:
            factor, info = torch.linalg.cholesky_ex(kernel + ridge_lambda * identity)
            if int(info.max()) != 0:
                lambda_reports.append({"lambda": ridge_lambda, "status": "cholesky_failed"})
                continue
            alpha = torch.cholesky_solve(target, factor)
            predicted = cross @ alpha
            val_target = (teacher_val.float() - provisional_val.float()).reshape(-1, HIDDEN)
            residual = predicted - val_target
            nrmse = float(torch.sqrt(residual.square().mean()) / torch.sqrt(val_target.square().mean()).clamp_min(1e-20))
            row = {"lambda": ridge_lambda, "status": "complete", "validation_correction_nrmse_fp32_kernel": nrmse}
            lambda_reports.append(row)
            if chosen is None or nrmse < chosen[0]:
                chosen = (nrmse, ridge_lambda, alpha)
        if chosen is None:
            raise RuntimeError(f"all ridge solves failed at layer {layer_index}")
        _, selected_lambda, alpha = chosen
        weight = ((x.T @ alpha) / scale).to(torch.bfloat16)
        a = weight[:HIDDEN].T.contiguous()
        b = weight[HIDDEN:].T.contiguous()
        repair_path = REPAIR_DIR / f"layer_{layer_index:02d}.safetensors"
        save_file(
            {"A": a.cpu(), "B": b.cpu()},
            repair_path,
            metadata={"layer": str(layer_index), "lambda": repr(selected_lambda), "kind": "bitflow_c1_q4_bf16"},
        )
        repaired_train = apply_repair(provisional_train, routed_train, a, b)
        repaired_val = apply_repair(provisional_val, routed_val, a, b)
        local_val_target = teacher_val.float() - provisional_val.float()
        local_val_residual = repaired_val.float() - teacher_val.float()
        selected_bf16_nrmse = float(
            torch.sqrt(local_val_residual.square().mean())
            / torch.sqrt(local_val_target.square().mean()).clamp_min(1e-20)
        )
        correction = repaired_val.float() - provisional_val.float()
        base_hidden = regression_metrics(baseline_val.cpu(), teacher_val.cpu())
        repaired_hidden = regression_metrics(repaired_val.cpu(), teacher_val.cpu())
        layer_report = {
            "layer": layer_index,
            "selected_lambda": selected_lambda,
            "lambda_validation": lambda_reports,
            "selected_bf16_validation_correction_nrmse": selected_bf16_nrmse,
            "baseline_hidden": base_hidden,
            "repaired_hidden": repaired_hidden,
            "router_overlap_train_student_teacher": topk_overlap(train_route_ids.cpu(), teacher_train_ids.cpu()),
            "router_overlap_validation_student_teacher": topk_overlap(val_route_ids.cpu(), teacher_val_ids.cpu()),
            "correction_to_provisional_norm": float(correction.norm() / provisional_val.float().norm().clamp_min(1e-20)),
            "contraction": contraction_summary(teacher_train_pre, student_train_pre, teacher_train, repaired_train),
            "official_decomposition_max_abs": control_max_abs,
            "repair_artifact": str(repair_path.relative_to(ROOT)).replace("\\", "/"),
            "repair_sha256": sha256(repair_path),
            "elapsed_seconds": time.perf_counter() - layer_timer,
        }
        layers.append(layer_report)
        manifests[str(layer_index)] = {"path": layer_report["repair_artifact"], "sha256": layer_report["repair_sha256"]}
        student_train, student_val = repaired_train, repaired_val
        print(json.dumps({"layer": layer_index, "lambda": selected_lambda, "base_nrmse": base_hidden["nrmse"], "repair_nrmse": repaired_hidden["nrmse"], "elapsed_seconds": layer_report["elapsed_seconds"]}), flush=True)
        del layer, routed_train, routed_val, provisional_train, provisional_val, target, phi_train, phi_val, x, xv, kernel, cross, identity, alpha, weight, a, b, correction
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.max_memory_allocated(device) > 7.5 * 2**30 or process.memory_info().rss > 48 * 2**30:
            raise MemoryError("BITFLOW resource ceiling exceeded")

    teacher_logits = final_logits(teacher_val)
    baseline_logits = final_logits(baseline_val)
    repaired_logits = final_logits(student_val)
    teacher_metrics = final_metrics(teacher_logits, teacher_logits, val_ids)
    baseline_metrics = final_metrics(baseline_logits, teacher_logits, val_ids)
    repaired_metrics = final_metrics(repaired_logits, teacher_logits, val_ids)
    damage = baseline_metrics["next_token_cross_entropy"] - teacher_metrics["next_token_cross_entropy"]
    recovery = (baseline_metrics["next_token_cross_entropy"] - repaired_metrics["next_token_cross_entropy"]) / damage if damage > 0 else float("nan")
    early_median = float(torch.tensor([row["repaired_hidden"]["nrmse"] for row in layers if 7 <= row["layer"] <= 20]).median())
    late_max = max(row["repaired_hidden"]["nrmse"] for row in layers if row["layer"] >= 21)
    explosion = late_max / early_median if early_median > 0 else float("inf")
    progression = recovery >= 0.50
    save_file(
        {"input_ids": val_ids.int(), "teacher_logits": teacher_logits.cpu(), "baseline_q4_logits": baseline_logits.cpu(), "repaired_logits": repaired_logits.cpu()},
        LOGITS,
        metadata={"kind": "bitflow_p0_c1_q4_validation_logits"},
    )
    payload = {
        "kind": "bitflow_moe_p0_c1_q4_validation",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "verdict": "validation_progression_pass" if progression else "validation_progression_fail",
        "test_open_authorized": True,
        "full_slate_authorized": False,
        "inputs": {"input_lock_sha256": sha256(INPUT_LOCK), "input_artifact_sha256": sha256(INPUTS), "preregistration_sha256": sha256(PREREG)},
        "candidate": "C1_Q4_two_full_rank_BF16_matrices",
        "repair_artifacts": manifests,
        "repair_parameter_count": 26 * 2 * HIDDEN * HIDDEN,
        "repair_bf16_bytes": 26 * 2 * HIDDEN * HIDDEN * 2,
        "empirical_design_rank_upper_bound": inputs["train"].numel(),
        "layers": layers,
        "final": {"teacher": teacher_metrics, "baseline_q4": baseline_metrics, "repaired": repaired_metrics, "ce_damage_recovery": recovery, "late_layer_explosion_ratio": explosion},
        "gates": {"validation_ce_damage_recovery_ge_0_50": progression, "all_official_decomposition_controls_exact": all_controls},
        "logits_artifact": str(LOGITS.relative_to(ROOT)).replace("\\", "/"),
        "logits_artifact_sha256": sha256(LOGITS),
        "elapsed_seconds": time.perf_counter() - timer,
        "hardware": {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "safetensors": safetensors.__version__, "device": torch.cuda.get_device_name(device), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_process_rss_bytes": process.memory_info().rss},
        "claim_boundary": "Validation only. Test metrics remain unopened; data-limited dense linear C1 screen, not runtime or Qwen proof.",
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text("\n".join([
        "# BITFLOW-MoE P0 C1/Q4 — validation", "",
        f"Uitkomst: **{payload['verdict']}**.", "",
        f"Q4 CE: {baseline_metrics['next_token_cross_entropy']:.6f}; repaired: {repaired_metrics['next_token_cross_entropy']:.6f}; teacher: {teacher_metrics['next_token_cross_entropy']:.6f}.",
        f"Validation CE-schadeherstel: **{recovery:.3%}** (progression gate 50%).",
        f"Top-1 agreement repaired: {repaired_metrics['top1_token_agreement']:.3%}; late-layer explosieratio: {explosion:.4f}.", "",
        "De eerste 256 testtokens zijn nog niet gebruikt. C0/C2/Q3/syndrome blijven gesloten tot test de progression gate eveneens bevestigt.", "",
    ]), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "ce_damage_recovery": recovery, "baseline_ce": baseline_metrics["next_token_cross_entropy"], "repaired_ce": repaired_metrics["next_token_cross_entropy"], "teacher_ce": teacher_metrics["next_token_cross_entropy"], "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
