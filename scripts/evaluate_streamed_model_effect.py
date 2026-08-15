from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.aggregate_student import ResidualBasisStudent, dense_router_features
from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import load_trace


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
BLOCK_SIZE = 128
BLOCKS = 2
HOT_EXPERTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student",
        choices=(
            "mixed_quant",
            "mixed_quant_all_layers",
            "uniform4_all_layers",
            "uniform3_all_layers",
            "uniform4_late3_8_all_layers",
            "uniform4_late6_8_all_layers",
            "uniform4_early3_8_all_layers",
            "uniform4_early6_8_all_layers",
            "uniform4_edges3_8_all_layers",
            "residual_basis_rank256",
        ),
        default="mixed_quant",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--offset-blocks", type=int, default=0)
    return parser.parse_args()


def corpus_blocks(model_dir: Path, split: str, offset_blocks: int) -> torch.Tensor:
    parquet = (
        ROOT
        / "data"
        / "corpora"
        / "wikitext"
        / "wikitext-2-raw-v1"
        / f"{split}-00000-of-00001.parquet"
    )
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    start = offset_blocks * BLOCK_SIZE
    ids = tokenizer.encode(joined).ids[start : start + BLOCK_SIZE * BLOCKS]
    return torch.tensor(ids, dtype=torch.long).view(BLOCKS, BLOCK_SIZE)


def hot_expert_ids() -> set[int]:
    train = load_trace(ROOT / "data" / "traces" / "wikitext_train_layer_1.safetensors")
    importance = torch.zeros(64, dtype=torch.float64)
    importance.scatter_add_(
        0, train.router_ids.long().reshape(-1), train.router_weights.double().reshape(-1)
    )
    return set(torch.argsort(importance, descending=True)[:HOT_EXPERTS].tolist())


@torch.inference_mode()
def forward_with_router(layer, hidden_states):
    batch, sequence, _ = hidden_states.shape
    position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    captured = []

    def hook(_module, _inputs, output):
        captured.append((output[0].detach().cpu(), output[1].detach().cpu()))

    handle = layer.mlp.gate.register_forward_hook(hook)
    try:
        output = layer(
            hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )[0]
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one router call, received {len(captured)}")
    return output, captured[0][0], captured[0][1]


def quantize_official_layer(layer, hot_ids: set[int]) -> None:
    moe = loaded_moe_from_official_module(layer.mlp, layer=1)
    for expert_id, expert in enumerate(moe.experts):
        bits = 3 if expert_id in hot_ids else 1
        for weight in (expert.gate, expert.up, expert.down):
            fake_quantize_symmetric_per_row_(weight, bits)


def quantize_official_layer_uniform(layer, bits: int) -> None:
    moe = loaded_moe_from_official_module(layer.mlp, layer=1)
    for expert in moe.experts:
        for weight in (expert.gate, expert.up, expert.down):
            fake_quantize_symmetric_per_row_(weight, bits)


def load_residual_student(device: torch.device) -> ResidualBasisStudent:
    model = ResidualBasisStudent(2048, 1408, 64, 256)
    state = load_file(
        ROOT / "data" / "models" / "layer1_residual_basis_rank256.safetensors",
        device="cpu",
    )
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


@torch.inference_mode()
def forward_with_residual_student(layer, hidden_states, student_model):
    batch, sequence, hidden = hidden_states.shape
    position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention_output = layer.self_attn(
        hidden_states=normalized,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )[0]
    post_attention = residual + attention_output
    moe_input = layer.post_attention_layernorm(post_attention)
    router_ids, router_weights, _ = layer.mlp.gate(moe_input)
    routed = student_model(
        moe_input.reshape(-1, hidden), router_ids, router_weights
    ).view(batch, sequence, hidden)
    shared = layer.mlp.shared_experts(moe_input)
    return post_attention + routed + shared, router_ids.cpu(), router_weights.cpu()


@torch.inference_mode()
def final_logits(model_dir: Path, hidden_states: torch.Tensor) -> torch.Tensor:
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device=hidden_states.device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(
        device=hidden_states.device
    )
    normalized = hidden_states.float()
    variance = normalized.pow(2).mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + 1e-6)
    normalized = normalized.to(hidden_states.dtype) * norm_weight
    return F.linear(normalized, lm_head)


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("streamed model evaluation requires CUDA")
    device = torch.device("cuda")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    input_ids = corpus_blocks(model_dir, args.split, args.offset_blocks)
    hidden = load_token_embeddings(model_dir, input_ids, device)

    layer_zero, _ = load_decoder_layer(model_dir, 0, device)
    batch, sequence, _ = hidden.shape
    position_ids = torch.arange(sequence, device=device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(None, (batch, sequence), hidden, 0)
    teacher = layer_zero(
        hidden,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
    )[0]
    student = teacher.clone()
    del layer_zero, hidden
    gc.collect()
    torch.cuda.empty_cache()

    hot_ids = hot_expert_ids()
    all_layer_hot = None
    if args.student == "mixed_quant_all_layers":
        calibration = json.loads(
            (ROOT / "reports" / "baseline" / "router_calibration_all_layers.json").read_text(
                encoding="utf-8"
            )
        )["payload"]
        all_layer_hot = {
            int(row["layer"]): set(row["hot_expert_ids"])
            for row in calibration["layers"]
        }
    residual_student = (
        load_residual_student(device)
        if args.student == "residual_basis_rank256"
        else None
    )
    layer_reports = []
    for layer_idx in range(1, 27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        teacher, teacher_ids, teacher_weights = forward_with_router(layer, teacher)
        if layer_idx == 1 and args.student == "mixed_quant":
            quantize_official_layer(layer, hot_ids)
        if args.student == "mixed_quant_all_layers":
            quantize_official_layer(layer, all_layer_hot[layer_idx])
        if args.student == "uniform4_all_layers":
            quantize_official_layer_uniform(layer, 4)
        if args.student == "uniform3_all_layers":
            quantize_official_layer_uniform(layer, 3)
        if args.student == "uniform4_late3_8_all_layers":
            quantize_official_layer_uniform(layer, 8 if layer_idx >= 24 else 4)
        if args.student == "uniform4_late6_8_all_layers":
            quantize_official_layer_uniform(layer, 8 if layer_idx >= 21 else 4)
        if args.student == "uniform4_early3_8_all_layers":
            quantize_official_layer_uniform(layer, 8 if layer_idx <= 3 else 4)
        if args.student == "uniform4_early6_8_all_layers":
            quantize_official_layer_uniform(layer, 8 if layer_idx <= 6 else 4)
        if args.student == "uniform4_edges3_8_all_layers":
            quantize_official_layer_uniform(
                layer, 8 if layer_idx <= 3 or layer_idx >= 24 else 4
            )
        if layer_idx == 1 and residual_student is not None:
            student, student_ids, student_weights = forward_with_residual_student(
                layer, student, residual_student
            )
            del residual_student
            residual_student = None
        else:
            student, student_ids, student_weights = forward_with_router(layer, student)
        hidden_metrics = regression_metrics(student.cpu(), teacher.cpu())
        layer_report = {
            "layer": layer_idx,
            "hidden": hidden_metrics,
            "router_topk_overlap": topk_overlap(student_ids, teacher_ids),
            "router_exact_set_fraction": float(
                torch.tensor(
                    [
                        set(left) == set(right)
                        for left, right in zip(
                            student_ids.tolist(), teacher_ids.tolist(), strict=True
                        )
                    ],
                    dtype=torch.float32,
                ).mean()
            ),
            "router_weight_nrmse": regression_metrics(
                dense_router_features(student_ids, student_weights, 64),
                dense_router_features(teacher_ids, teacher_weights, 64),
            )["nrmse"],
        }
        layer_reports.append(layer_report)
        print(
            f"layer={layer_idx:02d} hidden_nrmse={hidden_metrics['nrmse']:.6f} "
            f"router_overlap={layer_report['router_topk_overlap']:.6f}",
            flush=True,
        )
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    teacher_logits = final_logits(model_dir, teacher)
    student_logits = final_logits(model_dir, student)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    labels = input_ids[:, 1:].to(device)
    teacher_ce = F.cross_entropy(
        teacher_logits[:, :-1].float().reshape(-1, teacher_logits.shape[-1]),
        labels.reshape(-1),
    )
    student_ce = F.cross_entropy(
        student_logits[:, :-1].float().reshape(-1, student_logits.shape[-1]),
        labels.reshape(-1),
    )
    final = {
        "logits": regression_metrics(student_logits.cpu(), teacher_logits.cpu()),
        "teacher_to_student_kl_mean": float(kl.mean().item()),
        "teacher_to_student_kl_p95": float(torch.quantile(kl, 0.95).item()),
        "top1_token_agreement": float(
            (student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1))
            .float()
            .mean()
            .item()
        ),
        "teacher_next_token_cross_entropy": float(teacher_ce.item()),
        "student_next_token_cross_entropy": float(student_ce.item()),
        "next_token_cross_entropy_delta": float((student_ce - teacher_ce).item()),
    }
    report = {
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "split": args.split,
        "offset_blocks": args.offset_blocks,
        "blocks": BLOCKS,
        "block_size": BLOCK_SIZE,
        "tokens": BLOCKS * BLOCK_SIZE,
        "student_change": (
            "layer 1 routed experts only: train-frequency hot 32 at 3-bit, remaining 32 binary per-row"
            if args.student == "mixed_quant"
            else (
                "all 26 routed expert banks: per-layer train-frequency hot 32 at 3-bit, remaining 32 binary per-row"
                if args.student == "mixed_quant_all_layers"
                else (
                    "all 26 routed expert banks uniformly quantized per-row to 4-bit"
                    if args.student == "uniform4_all_layers"
                    else (
                        "all 26 routed expert banks uniformly quantized per-row to 3-bit"
                        if args.student == "uniform3_all_layers"
                        else (
                            "routed banks at 4-bit except layers 24-26 at 8-bit"
                            if args.student == "uniform4_late3_8_all_layers"
                            else (
                                "routed banks at 4-bit except layers 21-26 at 8-bit"
                                if args.student == "uniform4_late6_8_all_layers"
                                else (
                                    "routed banks at 4-bit except layers 1-3 at 8-bit"
                                    if args.student == "uniform4_early3_8_all_layers"
                                    else (
                                        "routed banks at 4-bit except layers 1-6 at 8-bit"
                                        if args.student == "uniform4_early6_8_all_layers"
                                        else (
                                            "routed banks at 4-bit except layers 1-3 and 24-26 at 8-bit"
                                            if args.student == "uniform4_edges3_8_all_layers"
                                            else "layer 1 routed output replaced by shared SwiGLU plus rank-256 expert-specific residual basis"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "downstream_layers": "original BF16 streamed one layer at a time",
        "teacher_forced": True,
        "layer_reports": layer_reports,
        "final": final,
    }
    suffix = "all_layers" if args.student.endswith("all_layers") else "layer1"
    sample_suffix = (
        ""
        if args.split == "test" and args.offset_blocks == 0
        else f"_{args.split}_offset{args.offset_blocks}"
    )
    path = write_json(
        f"streamed_model_{args.student}_{suffix}{sample_suffix}.json",
        envelope("model_effect", report),
    )
    print(final)
    print(path)
