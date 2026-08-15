from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.metrics import topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_
from moe_lab.reporting import ROOT, envelope, write_json


PROMPT = "The capital of France is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--report-name")
    return parser.parse_args()


@torch.inference_mode()
def forward_layer(layer, hidden_states):
    batch, sequence, _ = hidden_states.shape
    positions = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output[0].detach().cpu())

    handle = layer.mlp.gate.register_forward_hook(hook)
    try:
        output = layer(
            hidden_states,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            output_attentions=False,
        )[0]
    finally:
        handle.remove()
    return output, captured[0]


def quantize_edge_policy(layer, layer_idx: int) -> None:
    bits = 8 if layer_idx <= 3 or layer_idx >= 24 else 4
    moe = loaded_moe_from_official_module(layer.mlp, layer=layer_idx)
    for expert in moe.experts:
        for weight in (expert.gate, expert.up, expert.down):
            fake_quantize_symmetric_per_row_(weight, bits)


@torch.inference_mode()
def logits(model_dir, hidden_states):
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        hidden_states.device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(
        hidden_states.device
    )
    normalized = hidden_states.float()
    normalized *= torch.rsqrt(normalized.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    normalized = normalized.to(hidden_states.dtype) * norm_weight
    return F.linear(normalized[:, -1], lm_head)


@torch.inference_mode()
def next_logits(model_dir, teacher_ids, student_ids, device):
    combined_ids = torch.cat((teacher_ids, student_ids), dim=0)
    hidden = load_token_embeddings(model_dir, combined_ids, device)
    batch, sequence, _ = hidden.shape
    positions = torch.arange(sequence, device=device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(None, (batch, sequence), hidden, 0)
    layer_zero, _ = load_decoder_layer(model_dir, 0, device)
    hidden = layer_zero(
        hidden,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        output_attentions=False,
    )[0]
    teacher, student = hidden[:1], hidden[1:]
    del layer_zero, hidden
    gc.collect()
    torch.cuda.empty_cache()
    last_token_router_overlap = []
    for layer_idx in range(1, 27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        teacher, teacher_router = forward_layer(layer, teacher)
        quantize_edge_policy(layer, layer_idx)
        student, student_router = forward_layer(layer, student)
        overlap = topk_overlap(student_router[-1:], teacher_router[-1:])
        last_token_router_overlap.append(overlap)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    return (
        logits(model_dir, teacher),
        logits(model_dir, student),
        last_token_router_overlap,
    )


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("streamed rollout requires CUDA")
    device = torch.device("cuda")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    previous_wall_seconds = 0.0
    if args.resume_report is not None:
        previous = json.loads(args.resume_report.read_text(encoding="utf-8"))["payload"]
        args.prompt = previous["prompt"]
        prompt_ids = previous["prompt_token_ids"]
        steps = previous["steps"]
        teacher_tokens = prompt_ids + [row["teacher_token_id"] for row in steps]
        student_tokens = prompt_ids + [row["student_token_id"] for row in steps]
        teacher_ids = torch.tensor(teacher_tokens, dtype=torch.long, device=device).unsqueeze(0)
        student_ids = torch.tensor(student_tokens, dtype=torch.long, device=device).unsqueeze(0)
        previous_wall_seconds = float(previous.get("wall_seconds", 0.0))
    else:
        prompt_ids = tokenizer.encode(args.prompt).ids
        teacher_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        student_ids = teacher_ids.clone()
        steps = []
    started = time.perf_counter()
    for step in range(len(steps), args.max_new_tokens):
        step_started = time.perf_counter()
        teacher_logits, student_logits, router_overlap = next_logits(
            model_dir, teacher_ids, student_ids, device
        )
        teacher_next = teacher_logits.argmax(dim=-1)
        student_next = student_logits.argmax(dim=-1)
        teacher_ids = torch.cat((teacher_ids, teacher_next.unsqueeze(1)), dim=1)
        student_ids = torch.cat((student_ids, student_next.unsqueeze(1)), dim=1)
        row = {
            "step": step + 1,
            "teacher_token_id": int(teacher_next.item()),
            "student_token_id": int(student_next.item()),
            "token_agreement": bool(teacher_next.item() == student_next.item()),
            "teacher_token": tokenizer.decode([int(teacher_next.item())]),
            "student_token": tokenizer.decode([int(student_next.item())]),
            "last_token_router_overlap_by_layer": router_overlap,
            "mean_last_token_router_overlap": sum(router_overlap) / len(router_overlap),
            "wall_seconds": time.perf_counter() - step_started,
        }
        steps.append(row)
        print(row, flush=True)
    report = {
        "status": "complete",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids,
        "policy": "layers 1-3 and 24-26 at per-row 8-bit; layers 4-23 at per-row 4-bit",
        "decoding": "greedy, independent teacher/student prefixes, no KV cache",
        "teacher_text": tokenizer.decode(teacher_ids[0].tolist()),
        "student_text": tokenizer.decode(student_ids[0].tolist()),
        "steps": steps,
        "all_generated_tokens_agree": all(row["token_agreement"] for row in steps),
        "wall_seconds": previous_wall_seconds + time.perf_counter() - started,
    }
    report_name = args.report_name or (
        f"streamed_greedy_rollout_edge_policy_{args.max_new_tokens}tokens.json"
    )
    path = write_json(
        report_name,
        envelope("autoregressive_rollout", report),
    )
    print(report)
    print(path)
