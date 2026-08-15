from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.behavioral import rmsnorm
from moe_lab.metrics import topk_overlap
from moe_lab.moe_layer import loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


PROMPT = "Explain in one paragraph why mixture-of-experts routing can be redundant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--report-name", default="cache_aware_kv_rollout.json")
    return parser.parse_args()


def touch_route(cache: list[int], route: list[int], capacity: int) -> int:
    misses = 0
    for expert in route:
        if expert in cache:
            cache.remove(expert)
        else:
            misses += 1
        cache.append(expert)
        if len(cache) > capacity:
            cache.pop(0)
    return misses


def persistent_cache_positions(
    top7_ids: torch.Tensor,
    capacity: int,
    adaptive_cache: list[int],
    strict_cache: list[int],
    strict_reference_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    ids = top7_ids.cpu()
    strict_ids = (
        ids[:, :6]
        if strict_reference_ids is None
        else strict_reference_ids.reshape(-1, 6).cpu()
    )
    positions = torch.arange(6).view(1, 6).expand(ids.shape[0], 6).clone()
    strict_loads = 0
    adaptive_loads = 0
    substitutions = 0
    for token in range(ids.shape[0]):
        original = ids[token, :6].tolist()
        strict_loads += touch_route(
            strict_cache, strict_ids[token].tolist(), capacity
        )
        alternative_route = original.copy()
        alternative = int(ids[token, 6].item())
        alternative_route[-1] = alternative
        original_misses = touch_route(list(adaptive_cache), original, capacity)
        alternative_misses = touch_route(
            list(adaptive_cache), alternative_route, capacity
        )
        if alternative_misses < original_misses:
            chosen = alternative_route
            positions[token, -1] = 6
            substitutions += 1
        else:
            chosen = original
        adaptive_loads += touch_route(adaptive_cache, chosen, capacity)
    return positions.to(top7_ids.device), {
        "strict_expert_loads": strict_loads,
        "adaptive_expert_loads": adaptive_loads,
        "substitutions": substitutions,
    }


@torch.inference_mode()
def student_cache_mlp(
    layer,
    moe_input: torch.Tensor,
    capacity: int,
    adaptive_cache: list[int],
    strict_cache: list[int],
    strict_reference_ids: torch.Tensor | None = None,
):
    moe = loaded_moe_from_official_module(layer.mlp, layer=0)
    flat_input = moe_input.reshape(-1, moe_input.shape[-1])
    scores = F.linear(flat_input.float(), moe.gate_weight.float()).softmax(-1)
    top_weights, top_ids = torch.topk(scores, 7, dim=-1, sorted=True)
    positions, stats = persistent_cache_positions(
        top_ids,
        capacity,
        adaptive_cache,
        strict_cache,
        strict_reference_ids,
    )
    router_ids = top_ids.gather(1, positions)
    router_weights = top_weights.gather(1, positions)
    if moe.norm_topk_prob:
        router_weights = router_weights / router_weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-20)
    else:
        router_weights = router_weights * moe.routed_scaling_factor
    selected = torch.empty(
        flat_input.shape[0],
        6,
        flat_input.shape[-1],
        dtype=flat_input.dtype,
        device=flat_input.device,
    )
    for expert_id, expert in enumerate(moe.experts):
        locations = (router_ids == expert_id).nonzero(as_tuple=False)
        if locations.numel():
            token_indices = locations[:, 0]
            slots = locations[:, 1]
            selected[token_indices, slots] = moe.expert_forward(
                flat_input[token_indices], expert
            )
    routed = (
        selected.float() * router_weights.unsqueeze(-1)
    ).sum(1).to(flat_input.dtype)
    shared = moe.expert_forward(flat_input, moe.shared)
    return (
        (routed + shared).view_as(moe_input),
        router_ids,
        router_weights,
        stats,
    )


@torch.inference_mode()
def forward_cached_layer(
    layer,
    hidden_states: torch.Tensor,
    attention_cache: DynamicCache,
    layer_idx: int,
    capacity: int,
    expert_caches: dict[int, dict[str, list[int]]],
):
    batch, sequence, _ = hidden_states.shape
    past_length = attention_cache.get_seq_length(layer_idx)
    positions = torch.arange(
        past_length, past_length + sequence, device=hidden_states.device
    ).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, past_length
    )
    if mask is None:
        mask = torch.zeros(
            batch,
            1,
            sequence,
            past_length + sequence,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention = layer.self_attn(
        hidden_states=normalized,
        attention_mask=mask,
        position_ids=positions,
        past_key_value=attention_cache,
        output_attentions=False,
        use_cache=True,
    )[0]
    post_attention = residual + attention
    moe_input = layer.post_attention_layernorm(post_attention)
    if layer_idx == 0:
        return post_attention + layer.mlp(moe_input), None, None, None
    teacher_mlp = layer.mlp(moe_input[:1])
    teacher_ids, teacher_weights, _ = layer.mlp.gate(moe_input[:1])
    student_mlp, student_ids, student_weights, stats = student_cache_mlp(
        layer,
        moe_input[1:],
        capacity,
        expert_caches[layer_idx]["adaptive"],
        expert_caches[layer_idx]["strict"],
        teacher_ids,
    )
    combined_mlp = torch.cat((teacher_mlp, student_mlp), dim=0)
    return (
        post_attention + combined_mlp,
        teacher_ids,
        student_ids,
        stats,
    )


@torch.inference_mode()
def final_logits(hidden_states, norm_weight, lm_head):
    return F.linear(rmsnorm(hidden_states[:, -1], norm_weight), lm_head).float()


@torch.inference_mode()
def streamed_step(
    model_dir,
    hidden_states,
    attention_cache,
    capacity,
    expert_caches,
):
    router_overlaps = []
    step_stats = {"strict_expert_loads": 0, "adaptive_expert_loads": 0, "substitutions": 0}
    for layer_idx in range(27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, hidden_states.device)
        hidden_states, teacher_ids, student_ids, stats = forward_cached_layer(
            layer,
            hidden_states,
            attention_cache,
            layer_idx,
            capacity,
            expert_caches,
        )
        if layer_idx:
            router_overlaps.append(
                topk_overlap(student_ids[-1:].cpu(), teacher_ids[-1:].cpu())
            )
            for key in step_stats:
                step_stats[key] += stats[key]
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    return hidden_states, router_overlaps, step_stats


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("KV rollout requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_ids = tokenizer.encode(args.prompt).ids
    embedding = checkpoint_state_for_prefix(model_dir, "model.embed_tokens")[
        "weight"
    ].to(device)
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    teacher_tokens = prompt_ids.copy()
    student_tokens = prompt_ids.copy()
    combined_ids = torch.tensor(
        [teacher_tokens, student_tokens], dtype=torch.long, device=device
    )
    attention_cache = DynamicCache()
    expert_caches = {
        layer: {"adaptive": [], "strict": []} for layer in range(1, 27)
    }
    total_stats = {"strict_expert_loads": 0, "adaptive_expert_loads": 0, "substitutions": 0}
    started = time.perf_counter()
    hidden, _, prefill_stats = streamed_step(
        model_dir,
        embedding[combined_ids],
        attention_cache,
        args.capacity,
        expert_caches,
    )
    for key in total_stats:
        total_stats[key] += prefill_stats[key]
    logits = final_logits(hidden, norm_weight, lm_head)
    steps = []
    for step in range(args.max_new_tokens):
        step_started = time.perf_counter()
        teacher_log_probs = F.log_softmax(logits[0].float(), dim=-1)
        student_log_probs = F.log_softmax(logits[1].float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        kl = float(
            (teacher_probs * (teacher_log_probs - student_log_probs)).sum().item()
        )
        teacher_next = int(logits[0].argmax().item())
        student_next = int(logits[1].argmax().item())
        teacher_tokens.append(teacher_next)
        student_tokens.append(student_next)
        next_ids = torch.tensor(
            [[teacher_next], [student_next]], dtype=torch.long, device=device
        )
        hidden, router_overlaps, step_stats = streamed_step(
            model_dir,
            embedding[next_ids],
            attention_cache,
            args.capacity,
            expert_caches,
        )
        logits = final_logits(hidden, norm_weight, lm_head)
        for key in total_stats:
            total_stats[key] += step_stats[key]
        row = {
            "step": step + 1,
            "teacher_token_id": teacher_next,
            "student_token_id": student_next,
            "teacher_token": tokenizer.decode([teacher_next]),
            "student_token": tokenizer.decode([student_next]),
            "token_agreement": teacher_next == student_next,
            "pre_decision_teacher_to_student_kl": kl,
            "mean_last_token_router_overlap": sum(router_overlaps)
            / len(router_overlaps),
            "minimum_last_token_router_overlap": min(router_overlaps),
            "step_cache": step_stats,
            "wall_seconds": time.perf_counter() - step_started,
        }
        steps.append(row)
        print(row, flush=True)

    total_stats["expert_load_reduction_fraction"] = 1.0 - (
        total_stats["adaptive_expert_loads"] / total_stats["strict_expert_loads"]
    )
    report = {
        "status": "complete",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids,
        "capacity_experts_per_layer": args.capacity,
        "policy": "compare exact within-token LRU misses for original top6 versus top5+rank7; substitute only when the latter has fewer misses",
        "strict_cache_baseline": "unmodified teacher top6 routes with an independent persistent LRU cache",
        "decoding": "greedy independent teacher/student tokens with a persistent DynamicCache KV cache",
        "teacher_text": tokenizer.decode(teacher_tokens),
        "student_text": tokenizer.decode(student_tokens),
        "steps": steps,
        "all_generated_tokens_agree": all(row["token_agreement"] for row in steps),
        "attention_cache_length": attention_cache.get_seq_length(0),
        "total_cache_statistics": total_stats,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        args.report_name,
        envelope("autoregressive_rollout", report),
    )
    print(report)
    print(path)
