from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.behavioral import rmsnorm
from moe_lab.cache_routing import (
    CacheRoutingPolicy,
    parse_policy,
    select_route,
    touch_route,
)
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
    parser.add_argument("--policy", action="append")
    parser.add_argument(
        "--delta-calibration-report",
        default="confirm8_wikitext_mass_budget_pareto.json",
    )
    parser.add_argument(
        "--report-name", default="cache_policy_kv_rollout.json"
    )
    return parser.parse_args()


def load_delta_calibration(name: str) -> dict[int, float]:
    path = ROOT / "reports" / "baseline" / name
    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    return {
        int(row["layer"]): float(row["delta_average_from_validation_teacher"])
        for row in payload["layer_reports"]
    }


def _router(layer, moe_input: torch.Tensor):
    moe = loaded_moe_from_official_module(layer.mlp, layer=0)
    flat = moe_input.reshape(-1, moe_input.shape[-1])
    logits = F.linear(flat.float(), moe.gate_weight.float())
    probabilities = logits.softmax(-1)
    ranked_probabilities, ranked_ids = probabilities.sort(
        dim=-1, descending=True, stable=True
    )
    return moe, logits, probabilities, ranked_probabilities, ranked_ids


@torch.inference_mode()
def policy_mlp(
    layer,
    teacher_input: torch.Tensor,
    student_input: torch.Tensor,
    policy: CacheRoutingPolicy,
    delta_average: float,
    capacity: int,
    caches: dict[str, list[int]],
):
    teacher = _router(layer, teacher_input)
    student = _router(layer, student_input)
    moe, raw_logits, probabilities, ranked_probabilities, ranked_ids = student
    teacher_ranked_ids = teacher[-1]
    sequence = student_input.shape[1]
    chosen_rows = []
    strict_loads = 0
    adaptive_loads = 0
    substitutions = 0
    mass_loss_sum = 0.0
    for token in range(sequence):
        strict_route = teacher_ranked_ids[token, : moe.top_k].tolist()
        strict_loads += touch_route(caches["strict"], strict_route, capacity)
        token_ids = ranked_ids[token].tolist()
        token_probabilities = ranked_probabilities[token].tolist()
        original = token_ids[: moe.top_k]
        chosen = select_route(
            token_ids,
            token_probabilities,
            raw_logits[token].tolist(),
            set(caches["adaptive"]),
            policy,
            delta_average,
            moe.top_k,
        )
        chosen_rows.append(chosen)
        substitutions += int(chosen != original)
        probability_by_expert = dict(
            zip(token_ids, token_probabilities, strict=True)
        )
        mass_loss_sum += max(
            0.0,
            sum(probability_by_expert[expert] for expert in original)
            - sum(probability_by_expert[expert] for expert in chosen),
        )
        original_rank = {expert: rank for rank, expert in enumerate(token_ids)}
        touch_order = sorted(chosen, key=original_rank.__getitem__)
        adaptive_loads += touch_route(caches["adaptive"], touch_order, capacity)
    chosen_ids = torch.tensor(
        chosen_rows, dtype=torch.long, device=student_input.device
    )

    if policy.method == "original":
        output = layer.mlp(student_input)
        _, official_weights, _ = layer.mlp.gate(student_input)
        official_ids, _, _ = layer.mlp.gate(student_input)
        diagnostic_ids = official_ids
        diagnostic_weights = official_weights
    else:
        weights = probabilities.gather(1, chosen_ids)
        if moe.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        else:
            weights = weights * moe.routed_scaling_factor
        flat = student_input.reshape(-1, student_input.shape[-1])
        routed = layer.mlp.moe_infer(flat, chosen_ids, weights).view_as(student_input)
        output = routed + layer.mlp.shared_experts(student_input)
        diagnostic_ids = chosen_ids
        diagnostic_weights = weights
    return output, teacher_ranked_ids[:, : moe.top_k], diagnostic_ids, {
        "strict_expert_loads": strict_loads,
        "adaptive_expert_loads": adaptive_loads,
        "substitutions": substitutions,
        "mean_original_probability_mass_loss": mass_loss_sum / sequence,
    }


@torch.inference_mode()
def forward_cached_layer(
    layer,
    hidden_states: torch.Tensor,
    attention_cache: DynamicCache,
    layer_index: int,
    policies: tuple[CacheRoutingPolicy, ...],
    capacity: int,
    expert_caches: dict[str, dict[int, dict[str, list[int]]]],
    delta_average: float,
):
    batch, sequence, _ = hidden_states.shape
    past_length = attention_cache.get_seq_length(layer_index)
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
    if layer_index == 0:
        return post_attention + layer.mlp(moe_input), {}, {}

    teacher_mlp = layer.mlp(moe_input[:1])
    student_outputs = []
    overlaps = {}
    stats = {}
    for stream_index, policy in enumerate(policies, start=1):
        output, teacher_ids, student_ids, policy_stats = policy_mlp(
            layer,
            moe_input[:1],
            moe_input[stream_index : stream_index + 1],
            policy,
            delta_average,
            capacity,
            expert_caches[policy.name][layer_index],
        )
        student_outputs.append(output)
        overlaps[policy.name] = topk_overlap(
            student_ids[-1:].cpu(), teacher_ids[-1:].cpu()
        )
        stats[policy.name] = policy_stats
    combined_mlp = torch.cat((teacher_mlp, *student_outputs), dim=0)
    return post_attention + combined_mlp, overlaps, stats


@torch.inference_mode()
def final_logits(hidden_states, norm_weight, lm_head):
    return F.linear(rmsnorm(hidden_states[:, -1], norm_weight), lm_head).float()


@torch.inference_mode()
def streamed_step(
    model_dir: Path,
    hidden_states: torch.Tensor,
    attention_cache: DynamicCache,
    policies: tuple[CacheRoutingPolicy, ...],
    capacity: int,
    expert_caches: dict[str, dict[int, dict[str, list[int]]]],
    delta_calibration: dict[int, float],
):
    overlaps = {policy.name: [] for policy in policies}
    step_stats = {
        policy.name: {
            "strict_expert_loads": 0,
            "adaptive_expert_loads": 0,
            "substitutions": 0,
            "probability_mass_loss_sum": 0.0,
            "moe_tokens": 0,
        }
        for policy in policies
    }
    for layer_index in range(27):
        layer, _ = load_decoder_layer(model_dir, layer_index, hidden_states.device)
        sequence = hidden_states.shape[1]
        hidden_states, layer_overlaps, layer_stats = forward_cached_layer(
            layer,
            hidden_states,
            attention_cache,
            layer_index,
            policies,
            capacity,
            expert_caches,
            delta_calibration.get(layer_index, 0.0),
        )
        if layer_index:
            for policy in policies:
                name = policy.name
                overlaps[name].append(layer_overlaps[name])
                for key in (
                    "strict_expert_loads",
                    "adaptive_expert_loads",
                    "substitutions",
                ):
                    step_stats[name][key] += layer_stats[name][key]
                step_stats[name]["probability_mass_loss_sum"] += (
                    layer_stats[name]["mean_original_probability_mass_loss"]
                    * sequence
                )
                step_stats[name]["moe_tokens"] += sequence
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    return hidden_states, overlaps, step_stats


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("KV rollout requires CUDA")
    specifications = tuple(
        args.policy
        or ("original", "cache_prior:j2:0.085", "mass_budget:j2:0.016")
    )
    policies = tuple(parse_policy(specification) for specification in specifications)
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("policies must be unique")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    delta_calibration = load_delta_calibration(args.delta_calibration_report)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_ids = tokenizer.encode(args.prompt).ids
    embedding = checkpoint_state_for_prefix(model_dir, "model.embed_tokens")[
        "weight"
    ].to(device)
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    stream_names = ("teacher", *(policy.name for policy in policies))
    tokens = {name: prompt_ids.copy() for name in stream_names}
    combined_ids = torch.tensor(
        [tokens[name] for name in stream_names], dtype=torch.long, device=device
    )
    attention_cache = DynamicCache()
    expert_caches = {
        policy.name: {
            layer: {"adaptive": [], "strict": []} for layer in range(1, 27)
        }
        for policy in policies
    }
    total_stats = {
        policy.name: {
            "strict_expert_loads": 0,
            "adaptive_expert_loads": 0,
            "substitutions": 0,
            "probability_mass_loss_sum": 0.0,
            "moe_tokens": 0,
        }
        for policy in policies
    }
    started = time.perf_counter()
    hidden, _, prefill_stats = streamed_step(
        model_dir,
        embedding[combined_ids],
        attention_cache,
        policies,
        args.capacity,
        expert_caches,
        delta_calibration,
    )
    for policy in policies:
        for key in total_stats[policy.name]:
            total_stats[policy.name][key] += prefill_stats[policy.name][key]
    logits = final_logits(hidden, norm_weight, lm_head)
    steps = []
    for step_index in range(args.max_new_tokens):
        step_started = time.perf_counter()
        teacher_log_probs = F.log_softmax(logits[0].float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        teacher_next = int(logits[0].argmax().item())
        next_tokens = [teacher_next]
        policy_decisions = {}
        for stream_index, policy in enumerate(policies, start=1):
            student_log_probs = F.log_softmax(logits[stream_index].float(), dim=-1)
            kl = float(
                (
                    teacher_probs
                    * (teacher_log_probs - student_log_probs)
                ).sum().item()
            )
            student_next = int(logits[stream_index].argmax().item())
            next_tokens.append(student_next)
            policy_decisions[policy.name] = {
                "token_id": student_next,
                "token": tokenizer.decode([student_next]),
                "token_agreement": student_next == teacher_next,
                "pre_decision_teacher_to_student_kl": kl,
            }
        for name, token in zip(stream_names, next_tokens, strict=True):
            tokens[name].append(token)
        next_ids = torch.tensor(
            [[token] for token in next_tokens], dtype=torch.long, device=device
        )
        hidden, overlaps, step_stats = streamed_step(
            model_dir,
            embedding[next_ids],
            attention_cache,
            policies,
            args.capacity,
            expert_caches,
            delta_calibration,
        )
        logits = final_logits(hidden, norm_weight, lm_head)
        for policy in policies:
            name = policy.name
            for key in total_stats[name]:
                total_stats[name][key] += step_stats[name][key]
            policy_decisions[name]["mean_last_token_router_overlap"] = sum(
                overlaps[name]
            ) / len(overlaps[name])
            policy_decisions[name]["minimum_last_token_router_overlap"] = min(
                overlaps[name]
            )
            policy_decisions[name]["step_cache"] = step_stats[name]
        row = {
            "step": step_index + 1,
            "teacher_token_id": teacher_next,
            "teacher_token": tokenizer.decode([teacher_next]),
            "policies": policy_decisions,
            "wall_seconds": time.perf_counter() - step_started,
        }
        steps.append(row)
        print(row, flush=True)

    for policy in policies:
        stats = total_stats[policy.name]
        stats["expert_load_reduction_fraction"] = 1.0 - (
            stats["adaptive_expert_loads"] / stats["strict_expert_loads"]
        )
        stats["mean_original_probability_mass_loss"] = (
            stats.pop("probability_mass_loss_sum") / stats["moe_tokens"]
        )
    report = {
        "status": "complete",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids,
        "capacity_experts_per_layer": args.capacity,
        "policies": {
            policy.name: {
                "method": policy.method,
                "top_j": policy.top_j,
                "parameter": policy.parameter,
            }
            for policy in policies
        },
        "delta_calibration_report": str(
            (ROOT / "reports" / "baseline" / args.delta_calibration_report).resolve()
        ),
        "strict_cache_baseline": "unmodified teacher top-6 routes with an independent persistent LRU cache per policy",
        "decoding": "greedy independent teacher/policy prefixes with one batched persistent DynamicCache KV cache",
        "texts": {name: tokenizer.decode(value) for name, value in tokens.items()},
        "steps": steps,
        "all_generated_tokens_agree": {
            policy.name: all(
                row["policies"][policy.name]["token_agreement"] for row in steps
            )
            for policy in policies
        },
        "attention_cache_length": attention_cache.get_seq_length(0),
        "total_cache_statistics": total_stats,
        "scope_caveat": "short greedy smoke; no sampling, long-horizon task score, packed weights, or wall-clock kernel speedup claim",
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        args.report_name, envelope("autoregressive_policy_rollout", report)
    )
    print(report)
    print(path)
