from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow
import safetensors
import torch
import torch.nn.functional as F
import tokenizers
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from evaluate_crcq_oracle import (
    DATASET_REVISION,
    MODEL_REVISION,
    SEED,
    TARGET_MULTIPLIER,
    choices_for_schedule,
    command_result,
    corpus_tokens,
    evaluate_hidden,
    gate_bootstrap,
    git_state,
    hardware_state,
    make_teacher_reference,
    regression_summary,
    selected_precision_outputs,
    sequence_blocks,
    sha256_file,
    solution_json,
    write_json_once,
)
from moe_lab.craft_moe.crcq import (
    best_by_upgrade_count,
    best_schedule_within_fraction,
    local_routed_mean_squared_error,
    mean_gap_closure,
    mixed_precision_routed,
    natural_subset_index,
    routed_for_routes,
    routed_from_choices,
    six_of_twelve_subsets,
    solve_minimum_budget,
)
from moe_lab.dynamic_precision import binary_upgrade_masks
from moe_lab.metrics import topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT


FULL_TOKENS_PER_SPLIT = 256
BLOCK_SIZE = 128
ROUTE_CHUNK = 64
BOOTSTRAP_RESAMPLES = 10_000
POLICY_ORDER = (
    "natural_bf16_patch_control",
    "natural_all_q3",
    "natural_all_q4",
    "natural_minimum_local_q4_quality",
    "joint_minimum_local_q4_quality",
    "natural_matched_joint_minimum_budget",
    "natural_budget_0_15",
    "joint_budget_0_15",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-23 exhaustive local CRCQ selection through exact layers 24-26."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--tokens-per-split", type=int, default=FULL_TOKENS_PER_SPLIT
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--route-chunk", type=int, default=ROUTE_CHUNK)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/craft_moe/crcq_layer23_downstream.json"),
    )
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if args.stage == "smoke" and (
        not 1 <= args.tokens_per_split <= 32 or args.splits != ("validation",)
    ):
        raise ValueError("smoke must use at most 32 validation tokens only")
    if args.stage == "full" and (
        args.tokens_per_split != FULL_TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
    ):
        raise ValueError("the preregistered full layer-23 configuration is immutable")
    if args.route_chunk != ROUTE_CHUNK or args.bootstrap_resamples < 1:
        raise ValueError("route chunk is fixed and bootstrap count must be positive")
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output = output.resolve()
    if (ROOT / "reports").resolve() not in output.parents:
        raise ValueError("output-json must be inside reports/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    args.output_json = output
    return args


@torch.inference_mode()
def forward_layer(layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    batch, sequence, _ = hidden_states.shape
    position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    return layer(
        hidden_states,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
    )[0]


@torch.inference_mode()
def layer_components(
    layer: torch.nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, _ = hidden_states.shape
    position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention = layer.self_attn(
        hidden_states=normalized,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )[0]
    post_attention = residual + attention
    return post_attention, layer.post_attention_layernorm(post_attention)


@torch.inference_mode()
def forward_with_router(
    layer: torch.nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence, _ = hidden_states.shape
    positions = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def hook(_module, _inputs, output):
        captured.append((output[0].detach(), output[1].detach()))

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
    if len(captured) != 1:
        raise RuntimeError("router hook did not capture exactly one call")
    return output, captured[0][0], captured[0][1]


@torch.inference_mode()
def top12_route(
    moe, flat_input: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(flat_input.float(), moe.gate_weight.float()).softmax(dim=-1)
    weights, ids = torch.topk(scores, 12, dim=-1, sorted=True)
    if moe.norm_topk_prob:
        raise RuntimeError("experiment requires the pinned unnormalized router")
    return ids, weights * moe.routed_scaling_factor


@torch.inference_mode()
def full_local_damage(
    *,
    label: str,
    q3: torch.Tensor,
    q4: torch.Tensor,
    weights: torch.Tensor,
    subsets: torch.Tensor,
    masks: torch.Tensor,
    natural_bf16: torch.Tensor,
    device: torch.device,
    route_chunk: int,
) -> torch.Tensor:
    damage = torch.empty(
        q3.shape[0], subsets.shape[0], masks.shape[0], dtype=torch.float32
    )
    masks_device = masks.to(device)
    subsets_device = subsets.to(device)
    for token in range(q3.shape[0]):
        token_q3 = q3[token].to(device)
        token_q4 = q4[token].to(device)
        token_weights = weights[token].to(device)
        target = natural_bf16[token].to(device)
        for route_start in range(0, subsets.shape[0], route_chunk):
            route_stop = min(route_start + route_chunk, subsets.shape[0])
            routed = mixed_precision_routed(
                token_q3,
                token_q4,
                token_weights,
                subsets_device[route_start:route_stop],
                masks_device,
            )
            damage[token, route_start:route_stop] = local_routed_mean_squared_error(
                routed, target
            ).cpu()
        if token % 4 == 3 or token + 1 == q3.shape[0]:
            print(f"local_oracle[{label}]={token + 1}/{q3.shape[0]}", flush=True)
    return damage


def split_shaped_ids(model_dir: Path, split: str, tokens: int) -> torch.Tensor:
    flat = corpus_tokens(model_dir, split, tokens)
    sequence = BLOCK_SIZE if tokens % BLOCK_SIZE == 0 else tokens
    return flat.view(-1, sequence)


def schedule_choices(
    *,
    solution,
    best_route: torch.Tensor,
    best_mask: torch.Tensor,
    route_space: torch.Tensor,
    tokens: int,
    requested_fraction: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if requested_fraction is None:
        if solution.per_token_cost is None or solution.total_cost is None:
            raise RuntimeError("target solution is unreachable")
        schedule = solution.per_token_cost
        cost = solution.total_cost
    else:
        cost, schedule = best_schedule_within_fraction(
            solution, tokens, requested_fraction
        )
    routes, masks = choices_for_schedule(
        schedule, best_route, best_mask, route_space
    )
    return routes, masks, schedule, cost


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("layer-23 CRCQ intervention requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    model_dir = ROOT / "models/deepseek-v2-lite"
    authorization = ROOT / "reports/craft_moe/crcq_full_oracle.json"
    preregistration = ROOT / "reports/craft_moe/H1_CRCQ_LAYER23_PREREGISTRATION.md"
    if not model_dir.is_dir() or not authorization.is_file() or not preregistration.is_file():
        raise FileNotFoundError("model, full-oracle authorization, or preregistration missing")
    with authorization.open("r", encoding="utf-8") as handle:
        # Reading only the small header without loading the 1 GiB raw result is
        # unsafe with ordinary JSON, so validate via a bounded prefix string.
        prefix = handle.read(4096)
    if '"layer23_eligible": true' not in prefix:
        raise RuntimeError("full layer-26 oracle did not authorize layer 23")

    initial_hardware = hardware_state()
    repository = git_state()
    disk_before = psutil.disk_usage(str(ROOT))
    input_hashes = {
        str(preregistration.resolve()): sha256_file(preregistration),
        str((model_dir / "config.json").resolve()): sha256_file(model_dir / "config.json"),
    }
    split_ids = {
        split: split_shaped_ids(model_dir, split, args.tokens_per_split)
        for split in args.splits
    }
    input_ids = torch.cat(tuple(split_ids.values()), dim=0)
    sequence = input_ids.shape[1]
    blocks_per_split = input_ids.shape[0] // len(args.splits)
    total_blocks = input_ids.shape[0]
    total_tokens = input_ids.numel()

    phase = time.perf_counter()
    hidden = load_token_embeddings(model_dir, input_ids, device)
    for layer_index in range(23):
        layer, _ = load_decoder_layer(model_dir, layer_index, device)
        hidden = forward_layer(layer, hidden)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"prefix_layer={layer_index:02d}", flush=True)
    timings["exact_prefix_layers_0_to_22_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    layer23, _ = load_decoder_layer(model_dir, 23, device)
    official_teacher23 = forward_layer(layer23, hidden)
    _, moe_input = layer_components(layer23, hidden)
    moe = loaded_moe_from_official_module(layer23.mlp, layer=23)
    flat_input = moe_input.reshape(-1, moe_input.shape[-1])
    top12_ids, top12_weights = top12_route(moe, flat_input)
    natural_ids, natural_weights = moe.route(flat_input)
    bf16, q3, q4 = selected_precision_outputs(moe, flat_input, top12_ids)
    first6_match = torch.sort(top12_ids[:, :6], dim=1).values == torch.sort(
        natural_ids, dim=1
    ).values
    natural_match = top12_ids[:, :6].unsqueeze(2) == natural_ids.unsqueeze(1)
    natural_positions = natural_match.long().argmax(dim=2)
    aligned_weights = natural_weights.gather(1, natural_positions)
    routing_control = {
        "top6_id_sets_exact": bool(first6_match.all().item()),
        "top6_router_weight_max_abs": float(
            (aligned_weights - top12_weights[:, :6]).abs().max().item()
        ),
    }
    subsets = six_of_twelve_subsets()
    natural_index = natural_subset_index(subsets)
    masks = binary_upgrade_masks(6)
    natural_route = subsets[natural_index].unsqueeze(0)
    natural_bf16 = torch.stack(
        [
            routed_for_routes(bf16[token], top12_weights[token].cpu(), natural_route)[0]
            for token in range(total_tokens)
        ]
    )
    timings["layer23_components_and_precision_outputs_seconds"] = (
        time.perf_counter() - phase
    )
    del layer23, moe, hidden, moe_input, flat_input
    gc.collect()
    torch.cuda.empty_cache()

    split_flat_slices: dict[str, slice] = {}
    offset = 0
    local_results: dict[str, Any] = {}
    split_policy_choices: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    split_solutions: dict[str, Any] = {}
    local_started = time.perf_counter()
    for split_index, split in enumerate(args.splits):
        selected = slice(offset, offset + args.tokens_per_split)
        split_flat_slices[split] = selected
        split_q3 = q3[selected]
        split_q4 = q4[selected]
        split_weights = top12_weights[selected].cpu()
        split_bf16 = natural_bf16[selected]
        damage = full_local_damage(
            label=split,
            q3=split_q3,
            q4=split_q4,
            weights=split_weights,
            subsets=subsets,
            masks=masks,
            natural_bf16=split_bf16,
            device=device,
            route_chunk=args.route_chunk,
        )
        natural_damage = damage[:, natural_index]
        natural_q3_damage = natural_damage[:, 0]
        natural_q4_damage = natural_damage[:, -1]
        alternative = damage[:, :, 0].clone()
        alternative[:, natural_index] = torch.inf
        best_alternative_q3, best_alternative_route = alternative.min(dim=1)
        full_best, full_best_route, full_best_mask = best_by_upgrade_count(
            damage, masks
        )
        natural_best, _, natural_best_mask = best_by_upgrade_count(
            natural_damage.unsqueeze(1), masks
        )
        q4_mean = float(natural_q4_damage.double().mean().item())
        natural_solution = solve_minimum_budget(
            natural_best, q4_mean, tolerance_multiplier=TARGET_MULTIPLIER
        )
        joint_solution = solve_minimum_budget(
            full_best, q4_mean, tolerance_multiplier=TARGET_MULTIPLIER
        )
        if natural_solution.per_token_cost is None or joint_solution.per_token_cost is None:
            raise RuntimeError("local natural all-Q4 target is unreachable")
        natural_space = torch.full(
            (args.tokens_per_split, 1), natural_index, dtype=torch.long
        )
        full_space = torch.arange(subsets.shape[0]).unsqueeze(0).expand(
            args.tokens_per_split, -1
        )
        zero_routes = torch.zeros_like(natural_best_mask)

        natural_min_route, natural_min_mask, _, natural_min_cost = schedule_choices(
            solution=natural_solution,
            best_route=zero_routes,
            best_mask=natural_best_mask,
            route_space=natural_space,
            tokens=args.tokens_per_split,
        )
        joint_min_route, joint_min_mask, _, joint_min_cost = schedule_choices(
            solution=joint_solution,
            best_route=full_best_route,
            best_mask=full_best_mask,
            route_space=full_space,
            tokens=args.tokens_per_split,
        )
        matched_fraction = joint_min_cost / (args.tokens_per_split * 6)
        matched_route, matched_mask, _, matched_cost = schedule_choices(
            solution=natural_solution,
            best_route=zero_routes,
            best_mask=natural_best_mask,
            route_space=natural_space,
            tokens=args.tokens_per_split,
            requested_fraction=matched_fraction,
        )
        natural15_route, natural15_mask, _, natural15_cost = schedule_choices(
            solution=natural_solution,
            best_route=zero_routes,
            best_mask=natural_best_mask,
            route_space=natural_space,
            tokens=args.tokens_per_split,
            requested_fraction=0.15,
        )
        joint15_route, joint15_mask, _, joint15_cost = schedule_choices(
            solution=joint_solution,
            best_route=full_best_route,
            best_mask=full_best_mask,
            route_space=full_space,
            tokens=args.tokens_per_split,
            requested_fraction=0.15,
        )
        natural_routes = torch.full(
            (args.tokens_per_split,), natural_index, dtype=torch.long
        )
        zero_masks = torch.zeros(args.tokens_per_split, dtype=torch.long)
        all_q4_masks = torch.full(
            (args.tokens_per_split,), masks.shape[0] - 1, dtype=torch.long
        )
        split_policy_choices[split] = {
            "natural_bf16_patch_control": (natural_routes, zero_masks),
            "natural_all_q3": (natural_routes, zero_masks),
            "natural_all_q4": (natural_routes, all_q4_masks),
            "natural_minimum_local_q4_quality": (natural_min_route, natural_min_mask),
            "joint_minimum_local_q4_quality": (joint_min_route, joint_min_mask),
            "natural_matched_joint_minimum_budget": (matched_route, matched_mask),
            "natural_budget_0_15": (natural15_route, natural15_mask),
            "joint_budget_0_15": (joint15_route, joint15_mask),
        }
        selected_local = damage[
            torch.arange(args.tokens_per_split), joint_min_route, joint_min_mask
        ].double().mean()
        dp_local = joint_solution.exact_cost_curve[joint_min_cost] / args.tokens_per_split
        if abs(float(selected_local.item()) - float(dp_local)) > 1e-7:
            raise RuntimeError("local joint schedule and DP disagree")
        bootstrap = gate_bootstrap(
            natural_q3=natural_q3_damage,
            natural_q4=natural_q4_damage,
            alternative_q3=best_alternative_q3,
            natural_best=natural_best,
            joint_best=full_best,
            blocks=sequence_blocks(args.tokens_per_split),
            resamples=args.bootstrap_resamples,
            seed=args.seed + split_index,
        )
        bootstrap["intervals"]["joint_full_minimum_upgrade_fraction"] = bootstrap[
            "intervals"
        ].pop("joint_top32_minimum_upgrade_fraction")
        local_results[split] = {
            "raw_full_route_mask_mse": damage.tolist(),
            "best_alternative_all_q3_route": best_alternative_route.tolist(),
            "best_alternative_all_q3_mse": best_alternative_q3.tolist(),
            "full_best_mse_by_upgrade_count": full_best.tolist(),
            "full_best_route_by_upgrade_count": full_best_route.tolist(),
            "full_best_mask_by_upgrade_count": full_best_mask.tolist(),
            "natural_best_mse_by_upgrade_count": natural_best.tolist(),
            "natural_best_mask_by_upgrade_count": natural_best_mask.tolist(),
            "natural_solution": solution_json(natural_solution),
            "joint_solution": solution_json(joint_solution),
            "all_q3_alternative_mean_gap_closure": mean_gap_closure(
                natural_q3_damage, natural_q4_damage, best_alternative_q3
            ),
            "joint_vs_natural_relative_upgrade_reduction": 1.0
            - float(joint_solution.upgrade_fraction)
            / float(natural_solution.upgrade_fraction),
            "local_dp_direct_abs_error": abs(float(selected_local.item()) - float(dp_local)),
            "policy_costs": {
                "natural_minimum": natural_min_cost,
                "joint_minimum": joint_min_cost,
                "natural_matched_joint_budget": matched_cost,
                "natural_budget_0_15": natural15_cost,
                "joint_budget_0_15": joint15_cost,
            },
            "gate_bootstrap_95": bootstrap,
        }
        split_solutions[split] = {
            "natural": natural_solution,
            "joint": joint_solution,
        }
        offset += args.tokens_per_split
    timings["full_local_route_mask_oracle_seconds"] = time.perf_counter() - local_started

    combined_choices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for policy in POLICY_ORDER:
        combined_choices[policy] = (
            torch.cat([split_policy_choices[split][policy][0] for split in args.splits]),
            torch.cat([split_policy_choices[split][policy][1] for split in args.splits]),
        )

    teacher_flat = official_teacher23.reshape(-1, official_teacher23.shape[-1]).cpu()
    candidate_states = {}
    policy_choice_raw = {}
    for policy in POLICY_ORDER:
        routes, mask_indices = combined_choices[policy]
        source_q3 = bf16 if policy == "natural_bf16_patch_control" else q3
        source_q4 = bf16 if policy == "natural_bf16_patch_control" else q4
        routed = routed_from_choices(
            source_q3,
            source_q4,
            top12_weights.cpu(),
            subsets,
            routes,
            masks,
            mask_indices,
        )
        candidate = (
            teacher_flat.float() + (routed.float() - natural_bf16.float())
        ).to(teacher_flat.dtype)
        candidate_states[policy] = candidate.view_as(official_teacher23)
        policy_choice_raw[policy] = {
            "route_index": routes.tolist(),
            "mask_index": mask_indices.tolist(),
            "upgrade_count": masks[mask_indices].sum(dim=1).tolist(),
        }

    phase = time.perf_counter()
    combined = torch.cat(
        [official_teacher23, *(candidate_states[policy].to(device) for policy in POLICY_ORDER)],
        dim=0,
    )
    downstream = []
    tokens_per_policy = total_tokens
    for layer_index in range(24, 27):
        layer, _ = load_decoder_layer(model_dir, layer_index, device)
        combined, router_ids, router_weights = forward_with_router(layer, combined)
        teacher_router_ids = router_ids[:tokens_per_policy]
        teacher_router_weights = router_weights[:tokens_per_policy]
        layer_rows = {}
        for policy_index, policy in enumerate(POLICY_ORDER, start=1):
            token_start = policy_index * tokens_per_policy
            token_stop = (policy_index + 1) * tokens_per_policy
            candidate_ids = router_ids[token_start:token_stop]
            candidate_weights = router_weights[token_start:token_stop]
            candidate_hidden = combined[
                policy_index * total_blocks : (policy_index + 1) * total_blocks
            ]
            teacher_hidden = combined[:total_blocks]
            split_rows = {}
            flat_offset = 0
            for split in args.splits:
                block_start = flat_offset // sequence
                block_stop = block_start + blocks_per_split
                token_slice = slice(flat_offset, flat_offset + args.tokens_per_split)
                split_rows[split] = {
                    "hidden": regression_summary(
                        teacher_hidden[block_start:block_stop].cpu(),
                        candidate_hidden[block_start:block_stop].cpu(),
                    ),
                    "router_top6_overlap": topk_overlap(
                        candidate_ids[token_slice].cpu(),
                        teacher_router_ids[token_slice].cpu(),
                    ),
                    "router_weight_nrmse": regression_summary(
                        teacher_router_weights[token_slice].cpu(),
                        candidate_weights[token_slice].cpu(),
                    )["nrmse"],
                }
                flat_offset += args.tokens_per_split
            layer_rows[policy] = split_rows
        downstream.append({"layer": layer_index, "policies": layer_rows})
        print(f"exact_tail_layer={layer_index}", flush=True)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    timings["exact_tail_layers_24_to_26_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    final_results: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    block_offset = 0
    for split_index, split in enumerate(args.splits):
        block_slice = slice(block_offset, block_offset + blocks_per_split)
        teacher_final = combined[block_slice].reshape(-1, combined.shape[-1]).cpu()
        reference = make_teacher_reference(
            teacher_final,
            split_ids[split].reshape(-1),
            sequence_blocks(args.tokens_per_split),
            norm_weight,
            lm_head,
            128,
        )
        rows = {}
        for policy_index, policy in enumerate(POLICY_ORDER, start=1):
            policy_block_start = policy_index * total_blocks + block_offset
            policy_block_stop = policy_block_start + blocks_per_split
            candidate_final = combined[policy_block_start:policy_block_stop].reshape(
                -1, combined.shape[-1]
            ).cpu()
            rows[policy] = evaluate_hidden(
                candidate_final,
                reference,
                norm_weight,
                lm_head,
                128,
                args.bootstrap_resamples,
                args.seed + split_index,
            )
        control = rows["natural_bf16_patch_control"]
        control_exact = (
            max(control["raw"]["teacher_to_candidate_kl"]) == 0.0
            and all(control["raw"]["top1_agreement"])
            and control["aggregate"]["cross_entropy_delta"] == 0.0
        )
        natural_q4_kl = rows["natural_all_q4"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        joint_min = rows["joint_minimum_local_q4_quality"]["aggregate"]
        natural15_kl = rows["natural_budget_0_15"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        joint15_kl = rows["joint_budget_0_15"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        local_fraction = float(
            split_solutions[split]["joint"].upgrade_fraction
        )
        criteria = {
            "local_joint_upgrade_fraction_le_0_15": local_fraction <= 0.15,
            "joint_min_final_kl_le_1_10x_natural_q4": joint_min[
                "teacher_to_candidate_kl"
            ]
            <= 1.10 * natural_q4_kl,
            "joint_min_abs_relative_ce_lt_0_02": abs(
                joint_min["relative_cross_entropy_delta"]
            )
            < 0.02,
            "joint_15pct_final_kl_le_natural_15pct": joint15_kl <= natural15_kl,
            "bf16_control_exact": control_exact,
        }
        hard_falsification = (
            joint_min["teacher_to_candidate_kl"] > 1.25 * natural_q4_kl
            or abs(joint_min["relative_cross_entropy_delta"]) >= 0.02
            or local_fraction > 0.25
        )
        gates[split] = {
            "criteria": criteria,
            "passed": all(criteria.values()),
            "hard_falsification": hard_falsification,
            "local_joint_upgrade_fraction": local_fraction,
            "joint_min_to_natural_q4_final_kl_ratio": joint_min[
                "teacher_to_candidate_kl"
            ]
            / natural_q4_kl,
            "natural_q4_final_kl": natural_q4_kl,
            "joint_min_final_kl": joint_min["teacher_to_candidate_kl"],
            "natural_15pct_final_kl": natural15_kl,
            "joint_15pct_final_kl": joint15_kl,
        }
        final_results[split] = rows
        block_offset += blocks_per_split
    timings["final_projection_and_metrics_seconds"] = time.perf_counter() - phase

    if args.stage == "smoke":
        verdict = "smoke_passed_not_adjudicated"
    elif all(gates[split]["passed"] for split in args.splits):
        verdict = "downstream_positive"
    elif any(gates[split]["hard_falsification"] for split in args.splits):
        verdict = "downstream_falsified"
    else:
        verdict = "inconclusive"
    timings["total_compute_seconds_before_json"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    disk_after_compute = psutil.disk_usage(str(ROOT))
    report = {
        "schema_version": 1,
        "kind": "craft_moe_crcq_layer23_exact_downstream",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H1_CRCQ_LAYER23_EXACT_TAIL",
        "verdict": verdict,
        "candidate_validation_eligible": verdict == "downstream_positive",
        "preregistration": str(preregistration.resolve()),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "intervention_layer": 23,
            "exact_tail_layers": [24, 25, 26],
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "windows": {
                split: f"first {args.tokens_per_split} tokens" for split in args.splits
            },
            "sequence_length": sequence,
            "blocks_per_split": blocks_per_split,
        },
        "configuration": {
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
            "routes": subsets.shape[0],
            "masks": masks.shape[0],
            "local_candidates_per_token": subsets.shape[0] * masks.shape[0],
            "local_objective": "routed-output mean squared error to natural BF16",
            "local_target_multiplier_vs_natural_all_q4": TARGET_MULTIPLIER,
            "route_chunk": args.route_chunk,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "router_weights_renormalized": False,
            "counterfactual_patch": (
                "BF16(official_teacher23 + candidate_routed - natural_BF16_routed)"
            ),
            "policy_order_after_teacher": list(POLICY_ORDER),
        },
        "route_space": {
            "subsets": subsets.tolist(),
            "upgrade_masks": masks.tolist(),
            "natural_route_index": natural_index,
            "top12_expert_ids": top12_ids.cpu().tolist(),
            "top12_router_weights": top12_weights.cpu().tolist(),
        },
        "routing_control": routing_control,
        "local_oracle": local_results,
        "policy_choices": policy_choice_raw,
        "downstream_layers": downstream,
        "final": final_results,
        "gates": gates,
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "input_hashes": input_hashes,
            "authorization_result": str(authorization.resolve()),
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "safetensors": safetensors.__version__,
                "tokenizers": tokenizers.__version__,
                "pyarrow": pyarrow.__version__,
                "psutil": psutil.__version__,
            },
            "disk": {
                "free_bytes_before": disk_before.free,
                "free_bytes_after_compute": disk_after_compute.free,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "exhaustive local-MSE oracle followed by exact tail, not per-candidate final-KL enumeration",
            "teacher oracle with no cheap selector",
            "256-token existing windows are exploratory, not confirmation",
            "no packed-runtime or wall-clock speedup claim",
        ],
    }
    print("serializing_layer23_raw_json=true", flush=True)
    serialization_started = time.perf_counter()
    write_json_once(args.output_json, report)
    print(
        f"serialization_seconds={time.perf_counter() - serialization_started:.2f}",
        flush=True,
    )
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    for split in args.splits:
        print(f"gate[{split}]={gates[split]}", flush=True)


if __name__ == "__main__":
    main()
