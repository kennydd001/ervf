from __future__ import annotations

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from estimate_layer26_observability import behavioral_metrics, corpus_blocks, forward_layer
from moe_lab.behavioral import rmsnorm
from moe_lab.dynamic_precision import (
    best_mask_per_cardinality,
    binary_upgrade_masks,
    discrete_rate_distortion,
    recover_cost_schedule,
)
from moe_lab.metrics import regression_metrics
from moe_lab.moe_layer import (
    LoadedMoELayer,
    ProjectionWeights,
    load_token_embeddings,
    loaded_moe_from_official_module,
)
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_
from moe_lab.reporting import ROOT, envelope, write_json


BLOCKS_PER_SPLIT = 8
BLOCK_SIZE = 128
BUDGET_FRACTIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1 / 3, 0.50, 2 / 3, 0.75, 1.0)


@torch.inference_mode()
def layer_components(layer, hidden_states):
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
    moe_input = layer.post_attention_layernorm(post_attention)
    return post_attention, moe_input


@torch.inference_mode()
def quantized_copy(weights: ProjectionWeights, bits: int) -> ProjectionWeights:
    copied = ProjectionWeights(
        gate=weights.gate.detach().clone(),
        up=weights.up.detach().clone(),
        down=weights.down.detach().clone(),
    )
    for weight in (copied.gate, copied.up, copied.down):
        fake_quantize_symmetric_per_row_(weight, bits)
    return copied


@torch.inference_mode()
def selected_quantized_outputs(
    moe: LoadedMoELayer,
    flat_input: torch.Tensor,
    router_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, slots = router_ids.shape
    hidden = flat_input.shape[-1]
    three_bit = torch.empty(tokens, slots, hidden, dtype=flat_input.dtype, device=flat_input.device)
    four_bit = torch.empty_like(three_bit)
    for expert_id, expert in enumerate(moe.experts):
        positions = (router_ids == expert_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_indices = positions[:, 0]
        slot_indices = positions[:, 1]
        expert_input = flat_input[token_indices]
        quant3 = quantized_copy(expert, 3)
        three_bit[token_indices, slot_indices] = moe.expert_forward(expert_input, quant3)
        del quant3
        quant4 = quantized_copy(expert, 4)
        four_bit[token_indices, slot_indices] = moe.expert_forward(expert_input, quant4)
        del quant4
        if expert_id % 8 == 7:
            print(f"quantized_experts={expert_id + 1}/64", flush=True)
    return three_bit, four_bit


@torch.inference_mode()
def combine_selected(
    post_attention: torch.Tensor,
    shared: torch.Tensor,
    selected: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    routed = (selected.float() * router_weights.unsqueeze(-1)).sum(dim=1)
    routed = routed.to(selected.dtype)
    return post_attention + (routed + shared)


@torch.inference_mode()
def proxy_damage_all_masks(
    teacher: torch.Tensor,
    post_attention: torch.Tensor,
    shared: torch.Tensor,
    three_bit: torch.Tensor,
    four_bit: torch.Tensor,
    router_weights: torch.Tensor,
    gradients: torch.Tensor,
    masks: torch.Tensor,
    token_batch: int = 8,
) -> torch.Tensor:
    damage = torch.empty(teacher.shape[0], masks.shape[0], dtype=torch.float32)
    device = teacher.device
    masks_on_device = masks.to(device)
    for start in range(0, teacher.shape[0], token_batch):
        stop = min(start + token_batch, teacher.shape[0])
        selected = torch.where(
            masks_on_device.view(1, masks.shape[0], masks.shape[1], 1),
            four_bit[start:stop].unsqueeze(1),
            three_bit[start:stop].unsqueeze(1),
        )
        routed = (
            selected.float()
            * router_weights[start:stop].view(stop - start, 1, masks.shape[1], 1)
        ).sum(dim=2).to(three_bit.dtype)
        candidates = post_attention[start:stop].unsqueeze(1) + (
            routed + shared[start:stop].unsqueeze(1)
        )
        error = teacher[start:stop].unsqueeze(1).float() - candidates.float()
        scores = torch.einsum(
            "kbd,bmd->bmk", gradients[:, start:stop], error
        )
        damage[start:stop] = (0.5 * scores.square().mean(dim=-1)).cpu()
    return damage


@torch.inference_mode()
def exact_kl_all_masks(
    teacher: torch.Tensor,
    post_attention: torch.Tensor,
    shared: torch.Tensor,
    three_bit: torch.Tensor,
    four_bit: torch.Tensor,
    router_weights: torch.Tensor,
    masks: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    token_batch: int = 4,
) -> torch.Tensor:
    """Measure direct teacher-to-candidate KL for every 3/4-bit mask."""
    damage = torch.empty(teacher.shape[0], masks.shape[0], dtype=torch.float32)
    masks_on_device = masks.to(teacher.device)
    for start in range(0, teacher.shape[0], token_batch):
        stop = min(start + token_batch, teacher.shape[0])
        batch = stop - start
        selected = torch.where(
            masks_on_device.view(1, masks.shape[0], masks.shape[1], 1),
            four_bit[start:stop].unsqueeze(1),
            three_bit[start:stop].unsqueeze(1),
        )
        routed = (
            selected.float()
            * router_weights[start:stop].view(batch, 1, masks.shape[1], 1)
        ).sum(dim=2).to(three_bit.dtype)
        candidates = post_attention[start:stop].unsqueeze(1) + (
            routed + shared[start:stop].unsqueeze(1)
        )
        teacher_logits = F.linear(
            rmsnorm(teacher[start:stop], norm_weight), lm_head
        ).float()
        candidate_logits = F.linear(
            rmsnorm(candidates.reshape(-1, candidates.shape[-1]), norm_weight),
            lm_head,
        ).float()
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=-1).view(
            batch, masks.shape[0], -1
        )
        kl = (
            teacher_log_probs.exp().unsqueeze(1)
            * (teacher_log_probs.unsqueeze(1) - candidate_log_probs)
        ).sum(dim=-1)
        damage[start:stop] = kl.cpu()
    return damage


@torch.inference_mode()
def candidate_from_schedule(
    post_attention: torch.Tensor,
    shared: torch.Tensor,
    three_bit: torch.Tensor,
    four_bit: torch.Tensor,
    router_weights: torch.Tensor,
    chosen_masks: torch.Tensor,
) -> torch.Tensor:
    selected = torch.where(
        chosen_masks.to(three_bit.device).unsqueeze(-1), four_bit, three_bit
    )
    return combine_selected(post_attention, shared, selected, router_weights)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("dynamic precision oracle requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    split_ids = {
        "train": corpus_blocks(model_dir, "train", 16),
        "validation": corpus_blocks(model_dir, "validation", BLOCKS_PER_SPLIT),
        "test": corpus_blocks(model_dir, "test", BLOCKS_PER_SPLIT),
    }
    combined_ids = torch.cat(
        (split_ids["train"], split_ids["validation"], split_ids["test"]), dim=0
    )
    hidden = load_token_embeddings(model_dir, combined_ids, device)
    for layer_idx in range(26):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        hidden = forward_layer(layer, hidden)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"prefix_layer={layer_idx:02d}", flush=True)

    layer, _ = load_decoder_layer(model_dir, 26, device)
    teacher = forward_layer(layer, hidden)
    post_attention, moe_input = layer_components(layer, hidden)
    moe = loaded_moe_from_official_module(layer.mlp, layer=26)
    flat_input = moe_input.reshape(-1, moe_input.shape[-1])
    router_ids, router_weights = moe.route(flat_input)
    shared = moe.expert_forward(flat_input, moe.shared)
    three_bit, four_bit = selected_quantized_outputs(moe, flat_input, router_ids)
    flat_post_attention = post_attention.reshape(-1, post_attention.shape[-1])
    flat_teacher = teacher.reshape(-1, teacher.shape[-1])
    all_three = combine_selected(flat_post_attention, shared, three_bit, router_weights)
    all_four = combine_selected(flat_post_attention, shared, four_bit, router_weights)

    evaluation_start = 16 * BLOCK_SIZE
    flat_teacher = flat_teacher[evaluation_start:]
    flat_post_attention = flat_post_attention[evaluation_start:]
    flat_input = flat_input[evaluation_start:]
    shared = shared[evaluation_start:]
    router_ids = router_ids[evaluation_start:]
    three_bit = three_bit[evaluation_start:]
    four_bit = four_bit[evaluation_start:]
    router_weights = router_weights[evaluation_start:]
    all_three = all_three[evaluation_start:]
    all_four = all_four[evaluation_start:]

    prior_states = load_file(
        ROOT / "data" / "traces" / "layer26_teacher_quant3_final_states.safetensors",
        device="cpu",
    )
    prior_teacher = torch.cat(
        (prior_states["teacher_validation"], prior_states["teacher_test"]), dim=0
    ).reshape(-1, 2048)
    prior_three = torch.cat(
        (prior_states["quantized_validation"], prior_states["quantized_test"]), dim=0
    ).reshape(-1, 2048)
    reproducibility = {
        "teacher": regression_metrics(flat_teacher.cpu(), prior_teacher),
        "quant3": regression_metrics(all_three.cpu(), prior_three),
        "teacher_max_abs": float(
            (flat_teacher.cpu().float() - prior_teacher.float()).abs().max().item()
        ),
        "quant3_max_abs": float(
            (all_three.cpu().float() - prior_three.float()).abs().max().item()
        ),
    }
    print("reproducibility", reproducibility, flush=True)

    component_path = (
        ROOT / "data" / "traces" / "layer26_dynamic_precision_components.safetensors"
    )
    save_file(
        {
            "teacher": flat_teacher.cpu().contiguous(),
            "post_attention": flat_post_attention.cpu().contiguous(),
            "moe_input": flat_input.cpu().contiguous(),
            "shared": shared.cpu().contiguous(),
            "router_ids": router_ids.cpu().contiguous(),
            "router_weights": router_weights.cpu().contiguous(),
            "selected_quant3": three_bit.cpu().contiguous(),
            "selected_quant4": four_bit.cpu().contiguous(),
        },
        component_path,
        metadata={
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "splits": "validation:8x128,test:8x128",
            "layer": "26",
        },
    )

    del layer, moe, hidden, moe_input, flat_input, prior_states
    gc.collect()
    torch.cuda.empty_cache()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    masks = binary_upgrade_masks(6)
    split_results = {}
    offset = 0
    for split in ("validation", "test"):
        tokens = BLOCKS_PER_SPLIT * BLOCK_SIZE
        sl = slice(offset, offset + tokens)
        split_teacher_flat = flat_teacher[sl]
        split_three = all_three[sl]
        split_four = all_four[sl]
        exact_damage = exact_kl_all_masks(
            split_teacher_flat,
            flat_post_attention[sl],
            shared[sl],
            three_bit[sl],
            four_bit[sl],
            router_weights[sl],
            masks,
            norm_weight,
            lm_head,
        )
        best_damage, best_masks = best_mask_per_cardinality(exact_damage, masks)
        curve, backpointers = discrete_rate_distortion(
            best_damage.detach().double().numpy()
        )
        shaped_teacher = split_teacher_flat.view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
        shaped_three = split_three.view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
        shaped_four = split_four.view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
        baseline_three = behavioral_metrics(
            shaped_teacher, shaped_three, split_ids[split], norm_weight, lm_head
        )
        baseline_four = behavioral_metrics(
            shaped_teacher, shaped_four, split_ids[split], norm_weight, lm_head
        )
        rate_rows = []
        for requested_fraction in BUDGET_FRACTIONS:
            budget = min(int(requested_fraction * 6 * tokens), 6 * tokens)
            exact_cost = int(np.argmin(curve[: budget + 1]))
            per_token_cost = recover_cost_schedule(backpointers, exact_cost)
            token_indices = torch.arange(tokens)
            chosen_mask_indices = best_masks[token_indices, torch.from_numpy(per_token_cost).long()]
            chosen_masks = masks[chosen_mask_indices]
            candidate = candidate_from_schedule(
                flat_post_attention[sl],
                shared[sl],
                three_bit[sl],
                four_bit[sl],
                router_weights[sl],
                chosen_masks,
            ).view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
            metrics = behavioral_metrics(
                shaped_teacher, candidate, split_ids[split], norm_weight, lm_head
            )
            actual_fraction = exact_cost / (6 * tokens)
            row = {
                "requested_upgrade_fraction": requested_fraction,
                "actual_upgrade_fraction": actual_fraction,
                "average_effective_bits": 3.0 + actual_fraction,
                "dynamic_program_kl_mean": float(curve[exact_cost] / tokens),
                "dp_vs_direct_kl_abs_error": abs(
                    float(curve[exact_cost] / tokens)
                    - metrics["teacher_to_candidate_kl"]
                ),
                **metrics,
            }
            rate_rows.append(row)
            print(
                f"{split} upgrades={actual_fraction:.3f} "
                f"KL={metrics['teacher_to_candidate_kl']:.6f} "
                f"CEdelta={metrics['cross_entropy_delta']:.6f}",
                flush=True,
            )
        target_total_kl = (
            baseline_four["teacher_to_candidate_kl"] * 1.01 * tokens
        )
        qualifying_costs = np.flatnonzero(curve <= target_total_kl)
        minimum_cost = int(qualifying_costs[0]) if qualifying_costs.size else None
        split_results[split] = {
            "all_3bit": baseline_three,
            "all_4bit": baseline_four,
            "rate_distortion": rate_rows,
            "minimum_exact_upgrade_fraction_to_match_4bit_kl_within_1pct": (
                minimum_cost / (6 * tokens) if minimum_cost is not None else None
            ),
        }
        offset += tokens
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "status": "complete",
        "experiment": "layer26_token_expert_dynamic_3_to_4bit_oracle",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "blocks_per_split": BLOCKS_PER_SPLIT,
        "block_size": BLOCK_SIZE,
        "tokens_per_split": BLOCKS_PER_SPLIT * BLOCK_SIZE,
        "selection": "direct LM-head KL for all 64 masks per token; exact global cost dynamic program; direct KL/CE evaluation",
        "reproducibility": reproducibility,
        "component_artifact": str(component_path.resolve()),
        "results": split_results,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_dynamic_precision_exact_oracle.json",
        envelope("dynamic_precision_oracle", report),
    )
    print(path)
