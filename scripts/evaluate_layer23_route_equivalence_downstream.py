from __future__ import annotations

import gc
import itertools
import time

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from estimate_layer26_observability import behavioral_metrics, corpus_blocks, forward_layer
from evaluate_layer26_dynamic_precision_oracle import layer_components
from evaluate_layer26_route_equivalence import top_candidate_outputs
from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import load_token_embeddings
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


BLOCKS_PER_SPLIT = 2
BLOCK_SIZE = 128


@torch.inference_mode()
def choose_routes(
    weighted_outputs: torch.Tensor,
    subsets: torch.Tensor,
    allowed: torch.Tensor,
    exclude_original: bool,
    token_batch: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    selection = torch.zeros(
        subsets.shape[0], weighted_outputs.shape[1], device=weighted_outputs.device
    )
    selection.scatter_(1, subsets, 1.0)
    original = weighted_outputs[:, :6].float().sum(1)
    allowed_indices = allowed.nonzero(as_tuple=False).squeeze(1)
    original_subset = torch.arange(6, device=subsets.device)
    original_index = int(
        (subsets == original_subset).all(1).nonzero(as_tuple=False).item()
    )
    chosen_indices = []
    chosen_routed = []
    for start in range(0, weighted_outputs.shape[0], token_batch):
        stop = min(start + token_batch, weighted_outputs.shape[0])
        routed = torch.einsum(
            "ms,bsd->bmd",
            selection[allowed_indices],
            weighted_outputs[start:stop].float(),
        )
        damage = (routed - original[start:stop].unsqueeze(1)).square().mean(-1)
        if exclude_original:
            original_positions = (allowed_indices == original_index).nonzero(
                as_tuple=False
            )
            if original_positions.numel():
                damage[:, original_positions.item()] = float("inf")
        local = damage.argmin(1)
        chosen_indices.append(allowed_indices[local].cpu())
        chosen_routed.append(routed[torch.arange(stop - start), local].to(weighted_outputs.dtype))
    return torch.cat(chosen_indices), torch.cat(chosen_routed)


@torch.inference_mode()
def routed_for_fixed_subset(
    weighted_outputs: torch.Tensor, positions: tuple[int, ...]
) -> torch.Tensor:
    return weighted_outputs[:, list(positions)].float().sum(1).to(weighted_outputs.dtype)


@torch.inference_mode()
def forward_with_router(layer, hidden_states):
    batch, sequence, _ = hidden_states.shape
    positions = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    captured = []

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
    return output, captured[0][0], captured[0][1]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("downstream route experiment requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    split_ids = {
        split: corpus_blocks(model_dir, split, BLOCKS_PER_SPLIT)
        for split in ("validation", "test")
    }
    input_ids = torch.cat((split_ids["validation"], split_ids["test"]), dim=0)
    hidden = load_token_embeddings(model_dir, input_ids, device)
    for layer_idx in range(23):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        hidden = forward_layer(layer, hidden)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"prefix_layer={layer_idx:02d}", flush=True)

    layer23, _ = load_decoder_layer(model_dir, 23, device)
    official_teacher23 = forward_layer(layer23, hidden)
    post_attention, moe_input = layer_components(layer23, hidden)
    top_ids, top_weights, outputs = top_candidate_outputs(
        layer23, moe_input.reshape(-1, moe_input.shape[-1])
    )
    flat_post = post_attention.reshape(-1, post_attention.shape[-1])
    flat_shared = layer23.mlp.shared_experts(moe_input).reshape(-1, moe_input.shape[-1])
    weighted_outputs = outputs.float() * top_weights.unsqueeze(-1)
    original_routed = weighted_outputs[:, :6].sum(1).to(outputs.dtype)
    manual_teacher23 = flat_post + (original_routed + flat_shared)
    reconstruction = regression_metrics(
        manual_teacher23.cpu(), official_teacher23.reshape(-1, 2048).cpu()
    )
    subsets = torch.tensor(
        list(itertools.combinations(range(12), 6)), dtype=torch.long, device=device
    )
    intersections = (subsets < 6).sum(1)
    all_allowed = torch.ones(subsets.shape[0], dtype=torch.bool, device=device)
    low_overlap_allowed = intersections <= 4
    best_any_indices, best_any_routed = choose_routes(
        weighted_outputs, subsets, all_allowed, exclude_original=True
    )
    best_low_indices, best_low_routed = choose_routes(
        weighted_outputs, subsets, low_overlap_allowed, exclude_original=False
    )
    best_any_routed = best_any_routed.to(outputs.dtype)
    best_low_routed = best_low_routed.to(outputs.dtype)
    bottom_two_swap_routed = routed_for_fixed_subset(
        weighted_outputs, (0, 1, 2, 3, 6, 7)
    ).to(outputs.dtype)
    policy_routed = {
        "best_euclidean_alternative_oracle": best_any_routed,
        "best_euclidean_jaccard_le_0_5_oracle": best_low_routed,
        "replace_bottom_two_with_rank7_8": bottom_two_swap_routed,
    }
    policy_hidden23 = {
        name: flat_post + (routed + flat_shared)
        for name, routed in policy_routed.items()
    }
    local = {}
    for name, candidate in policy_hidden23.items():
        local[name] = regression_metrics(candidate.cpu(), manual_teacher23.cpu())

    policy_names = tuple(policy_hidden23)
    original_shaped = manual_teacher23.view_as(official_teacher23)
    combined = torch.cat(
        (
            original_shaped,
            *(policy_hidden23[name].view_as(official_teacher23) for name in policy_names),
        ),
        dim=0,
    )
    batch_per_policy = input_ids.shape[0]
    downstream = []
    for layer_idx in range(24, 27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        combined, router_ids, router_weights = forward_with_router(layer, combined)
        teacher_ids = router_ids[: batch_per_policy * BLOCK_SIZE]
        teacher_weights = router_weights[: batch_per_policy * BLOCK_SIZE]
        layer_rows = {}
        for policy_index, name in enumerate(policy_names, start=1):
            token_start = policy_index * batch_per_policy * BLOCK_SIZE
            token_stop = (policy_index + 1) * batch_per_policy * BLOCK_SIZE
            candidate_ids = router_ids[token_start:token_stop]
            candidate_weights = router_weights[token_start:token_stop]
            teacher_hidden = combined[:batch_per_policy]
            candidate_hidden = combined[
                policy_index * batch_per_policy : (policy_index + 1) * batch_per_policy
            ]
            layer_rows[name] = {
                "hidden": regression_metrics(
                    candidate_hidden.cpu(), teacher_hidden.cpu()
                ),
                "router_topk_overlap": topk_overlap(candidate_ids.cpu(), teacher_ids.cpu()),
                "router_weight_nrmse": regression_metrics(
                    candidate_weights.cpu(), teacher_weights.cpu()
                )["nrmse"],
            }
        downstream.append({"layer": layer_idx, "policies": layer_rows})
        print(f"downstream_layer={layer_idx:02d}", flush=True)
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    final_results = {}
    for split_index, split in enumerate(("validation", "test")):
        block_start = split_index * BLOCKS_PER_SPLIT
        block_stop = (split_index + 1) * BLOCKS_PER_SPLIT
        teacher_final = combined[block_start:block_stop].cpu()
        split_rows = {}
        for policy_index, name in enumerate(policy_names, start=1):
            policy_start = policy_index * batch_per_policy + block_start
            policy_stop = policy_index * batch_per_policy + block_stop
            candidate_final = combined[policy_start:policy_stop].cpu()
            split_rows[name] = behavioral_metrics(
                teacher_final,
                candidate_final,
                split_ids[split],
                norm_weight,
                lm_head,
            )
        final_results[split] = split_rows

    original_set = set(range(6))
    selection_stats = {
        "best_any_mean_jaccard": float(
            torch.tensor(
                [
                    len(original_set & set(subsets[index].tolist()))
                    / len(original_set | set(subsets[index].tolist()))
                    for index in best_any_indices
                ]
            ).mean()
        ),
        "best_low_overlap_mean_jaccard": float(
            torch.tensor(
                [
                    len(original_set & set(subsets[index].tolist()))
                    / len(original_set | set(subsets[index].tolist()))
                    for index in best_low_indices
                ]
            ).mean()
        ),
    }
    report = {
        "status": "complete",
        "experiment": "layer23_route_equivalence_through_layers24_to26",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "blocks_per_split": BLOCKS_PER_SPLIT,
        "manual_layer23_reference_vs_official": reconstruction,
        "selection": "natural raw top12 router weights; no renormalization; local Euclidean route-output oracle",
        "selection_statistics": selection_stats,
        "local_layer23_hidden": local,
        "downstream_layers": downstream,
        "final": final_results,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer23_route_equivalence_downstream.json",
        envelope("route_equivalence_downstream", report),
    )
    print(path)
    print(final_results)
