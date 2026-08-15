from __future__ import annotations

import gc
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from estimate_layer26_observability import behavioral_metrics, corpus_blocks
from evaluate_layer26_dynamic_precision_oracle import (
    BLOCK_SIZE,
    BLOCKS_PER_SPLIT,
    candidate_from_schedule,
    exact_kl_all_masks,
    quantized_copy,
)
from moe_lab.dynamic_precision import (
    best_mask_per_cardinality,
    binary_upgrade_masks,
    discrete_rate_distortion,
    recover_cost_schedule,
)
from moe_lab.moe_layer import loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


FRACTIONS = (0.0, 0.10, 0.15, 0.20, 0.25, 1 / 3, 0.50, 2 / 3, 0.75, 1.0)


@torch.inference_mode()
def selected_two_bit_outputs(layer, moe_input, router_ids):
    moe = loaded_moe_from_official_module(layer.mlp, layer=26)
    output = torch.empty(
        router_ids.shape[0],
        router_ids.shape[1],
        moe_input.shape[-1],
        dtype=moe_input.dtype,
        device=moe_input.device,
    )
    for expert_id, expert in enumerate(moe.experts):
        positions = (router_ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            quant2 = quantized_copy(expert, 2)
            output[token_indices, slot_indices] = moe.expert_forward(
                moe_input[token_indices], quant2
            )
            del quant2
        if expert_id % 8 == 7:
            print(f"quant2_experts={expert_id + 1}/64", flush=True)
    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("2-to-4-bit oracle requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    component_path = (
        ROOT / "data" / "traces" / "layer26_dynamic_precision_components.safetensors"
    )
    components_cpu = load_file(component_path, device="cpu")
    components = {key: value.to(device) for key, value in components_cpu.items()}
    layer, _ = load_decoder_layer(model_dir, 26, device)
    quant2 = selected_two_bit_outputs(
        layer, components["moe_input"], components["router_ids"].long()
    )
    del layer
    gc.collect()
    torch.cuda.empty_cache()
    quant2_path = ROOT / "data" / "traces" / "layer26_selected_quant2.safetensors"
    save_file(
        {"selected_quant2": quant2.cpu().contiguous()},
        quant2_path,
        metadata={"layer": "26", "quantization": "symmetric per-row 2-bit"},
    )

    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    masks = binary_upgrade_masks(6)
    split_ids = {
        split: corpus_blocks(model_dir, split, BLOCKS_PER_SPLIT)
        for split in ("validation", "test")
    }
    results = {}
    offset = 0
    for split in ("validation", "test"):
        tokens = BLOCKS_PER_SPLIT * BLOCK_SIZE
        sl = slice(offset, offset + tokens)
        damage = exact_kl_all_masks(
            components["teacher"][sl],
            components["post_attention"][sl],
            components["shared"][sl],
            quant2[sl],
            components["selected_quant4"][sl],
            components["router_weights"][sl],
            masks,
            norm_weight,
            lm_head,
        )
        best_damage, best_masks = best_mask_per_cardinality(damage, masks)
        curve, backpointers = discrete_rate_distortion(best_damage.double().numpy())
        teacher_shaped = components_cpu["teacher"][sl].view(
            BLOCKS_PER_SPLIT, BLOCK_SIZE, -1
        )
        rows = []
        for fraction in FRACTIONS:
            budget = int(fraction * tokens * 6)
            cost = int(np.argmin(curve[: budget + 1]))
            token_cost = torch.from_numpy(
                recover_cost_schedule(backpointers, cost)
            ).long()
            mask_indices = best_masks[torch.arange(tokens), token_cost]
            chosen_masks = masks[mask_indices]
            candidate = candidate_from_schedule(
                components["post_attention"][sl],
                components["shared"][sl],
                quant2[sl],
                components["selected_quant4"][sl],
                components["router_weights"][sl],
                chosen_masks,
            ).view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
            metrics = behavioral_metrics(
                teacher_shaped,
                candidate,
                split_ids[split],
                norm_weight,
                lm_head,
            )
            actual_fraction = cost / (tokens * 6)
            rows.append(
                {
                    "requested_upgrade_fraction": fraction,
                    "actual_upgrade_fraction": actual_fraction,
                    "average_active_bits": 2.0 + 2.0 * actual_fraction,
                    **metrics,
                }
            )
            print(
                f"{split} f={actual_fraction:.3f} "
                f"bits={2 + 2 * actual_fraction:.3f} "
                f"KL={metrics['teacher_to_candidate_kl']:.6f}",
                flush=True,
            )
        q4_mean = float(damage[:, -1].mean().item())
        qualifying = np.flatnonzero(curve <= q4_mean * 1.01 * tokens)
        minimum_cost = int(qualifying[0])
        results[split] = {
            "all_2bit_kl": float(damage[:, 0].mean().item()),
            "all_4bit_kl": q4_mean,
            "minimum_exact_upgrade_fraction_to_match_4bit_kl_within_1pct": minimum_cost
            / (tokens * 6),
            "minimum_average_active_bits": 2.0 + 2.0 * minimum_cost / (tokens * 6),
            "rate_distortion": rows,
        }
        offset += tokens

    report = {
        "status": "complete",
        "experiment": "layer26_token_expert_dynamic_2_to_4bit_exact_oracle",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "selection": "direct LM-head KL over all 64 masks and exact global cost dynamic program",
        "results": results,
        "quant2_artifact": str(quant2_path.resolve()),
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_dynamic_2to4_exact_oracle.json",
        envelope("dynamic_precision_oracle", report),
    )
    print(path)
