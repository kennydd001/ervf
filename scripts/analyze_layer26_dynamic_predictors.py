from __future__ import annotations

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
)
from moe_lab.dynamic_precision import (
    best_mask_per_cardinality,
    binary_upgrade_masks,
    discrete_rate_distortion,
    recover_cost_schedule,
)
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT, envelope, write_json


FRACTIONS = (0.15, 0.20, 0.25)
RIDGE = 1e-2


def fixed_budget_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = scores.reshape(-1)
    count = int(fraction * flat.numel())
    mask = torch.zeros(flat.numel(), dtype=torch.bool)
    if count:
        selected = torch.topk(flat, count, sorted=False).indices
        mask[selected] = True
    return mask.view_as(scores)


def oracle_mask_for_budget(
    best_masks: torch.Tensor,
    curve: np.ndarray,
    backpointers: np.ndarray,
    masks: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, int]:
    tokens = best_masks.shape[0]
    budget = int(fraction * tokens * masks.shape[1])
    exact_cost = int(np.argmin(curve[: budget + 1]))
    per_token_cost = torch.from_numpy(
        recover_cost_schedule(backpointers, exact_cost)
    ).long()
    token_indices = torch.arange(tokens)
    mask_indices = best_masks[token_indices, per_token_cost]
    return masks[mask_indices], exact_cost


def continuous_features(
    moe_input: torch.Tensor,
    quant3: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    inputs = moe_input.float()
    outputs = quant3.float()
    weights = router_weights.float()
    input_rms = inputs.square().mean(-1).sqrt().unsqueeze(1).expand_as(weights)
    output_rms = outputs.square().mean(-1).sqrt()
    input_norm = inputs.norm(dim=-1).unsqueeze(1)
    output_norm = outputs.norm(dim=-1)
    cosine = (outputs * inputs.unsqueeze(1)).sum(-1) / (
        input_norm * output_norm
    ).clamp_min(1e-12)
    normalized_weight = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    maximum_weight = weights.max(-1, keepdim=True).values
    order = torch.argsort(weights, dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(weights.shape[1], device=weights.device).expand_as(order),
    )
    return torch.stack(
        (
            weights.log().clamp_min(-30),
            weights,
            normalized_weight,
            maximum_weight - weights,
            input_rms,
            output_rms,
            weights * output_rms,
            cosine,
            ranks.float() / max(weights.shape[1] - 1, 1),
        ),
        dim=-1,
    )


def design_matrix(
    continuous: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    mean: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_continuous = continuous.reshape(-1, continuous.shape[-1]).double()
    if mean is None:
        mean = flat_continuous.mean(0)
    if scale is None:
        scale = flat_continuous.std(0).clamp_min(1e-6)
    standardized = (flat_continuous - mean) / scale
    expert_one_hot = torch.nn.functional.one_hot(
        expert_ids.reshape(-1).long(), num_classes=64
    ).double()
    order = torch.argsort(router_weights, dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(router_weights.shape[1]).expand_as(order))
    rank_one_hot = torch.nn.functional.one_hot(
        ranks.reshape(-1).long(), num_classes=router_weights.shape[1]
    ).double()
    intercept = torch.ones(standardized.shape[0], 1, dtype=torch.float64)
    return (
        torch.cat((intercept, standardized, expert_one_hot, rank_one_hot), dim=1),
        mean,
        scale,
    )


def fit_ridge(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gram = design.T @ design
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype) * RIDGE
    penalty[0, 0] = 0
    return torch.linalg.solve(gram + penalty, design.T @ target.double())


def metrics_for_mask(
    components: dict[str, torch.Tensor],
    split_slice: slice,
    chosen_mask: torch.Tensor,
    teacher_shaped: torch.Tensor,
    token_ids: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
) -> dict[str, float]:
    candidate = candidate_from_schedule(
        components["post_attention"][split_slice],
        components["shared"][split_slice],
        components["selected_quant3"][split_slice],
        components["selected_quant4"][split_slice],
        components["router_weights"][split_slice],
        chosen_mask,
    ).view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
    return behavioral_metrics(
        teacher_shaped, candidate, token_ids, norm_weight, lm_head
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("dynamic predictor analysis requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    component_path = (
        ROOT / "data" / "traces" / "layer26_dynamic_precision_components.safetensors"
    )
    components_cpu = load_file(component_path, device="cpu")
    components = {key: value.to(device) for key, value in components_cpu.items()}
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    split_ids = {
        split: corpus_blocks(model_dir, split, BLOCKS_PER_SPLIT)
        for split in ("validation", "test")
    }
    masks = binary_upgrade_masks(6)
    split_data = {}
    offset = 0
    for split in ("validation", "test"):
        tokens = BLOCKS_PER_SPLIT * BLOCK_SIZE
        sl = slice(offset, offset + tokens)
        exact_damage = exact_kl_all_masks(
            components["teacher"][sl],
            components["post_attention"][sl],
            components["shared"][sl],
            components["selected_quant3"][sl],
            components["selected_quant4"][sl],
            components["router_weights"][sl],
            masks,
            norm_weight,
            lm_head,
        )
        best_damage, best_masks = best_mask_per_cardinality(exact_damage, masks)
        curve, backpointers = discrete_rate_distortion(best_damage.double().numpy())
        q4_target = float(exact_damage[:, -1].mean().item()) * 1.01 * tokens
        qualifying = np.flatnonzero(curve <= q4_target)
        minimum_cost = int(qualifying[0])
        minimum_schedule = torch.from_numpy(
            recover_cost_schedule(backpointers, minimum_cost)
        ).long()
        minimum_mask_indices = best_masks[
            torch.arange(tokens), minimum_schedule
        ]
        minimum_masks = masks[minimum_mask_indices]
        singleton_indices = torch.tensor([1 << slot for slot in range(6)])
        singleton_benefit = (
            exact_damage[:, :1] - exact_damage[:, singleton_indices]
        )
        continuous = continuous_features(
            components["moe_input"][sl],
            components["selected_quant3"][sl],
            components["router_weights"][sl],
        ).cpu()
        split_data[split] = {
            "slice": sl,
            "damage": exact_damage,
            "best_masks": best_masks,
            "curve": curve,
            "backpointers": backpointers,
            "minimum_cost": minimum_cost,
            "minimum_masks": minimum_masks,
            "singleton_benefit": singleton_benefit,
            "continuous": continuous,
        }
        offset += tokens
        print(
            f"{split} exact_min_fraction={minimum_cost / (6 * tokens):.6f}",
            flush=True,
        )

    train = split_data["validation"]
    test = split_data["test"]
    train_slice = train["slice"]
    test_slice = test["slice"]
    train_design, feature_mean, feature_scale = design_matrix(
        train["continuous"],
        components_cpu["router_ids"][train_slice],
        components_cpu["router_weights"][train_slice],
    )
    test_design, _, _ = design_matrix(
        test["continuous"],
        components_cpu["router_ids"][test_slice],
        components_cpu["router_weights"][test_slice],
        feature_mean,
        feature_scale,
    )
    coefficients = fit_ridge(train_design, train["singleton_benefit"].reshape(-1))
    ridge_scores = (test_design @ coefficients).view(-1, 6).float()

    global_benefit = float(train["singleton_benefit"].mean().item())
    expert_benefit = torch.full((64,), global_benefit)
    train_expert_ids = components_cpu["router_ids"][train_slice].long()
    for expert_id in range(64):
        selected = train_expert_ids == expert_id
        if selected.any():
            expert_benefit[expert_id] = train["singleton_benefit"][selected].mean()
    test_expert_ids = components_cpu["router_ids"][test_slice].long()
    expert_prior_scores = expert_benefit[test_expert_ids]
    test_router_weights = components_cpu["router_weights"][test_slice].float()
    delta_energy_scores = (
        (
            components_cpu["selected_quant4"][test_slice].float()
            - components_cpu["selected_quant3"][test_slice].float()
        )
        * test_router_weights.unsqueeze(-1)
    ).square().mean(-1)

    test_teacher = components_cpu["teacher"][test_slice].view(
        BLOCKS_PER_SPLIT, BLOCK_SIZE, -1
    )
    predictor_scores = {
        "router_weight": test_router_weights,
        "validation_expert_prior": expert_prior_scores,
        "calibrated_ridge_available_features": ridge_scores,
        "q4_delta_energy_non_deployable": delta_energy_scores,
        "singleton_kl_benefit_teacher_oracle": test["singleton_benefit"],
    }
    method_rows = {}
    for method, scores in predictor_scores.items():
        rows = []
        for fraction in FRACTIONS:
            chosen = fixed_budget_mask(scores, fraction)
            metrics = metrics_for_mask(
                components,
                test_slice,
                chosen,
                test_teacher,
                split_ids["test"],
                norm_weight,
                lm_head,
            )
            row = {
                "upgrade_fraction": float(chosen.float().mean().item()),
                "average_effective_bits": 3.0 + float(chosen.float().mean().item()),
                **metrics,
            }
            rows.append(row)
            print(
                f"test {method} f={fraction:.2f} "
                f"KL={metrics['teacher_to_candidate_kl']:.6f}",
                flush=True,
            )
        method_rows[method] = rows

    oracle_rows = []
    for fraction in FRACTIONS:
        chosen, exact_cost = oracle_mask_for_budget(
            test["best_masks"],
            test["curve"],
            test["backpointers"],
            masks,
            fraction,
        )
        metrics = metrics_for_mask(
            components,
            test_slice,
            chosen,
            test_teacher,
            split_ids["test"],
            norm_weight,
            lm_head,
        )
        oracle_rows.append(
            {
                "upgrade_fraction": exact_cost / chosen.numel(),
                "average_effective_bits": 3.0 + exact_cost / chosen.numel(),
                **metrics,
            }
        )
    method_rows["exact_mask_oracle"] = oracle_rows

    schedule_stats = {}
    expert_rates = {}
    for split in ("validation", "test"):
        data = split_data[split]
        sl = data["slice"]
        chosen = data["minimum_masks"]
        ids = components_cpu["router_ids"][sl].long()
        weights = components_cpu["router_weights"][sl].float()
        order = torch.argsort(weights, dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(6).expand_as(order))
        rates = torch.zeros(64)
        counts = torch.zeros(64)
        rates.scatter_add_(0, ids.reshape(-1), chosen.float().reshape(-1))
        counts.scatter_add_(0, ids.reshape(-1), torch.ones(ids.numel()))
        rates = rates / counts.clamp_min(1)
        expert_rates[split] = rates
        schedule_stats[split] = {
            "minimum_fraction": data["minimum_cost"] / chosen.numel(),
            "upgrades_per_token_histogram": torch.bincount(
                chosen.sum(1), minlength=7
            ).tolist(),
            "upgrade_fraction_by_router_weight_rank": [
                float(chosen[ranks == rank].float().mean().item())
                for rank in range(6)
            ],
            "router_mass_fraction_upgraded": float(
                ((weights * chosen).sum() / weights.sum()).item()
            ),
            "distinct_experts_ever_upgraded": int(ids[chosen].unique().numel()),
            "expert_upgrade_rates": rates.tolist(),
        }
    expert_rate_correlation = float(
        torch.corrcoef(
            torch.stack((expert_rates["validation"], expert_rates["test"]))
        )[0, 1].item()
    )

    schedule_path = ROOT / "data" / "traces" / "layer26_dynamic_oracle_schedules.safetensors"
    save_file(
        {
            "validation_minimum_mask": split_data["validation"]["minimum_masks"],
            "test_minimum_mask": split_data["test"]["minimum_masks"],
            "validation_exact_mask_kl": split_data["validation"]["damage"],
            "test_exact_mask_kl": split_data["test"]["damage"],
        },
        schedule_path,
        metadata={"selection": "direct layer26 H0 teacher KL over all 64 masks"},
    )
    report = {
        "status": "complete",
        "experiment": "layer26_dynamic_precision_oracle_predictability",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "train_split_for_predictors": "WikiText validation, 1024 tokens",
        "untouched_predictor_test_split": "WikiText test, 1024 tokens",
        "ridge_features": "router weight/rank, expert id, MoE-input statistics, 3-bit expert-output statistics",
        "schedule_statistics": schedule_stats,
        "expert_upgrade_rate_validation_test_correlation": expert_rate_correlation,
        "test_predictor_rate_distortion": method_rows,
        "schedule_artifact": str(schedule_path.resolve()),
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_dynamic_precision_predictors.json",
        envelope("dynamic_precision_predictors", report),
    )
    print(path)
