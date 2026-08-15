from __future__ import annotations

import time

import numpy as np
import torch
from safetensors.torch import load_file

from moe_lab.dynamic_precision import (
    best_mask_per_cardinality,
    binary_upgrade_masks,
    discrete_rate_distortion,
    recover_cost_schedule,
)
from moe_lab.reporting import ROOT, envelope, write_json


FRACTIONS = (0.15, 0.20, 0.25)
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TOKENS_PER_SPLIT = 1024


def fit_pca(train: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.float().mean(0)
    torch.manual_seed(20260809 + rank)
    _, _, basis = torch.pca_lowrank(
        train.float() - mean, q=rank, center=False, niter=6
    )
    return mean, basis


def pca_features(
    values: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    return (values.float() - mean) @ basis


def token_features(
    components: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    train_slice = slice(0, TOKENS_PER_SPLIT)
    test_slice = slice(TOKENS_PER_SPLIT, 2 * TOKENS_PER_SPLIT)
    train_input = components["moe_input"][train_slice]
    input_mean, input_basis = fit_pca(train_input, 64)

    weights = components["router_weights"].float()
    quant3 = components["selected_quant3"].float()
    aggregate = (quant3 * weights.unsqueeze(-1)).sum(1)
    aggregate_mean, aggregate_basis = fit_pca(aggregate[train_slice], 32)

    train_selected = quant3[train_slice].reshape(-1, quant3.shape[-1])
    selected_mean, selected_basis = fit_pca(train_selected, 16)

    feature_sets = []
    for split_slice in (train_slice, test_slice):
        split_input = components["moe_input"][split_slice].float()
        split_quant3 = quant3[split_slice]
        split_weights = weights[split_slice]
        split_ids = components["router_ids"][split_slice].long()
        input_projection = pca_features(split_input, input_mean, input_basis)
        aggregate_projection = pca_features(
            aggregate[split_slice], aggregate_mean, aggregate_basis
        )
        selected_projection = pca_features(
            split_quant3.reshape(-1, split_quant3.shape[-1]),
            selected_mean,
            selected_basis,
        ).view(TOKENS_PER_SPLIT, 6 * selected_basis.shape[1])
        input_rms = split_input.square().mean(-1).sqrt().unsqueeze(1).expand_as(
            split_weights
        )
        output_rms = split_quant3.square().mean(-1).sqrt()
        normalized_weights = split_weights / split_weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-12)
        order = torch.argsort(split_weights, dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(6, device=order.device).expand_as(order),
        )
        continuous = torch.stack(
            (
                split_weights.log().clamp_min(-30),
                split_weights,
                normalized_weights,
                input_rms,
                output_rms,
                split_weights * output_rms,
            ),
            dim=-1,
        ).reshape(TOKENS_PER_SPLIT, -1)
        expert_one_hot = torch.nn.functional.one_hot(
            split_ids, num_classes=64
        ).reshape(TOKENS_PER_SPLIT, -1).float()
        rank_one_hot = torch.nn.functional.one_hot(
            ranks, num_classes=6
        ).reshape(TOKENS_PER_SPLIT, -1).float()
        feature_sets.append(
            torch.cat(
                (
                    input_projection,
                    aggregate_projection,
                    selected_projection,
                    continuous,
                    expert_one_hot,
                    rank_one_hot,
                ),
                dim=1,
            ).cpu()
        )
    return feature_sets[0], feature_sets[1]


def quadratic_mask_design(masks: torch.Tensor) -> torch.Tensor:
    remaining = (~masks).float()
    columns = [torch.ones(remaining.shape[0])]
    columns.extend(remaining[:, slot] for slot in range(remaining.shape[1]))
    columns.extend(
        remaining[:, left] * remaining[:, right]
        for left in range(remaining.shape[1])
        for right in range(left + 1, remaining.shape[1])
    )
    return torch.stack(columns, dim=1)


def standardize(
    train: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(0)
    scale = train.std(0).clamp_min(1e-4)
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def ridge_fit(
    features: torch.Tensor, target: torch.Tensor, regularization: float
) -> torch.Tensor:
    design = torch.cat((torch.ones(features.shape[0], 1), features), dim=1).double()
    target = target.double()
    gram = design.T @ design
    penalty = torch.eye(gram.shape[0], dtype=torch.float64) * regularization
    penalty[0, 0] = 0
    return torch.linalg.solve(gram + penalty, design.T @ target)


def ridge_predict(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    design = torch.cat((torch.ones(features.shape[0], 1), features), dim=1).double()
    return (design @ coefficients).float()


def schedule_from_predicted_damage(
    predicted_damage: torch.Tensor,
    masks: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, int]:
    best_damage, best_masks = best_mask_per_cardinality(
        predicted_damage.clamp_min(0), masks
    )
    curve, backpointers = discrete_rate_distortion(best_damage.double().numpy())
    budget = int(fraction * predicted_damage.shape[0] * masks.shape[1])
    cost = int(np.argmin(curve[: budget + 1]))
    token_cost = torch.from_numpy(recover_cost_schedule(backpointers, cost)).long()
    indices = best_masks[torch.arange(predicted_damage.shape[0]), token_cost]
    return indices, cost


def evaluate_prediction(
    predicted_damage: torch.Tensor,
    exact_damage: torch.Tensor,
    masks: torch.Tensor,
) -> list[dict[str, float]]:
    rows = []
    for fraction in FRACTIONS:
        indices, cost = schedule_from_predicted_damage(
            predicted_damage, masks, fraction
        )
        actual = exact_damage[torch.arange(exact_damage.shape[0]), indices]
        rows.append(
            {
                "requested_fraction": fraction,
                "actual_fraction": cost / (exact_damage.shape[0] * masks.shape[1]),
                "actual_kl_mean": float(actual.mean().item()),
                "actual_kl_p95": float(torch.quantile(actual, 0.95).item()),
            }
        )
    return rows


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("quadratic predictor training requires CUDA")
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    component_path = (
        ROOT / "data" / "traces" / "layer26_dynamic_precision_components.safetensors"
    )
    schedule_path = (
        ROOT / "data" / "traces" / "layer26_dynamic_oracle_schedules.safetensors"
    )
    components = {
        key: value.to("cuda")
        for key, value in load_file(component_path, device="cpu").items()
    }
    schedules = load_file(schedule_path, device="cpu")
    masks = binary_upgrade_masks(6)
    mask_design = quadratic_mask_design(masks)
    design_pinv = torch.linalg.pinv(mask_design)
    validation_damage = schedules["validation_exact_mask_kl"].float()
    test_damage = schedules["test_exact_mask_kl"].float()
    validation_coefficients = validation_damage @ design_pinv.T
    test_coefficients = test_damage @ design_pinv.T
    validation_reconstruction = validation_coefficients @ mask_design.T
    test_reconstruction = test_coefficients @ mask_design.T

    validation_features, test_features = token_features(components)
    validation_features, test_features, feature_mean, feature_scale = standardize(
        validation_features, test_features
    )
    target_mean = validation_coefficients.mean(0)
    target_scale = validation_coefficients.std(0).clamp_min(1e-8)
    standardized_targets = (validation_coefficients - target_mean) / target_scale

    calibration_train = slice(0, 768)
    calibration_holdout = slice(768, TOKENS_PER_SPLIT)
    tuning_rows = []
    best_regularization = None
    best_holdout_kl = float("inf")
    for regularization in RIDGE_GRID:
        coefficients = ridge_fit(
            validation_features[calibration_train],
            standardized_targets[calibration_train],
            regularization,
        )
        predicted_standard = ridge_predict(
            validation_features[calibration_holdout], coefficients
        )
        predicted_coefficients = predicted_standard * target_scale + target_mean
        predicted_damage = predicted_coefficients @ mask_design.T
        rows = evaluate_prediction(
            predicted_damage,
            validation_damage[calibration_holdout],
            masks,
        )
        holdout_25 = next(row for row in rows if row["requested_fraction"] == 0.25)
        tuning_rows.append(
            {
                "regularization": regularization,
                "holdout_rate_distortion": rows,
            }
        )
        if holdout_25["actual_kl_mean"] < best_holdout_kl:
            best_holdout_kl = holdout_25["actual_kl_mean"]
            best_regularization = regularization

    final_coefficients = ridge_fit(
        validation_features, standardized_targets, float(best_regularization)
    )
    predicted_test_standard = ridge_predict(test_features, final_coefficients)
    predicted_test_coefficients = (
        predicted_test_standard * target_scale + target_mean
    )
    predicted_test_damage = predicted_test_coefficients @ mask_design.T
    test_rows = evaluate_prediction(predicted_test_damage, test_damage, masks)

    oracle_test_rows = evaluate_prediction(test_damage, test_damage, masks)
    report = {
        "status": "complete",
        "experiment": "predict_layer26_mask_interactions_via_quadratic_kl_model",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "quadratic_terms": mask_design.shape[1],
        "feature_count": validation_features.shape[1],
        "features": "PCA64 MoE input, PCA32 aggregate 3-bit output, PCA16 per selected 3-bit output, router statistics, expert IDs and router ranks",
        "validation_quadratic_reconstruction": {
            "rmse": float(
                (validation_reconstruction - validation_damage)
                .square()
                .mean()
                .sqrt()
                .item()
            ),
            "relative_rmse_vs_damage_rms": float(
                (
                    (validation_reconstruction - validation_damage)
                    .square()
                    .mean()
                    .sqrt()
                    / validation_damage.square().mean().sqrt()
                ).item()
            ),
        },
        "test_per_token_fitted_quadratic_reconstruction": {
            "rmse": float(
                (test_reconstruction - test_damage).square().mean().sqrt().item()
            ),
            "relative_rmse_vs_damage_rms": float(
                (
                    (test_reconstruction - test_damage).square().mean().sqrt()
                    / test_damage.square().mean().sqrt()
                ).item()
            ),
        },
        "ridge_tuning": tuning_rows,
        "selected_regularization": best_regularization,
        "test_predicted_rate_distortion": test_rows,
        "test_exact_oracle_rate_distortion": oracle_test_rows,
        "all_3bit_test_kl": float(test_damage[:, 0].mean().item()),
        "all_4bit_test_kl": float(test_damage[:, -1].mean().item()),
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_quadratic_mask_predictor.json",
        envelope("dynamic_precision_predictor", report),
    )
    print(
        f"quadratic_relative_rmse={report['test_per_token_fitted_quadratic_reconstruction']['relative_rmse_vs_damage_rms']:.6f}",
        flush=True,
    )
    print(f"selected_regularization={best_regularization}", flush=True)
    for row in test_rows:
        print(
            f"test predicted f={row['actual_fraction']:.3f} "
            f"KL={row['actual_kl_mean']:.6f}",
            flush=True,
        )
    print(path)
