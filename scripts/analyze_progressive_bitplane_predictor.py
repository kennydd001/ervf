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
from moe_lab.metrics import regression_metrics
from moe_lab.reporting import ROOT, envelope, write_json


TOKENS = 1024
FRACTIONS = (0.15, 0.20, 0.25)


def fit_delta_calibrators(
    previous: torch.Tensor,
    target: torch.Tensor,
    expert_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    denominator = previous.square().sum().clamp_min(1e-20)
    shared_scalar = (previous * target).sum() / denominator
    expert_scalar = torch.ones(64, device=previous.device) * shared_scalar
    expert_channel = (
        torch.ones(64, previous.shape[-1], device=previous.device) * shared_scalar
    )
    for expert_id in range(64):
        selected = expert_ids == expert_id
        if not selected.any():
            continue
        source = previous[selected]
        destination = target[selected]
        expert_scalar[expert_id] = (source * destination).sum() / source.square().sum().clamp_min(
            1e-20
        )
        expert_channel[expert_id] = (source * destination).sum(0) / source.square().sum(
            0
        ).clamp_min(1e-20)
    return {
        "previous_delta_identity": torch.tensor(1.0, device=previous.device),
        "shared_scalar": shared_scalar,
        "expert_scalar": expert_scalar,
        "expert_channel_diagonal": expert_channel,
    }


def apply_calibrator(
    previous: torch.Tensor,
    expert_ids: torch.Tensor,
    calibrator: torch.Tensor,
) -> torch.Tensor:
    if calibrator.ndim == 0:
        return previous * calibrator
    if calibrator.ndim == 1:
        return previous * calibrator[expert_ids].unsqueeze(-1)
    return previous * calibrator[expert_ids]


def euclidean_mask_damage(
    predicted_delta: torch.Tensor,
    masks: torch.Tensor,
    token_batch: int = 64,
) -> torch.Tensor:
    remaining = (~masks).to(predicted_delta.device).float()
    damage = torch.empty(predicted_delta.shape[0], masks.shape[0])
    for start in range(0, predicted_delta.shape[0], token_batch):
        stop = min(start + token_batch, predicted_delta.shape[0])
        residual = torch.einsum(
            "ms,bsd->bmd", remaining, predicted_delta[start:stop]
        )
        damage[start:stop] = residual.square().mean(-1).cpu()
    return damage


def schedule_indices(
    predicted_damage: torch.Tensor,
    masks: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, int]:
    best_damage, best_masks = best_mask_per_cardinality(predicted_damage, masks)
    curve, backpointers = discrete_rate_distortion(best_damage.double().numpy())
    budget = int(fraction * predicted_damage.shape[0] * masks.shape[1])
    cost = int(np.argmin(curve[: budget + 1]))
    per_token_cost = torch.from_numpy(recover_cost_schedule(backpointers, cost)).long()
    indices = best_masks[torch.arange(predicted_damage.shape[0]), per_token_cost]
    return indices, cost


def evaluate_schedule_method(
    predicted_damage: torch.Tensor,
    exact_test_kl: torch.Tensor,
    masks: torch.Tensor,
) -> list[dict[str, float]]:
    rows = []
    for fraction in FRACTIONS:
        indices, cost = schedule_indices(predicted_damage, masks, fraction)
        actual = exact_test_kl[torch.arange(TOKENS), indices]
        rows.append(
            {
                "requested_fraction": fraction,
                "actual_fraction": cost / (TOKENS * 6),
                "actual_kl_mean": float(actual.mean().item()),
                "actual_kl_p95": float(torch.quantile(actual, 0.95).item()),
            }
        )
    return rows


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("progressive bitplane analysis requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    components = {
        key: value.to(device)
        for key, value in load_file(
            ROOT
            / "data"
            / "traces"
            / "layer26_dynamic_precision_components.safetensors",
            device="cpu",
        ).items()
    }
    quant2 = load_file(
        ROOT / "data" / "traces" / "layer26_selected_quant2.safetensors",
        device="cpu",
    )["selected_quant2"].to(device)
    schedules = load_file(
        ROOT / "data" / "traces" / "layer26_dynamic_oracle_schedules.safetensors",
        device="cpu",
    )
    masks = binary_upgrade_masks(6)
    weights = components["router_weights"].float().unsqueeze(-1)
    previous_delta = weights * (
        components["selected_quant3"].float() - quant2.float()
    )
    target_delta = weights * (
        components["selected_quant4"].float()
        - components["selected_quant3"].float()
    )
    train_slice = slice(0, TOKENS)
    test_slice = slice(TOKENS, 2 * TOKENS)
    calibrators = fit_delta_calibrators(
        previous_delta[train_slice],
        target_delta[train_slice],
        components["router_ids"][train_slice].long(),
    )
    exact_test_kl = schedules["test_exact_mask_kl"].float()
    methods = {}
    prediction_quality = {}
    test_ids = components["router_ids"][test_slice].long()
    for name, calibrator in calibrators.items():
        predicted = apply_calibrator(
            previous_delta[test_slice], test_ids, calibrator
        )
        prediction_quality[name] = regression_metrics(
            predicted.cpu(), target_delta[test_slice].cpu()
        )
        predicted_damage = euclidean_mask_damage(predicted, masks)
        methods[name] = evaluate_schedule_method(
            predicted_damage, exact_test_kl, masks
        )
        row_25 = next(
            row for row in methods[name] if row["requested_fraction"] == 0.25
        )
        print(
            f"{name} delta_cos={prediction_quality[name]['cosine']:.4f} "
            f"KL25={row_25['actual_kl_mean']:.6f}",
            flush=True,
        )

    true_delta_damage = euclidean_mask_damage(target_delta[test_slice], masks)
    methods["true_q4_delta_euclidean_non_deployable"] = evaluate_schedule_method(
        true_delta_damage, exact_test_kl, masks
    )
    oracle_rows = evaluate_schedule_method(exact_test_kl, exact_test_kl, masks)
    methods["direct_teacher_kl_oracle"] = oracle_rows

    router_scores = components["router_weights"][test_slice].float()
    router_rows = []
    for fraction in FRACTIONS:
        count = int(fraction * router_scores.numel())
        chosen = torch.zeros(router_scores.numel(), dtype=torch.bool)
        chosen[torch.topk(router_scores.cpu().reshape(-1), count).indices] = True
        chosen = chosen.view(TOKENS, 6)
        mask_indices = (
            chosen.long() * (1 << torch.arange(6)).view(1, 6)
        ).sum(1)
        actual = exact_test_kl[torch.arange(TOKENS), mask_indices]
        router_rows.append(
            {
                "requested_fraction": fraction,
                "actual_fraction": float(chosen.float().mean().item()),
                "actual_kl_mean": float(actual.mean().item()),
                "actual_kl_p95": float(torch.quantile(actual, 0.95).item()),
            }
        )
    methods["router_weight"] = router_rows

    report = {
        "status": "complete",
        "experiment": "predict_3_to_4bit_mask_from_observed_2_to_3bit_delta",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "calibration": "WikiText validation, 1024 tokens",
        "test": "untouched WikiText test, 1024 tokens",
        "prediction_quality": prediction_quality,
        "test_rate_distortion": methods,
        "all_3bit_test_kl": float(exact_test_kl[:, 0].mean().item()),
        "all_4bit_test_kl": float(exact_test_kl[:, -1].mean().item()),
        "calibrator_storage_values": {
            "shared_scalar": 1,
            "expert_scalar": 64,
            "expert_channel_diagonal": 64 * 2048,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_progressive_bitplane_predictor.json",
        envelope("dynamic_precision_predictor", report),
    )
    true_row = next(
        row
        for row in methods["true_q4_delta_euclidean_non_deployable"]
        if row["requested_fraction"] == 0.25
    )
    print(f"true_delta_euclidean_KL25={true_row['actual_kl_mean']:.6f}")
    print(path)
