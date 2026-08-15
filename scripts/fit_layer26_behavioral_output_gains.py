from __future__ import annotations

import time

import torch
from safetensors.torch import load_file

from estimate_layer26_observability import behavioral_metrics, corpus_blocks
from evaluate_layer26_dynamic_precision_oracle import (
    BLOCK_SIZE,
    BLOCKS_PER_SPLIT,
    combine_selected,
)
from moe_lab.behavioral import sample_fisher_score_gradient_replicates
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT, envelope, write_json


TOKENS = BLOCKS_PER_SPLIT * BLOCK_SIZE
FISHER_SAMPLES = 8
LAMBDA_FRACTIONS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def router_ranks(weights: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(weights, dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(weights.shape[1], device=weights.device).expand_as(order),
    )
    return ranks


def category_ids(
    mode: str, expert_ids: torch.Tensor, ranks: torch.Tensor
) -> tuple[torch.Tensor, int]:
    if mode == "router_rank":
        return ranks, 6
    if mode == "expert":
        return expert_ids, 64
    if mode == "expert_x_router_rank":
        return expert_ids * 6 + ranks, 64 * 6
    raise ValueError(mode)


def fisher_design(
    gradients: torch.Tensor,
    selected_quant3: torch.Tensor,
    router_weights: torch.Tensor,
    categories: torch.Tensor,
    category_count: int,
) -> torch.Tensor:
    contributions = selected_quant3.float() * router_weights.float().unsqueeze(-1)
    scores = torch.einsum("knd,nsd->kns", gradients, contributions)
    expanded_categories = categories.unsqueeze(0).expand(gradients.shape[0], -1, -1)
    design = torch.zeros(
        gradients.shape[0],
        gradients.shape[1],
        category_count,
        device=gradients.device,
    )
    design.scatter_add_(2, expanded_categories, scores)
    return design


def solve_correction(
    design: torch.Tensor,
    target: torch.Tensor,
    lambda_fraction: float,
) -> torch.Tensor:
    flat_design = design.reshape(-1, design.shape[-1]).double()
    flat_target = target.reshape(-1).double()
    gram = flat_design.T @ flat_design
    scale = torch.diagonal(gram).mean().clamp_min(1e-20)
    regularized = gram + torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    ) * (max(lambda_fraction, 1e-10) * scale)
    return torch.linalg.solve(regularized, flat_design.T @ flat_target).float()


@torch.inference_mode()
def apply_gains(
    components: dict[str, torch.Tensor],
    token_slice: slice,
    mode: str,
    correction: torch.Tensor,
) -> torch.Tensor:
    ids = components["router_ids"][token_slice].long()
    weights = components["router_weights"][token_slice]
    ranks = router_ranks(weights)
    categories, _ = category_ids(mode, ids, ranks)
    gains = (1.0 + correction[categories]).to(
        components["selected_quant3"].dtype
    )
    selected = components["selected_quant3"][token_slice] * gains.unsqueeze(-1)
    return combine_selected(
        components["post_attention"][token_slice],
        components["shared"][token_slice],
        selected,
        weights,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("behavioral gain fitting requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
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
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    split_ids = {
        split: corpus_blocks(model_dir, split, BLOCKS_PER_SPLIT)
        for split in ("validation", "test")
    }
    validation_slice = slice(0, TOKENS)
    test_slice = slice(TOKENS, 2 * TOKENS)
    validation_q3 = combine_selected(
        components["post_attention"][validation_slice],
        components["shared"][validation_slice],
        components["selected_quant3"][validation_slice],
        components["router_weights"][validation_slice],
    )
    test_q3 = combine_selected(
        components["post_attention"][test_slice],
        components["shared"][test_slice],
        components["selected_quant3"][test_slice],
        components["router_weights"][test_slice],
    )
    test_q4 = combine_selected(
        components["post_attention"][test_slice],
        components["shared"][test_slice],
        components["selected_quant4"][test_slice],
        components["router_weights"][test_slice],
    )
    gradients = sample_fisher_score_gradient_replicates(
        components["teacher"][validation_slice],
        norm_weight,
        lm_head,
        batch_size=32,
        seed=20260809,
        samples_per_state=FISHER_SAMPLES,
    ).to(device)
    visible_target = torch.einsum(
        "knd,nd->kn",
        gradients,
        components["teacher"][validation_slice].float() - validation_q3.float(),
    )

    train_tokens = 768
    holdout_slice = slice(train_tokens, TOKENS)
    holdout_teacher = components["teacher"][validation_slice][holdout_slice].view(
        2, BLOCK_SIZE, -1
    ).cpu()
    holdout_ids = split_ids["validation"][6:8]
    modes = {}
    for mode in ("router_rank", "expert", "expert_x_router_rank"):
        ids = components["router_ids"][validation_slice].long()
        ranks = router_ranks(components["router_weights"][validation_slice])
        categories, count = category_ids(mode, ids, ranks)
        design = fisher_design(
            gradients,
            components["selected_quant3"][validation_slice],
            components["router_weights"][validation_slice],
            categories,
            count,
        )
        tuning = []
        best_lambda = None
        best_kl = float("inf")
        for lambda_fraction in LAMBDA_FRACTIONS:
            correction = solve_correction(
                design[:, :train_tokens],
                visible_target[:, :train_tokens],
                lambda_fraction,
            )
            candidate = apply_gains(
                components,
                slice(train_tokens, TOKENS),
                mode,
                correction,
            ).view(2, BLOCK_SIZE, -1).cpu()
            metrics = behavioral_metrics(
                holdout_teacher,
                candidate,
                holdout_ids,
                norm_weight,
                lm_head,
            )
            tuning.append({"lambda_fraction": lambda_fraction, **metrics})
            if metrics["teacher_to_candidate_kl"] < best_kl:
                best_kl = metrics["teacher_to_candidate_kl"]
                best_lambda = lambda_fraction
        correction = solve_correction(design, visible_target, float(best_lambda))
        validation_candidate = apply_gains(
            components, validation_slice, mode, correction
        ).view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
        test_candidate = apply_gains(
            components, test_slice, mode, correction
        ).view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu()
        validation_metrics = behavioral_metrics(
            components["teacher"][validation_slice].view(
                BLOCKS_PER_SPLIT, BLOCK_SIZE, -1
            ).cpu(),
            validation_candidate,
            split_ids["validation"],
            norm_weight,
            lm_head,
        )
        test_metrics = behavioral_metrics(
            components["teacher"][test_slice].view(
                BLOCKS_PER_SPLIT, BLOCK_SIZE, -1
            ).cpu(),
            test_candidate,
            split_ids["test"],
            norm_weight,
            lm_head,
        )
        modes[mode] = {
            "parameter_count": count,
            "selected_lambda_fraction": best_lambda,
            "holdout_tuning": tuning,
            "correction_statistics": {
                "minimum": float(correction.min().item()),
                "maximum": float(correction.max().item()),
                "mean": float(correction.mean().item()),
                "standard_deviation": float(correction.std().item()),
            },
            "validation": validation_metrics,
            "test": test_metrics,
        }
        print(
            f"{mode} lambda={best_lambda} test_KL={test_metrics['teacher_to_candidate_kl']:.6f}",
            flush=True,
        )

    test_teacher = components["teacher"][test_slice].view(
        BLOCKS_PER_SPLIT, BLOCK_SIZE, -1
    ).cpu()
    baselines = {
        "all_3bit": behavioral_metrics(
            test_teacher,
            test_q3.view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu(),
            split_ids["test"],
            norm_weight,
            lm_head,
        ),
        "all_4bit": behavioral_metrics(
            test_teacher,
            test_q4.view(BLOCKS_PER_SPLIT, BLOCK_SIZE, -1).cpu(),
            split_ids["test"],
            norm_weight,
            lm_head,
        ),
    }
    report = {
        "status": "complete",
        "experiment": "layer26_static_behavioral_output_gain_calibration",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "fisher_samples_per_validation_state": FISHER_SAMPLES,
        "selection_protocol": "fit first 768 validation tokens, select ridge on last 256, refit all 1024 validation tokens, evaluate untouched 1024 test tokens",
        "baselines": baselines,
        "methods": modes,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_behavioral_output_gains.json",
        envelope("co_routed_error_calibration", report),
    )
    print(path)
