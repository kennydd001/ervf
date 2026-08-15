from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from moe_lab.cache_routing import CacheRoutingPolicy, route_batch, touch_route
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.risk_selector import (
    RouteRiskMLP,
    expert_identity_features,
    finite_sample_upper_quantile,
    route_risk_features,
    subset_mask,
    top_j_slate,
)


SEED = 20260810
BLOCK_SIZE = 128
EPSILON = 1e-6
TRAIN_TOKENS = slice(0, 640)
MODEL_SELECTION_TOKENS = slice(640, 768)
CONFORMAL_TOKENS = slice(768, 1024)
TEST_TOKENS = slice(1024, 2048)
ALPHAS = (0.005, 0.05, 0.10)
RISK_LIMITS = (1e-3, 3e-3, 1e-2)
MAX_SWAPS = (1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--report-name", default="layer26_conformal_cache_selector.json"
    )
    return parser.parse_args()


def _log_target(risk: torch.Tensor) -> torch.Tensor:
    return torch.log10(risk.float().clamp_min(0) + EPSILON)


def _quality(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.flatten().float()
    target = target.flatten().float()
    difference = prediction - target
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    correlation = (
        centered_prediction.mul(centered_target).mean()
        / (
            centered_prediction.square().mean().sqrt()
            * centered_target.square().mean().sqrt()
        ).clamp_min(1e-12)
    )
    return {
        "log10_rmse": float(difference.square().mean().sqrt().item()),
        "log10_mae": float(difference.abs().mean().item()),
        "log10_pearson": float(correlation.item()),
        "mean_true_kl": float(
            (torch.pow(10.0, target) - EPSILON).clamp_min(0).mean().item()
        ),
        "mean_predicted_kl": float(
            (torch.pow(10.0, prediction) - EPSILON).clamp_min(0).mean().item()
        ),
    }


def train_predictor(
    features: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RouteRiskMLP, torch.Tensor, torch.Tensor, dict[str, object]]:
    train_x = features[TRAIN_TOKENS].reshape(-1, features.shape[-1]).float()
    train_y = targets[TRAIN_TOKENS].reshape(-1).float()
    selection_x = features[MODEL_SELECTION_TOKENS].reshape(
        -1, features.shape[-1]
    ).float()
    selection_y = targets[MODEL_SELECTION_TOKENS].reshape(-1).float()
    feature_mean = train_x.mean(0)
    feature_scale = train_x.std(0).clamp_min(1e-5)
    train_x = ((train_x - feature_mean) / feature_scale).to(device)
    train_y = train_y.to(device)
    selection_x = ((selection_x - feature_mean) / feature_scale).to(device)
    selection_y = selection_y.to(device)

    torch.manual_seed(SEED)
    model = RouteRiskMLP(features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    generator = torch.Generator().manual_seed(SEED)
    history = []
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(train_x.shape[0], generator=generator)
        total_loss = 0.0
        examples = 0
        for start in range(0, permutation.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            prediction = model(train_x[indices])
            loss = torch.nn.functional.huber_loss(
                prediction, train_y[indices], delta=0.5
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * indices.numel()
            examples += indices.numel()
        model.eval()
        with torch.inference_mode():
            selection_prediction = model(selection_x)
            selection_loss = float(
                (selection_prediction - selection_y).square().mean().item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_huber": total_loss / examples,
                "selection_mse": selection_loss,
            }
        )
        if selection_loss < best_loss - 1e-6:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("predictor training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, feature_mean, feature_scale, {
        "best_epoch": best_epoch,
        "best_selection_mse": best_loss,
        "epochs_run": len(history),
        "last_five_epochs": history[-5:],
    }


@torch.inference_mode()
def predict_all(
    model: RouteRiskMLP,
    features: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    batch_size: int = 8192,
) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1]).float()
    output = []
    for start in range(0, flat.shape[0], batch_size):
        normalized = (flat[start : start + batch_size] - mean) / scale
        output.append(model(normalized.to(device)).cpu())
    return torch.cat(output).view(features.shape[:-1])


def _route_summary(
    risks: list[float],
    bounds: list[float],
    threshold: float,
    strict_loads: int,
    adaptive_loads: int,
    substitutions: int,
    tokens: int,
) -> dict[str, object]:
    risk = torch.tensor(risks, dtype=torch.float32)
    bound = torch.tensor(bounds, dtype=torch.float32)
    return {
        "strict_expert_loads": strict_loads,
        "adaptive_expert_loads": adaptive_loads,
        "expert_load_reduction_fraction": 1.0 - adaptive_loads / strict_loads,
        "strict_cache_miss_fraction": strict_loads / (tokens * 6),
        "adaptive_cache_miss_fraction": adaptive_loads / (tokens * 6),
        "substituted_token_fraction": substitutions / tokens,
        "exact_local_kl": {
            "mean": float(risk.mean().item()),
            "median": float(risk.median().item()),
            "p95": float(torch.quantile(risk, 0.95).item()),
            "maximum": float(risk.max().item()),
            "threshold_exceedance_fraction": float((risk > threshold).float().mean().item()),
        },
        "chosen_bound_coverage": float((risk <= bound + 1e-12).float().mean().item()),
        "mean_upper_bound": float(bound.mean().item()),
    }


def simulate_selector(
    top_ids: torch.Tensor,
    subsets: torch.Tensor,
    exact_risk: torch.Tensor,
    upper_bound: torch.Tensor,
    threshold: float,
    capacity: int = 32,
) -> dict[str, object]:
    if top_ids.shape[0] % BLOCK_SIZE:
        raise ValueError("token count must be a multiple of the block size")
    original_position = int(
        (subsets == torch.arange(6)).all(1).nonzero(as_tuple=False).item()
    )
    strict_loads = 0
    adaptive_loads = 0
    substitutions = 0
    chosen_risks: list[float] = []
    chosen_bounds: list[float] = []
    chosen_positions: list[int] = []
    per_block = []
    for block_start in range(0, top_ids.shape[0], BLOCK_SIZE):
        strict_cache: list[int] = []
        adaptive_cache: list[int] = []
        block_strict = 0
        block_adaptive = 0
        block_substitutions = 0
        for token in range(block_start, block_start + BLOCK_SIZE):
            strict_route = top_ids[token, :6].tolist()
            misses = touch_route(strict_cache, strict_route, capacity)
            strict_loads += misses
            block_strict += misses

            allowed = (upper_bound[token] <= threshold).nonzero(
                as_tuple=False
            ).squeeze(1).tolist()
            if original_position not in allowed:
                allowed.append(original_position)
            best = None
            best_key = None
            cache_set = set(adaptive_cache)
            for position in allowed:
                route = top_ids[token, subsets[position]].tolist()
                misses = sum(expert not in cache_set for expert in route)
                key = (misses, float(upper_bound[token, position].item()), position)
                if best_key is None or key < best_key:
                    best = position
                    best_key = key
            if best is None:
                raise RuntimeError("selector has no route")
            route = top_ids[token, subsets[best]].tolist()
            misses = touch_route(adaptive_cache, route, capacity)
            adaptive_loads += misses
            block_adaptive += misses
            changed = best != original_position
            substitutions += int(changed)
            block_substitutions += int(changed)
            chosen_positions.append(best)
            chosen_risks.append(
                0.0
                if best == original_position
                else float(exact_risk[token, best].clamp_min(0).item())
            )
            chosen_bounds.append(
                0.0
                if best == original_position
                else float(upper_bound[token, best].clamp_min(0).item())
            )
        per_block.append(
            {
                "strict_expert_loads": block_strict,
                "adaptive_expert_loads": block_adaptive,
                "expert_load_reduction_fraction": 1.0
                - block_adaptive / block_strict,
                "substituted_token_fraction": block_substitutions / BLOCK_SIZE,
            }
        )
    summary = _route_summary(
        chosen_risks,
        chosen_bounds,
        threshold,
        strict_loads,
        adaptive_loads,
        substitutions,
        top_ids.shape[0],
    )
    summary["per_block"] = per_block
    summary["chosen_slate_positions"] = chosen_positions
    return summary


def fixed_max_rank_baseline(
    top_ids: torch.Tensor,
    top_weights: torch.Tensor,
    all_subsets: torch.Tensor,
    all_risk: torch.Tensor,
    policy: CacheRoutingPolicy,
    capacity: int = 32,
) -> dict[str, object]:
    blocks = top_ids.shape[0] // BLOCK_SIZE
    chosen, stats = route_batch(
        top_ids.view(blocks, BLOCK_SIZE, -1),
        top_weights.view(blocks, BLOCK_SIZE, -1),
        top_weights.clamp_min(1e-12).log().view(blocks, BLOCK_SIZE, -1),
        policy,
        capacity,
        0.0,
        6,
    )
    subset_lookup = {
        tuple(route.tolist()): index for index, route in enumerate(all_subsets)
    }
    risks = []
    substitutions = 0
    original = tuple(range(6))
    for token in range(top_ids.shape[0]):
        rank_by_expert = {
            int(expert): rank for rank, expert in enumerate(top_ids[token].tolist())
        }
        positions = tuple(sorted(rank_by_expert[int(expert)] for expert in chosen[token]))
        subset_index = subset_lookup[positions]
        risks.append(float(all_risk[token, subset_index].clamp_min(0).item()))
        substitutions += int(positions != original)
    original_policy = CacheRoutingPolicy("original")
    _, strict = route_batch(
        top_ids.view(blocks, BLOCK_SIZE, -1),
        top_weights.view(blocks, BLOCK_SIZE, -1),
        top_weights.clamp_min(1e-12).log().view(blocks, BLOCK_SIZE, -1),
        original_policy,
        capacity,
        0.0,
        6,
    )
    return _route_summary(
        risks,
        risks,
        float("inf"),
        strict["expert_loads"],
        stats["expert_loads"],
        substitutions,
        top_ids.shape[0],
    )


if __name__ == "__main__":
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_path = (
        ROOT / "data" / "traces" / "layer26_route_equivalence_full.safetensors"
    )
    artifact = load_file(artifact_path, device="cpu")
    all_subsets = artifact["subsets"].to(torch.long)
    all_risk = artifact["subset_kl"].float().clamp_min(0)
    top_ids = artifact["top12_expert_ids"].to(torch.long)
    top_weights = artifact["top12_router_weights"].float()

    base_slate_mask = top_j_slate(all_subsets, top_j=2)
    base_indices = base_slate_mask.nonzero(as_tuple=False).squeeze(1)
    base_subsets = all_subsets[base_indices]
    base_risk = all_risk[:, base_indices]
    rank_features = route_risk_features(top_weights, base_subsets, original_k=6)
    identity_features = expert_identity_features(
        top_weights,
        top_ids,
        base_subsets,
        total_experts=64,
        original_k=6,
    )
    components = load_file(
        ROOT
        / "data"
        / "traces"
        / "layer26_dynamic_precision_components.safetensors",
        device="cpu",
    )
    hidden = components["moe_input"].float()
    del components
    hidden_mean = hidden[TRAIN_TOKENS].mean(0)
    centered_train = hidden[TRAIN_TOKENS] - hidden_mean
    torch.manual_seed(SEED)
    _, _, hidden_basis = torch.pca_lowrank(
        centered_train, q=32, center=False, niter=6
    )
    hidden_projection = (hidden - hidden_mean) @ hidden_basis
    expanded_hidden = hidden_projection.unsqueeze(1).expand(
        -1, base_subsets.shape[0], -1
    )
    features = torch.cat(
        (rank_features, identity_features, expanded_hidden), dim=-1
    )
    del rank_features, identity_features, expanded_hidden
    log_risk = _log_target(base_risk)
    model, feature_mean, feature_scale, training = train_predictor(
        features, log_risk, args, device
    )
    predicted_log_risk = predict_all(
        model, features, feature_mean, feature_scale, device
    )

    predictor_quality = {
        "train": _quality(
            predicted_log_risk[TRAIN_TOKENS], log_risk[TRAIN_TOKENS]
        ),
        "model_selection": _quality(
            predicted_log_risk[MODEL_SELECTION_TOKENS],
            log_risk[MODEL_SELECTION_TOKENS],
        ),
        "conformal_calibration": _quality(
            predicted_log_risk[CONFORMAL_TOKENS], log_risk[CONFORMAL_TOKENS]
        ),
        "untouched_test": _quality(
            predicted_log_risk[TEST_TOKENS], log_risk[TEST_TOKENS]
        ),
    }

    original_global_index = int(
        (all_subsets == torch.arange(6)).all(1).nonzero(as_tuple=False).item()
    )
    original_base_position = int(
        (base_indices == original_global_index).nonzero(as_tuple=False).item()
    )
    slate_results = {}
    for max_swaps in MAX_SWAPS:
        overlap = subset_mask(base_subsets)[:, :6].sum(1)
        local_mask = overlap >= 6 - max_swaps
        local_positions = local_mask.nonzero(as_tuple=False).squeeze(1)
        local_subsets = base_subsets[local_positions]
        local_risk = base_risk[:, local_positions]
        local_prediction = predicted_log_risk[:, local_positions]
        original_local_position = int(
            (local_positions == original_base_position).nonzero(as_tuple=False).item()
        )
        calibration_residual_by_candidate = (
            log_risk[CONFORMAL_TOKENS][:, local_positions]
            - local_prediction[CONFORMAL_TOKENS]
        ).clone()
        calibration_residual_by_candidate[:, original_local_position] = -torch.inf
        calibration_residual = calibration_residual_by_candidate.amax(dim=1)
        alpha_rows = {}
        for alpha in ALPHAS:
            correction = finite_sample_upper_quantile(calibration_residual, alpha)
            upper_log = local_prediction + correction
            upper = (torch.pow(10.0, upper_log) - EPSILON).clamp_min(0)
            upper[:, original_local_position] = 0.0
            coverage_upper_log = upper_log.clone()
            coverage_upper_log[:, original_local_position] = torch.inf
            test_true_log = log_risk[TEST_TOKENS][:, local_positions]
            simultaneous_coverage = float(
                (test_true_log <= coverage_upper_log[TEST_TOKENS] + 1e-12)
                .all(dim=1)
                .float()
                .mean()
                .item()
            )
            threshold_rows = {}
            for threshold in RISK_LIMITS:
                predictor_result = simulate_selector(
                    top_ids[TEST_TOKENS],
                    local_subsets,
                    local_risk[TEST_TOKENS],
                    upper[TEST_TOKENS],
                    threshold,
                )
                oracle_bound = local_risk[TEST_TOKENS].clone()
                oracle_bound[:, original_local_position] = 0.0
                oracle_result = simulate_selector(
                    top_ids[TEST_TOKENS],
                    local_subsets,
                    local_risk[TEST_TOKENS],
                    oracle_bound,
                    threshold,
                )
                threshold_rows[str(threshold)] = {
                    "conformal_predictor": predictor_result,
                    "exact_risk_oracle": oracle_result,
                }
            alpha_rows[str(alpha)] = {
                "log10_additive_conformal_correction": correction,
                "calibration_token_count": calibration_residual.numel(),
                "calibration_simultaneous_coverage": float(
                    (
                        log_risk[CONFORMAL_TOKENS][:, local_positions]
                        <= coverage_upper_log[CONFORMAL_TOKENS] + 1e-12
                    )
                    .all(dim=1)
                    .float()
                    .mean()
                    .item()
                ),
                "untouched_test_simultaneous_slate_coverage": simultaneous_coverage,
                "risk_limits": threshold_rows,
            }
        slate_results[f"top2_max{max_swaps}_swaps"] = {
            "candidate_count_including_original": local_subsets.shape[0],
            "alphas": alpha_rows,
        }

    fixed_baselines = {}
    for specification in (
        CacheRoutingPolicy("max_rank", top_j=5, parameter=7),
        CacheRoutingPolicy("max_rank", top_j=2, parameter=7),
        CacheRoutingPolicy("max_rank", top_j=2, parameter=12),
    ):
        fixed_baselines[specification.name] = fixed_max_rank_baseline(
            top_ids[TEST_TOKENS],
            top_weights[TEST_TOKENS],
            all_subsets,
            all_risk[TEST_TOKENS],
            specification,
        )

    model_path = ROOT / "data" / "traces" / "layer26_route_risk_mlp.safetensors"
    model_tensors = {
        f"model.{name}": value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    model_tensors.update(
        {
            "feature_mean": feature_mean.contiguous(),
            "feature_scale": feature_scale.contiguous(),
            "hidden_mean": hidden_mean.contiguous(),
            "hidden_basis": hidden_basis.contiguous(),
        }
    )
    save_file(
        model_tensors,
        model_path,
        metadata={
            "training_tokens": "validation tokens 0:640",
            "model_selection_tokens": "validation tokens 640:768",
            "conformal_tokens": "validation tokens 768:1024",
            "test_tokens": "untouched test tokens 0:1024",
            "target": "log10(exact full-vocabulary KL + 1e-6)",
        },
    )
    report = {
        "status": "complete",
        "experiment": "layer26_teacher_free_conformal_route_risk_selector",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "artifact": str(artifact_path.resolve()),
        "model_artifact": str(model_path.resolve()),
        "data_partition": {
            "predictor_train": "WikiText validation blocks 0:5 (640 tokens)",
            "early_stop_model_selection": "WikiText validation block 5 (128 tokens)",
            "split_conformal_calibration": "WikiText validation blocks 6:8 (256 tokens)",
            "untouched_test": "all eight WikiText test blocks (1024 tokens)",
            "partition_unit": "contiguous 128-token blocks; no block crosses partitions",
        },
        "candidate_family": {
            "base": "all top-12 choose-6 routes that preserve router ranks 1 and 2",
            "base_candidate_count": base_subsets.shape[0],
            "runtime_features": "top-12 router weights, candidate rank/mask summaries, concrete expert identities, and a 32D train-only PCA projection of the current MoE input",
            "forbidden_runtime_features": "teacher logits, exact KL, expert outputs, cache outcomes",
        },
        "training": training,
        "predictor_quality": predictor_quality,
        "conformal_method": {
            "score": "per-token maximum over the entire candidate slate of true_log10_KL minus predicted_log10_KL",
            "quantile": "one-sided split conformal with ceil((n+1)*(1-alpha)) finite-sample correction",
            "guarantee_caveat": "marginal exchangeability is only approximate for contiguous language tokens; empirical held-out coverage is reported",
            "original_route": "always allowed with structural zero risk",
        },
        "slates": slate_results,
        "fixed_test_baselines": fixed_baselines,
        "scope_caveat": "direct layer-26 same-input full-vocabulary KL and expert-granularity LRU only; not yet a model-wide or autoregressive guarantee",
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        args.report_name, envelope("conformal_route_risk", report)
    )
    print(path)
    print({"predictor_quality": predictor_quality["untouched_test"]})
    for slate_name, slate in slate_results.items():
        row = slate["alphas"]["0.1"]["risk_limits"]["0.003"]
        print(
            slate_name,
            {
                "coverage": slate["alphas"]["0.1"][
                    "untouched_test_simultaneous_slate_coverage"
                ],
                "predictor": {
                    key: row["conformal_predictor"][key]
                    for key in (
                        "expert_load_reduction_fraction",
                        "substituted_token_fraction",
                        "chosen_bound_coverage",
                        "exact_local_kl",
                    )
                },
                "oracle_load_reduction": row["exact_risk_oracle"][
                    "expert_load_reduction_fraction"
                ],
            },
        )
