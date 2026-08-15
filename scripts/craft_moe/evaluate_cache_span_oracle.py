from __future__ import annotations

import argparse
import gc
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import safetensors
import scipy
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from evaluate_crcq_oracle import (
    COMPONENT_RELATIVE,
    DATASET_REVISION,
    MODEL_REVISION,
    SEED,
    corpus_tokens,
    evaluate_hidden,
    git_state,
    hardware_state,
    make_teacher_reference,
    metadata,
    regression_summary,
    sequence_blocks,
    sha256_file,
    write_json_once,
)
from moe_lab.behavioral import rmsnorm
from moe_lab.cache_routing import CacheRoutingPolicy, route_batch
from moe_lab.craft_moe.cache_span import (
    anchored_candidate,
    choose_lowest_mse_candidate,
    fit_span,
    nonempty_subsets,
    omp_cached_order,
    simulate_mass_budget_trace,
)
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT


LAYER = 26
EXPERTS = 64
ACTIVE_EXPERTS = 6
HIDDEN_SIZE = 2048
FULL_TOKENS_PER_SPLIT = 256
SMOKE_TOKENS = 32
TRACE_TOKENS_PER_SPLIT = 1024
FULL_BLOCK_SIZE = 128
CACHE_CAPACITY = 32
MASS_BUDGET_TOP_J = 2
MASS_BUDGET_DELTA = 0.004
DELTA_AVERAGE = 5.425435543060303
LOCAL_KL_THRESHOLD = 0.001
MAX_CACHED_EXTRAS = 4
MAX_EXTRA_PER_AVOIDED = 2
RIDGE_RELATIVE = 1e-4
COEFFICIENT_BOUND = 1.0
ROUTER_WEIGHT_TOLERANCE = 1e-6
METHODS = ("ridge", "nnls", "bounded")
BASIS_VARIANTS = (
    ("resident_selected", False),
    ("available_selected", False),
    ("resident_selected", True),
    ("available_selected", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H8 layer-26 optimistic ghost-cache span oracle."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--tokens-per-split", type=int)
    parser.add_argument("--candidate-batch", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--reuse-capture", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-capture", type=Path)
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.tokens_per_split is None:
        args.tokens_per_split = (
            SMOKE_TOKENS if args.stage == "smoke" else FULL_TOKENS_PER_SPLIT
        )
    if args.bootstrap_resamples is None:
        args.bootstrap_resamples = 500 if args.stage == "smoke" else 10_000
    if args.stage == "smoke":
        if not 1 <= args.tokens_per_split <= SMOKE_TOKENS:
            raise ValueError("smoke must use 1-32 validation tokens")
    elif (
        args.tokens_per_split != FULL_TOKENS_PER_SPLIT
        or args.bootstrap_resamples != 10_000
    ):
        raise ValueError("full run is fixed at 256+256 tokens and 10k bootstrap")
    if args.candidate_batch < 1 or args.bootstrap_resamples < 1:
        raise ValueError("batch sizes and bootstrap count must be positive")
    if args.output_json is None:
        args.output_json = ROOT / (
            "reports/runs/craft_moe/cache_span_smoke.json"
            if args.stage == "smoke"
            else "reports/craft_moe/cache_span.json"
        )
    elif not args.output_json.is_absolute():
        args.output_json = ROOT / args.output_json
    if args.output_capture is None:
        args.output_capture = ROOT / (
            "reports/runs/craft_moe/cache_span_layer26_capture_smoke.safetensors"
            if args.stage == "smoke"
            else "reports/runs/craft_moe/cache_span_layer26_capture.safetensors"
        )
    elif not args.output_capture.is_absolute():
        args.output_capture = ROOT / args.output_capture
    args.output_json = args.output_json.resolve()
    args.output_capture = args.output_capture.resolve()
    report_root = (ROOT / "reports").resolve()
    for path in (args.output_json, args.output_capture):
        if report_root not in path.parents:
            raise ValueError("outputs must remain under reports/")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output_json}")
    if args.output_capture.exists() and not args.reuse_capture:
        raise FileExistsError(
            f"capture exists; pass --reuse-capture after auditing it: {args.output_capture}"
        )
    if args.reuse_capture and not args.output_capture.exists():
        raise FileNotFoundError(args.output_capture)
    return args


def numeric_summary(values: list[float] | torch.Tensor) -> dict[str, float] | None:
    tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0:
        return None
    if not torch.isfinite(tensor).all():
        raise ValueError("summary values must be finite")
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def configuration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order = 0
    for basis, cached in BASIS_VARIANTS:
        for method in METHODS:
            rows.append(
                {
                    "name": (
                        f"{basis}_{'plus_cached_omp' if cached else 'no_cached'}_{method}"
                    ),
                    "basis": basis,
                    "use_cached": cached,
                    "method": method,
                    "tie_order": order,
                }
            )
            order += 1
    return rows


def routed_from_all(
    all_outputs: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    if expert_ids.shape != router_weights.shape:
        raise ValueError("route IDs and weights must align")
    tokens, slots = expert_ids.shape
    selected = all_outputs[
        torch.arange(tokens).view(tokens, 1), expert_ids.long()
    ]
    if selected.shape[:2] != (tokens, slots):
        raise RuntimeError("expert gather shape mismatch")
    return (
        selected.float() * router_weights.float().unsqueeze(-1)
    ).sum(1).to(all_outputs.dtype)


@torch.inference_mode()
def capture_all_expert_outputs(
    moe: LoadedMoELayer, inputs: torch.Tensor
) -> torch.Tensor:
    inputs_device = inputs.to(moe.device)
    outputs = torch.empty(
        inputs.shape[0], EXPERTS, HIDDEN_SIZE, dtype=inputs.dtype, device="cpu"
    )
    for expert_id, expert in enumerate(moe.experts):
        outputs[:, expert_id] = moe.expert_forward(inputs_device, expert).cpu()
        if expert_id % 4 == 3:
            print(f"cache_span_capture_experts={expert_id + 1}/64", flush=True)
    if not torch.isfinite(outputs.float()).all():
        raise RuntimeError("non-finite all-expert capture")
    return outputs


def dense_route(
    ids: torch.Tensor, weights: torch.Tensor, experts: int = EXPERTS
) -> torch.Tensor:
    dense = torch.zeros(ids.shape[0], experts, dtype=torch.float32)
    dense.scatter_add_(1, ids.long(), weights.float())
    return dense


def split_mass_budget_trace(
    ranked_ids: torch.Tensor,
    ranked_probabilities: torch.Tensor,
    raw_logits: torch.Tensor,
    block_size: int,
) -> tuple[dict[str, object], dict[str, object]]:
    trace = simulate_mass_budget_trace(
        ranked_ids,
        ranked_probabilities,
        raw_logits,
        capacity=CACHE_CAPACITY,
        delta_average=DELTA_AVERAGE,
        block_size=block_size,
        top_k=ACTIVE_EXPERTS,
        top_j=MASS_BUDGET_TOP_J,
        delta=MASS_BUDGET_DELTA,
    )
    batch = ranked_ids.shape[0] // block_size
    independent_routes, independent_stats = route_batch(
        ranked_ids.view(batch, block_size, EXPERTS),
        ranked_probabilities.view(batch, block_size, EXPERTS),
        raw_logits.view(batch, block_size, EXPERTS),
        CacheRoutingPolicy(
            "mass_budget", top_j=MASS_BUDGET_TOP_J, parameter=MASS_BUDGET_DELTA
        ),
        CACHE_CAPACITY,
        DELTA_AVERAGE,
        ACTIVE_EXPERTS,
    )
    audit = {
        "route_ids_exact": bool(torch.equal(trace["routes"], independent_routes)),
        "expert_loads_exact": bool(
            int(trace["expert_loads"]) == int(independent_stats["expert_loads"])
        ),
        "primary_expert_loads": int(trace["expert_loads"]),
        "independent_expert_loads": int(independent_stats["expert_loads"]),
    }
    if not audit["route_ids_exact"] or not audit["expert_loads_exact"]:
        raise RuntimeError("Mass-Budget trace audit failed")
    return trace, audit


def build_capture(
    moe: LoadedMoELayer,
    components: dict[str, torch.Tensor],
    split_slices: dict[str, slice],
    block_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = moe.device
    inputs_device = components["moe_input"].to(device)
    logits_device = F.linear(inputs_device.float(), moe.gate_weight.float())
    probabilities_device = logits_device.softmax(-1)
    ranked_probabilities_device, ranked_ids_device = probabilities_device.sort(
        dim=-1, descending=True, stable=True
    )
    official_ids, official_weights = moe.route(inputs_device)
    stored_ids = components["router_ids"].long().to(device)
    stored_weights = components["router_weights"].float().to(device)
    official_control = {
        "slot_order_ids_exact": bool(torch.equal(official_ids, stored_ids)),
        "router_weight_maximum_absolute_error": float(
            (official_weights.float() - stored_weights).abs().max().item()
        ),
    }
    natural_ids = ranked_ids_device[:, :ACTIVE_EXPERTS]
    natural_weights = probabilities_device.gather(1, natural_ids)
    if moe.norm_topk_prob:
        natural_weights = natural_weights / natural_weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-20)
    else:
        natural_weights = natural_weights * moe.routed_scaling_factor
    sorted_set_control = torch.equal(
        natural_ids.sort(1).values, stored_ids.sort(1).values
    )
    dense_weight_error = float(
        (
            dense_route(natural_ids.cpu(), natural_weights.cpu())
            - dense_route(components["router_ids"], components["router_weights"])
        )
        .abs()
        .max()
        .item()
    )
    if (
        not official_control["slot_order_ids_exact"]
        or official_control["router_weight_maximum_absolute_error"]
        > ROUTER_WEIGHT_TOLERANCE
        or not sorted_set_control
        or dense_weight_error > ROUTER_WEIGHT_TOLERANCE
    ):
        raise RuntimeError(
            "natural router control failed: "
            f"official={official_control}, "
            f"sorted_top6_set_exact={sorted_set_control}, "
            f"dense_weight_max_abs={dense_weight_error}"
        )

    all_outputs = capture_all_expert_outputs(moe, components["moe_input"])
    natural_ids_cpu = natural_ids.cpu()
    natural_weights_cpu = natural_weights.cpu()
    natural_routed = routed_from_all(
        all_outputs, natural_ids_cpu, natural_weights_cpu
    )
    route_parts = []
    cache_parts = []
    miss_parts = []
    weight_parts = []
    split_stats: dict[str, Any] = {}
    split_audits: dict[str, Any] = {}
    for split, sl in split_slices.items():
        trace, audit = split_mass_budget_trace(
            ranked_ids_device[sl].cpu(),
            ranked_probabilities_device[sl].cpu(),
            logits_device[sl].cpu(),
            block_size,
        )
        routes = trace["routes"]
        route_weights = probabilities_device[sl].cpu().gather(1, routes.long())
        if moe.norm_topk_prob:
            route_weights = route_weights / route_weights.sum(
                -1, keepdim=True
            ).clamp_min(1e-20)
        else:
            route_weights = route_weights * moe.routed_scaling_factor
        route_parts.append(routes)
        cache_parts.append(trace["cache_before"])
        miss_parts.append(trace["miss_mask"])
        weight_parts.append(route_weights)
        split_stats[split] = {
            key: value
            for key, value in trace.items()
            if key not in {"routes", "cache_before", "miss_mask"}
        }
        split_audits[split] = audit
    mass_budget_ids = torch.cat(route_parts)
    mass_budget_weights = torch.cat(weight_parts)
    cache_before = torch.cat(cache_parts)
    miss_mask = torch.cat(miss_parts)
    mass_budget_routed = routed_from_all(
        all_outputs, mass_budget_ids, mass_budget_weights
    )
    original = anchored_candidate(
        components["teacher"], natural_routed, natural_routed
    )
    original_bit_exact = torch.equal(original, components["teacher"])
    if not original_bit_exact:
        raise RuntimeError("official teacher-delta original control failed")
    tensors = {
        "all_expert_outputs": all_outputs.contiguous(),
        "natural_ids": natural_ids_cpu.to(torch.int16).contiguous(),
        "natural_weights": natural_weights_cpu.contiguous(),
        "natural_routed": natural_routed.contiguous(),
        "mass_budget_ids": mass_budget_ids.to(torch.int16).contiguous(),
        "mass_budget_weights": mass_budget_weights.contiguous(),
        "mass_budget_routed": mass_budget_routed.contiguous(),
        "cache_before": cache_before.contiguous(),
        "miss_mask": miss_mask.contiguous(),
        "teacher": components["teacher"].contiguous(),
        "trace_indices": components["trace_indices"].contiguous(),
    }
    controls = {
        "official_router": official_control,
        "sorted_top6_set_exact": sorted_set_control,
        "dense_router_weight_maximum_absolute_error": dense_weight_error,
        "original_teacher_delta_bit_exact": original_bit_exact,
        "mass_budget_independent_audit": split_audits,
        "mass_budget": split_stats,
        "all_outputs_finite": bool(torch.isfinite(all_outputs.float()).all()),
    }
    return tensors, controls


def candidate_record(
    *,
    token: int,
    subset_slots: tuple[int, ...],
    route_ids: torch.Tensor,
    retained_slots: list[int],
    base_ids: list[int],
    extra_ids: list[int],
    fit: Any,
    retained_routed: torch.Tensor,
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
) -> dict[str, Any]:
    candidate_routed = (retained_routed.float() + fit.prediction.float()).to(
        natural_routed.dtype
    )
    candidate_hidden = anchored_candidate(teacher, candidate_routed, natural_routed)
    if not torch.isfinite(candidate_hidden.float()).all():
        raise RuntimeError("non-finite cache-span candidate")
    return {
        "token": token,
        "avoid_count": len(subset_slots),
        "reconstructed_slots": list(subset_slots),
        "reconstructed_expert_ids": [int(route_ids[slot]) for slot in subset_slots],
        "retained_expert_ids": [int(route_ids[slot]) for slot in retained_slots],
        "base_expert_ids": base_ids,
        "cache_extra_expert_ids": extra_ids,
        "extra_computations": len(extra_ids),
        "coefficients": [float(value) for value in fit.coefficients.tolist()],
        "target_squared_error": float(fit.squared_error),
        "target_normalized_squared_error": float(fit.normalized_squared_error),
        "target_nrmse": float(math.sqrt(max(0.0, fit.normalized_squared_error))),
        "target_cosine": float(fit.cosine),
        "_candidate_state": candidate_hidden,
    }


def build_configuration_candidates(
    *,
    configuration: dict[str, Any],
    all_outputs: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    cache_before: torch.Tensor,
    miss_mask: torch.Tensor,
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
) -> list[dict[str, Any]]:
    zero_fill = configuration["name"] == "zero_fill"
    rows: list[dict[str, Any]] = []
    for token in range(route_ids.shape[0]):
        misses = miss_mask[token].nonzero(as_tuple=False).squeeze(1).tolist()
        if not misses:
            continue
        token_route = route_ids[token].long()
        route_outputs = all_outputs[token, token_route].float()
        contributions = route_outputs * route_weights[token].float().unsqueeze(1)
        best_by_count: dict[int, list[dict[str, Any]]] = {}
        for subset in nonempty_subsets(misses):
            reconstructed = set(subset)
            retained_slots = [
                slot for slot in range(ACTIVE_EXPERTS) if slot not in reconstructed
            ]
            target = contributions[list(subset)].sum(0)
            retained_routed = (
                contributions[retained_slots].sum(0)
                if retained_slots
                else torch.zeros_like(target)
            )
            if zero_fill:
                fit = fit_span(
                    torch.empty(HIDDEN_SIZE, 0), target, "ridge",
                    ridge_relative=RIDGE_RELATIVE,
                )
                row = candidate_record(
                    token=token,
                    subset_slots=subset,
                    route_ids=token_route,
                    retained_slots=retained_slots,
                    base_ids=[],
                    extra_ids=[],
                    fit=fit,
                    retained_routed=retained_routed,
                    teacher=teacher[token],
                    natural_routed=natural_routed[token],
                )
                best_by_count.setdefault(len(subset), []).append(row)
                continue

            if configuration["basis"] == "available_selected":
                base_slots = retained_slots
            elif configuration["basis"] == "resident_selected":
                base_slots = [
                    slot for slot in retained_slots if not bool(miss_mask[token, slot])
                ]
            else:
                raise ValueError(configuration["basis"])
            base_ids = [int(token_route[slot]) for slot in base_slots]
            base_basis = (
                route_outputs[base_slots].T.contiguous()
                if base_slots
                else torch.empty(HIDDEN_SIZE, 0)
            )
            route_set = set(int(value) for value in token_route.tolist())
            cached_ids = [
                int(value)
                for value in cache_before[token].nonzero(as_tuple=False).squeeze(1).tolist()
                if int(value) not in route_set
            ]
            extra_limit = (
                min(
                    MAX_CACHED_EXTRAS,
                    MAX_EXTRA_PER_AVOIDED * len(subset),
                    len(cached_ids),
                )
                if configuration["use_cached"]
                else 0
            )
            cached_basis = (
                all_outputs[token, cached_ids].float().T.contiguous()
                if cached_ids
                else torch.empty(HIDDEN_SIZE, 0)
            )
            omp_order = (
                omp_cached_order(
                    base_basis, cached_basis, target, max_extra=extra_limit
                )
                if extra_limit
                else []
            )
            nested: list[dict[str, Any]] = []
            for extra_count in range(extra_limit + 1):
                chosen_positions = omp_order[:extra_count]
                chosen_ids = [cached_ids[position] for position in chosen_positions]
                basis = (
                    torch.cat(
                        (base_basis, cached_basis[:, chosen_positions]), dim=1
                    )
                    if chosen_positions
                    else base_basis
                )
                fit = fit_span(
                    basis,
                    target,
                    configuration["method"],
                    ridge_relative=RIDGE_RELATIVE,
                    coefficient_bound=COEFFICIENT_BOUND,
                )
                nested.append(
                    candidate_record(
                        token=token,
                        subset_slots=subset,
                        route_ids=token_route,
                        retained_slots=retained_slots,
                        base_ids=base_ids,
                        extra_ids=chosen_ids,
                        fit=fit,
                        retained_routed=retained_routed,
                        teacher=teacher[token],
                        natural_routed=natural_routed[token],
                    )
                )
            best_by_count.setdefault(len(subset), []).append(
                choose_lowest_mse_candidate(nested)
            )
        for avoid_count in sorted(best_by_count):
            rows.append(choose_lowest_mse_candidate(best_by_count[avoid_count]))
    return rows


@torch.inference_mode()
def attach_exact_indexed_kl(
    rows: list[dict[str, Any]],
    reference: Any,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int,
    label: str,
) -> None:
    if not rows:
        return
    states = torch.stack([row["_candidate_state"] for row in rows])
    token_indices = torch.tensor([int(row["token"]) for row in rows])
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_states = states[start:stop].to(lm_head.device)
        indices = token_indices[start:stop]
        logits = F.linear(rmsnorm(batch_states, norm_weight), lm_head).float()
        candidate_log_probs = F.log_softmax(logits, dim=-1)
        teacher_log_probs = reference.log_probs[indices].to(lm_head.device)
        kl = (
            teacher_log_probs.exp() * (teacher_log_probs - candidate_log_probs)
        ).sum(-1).clamp_min(0.0).cpu()
        for offset, value in enumerate(kl.tolist(), start=start):
            rows[offset]["teacher_to_candidate_kl"] = float(value)
        if stop == len(rows) or stop % 512 == 0:
            print(f"{label}_exact_kl={stop}/{len(rows)}", flush=True)


def public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def adjudicate_configuration(
    *,
    configuration: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline_states: torch.Tensor,
    baseline_misses: int,
    miss_mask: torch.Tensor,
    reference: Any,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    block_size: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    by_token: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_token.setdefault(int(row["token"]), []).append(row)
    selected_states = baseline_states.clone()
    selected_records: list[dict[str, Any]] = []
    avoided = 0
    extras = 0
    per_block = []
    for token in range(baseline_states.shape[0]):
        eligible = [
            row
            for row in by_token.get(token, [])
            if math.isfinite(float(row["teacher_to_candidate_kl"]))
            and float(row["teacher_to_candidate_kl"]) <= LOCAL_KL_THRESHOLD
        ]
        chosen = (
            min(
                eligible,
                key=lambda row: (
                    -int(row["avoid_count"]),
                    int(row["extra_computations"]),
                    float(row["teacher_to_candidate_kl"]),
                    float(row["target_squared_error"]),
                ),
            )
            if eligible
            else None
        )
        if chosen is not None:
            selected_states[token] = chosen["_candidate_state"]
            avoided += int(chosen["avoid_count"])
            extras += int(chosen["extra_computations"])
            selected_records.append(public_candidate(chosen))
        else:
            selected_records.append(
                {
                    "token": token,
                    "avoid_count": 0,
                    "extra_computations": 0,
                    "baseline_misses": int(miss_mask[token].sum().item()),
                }
            )
    for block_start in range(0, baseline_states.shape[0], block_size):
        stop = block_start + block_size
        block_records = selected_records[block_start:stop]
        block_baseline = int(miss_mask[block_start:stop].sum().item())
        block_avoided = sum(int(row["avoid_count"]) for row in block_records)
        block_extras = sum(int(row["extra_computations"]) for row in block_records)
        per_block.append(
            {
                "block_start": block_start,
                "baseline_misses": block_baseline,
                "avoided_misses": block_avoided,
                "miss_reduction_fraction": (
                    block_avoided / block_baseline if block_baseline else 0.0
                ),
                "extra_computations": block_extras,
                "extra_computations_per_avoided_load": (
                    block_extras / block_avoided if block_avoided else None
                ),
            }
        )
    quality = evaluate_hidden(
        selected_states,
        reference,
        norm_weight,
        lm_head,
        candidate_batch,
        bootstrap_resamples,
        bootstrap_seed,
    )
    accepted = [row for row in selected_records if int(row["avoid_count"]) > 0]
    report = {
        "configuration": configuration,
        "baseline_misses": baseline_misses,
        "avoided_misses": avoided,
        "remaining_loads": baseline_misses - avoided,
        "miss_reduction_fraction": avoided / baseline_misses if baseline_misses else 0.0,
        "extra_computations": extras,
        "extra_computations_per_avoided_load": extras / avoided if avoided else None,
        "accepted_token_fraction": len(accepted) / baseline_states.shape[0],
        "target_nrmse": numeric_summary(
            [float(row["target_nrmse"]) for row in accepted]
        ),
        "target_cosine": numeric_summary(
            [float(row["target_cosine"]) for row in accepted]
        ),
        "selected_local_kl": numeric_summary(
            [float(row["teacher_to_candidate_kl"]) for row in accepted]
        ),
        "quality": quality,
        "per_block": per_block,
        "selected": selected_records,
        "oracle_candidates": [public_candidate(row) for row in rows],
    }
    return report, selected_states


def select_validation_configuration(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if row["avoided_misses"] > 0
        and row["quality"]["aggregate"]["teacher_to_candidate_kl"]
        <= LOCAL_KL_THRESHOLD
        and row["extra_computations_per_avoided_load"] is not None
        and row["extra_computations_per_avoided_load"] <= MAX_EXTRA_PER_AVOIDED
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -float(row["miss_reduction_fraction"]),
            float(row["extra_computations_per_avoided_load"]),
            float(row["quality"]["aggregate"]["teacher_to_candidate_kl"]),
            int(row["configuration"]["tie_order"]),
        ),
    )


@torch.inference_mode()
def measured_hardware_model(
    moe: LoadedMoELayer, extra_per_avoided: float | None
) -> dict[str, Any]:
    device = moe.device
    weights_per_expert = sum(
        int(weight.numel())
        for weight in (
            moe.experts[0].gate,
            moe.experts[0].up,
            moe.experts[0].down,
        )
    )
    packed_int4_bytes = (weights_per_expert + 1) // 2
    host = torch.empty(packed_int4_bytes, dtype=torch.uint8, pin_memory=True)
    destination = torch.empty_like(host, device=device)
    expert_input = torch.randn(1, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    span_outputs = torch.randn(
        ACTIVE_EXPERTS + MAX_CACHED_EXTRAS,
        HIDDEN_SIZE,
        dtype=torch.float16,
        device=device,
    )
    coefficients = torch.randn(
        ACTIVE_EXPERTS + MAX_CACHED_EXTRAS, dtype=torch.float16, device=device
    )
    for _ in range(15):
        destination.copy_(host, non_blocking=True)
        moe.expert_forward(expert_input, moe.experts[0])
        (span_outputs * coefficients.unsqueeze(1)).sum(0)
    torch.cuda.synchronize()
    repetitions = 7
    transfer_inner = 100
    compute_inner = 100
    combine_inner = 200
    transfer_ms: list[float] = []
    expert_ms: list[float] = []
    combine_ms: list[float] = []

    def elapsed(operation: Any, inner: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            operation()
        stop.record()
        stop.synchronize()
        return float(start.elapsed_time(stop) / inner)

    for _ in range(repetitions):
        transfer_ms.append(
            elapsed(lambda: destination.copy_(host, non_blocking=True), transfer_inner)
        )
        expert_ms.append(
            elapsed(
                lambda: moe.expert_forward(expert_input, moe.experts[0]),
                compute_inner,
            )
        )
        combine_ms.append(
            elapsed(
                lambda: (span_outputs * coefficients.unsqueeze(1)).sum(0),
                combine_inner,
            )
        )
    transfer_median = float(np.median(transfer_ms))
    expert_median = float(np.median(expert_ms))
    combine_median = float(np.median(combine_ms))
    projected_compute = (
        float(extra_per_avoided) * expert_median + combine_median
        if extra_per_avoided is not None
        else float("inf")
    )
    ratio = projected_compute / transfer_median if transfer_median > 0 else float("inf")
    del host, destination, expert_input, span_outputs, coefficients
    torch.cuda.empty_cache()
    return {
        "model": (
            "batch-1 packed-int4 host-to-device expert load versus resident BF16 "
            "expert forward plus ten-vector span combination"
        ),
        "not_a_runtime_claim": True,
        "weights_per_expert": weights_per_expert,
        "packed_int4_bytes_per_avoided_load": packed_int4_bytes,
        "extra_computations_per_avoided_load": extra_per_avoided,
        "repetitions": repetitions,
        "transfer_milliseconds_raw": transfer_ms,
        "resident_expert_forward_milliseconds_raw": expert_ms,
        "span_combine_milliseconds_raw": combine_ms,
        "transfer_milliseconds_median": transfer_median,
        "resident_expert_forward_milliseconds_median": expert_median,
        "span_combine_milliseconds_median": combine_median,
        "projected_compute_milliseconds_per_avoided_load": projected_compute,
        "compute_over_avoided_transfer_time": ratio,
        "passes_compute_lt_transfer": bool(ratio < 1.0),
    }


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("H8 cache-span oracle requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    device = torch.device("cuda")
    model_dir = ROOT / "models/deepseek-v2-lite"
    component_path = ROOT / COMPONENT_RELATIVE
    preregistration = ROOT / "reports/craft_moe/H8_CACHE_SPAN_LAYER26_PREREGISTRATION.md"
    baseline_source = (
        ROOT / "reports/baseline/preregistered_wikitext_offset4096_mass_budget_confirmation.json"
    )
    for path in (model_dir, component_path, preregistration, baseline_source):
        if not path.exists():
            raise FileNotFoundError(path)
    input_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in (
            component_path,
            preregistration,
            baseline_source,
            model_dir / "config.json",
            model_dir / "model.safetensors.index.json",
        )
    }
    repository = git_state()
    initial_hardware = hardware_state()
    splits = ("validation",) if args.stage == "smoke" else ("validation", "test")
    block_size = (
        args.tokens_per_split if args.stage == "smoke" else FULL_BLOCK_SIZE
    )

    phase = time.perf_counter()
    components_all = load_file(component_path, device="cpu")
    indices: list[int] = []
    trace_indices: dict[str, list[int]] = {}
    split_slices: dict[str, slice] = {}
    offset = 0
    for split in splits:
        base = 0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        chosen = list(range(base, base + args.tokens_per_split))
        trace_indices[split] = chosen
        indices.extend(chosen)
        split_slices[split] = slice(offset, offset + args.tokens_per_split)
        offset += args.tokens_per_split
    index = torch.tensor(indices, dtype=torch.long)
    components = {
        key: components_all[key].index_select(0, index)
        for key in ("moe_input", "router_ids", "router_weights", "teacher")
    }
    components["trace_indices"] = index
    del components_all
    moe = load_moe_layer(model_dir, LAYER, device)
    timings["load_inputs_and_layer_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    if args.reuse_capture:
        capture = load_file(args.output_capture, device="cpu")
        expected_tokens = len(splits) * args.tokens_per_split
        if capture["all_expert_outputs"].shape != (
            expected_tokens,
            EXPERTS,
            HIDDEN_SIZE,
        ):
            raise RuntimeError("reused capture shape does not match this run")
        if not torch.equal(capture["trace_indices"], index):
            raise RuntimeError("reused capture trace indices do not match")
        capture_controls: dict[str, Any] = {
            "reused_existing_capture": True,
            "shape_and_trace_indices_exact": True,
        }
    else:
        capture, capture_controls = build_capture(
            moe, components, split_slices, block_size
        )
        args.output_capture.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            capture,
            args.output_capture,
            metadata={
                "model_revision": MODEL_REVISION,
                "dataset_revision": DATASET_REVISION,
                "layer": str(LAYER),
                "stage": args.stage,
                "splits": ",".join(splits),
                "tokens_per_split": str(args.tokens_per_split),
                "mass_budget_policy": "mass_budget:j2:0.004",
                "capacity": str(CACHE_CAPACITY),
                "cache_semantics": "baseline pre-touch ghost cache",
            },
        )
    capture_hash = sha256_file(args.output_capture)
    timings["capture_or_reuse_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    references: dict[str, Any] = {}
    baseline_states: dict[str, torch.Tensor] = {}
    baseline_reports: dict[str, Any] = {}
    for split, sl in split_slices.items():
        token_ids = corpus_tokens(model_dir, split, args.tokens_per_split)
        references[split] = make_teacher_reference(
            capture["teacher"][sl],
            token_ids,
            sequence_blocks(args.tokens_per_split),
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        baseline_states[split] = anchored_candidate(
            capture["teacher"][sl],
            capture["mass_budget_routed"][sl],
            capture["natural_routed"][sl],
        )
        baseline_reports[split] = evaluate_hidden(
            baseline_states[split],
            references[split],
            norm_weight,
            lm_head,
            args.candidate_batch,
            args.bootstrap_resamples,
            SEED + (0 if split == "validation" else 1),
        )
    original = anchored_candidate(
        capture["teacher"], capture["natural_routed"], capture["natural_routed"]
    )
    original_bit_exact = torch.equal(original, capture["teacher"])
    if not original_bit_exact:
        raise RuntimeError("post-capture original control failed")
    timings["teacher_references_and_baseline_seconds"] = time.perf_counter() - phase

    configurations = configuration_rows()
    zero_configuration = {
        "name": "zero_fill",
        "basis": "none",
        "use_cached": False,
        "method": "zero",
        "tie_order": -1,
    }
    validation_results: list[dict[str, Any]] = []
    phase = time.perf_counter()
    validation_slice = split_slices["validation"]
    validation_misses = int(capture["miss_mask"][validation_slice].sum().item())
    for config_index, configuration in enumerate([zero_configuration, *configurations]):
        print(
            f"validation_configuration={config_index + 1}/{len(configurations) + 1} "
            f"name={configuration['name']}",
            flush=True,
        )
        rows = build_configuration_candidates(
            configuration=configuration,
            all_outputs=capture["all_expert_outputs"][validation_slice],
            route_ids=capture["mass_budget_ids"][validation_slice],
            route_weights=capture["mass_budget_weights"][validation_slice],
            cache_before=capture["cache_before"][validation_slice],
            miss_mask=capture["miss_mask"][validation_slice],
            teacher=capture["teacher"][validation_slice],
            natural_routed=capture["natural_routed"][validation_slice],
        )
        attach_exact_indexed_kl(
            rows,
            references["validation"],
            norm_weight,
            lm_head,
            args.candidate_batch,
            configuration["name"],
        )
        adjudicated, _ = adjudicate_configuration(
            configuration=configuration,
            rows=rows,
            baseline_states=baseline_states["validation"],
            baseline_misses=validation_misses,
            miss_mask=capture["miss_mask"][validation_slice],
            reference=references["validation"],
            norm_weight=norm_weight,
            lm_head=lm_head,
            candidate_batch=args.candidate_batch,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=SEED + 10 + config_index,
            block_size=block_size,
        )
        validation_results.append(adjudicated)
        print(
            f"validation_result name={configuration['name']} "
            f"reduction={adjudicated['miss_reduction_fraction']:.6f} "
            f"kl={adjudicated['quality']['aggregate']['teacher_to_candidate_kl']:.6g}",
            flush=True,
        )
    zero_validation = validation_results[0]
    selected_validation = select_validation_configuration(validation_results[1:])
    timings["validation_oracle_seconds"] = time.perf_counter() - phase

    test_results: dict[str, Any] | None = None
    zero_test: dict[str, Any] | None = None
    if "test" in splits and selected_validation is not None:
        selected_name = selected_validation["configuration"]["name"]
        selected_configuration = next(
            row for row in configurations if row["name"] == selected_name
        )
        print(f"validation_selected={selected_name}", flush=True)
        phase = time.perf_counter()
        test_slice = split_slices["test"]
        test_misses = int(capture["miss_mask"][test_slice].sum().item())
        evaluated_test = []
        for config_index, configuration in enumerate(
            (zero_configuration, selected_configuration)
        ):
            print(f"test_configuration={configuration['name']}", flush=True)
            rows = build_configuration_candidates(
                configuration=configuration,
                all_outputs=capture["all_expert_outputs"][test_slice],
                route_ids=capture["mass_budget_ids"][test_slice],
                route_weights=capture["mass_budget_weights"][test_slice],
                cache_before=capture["cache_before"][test_slice],
                miss_mask=capture["miss_mask"][test_slice],
                teacher=capture["teacher"][test_slice],
                natural_routed=capture["natural_routed"][test_slice],
            )
            attach_exact_indexed_kl(
                rows,
                references["test"],
                norm_weight,
                lm_head,
                args.candidate_batch,
                configuration["name"],
            )
            adjudicated, _ = adjudicate_configuration(
                configuration=configuration,
                rows=rows,
                baseline_states=baseline_states["test"],
                baseline_misses=test_misses,
                miss_mask=capture["miss_mask"][test_slice],
                reference=references["test"],
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=SEED + 100 + config_index,
                block_size=block_size,
            )
            evaluated_test.append(adjudicated)
        zero_test, test_results = evaluated_test
        timings["heldout_test_oracle_seconds"] = time.perf_counter() - phase

    primary_for_hardware = test_results or selected_validation
    extra_ratio = (
        primary_for_hardware["extra_computations_per_avoided_load"]
        if primary_for_hardware is not None
        else None
    )
    phase = time.perf_counter()
    hardware_model = measured_hardware_model(moe, extra_ratio)
    timings["hardware_model_seconds"] = time.perf_counter() - phase

    controls_pass = bool(original_bit_exact)
    if not args.reuse_capture:
        controls_pass = controls_pass and all(
            (
                capture_controls["official_router"]["slot_order_ids_exact"],
                capture_controls["official_router"][
                    "router_weight_maximum_absolute_error"
                ]
                <= ROUTER_WEIGHT_TOLERANCE,
                capture_controls["sorted_top6_set_exact"],
                capture_controls["dense_router_weight_maximum_absolute_error"]
                <= ROUTER_WEIGHT_TOLERANCE,
                capture_controls["all_outputs_finite"],
                all(
                    row["route_ids_exact"] and row["expert_loads_exact"]
                    for row in capture_controls["mass_budget_independent_audit"].values()
                ),
            )
        )
    gates: dict[str, Any] = {
        "exact_controls_pass": controls_pass,
        "validation_candidate_selected_without_test": selected_validation is not None,
        "miss_reduction_ge_0_50": {"validation": False, "test": None},
        "mean_kl_le_0_001": {"validation": False, "test": None},
        "extra_compute_per_avoided_le_2": {"validation": False, "test": None},
        "span_uplift_over_zero_fill_ge_0_10": {"validation": False, "test": None},
        "projected_compute_lt_avoided_transfer": hardware_model[
            "passes_compute_lt_transfer"
        ],
        "full_depth_ce_lt_0_02": {
            "evaluated": False,
            "reason": "opens only after positive layer-26 oracle and causal predictor",
        },
    }
    if selected_validation is not None:
        gates["miss_reduction_ge_0_50"]["validation"] = (
            selected_validation["miss_reduction_fraction"] >= 0.50
        )
        gates["mean_kl_le_0_001"]["validation"] = (
            selected_validation["quality"]["aggregate"]["teacher_to_candidate_kl"]
            <= LOCAL_KL_THRESHOLD
        )
        gates["extra_compute_per_avoided_le_2"]["validation"] = (
            selected_validation["extra_computations_per_avoided_load"]
            <= MAX_EXTRA_PER_AVOIDED
        )
        gates["span_uplift_over_zero_fill_ge_0_10"]["validation"] = (
            selected_validation["miss_reduction_fraction"]
            - zero_validation["miss_reduction_fraction"]
            >= 0.10
        )
    if test_results is not None and zero_test is not None:
        gates["miss_reduction_ge_0_50"]["test"] = (
            test_results["miss_reduction_fraction"] >= 0.50
        )
        gates["mean_kl_le_0_001"]["test"] = (
            test_results["quality"]["aggregate"]["teacher_to_candidate_kl"]
            <= LOCAL_KL_THRESHOLD
        )
        gates["extra_compute_per_avoided_le_2"]["test"] = (
            test_results["extra_computations_per_avoided_load"]
            <= MAX_EXTRA_PER_AVOIDED
        )
        gates["span_uplift_over_zero_fill_ge_0_10"]["test"] = (
            test_results["miss_reduction_fraction"]
            - zero_test["miss_reduction_fraction"]
            >= 0.10
        )
    screen_gate_values = [
        gates["exact_controls_pass"],
        gates["validation_candidate_selected_without_test"],
        gates["miss_reduction_ge_0_50"]["validation"],
        gates["mean_kl_le_0_001"]["validation"],
        gates["extra_compute_per_avoided_le_2"]["validation"],
        gates["span_uplift_over_zero_fill_ge_0_10"]["validation"],
        gates["projected_compute_lt_avoided_transfer"],
    ]
    if test_results is not None:
        screen_gate_values.extend(
            (
                gates["miss_reduction_ge_0_50"]["test"],
                gates["mean_kl_le_0_001"]["test"],
                gates["extra_compute_per_avoided_le_2"]["test"],
                gates["span_uplift_over_zero_fill_ge_0_10"]["test"],
            )
        )
    screen_positive = bool(all(screen_gate_values)) and args.stage == "full"
    if screen_positive:
        verdict = "oracle_positive_requires_causal_cache_and_predictor"
    elif not controls_pass:
        verdict = "invalid_exact_control"
    elif test_results is not None and (
        test_results["miss_reduction_fraction"] < 0.40
        or test_results["quality"]["aggregate"]["teacher_to_candidate_kl"]
        > LOCAL_KL_THRESHOLD
        or test_results["miss_reduction_fraction"]
        - zero_test["miss_reduction_fraction"]
        <= 0.0
    ):
        verdict = "falsified_optimistic_ghost_cache_oracle"
    elif args.stage == "smoke":
        verdict = "smoke_only_not_adjudicated"
    else:
        verdict = "inconclusive_negative_layer26_screen"

    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report = {
        "schema_version": 1,
        "kind": "craft_moe_h8_cache_span_layer26_oracle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "verdict": verdict,
        "experiment": "H8_CACHE_SPAN",
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "layer": LAYER,
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "splits": list(splits),
            "tokens_per_split": args.tokens_per_split,
            "trace_indices": trace_indices,
        },
        "protocol": {
            "cache": "baseline pre-touch ghost cache; reset per sequence block",
            "block_size": block_size,
            "capacity": CACHE_CAPACITY,
            "mass_budget_policy": "mass_budget:j2:0.004",
            "delta_average_fixed_from_prior_validation": DELTA_AVERAGE,
            "local_kl_threshold": LOCAL_KL_THRESHOLD,
            "max_cached_extras": MAX_CACHED_EXTRAS,
            "max_extra_per_avoided_load": MAX_EXTRA_PER_AVOIDED,
            "ridge_relative": RIDGE_RELATIVE,
            "coefficient_bound": [-COEFFICIENT_BOUND, COEFFICIENT_BOUND],
            "router_weight_absolute_tolerance": ROUTER_WEIGHT_TOLERANCE,
            "validation_selects_one_global_configuration": True,
            "test_selection": "none; fixed validation winner only",
        },
        "controls": capture_controls
        | {
            "post_capture_original_teacher_delta_bit_exact": original_bit_exact,
            "all_adjudicated_candidates_finite": True,
        },
        "mass_budget_baseline_quality": baseline_reports,
        "validation": {
            "zero_fill": zero_validation,
            "all_span_configurations": validation_results[1:],
            "selected": selected_validation,
        },
        "heldout_test": {
            "opened": test_results is not None,
            "zero_fill": zero_test,
            "selected_configuration": test_results,
        },
        "hardware_model": hardware_model,
        "gates": gates,
        "screen_positive": screen_positive,
        "capture_artifact": {
            "path": str(args.output_capture),
            "bytes": args.output_capture.stat().st_size,
            "sha256": capture_hash,
            "metadata": metadata(args.output_capture),
        },
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "seed": SEED,
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "safetensors": safetensors.__version__,
                "psutil": psutil.__version__,
            },
            "inputs_sha256": input_hashes,
            "component_metadata": metadata(component_path),
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "the primary oracle uses the true missing output signature and cannot be deployed",
            "ghost-cache state grants future residency even after a load is counted as avoided",
            "the per-token subset decision sees exact full-vocabulary KL",
            "only layer 26 and 256-token exploratory windows are evaluated in the full screen",
            "the hardware result is a batch-1 microbenchmark model, not packed end-to-end runtime",
            "full-depth CE, causal cache evolution, coefficient prediction, and autoregressive stability remain unopened unless every screen gate passes",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}")
    print(f"capture={args.output_capture}")
    print(f"verdict={verdict}")
    if selected_validation is not None:
        print(
            "validation_selected="
            f"{selected_validation['configuration']['name']} "
            f"reduction={selected_validation['miss_reduction_fraction']:.6f} "
            f"kl={selected_validation['quality']['aggregate']['teacher_to_candidate_kl']:.6g}"
        )
    if test_results is not None:
        print(
            f"test_reduction={test_results['miss_reduction_fraction']:.6f} "
            f"test_kl={test_results['quality']['aggregate']['teacher_to_candidate_kl']:.6g}"
        )


if __name__ == "__main__":
    main()
