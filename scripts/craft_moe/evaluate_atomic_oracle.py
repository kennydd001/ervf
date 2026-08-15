from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import math
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
from safetensors.torch import load_file

from evaluate_crcq_oracle import (
    BLOCK_SIZE,
    COMPONENT_RELATIVE,
    DATASET_REVISION,
    MODEL_REVISION,
    SEED,
    TRACE_TOKENS_PER_SPLIT,
    corpus_tokens,
    evaluate_hidden,
    git_state,
    hardware_state,
    make_teacher_reference,
    metadata,
    nullable,
    percentile,
    regression_summary,
    sequence_blocks,
    sha256_file,
    write_json_once,
)
from moe_lab.craft_moe.atomic import (
    atomic_selector_masks,
    delta_patched_hidden,
    relative_routed_l2,
    support_known_accounting,
)
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT


FULL_TOKENS_PER_SPLIT = 256
SMOKE_TOKENS = 32
BOOTSTRAP_RESAMPLES = 10_000
CANDIDATE_BATCH = 128
POLICY_BATCH = 4
LAYER = 26
HIDDEN_SIZE = 2048
ATOMS_PER_EXPERT = 1408
ACTIVE_EXPERTS = 6
FRACTIONS = (1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05)
METHODS = (
    "per_expert_activation",
    "per_expert_contribution",
    "global_contribution",
    "tile16_contribution",
    "tile32_contribution",
    "tile64_contribution",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact layer-26 atomic routed-expert oracle screen."
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
    parser.add_argument("--candidate-batch", type=int, default=CANDIDATE_BATCH)
    parser.add_argument("--policy-batch", type=int, default=POLICY_BATCH)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if not 1 <= args.tokens_per_split <= TRACE_TOKENS_PER_SPLIT:
        raise ValueError("tokens-per-split is outside the existing trace")
    if args.stage == "smoke":
        if args.splits != ("validation",) or args.tokens_per_split > SMOKE_TOKENS:
            raise ValueError("smoke is limited to at most 32 validation tokens")
    elif (
        args.tokens_per_split != FULL_TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
        or args.candidate_batch != CANDIDATE_BATCH
        or args.policy_batch != POLICY_BATCH
    ):
        raise ValueError(
            "full is fixed at 256 validation + 256 test, 10k bootstrap, "
            "candidate batch 128, policy batch 4, and the preregistered seed"
        )
    if args.candidate_batch < 1 or args.policy_batch < 1:
        raise ValueError("batch sizes must be positive")
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    if args.output_json is None:
        relative = (
            Path("reports/craft_moe/atomic_oracle.json")
            if args.stage == "full"
            else Path("reports/runs/craft_moe/atomic_layer26_smoke.json")
        )
        args.output_json = ROOT / relative
    elif not args.output_json.is_absolute():
        args.output_json = ROOT / args.output_json
    args.output_json = args.output_json.resolve()
    reports = (ROOT / "reports").resolve()
    if reports not in args.output_json.parents:
        raise ValueError("output-json must be inside reports/")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output_json}")
    return args


def policy_id(method: str, fraction: float) -> str:
    return f"{method}__f{fraction:.3f}".replace(".", "p")


def numeric_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().double().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("summary values must be non-empty and finite")
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


def block_bootstrap_mean(
    values: torch.Tensor,
    blocks: list[tuple[int, int]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    array = values.detach().cpu().double().numpy()
    sums = np.asarray([array[start:stop].sum() for start, stop in blocks])
    counts = np.asarray([stop - start for start, stop in blocks])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(resamples, len(blocks)))
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "method": "paired sequence-block percentile bootstrap",
        "confidence": 0.95,
        "resamples": resamples,
        "seed": seed,
        "interval": percentile(estimates),
    }


def packed_mask_record(mask: torch.Tensor) -> dict[str, Any]:
    if mask.ndim != 3 or mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean [tokens, experts, atoms]")
    flattened = mask.detach().cpu().numpy().reshape(mask.shape[0], -1)
    packed = np.packbits(flattened, axis=1, bitorder="little")
    payload = packed.tobytes(order="C")
    return {
        "encoding": "base64(numpy.packbits(bitorder=little, axis=flattened_atoms))",
        "logical_shape": list(mask.shape),
        "flattened_order": "token, expert_slot, neuron",
        "packed_row_bytes": int(packed.shape[1]),
        "sha256_packed": hashlib.sha256(payload).hexdigest(),
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


@torch.inference_mode()
def exact_activations_and_outputs(
    moe: LoadedMoELayer,
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact SwiGLU activations, selected outputs, and down norms."""

    if inputs.ndim != 2 or inputs.shape[1] != HIDDEN_SIZE:
        raise ValueError("inputs must have shape [tokens, 2048]")
    if expert_ids.shape != (inputs.shape[0], ACTIVE_EXPERTS):
        raise ValueError("expert_ids must have shape [tokens, 6]")
    tokens = inputs.shape[0]
    activations = torch.empty(
        tokens, ACTIVE_EXPERTS, ATOMS_PER_EXPERT, dtype=inputs.dtype
    )
    selected = torch.empty(
        tokens, ACTIVE_EXPERTS, HIDDEN_SIZE, dtype=inputs.dtype
    )
    down_norms = torch.empty(len(moe.experts), ATOMS_PER_EXPERT, dtype=torch.float32)
    touched = 0
    for expert_id, expert in enumerate(moe.experts):
        down_norms[expert_id] = expert.down.float().square().sum(dim=0).sqrt().cpu()
        positions = (expert_ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            touched += 1
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            x = inputs[token_indices].to(moe.device)
            activation = F.silu(F.linear(x, expert.gate)) * F.linear(x, expert.up)
            output = F.linear(activation, expert.down)
            activations[token_indices, slot_indices] = activation.cpu()
            selected[token_indices, slot_indices] = output.cpu()
        if expert_id % 8 == 7:
            print(f"atomic_activation_experts={expert_id + 1}/64", flush=True)
    if not torch.isfinite(activations.float()).all():
        raise RuntimeError("non-finite exact atomic activations")
    if not torch.isfinite(selected.float()).all():
        raise RuntimeError("non-finite exact selected output")
    print(f"atomic_experts_touched={touched}/64", flush=True)
    return activations, selected, down_norms


def release_non_down_weights(moe: LoadedMoELayer) -> None:
    """Keep only routed-expert down matrices needed by atom reconstruction."""

    empty = torch.empty(0)
    moe.gate_weight = empty
    moe.shared.gate = empty
    moe.shared.up = empty
    moe.shared.down = empty
    for expert in moe.experts:
        expert.gate = empty
        expert.up = empty
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def reconstruct_policy_masks(
    moe: LoadedMoELayer,
    activations: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    masks: list[torch.Tensor],
    policy_batch: int,
) -> torch.Tensor:
    """Dense-evaluation implementation of exact masked atom sums.

    Zero-masked dense GEMMs are used only to evaluate quality. They are not a
    packed sparse runtime and their wall time is never interpreted as speedup.
    """

    if not masks:
        raise ValueError("at least one policy mask is required")
    expected = activations.shape
    for mask in masks:
        if mask.shape != expected or mask.dtype is not torch.bool:
            raise ValueError("every policy mask must be boolean and match activations")
    tokens, active_experts, atoms = activations.shape
    if expert_ids.shape != (tokens, active_experts):
        raise ValueError("expert_ids must match activation token/expert dimensions")
    if router_weights.shape != (tokens, active_experts):
        raise ValueError("router_weights must match activation token/expert dimensions")
    device = moe.experts[0].down.device
    hidden = moe.experts[0].down.shape[0]
    result = torch.empty(len(masks), tokens, hidden, dtype=activations.dtype)
    weights_device = router_weights.to(device).float()
    for policy_start in range(0, len(masks), policy_batch):
        policy_stop = min(policy_start + policy_batch, len(masks))
        batch_masks = masks[policy_start:policy_stop]
        batch = len(batch_masks)
        selected = torch.empty(
            batch,
            tokens,
            active_experts,
            hidden,
            dtype=activations.dtype,
            device=device,
        )
        for expert_id, expert in enumerate(moe.experts):
            positions = (expert_ids == expert_id).nonzero(as_tuple=False)
            if not positions.numel():
                continue
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            exact = activations[token_indices, slot_indices].to(device)
            selected_masks = torch.stack(
                [mask[token_indices, slot_indices] for mask in batch_masks]
            ).to(device)
            sparse = torch.where(
                selected_masks,
                exact.unsqueeze(0),
                torch.zeros((), dtype=exact.dtype, device=exact.device),
            )
            output = F.linear(
                sparse.reshape(batch * positions.shape[0], atoms), expert.down
            ).reshape(batch, positions.shape[0], hidden)
            selected[:, token_indices, slot_indices] = output
        routed = (
            selected.float() * weights_device.unsqueeze(0).unsqueeze(-1)
        ).sum(dim=2).to(activations.dtype)
        result[policy_start:policy_stop] = routed.cpu()
        print(
            f"atomic_reconstructed_policies={policy_stop}/{len(masks)}",
            flush=True,
        )
        del selected, routed
    return result


def build_policies(
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    selected_down_norms: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    control = torch.ones_like(activations, dtype=torch.bool)
    specifications: list[dict[str, Any]] = [
        {
            "id": policy_id("exact_all_atoms", 1.0),
            "method": "exact_all_atoms",
            "requested_fraction": 1.0,
        }
    ]
    masks = [control]
    for fraction in FRACTIONS[1:]:
        selected = atomic_selector_masks(
            activations, router_weights, selected_down_norms, fraction
        )
        for method in METHODS:
            mask = selected[method]
            specifications.append(
                {
                    "id": policy_id(method, fraction),
                    "method": method,
                    "requested_fraction": fraction,
                }
            )
            masks.append(mask)
    return specifications, masks


def accounting_summary(accounting: dict[str, Any]) -> dict[str, Any]:
    raw_keys = (
        "retained_atoms",
        "retained_atom_fraction",
        "ideal_weight_bytes",
        "ideal_weight_byte_fraction",
        "ideal_macs",
        "ideal_mac_fraction",
        "tensor_local_pages_4k",
        "tensor_local_page_bytes",
        "tensor_local_page_byte_fraction",
    )
    summary = {
        key: numeric_summary(torch.tensor(accounting[key], dtype=torch.float64))
        for key in raw_keys
    }
    return {
        "assumption": accounting["assumption"],
        "full_routed_atoms": accounting["full_routed_atoms"],
        "full_ideal_weight_bytes": accounting["full_ideal_weight_bytes"],
        "full_ideal_macs": accounting["full_ideal_macs"],
        "full_tensor_local_pages_4k": accounting["full_tensor_local_pages_4k"],
        "full_tensor_local_page_bytes": accounting[
            "full_tensor_local_page_bytes"
        ],
        "aggregate": summary,
        "raw": {key: accounting[key] for key in raw_keys},
    }


def evaluate_policy(
    candidate_routed: torch.Tensor,
    original_routed: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    reference: Any,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    local = relative_routed_l2(original_routed, candidate_routed)
    patched = delta_patched_hidden(teacher, original_routed, candidate_routed)
    metrics = evaluate_hidden(
        patched,
        reference,
        norm_weight,
        lm_head,
        candidate_batch,
        bootstrap_resamples,
        seed,
    )
    counts = mask.sum(dim=2).to(torch.int64)
    accounting = support_known_accounting(
        counts,
        atoms_per_expert=ATOMS_PER_EXPERT,
        hidden_size=HIDDEN_SIZE,
    )
    return {
        "local_routed_relative_l2": {
            "aggregate": numeric_summary(local),
            "bootstrap_95_mean": block_bootstrap_mean(
                local, reference.blocks, bootstrap_resamples, seed
            ),
            "raw": local.tolist(),
        },
        "full_model": metrics,
        "support": packed_mask_record(mask),
        "accounting": accounting_summary(accounting),
    }


def record_at(
    split_results: dict[str, Any], method: str, fraction: float
) -> dict[str, Any]:
    selected_id = (
        policy_id("exact_all_atoms", 1.0)
        if fraction == 1.0
        else policy_id(method, fraction)
    )
    return split_results["policies"][selected_id]


def adjudicate(results: dict[str, Any], stage: str) -> tuple[str, dict[str, Any]]:
    if stage == "smoke":
        return "smoke_passed_not_adjudicated", {
            "adjudicated": False,
            "reason": "the preregistered gates require fixed validation and test windows",
        }

    primary_by_split = {}
    moonshot_by_split = {}
    tile64_by_split = {}
    neuron_25_by_split: dict[str, Any] = {}
    for split in ("validation", "test"):
        global_25 = record_at(results[split], "global_contribution", 0.25)
        global_10 = record_at(results[split], "global_contribution", 0.10)
        tile64_25 = record_at(results[split], "tile64_contribution", 0.25)
        global_ce25 = global_25["full_model"]["aggregate"][
            "relative_cross_entropy_delta"
        ]
        global_ce10 = global_10["full_model"]["aggregate"][
            "relative_cross_entropy_delta"
        ]
        global_kl = global_25["full_model"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        tile_kl = tile64_25["full_model"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        primary_by_split[split] = {
            "relative_cross_entropy_delta": global_ce25,
            "passes_lt_0_02": global_ce25 < 0.02,
        }
        moonshot_by_split[split] = {
            "relative_cross_entropy_delta": global_ce10,
            "passes_lt_0_03": global_ce10 < 0.03,
        }
        ratio = tile_kl / global_kl if global_kl > 0 else None
        tile64_by_split[split] = {
            "global_neuron_mean_kl": global_kl,
            "tile64_mean_kl": tile_kl,
            "kl_ratio": ratio,
            "absolute_kl_difference": tile_kl - global_kl,
            "passes_le_1_20x": tile_kl <= 1.20 * global_kl,
        }
        neuron_records = {
            method: record_at(results[split], method, 0.25)
            for method in METHODS[:3]
        }
        neuron_25_by_split[split] = {
            method: {
                "relative_cross_entropy_delta": record["full_model"]["aggregate"][
                    "relative_cross_entropy_delta"
                ],
                "mean_kl": record["full_model"]["aggregate"][
                    "teacher_to_candidate_kl"
                ],
            }
            for method, record in neuron_records.items()
        }

    primary = all(item["passes_lt_0_02"] for item in primary_by_split.values())
    moonshot = all(item["passes_lt_0_03"] for item in moonshot_by_split.values())
    tile64 = all(item["passes_le_1_20x"] for item in tile64_by_split.values())
    validation_neurons = neuron_25_by_split["validation"]
    no_neuron_ce_pass = all(
        item["relative_cross_entropy_delta"] >= 0.02
        for item in validation_neurons.values()
    )
    best_validation_kl = min(item["mean_kl"] for item in validation_neurons.values())
    hard_stop = no_neuron_ce_pass and best_validation_kl > 0.01
    if primary:
        verdict = "oracle_positive_opens_depth_and_domain_expansion"
    elif hard_stop:
        verdict = "oracle_negative_hard_stop"
    else:
        verdict = "inconclusive_negative_no_expansion"
    gates = {
        "adjudicated": True,
        "primary_global_25pct_relative_ce_lt_2pct_both_splits": primary,
        "primary_by_split": primary_by_split,
        "moonshot_global_10pct_relative_ce_lt_3pct_both_splits": moonshot,
        "moonshot_by_split": moonshot_by_split,
        "tile64_25pct_kl_le_1_20x_global_both_splits": tile64,
        "tile64_by_split": tile64_by_split,
        "neuron_selectors_25pct": neuron_25_by_split,
        "hard_stop": {
            "triggered": hard_stop,
            "no_validation_neuron_selector_passes_ce_gate": no_neuron_ce_pass,
            "best_validation_neuron_mean_kl": best_validation_kl,
            "best_validation_neuron_mean_kl_gt_0_01": best_validation_kl > 0.01,
        },
    }
    return verdict, gates


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("the exact atomic oracle requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    model_dir = ROOT / "models/deepseek-v2-lite"
    component_path = ROOT / COMPONENT_RELATIVE
    for path in (model_dir, component_path):
        if not path.exists():
            raise FileNotFoundError(path)

    phase = time.perf_counter()
    input_hashes = {
        str(component_path.resolve()): sha256_file(component_path),
        str((model_dir / "config.json").resolve()): sha256_file(
            model_dir / "config.json"
        ),
        str((model_dir / "model.safetensors.index.json").resolve()): sha256_file(
            model_dir / "model.safetensors.index.json"
        ),
    }
    timings["input_sha256_seconds"] = time.perf_counter() - phase
    initial_hardware = hardware_state()
    repository = git_state()

    phase = time.perf_counter()
    component_all = load_file(component_path, device="cpu")
    trace_indices: dict[str, list[int]] = {}
    indices: list[int] = []
    for split in args.splits:
        base = 0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        selected = list(range(base, base + args.tokens_per_split))
        trace_indices[split] = selected
        indices.extend(selected)
    index = torch.tensor(indices, dtype=torch.long)
    needed = ("moe_input", "router_ids", "router_weights", "teacher")
    components = {
        key: component_all[key].index_select(0, index) for key in needed
    }
    del component_all
    token_ids = {
        split: corpus_tokens(model_dir, split, args.tokens_per_split)
        for split in args.splits
    }
    timings["load_inputs_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    moe = load_moe_layer(model_dir, LAYER, device)
    recomputed_ids, recomputed_weights = moe.route(components["moe_input"].to(device))
    route_control = {
        "slot_order_ids_exact": bool(
            torch.equal(recomputed_ids.cpu(), components["router_ids"].long())
        ),
        "set_ids_exact": bool(
            torch.equal(
                recomputed_ids.cpu().sort(dim=1).values,
                components["router_ids"].long().sort(dim=1).values,
            )
        ),
        "router_weight_max_absolute_error_in_trace_slot_order": None,
    }
    if route_control["slot_order_ids_exact"]:
        route_control["router_weight_max_absolute_error_in_trace_slot_order"] = float(
            (
                recomputed_weights.cpu().float()
                - components["router_weights"].float()
            )
            .abs()
            .max()
            .item()
        )
    activations, selected_outputs, down_norm_bank = exact_activations_and_outputs(
        moe, components["moe_input"], components["router_ids"].long()
    )
    direct_routed = (
        selected_outputs.float()
        * components["router_weights"].float().unsqueeze(-1)
    ).sum(dim=1).to(selected_outputs.dtype)
    selected_down_norms = down_norm_bank[components["router_ids"].long()]
    timings["load_layer_route_and_exact_activations_seconds"] = (
        time.perf_counter() - phase
    )

    phase = time.perf_counter()
    specifications, masks = build_policies(
        activations,
        components["router_weights"].float(),
        selected_down_norms,
    )
    timings["selector_masks_seconds"] = time.perf_counter() - phase
    del selected_down_norms, down_norm_bank
    release_non_down_weights(moe)

    phase = time.perf_counter()
    exact_routed = reconstruct_policy_masks(
        moe,
        activations,
        components["router_ids"].long(),
        components["router_weights"].float(),
        [masks[0]],
        1,
    )[0]
    decomposition_regression = regression_summary(direct_routed, exact_routed)
    decomposition_bit_exact = torch.equal(direct_routed, exact_routed)
    if (
        decomposition_regression["nrmse"] > 1e-4
        or decomposition_regression["maximum_absolute_error"] > 0.01
    ):
        raise RuntimeError(
            "the 100% atom decomposition exceeded the fixed BF16 kernel regression "
            f"tolerance: {decomposition_regression}"
        )
    candidate_routed = reconstruct_policy_masks(
        moe,
        activations,
        components["router_ids"].long(),
        components["router_weights"].float(),
        masks[1:],
        args.policy_batch,
    )
    all_routed = torch.cat((exact_routed.unsqueeze(0), candidate_routed), dim=0)
    del candidate_routed, selected_outputs, direct_routed, activations, moe
    gc.collect()
    torch.cuda.empty_cache()
    timings["dense_masked_quality_reconstruction_seconds"] = (
        time.perf_counter() - phase
    )

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    timings["load_final_projection_seconds"] = time.perf_counter() - phase

    split_slices: dict[str, slice] = {}
    references: dict[str, Any] = {}
    offset = 0
    for split in args.splits:
        selected = slice(offset, offset + args.tokens_per_split)
        split_slices[split] = selected
        references[split] = make_teacher_reference(
            components["teacher"][selected],
            token_ids[split],
            sequence_blocks(args.tokens_per_split),
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        offset += args.tokens_per_split

    phase = time.perf_counter()
    results: dict[str, Any] = {}
    for split_index, split in enumerate(args.splits):
        selected = split_slices[split]
        original = all_routed[0, selected]
        split_results: dict[str, Any] = {
            "teacher_reference": {
                "token_ids": token_ids[split].tolist(),
                "true_token_nll": nullable(references[split].true_token_nll),
                "sequence_blocks": [list(block) for block in references[split].blocks],
            },
            "policies": {},
        }
        for policy_index, specification in enumerate(specifications):
            record = evaluate_policy(
                all_routed[policy_index, selected],
                original,
                components["teacher"][selected],
                masks[policy_index][selected],
                references[split],
                norm_weight,
                lm_head,
                args.candidate_batch,
                args.bootstrap_resamples,
                args.seed + split_index,
            )
            record["method"] = specification["method"]
            record["requested_fraction"] = specification["requested_fraction"]
            split_results["policies"][specification["id"]] = record
            print(
                f"atomic_evaluated[{split}]={policy_index + 1}/{len(specifications)}",
                flush=True,
            )
        control = split_results["policies"][policy_id("exact_all_atoms", 1.0)]
        control_metrics = control["full_model"]
        if (
            max(control_metrics["raw"]["teacher_to_candidate_kl"]) != 0.0
            or not all(control_metrics["raw"]["top1_agreement"])
            or control_metrics["aggregate"]["cross_entropy_delta"] != 0.0
            or control["local_routed_relative_l2"]["aggregate"]["maximum"] != 0.0
        ):
            raise RuntimeError(f"exact 100% atomic control failed on {split}")
        split_results["curve_index"] = {
            method: [
                {
                    "requested_fraction": fraction,
                    "policy_id": (
                        policy_id("exact_all_atoms", 1.0)
                        if fraction == 1.0
                        else policy_id(method, fraction)
                    ),
                }
                for fraction in FRACTIONS
            ]
            for method in METHODS
        }
        results[split] = split_results
    timings["final_projection_and_metrics_seconds"] = time.perf_counter() - phase

    verdict, gates = adjudicate(results, args.stage)
    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report = {
        "schema_version": 1,
        "kind": "craft_moe_exact_atomic_layer26_oracle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H3_ATOMIC_ORACLE",
        "verdict": verdict,
        "preregistration": str(
            (ROOT / "reports/craft_moe/H3_ATOMIC_LAYER26_PREREGISTRATION.md").resolve()
        ),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "local_path": str(model_dir.resolve()),
            "layer": LAYER,
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "window": {
                split: f"first {args.tokens_per_split} tokens"
                for split in args.splits
            },
            "block_size": BLOCK_SIZE,
        },
        "configuration": {
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
            "fractions": list(FRACTIONS),
            "methods": list(METHODS),
            "candidate_batch": args.candidate_batch,
            "policy_batch": args.policy_batch,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "active_routed_experts": ACTIVE_EXPERTS,
            "atoms_per_expert": ATOMS_PER_EXPERT,
            "shared_experts": "exact via official teacher delta patch",
            "router_weights_renormalized": False,
            "counterfactual_patch": (
                "BF16(official_teacher + sparse_routed - manual_full_routed)"
            ),
            "tile_score": "sum of squared p*abs(a)*L2(down_column) atom norms",
            "ties": "stable original expert-slot/neuron order",
            "quality_evaluation_implementation": (
                "dense BF16 GEMM with exactly zero-masked activations; not sparse runtime"
            ),
        },
        "decomposition": {
            "activation": "a_j = silu(gate_j(x)) * up_j(x)",
            "atom": "v_(e,j) = p_e * a_(e,j) * down_column_(e,j)",
            "separate_direct_gemm_bit_exact": decomposition_bit_exact,
            "separate_direct_gemm_regression": decomposition_regression,
        },
        "policy_order": specifications,
        "gates": gates,
        "results": results,
        "controls": {
            "route_recomputation": route_control,
            "full_atom_vs_separate_direct_expert_output": {
                "bit_exact": decomposition_bit_exact,
                "regression": decomposition_regression,
                "fixed_tolerance": {"nrmse_le": 1e-4, "maximum_absolute_error_le": 0.01},
            },
            "official_teacher_exact_delta": "passed on every evaluated split",
        },
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "safetensors": safetensors.__version__,
                "tokenizers": tokenizers.__version__,
                "pyarrow": pyarrow.__version__,
                "psutil": psutil.__version__,
            },
            "inputs": {
                "sha256": input_hashes,
                "component_metadata": metadata(component_path),
                "trace_indices": trace_indices,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "late-layer exploratory oracle; not yet layer 23, spread layers, or full depth",
            "support is selected from exact activations and therefore is not deployable",
            "gate/up must currently execute before support is known; this is an oracle ceiling",
            "reported ideal bytes/MACs and tensor-local pages are analytical, not wall-clock",
            "dense zero-masked GEMMs evaluate quality but are not a packed sparse kernel",
            "a separately launched direct BF16 GEMM can differ by one BF16 rounding step; the exact original control uses the same manual full-route kernel as its delta baseline",
            "256-token existing windows are exploratory replication, not confirmation",
            "task accuracy and autoregressive stability are outside this initial screen",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    print(f"gates={json.dumps(gates, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
