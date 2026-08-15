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
import torch
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
    quantized_copy,
    regression_summary,
    sequence_blocks,
    sha256_file,
    write_json_once,
)
from moe_lab.craft_moe.qerc import routed_output
from moe_lab.craft_moe.reduction_order import (
    SCHEMES,
    anchored_reduction_candidate,
    q3_q4_gap_closure,
    reduce_permutation_batch,
    routed_mse_by_order,
    six_term_permutations,
)
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT


LAYER = 26
HIDDEN_SIZE = 2048
EXPERTS = 64
ACTIVE_EXPERTS = 6
TRACE_TOKENS_PER_SPLIT = 1024
FULL_TOKENS_PER_SPLIT = 256
SMOKE_TOKENS = 32
BLOCK_SIZE = 128
ROUTER_WEIGHT_TOLERANCE = 1e-6
STRONG_CLOSURE = 0.20
HARD_CLOSURE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H10 all-720 low-precision expert reduction-order screen."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--tokens-per-split", type=int)
    parser.add_argument("--permutation-batch", type=int, default=16)
    parser.add_argument("--candidate-batch", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--reuse-capture", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-capture", type=Path)
    parser.add_argument("--output-raw", type=Path)
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
        raise ValueError("full is fixed at 256+256 tokens and 10k bootstrap")
    if args.permutation_batch < 1 or args.candidate_batch < 1:
        raise ValueError("batch sizes must be positive")
    defaults = {
        "output_json": (
            "reports/runs/craft_moe/reduction_order_smoke.json"
            if args.stage == "smoke"
            else "reports/craft_moe/reduction_order.json"
        ),
        "output_capture": (
            "reports/runs/craft_moe/reduction_order_capture_smoke.safetensors"
            if args.stage == "smoke"
            else "reports/runs/craft_moe/reduction_order_capture.safetensors"
        ),
        "output_raw": (
            "reports/runs/craft_moe/reduction_order_raw_smoke.safetensors"
            if args.stage == "smoke"
            else "reports/runs/craft_moe/reduction_order_raw.safetensors"
        ),
    }
    for name, relative in defaults.items():
        value = getattr(args, name)
        path = ROOT / relative if value is None else value
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if (ROOT / "reports").resolve() not in path.parents:
            raise ValueError("all outputs must remain under reports/")
        setattr(args, name, path)
    if args.output_json.exists() or args.output_raw.exists():
        raise FileExistsError("refusing to overwrite H10 result/raw artifact")
    if args.output_capture.exists() and not args.reuse_capture:
        raise FileExistsError("capture exists; use --reuse-capture after audit")
    if args.reuse_capture and not args.output_capture.exists():
        raise FileNotFoundError(args.output_capture)
    return args


@torch.inference_mode()
def capture_precisions(
    moe: LoadedMoELayer,
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, slots = expert_ids.shape
    shape = (tokens, slots, HIDDEN_SIZE)
    bf16 = torch.empty(shape, dtype=inputs.dtype)
    q3 = torch.empty_like(bf16)
    q4 = torch.empty_like(bf16)
    for expert_id, expert in enumerate(moe.experts):
        positions = (expert_ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            x = inputs[token_indices].to(moe.device)
            bf16[token_indices, slot_indices] = moe.expert_forward(x, expert).cpu()
            three = quantized_copy(expert, 3)
            q3[token_indices, slot_indices] = moe.expert_forward(x, three).cpu()
            del three
            four = quantized_copy(expert, 4)
            q4[token_indices, slot_indices] = moe.expert_forward(x, four).cpu()
            del four
        if expert_id % 4 == 3:
            print(f"reduction_capture_experts={expert_id + 1}/64", flush=True)
    for name, tensor in (("bf16", bf16), ("q3", q3), ("q4", q4)):
        if not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"non-finite {name} capture")
    return bf16, q3, q4


def build_capture(
    moe: LoadedMoELayer, components: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    route_ids, route_weights = moe.route(components["moe_input"].to(moe.device))
    ids_exact = torch.equal(route_ids.cpu(), components["router_ids"].long())
    weight_error = float(
        (route_weights.cpu().float() - components["router_weights"].float())
        .abs()
        .max()
        .item()
    )
    if not ids_exact or weight_error > ROUTER_WEIGHT_TOLERANCE:
        raise RuntimeError(
            f"route control failed: ids_exact={ids_exact}, weight_error={weight_error}"
        )
    bf16, q3, q4 = capture_precisions(
        moe, components["moe_input"], components["router_ids"].long()
    )
    weights = components["router_weights"].float()
    natural_routed = routed_output(bf16, weights)
    q3_reference = routed_output(q3, weights)
    q4_reference = routed_output(q4, weights)
    original = anchored_reduction_candidate(
        components["teacher"], natural_routed, natural_routed
    )
    original_exact = torch.equal(original, components["teacher"])
    if not original_exact:
        raise RuntimeError("reduction-order original control failed")
    tensors = {
        "bf16_selected": bf16.contiguous(),
        "q3_selected": q3.contiguous(),
        "q4_selected": q4.contiguous(),
        "router_ids": components["router_ids"].to(torch.int16).contiguous(),
        "router_weights": weights.contiguous(),
        "natural_routed": natural_routed.contiguous(),
        "q3_reference_routed": q3_reference.contiguous(),
        "q4_reference_routed": q4_reference.contiguous(),
        "teacher": components["teacher"].contiguous(),
        "trace_indices": components["trace_indices"].contiguous(),
    }
    controls = {
        "route_slot_ids_exact": ids_exact,
        "router_weight_maximum_absolute_error": weight_error,
        "original_teacher_delta_bit_exact": original_exact,
        "capture_finite": True,
        "stored_q3_batch_shape_regression": regression_summary(
            components["stored_q3"], q3
        ),
        "stored_q4_batch_shape_regression": regression_summary(
            components["stored_q4"], q4
        ),
    }
    return tensors, controls


def fixed_reduction(
    selected: torch.Tensor,
    weights: torch.Tensor,
    scheme_index: int,
    permutation_index: int,
    permutations: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    terms = selected.to(device).float() * weights.to(device).float().unsqueeze(-1)
    reduced = reduce_permutation_batch(
        terms,
        permutations[permutation_index : permutation_index + 1].to(device),
        SCHEMES[scheme_index],
    )[0]
    return reduced.to(torch.bfloat16).cpu()


@torch.inference_mode()
def sweep_split_precision(
    *,
    selected: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
    permutations: torch.Tensor,
    permutation_batch: int,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = selected.shape[0]
    raw_mse = torch.empty(len(SCHEMES), 720, tokens, dtype=torch.float32)
    oracle_mse = torch.full((tokens,), float("inf"), dtype=torch.float32)
    oracle_scheme = torch.full((tokens,), -1, dtype=torch.int16)
    oracle_order = torch.full((tokens,), -1, dtype=torch.int16)
    oracle_routed = torch.empty(tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
    scheme_reports: dict[str, Any] = {}
    selected_device = selected.to(device)
    weights_device = weights.to(device)
    target_device = target.to(device)
    terms = selected_device.float() * weights_device.float().unsqueeze(-1)
    token_index = torch.arange(tokens, device=device)
    for scheme_index, scheme in enumerate(SCHEMES):
        coordinate_min = torch.full_like(target_device.float(), float("inf"))
        coordinate_max = torch.full_like(target_device.float(), -float("inf"))
        for start in range(0, 720, permutation_batch):
            stop = min(start + permutation_batch, 720)
            reduced = reduce_permutation_batch(
                terms, permutations[start:stop].to(device), scheme
            ).to(torch.bfloat16)
            mse = routed_mse_by_order(reduced, target_device).cpu()
            raw_mse[scheme_index, start:stop] = mse
            reduced_float = reduced.float()
            coordinate_min = torch.minimum(
                coordinate_min, reduced_float.amin(dim=0)
            )
            coordinate_max = torch.maximum(
                coordinate_max, reduced_float.amax(dim=0)
            )
            local_mse, local_order = mse.min(dim=0)
            improved = local_mse < oracle_mse
            if improved.any():
                positions = improved.nonzero(as_tuple=False).squeeze(1)
                oracle_mse[positions] = local_mse[positions]
                oracle_scheme[positions] = scheme_index
                absolute_order = local_order[positions] + start
                oracle_order[positions] = absolute_order.to(torch.int16)
                oracle_routed[positions] = reduced[
                    local_order[positions].to(device), token_index[positions]
                ].cpu()
            if stop == 720 or stop % 240 == 0:
                print(
                    f"{label}_{scheme.name}_orders={stop}/720", flush=True
                )
        aggregate_by_order = raw_mse[scheme_index].double().sum(dim=1)
        best_order = int(aggregate_by_order.argmin().item())
        identity = raw_mse[scheme_index, 0]
        best = raw_mse[scheme_index, best_order]
        scheme_reports[scheme.name] = {
            "best_validation_candidate_permutation_index": best_order,
            "best_validation_candidate_permutation": permutations[best_order].tolist(),
            "identity_mse_mean": float(identity.double().mean().item()),
            "best_fixed_mse_mean": float(best.double().mean().item()),
            "best_fixed_relative_mse_reduction": float(
                1.0 - best.double().sum().item() / identity.double().sum().item()
            )
            if identity.double().sum().item() > 0
            else 0.0,
            "unique_mse_profiles": int(torch.unique(raw_mse[scheme_index], dim=0).shape[0]),
            "maximum_order_spread_linf_after_bf16_cast": float(
                (coordinate_max - coordinate_min).max().item()
            ),
        }
        del coordinate_min, coordinate_max
        torch.cuda.empty_cache()
    overall_scores = raw_mse.double().sum(dim=2)
    flat_best = int(overall_scores.reshape(-1).argmin().item())
    best_scheme = flat_best // 720
    best_order = flat_best % 720
    summary = {
        "schemes": scheme_reports,
        "best_fixed_by_local_mse": {
            "scheme_index": best_scheme,
            "scheme": SCHEMES[best_scheme].name,
            "permutation_index": best_order,
            "permutation": permutations[best_order].tolist(),
            "mse_mean": float(
                raw_mse[best_scheme, best_order].double().mean().item()
            ),
        },
        "per_token_oracle_mse_mean": float(oracle_mse.double().mean().item()),
        "per_token_oracle_scheme_histogram": {
            SCHEMES[index].name: int((oracle_scheme == index).sum().item())
            for index in range(len(SCHEMES))
        },
    }
    del selected_device, weights_device, target_device, terms
    torch.cuda.empty_cache()
    return raw_mse, summary, oracle_routed, oracle_scheme, oracle_order


def quality_record(
    routed: torch.Tensor,
    teacher: torch.Tensor,
    natural_routed: torch.Tensor,
    reference: Any,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    states = anchored_reduction_candidate(teacher, routed, natural_routed)
    return evaluate_hidden(
        states,
        reference,
        norm_weight,
        lm_head,
        candidate_batch,
        bootstrap_resamples,
        bootstrap_seed,
    )


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("H10 reduction-order screen requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    device = torch.device("cuda")
    model_dir = ROOT / "models/deepseek-v2-lite"
    component_path = ROOT / COMPONENT_RELATIVE
    preregistration = ROOT / "reports/craft_moe/H10_REDUCTION_ORDER_LAYER26_PREREGISTRATION.md"
    prior_art = ROOT / "docs/PRIOR_ART.md"
    for path in (model_dir, component_path, preregistration, prior_art):
        if not path.exists():
            raise FileNotFoundError(path)
    inputs_sha256 = {
        str(path.resolve()): sha256_file(path)
        for path in (
            component_path,
            preregistration,
            prior_art,
            model_dir / "config.json",
            model_dir / "model.safetensors.index.json",
        )
    }
    repository = git_state()
    initial_hardware = hardware_state()
    splits = ("validation",) if args.stage == "smoke" else ("validation", "test")

    phase = time.perf_counter()
    all_components = load_file(component_path, device="cpu")
    indices: list[int] = []
    split_slices: dict[str, slice] = {}
    trace_indices: dict[str, list[int]] = {}
    offset = 0
    for split in splits:
        base = 0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        chosen = list(range(base, base + args.tokens_per_split))
        indices.extend(chosen)
        trace_indices[split] = chosen
        split_slices[split] = slice(offset, offset + args.tokens_per_split)
        offset += args.tokens_per_split
    index = torch.tensor(indices, dtype=torch.long)
    components = {
        "moe_input": all_components["moe_input"].index_select(0, index),
        "router_ids": all_components["router_ids"].index_select(0, index),
        "router_weights": all_components["router_weights"].index_select(0, index),
        "teacher": all_components["teacher"].index_select(0, index),
        "stored_q3": all_components["selected_quant3"].index_select(0, index),
        "stored_q4": all_components["selected_quant4"].index_select(0, index),
        "trace_indices": index,
    }
    del all_components
    moe = load_moe_layer(model_dir, LAYER, device)
    timings["load_inputs_and_layer_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    if args.reuse_capture:
        capture = load_file(args.output_capture, device="cpu")
        if not torch.equal(capture["trace_indices"], index):
            raise RuntimeError("reused capture indices differ")
        capture_controls: dict[str, Any] = {
            "reused_capture": True,
            "trace_indices_exact": True,
        }
    else:
        capture, capture_controls = build_capture(moe, components)
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
                "quantization": "symmetric per-output-row Q3/Q4",
                "capture": "BF16/Q3/Q4 same batch and route",
            },
        )
    capture_hash = sha256_file(args.output_capture)
    timings["capture_or_reuse_seconds"] = time.perf_counter() - phase
    del moe
    gc.collect()
    torch.cuda.empty_cache()

    phase = time.perf_counter()
    permutations = six_term_permutations()
    raw_artifact: dict[str, torch.Tensor] = {
        "permutations": permutations.to(torch.int16).contiguous()
    }
    sweep_reports: dict[str, dict[str, Any]] = {"q3": {}, "q4": {}}
    oracle_outputs: dict[str, dict[str, torch.Tensor]] = {"q3": {}, "q4": {}}
    for bit_name, selected_key in (("q3", "q3_selected"), ("q4", "q4_selected")):
        for split, sl in split_slices.items():
            raw_mse, summary, oracle_routed, oracle_scheme, oracle_order = (
                sweep_split_precision(
                    selected=capture[selected_key][sl],
                    weights=capture["router_weights"][sl],
                    target=capture["natural_routed"][sl],
                    permutations=permutations,
                    permutation_batch=args.permutation_batch,
                    device=device,
                    label=f"{bit_name}_{split}",
                )
            )
            raw_artifact[f"{bit_name}_{split}_mse"] = raw_mse.contiguous()
            raw_artifact[f"{bit_name}_{split}_oracle_scheme"] = oracle_scheme.contiguous()
            raw_artifact[f"{bit_name}_{split}_oracle_order"] = oracle_order.contiguous()
            raw_artifact[f"{bit_name}_{split}_oracle_routed"] = oracle_routed.contiguous()
            sweep_reports[bit_name][split] = summary
            oracle_outputs[bit_name][split] = oracle_routed
    validation_scores = raw_artifact["q3_validation_mse"].double().sum(dim=2)
    selected_flat = int(validation_scores.reshape(-1).argmin().item())
    selected_scheme_index = selected_flat // 720
    selected_order_index = selected_flat % 720
    fp32_scores = validation_scores[:2]
    fp32_flat = int(fp32_scores.reshape(-1).argmin().item())
    fp32_scheme_index = fp32_flat // 720
    fp32_order_index = fp32_flat % 720
    selection = {
        "scheme_index": selected_scheme_index,
        "scheme": SCHEMES[selected_scheme_index].name,
        "permutation_index": selected_order_index,
        "permutation": permutations[selected_order_index].tolist(),
        "criterion": "minimum sum routed MSE to same-batch BF16 natural route on validation",
    }
    fp32_selection = {
        "scheme_index": fp32_scheme_index,
        "scheme": SCHEMES[fp32_scheme_index].name,
        "permutation_index": fp32_order_index,
        "permutation": permutations[fp32_order_index].tolist(),
    }
    save_file(
        raw_artifact,
        args.output_raw,
        metadata={
            "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "stage": args.stage,
            "schemes_in_axis_order": ",".join(scheme.name for scheme in SCHEMES),
            "selection": str(selection),
        },
    )
    raw_hash = sha256_file(args.output_raw)
    timings["all_720_order_sweeps_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    exact_results: dict[str, Any] = {}
    for split_index, (split, sl) in enumerate(split_slices.items()):
        reference = make_teacher_reference(
            capture["teacher"][sl],
            corpus_tokens(model_dir, split, args.tokens_per_split),
            sequence_blocks(args.tokens_per_split),
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        teacher = capture["teacher"][sl]
        natural = capture["natural_routed"][sl]
        q3_fixed = fixed_reduction(
            capture["q3_selected"][sl],
            capture["router_weights"][sl],
            selected_scheme_index,
            selected_order_index,
            permutations,
            device,
        )
        q4_fixed = fixed_reduction(
            capture["q4_selected"][sl],
            capture["router_weights"][sl],
            selected_scheme_index,
            selected_order_index,
            permutations,
            device,
        )
        q3_fp32_control = fixed_reduction(
            capture["q3_selected"][sl],
            capture["router_weights"][sl],
            fp32_scheme_index,
            fp32_order_index,
            permutations,
            device,
        )
        candidates = {
            "q3_reference_vectorized_fp32": capture["q3_reference_routed"][sl],
            "q4_reference_vectorized_fp32": capture["q4_reference_routed"][sl],
            "q3_fixed_validation_order": q3_fixed,
            "q4_same_fixed_order": q4_fixed,
            "q3_validation_selected_fp32_control": q3_fp32_control,
            "q3_per_token_local_mse_oracle": oracle_outputs["q3"][split],
            "q4_per_token_local_mse_oracle": oracle_outputs["q4"][split],
        }
        for scheme_index, scheme in enumerate(SCHEMES):
            candidates[f"q3_identity__{scheme.name}"] = fixed_reduction(
                capture["q3_selected"][sl],
                capture["router_weights"][sl],
                scheme_index,
                0,
                permutations,
                device,
            )
            candidates[f"q4_identity__{scheme.name}"] = fixed_reduction(
                capture["q4_selected"][sl],
                capture["router_weights"][sl],
                scheme_index,
                0,
                permutations,
                device,
            )
        split_quality = {}
        for candidate_index, (name, routed) in enumerate(candidates.items()):
            print(f"exact_quality split={split} candidate={name}", flush=True)
            split_quality[name] = quality_record(
                routed,
                teacher,
                natural,
                reference,
                norm_weight,
                lm_head,
                args.candidate_batch,
                args.bootstrap_resamples,
                SEED + split_index * 100 + candidate_index,
            )
        q3_kl = split_quality["q3_reference_vectorized_fp32"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        q4_kl = split_quality["q4_reference_vectorized_fp32"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        fixed_kl = split_quality["q3_fixed_validation_order"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        fp32_kl = split_quality["q3_validation_selected_fp32_control"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        denominator_positive = q3_kl > q4_kl
        split_quality["gap_analysis"] = {
            "q3_reference_kl": q3_kl,
            "q4_reference_kl": q4_kl,
            "q3_to_q4_gap": q3_kl - q4_kl,
            "denominator_positive": denominator_positive,
            "fixed_q3_kl": fixed_kl,
            "fixed_gap_closure": (
                q3_q4_gap_closure(q3_kl, q4_kl, fixed_kl)
                if denominator_positive
                else None
            ),
            "fp32_control_q3_kl": fp32_kl,
            "fp32_control_gap_closure": (
                q3_q4_gap_closure(q3_kl, q4_kl, fp32_kl)
                if denominator_positive
                else None
            ),
        }
        exact_results[split] = split_quality
    timings["exact_quality_seconds"] = time.perf_counter() - phase

    original_bit_exact = torch.equal(
        anchored_reduction_candidate(
            capture["teacher"],
            capture["natural_routed"],
            capture["natural_routed"],
        ),
        capture["teacher"],
    )
    controls_pass = original_bit_exact and bool(
        capture_controls.get("route_slot_ids_exact", True)
    ) and float(
        capture_controls.get("router_weight_maximum_absolute_error", 0.0)
    ) <= ROUTER_WEIGHT_TOLERANCE
    gates: dict[str, Any] = {
        "exact_controls_pass": controls_pass,
        "q3_to_q4_denominator_positive": {},
        "fixed_gap_closure_ge_0_20": {},
        "fixed_q3_not_worse_than_reference": {},
        "fp32_control_closure_ge_0_10": {},
        "same_weight_bytes_and_no_metadata": True,
        "throughput_ratio_le_1_05": {
            "evaluated": False,
            "reason": "physical reducer benchmark opens only after content gates pass",
        },
    }
    for split in splits:
        gap = exact_results[split]["gap_analysis"]
        gates["q3_to_q4_denominator_positive"][split] = gap[
            "denominator_positive"
        ]
        gates["fixed_gap_closure_ge_0_20"][split] = bool(
            gap["fixed_gap_closure"] is not None
            and gap["fixed_gap_closure"] >= STRONG_CLOSURE
        )
        gates["fixed_q3_not_worse_than_reference"][split] = (
            gap["fixed_q3_kl"] <= gap["q3_reference_kl"]
        )
        gates["fp32_control_closure_ge_0_10"][split] = bool(
            gap["fp32_control_gap_closure"] is not None
            and gap["fp32_control_gap_closure"] >= HARD_CLOSURE
        )
    content_positive = bool(
        args.stage == "full"
        and controls_pass
        and all(gates["q3_to_q4_denominator_positive"].values())
        and all(gates["fixed_gap_closure_ge_0_20"].values())
        and all(gates["fixed_q3_not_worse_than_reference"].values())
        and gates["same_weight_bytes_and_no_metadata"]
    )
    if content_positive:
        verdict = "content_positive_requires_physical_reducer_benchmark"
    elif not controls_pass:
        verdict = "invalid_exact_control"
    elif args.stage == "smoke":
        verdict = "smoke_only_not_adjudicated"
    else:
        test_gap = exact_results["test"]["gap_analysis"]
        if (
            not test_gap["denominator_positive"]
            or test_gap["fixed_gap_closure"] is None
            or test_gap["fixed_gap_closure"] < HARD_CLOSURE
            or test_gap["fixed_q3_kl"] > test_gap["q3_reference_kl"]
            or test_gap["fp32_control_gap_closure"] is None
            or test_gap["fp32_control_gap_closure"] < HARD_CLOSURE
        ):
            verdict = "falsified_fixed_heldout_reduction_order"
        else:
            verdict = "inconclusive_negative_layer26_screen"

    protected_order_invariance: dict[str, Any] = {}
    for bit in ("q3", "q4"):
        protected_order_invariance[bit] = {}
        for split in splits:
            protected_order_invariance[bit][split] = {
                name: sweep_reports[bit][split]["schemes"][name][
                    "maximum_order_spread_linf_after_bf16_cast"
                ]
                for name in (
                    "bf16_operands_fp32_sequential",
                    "bf16_operands_fp32_tree",
                )
            }

    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report = {
        "schema_version": 1,
        "kind": "craft_moe_h10_reduction_order_layer26",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "verdict": verdict,
        "content_positive": content_positive,
        "experiment": "H10_REDUCTION_ORDER",
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
            "permutations": 720,
            "permutation_order": "lexicographic slots 0..5",
            "schemes": [
                {
                    "name": scheme.name,
                    "operand_dtype": str(scheme.operand_dtype),
                    "accumulator_dtype": str(scheme.accumulator_dtype),
                    "topology": scheme.topology,
                }
                for scheme in SCHEMES
            ],
            "final_routed_cast": "BF16",
            "selection_uses": "validation Q3 routed MSE only",
            "test_tuning": "none",
            "gate_gap_closure": STRONG_CLOSURE,
            "hard_gap_closure": HARD_CLOSURE,
        },
        "controls": capture_controls
        | {"post_capture_original_teacher_delta_bit_exact": original_bit_exact},
        "validation_selection": selection,
        "validation_selected_fp32_control": fp32_selection,
        "sweeps": sweep_reports,
        "protected_fp32_order_invariance": protected_order_invariance,
        "exact_quality": exact_results,
        "gates": gates,
        "accounting": {
            "expert_weight_bytes_changed": 0,
            "metadata_bytes": 0,
            "terms_changed": False,
            "additions_per_reduction": 5,
            "physical_throughput_measured": False,
        },
        "capture_artifact": {
            "path": str(args.output_capture),
            "bytes": args.output_capture.stat().st_size,
            "sha256": capture_hash,
            "metadata": metadata(args.output_capture),
        },
        "raw_sweep_artifact": {
            "path": str(args.output_raw),
            "bytes": args.output_raw.stat().st_size,
            "sha256": raw_hash,
            "metadata": metadata(args.output_raw),
        },
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "seed": SEED,
            "inputs_sha256": inputs_sha256,
            "component_metadata": metadata(component_path),
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "safetensors": safetensors.__version__,
                "psutil": psutil.__version__,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "prior_art": {
            "primary": "https://arxiv.org/abs/2607.28097",
            "boundary": (
                "prior work establishes causal divergence and numerical compatibility; "
                "H10 tests systematic held-out Q3 quality compensation"
            ),
        },
        "limitations": [
            "fixed order is selected by routed MSE, not exhaustive full-vocabulary KL over all 720 orders",
            "the per-token best scheme/order uses the true BF16 target and is not deployable",
            "only layer 26 and 256-token windows are evaluated unless the content gate passes",
            "no physical reducer or throughput claim is opened before a positive content screen",
            "the experiment tests one quantizer and one GPU/software numerical environment",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}")
    print(f"capture={args.output_capture}")
    print(f"raw={args.output_raw}")
    print(f"verdict={verdict}")
    print(
        f"selected={selection['scheme']} order={selection['permutation']}"
    )
    for split in splits:
        gap = exact_results[split]["gap_analysis"]
        print(
            f"{split}_q3={gap['q3_reference_kl']:.8g} "
            f"q4={gap['q4_reference_kl']:.8g} "
            f"fixed={gap['fixed_q3_kl']:.8g} "
            f"closure={gap['fixed_gap_closure']}"
        )


if __name__ == "__main__":
    main()
