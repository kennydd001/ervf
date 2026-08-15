from __future__ import annotations

import argparse
import gc
import json
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
import tokenizers
from safetensors.torch import load_file

from evaluate_crcq_oracle import (
    BLOCK_SIZE,
    COMPONENT_RELATIVE,
    DATASET_REVISION,
    MODEL_REVISION,
    RATE_FRACTIONS,
    ROUTE_RELATIVE,
    SEED,
    TARGET_MULTIPLIER,
    TRACE_TOKENS_PER_SPLIT,
    aligned_natural_outputs,
    choices_for_schedule,
    command_result,
    corpus_tokens,
    evaluate_hidden,
    exact_kl_for_states,
    gate_bootstrap,
    git_state,
    hardware_state,
    make_teacher_reference,
    metadata,
    patched_candidates,
    regression_summary,
    schedule_record,
    selected_precision_outputs,
    sequence_blocks,
    sha256_file,
    solution_json,
    write_json_once,
)
from moe_lab.craft_moe.crcq import (
    best_by_upgrade_count,
    best_schedule_within_fraction,
    mean_gap_closure,
    mixed_precision_routed,
    natural_subset_index,
    routed_for_routes,
    routed_from_choices,
    six_of_twelve_subsets,
    solve_minimum_budget,
)
from moe_lab.dynamic_precision import binary_upgrade_masks
from moe_lab.moe_layer import load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT


TOKENS_PER_SPLIT = 256
ROUTE_CHUNK = 64
CANDIDATE_BATCH = 128
BOOTSTRAP_RESAMPLES = 10_000
SCREEN_RELATIVE = Path("reports/craft_moe/crcq_oracle.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full 924-route x 64-mask exact-KL CRCQ oracle."
    )
    parser.add_argument("--tokens-per-split", type=int, default=TOKENS_PER_SPLIT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--route-chunk", type=int, default=ROUTE_CHUNK)
    parser.add_argument("--candidate-batch", type=int, default=CANDIDATE_BATCH)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/craft_moe/crcq_full_oracle.json"),
    )
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if (
        args.tokens_per_split != TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.route_chunk != ROUTE_CHUNK
        or args.candidate_batch != CANDIDATE_BATCH
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
    ):
        raise ValueError("the preregistered full-oracle configuration is immutable")
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output = output.resolve()
    if (ROOT / "reports").resolve() not in output.parents:
        raise ValueError("output-json must be inside reports/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    args.output_json = output
    return args


@torch.inference_mode()
def full_route_mask_damage(
    *,
    label: str,
    q3: torch.Tensor,
    q4: torch.Tensor,
    weights: torch.Tensor,
    subsets: torch.Tensor,
    masks: torch.Tensor,
    teacher: torch.Tensor,
    natural_bf16: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    route_chunk: int,
    candidate_batch: int,
) -> torch.Tensor:
    damage = torch.empty(
        q3.shape[0], subsets.shape[0], masks.shape[0], dtype=torch.float32
    )
    for token in range(q3.shape[0]):
        for route_start in range(0, subsets.shape[0], route_chunk):
            route_stop = min(route_start + route_chunk, subsets.shape[0])
            routes = subsets[route_start:route_stop]
            mixed = mixed_precision_routed(
                q3[token], q4[token], weights[token], routes, masks
            )
            states = patched_candidates(
                teacher[token],
                natural_bf16[token],
                mixed.reshape(-1, mixed.shape[-1]),
            )
            damage[token, route_start:route_stop] = exact_kl_for_states(
                states,
                teacher_log_probs[token],
                norm_weight,
                lm_head,
                candidate_batch,
            ).view(route_stop - route_start, masks.shape[0])
        print(f"full_stage[{label}]={token + 1}/{q3.shape[0]}", flush=True)
    return damage


def natural_control(
    *,
    bf16: torch.Tensor,
    weights: torch.Tensor,
    subsets: torch.Tensor,
    natural_index: int,
    masks: torch.Tensor,
    teacher: torch.Tensor,
    natural_bf16: torch.Tensor,
    reference,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    args: argparse.Namespace,
    bootstrap_seed: int,
) -> dict[str, Any]:
    routes = torch.full((bf16.shape[0],), natural_index, dtype=torch.long)
    mask_indices = torch.zeros(bf16.shape[0], dtype=torch.long)
    routed = routed_from_choices(
        bf16, bf16, weights, subsets, routes, masks, mask_indices
    )
    hidden = (teacher.float() + (routed.float() - natural_bf16.float())).to(
        teacher.dtype
    )
    result = evaluate_hidden(
        hidden,
        reference,
        norm_weight,
        lm_head,
        args.candidate_batch,
        args.bootstrap_resamples,
        bootstrap_seed,
    )
    if (
        max(result["raw"]["teacher_to_candidate_kl"]) != 0.0
        or not all(result["raw"]["top1_agreement"])
        or result["aggregate"]["cross_entropy_delta"] != 0.0
    ):
        raise RuntimeError("natural BF16 exact control failed")
    return result


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("full CRCQ oracle requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    model_dir = ROOT / "models/deepseek-v2-lite"
    component_path = ROOT / COMPONENT_RELATIVE
    route_path = ROOT / ROUTE_RELATIVE
    screen_path = ROOT / SCREEN_RELATIVE
    for path in (model_dir, component_path, route_path, screen_path):
        if not path.exists():
            raise FileNotFoundError(path)
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if screen["verdict"] != "strong_positive" or not screen[
        "full_59136_search_eligible"
    ]:
        raise RuntimeError("the prerequisite top-32 screen did not authorize full search")

    phase = time.perf_counter()
    input_hashes = {
        str(component_path.resolve()): sha256_file(component_path),
        str(route_path.resolve()): sha256_file(route_path),
        str(screen_path.resolve()): sha256_file(screen_path),
    }
    timings["input_sha256_seconds"] = time.perf_counter() - phase
    initial_hardware = hardware_state()
    repository = git_state()
    disk_before = psutil.disk_usage(str(ROOT))

    phase = time.perf_counter()
    component_all = load_file(component_path, device="cpu")
    route_all = load_file(route_path, device="cpu")
    trace_indices: dict[str, list[int]] = {}
    indices = []
    for split in args.splits:
        base = 0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        selected = list(range(base, base + args.tokens_per_split))
        trace_indices[split] = selected
        indices.extend(selected)
    index = torch.tensor(indices, dtype=torch.long)
    components = {key: value.index_select(0, index) for key, value in component_all.items()}
    top12_ids = route_all["top12_expert_ids"].index_select(0, index).long()
    top12_weights = route_all["top12_router_weights"].index_select(0, index).float()
    subsets = route_all["subsets"].long()
    if not torch.equal(subsets, six_of_twelve_subsets()):
        raise RuntimeError("unexpected route enumeration")
    natural_index = natural_subset_index(subsets)
    masks = binary_upgrade_masks(6)
    del component_all, route_all
    token_ids = {
        split: corpus_tokens(model_dir, split, args.tokens_per_split)
        for split in args.splits
    }
    timings["load_inputs_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    moe = load_moe_layer(model_dir, 26, device)
    bf16, q3, q4 = selected_precision_outputs(
        moe, components["moe_input"], top12_ids
    )
    del moe
    gc.collect()
    torch.cuda.empty_cache()
    timings["bf16_q3_q4_expert_forward_seconds"] = time.perf_counter() - phase

    aligned_q3, natural_positions = aligned_natural_outputs(
        top12_ids, components["router_ids"].long(), components["selected_quant3"]
    )
    aligned_q4, _ = aligned_natural_outputs(
        top12_ids, components["router_ids"].long(), components["selected_quant4"]
    )
    aligned_weights = components["router_weights"].gather(1, natural_positions)
    trace_reproducibility = {
        "top6_ids_exact": bool(
            torch.equal(
                torch.sort(top12_ids[:, :6], dim=1).values,
                torch.sort(components["router_ids"].long(), dim=1).values,
            )
        ),
        "top6_router_weight_max_abs": float(
            (aligned_weights.float() - top12_weights[:, :6]).abs().max().item()
        ),
        "selected_q3": regression_summary(aligned_q3, q3[:, :6]),
        "selected_q4": regression_summary(aligned_q4, q4[:, :6]),
    }
    natural_route = subsets[natural_index].unsqueeze(0)
    natural_bf16 = torch.stack(
        [
            routed_for_routes(bf16[token], top12_weights[token], natural_route)[0]
            for token in range(bf16.shape[0])
        ]
    )

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    timings["load_final_projection_seconds"] = time.perf_counter() - phase
    split_slices: dict[str, slice] = {}
    references = {}
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

    results: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for split_index, split in enumerate(args.splits):
        split_started = time.perf_counter()
        selected = split_slices[split]
        split_bf16 = bf16[selected]
        split_q3 = q3[selected]
        split_q4 = q4[selected]
        split_weights = top12_weights[selected]
        split_teacher = components["teacher"][selected]
        split_natural_bf16 = natural_bf16[selected]
        reference = references[split]

        full_started = time.perf_counter()
        full_damage = full_route_mask_damage(
            label=split,
            q3=split_q3,
            q4=split_q4,
            weights=split_weights,
            subsets=subsets,
            masks=masks,
            teacher=split_teacher,
            natural_bf16=split_natural_bf16,
            teacher_log_probs=reference.log_probs,
            norm_weight=norm_weight,
            lm_head=lm_head,
            route_chunk=args.route_chunk,
            candidate_batch=args.candidate_batch,
        )
        full_seconds = time.perf_counter() - full_started
        natural_damage = full_damage[:, natural_index]
        natural_q3 = natural_damage[:, 0]
        natural_q4 = natural_damage[:, -1]
        alternative_all_q3 = full_damage[:, :, 0].clone()
        alternative_all_q3[:, natural_index] = torch.inf
        best_alternative_q3, best_alternative_route = alternative_all_q3.min(dim=1)
        full_best, full_best_route, full_best_mask = best_by_upgrade_count(
            full_damage, masks
        )
        natural_best, _, natural_best_mask = best_by_upgrade_count(
            natural_damage.unsqueeze(1), masks
        )
        reference_q4_mean = float(natural_q4.double().mean().item())
        natural_solution = solve_minimum_budget(
            natural_best,
            reference_q4_mean,
            tolerance_multiplier=TARGET_MULTIPLIER,
        )
        full_solution = solve_minimum_budget(
            full_best,
            reference_q4_mean,
            tolerance_multiplier=TARGET_MULTIPLIER,
        )
        if natural_solution.per_token_cost is None or full_solution.per_token_cost is None:
            raise RuntimeError("natural all-Q4 target must be reachable")

        screen_split = screen["results"][split]
        screen_natural_q3 = screen_split["baselines"]["natural_all_q3"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        screen_natural_q4 = screen_split["baselines"]["natural_all_q4"]["aggregate"][
            "teacher_to_candidate_kl"
        ]
        natural_q3_abs_error = abs(float(natural_q3.double().mean()) - screen_natural_q3)
        natural_q4_abs_error = abs(float(natural_q4.double().mean()) - screen_natural_q4)
        screen_natural_fraction = screen_split["solutions"]["natural_route"][
            "upgrade_fraction"
        ]
        screen_joint_fraction = screen_split["solutions"]["joint_top32"][
            "upgrade_fraction"
        ]
        discrete_tolerance = 1 / (args.tokens_per_split * 6)
        if (
            natural_q3_abs_error > 1e-7
            or natural_q4_abs_error > 1e-7
            or abs(natural_solution.upgrade_fraction - screen_natural_fraction)
            > discrete_tolerance
            or full_solution.upgrade_fraction > screen_joint_fraction + 1e-12
        ):
            raise RuntimeError("full oracle failed its preregistered screen reproduction")

        full_routes, full_masks = choices_for_schedule(
            full_solution.per_token_cost,
            full_best_route,
            full_best_mask,
            torch.arange(subsets.shape[0]).unsqueeze(0).expand(args.tokens_per_split, -1),
        )
        target_record = schedule_record(
            q3=split_q3,
            q4=split_q4,
            weights=split_weights,
            subsets=subsets,
            masks=masks,
            routes=full_routes,
            mask_indices=full_masks,
            teacher=split_teacher,
            natural_bf16=split_natural_bf16,
            reference=reference,
            norm_weight=norm_weight,
            lm_head=lm_head,
            candidate_batch=args.candidate_batch,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed + split_index,
        )
        direct_kl = target_record["aggregate"]["teacher_to_candidate_kl"]
        dp_kl = float(
            full_solution.exact_cost_curve[full_solution.total_cost]
            / args.tokens_per_split
        )
        direct_dp_abs_error = abs(direct_kl - dp_kl)
        if direct_dp_abs_error > 1e-6:
            raise RuntimeError("full DP and direct schedule KL disagree")

        selected_bf16_routed = routed_from_choices(
            split_bf16,
            split_bf16,
            split_weights,
            subsets,
            full_routes,
            masks,
            torch.zeros_like(full_masks),
        )
        selected_bf16_hidden = (
            split_teacher.float()
            + (selected_bf16_routed.float() - split_natural_bf16.float())
        ).to(split_teacher.dtype)
        selected_bf16_metrics = evaluate_hidden(
            selected_bf16_hidden,
            reference,
            norm_weight,
            lm_head,
            args.candidate_batch,
            args.bootstrap_resamples,
            args.seed + split_index,
        )
        control = natural_control(
            bf16=split_bf16,
            weights=split_weights,
            subsets=subsets,
            natural_index=natural_index,
            masks=masks,
            teacher=split_teacher,
            natural_bf16=split_natural_bf16,
            reference=reference,
            norm_weight=norm_weight,
            lm_head=lm_head,
            args=args,
            bootstrap_seed=args.seed + split_index,
        )

        rate_distortion = []
        route_space = torch.arange(subsets.shape[0]).unsqueeze(0).expand(
            args.tokens_per_split, -1
        )
        for requested_fraction in RATE_FRACTIONS:
            cost, schedule = best_schedule_within_fraction(
                full_solution, args.tokens_per_split, requested_fraction
            )
            routes, selected_masks = choices_for_schedule(
                schedule, full_best_route, full_best_mask, route_space
            )
            record = schedule_record(
                q3=split_q3,
                q4=split_q4,
                weights=split_weights,
                subsets=subsets,
                masks=masks,
                routes=routes,
                mask_indices=selected_masks,
                teacher=split_teacher,
                natural_bf16=split_natural_bf16,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
            )
            rate_dp_kl = float(full_solution.exact_cost_curve[cost] / args.tokens_per_split)
            if abs(record["aggregate"]["teacher_to_candidate_kl"] - rate_dp_kl) > 1e-6:
                raise RuntimeError("rate-curve DP/direct mismatch")
            record |= {
                "requested_upgrade_fraction": requested_fraction,
                "total_upgrade_count": cost,
                "actual_upgrade_fraction": cost / (args.tokens_per_split * 6),
                "average_active_bits": 3.0 + cost / (args.tokens_per_split * 6),
                "dynamic_program_kl_mean": rate_dp_kl,
            }
            rate_distortion.append(record)

        closure = mean_gap_closure(natural_q3, natural_q4, best_alternative_q3)
        bootstrap_gate = gate_bootstrap(
            natural_q3=natural_q3,
            natural_q4=natural_q4,
            alternative_q3=best_alternative_q3,
            natural_best=natural_best,
            joint_best=full_best,
            blocks=reference.blocks,
            resamples=args.bootstrap_resamples,
            seed=args.seed + split_index,
        )
        point_pass = float(full_solution.upgrade_fraction) <= 0.15
        no_worse = float(full_solution.upgrade_fraction) <= screen_joint_fraction
        direct_pass = direct_dp_abs_error <= 1e-6
        gates[split] = {
            "full_upgrade_fraction_le_0_15": point_pass,
            "no_worse_than_top32": no_worse,
            "direct_dp_abs_error_le_1e_6": direct_pass,
            "passed": point_pass and no_worse and direct_pass,
            "minimum_upgrade_fraction": full_solution.upgrade_fraction,
            "average_active_bits": full_solution.average_active_bits,
            "top32_upgrade_fraction": screen_joint_fraction,
            "absolute_fraction_improvement_vs_top32": screen_joint_fraction
            - float(full_solution.upgrade_fraction),
            "all_q3_alternative_mean_gap_closure": closure,
            "direct_dp_abs_error": direct_dp_abs_error,
        }
        results[split] = {
            "controls": {
                "natural_bf16_exact": control,
                "screen_reproduction": {
                    "natural_q3_mean_kl_abs_error": natural_q3_abs_error,
                    "natural_q4_mean_kl_abs_error": natural_q4_abs_error,
                    "natural_upgrade_fraction_full": natural_solution.upgrade_fraction,
                    "natural_upgrade_fraction_screen": screen_natural_fraction,
                },
            },
            "raw": {
                "full_route_mask_kl": full_damage.tolist(),
                "best_alternative_all_q3_route": best_alternative_route.tolist(),
                "best_alternative_all_q3_kl": best_alternative_q3.tolist(),
                "full_best_kl_by_exact_upgrade_count": full_best.tolist(),
                "full_best_route_by_exact_upgrade_count": full_best_route.tolist(),
                "full_best_mask_by_exact_upgrade_count": full_best_mask.tolist(),
                "natural_best_kl_by_exact_upgrade_count": natural_best.tolist(),
                "natural_best_mask_by_exact_upgrade_count": natural_best_mask.tolist(),
            },
            "solutions": {
                "natural_route": solution_json(natural_solution),
                "full_joint": solution_json(full_solution),
                "top32_upgrade_fraction": screen_joint_fraction,
                "full_vs_top32_absolute_fraction_improvement": screen_joint_fraction
                - float(full_solution.upgrade_fraction),
                "full_vs_natural_relative_upgrade_reduction": 1.0
                - float(full_solution.upgrade_fraction)
                / float(natural_solution.upgrade_fraction),
                "all_q3_alternative_mean_gap_closure": closure,
            },
            "target_schedule": target_record,
            "selected_routes_in_bf16": selected_bf16_metrics,
            "gate_bootstrap_95": bootstrap_gate,
            "rate_distortion": rate_distortion,
            "full_enumeration_wall_seconds": full_seconds,
            "split_wall_seconds": time.perf_counter() - split_started,
        }
        print(f"gate[{split}]={gates[split]}", flush=True)

    verdict = (
        "full_oracle_positive"
        if all(gates[split]["passed"] for split in args.splits)
        else "full_oracle_gate_failed"
    )
    timings["total_compute_seconds_before_json"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    disk_after_compute = psutil.disk_usage(str(ROOT))
    report = {
        "schema_version": 1,
        "kind": "craft_moe_crcq_full_route_mask_oracle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "experiment": "H1_CRCQ_FULL_59136",
        "verdict": verdict,
        "layer23_eligible": verdict == "full_oracle_positive",
        "preregistration": str(
            (ROOT / "reports/craft_moe/H1_CRCQ_FULL_PREREGISTRATION.md").resolve()
        ),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "local_path": str(model_dir.resolve()),
            "layer": 26,
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "windows": {
                split: f"first {args.tokens_per_split} tokens" for split in args.splits
            },
            "block_size": BLOCK_SIZE,
        },
        "configuration": {
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
            "routes": subsets.shape[0],
            "masks": masks.shape[0],
            "candidates_per_token": subsets.shape[0] * masks.shape[0],
            "route_chunk": args.route_chunk,
            "candidate_batch": args.candidate_batch,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "target_multiplier_vs_natural_all_q4_mean_kl": TARGET_MULTIPLIER,
            "router_weights_renormalized": False,
            "counterfactual_patch": (
                "BF16(official_teacher + candidate_routed - natural_BF16_routed)"
            ),
        },
        "route_space": {
            "subsets": subsets.tolist(),
            "upgrade_masks": masks.tolist(),
            "natural_route_index": natural_index,
        },
        "gates": gates,
        "results": results,
        "controls": {
            "trace_reproducibility": trace_reproducibility,
            "natural_bf16_exact": "passed on validation and test",
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
                "route_metadata": metadata(route_path),
                "trace_indices": trace_indices,
            },
            "disk": {
                "free_bytes_before": disk_before.free,
                "free_bytes_after_compute": disk_after_compute.free,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "full late-layer teacher oracle, not a cheap route/bit selector",
            "no layer-23 exact-downstream intervention in this artifact",
            "256-token existing windows are exploratory, not confirmation",
            "bit savings are not a measured packed-runtime speedup",
        ],
    }
    print("serializing_full_raw_json=true", flush=True)
    serialization_started = time.perf_counter()
    write_json_once(args.output_json, report)
    print(
        f"serialization_seconds={time.perf_counter() - serialization_started:.2f}",
        flush=True,
    )
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()
