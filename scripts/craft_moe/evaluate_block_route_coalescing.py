from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import scipy
import torch
from safetensors.torch import load_file

from evaluate_crcq_oracle import (
    DATASET_REVISION,
    MODEL_REVISION,
    SEED,
    git_state,
    hardware_state,
    metadata,
    percentile,
    sha256_file,
    write_json_once,
)
from moe_lab.craft_moe.block_coalescing import (
    BlockSolution,
    beam_union_solution,
    build_candidate_slates,
    eligible_set_pruning,
    exact_ilp_solution,
    experts_to_mask,
    fixed_cache_prior_solution,
    highs_optimal_control,
    marginal_union_greedy,
    mass_budget_solution,
    natural_solution,
    original_subset_index,
    solution_metrics,
)
from moe_lab.reporting import ROOT


TOKENS_PER_SPLIT = 256
SMOKE_TOKENS = 32
TRACE_TOKENS_PER_SPLIT = 256
SEQUENCE_BLOCK = 128
THRESHOLDS = (1e-4, 1e-3, 3e-3)
SLATE_CAPS = (16, 32, 64)
VERIFICATION_BLOCKS = (2, 4, 8, 16)
PRIMARY_THRESHOLD = 1e-3
PRIMARY_CAP = 32
PRIMARY_BLOCK = 8
MASS_BUDGET_DELTA = 0.004
BEAM_WIDTH = 1024
BOOTSTRAP_RESAMPLES = 10_000
EXPERTS = 64
WEIGHTS_PER_EXPERT = 3 * 1408 * 2048
BF16_BYTES_PER_EXPERT = WEIGHTS_PER_EXPERT * 2
INT4_BYTES_PER_EXPERT = WEIGHTS_PER_EXPERT // 2
ARTIFACT_RELATIVE = Path(
    "data/traces/layer26_route_equivalence.safetensors"
)
CALIBRATION_RELATIVE = Path(
    "reports/baseline/router_calibration_all_layers.json"
)
PREREGISTRATION_RELATIVE = Path(
    "reports/craft_moe/H2_BLOCK_COALESCING_LAYER26_PREREGISTRATION.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered H2 exact block route-union oracle."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--tokens-per-split", type=int, default=TOKENS_PER_SPLIT
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if not 1 <= args.tokens_per_split <= TRACE_TOKENS_PER_SPLIT:
        raise ValueError("tokens-per-split is outside the route trace")
    if args.stage == "smoke":
        if args.splits != ("validation",) or args.tokens_per_split > SMOKE_TOKENS:
            raise ValueError("smoke is at most 32 validation tokens")
    elif (
        args.tokens_per_split != TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.beam_width != BEAM_WIDTH
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
    ):
        raise ValueError(
            "full is fixed at 256 validation + 256 test, beam 1024, "
            "10k bootstrap, and seed 20260810"
        )
    if args.tokens_per_split % 16:
        raise ValueError("tokens-per-split must be divisible by 16")
    if min(args.beam_width, args.bootstrap_resamples) < 1:
        raise ValueError("beam width and bootstrap count must be positive")
    if args.output_json is None:
        relative = (
            Path("reports/craft_moe/block_route_coalescing.json")
            if args.stage == "full"
            else Path("reports/runs/craft_moe/block_route_coalescing_smoke.json")
        )
        args.output_json = ROOT / relative
    elif not args.output_json.is_absolute():
        args.output_json = ROOT / args.output_json
    args.output_json = args.output_json.resolve()
    if (ROOT / "reports").resolve() not in args.output_json.parents:
        raise ValueError("output-json must be inside reports/")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output_json}")
    return args


def configuration_id(threshold: float, cap: int) -> str:
    return f"kl{threshold:g}_cap{cap}".replace(".", "p").replace("-", "m")


def block_ranges(tokens: int, verification_block: int) -> list[tuple[int, int]]:
    if SEQUENCE_BLOCK % verification_block:
        raise ValueError("verification block must divide the sequence block")
    ranges = []
    for sequence_start in range(0, tokens, SEQUENCE_BLOCK):
        sequence_stop = min(sequence_start + SEQUENCE_BLOCK, tokens)
        for start in range(sequence_start, sequence_stop, verification_block):
            stop = start + verification_block
            if stop <= sequence_stop:
                ranges.append((start, stop))
    if sum(stop - start for start, stop in ranges) != tokens:
        raise RuntimeError("verification blocks did not cover every token")
    return ranges


def numeric_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("summary values must be non-empty and finite")
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def paired_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    resamples: int,
    seed: int,
    transform: str,
    token_count_per_block: int | None = None,
) -> dict[str, Any]:
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("paired arrays must be equally shaped vectors")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, numerator.size, size=(resamples, numerator.size))
    num = numerator[sampled].sum(axis=1)
    den = denominator[sampled].sum(axis=1)
    if transform == "reduction":
        estimates = 1.0 - num / den
    elif transform == "mean":
        if token_count_per_block is None:
            raise ValueError("mean transform needs token_count_per_block")
        estimates = num / (numerator.size * token_count_per_block)
    else:
        raise ValueError(transform)
    return {
        "method": "paired verification-block percentile bootstrap",
        "confidence": 0.95,
        "resamples": resamples,
        "seed": seed,
        "interval": percentile(estimates),
    }


def aggregate_method(
    raw: list[dict[str, Any]],
    natural_raw: list[dict[str, Any]],
    *,
    verification_block: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    union = np.asarray([row["union_count"] for row in raw], dtype=np.float64)
    natural_union = np.asarray(
        [row["union_count"] for row in natural_raw], dtype=np.float64
    )
    cold = np.asarray([row["cold_union_count"] for row in raw], dtype=np.float64)
    natural_cold = np.asarray(
        [row["cold_union_count"] for row in natural_raw], dtype=np.float64
    )
    total_kl = np.asarray([row["total_local_kl"] for row in raw], dtype=np.float64)
    total_union = int(union.sum())
    total_natural = int(natural_union.sum())
    union_reduction = 1.0 - total_union / total_natural
    total_cold = int(cold.sum())
    total_natural_cold = int(natural_cold.sum())
    cold_reduction = (
        1.0 - total_cold / total_natural_cold
        if total_natural_cold > 0
        else None
    )
    result = {
        "blocks": len(raw),
        "tokens": len(raw) * verification_block,
        "total_union_expert_instances": total_union,
        "total_natural_union_expert_instances": total_natural,
        "union_reduction_vs_natural": union_reduction,
        "union_reduction_bootstrap_95": paired_bootstrap(
            union,
            natural_union,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            transform="reduction",
        ),
        "mean_local_kl": float(total_kl.sum() / (len(raw) * verification_block)),
        "mean_local_kl_bootstrap_95": paired_bootstrap(
            total_kl,
            np.ones_like(total_kl),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            transform="mean",
            token_count_per_block=verification_block,
        ),
        "union_count": numeric_summary(union.tolist()),
        "cold_union_count": numeric_summary(cold.tolist()),
        "total_cold_union_expert_instances": total_cold,
        "total_natural_cold_union_expert_instances": total_natural_cold,
        "cold_union_reduction_vs_natural": cold_reduction,
        "changed_token_fraction": float(
            np.mean([row["changed_token_fraction"] for row in raw])
        ),
        "mean_natural_route_jaccard": float(
            np.mean([row["mean_natural_route_jaccard"] for row in raw])
        ),
        "mean_router_mass_loss": float(
            np.mean([row["mean_router_mass_loss"] for row in raw])
        ),
        "bf16_union_bytes": total_union * BF16_BYTES_PER_EXPERT,
        "int4_union_bytes": total_union * INT4_BYTES_PER_EXPERT,
        "bf16_natural_union_bytes": total_natural * BF16_BYTES_PER_EXPERT,
        "int4_natural_union_bytes": total_natural * INT4_BYTES_PER_EXPERT,
        "raw_blocks": raw,
    }
    return result


def compare_to_mass_budget(
    oracle_raw: list[dict[str, Any]],
    mass_raw: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    oracle = np.asarray(
        [row["union_count"] for row in oracle_raw], dtype=np.float64
    )
    mass = np.asarray([row["union_count"] for row in mass_raw], dtype=np.float64)
    return {
        "oracle_total_union": int(oracle.sum()),
        "mass_budget_total_union": int(mass.sum()),
        "additional_union_reduction_vs_mass_budget": float(
            1.0 - oracle.sum() / mass.sum()
        ),
        "bootstrap_95": paired_bootstrap(
            oracle,
            mass,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            transform="reduction",
        ),
    }


def evaluate_blocks(
    slates: list[list[Any]],
    *,
    verification_block: int,
    beam_width: int,
    hot_cache_mask: int,
    exact: bool,
    exact_fixed_cache: bool,
) -> dict[str, Any]:
    raw_by_method: dict[str, list[dict[str, Any]]] = {}
    ranges = block_ranges(len(slates), verification_block)
    for block_index, (start, stop) in enumerate(ranges):
        block = slates[start:stop]
        natural = natural_solution(block)
        solutions: list[BlockSolution] = [
            natural,
            mass_budget_solution(block, MASS_BUDGET_DELTA),
            fixed_cache_prior_solution(block, hot_cache_mask),
            marginal_union_greedy(block),
            eligible_set_pruning(block),
            beam_union_solution(block, beam_width),
        ]
        if exact:
            solutions.append(exact_ilp_solution(block))
        if exact_fixed_cache:
            solutions.append(exact_ilp_solution(block, cache_mask=hot_cache_mask))
        for solution in solutions:
            record = solution_metrics(
                block,
                solution,
                natural=natural,
                cache_mask=hot_cache_mask,
            )
            record["block_index"] = block_index
            record["token_range"] = [start, stop]
            raw_by_method.setdefault(solution.method, []).append(record)
    return {
        "verification_block": verification_block,
        "block_ranges": [list(item) for item in ranges],
        "raw_by_method": raw_by_method,
    }


def main() -> None:
    args = checked_args(parse_args())
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    artifact_path = ROOT / ARTIFACT_RELATIVE
    calibration_path = ROOT / CALIBRATION_RELATIVE
    preregistration_path = ROOT / PREREGISTRATION_RELATIVE
    for path in (artifact_path, calibration_path, preregistration_path):
        if not path.exists():
            raise FileNotFoundError(path)

    phase = time.perf_counter()
    input_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in (artifact_path, calibration_path, preregistration_path)
    }
    repository = git_state()
    initial_hardware = hardware_state()
    artifact = load_file(artifact_path, device="cpu")
    subsets = artifact["subsets"].long()
    original_index = original_subset_index(subsets)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))[
        "payload"
    ]
    layer_calibration = next(
        row for row in calibration["layers"] if int(row["layer"]) == 26
    )
    hot_experts = tuple(int(value) for value in layer_calibration["hot_expert_ids"])
    if len(hot_experts) != 32:
        raise RuntimeError("fixed cache calibration must contain 32 experts")
    hot_cache_mask = experts_to_mask(hot_experts)
    timings["load_inputs_and_environment_seconds"] = time.perf_counter() - phase

    thresholds = (PRIMARY_THRESHOLD,) if args.stage == "smoke" else THRESHOLDS
    caps = (PRIMARY_CAP,) if args.stage == "smoke" else SLATE_CAPS
    verification_blocks = (4, 8) if args.stage == "smoke" else VERIFICATION_BLOCKS
    results: dict[str, Any] = {}
    exact_solver_records: list[dict[str, Any]] = []
    phase = time.perf_counter()
    for split_index, split in enumerate(args.splits):
        trace_base = 0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        selected = slice(trace_base, trace_base + args.tokens_per_split)
        top12_ids = artifact["top12_expert_ids"][selected].long()
        top12_weights = artifact["top12_router_weights"][selected].float()
        subset_kl = artifact["subset_kl"][selected].float()
        original_kl = subset_kl[:, original_index]
        split_result: dict[str, Any] = {
            "trace_indices": list(
                range(trace_base, trace_base + args.tokens_per_split)
            ),
            "original_route_control": {
                "subset_index": original_index,
                "present_once_in_subset_table": True,
                "local_kl": numeric_summary(original_kl.tolist()),
                "maximum_le_1e_5": bool(original_kl.max().item() <= 1e-5),
            },
            "configurations": {},
        }
        for threshold in thresholds:
            for cap in caps:
                config_id = configuration_id(threshold, cap)
                slates = build_candidate_slates(
                    top12_ids,
                    top12_weights,
                    subset_kl,
                    subsets,
                    threshold=threshold,
                    cap=cap,
                )
                counts = [len(candidates) for candidates in slates]
                slate_digest = hashlib.sha256()
                for candidates in slates:
                    for candidate in candidates:
                        slate_digest.update(
                            int(candidate.subset_index).to_bytes(2, "little")
                        )
                config: dict[str, Any] = {
                    "threshold": threshold,
                    "slate_cap": cap,
                    "candidate_count": numeric_summary(counts),
                    "all_slates_contain_natural_once": all(
                        sum(candidate.natural for candidate in candidates) == 1
                        for candidates in slates
                    ),
                    "slate_subset_index_sha256": slate_digest.hexdigest(),
                    "block_sizes": {},
                }
                for verification_block in verification_blocks:
                    is_primary = (
                        math.isclose(threshold, PRIMARY_THRESHOLD)
                        and cap == PRIMARY_CAP
                        and verification_block == PRIMARY_BLOCK
                    )
                    exact = verification_block == 4 or is_primary
                    raw = evaluate_blocks(
                        slates,
                        verification_block=verification_block,
                        beam_width=args.beam_width,
                        hot_cache_mask=hot_cache_mask,
                        exact=exact,
                        exact_fixed_cache=is_primary,
                    )
                    aggregated = {
                        method: aggregate_method(
                            records,
                            raw["raw_by_method"]["natural"],
                            verification_block=verification_block,
                            bootstrap_resamples=args.bootstrap_resamples,
                            bootstrap_seed=args.seed + split_index,
                        )
                        for method, records in raw["raw_by_method"].items()
                    }
                    cell: dict[str, Any] = {
                        "verification_block": verification_block,
                        "methods": aggregated,
                    }
                    if exact:
                        cell["exact_vs_mass_budget"] = compare_to_mass_budget(
                            raw["raw_by_method"]["exact_ilp"],
                            raw["raw_by_method"][
                                f"mass_budget_delta_{MASS_BUDGET_DELTA:g}"
                            ],
                            bootstrap_resamples=args.bootstrap_resamples,
                            bootstrap_seed=args.seed + split_index,
                        )
                        exact_total = aggregated["exact_ilp"][
                            "total_union_expert_instances"
                        ]
                        beam_total = aggregated[f"beam_{args.beam_width}"][
                            "total_union_expert_instances"
                        ]
                        cell["beam_vs_exact"] = {
                            "beam_total_union": beam_total,
                            "exact_total_union": exact_total,
                            "relative_union_gap": (beam_total - exact_total)
                            / exact_total,
                            "blocks_matching_exact_union_count": sum(
                                beam_row["union_count"] == exact_row["union_count"]
                                for beam_row, exact_row in zip(
                                    raw["raw_by_method"][f"beam_{args.beam_width}"],
                                    raw["raw_by_method"]["exact_ilp"],
                                    strict=True,
                                )
                            ),
                            "blocks": len(raw["raw_by_method"]["exact_ilp"]),
                        }
                        exact_solver_records.extend(
                            row["diagnostics"]
                            for row in raw["raw_by_method"]["exact_ilp"]
                        )
                    if is_primary:
                        exact_cache = aggregated["exact_ilp_fixed_cache"]
                        natural_cache = aggregated["natural"]
                        cell["fixed_cache_exact_oracle"] = {
                            "hot_expert_ids": list(hot_experts),
                            "exact_total_cold_union": exact_cache[
                                "total_cold_union_expert_instances"
                            ],
                            "natural_total_cold_union": natural_cache[
                                "total_cold_union_expert_instances"
                            ],
                            "cold_union_reduction_vs_natural": exact_cache[
                                "cold_union_reduction_vs_natural"
                            ],
                        }
                        exact_solver_records.extend(
                            row["diagnostics"]
                            for row in raw["raw_by_method"][
                                "exact_ilp_fixed_cache"
                            ]
                        )
                    config["block_sizes"][str(verification_block)] = cell
                    print(
                        f"coalescing[{split}][{config_id}][b{verification_block}] "
                        f"beam_reduction={aggregated[f'beam_{args.beam_width}']['union_reduction_vs_natural']:.4f}",
                        flush=True,
                    )
                split_result["configurations"][config_id] = config
        results[split] = split_result
    timings["all_block_optimizations_seconds"] = time.perf_counter() - phase

    if args.stage == "smoke":
        verdict = "smoke_passed_not_adjudicated"
        gates: dict[str, Any] = {
            "adjudicated": False,
            "reason": "fixed validation and test windows are required",
        }
    else:
        primary_id = configuration_id(PRIMARY_THRESHOLD, PRIMARY_CAP)
        primary_by_split = {}
        for split in ("validation", "test"):
            cell = results[split]["configurations"][primary_id]["block_sizes"][
                str(PRIMARY_BLOCK)
            ]
            exact = cell["methods"]["exact_ilp"]
            additional = cell["exact_vs_mass_budget"]
            primary_by_split[split] = {
                "union_reduction_vs_natural": exact[
                    "union_reduction_vs_natural"
                ],
                "union_reduction_bootstrap_95": exact[
                    "union_reduction_bootstrap_95"
                ],
                "additional_union_reduction_vs_mass_budget": additional[
                    "additional_union_reduction_vs_mass_budget"
                ],
                "additional_reduction_bootstrap_95": additional["bootstrap_95"],
                "mean_local_kl": exact["mean_local_kl"],
                "mean_local_kl_bootstrap_95": exact[
                    "mean_local_kl_bootstrap_95"
                ],
                "beam_relative_union_gap": cell["beam_vs_exact"][
                    "relative_union_gap"
                ],
                "exact_total_union": exact["total_union_expert_instances"],
                "natural_total_union": exact[
                    "total_natural_union_expert_instances"
                ],
                "mass_budget_total_union": additional[
                    "mass_budget_total_union"
                ],
            }
        natural_gate = all(
            row["union_reduction_vs_natural"] >= 0.40
            for row in primary_by_split.values()
        )
        mass_gate = all(
            row["additional_union_reduction_vs_mass_budget"] >= 0.25
            for row in primary_by_split.values()
        )
        kl_gate = all(
            row["mean_local_kl"] <= PRIMARY_THRESHOLD
            for row in primary_by_split.values()
        )
        solver_gate = bool(
            exact_solver_records
            and all(
                record["status"] == 0
                and record["success"]
                and abs(record["mip_gap"]) <= 1e-12
                for record in exact_solver_records
            )
        )
        original_gate = all(
            results[split]["original_route_control"]["maximum_le_1e_5"]
            for split in ("validation", "test")
        )
        controls_gate = solver_gate and original_gate
        hard_falsification = {
            "any_split_union_reduction_lt_0_25": any(
                row["union_reduction_vs_natural"] < 0.25
                for row in primary_by_split.values()
            ),
            "both_splits_additional_reduction_lt_0_10": all(
                row["additional_union_reduction_vs_mass_budget"] < 0.10
                for row in primary_by_split.values()
            ),
            "any_split_mean_kl_gt_0_001": any(
                row["mean_local_kl"] > PRIMARY_THRESHOLD
                for row in primary_by_split.values()
            ),
            "exact_control_failed": not controls_gate,
        }
        hard = any(hard_falsification.values())
        positive = natural_gate and mass_gate and kl_gate and controls_gate
        if positive:
            verdict = "layer26_positive_opens_layer23_preregistration"
        elif hard:
            verdict = "oracle_negative_hard_falsification"
        else:
            verdict = "inconclusive_negative_no_downstream"
        gates = {
            "adjudicated": True,
            "primary": {
                "verification_block": PRIMARY_BLOCK,
                "threshold": PRIMARY_THRESHOLD,
                "slate_cap": PRIMARY_CAP,
            },
            "primary_by_split": primary_by_split,
            "union_reduction_ge_0_40_both_splits": natural_gate,
            "additional_reduction_vs_mass_budget_ge_0_25_both_splits": mass_gate,
            "mean_local_kl_le_0_001_both_splits": kl_gate,
            "exact_controls_pass": controls_gate,
            "all_highs_solutions_optimal": solver_gate,
            "original_route_numerical_control_pass": original_gate,
            "hard_falsification": {**hard_falsification, "triggered": hard},
        }

    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    report = {
        "schema_version": 1,
        "kind": "craft_moe_h2_block_route_coalescing_layer26",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H2_BLOCK_COALESCING",
        "verdict": verdict,
        "preregistration": str(preregistration_path.resolve()),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "layer": 26,
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "tokens_per_split": args.tokens_per_split,
            "window": {
                split: f"first {args.tokens_per_split} route-trace tokens"
                for split in args.splits
            },
            "sequence_block": SEQUENCE_BLOCK,
        },
        "configuration": {
            "thresholds": list(thresholds),
            "slate_caps": list(caps),
            "verification_blocks": list(verification_blocks),
            "primary": {
                "threshold": PRIMARY_THRESHOLD,
                "slate_cap": PRIMARY_CAP,
                "verification_block": PRIMARY_BLOCK,
            },
            "beam_width": args.beam_width,
            "mass_budget_delta": MASS_BUDGET_DELTA,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "fixed_cache_hot_experts": list(hot_experts),
            "route_weights_renormalized": False,
            "slate_order": "stable (local KL, subset index), natural forced present",
            "exact_cells": "all block-4 cells and primary block-8 cell",
        },
        "gates": gates,
        "results": results,
        "solver": {
            "implementation": "scipy.optimize.milp backed by HiGHS",
            "scipy_version": scipy.__version__,
            "exact_solution_count": len(exact_solver_records),
            "solve_seconds": numeric_summary(
                [record["solve_seconds"] for record in exact_solver_records]
            ),
            "mip_node_count": numeric_summary(
                [record["mip_node_count"] for record in exact_solver_records]
            ),
            "all_status_optimal": all(
                record["status"] == 0 for record in exact_solver_records
            ),
            "all_mip_gap_le_1e_12": all(
                abs(record["mip_gap"]) <= 1e-12
                for record in exact_solver_records
            ),
        },
        "accounting": {
            "weights_per_routed_expert": WEIGHTS_PER_EXPERT,
            "bf16_bytes_per_routed_expert": BF16_BYTES_PER_EXPERT,
            "int4_bytes_per_routed_expert": INT4_BYTES_PER_EXPERT,
            "scope": "unique expert instances per layer-26 verification block",
            "not_wallclock": True,
        },
        "controls": {
            "original_subset_index": original_index,
            "subset_count": int(subsets.shape[0]),
            "subset_table_is_12_choose_6": int(subsets.shape[0]) == math.comb(12, 6),
            "hot_cache_calibration": layer_calibration,
        },
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "psutil": psutil.__version__,
            },
            "inputs": {
                "sha256": input_hashes,
                "artifact_metadata": metadata(artifact_path),
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "late-layer local route-union oracle; no layer-23 downstream run unless the primary gate passes",
            "future draft routes are assumed known and every route is selected using exact teacher KL, so the exact oracle is not deployable",
            "union counts and BF16/int4 bytes are deterministic accounting, not cache traffic or wall-clock",
            "no speculative acceptance tree, commitment probability, packed runtime, or accepted-token throughput is measured",
            "the route-equivalence artifact predates H2; this screen is exploratory rather than fresh confirmation",
            "no novelty claim is made because EcoSpec, AcceptMoE, and EdgeXpert are close prior art",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    print(f"gates={json.dumps(gates, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
