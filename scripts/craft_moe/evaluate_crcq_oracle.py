from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow
import pyarrow.parquet as pq
import safetensors
import torch
import torch.nn.functional as F
import tokenizers
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer

from moe_lab.behavioral import rmsnorm
from moe_lab.craft_moe.crcq import (
    BudgetSolution,
    best_by_upgrade_count,
    best_schedule_within_fraction,
    force_natural_shortlist,
    mean_gap_closure,
    mixed_precision_routed,
    natural_subset_index,
    routed_for_routes,
    routed_from_choices,
    six_of_twelve_subsets,
    solve_minimum_budget,
)
from moe_lab.dynamic_precision import binary_upgrade_masks
from moe_lab.moe_layer import LoadedMoELayer, ProjectionWeights, load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.quantization import fake_quantize_symmetric_per_row_
from moe_lab.reporting import ROOT, git_revision


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
COMPONENT_RELATIVE = Path(
    "data/traces/layer26_dynamic_precision_components.safetensors"
)
ROUTE_RELATIVE = Path("data/traces/layer26_route_equivalence_full.safetensors")
BLOCK_SIZE = 128
TRACE_TOKENS_PER_SPLIT = 1024
FULL_TOKENS_PER_SPLIT = 256
SHORTLIST_SIZE = 32
SEED = 20260810
TARGET_MULTIPLIER = 1.01
RATE_FRACTIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1 / 3, 0.50, 1.0)


@dataclass
class TeacherReference:
    log_probs: torch.Tensor
    top1: torch.Tensor
    true_token_nll: torch.Tensor
    token_ids: torch.Tensor
    blocks: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-KL top-32 CRCQ route and bit-width oracle screen."
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
    parser.add_argument("--shortlist-size", type=int, default=SHORTLIST_SIZE)
    parser.add_argument("--candidate-batch", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/craft_moe/crcq_oracle.json"),
    )
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if not 1 <= args.tokens_per_split <= TRACE_TOKENS_PER_SPLIT:
        raise ValueError("tokens-per-split is outside the existing trace")
    if args.stage == "smoke" and (
        args.tokens_per_split > 32 or args.splits != ("validation",)
    ):
        raise ValueError("smoke must use at most 32 validation tokens only")
    if args.stage == "full" and (
        args.tokens_per_split != FULL_TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.bootstrap_resamples != 10_000
        or args.seed != SEED
    ):
        raise ValueError(
            "full run is fixed at 256 validation + 256 test, 10k bootstrap, fixed seed"
        )
    if args.shortlist_size != SHORTLIST_SIZE:
        raise ValueError("the preregistered shortlist size is fixed at 32")
    if args.candidate_batch < 1 or args.bootstrap_resamples < 1:
        raise ValueError("batch size and bootstrap count must be positive")
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output = output.resolve()
    reports = (ROOT / "reports").resolve()
    if reports not in output.parents:
        raise ValueError("output-json must be inside reports/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    args.output_json = output
    return args


def corpus_tokens(model_dir: Path, split: str, count: int) -> torch.Tensor:
    parquet = (
        ROOT
        / "data"
        / "corpora"
        / "wikitext"
        / "wikitext-2-raw-v1"
        / f"{split}-00000-of-00001.parquet"
    )
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    ids = tokenizer.encode(joined).ids[:count]
    if len(ids) != count:
        raise RuntimeError(f"{split} yielded only {len(ids)} tokens")
    return torch.tensor(ids, dtype=torch.long)


def sequence_blocks(count: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + BLOCK_SIZE, count))
        for start in range(0, count, BLOCK_SIZE)
    ]


def prediction_mask(count: int, blocks: list[tuple[int, int]]) -> torch.Tensor:
    mask = torch.ones(count, dtype=torch.bool)
    for start, stop in blocks:
        if stop > start:
            mask[stop - 1] = False
    return mask


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def command_result(arguments: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            arguments,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return {"available": False, "error": str(error)}
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def git_state() -> dict[str, Any]:
    status = command_result(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    return {
        "revision": git_revision(),
        "dirty": bool(status.get("stdout")),
        "status_porcelain": status.get("stdout", "").splitlines(),
        "status_command": status,
    }


def hardware_state() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    gpu = command_result(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda |= {
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu": platform.processor(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "process_rss_bytes": process.memory_info().rss,
        "cuda": cuda,
        "nvidia_smi": gpu,
    }


def metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def quantized_copy(weights: ProjectionWeights, bits: int) -> ProjectionWeights:
    copied = ProjectionWeights(
        gate=weights.gate.detach().clone(),
        up=weights.up.detach().clone(),
        down=weights.down.detach().clone(),
    )
    for weight in (copied.gate, copied.up, copied.down):
        fake_quantize_symmetric_per_row_(weight, bits)
    return copied


@torch.inference_mode()
def selected_precision_outputs(
    moe: LoadedMoELayer, flat_input: torch.Tensor, top12_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = flat_input.to(moe.device)
    ids = top12_ids.to(moe.device).long()
    tokens, slots = ids.shape
    shape = (tokens, slots, inputs.shape[-1])
    bf16 = torch.empty(shape, dtype=inputs.dtype, device=moe.device)
    q3 = torch.empty_like(bf16)
    q4 = torch.empty_like(bf16)
    touched = 0
    for expert_id, expert in enumerate(moe.experts):
        positions = (ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            touched += 1
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            expert_input = inputs[token_indices]
            bf16[token_indices, slot_indices] = moe.expert_forward(expert_input, expert)
            three = quantized_copy(expert, 3)
            q3[token_indices, slot_indices] = moe.expert_forward(expert_input, three)
            del three
            four = quantized_copy(expert, 4)
            q4[token_indices, slot_indices] = moe.expert_forward(expert_input, four)
            del four
        if expert_id % 8 == 7:
            print(f"precision_experts={expert_id + 1}/64", flush=True)
    for name, output in (("bf16", bf16), ("q3", q3), ("q4", q4)):
        if not torch.isfinite(output.float()).all():
            raise RuntimeError(f"non-finite {name} expert output")
    print(f"experts_touched={touched}/64", flush=True)
    return bf16.cpu(), q3.cpu(), q4.cpu()


def regression_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    error = candidate.float() - reference.float()
    rmse = error.square().mean().sqrt()
    scale = reference.float().square().mean().sqrt().clamp_min(1e-30)
    return {
        "nrmse": float((rmse / scale).item()),
        "rmse": float(rmse.item()),
        "mean_absolute_error": float(error.abs().mean().item()),
        "maximum_absolute_error": float(error.abs().max().item()),
    }


def aligned_natural_outputs(
    top12_ids: torch.Tensor,
    natural_ids: torch.Tensor,
    natural_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    match = top12_ids[:, :6].unsqueeze(2) == natural_ids.unsqueeze(1)
    if not (match.sum(dim=2) == 1).all():
        raise RuntimeError("top-12 first six do not match the natural top-6 route")
    positions = match.long().argmax(dim=2)
    aligned = natural_outputs.gather(
        1, positions.unsqueeze(-1).expand(-1, -1, natural_outputs.shape[-1])
    )
    return aligned, positions


@torch.inference_mode()
def make_teacher_reference(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    blocks: list[tuple[int, int]],
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int,
) -> TeacherReference:
    tokens = hidden.shape[0]
    log_probs = torch.empty(tokens, lm_head.shape[0], dtype=torch.float32)
    top1 = torch.empty(tokens, dtype=torch.long)
    for start in range(0, tokens, batch_size):
        stop = min(start + batch_size, tokens)
        states = hidden[start:stop].to(lm_head.device)
        logits = F.linear(rmsnorm(states, norm_weight), lm_head).float()
        batch_log_probs = F.log_softmax(logits, dim=-1)
        log_probs[start:stop] = batch_log_probs.cpu()
        top1[start:stop] = logits.argmax(dim=-1).cpu()
    valid = prediction_mask(tokens, blocks)
    positions = valid.nonzero(as_tuple=False).squeeze(1)
    nll = torch.full((tokens,), float("nan"), dtype=torch.float32)
    nll[positions] = -log_probs[positions, token_ids[positions + 1]]
    return TeacherReference(log_probs, top1, nll, token_ids, blocks)


def patched_candidates(
    teacher: torch.Tensor, natural_bf16_routed: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    return (
        teacher.float().unsqueeze(0)
        + (candidates.float() - natural_bf16_routed.float().unsqueeze(0))
    ).to(teacher.dtype)


@torch.inference_mode()
def exact_kl_for_states(
    states: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    damage = torch.empty(states.shape[0], dtype=torch.float32)
    teacher_log_probs_device = teacher_log_probs.to(lm_head.device)
    teacher_probs = teacher_log_probs_device.exp()
    for start in range(0, states.shape[0], batch_size):
        stop = min(start + batch_size, states.shape[0])
        candidates = states[start:stop].to(lm_head.device)
        logits = F.linear(rmsnorm(candidates, norm_weight), lm_head).float()
        candidate_log_probs = F.log_softmax(logits, dim=-1)
        kl = (
            teacher_probs.unsqueeze(0)
            * (teacher_log_probs_device.unsqueeze(0) - candidate_log_probs)
        ).sum(dim=-1)
        damage[start:stop] = kl.clamp_min(0.0).cpu()
    return damage


def _block_sums_counts(
    values: np.ndarray, blocks: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    sums, counts = [], []
    for start, stop in blocks:
        selected = values[start:stop]
        finite = np.isfinite(selected)
        sums.append(float(selected[finite].sum()))
        counts.append(int(finite.sum()))
    return np.asarray(sums), np.asarray(counts)


def percentile(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, (0.025, 0.975), method="linear")
    return {"low": float(low), "high": float(high)}


def nullable(values: torch.Tensor) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


@torch.inference_mode()
def evaluate_hidden(
    hidden: torch.Tensor,
    reference: TeacherReference,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    tokens = hidden.shape[0]
    kl = torch.empty(tokens, dtype=torch.float32)
    agreement = torch.empty(tokens, dtype=torch.bool)
    nll = torch.full((tokens,), float("nan"), dtype=torch.float32)
    valid = prediction_mask(tokens, reference.blocks)
    for start in range(0, tokens, batch_size):
        stop = min(start + batch_size, tokens)
        states = hidden[start:stop].to(lm_head.device)
        logits = F.linear(rmsnorm(states, norm_weight), lm_head).float()
        log_probs = F.log_softmax(logits, dim=-1)
        teacher_log_probs = reference.log_probs[start:stop].to(lm_head.device)
        kl[start:stop] = (
            teacher_log_probs.exp() * (teacher_log_probs - log_probs)
        ).sum(dim=-1).clamp_min(0.0).cpu()
        agreement[start:stop] = logits.argmax(dim=-1).cpu() == reference.top1[start:stop]
        local_valid = valid[start:stop]
        if local_valid.any():
            local = local_valid.nonzero(as_tuple=False).squeeze(1)
            global_positions = start + local
            nll[global_positions] = -log_probs[
                local, reference.token_ids[global_positions + 1].to(lm_head.device)
            ].cpu()
    prediction = valid.nonzero(as_tuple=False).squeeze(1)
    teacher_ce = reference.true_token_nll[prediction].double().mean()
    candidate_ce = nll[prediction].double().mean()
    delta = candidate_ce - teacher_ce
    raw_series = {
        "teacher_to_candidate_kl": kl.double().numpy(),
        "top1_agreement": agreement.double().numpy(),
        "cross_entropy_delta": (nll.double() - reference.true_token_nll.double()).numpy(),
    }
    block_stats = {
        key: _block_sums_counts(value, reference.blocks)
        for key, value in raw_series.items()
    }
    teacher_stats = _block_sums_counts(
        reference.true_token_nll.double().numpy(), reference.blocks
    )
    candidate_stats = _block_sums_counts(nll.double().numpy(), reference.blocks)
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(
        0,
        len(reference.blocks),
        size=(bootstrap_resamples, len(reference.blocks)),
    )
    intervals = {}
    for key, (sums, counts) in block_stats.items():
        estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        intervals[key] = percentile(estimates)
    teacher_sums, teacher_counts = teacher_stats
    candidate_sums, candidate_counts = candidate_stats
    teacher_samples = teacher_sums[sampled].sum(axis=1) / teacher_counts[
        sampled
    ].sum(axis=1)
    candidate_samples = candidate_sums[sampled].sum(axis=1) / candidate_counts[
        sampled
    ].sum(axis=1)
    intervals["relative_cross_entropy_delta"] = percentile(
        (candidate_samples - teacher_samples) / teacher_samples
    )
    return {
        "aggregate": {
            "teacher_to_candidate_kl": float(kl.double().mean().item()),
            "top1_agreement": float(agreement.double().mean().item()),
            "teacher_cross_entropy": float(teacher_ce.item()),
            "candidate_cross_entropy": float(candidate_ce.item()),
            "cross_entropy_delta": float(delta.item()),
            "relative_cross_entropy_delta": float((delta / teacher_ce).item()),
        },
        "bootstrap_95": {
            "method": "paired sequence-block percentile bootstrap",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "sequence_blocks": len(reference.blocks),
            "intervals": intervals,
        },
        "raw": {
            "teacher_to_candidate_kl": kl.tolist(),
            "top1_agreement": [bool(value) for value in agreement.tolist()],
            "candidate_true_token_nll": nullable(nll),
        },
    }


def stage_a_all_q3(
    label: str,
    q3: torch.Tensor,
    weights: torch.Tensor,
    subsets: torch.Tensor,
    teacher: torch.Tensor,
    natural_bf16: torch.Tensor,
    reference: TeacherReference,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
) -> torch.Tensor:
    damage = torch.empty(q3.shape[0], subsets.shape[0], dtype=torch.float32)
    for token in range(q3.shape[0]):
        routed = routed_for_routes(q3[token], weights[token], subsets)
        candidates = patched_candidates(teacher[token], natural_bf16[token], routed)
        damage[token] = exact_kl_for_states(
            candidates,
            reference.log_probs[token],
            norm_weight,
            lm_head,
            candidate_batch,
        )
        if token % 4 == 3 or token + 1 == q3.shape[0]:
            print(f"stage_a[{label}]={token + 1}/{q3.shape[0]}", flush=True)
    return damage


def stage_c_route_bits(
    label: str,
    bf16: torch.Tensor,
    q3: torch.Tensor,
    q4: torch.Tensor,
    weights: torch.Tensor,
    subsets: torch.Tensor,
    shortlist: torch.Tensor,
    masks: torch.Tensor,
    teacher: torch.Tensor,
    natural_bf16: torch.Tensor,
    reference: TeacherReference,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    damage = torch.empty(
        q3.shape[0], shortlist.shape[1], masks.shape[0], dtype=torch.float32
    )
    bf16_damage = torch.empty(q3.shape[0], shortlist.shape[1], dtype=torch.float32)
    for token in range(q3.shape[0]):
        routes = subsets[shortlist[token]]
        mixed = mixed_precision_routed(
            q3[token], q4[token], weights[token], routes, masks
        )
        candidates = patched_candidates(
            teacher[token], natural_bf16[token], mixed.reshape(-1, mixed.shape[-1])
        )
        damage[token] = exact_kl_for_states(
            candidates,
            reference.log_probs[token],
            norm_weight,
            lm_head,
            candidate_batch,
        ).view(shortlist.shape[1], masks.shape[0])
        routed_bf16 = routed_for_routes(bf16[token], weights[token], routes)
        bf16_candidates = patched_candidates(
            teacher[token], natural_bf16[token], routed_bf16
        )
        bf16_damage[token] = exact_kl_for_states(
            bf16_candidates,
            reference.log_probs[token],
            norm_weight,
            lm_head,
            candidate_batch,
        )
        if token % 2 == 1 or token + 1 == q3.shape[0]:
            print(f"stage_c[{label}]={token + 1}/{q3.shape[0]}", flush=True)
    return damage, bf16_damage


def choices_for_schedule(
    per_token_cost: torch.Tensor,
    best_route: torch.Tensor,
    best_mask: torch.Tensor,
    shortlist: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = per_token_cost.numel()
    token_indices = torch.arange(tokens)
    local_routes = best_route[token_indices, per_token_cost]
    masks = best_mask[token_indices, per_token_cost]
    routes = shortlist[token_indices, local_routes]
    return routes, masks


def schedule_record(
    *,
    q3: torch.Tensor,
    q4: torch.Tensor,
    weights: torch.Tensor,
    subsets: torch.Tensor,
    masks: torch.Tensor,
    routes: torch.Tensor,
    mask_indices: torch.Tensor,
    teacher: torch.Tensor,
    natural_bf16: torch.Tensor,
    reference: TeacherReference,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    routed = routed_from_choices(
        q3, q4, weights, subsets, routes, masks, mask_indices
    )
    hidden = (
        teacher.float() + (routed.float() - natural_bf16.float())
    ).to(teacher.dtype)
    metrics = evaluate_hidden(
        hidden,
        reference,
        norm_weight,
        lm_head,
        candidate_batch,
        bootstrap_resamples,
        bootstrap_seed,
    )
    metrics["raw"]["route_index"] = routes.tolist()
    metrics["raw"]["mask_index"] = mask_indices.tolist()
    metrics["raw"]["upgrade_count"] = masks[mask_indices].sum(dim=1).tolist()
    return metrics


def gate_bootstrap(
    *,
    natural_q3: torch.Tensor,
    natural_q4: torch.Tensor,
    alternative_q3: torch.Tensor,
    natural_best: torch.Tensor,
    joint_best: torch.Tensor,
    blocks: list[tuple[int, int]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(resamples, len(blocks)))
    canonical = np.sort(sampled, axis=1)
    unique, inverse = np.unique(canonical, axis=0, return_inverse=True)
    closure_unique = np.empty(unique.shape[0])
    natural_fraction_unique = np.empty(unique.shape[0])
    joint_fraction_unique = np.empty(unique.shape[0])
    for row_index, row in enumerate(unique):
        token_indices = torch.cat(
            [torch.arange(*blocks[int(block)]) for block in row]
        )
        q3 = natural_q3[token_indices]
        q4 = natural_q4[token_indices]
        alternative = alternative_q3[token_indices]
        closure_unique[row_index] = mean_gap_closure(q3, q4, alternative)
        reference_mean = float(q4.double().mean().item())
        natural_solution = solve_minimum_budget(
            natural_best[token_indices],
            reference_mean,
            tolerance_multiplier=TARGET_MULTIPLIER,
        )
        joint_solution = solve_minimum_budget(
            joint_best[token_indices],
            reference_mean,
            tolerance_multiplier=TARGET_MULTIPLIER,
        )
        natural_fraction_unique[row_index] = float(natural_solution.upgrade_fraction)
        joint_fraction_unique[row_index] = float(joint_solution.upgrade_fraction)
    return {
        "method": "paired sequence-block percentile bootstrap with memoized exact DP",
        "confidence": 0.95,
        "resamples": resamples,
        "seed": seed,
        "sequence_blocks": len(blocks),
        "unique_block_multisets": int(unique.shape[0]),
        "intervals": {
            "all_q3_alternative_mean_gap_closure": percentile(
                closure_unique[inverse]
            ),
            "natural_minimum_upgrade_fraction": percentile(
                natural_fraction_unique[inverse]
            ),
            "joint_top32_minimum_upgrade_fraction": percentile(
                joint_fraction_unique[inverse]
            ),
        },
    }


def solution_json(solution: BudgetSolution) -> dict[str, Any]:
    return {
        "target_total_kl": solution.target_total_damage,
        "total_upgrade_count": solution.total_cost,
        "upgrade_fraction": solution.upgrade_fraction,
        "average_active_bits": solution.average_active_bits,
        "per_token_upgrade_count": (
            solution.per_token_cost.tolist()
            if solution.per_token_cost is not None
            else None
        ),
        "exact_total_kl_curve": solution.exact_cost_curve.tolist(),
    }


def split_gate(
    joint_solution: BudgetSolution, gap_closure: float
) -> dict[str, Any]:
    fraction = float(joint_solution.upgrade_fraction)
    bits = float(joint_solution.average_active_bits)
    strong = {
        "joint_upgrade_fraction_le_0_15": fraction <= 0.15,
        "average_active_bits_le_3_15": bits <= 3.15,
        "all_q3_gap_closure_ge_0_50": gap_closure >= 0.50,
    }
    return {
        "strong_criteria": strong,
        "any_strong_criterion": any(strong.values()),
        "route_axis_negative_gap_closure_lt_0_10": gap_closure < 0.10,
        "joint_axis_negative_upgrade_fraction_gt_0_25": fraction > 0.25,
        "all_q3_alternative_mean_gap_closure": gap_closure,
        "joint_minimum_upgrade_fraction": fraction,
        "joint_average_active_bits": bits,
    }


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("CRCQ requires CUDA")
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
    for path in (model_dir, component_path, route_path):
        if not path.exists():
            raise FileNotFoundError(path)

    phase = time.perf_counter()
    input_hashes = {
        str(component_path.resolve()): sha256_file(component_path),
        str(route_path.resolve()): sha256_file(route_path),
    }
    timings["input_sha256_seconds"] = time.perf_counter() - phase
    initial_hardware = hardware_state()
    repository = git_state()

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
    expected_subsets = six_of_twelve_subsets()
    if not torch.equal(subsets, expected_subsets):
        raise RuntimeError("route artifact subset order is not the fixed lexicographic order")
    natural_index = natural_subset_index(subsets)
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
    masks = binary_upgrade_masks(6)
    split_slices: dict[str, slice] = {}
    references: dict[str, TeacherReference] = {}
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

        stage_a_started = time.perf_counter()
        all_q3_damage = stage_a_all_q3(
            split,
            split_q3,
            split_weights,
            subsets,
            split_teacher,
            split_natural_bf16,
            reference,
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        stage_a_seconds = time.perf_counter() - stage_a_started
        shortlist, natural_forced = force_natural_shortlist(
            all_q3_damage, natural_index, args.shortlist_size
        )
        alternative_damage = all_q3_damage.clone()
        alternative_damage[:, natural_index] = torch.inf
        best_alternative_q3, best_alternative_routes = alternative_damage.min(dim=1)

        stage_c_started = time.perf_counter()
        stage_c_damage, shortlist_bf16_damage = stage_c_route_bits(
            split,
            split_bf16,
            split_q3,
            split_q4,
            split_weights,
            subsets,
            shortlist,
            masks,
            split_teacher,
            split_natural_bf16,
            reference,
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        stage_c_seconds = time.perf_counter() - stage_c_started
        natural_local = (shortlist == natural_index).nonzero(as_tuple=False)
        natural_local_by_token = torch.empty(args.tokens_per_split, dtype=torch.long)
        natural_local_by_token[natural_local[:, 0]] = natural_local[:, 1]
        token_index = torch.arange(args.tokens_per_split)
        natural_mask_damage = stage_c_damage[token_index, natural_local_by_token]
        natural_q3_damage = natural_mask_damage[:, 0]
        natural_q4_damage = natural_mask_damage[:, -1]
        stage_a_natural = all_q3_damage[:, natural_index]
        if float((natural_q3_damage - stage_a_natural).abs().max().item()) > 1e-7:
            raise RuntimeError("Stage A and Stage C natural all-Q3 KL disagree")

        joint_best, joint_route, joint_mask = best_by_upgrade_count(
            stage_c_damage, masks
        )
        natural_best, _, natural_best_mask = best_by_upgrade_count(
            natural_mask_damage.unsqueeze(1), masks
        )
        reference_q4_mean = float(natural_q4_damage.double().mean().item())
        natural_solution = solve_minimum_budget(
            natural_best,
            reference_q4_mean,
            tolerance_multiplier=TARGET_MULTIPLIER,
        )
        joint_solution = solve_minimum_budget(
            joint_best,
            reference_q4_mean,
            tolerance_multiplier=TARGET_MULTIPLIER,
        )
        closure = mean_gap_closure(
            natural_q3_damage, natural_q4_damage, best_alternative_q3
        )
        gates[split] = split_gate(joint_solution, closure)

        natural_routes = torch.full(
            (args.tokens_per_split,), natural_index, dtype=torch.long
        )
        all_q3_masks = torch.zeros(args.tokens_per_split, dtype=torch.long)
        all_q4_masks = torch.full(
            (args.tokens_per_split,), masks.shape[0] - 1, dtype=torch.long
        )
        baseline_records = {
            "natural_all_q3": schedule_record(
                q3=split_q3,
                q4=split_q4,
                weights=split_weights,
                subsets=subsets,
                masks=masks,
                routes=natural_routes,
                mask_indices=all_q3_masks,
                teacher=split_teacher,
                natural_bf16=split_natural_bf16,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
            ),
            "natural_all_q4": schedule_record(
                q3=split_q3,
                q4=split_q4,
                weights=split_weights,
                subsets=subsets,
                masks=masks,
                routes=natural_routes,
                mask_indices=all_q4_masks,
                teacher=split_teacher,
                natural_bf16=split_natural_bf16,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
            ),
            "best_alternative_all_q3": schedule_record(
                q3=split_q3,
                q4=split_q4,
                weights=split_weights,
                subsets=subsets,
                masks=masks,
                routes=best_alternative_routes,
                mask_indices=all_q3_masks,
                teacher=split_teacher,
                natural_bf16=split_natural_bf16,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
            ),
        }
        natural_control_routed = routed_from_choices(
            split_bf16,
            split_bf16,
            split_weights,
            subsets,
            natural_routes,
            masks,
            all_q3_masks,
        )
        natural_control_hidden = (
            split_teacher.float()
            + (natural_control_routed.float() - split_natural_bf16.float())
        ).to(split_teacher.dtype)
        natural_control = evaluate_hidden(
            natural_control_hidden,
            reference,
            norm_weight,
            lm_head,
            args.candidate_batch,
            args.bootstrap_resamples,
            args.seed + split_index,
        )
        if (
            max(natural_control["raw"]["teacher_to_candidate_kl"]) != 0.0
            or not all(natural_control["raw"]["top1_agreement"])
            or natural_control["aggregate"]["cross_entropy_delta"] != 0.0
        ):
            raise RuntimeError(f"natural BF16 exact control failed on {split}")

        rate_distortion: dict[str, list[dict[str, Any]]] = {
            "natural_route": [],
            "joint_top32": [],
        }
        for family, solution, best_route, best_mask, family_shortlist in (
            (
                "natural_route",
                natural_solution,
                torch.zeros_like(natural_best_mask),
                natural_best_mask,
                natural_routes.unsqueeze(1),
            ),
            ("joint_top32", joint_solution, joint_route, joint_mask, shortlist),
        ):
            for fraction_index, requested_fraction in enumerate(RATE_FRACTIONS):
                cost, schedule = best_schedule_within_fraction(
                    solution, args.tokens_per_split, requested_fraction
                )
                routes, selected_masks = choices_for_schedule(
                    schedule, best_route, best_mask, family_shortlist
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
                direct_kl = record["aggregate"]["teacher_to_candidate_kl"]
                dp_kl = float(solution.exact_cost_curve[cost] / args.tokens_per_split)
                if abs(direct_kl - dp_kl) > 1e-6:
                    raise RuntimeError("DP and direct KL disagree")
                record |= {
                    "requested_upgrade_fraction": requested_fraction,
                    "total_upgrade_count": cost,
                    "actual_upgrade_fraction": cost
                    / (args.tokens_per_split * 6),
                    "average_active_bits": 3.0
                    + cost / (args.tokens_per_split * 6),
                    "dynamic_program_kl_mean": dp_kl,
                }
                rate_distortion[family].append(record)

        if natural_solution.per_token_cost is None or joint_solution.per_token_cost is None:
            raise RuntimeError("natural all-Q4 target must be reachable")
        natural_target_routes, natural_target_masks = choices_for_schedule(
            natural_solution.per_token_cost,
            torch.zeros_like(natural_best_mask),
            natural_best_mask,
            natural_routes.unsqueeze(1),
        )
        joint_target_routes, joint_target_masks = choices_for_schedule(
            joint_solution.per_token_cost,
            joint_route,
            joint_mask,
            shortlist,
        )
        target_records = {
            "natural_minimum_at_q4_quality": schedule_record(
                q3=split_q3,
                q4=split_q4,
                weights=split_weights,
                subsets=subsets,
                masks=masks,
                routes=natural_target_routes,
                mask_indices=natural_target_masks,
                teacher=split_teacher,
                natural_bf16=split_natural_bf16,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
            ),
            "joint_top32_minimum_at_q4_quality": schedule_record(
                q3=split_q3,
                q4=split_q4,
                weights=split_weights,
                subsets=subsets,
                masks=masks,
                routes=joint_target_routes,
                mask_indices=joint_target_masks,
                teacher=split_teacher,
                natural_bf16=split_natural_bf16,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + split_index,
            ),
        }
        selected_bf16_routed = routed_from_choices(
            split_bf16,
            split_bf16,
            split_weights,
            subsets,
            joint_target_routes,
            masks,
            torch.zeros_like(joint_target_masks),
        )
        selected_bf16_hidden = (
            split_teacher.float()
            + (selected_bf16_routed.float() - split_natural_bf16.float())
        ).to(split_teacher.dtype)
        target_records["joint_selected_routes_in_bf16"] = evaluate_hidden(
            selected_bf16_hidden,
            reference,
            norm_weight,
            lm_head,
            args.candidate_batch,
            args.bootstrap_resamples,
            args.seed + split_index,
        )

        bootstrap_gate = gate_bootstrap(
            natural_q3=natural_q3_damage,
            natural_q4=natural_q4_damage,
            alternative_q3=best_alternative_q3,
            natural_best=natural_best,
            joint_best=joint_best,
            blocks=reference.blocks,
            resamples=args.bootstrap_resamples,
            seed=args.seed + split_index,
        )
        same_or_better_q4 = best_alternative_q3 <= natural_q4_damage * TARGET_MULTIPLIER
        results[split] = {
            "teacher_reference": {
                "token_ids": token_ids[split].tolist(),
                "true_token_nll": nullable(reference.true_token_nll),
                "sequence_blocks": [list(block) for block in reference.blocks],
            },
            "controls": {"natural_bf16_exact": natural_control},
            "stage_a": {
                "selection": "exact full-vocabulary KL over every all-Q3 route",
                "all_q3_kl": all_q3_damage.tolist(),
                "natural_route_index": natural_index,
                "best_alternative_route_index": best_alternative_routes.tolist(),
                "best_alternative_all_q3_kl": best_alternative_q3.tolist(),
                "shortlist_route_indices": shortlist.tolist(),
                "natural_route_forced": natural_forced.tolist(),
                "natural_route_forced_fraction": float(natural_forced.double().mean()),
                "wall_seconds": stage_a_seconds,
            },
            "stage_c": {
                "selection": "exact full-vocabulary KL for 32 routes x 64 masks",
                "kl": stage_c_damage.tolist(),
                "shortlist_bf16_kl": shortlist_bf16_damage.tolist(),
                "natural_local_route_index": natural_local_by_token.tolist(),
                "joint_best_kl_by_exact_upgrade_count": joint_best.tolist(),
                "joint_best_local_route_by_exact_upgrade_count": joint_route.tolist(),
                "joint_best_mask_by_exact_upgrade_count": joint_mask.tolist(),
                "natural_best_kl_by_exact_upgrade_count": natural_best.tolist(),
                "natural_best_mask_by_exact_upgrade_count": natural_best_mask.tolist(),
                "wall_seconds": stage_c_seconds,
            },
            "baselines": baseline_records,
            "solutions": {
                "natural_route": solution_json(natural_solution),
                "joint_top32": solution_json(joint_solution),
                "upgrade_fraction_absolute_reduction": float(
                    natural_solution.upgrade_fraction - joint_solution.upgrade_fraction
                ),
                "upgrade_fraction_relative_reduction": float(
                    1.0
                    - joint_solution.upgrade_fraction
                    / natural_solution.upgrade_fraction
                ),
                "all_q3_alternative_mean_gap_closure": closure,
                "all_q3_alternative_matches_q4_within_1pct_token_fraction": float(
                    same_or_better_q4.double().mean().item()
                ),
            },
            "rate_distortion": rate_distortion,
            "target_schedules": target_records,
            "gate_bootstrap_95": bootstrap_gate,
            "wall_seconds": time.perf_counter() - split_started,
        }

    if args.stage == "smoke":
        verdict = "smoke_passed_not_adjudicated"
        shared_positive: list[str] = []
    else:
        shared_positive = [
            criterion
            for criterion in gates["validation"]["strong_criteria"]
            if gates["validation"]["strong_criteria"][criterion]
            and gates["test"]["strong_criteria"][criterion]
        ]
        both_axes_negative = all(
            gates[split]["route_axis_negative_gap_closure_lt_0_10"]
            and gates[split]["joint_axis_negative_upgrade_fraction_gt_0_25"]
            for split in ("validation", "test")
        )
        if shared_positive:
            verdict = "strong_positive"
        elif both_axes_negative:
            verdict = "screen_negative"
        else:
            verdict = "inconclusive"

    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report = {
        "schema_version": 1,
        "kind": "craft_moe_crcq_top32_oracle_screen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H1_CRCQ",
        "verdict": verdict,
        "shared_positive_criteria": shared_positive,
        "full_59136_search_eligible": bool(
            args.stage == "full" and gates["validation"]["any_strong_criterion"]
        ),
        "preregistration": str(
            (ROOT / "reports/craft_moe/H1_CRCQ_PREREGISTRATION.md").resolve()
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
            "window": {
                split: f"first {args.tokens_per_split} tokens" for split in args.splits
            },
            "block_size": BLOCK_SIZE,
        },
        "configuration": {
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
            "route_count": subsets.shape[0],
            "shortlist_size": args.shortlist_size,
            "mask_count": masks.shape[0],
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
            "top12_expert_ids": {
                split: top12_ids[split_slices[split]].tolist() for split in args.splits
            },
            "top12_router_weights": {
                split: top12_weights[split_slices[split]].tolist()
                for split in args.splits
            },
        },
        "gates": gates,
        "results": results,
        "controls": {
            "trace_reproducibility": trace_reproducibility,
            "natural_bf16_exact": "passed on every evaluated split",
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
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "top-32 exact-KL screen, not the unopened full 59,136 route-mask oracle",
            "late-layer local oracle; no exact layer-23 downstream intervention yet",
            "teacher-KL chooses each route and mask, so no deployable selector exists",
            "projected bit savings are not a measured packed-runtime speedup",
            "256-token existing windows are exploratory replication, not confirmation",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    for split in args.splits:
        print(f"gate[{split}]={gates[split]}", flush=True)


if __name__ == "__main__":
    main()
