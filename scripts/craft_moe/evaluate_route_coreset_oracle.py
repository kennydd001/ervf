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
from moe_lab.craft_moe.route_coreset import (
    CandidateFit,
    delta_patched_hidden,
    enumerate_best_fits,
    higher_quantile,
    minimum_k_from_kl,
    ranked_original_coefficients,
    reconstruct_routed,
    relative_routed_error,
    rescaled_rank1_coefficients,
    routed_target,
)
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.partial_forward import checkpoint_state_for_prefix
from moe_lab.reporting import ROOT, git_revision


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TRACE_RELATIVE = Path("data/traces/layer26_dynamic_precision_components.safetensors")
BLOCK_SIZE = 128
TRACE_TOKENS_PER_SPLIT = 1024
DEFAULT_TOKENS_PER_SPLIT = 256
DEFAULT_SEED = 20260810
KL_GATE = 0.001
FALSIFICATION_KL = 0.003


@dataclass
class TeacherReference:
    log_probs: torch.Tensor
    top1: torch.Tensor
    true_token_nll: torch.Tensor
    token_ids: torch.Tensor
    blocks: list[tuple[int, int]]


@dataclass
class EvaluationCandidate:
    method: str
    k: int
    coefficients: torch.Tensor
    subset_mask: torch.Tensor
    squared_error: torch.Tensor
    selection: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustive route-local sparse-coreset oracle for layer 26."
    )
    parser.add_argument(
        "--stage", choices=("smoke", "full"), default="full"
    )
    parser.add_argument(
        "--tokens-per-split", type=int, default=DEFAULT_TOKENS_PER_SPLIT
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--box-multiplier", type=float, default=2.0)
    parser.add_argument("--box-iterations", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/craft_moe/route_coreset_oracle.json"),
    )
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if not 1 <= args.tokens_per_split <= TRACE_TOKENS_PER_SPLIT:
        raise ValueError(
            f"tokens-per-split must be in [1, {TRACE_TOKENS_PER_SPLIT}]"
        )
    if args.stage == "smoke" and (
        args.tokens_per_split > 32 or args.splits != ("validation",)
    ):
        raise ValueError("smoke must use at most 32 validation tokens and no test tokens")
    if args.stage == "full" and (
        args.tokens_per_split != DEFAULT_TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
    ):
        raise ValueError(
            "the preregistered full run is fixed at 256 validation + 256 test tokens"
        )
    if args.max_k != 5:
        raise ValueError("the preregistered search requires max-k=5")
    if args.box_multiplier != 2.0 or args.box_iterations != 256:
        raise ValueError("the preregistered bounded solver is fixed at 2x and 256 steps")
    if args.eval_batch < 1 or args.bootstrap_resamples < 1:
        raise ValueError("eval-batch and bootstrap-resamples must be positive")
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output = output.resolve()
    allowed = (ROOT / "reports").resolve()
    if allowed not in output.parents:
        raise ValueError("output-json must be inside reports/")
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing experiment result: {output}"
        )
    args.output_json = output
    return args


def corpus_tokens(model_dir: Path, split: str, token_count: int) -> torch.Tensor:
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
    ids = tokenizer.encode(joined).ids[:token_count]
    if len(ids) != token_count:
        raise RuntimeError(f"{split} yielded only {len(ids)} tokens")
    return torch.tensor(ids, dtype=torch.long)


def sequence_blocks(token_count: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + BLOCK_SIZE, token_count))
        for start in range(0, token_count, BLOCK_SIZE)
    ]


def prediction_mask(token_count: int, blocks: list[tuple[int, int]]) -> torch.Tensor:
    mask = torch.ones(token_count, dtype=torch.bool)
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
    gpu_query = command_result(
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
        "nvidia_smi": gpu_query,
    }


@torch.inference_mode()
def selected_bf16_outputs(
    moe: LoadedMoELayer,
    flat_input: torch.Tensor,
    router_ids: torch.Tensor,
) -> torch.Tensor:
    inputs = flat_input.to(moe.device)
    ids = router_ids.to(moe.device).long()
    tokens, slots = ids.shape
    outputs = torch.empty(
        tokens,
        slots,
        inputs.shape[-1],
        dtype=inputs.dtype,
        device=moe.device,
    )
    touched = 0
    for expert_id, expert in enumerate(moe.experts):
        positions = (ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            touched += 1
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            outputs[token_indices, slot_indices] = moe.expert_forward(
                inputs[token_indices], expert
            )
        if expert_id % 8 == 7:
            print(f"bf16_experts={expert_id + 1}/64", flush=True)
    if not torch.isfinite(outputs.float()).all():
        raise RuntimeError("non-finite selected BF16 expert output")
    print(f"experts_touched={touched}/64", flush=True)
    return outputs.cpu()


def regression_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    error = candidate.float() - reference.float()
    mse = error.square().mean()
    reference_rms = reference.float().square().mean().sqrt().clamp_min(1e-30)
    return {
        "nrmse": float(mse.sqrt().item() / reference_rms.item()),
        "rmse": float(mse.sqrt().item()),
        "mean_absolute_error": float(error.abs().mean().item()),
        "maximum_absolute_error": float(error.abs().max().item()),
    }


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
    vocabulary = lm_head.shape[0]
    log_probs = torch.empty(tokens, vocabulary, dtype=torch.float32)
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


def _block_sums_counts(
    values: np.ndarray, blocks: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    sums = []
    counts = []
    for start, stop in blocks:
        selected = values[start:stop]
        finite = np.isfinite(selected)
        sums.append(float(selected[finite].sum()))
        counts.append(int(finite.sum()))
    return np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.int64)


def _percentile_summary(estimates: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(estimates, (0.025, 0.975), method="linear")
    return {"low": float(low), "high": float(high)}


def sequence_block_bootstrap(
    *,
    kl: torch.Tensor,
    top1: torch.Tensor,
    teacher_nll: torch.Tensor,
    candidate_nll: torch.Tensor,
    routed_error: torch.Tensor,
    blocks: list[tuple[int, int]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    series = {
        "teacher_to_candidate_kl": kl.double().numpy(),
        "top1_agreement": top1.double().numpy(),
        "routed_relative_l2": routed_error.double().numpy(),
        "cross_entropy_delta": (
            candidate_nll.double() - teacher_nll.double()
        ).numpy(),
    }
    block_stats = {
        name: _block_sums_counts(values, blocks) for name, values in series.items()
    }
    teacher_stats = _block_sums_counts(teacher_nll.double().numpy(), blocks)
    candidate_stats = _block_sums_counts(candidate_nll.double().numpy(), blocks)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(resamples, len(blocks)))
    intervals: dict[str, Any] = {}
    for name, (sums, counts) in block_stats.items():
        estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        intervals[name] = _percentile_summary(estimates)
    teacher_sums, teacher_counts = teacher_stats
    candidate_sums, candidate_counts = candidate_stats
    sampled_teacher = teacher_sums[sampled].sum(axis=1) / teacher_counts[
        sampled
    ].sum(axis=1)
    sampled_candidate = candidate_sums[sampled].sum(axis=1) / candidate_counts[
        sampled
    ].sum(axis=1)
    relative = (sampled_candidate - sampled_teacher) / sampled_teacher
    intervals["relative_cross_entropy_delta"] = _percentile_summary(relative)
    return {
        "method": "paired sequence-block percentile bootstrap",
        "confidence": 0.95,
        "resamples": resamples,
        "seed": seed,
        "sequence_blocks": len(blocks),
        "intervals": intervals,
    }


def nullable_float_list(values: torch.Tensor) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


@torch.inference_mode()
def evaluate_candidate(
    candidate_hidden: torch.Tensor,
    reference: TeacherReference,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    routed_error: torch.Tensor,
    batch_size: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tokens = candidate_hidden.shape[0]
    kl = torch.empty(tokens, dtype=torch.float32)
    agreement = torch.empty(tokens, dtype=torch.bool)
    candidate_nll = torch.full((tokens,), float("nan"), dtype=torch.float32)
    valid = prediction_mask(tokens, reference.blocks)
    for start in range(0, tokens, batch_size):
        stop = min(start + batch_size, tokens)
        states = candidate_hidden[start:stop].to(lm_head.device)
        logits = F.linear(rmsnorm(states, norm_weight), lm_head).float()
        candidate_log_probs = F.log_softmax(logits, dim=-1)
        teacher_log_probs = reference.log_probs[start:stop].to(lm_head.device)
        divergence = (
            teacher_log_probs.exp()
            * (teacher_log_probs - candidate_log_probs)
        ).sum(dim=-1)
        kl[start:stop] = divergence.clamp_min(0.0).cpu()
        agreement[start:stop] = (
            logits.argmax(dim=-1).cpu() == reference.top1[start:stop]
        )
        local_valid = valid[start:stop]
        if local_valid.any():
            local_positions = local_valid.nonzero(as_tuple=False).squeeze(1)
            global_positions = start + local_positions
            candidate_nll[global_positions] = -candidate_log_probs[
                local_positions,
                reference.token_ids[global_positions + 1].to(lm_head.device),
            ].cpu()
    prediction = valid.nonzero(as_tuple=False).squeeze(1)
    teacher_ce = reference.true_token_nll[prediction].double().mean()
    candidate_ce = candidate_nll[prediction].double().mean()
    ce_delta = candidate_ce - teacher_ce
    aggregate = {
        "teacher_to_candidate_kl": float(kl.double().mean().item()),
        "top1_agreement": float(agreement.double().mean().item()),
        "teacher_cross_entropy": float(teacher_ce.item()),
        "candidate_cross_entropy": float(candidate_ce.item()),
        "cross_entropy_delta": float(ce_delta.item()),
        "relative_cross_entropy_delta": float((ce_delta / teacher_ce).item()),
        "routed_relative_l2_mean": float(routed_error.double().mean().item()),
        "routed_relative_l2_median": float(routed_error.double().median().item()),
        "routed_relative_l2_p95_higher": higher_quantile(routed_error, 0.95),
    }
    bootstrap = sequence_block_bootstrap(
        kl=kl,
        top1=agreement,
        teacher_nll=reference.true_token_nll,
        candidate_nll=candidate_nll,
        routed_error=routed_error,
        blocks=reference.blocks,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    raw = {
        "teacher_to_candidate_kl": kl.tolist(),
        "top1_agreement": [bool(value) for value in agreement.tolist()],
        "candidate_true_token_nll": nullable_float_list(candidate_nll),
        "routed_relative_l2": routed_error.tolist(),
    }
    return {"aggregate": aggregate, "bootstrap_95": bootstrap}, raw


def bitmask_rows(mask: torch.Tensor) -> list[int]:
    powers = 1 << torch.arange(mask.shape[1], dtype=torch.long)
    return (mask.long() * powers).sum(dim=1).tolist()


def make_candidate(
    method: str,
    k: int,
    coefficients: torch.Tensor,
    outputs: torch.Tensor,
    target: torch.Tensor,
    selection: str,
    fit: CandidateFit | None = None,
) -> EvaluationCandidate:
    coefficients = coefficients.detach().cpu().double()
    if fit is None:
        subset_mask = coefficients != 0
        reconstructed = reconstruct_routed(outputs, coefficients)
        squared_error = (reconstructed - target).square().sum(dim=-1)
    else:
        subset_mask = fit.subset_mask.detach().cpu()
        squared_error = fit.squared_error.detach().cpu()
    return EvaluationCandidate(
        method=method,
        k=k,
        coefficients=coefficients,
        subset_mask=subset_mask,
        squared_error=squared_error,
        selection=selection,
    )


def candidate_grid(
    outputs: torch.Tensor,
    weights: torch.Tensor,
    fits: dict[str, dict[int, CandidateFit]],
) -> list[EvaluationCandidate]:
    target = routed_target(outputs, weights)
    candidates = [
        make_candidate(
            "exact_top6",
            6,
            weights,
            outputs,
            target,
            "original six experts and original unnormalized router weights",
        )
    ]
    for method in ("original_drop", "free_ls", "nnls", "bounded_2x"):
        for k in range(1, 6):
            fit = fits[method][k]
            candidates.append(
                make_candidate(
                    method,
                    k,
                    fit.coefficients,
                    outputs,
                    target,
                    "minimum routed-output L2 over all selected-expert subsets",
                    fit,
                )
            )
    for k in range(1, 6):
        coefficients = ranked_original_coefficients(weights, k)
        candidates.append(
            make_candidate(
                "ranked_original",
                k,
                coefficients,
                outputs,
                target,
                "top-k by original router weight; coefficients unchanged",
            )
        )
    candidates.append(
        make_candidate(
            "rank1_rescaled",
            1,
            rescaled_rank1_coefficients(outputs, weights),
            outputs,
            target,
            "highest router-weight expert with optimal non-negative scalar",
        )
    )
    return candidates


def split_gate(method_rows: dict[str, Any]) -> dict[str, Any]:
    kl_by_k = {
        k: torch.tensor(
            method_rows["nnls"][str(k)]["raw"]["teacher_to_candidate_kl"],
            dtype=torch.float64,
        )
        for k in range(1, 6)
    }
    minimum = minimum_k_from_kl(kl_by_k, KL_GATE)
    median_higher = higher_quantile(minimum, 0.50)
    p95_higher = higher_quantile(minimum, 0.95)
    fraction_k5_bad = float((kl_by_k[5] > FALSIFICATION_KL).double().mean().item())
    primary_criterion = median_higher <= 3 or p95_higher <= 4
    falsification = fraction_k5_bad > 0.25
    if falsification:
        verdict = "falsified"
    elif primary_criterion:
        verdict = "oracle_positive"
    else:
        verdict = "inconclusive_negative"
    return {
        "verdict": verdict,
        "minimum_k_at_kl_le_0_001": minimum.tolist(),
        "minimum_k_distribution": {
            str(k): int((minimum == k).sum().item()) for k in range(1, 7)
        },
        "minimum_k_median_higher_empirical": median_higher,
        "minimum_k_p95_higher_empirical": p95_higher,
        "primary_criterion_passed": primary_criterion,
        "falsification_fraction_k5_kl_gt_0_003": fraction_k5_bad,
        "falsification_triggered": falsification,
    }


def trace_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


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
        raise RuntimeError("the route-coreset oracle requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    trace_path = ROOT / TRACE_RELATIVE
    if not model_dir.is_dir() or not trace_path.is_file():
        raise FileNotFoundError("pinned model or layer-26 component trace is missing")

    phase = time.perf_counter()
    artifact_hash = sha256_file(trace_path)
    timings["input_sha256_seconds"] = time.perf_counter() - phase
    initial_hardware = hardware_state()
    repository = git_state()
    metadata = trace_metadata(trace_path)

    phase = time.perf_counter()
    all_components = load_file(trace_path, device="cpu")
    trace_indices: dict[str, list[int]] = {}
    indices = []
    for split in args.splits:
        base = 0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        selected = list(range(base, base + args.tokens_per_split))
        trace_indices[split] = selected
        indices.extend(selected)
    index = torch.tensor(indices, dtype=torch.long)
    components = {key: value.index_select(0, index) for key, value in all_components.items()}
    del all_components
    token_ids = {
        split: corpus_tokens(model_dir, split, args.tokens_per_split)
        for split in args.splits
    }
    timings["load_inputs_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    moe = load_moe_layer(model_dir, 26, device)
    outputs = selected_bf16_outputs(
        moe, components["moe_input"], components["router_ids"]
    )
    del moe
    gc.collect()
    torch.cuda.empty_cache()
    timings["selected_bf16_expert_forward_seconds"] = time.perf_counter() - phase

    weights = components["router_weights"].double()
    original_routed = routed_target(outputs, weights)
    manual_hidden = components["post_attention"] + (
        original_routed.to(components["shared"].dtype) + components["shared"]
    )
    manual_vs_official = regression_summary(components["teacher"], manual_hidden)
    router_weight_summary = {
        "minimum": float(weights.min().item()),
        "maximum": float(weights.max().item()),
        "sum_per_token_mean": float(weights.sum(dim=1).mean().item()),
        "sum_per_token_minimum": float(weights.sum(dim=1).min().item()),
        "sum_per_token_maximum": float(weights.sum(dim=1).max().item()),
        "renormalized": False,
    }

    phase = time.perf_counter()
    fits = enumerate_best_fits(
        outputs,
        weights,
        max_k=args.max_k,
        box_multiplier=args.box_multiplier,
        box_iterations=args.box_iterations,
    )
    candidates = candidate_grid(outputs, weights, fits)
    timings["exhaustive_fit_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    timings["load_final_projection_seconds"] = time.perf_counter() - phase

    split_slices: dict[str, slice] = {}
    offset = 0
    references: dict[str, TeacherReference] = {}
    for split_index, split in enumerate(args.splits):
        split_slices[split] = slice(offset, offset + args.tokens_per_split)
        references[split] = make_teacher_reference(
            components["teacher"][split_slices[split]],
            token_ids[split],
            sequence_blocks(args.tokens_per_split),
            norm_weight,
            lm_head,
            args.eval_batch,
        )
        offset += args.tokens_per_split

    results: dict[str, Any] = {
        split: {
            "teacher_reference": {
                "true_token_nll": nullable_float_list(
                    references[split].true_token_nll
                ),
                "sequence_blocks": [list(block) for block in references[split].blocks],
                "token_ids": token_ids[split].tolist(),
            },
            "methods": {},
        }
        for split in args.splits
    }
    phase = time.perf_counter()
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_started = time.perf_counter()
        candidate_routed = reconstruct_routed(outputs, candidate.coefficients)
        candidate_hidden = delta_patched_hidden(
            components["teacher"], original_routed, candidate_routed
        )
        routed_error_all = relative_routed_error(outputs, weights, candidate.coefficients)
        for split_index, split in enumerate(args.splits):
            selected = split_slices[split]
            evaluation, raw = evaluate_candidate(
                candidate_hidden[selected],
                references[split],
                norm_weight,
                lm_head,
                routed_error_all[selected],
                args.eval_batch,
                args.bootstrap_resamples,
                args.seed + split_index,
            )
            raw |= {
                "subset_bitmask": bitmask_rows(candidate.subset_mask[selected]),
                "coefficients": candidate.coefficients[selected].tolist(),
                "squared_output_error": candidate.squared_error[selected].tolist(),
            }
            nonzero = (candidate.coefficients[selected].abs() > 1e-12).sum(dim=1)
            evaluation |= {
                "selection": candidate.selection,
                "coefficient_nonzero_mean": float(nonzero.double().mean().item()),
                "subset_cardinality_distribution": {
                    str(k): int((candidate.subset_mask[selected].sum(dim=1) == k).sum())
                    for k in range(7)
                },
                "raw": raw,
            }
            results[split]["methods"].setdefault(candidate.method, {})[
                str(candidate.k)
            ] = evaluation
        elapsed = time.perf_counter() - candidate_started
        for split in args.splits:
            results[split]["methods"][candidate.method][str(candidate.k)][
                "evaluation_wall_seconds_all_splits"
            ] = elapsed
        print(
            f"candidate={candidate_index:02d}/{len(candidates)} "
            f"method={candidate.method} k={candidate.k} seconds={elapsed:.2f}",
            flush=True,
        )
        if candidate.method == "exact_top6":
            for split in args.splits:
                control = results[split]["methods"]["exact_top6"]["6"]
                control_raw = control["raw"]
                if (
                    max(control_raw["teacher_to_candidate_kl"]) != 0.0
                    or not all(control_raw["top1_agreement"])
                    or control["aggregate"]["cross_entropy_delta"] != 0.0
                ):
                    raise RuntimeError(f"exact top-6 control failed on {split}")
    timings["candidate_evaluation_seconds"] = time.perf_counter() - phase

    gates = {split: split_gate(results[split]["methods"]) for split in args.splits}
    if args.stage == "smoke":
        overall_verdict = "smoke_passed_not_adjudicated"
    else:
        overall_verdict = gates["validation"]["verdict"]
    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report = {
        "schema_version": 1,
        "kind": "craft_moe_route_coreset_oracle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H7_ROUTE_CORESET",
        "verdict": overall_verdict,
        "preregistration": str(
            (ROOT / "reports/craft_moe/H7_ROUTE_CORESET_PREREGISTRATION.md").resolve()
        ),
        "hypothesis": (
            "The original top-6 routed sum can often be reconstructed by at most "
            "three already-selected expert outputs with fitted non-negative coefficients."
        ),
        "gate_definition": {
            "primary_split": "validation",
            "primary_method": "nnls",
            "local_kl_threshold": KL_GATE,
            "positive": "higher empirical median minimum k <=3 OR p95 <=4",
            "falsification": "fraction with NNLS k=5 KL >0.003 exceeds 0.25",
            "falsification_overrides_positive": True,
        },
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
            "stage": args.stage,
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
            "max_k": args.max_k,
            "box_multiplier": args.box_multiplier,
            "box_iterations": args.box_iterations,
            "eval_batch": args.eval_batch,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "candidate_count": len(candidates),
            "selection_objective": "per-token routed-output squared L2 only",
            "counterfactual_patch": (
                "BF16(official_teacher + candidate_routed - manual_top6_routed)"
            ),
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
            "input_trace": {
                "path": str(trace_path.resolve()),
                "sha256": artifact_hash,
                "bytes": trace_path.stat().st_size,
                "metadata": metadata,
                "indices": trace_indices,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "controls": {
            "exact_top6": "required and passed on every evaluated split",
            "manual_top6_reconstruction_vs_official_teacher": manual_vs_official,
            "router_weights": router_weight_summary,
        },
        "timings": timings,
        "gates": gates,
        "results": results,
        "limitations": [
            "late-layer local oracle; no downstream layer-23 intervention yet",
            "coefficients are oracle fits and have no cheap deployable predictor yet",
            "projected expert-count reduction is not a measured packed-runtime speedup",
            "256-token exploratory windows are not confirmatory evidence",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={overall_verdict}", flush=True)
    for split in args.splits:
        print(f"gate[{split}]={gates[split]}", flush=True)


if __name__ == "__main__":
    main()
