#!/usr/bin/env python3
"""TierFlow persistent-set functional-span oracle.

The experiment reconstructs P4D teacher states from local weights, fits a
per-token nonnegative unit-sum oracle on three sentinel layers, and propagates
each one-layer intervention through the untouched downstream model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import transformers
from safetensors.numpy import load_file as load_numpy
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_checkpoint_tensors,
    load_qwen_decoder_layer,
    load_token_embeddings,
)


MODEL = ROOT / "models" / "qwen3-30b-a3b-base"
R = ROOT / "reports" / "streamq5_moe"
RUNS = ROOT / "reports" / "runs" / "streamq5_moe"
PREREG = R / "TIERFLOW_PERSISTENT_SET_FUNCTIONAL_SPAN_PREREGISTRATION_2026-08-12.md"
INPUT_LOCK = R / "p4d_route_input_lock.json"
INPUT_IDS = RUNS / "p4d_fresh_route_input_ids.safetensors"
CAPTURE = R / "p4d_route_capture_result.json"
ROUTE_DIR = RUNS / "p4d_routes"
F0_VALIDATION = R / "tierflow_f0_validation.json"
F0_TEST = R / "tierflow_f0_result.json"
VALIDATION_OUT = R / "tierflow_persistent_set_functional_span_validation.json"
TEST_OUT = R / "tierflow_persistent_set_functional_span_test.json"
REPORT = R / "TIERFLOW_PERSISTENT_SET_FUNCTIONAL_SPAN_REPORT_2026-08-12.md"

EXPECTED_INPUT = "32838e94887f8572445159925e815f5353f55a20a954f9adc2f8cef48427af08"
EXPECTED_CAPTURE = "7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
PARTITIONS = {"validation": (512, 768), "test": (768, 1024)}
SENTINELS = (0, 24, 47)
LAYERS = 48
EXPERTS = 128
TOP_K = 8
TOKENS = 1024
HIDDEN = 2048
EXPERT_BATCH = 8
NEGATIVE_TOL = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


@torch.no_grad()
def quantize_groupwise_(value: torch.Tensor, bits: int, row_batch: int = 512) -> None:
    if value.ndim != 2 or value.shape[1] % 128:
        raise ValueError(f"matrix is not group-128 compatible: {tuple(value.shape)}")
    qmax = 15 if bits == 5 else 127 if bits == 8 else None
    if qmax is None:
        raise ValueError(bits)
    rows, columns = value.shape
    groups = columns // 128
    for start in range(0, rows, row_batch):
        end = min(rows, start + row_batch)
        work = value[start:end].float().reshape(end - start, groups, 128)
        maximum = work.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
        codes = torch.round(work / scale).clamp(-qmax, qmax)
        stored_scale = scale.to(torch.bfloat16).float()
        value[start:end].copy_((codes * stored_scale).reshape(end - start, columns).to(value.dtype))


def trunk_parameters(layer: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in layer.named_parameters()
        if parameter.ndim == 2 and ".experts." not in name
    ]


@torch.no_grad()
def quantize_candidate_(layer: torch.nn.Module) -> None:
    for _name, parameter in trunk_parameters(layer):
        quantize_groupwise_(parameter, 8)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for start in range(0, len(layer.mlp.experts), EXPERT_BATCH):
            experts = layer.mlp.experts[start : start + EXPERT_BATCH]
            weights = [getattr(expert, projection).weight for expert in experts]
            shape = weights[0].shape
            work = torch.stack(weights).reshape(-1, shape[1])
            quantize_groupwise_(work, 5, row_batch=work.shape[0])
            work = work.reshape(len(weights), *shape)
            for offset, weight in enumerate(weights):
                weight.copy_(work[offset])


def artifact_audit() -> dict[str, Any]:
    required = [MODEL / "model.safetensors.index.json", INPUT_LOCK, INPUT_IDS, CAPTURE, PREREG]
    missing = [str(path.relative_to(ROOT)).replace("\\", "/") for path in required if not path.is_file()]
    if missing:
        return {"pass": False, "missing": missing}
    index = json.loads((MODEL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (MODEL / name).is_file()]
    empty_shards = [name for name in shards if (MODEL / name).is_file() and (MODEL / name).stat().st_size == 0]
    route_missing = [
        str((ROUTE_DIR / f"layer_{layer:02d}.safetensors").relative_to(ROOT)).replace("\\", "/")
        for layer in range(LAYERS)
        if not (ROUTE_DIR / f"layer_{layer:02d}.safetensors").is_file()
    ]
    hashes = {
        "input_ids": sha256(INPUT_IDS),
        "capture": sha256(CAPTURE),
        "model_index": sha256(MODEL / "model.safetensors.index.json"),
    }
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    hash_ok = (
        hashes["input_ids"] == EXPECTED_INPUT
        and hashes["capture"] == EXPECTED_CAPTURE
        and hashes["model_index"] == lock["model_index_sha256"]
    )
    return {
        "pass": not missing_shards and not empty_shards and not route_missing and hash_ok,
        "missing": missing,
        "model_shards_indexed": len(shards),
        "missing_shards": missing_shards,
        "empty_shards": empty_shards,
        "missing_routes": route_missing,
        "hashes": hashes,
        "hash_contract_pass": hash_ok,
        "cuda_available": torch.cuda.is_available(),
        "transformers_version": transformers.__version__,
    }


def load_routes() -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    raw: dict[int, dict[str, np.ndarray]] = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        if sha256(path) != capture["manifests"][str(layer)]["artifact_sha256"]:
            raise RuntimeError(f"route hash mismatch at layer {layer}")
        tensors = load_numpy(path)
        raw[layer] = {}
        for domain in DOMAINS:
            routes = tensors[f"{domain}_router_ids"].astype(np.int64, copy=False)
            if routes.shape != (TOKENS, TOP_K):
                raise RuntimeError(f"route shape mismatch {layer}:{domain}")
            raw[layer][domain] = routes
    return raw, capture


def future_tables(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = observed.shape[0]
    present = np.zeros((length, EXPERTS), dtype=np.uint8)
    present[np.arange(length)[:, None], observed] = 1
    suffix = np.zeros((length + 1, EXPERTS), dtype=np.int32)
    for token in range(length - 1, -1, -1):
        suffix[token] = suffix[token + 1] + present[token]
    next_use = np.full((length, EXPERTS), length + 1, dtype=np.int32)
    next_position = np.full(EXPERTS, length + 1, dtype=np.int32)
    for token in range(length - 1, -1, -1):
        next_use[token] = next_position
        next_position[observed[token]] = token
    return next_use, suffix


def persistent_sequence(full: np.ndarray, start: int, end: int) -> dict[str, Any]:
    observed = full[start:end]
    next_use, suffix = future_tables(observed)
    state = set(int(x) for x in full[start - 1].tolist())
    previous = set(state)
    states: list[list[int]] = []
    baseline_new: list[int] = []
    oracle_new: list[int] = []
    overlaps: list[int] = []
    for token, row in enumerate(observed):
        requested = set(int(x) for x in row.tolist())
        baseline_new.append(len(requested - previous))
        missing = requested - state
        count = min(1, len(missing))

        def priority(expert: int) -> tuple[int, int, int]:
            return int(next_use[token, expert]), -int(suffix[token + 1, expert]), expert

        admissions = sorted(missing, key=priority)[:count]
        evictions = sorted(state - requested, key=priority, reverse=True)[:count]
        state.difference_update(evictions)
        state.update(admissions)
        if len(state) != TOP_K:
            raise AssertionError("persistent state cardinality changed")
        states.append(sorted(state))
        oracle_new.append(count)
        overlaps.append(len(state & requested))
        previous = requested
    return {
        "states": np.asarray(states, dtype=np.int16),
        "baseline_new": baseline_new,
        "oracle_new": oracle_new,
        "overlaps": overlaps,
    }


def build_persistent_sets(
    routes: dict[int, dict[str, np.ndarray]], phase: str
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, Any]]:
    start, end = PARTITIONS[phase]
    sets: dict[tuple[int, str], np.ndarray] = {}
    baseline: list[int] = []
    oracle: list[int] = []
    overlaps: list[int] = []
    for domain in DOMAINS:
        for layer in range(LAYERS):
            row = persistent_sequence(routes[layer][domain], start, end)
            if layer in SENTINELS:
                sets[(layer, domain)] = row["states"]
            baseline.extend(row["baseline_new"])
            oracle.extend(row["oracle_new"])
            overlaps.extend(row["overlaps"])
    baseline_total = sum(baseline)
    oracle_total = sum(oracle)
    metrics = {
        "transitions": len(overlaps),
        "baseline_new_loads": baseline_total,
        "oracle_new_loads": oracle_total,
        "critical_bytes_reduction_x": baseline_total / oracle_total,
        "worst_case_new_load_reduction_x": max(baseline) / max(oracle),
        "mean_route_overlap": sum(overlaps) / (len(overlaps) * TOP_K),
        "substitution_rate": 1.0 - sum(overlaps) / (len(overlaps) * TOP_K),
    }
    return sets, metrics


@torch.inference_mode()
def official_forward(
    layer: torch.nn.Module,
    rotary: torch.nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
    return_routes: bool,
) -> tuple[torch.Tensor, np.ndarray | None]:
    batch = hidden.to(device)
    positions = torch.arange(TOKENS, device=device).unsqueeze(0)
    embeddings = rotary(batch, positions)
    output = layer(
        batch,
        attention_mask=None,
        position_ids=positions,
        use_cache=False,
        output_attentions=False,
        output_router_logits=return_routes,
        cache_position=positions.squeeze(0),
        position_embeddings=embeddings,
    )
    routes = None
    if return_routes:
        probabilities = F.softmax(output[1], dim=-1, dtype=torch.float)
        routes = torch.topk(probabilities, TOP_K, dim=-1).indices.to(torch.int16).cpu().numpy()
    result = output[0].detach().cpu().contiguous()
    del batch, embeddings, output
    return result, routes


@torch.inference_mode()
def manual_sentinel(
    layer: torch.nn.Module,
    rotary: torch.nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    batch = hidden.to(device)
    positions = torch.arange(TOKENS, device=device).unsqueeze(0)
    embeddings = rotary(batch, positions)
    residual = batch
    normalized = layer.input_layernorm(batch)
    attention = layer.self_attn(
        hidden_states=normalized,
        attention_mask=None,
        position_ids=positions,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=positions.squeeze(0),
        position_embeddings=embeddings,
    )[0]
    post_attention = residual + attention
    mlp_input = layer.post_attention_layernorm(post_attention)
    natural_moe, router_logits = layer.mlp(mlp_input)
    natural = post_attention + natural_moe
    probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routes = torch.topk(probabilities, TOP_K, dim=-1).indices.to(torch.int16).cpu().numpy()
    return (
        natural.detach(),
        post_attention.detach(),
        mlp_input.detach(),
        natural_moe.detach(),
        routes,
    )


@torch.inference_mode()
def persistent_expert_outputs(
    layer: torch.nn.Module,
    mlp_input: torch.Tensor,
    persistent_ids: np.ndarray,
    start: int,
    end: int,
) -> torch.Tensor:
    selected_input = mlp_input[0, start:end]
    outputs = torch.empty(
        (end - start, TOP_K, HIDDEN), dtype=selected_input.dtype, device=selected_input.device
    )
    for expert in range(EXPERTS):
        locations = np.argwhere(persistent_ids == expert)
        if locations.size == 0:
            continue
        token_index = torch.as_tensor(locations[:, 0], dtype=torch.long, device=selected_input.device)
        slot_index = torch.as_tensor(locations[:, 1], dtype=torch.long, device=selected_input.device)
        value = layer.mlp.experts[expert](selected_input.index_select(0, token_index))
        outputs[token_index, slot_index] = value
    return outputs


def simplex_active_support_oracle(experts: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Exact support enumeration for nonnegative unit-sum least squares."""
    x = np.asarray(experts, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    count = x.shape[0]
    gram = np.einsum("nid,njd->nij", x, x, optimize=True)
    cross = np.einsum("nid,nd->ni", x, y, optimize=True)
    target_norm = np.einsum("nd,nd->n", y, y, optimize=True)
    best_objective = np.full(count, np.inf, dtype=np.float64)
    best_alpha = np.zeros((count, TOP_K), dtype=np.float64)
    best_mask = np.zeros(count, dtype=np.uint16)

    for mask in range(1, 1 << TOP_K):
        indices = np.asarray([index for index in range(TOP_K) if mask & (1 << index)], dtype=np.int64)
        width = len(indices)
        kkt = np.zeros((count, width + 1, width + 1), dtype=np.float64)
        kkt[:, :width, :width] = gram[:, indices[:, None], indices]
        kkt[:, :width, width] = 1.0
        kkt[:, width, :width] = 1.0
        rhs = np.zeros((count, width + 1), dtype=np.float64)
        rhs[:, :width] = cross[:, indices]
        rhs[:, width] = 1.0
        try:
            solution = np.linalg.solve(kkt, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            solution = np.stack(
                [np.linalg.lstsq(kkt[row], rhs[row], rcond=None)[0] for row in range(count)]
            )
        coefficients = solution[:, :width]
        feasible = np.isfinite(coefficients).all(axis=1) & (coefficients >= -NEGATIVE_TOL).all(axis=1)
        clipped = np.maximum(coefficients, 0.0)
        clipped /= np.maximum(clipped.sum(axis=1, keepdims=True), 1e-300)
        objective = target_norm.copy()
        objective -= 2.0 * np.einsum("ni,ni->n", cross[:, indices], clipped)
        objective += np.einsum(
            "ni,nij,nj->n", clipped, gram[:, indices[:, None], indices], clipped, optimize=True
        )
        improve = feasible & (objective < best_objective - 1e-12)
        if np.any(improve):
            rows = np.flatnonzero(improve)
            best_objective[rows] = objective[rows]
            best_alpha[rows] = 0.0
            best_alpha[np.ix_(rows, indices)] = clipped[rows]
            best_mask[rows] = mask

    if not np.isfinite(best_objective).all() or np.any(best_mask == 0):
        raise RuntimeError("simplex support enumeration failed")
    gradient = np.einsum("nij,nj->ni", gram, best_alpha) - cross
    violations = np.zeros(count, dtype=np.float64)
    for row in range(count):
        active = best_alpha[row] > 1e-9
        level = float(gradient[row, active].mean())
        active_error = float(np.max(np.abs(gradient[row, active] - level)))
        inactive_error = float(np.max(np.maximum(level - gradient[row, ~active], 0.0))) if np.any(~active) else 0.0
        scale = max(float(np.max(np.abs(gram[row]))), float(np.max(np.abs(cross[row]))), 1.0)
        violations[row] = max(active_error, inactive_error) / scale
    fitted = np.einsum("ni,nid->nd", best_alpha, x, optimize=True)
    return {
        "alpha": best_alpha,
        "mask": best_mask,
        "objective": best_objective,
        "kkt_violation": violations,
        "fitted": fitted,
    }


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def fit_sentinel_domain(
    layer: torch.nn.Module,
    mlp_input: torch.Tensor,
    natural_moe: torch.Tensor,
    post_attention: torch.Tensor,
    persistent_ids: np.ndarray,
    natural_ids: np.ndarray,
    start: int,
    end: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    expert_tensor = persistent_expert_outputs(layer, mlp_input, persistent_ids, start, end)
    target_tensor = natural_moe[0, start:end]
    expert_np = expert_tensor.float().cpu().numpy()
    target_np = target_tensor.float().cpu().numpy()
    solved = simplex_active_support_oracle(expert_np, target_np)
    fitted_bf16 = torch.from_numpy(solved["fitted"].astype(np.float32)).to(
        device=post_attention.device, dtype=torch.bfloat16
    )
    target_float = target_tensor.float()
    delta = fitted_bf16.float() - target_float
    relative = (
        torch.linalg.vector_norm(delta, dim=-1)
        / torch.linalg.vector_norm(target_float, dim=-1).clamp_min(1e-30)
    ).cpu().numpy()
    candidate = (post_attention + natural_moe).detach().clone()
    candidate[0, start:end] = post_attention[0, start:end] + fitted_bf16
    natural_sets = [set(int(x) for x in row) for row in natural_ids[start:end]]
    persistent_sets = [set(int(x) for x in row) for row in persistent_ids]
    overlap = [len(left & right) / TOP_K for left, right in zip(natural_sets, persistent_sets)]
    support = np.count_nonzero(solved["alpha"] > 1e-9, axis=1)
    row = {
        "relative_l2": relative.tolist(),
        "kkt_violation": solved["kkt_violation"].tolist(),
        "simplex_sum_error": np.abs(solved["alpha"].sum(axis=1) - 1.0).tolist(),
        "minimum_coefficient": solved["alpha"].min(axis=1).tolist(),
        "support_size": support.tolist(),
        "route_overlap": overlap,
        "persistent_ids": persistent_ids.astype(int).tolist(),
        "alpha": solved["alpha"].tolist(),
        "fitted_routed_sha256": array_sha(fitted_bf16.float().cpu().numpy()),
        "natural_routed_sha256": array_sha(target_float.cpu().numpy()),
    }
    del expert_tensor, target_tensor, expert_np, target_np, fitted_bf16, target_float, delta
    return candidate.detach().cpu().contiguous(), row


@torch.inference_mode()
def normalized_hidden(hidden: torch.Tensor, norm_weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    state = hidden.to(device).float()
    variance = state.pow(2).mean(dim=-1, keepdim=True)
    return (state * torch.rsqrt(variance + 1e-6)).to(torch.bfloat16) * norm_weight


@torch.inference_mode()
def final_metrics(
    baseline: dict[str, torch.Tensor],
    candidates: dict[int, dict[str, torch.Tensor]],
    inputs: dict[str, torch.Tensor],
    norm_weight: torch.Tensor,
    head: torch.Tensor,
    device: torch.device,
    start: int,
    end: int,
) -> tuple[dict[str, Any], dict[int, Any]]:
    baseline_rows: dict[str, Any] = {}
    candidate_rows: dict[int, Any] = {layer: {"domains": {}} for layer in SENTINELS}
    aggregate_base_ce: list[float] = []
    for domain in DOMAINS:
        positions = slice(start, end - 1)
        labels = inputs[domain][0, start + 1 : end].to(device)
        base_norm = normalized_hidden(baseline[domain][0, positions], norm_weight, device)
        base_logits = F.linear(base_norm, head).float()
        base_logp = F.log_softmax(base_logits, dim=-1)
        base_probability = base_logp.exp()
        base_ce = F.cross_entropy(base_logits, labels, reduction="none")
        base_top1 = base_logits.argmax(dim=-1)
        base_values = base_ce.cpu().numpy().tolist()
        baseline_rows[domain] = {
            "cross_entropy": base_values,
            "mean_cross_entropy": float(base_ce.mean()),
            "top1_sha256": array_sha(base_top1.cpu().numpy()),
        }
        aggregate_base_ce.extend(base_values)
        for sentinel in SENTINELS:
            cand_norm = normalized_hidden(candidates[sentinel][domain][0, positions], norm_weight, device)
            cand_logits = F.linear(cand_norm, head).float()
            cand_logp = F.log_softmax(cand_logits, dim=-1)
            cand_ce = F.cross_entropy(cand_logits, labels, reduction="none")
            cand_top1 = cand_logits.argmax(dim=-1)
            kl = (base_probability * (base_logp - cand_logp)).sum(dim=-1)
            agreement = cand_top1 == base_top1
            candidate_rows[sentinel]["domains"][domain] = {
                "candidate_cross_entropy": cand_ce.cpu().numpy().tolist(),
                "natural_to_candidate_kl": kl.cpu().numpy().tolist(),
                "top1_agreement": agreement.to(torch.uint8).cpu().numpy().tolist(),
                "mean_candidate_cross_entropy": float(cand_ce.mean()),
                "mean_kl": float(kl.mean()),
                "top1_agreement_rate": float(agreement.float().mean()),
            }
            del cand_norm, cand_logits, cand_logp, cand_ce, cand_top1, kl, agreement
        del labels, base_norm, base_logits, base_logp, base_probability, base_ce, base_top1

    baseline_mean = float(np.mean(aggregate_base_ce))
    for sentinel in SENTINELS:
        all_ce: list[float] = []
        all_kl: list[float] = []
        all_agreement: list[int] = []
        per_domain_relative: dict[str, float] = {}
        for domain in DOMAINS:
            row = candidate_rows[sentinel]["domains"][domain]
            all_ce.extend(row["candidate_cross_entropy"])
            all_kl.extend(row["natural_to_candidate_kl"])
            all_agreement.extend(row["top1_agreement"])
            base_mean = baseline_rows[domain]["mean_cross_entropy"]
            per_domain_relative[domain] = (row["mean_candidate_cross_entropy"] - base_mean) / base_mean
            row["relative_cross_entropy_regression"] = per_domain_relative[domain]
        candidate_mean = float(np.mean(all_ce))
        candidate_rows[sentinel]["aggregate"] = {
            "labels": len(all_ce),
            "baseline_cross_entropy": baseline_mean,
            "candidate_cross_entropy": candidate_mean,
            "relative_cross_entropy_regression": (candidate_mean - baseline_mean) / baseline_mean,
            "natural_to_candidate_kl": describe(all_kl),
            "top1_agreement": float(np.mean(all_agreement)),
            "per_domain_relative_cross_entropy": per_domain_relative,
        }
    return {"domains": baseline_rows, "aggregate_cross_entropy": baseline_mean}, candidate_rows


def summarize_oracle(raw_rows: dict[int, dict[str, Any]]) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for sentinel in SENTINELS:
        rel: list[float] = []
        kkt: list[float] = []
        sum_error: list[float] = []
        minima: list[float] = []
        support: list[float] = []
        overlap: list[float] = []
        for domain in DOMAINS:
            row = raw_rows[sentinel][domain]
            rel.extend(row["relative_l2"])
            kkt.extend(row["kkt_violation"])
            sum_error.extend(row["simplex_sum_error"])
            minima.extend(row["minimum_coefficient"])
            support.extend(row["support_size"])
            overlap.extend(row["route_overlap"])
        result[sentinel] = {
            "relative_l2": describe(rel),
            "kkt_violation": describe(kkt),
            "simplex_sum_error": describe(sum_error),
            "minimum_coefficient": describe(minima),
            "support_size": describe(support),
            "route_overlap": describe(overlap),
        }
    return result


def adjudicate(
    traffic: dict[str, Any],
    controls: dict[str, Any],
    oracle_summary: dict[int, Any],
    downstream: dict[int, Any],
) -> dict[str, bool]:
    gates: dict[str, bool] = {
        "traffic_reduction_at_least_4x": traffic["critical_bytes_reduction_x"] >= 4.0,
        "worst_case_new_load_reduction_at_least_8x": traffic["worst_case_new_load_reduction_x"] >= 8.0,
        "all_natural_routes_match_capture": controls["route_mismatch_count"] == 0,
        "manual_sentinel_natural_bitexact": controls["manual_natural_mismatch_values"] == 0,
        "all_finite": controls["all_finite"],
    }
    for sentinel in SENTINELS:
        oracle = oracle_summary[sentinel]
        quality = downstream[sentinel]["aggregate"]
        prefix = f"layer_{sentinel}"
        gates[f"{prefix}_mean_relative_l2_le_0_05"] = oracle["relative_l2"]["mean"] <= 0.05
        gates[f"{prefix}_p95_relative_l2_le_0_10"] = oracle["relative_l2"]["p95"] <= 0.10
        gates[f"{prefix}_mean_kl_le_0_001"] = quality["natural_to_candidate_kl"]["mean"] <= 0.001
        gates[f"{prefix}_relative_ce_le_0_01"] = quality["relative_cross_entropy_regression"] <= 0.01
        gates[f"{prefix}_top1_ge_0_99"] = quality["top1_agreement"] >= 0.99
        gates[f"{prefix}_every_domain_relative_ce_le_0_02"] = all(
            value <= 0.02 for value in quality["per_domain_relative_cross_entropy"].values()
        )
        gates[f"{prefix}_simplex_and_kkt"] = (
            oracle["minimum_coefficient"]["min"] if "min" in oracle["minimum_coefficient"] else oracle["minimum_coefficient"]["p50"]
        ) >= -NEGATIVE_TOL and oracle["simplex_sum_error"]["max"] <= 1e-9 and oracle["kkt_violation"]["max"] <= 1e-7
    gates["overall_pass"] = all(gates.values())
    return gates


def write_report(payload: dict[str, Any]) -> None:
    rows = []
    for sentinel in SENTINELS:
        oracle = payload["oracle_summary"][str(sentinel)]
        quality = payload["downstream"][str(sentinel)]["aggregate"]
        rows.append(
            f"| {sentinel} | {oracle['relative_l2']['mean']:.6f} | {oracle['relative_l2']['p95']:.6f} | "
            f"{quality['natural_to_candidate_kl']['mean']:.6f} | "
            f"{quality['relative_cross_entropy_regression']:.3%} | {quality['top1_agreement']:.3%} |"
        )
    gate_rows = "\n".join(
        f"| {name} | {'pass' if passed else 'fail'} |" for name, passed in payload["gates"].items()
    )
    test_text = (
        "Test was opened once after validation passed."
        if payload["phase"] == "test"
        else "Test remains closed." if not payload["gates"]["overall_pass"] else "Validation authorizes one test run."
    )
    report = f"""# TierFlow persistent-set functional-span oracle

## Verdict

**{payload['status']}**. {test_text}

| sentinel layer | mean routed rel-L2 | p95 rel-L2 | mean downstream KL | relative CE | top-1 agreement |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Traffic replication: `{payload['traffic']['critical_bytes_reduction_x']:.6f}x`
critical bytes and `{payload['traffic']['worst_case_new_load_reduction_x']:.1f}x`
worst-case new loads.

## Frozen gates

| gate | result |
|---|---|
{gate_rows}

## Claim boundary

This is a per-token non-causal coefficient oracle on three single-layer
interventions. It is not a trained TierFlow controller, full-48-layer
intervention, runtime, latency or deployment result. Validation/test inputs are
strictly disjoint but were previously used for P4D/TierFlow route analysis.

## Artifacts

- preregistration: `reports/streamq5_moe/TIERFLOW_PERSISTENT_SET_FUNCTIONAL_SPAN_PREREGISTRATION_2026-08-12.md`
- runner: `scripts/streamq5_moe/run_tierflow_persistent_set_functional_span.py`
- raw result: `{payload['output_artifact']}`
"""
    REPORT.write_text(report, encoding="utf-8")


def run(phase: str) -> dict[str, Any]:
    output_path = VALIDATION_OUT if phase == "validation" else TEST_OUT
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if phase == "test":
        if not VALIDATION_OUT.exists():
            raise RuntimeError("validation missing; test closed")
        validation = json.loads(VALIDATION_OUT.read_text(encoding="utf-8"))
        if validation["status"] != "validation_pass_test_authorized":
            raise RuntimeError("validation failed; test closed")
        if validation["inputs"]["preregistration_sha256"] != sha256(PREREG):
            raise RuntimeError("preregistration changed after validation")

    audit = artifact_audit()
    if not audit["pass"]:
        payload = {
            "kind": "tierflow_persistent_set_functional_span",
            "phase": phase,
            "status": "blocked_artifact",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_audit": audit,
            "output_artifact": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    if transformers.__version__ != "4.51.3" or not torch.cuda.is_available():
        raise RuntimeError("pinned Transformers 4.51.3 and CUDA required")

    started = time.perf_counter()
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()
    routes, capture = load_routes()
    persistent, traffic = build_persistent_sets(routes, phase)
    start, end = PARTITIONS[phase]
    inputs = load_file(INPUT_IDS)
    if set(inputs) != set(DOMAINS):
        raise RuntimeError("P4D input domain mismatch")

    device = torch.device("cuda")
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = "sdpa"
    if (config.num_hidden_layers, config.num_experts, config.num_experts_per_tok) != (LAYERS, EXPERTS, TOP_K):
        raise RuntimeError("unexpected Qwen configuration")
    weight_map = checkpoint_weight_map(MODEL)
    baseline: dict[str, torch.Tensor] = {}
    for domain in DOMAINS:
        hidden = load_token_embeddings(MODEL, inputs[domain], device, weight_map)
        quantize_groupwise_(hidden.reshape(-1, HIDDEN), 8)
        baseline[domain] = hidden.cpu().contiguous()
    candidates: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in SENTINELS}
    oracle_raw: dict[int, dict[str, Any]] = {layer: {} for layer in SENTINELS}
    route_mismatches = 0
    manual_mismatch_values = 0
    layer_timings: list[dict[str, Any]] = []

    for layer_index in range(LAYERS):
        layer_started = time.perf_counter()
        layer = load_qwen_decoder_layer(MODEL, config, layer_index, device, weight_map)
        quantize_candidate_(layer)
        rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
        for domain in DOMAINS:
            if layer_index in SENTINELS:
                natural_gpu, post_attention, mlp_input, natural_moe, observed_routes = manual_sentinel(
                    layer, rotary, baseline[domain], device
                )
                official, official_routes = official_forward(layer, rotary, baseline[domain], device, True)
                manual_cpu = natural_gpu.cpu().contiguous()
                manual_mismatch_values += int((manual_cpu != official).sum())
                baseline[domain] = official
                route_mismatches += int(np.count_nonzero(observed_routes != routes[layer_index][domain]))
                route_mismatches += int(np.count_nonzero(official_routes != routes[layer_index][domain]))
                candidate, raw_row = fit_sentinel_domain(
                    layer,
                    mlp_input,
                    natural_moe,
                    post_attention,
                    persistent[(layer_index, domain)],
                    observed_routes,
                    start,
                    end,
                )
                candidates[layer_index][domain] = candidate
                oracle_raw[layer_index][domain] = raw_row
                del natural_gpu, post_attention, mlp_input, natural_moe, manual_cpu, official, official_routes
            else:
                updated, observed_routes = official_forward(layer, rotary, baseline[domain], device, True)
                baseline[domain] = updated
                route_mismatches += int(np.count_nonzero(observed_routes != routes[layer_index][domain]))

            for sentinel in SENTINELS:
                if sentinel < layer_index:
                    candidates[sentinel][domain], _ = official_forward(
                        layer, rotary, candidates[sentinel][domain], device, False
                    )

        layer_timings.append({"layer": layer_index, "seconds": time.perf_counter() - layer_started})
        print(json.dumps({"layer": layer_index, "seconds": layer_timings[-1]["seconds"], "route_mismatches": route_mismatches}), flush=True)
        del layer, rotary
        gc.collect()
        torch.cuda.empty_cache()

    final = load_checkpoint_tensors(MODEL, ["model.norm.weight", "lm_head.weight"], weight_map)
    norm_weight = final["model.norm.weight"].to(device)
    head = final["lm_head.weight"].to(device)
    del final
    baseline_metrics, downstream = final_metrics(
        baseline, candidates, inputs, norm_weight, head, device, start, end
    )
    oracle_summary = summarize_oracle(oracle_raw)
    all_finite = all(
        math.isfinite(value)
        for sentinel in SENTINELS
        for domain in DOMAINS
        for key in ("relative_l2", "kkt_violation", "simplex_sum_error", "minimum_coefficient")
        for value in oracle_raw[sentinel][domain][key]
    ) and all(
        math.isfinite(row["aggregate"]["candidate_cross_entropy"])
        and math.isfinite(row["aggregate"]["natural_to_candidate_kl"]["mean"])
        for row in downstream.values()
    )
    controls = {
        "route_mismatch_count": route_mismatches,
        "manual_natural_mismatch_values": manual_mismatch_values,
        "all_finite": all_finite,
        "all_48_layers": len(layer_timings) == LAYERS,
        "strict_partition_labels": (end - 1 - start) == 255,
    }
    # Add min explicitly for adjudication without changing the common summary helper.
    for sentinel in SENTINELS:
        minima = [
            value
            for domain in DOMAINS
            for value in oracle_raw[sentinel][domain]["minimum_coefficient"]
        ]
        oracle_summary[sentinel]["minimum_coefficient"]["min"] = float(min(minima))
    gates = adjudicate(traffic, controls, oracle_summary, downstream)
    if phase == "validation":
        status = "validation_pass_test_authorized" if gates["overall_pass"] else "validation_negative_test_closed"
    else:
        status = "heldout_functional_span_pass" if gates["overall_pass"] else "heldout_functional_span_negative"
    payload = {
        "kind": "tierflow_persistent_set_functional_span",
        "phase": phase,
        "status": status,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "input_ids_sha256": sha256(INPUT_IDS),
            "input_lock_sha256": sha256(INPUT_LOCK),
            "capture_sha256": sha256(CAPTURE),
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "f0_reference_sha256": sha256(F0_VALIDATION if phase == "validation" else F0_TEST),
            "validation_sha256": sha256(VALIDATION_OUT) if phase == "test" else None,
        },
        "artifact_audit": audit,
        "partition": [start, end],
        "sentinel_layers": list(SENTINELS),
        "traffic": traffic,
        "controls": controls,
        "oracle_summary": {str(layer): row for layer, row in oracle_summary.items()},
        "oracle_raw": {str(layer): row for layer, row in oracle_raw.items()},
        "baseline": baseline_metrics,
        "downstream": {str(layer): row for layer, row in downstream.items()},
        "gates": gates,
        "layer_timings": layer_timings,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "rss_bytes": process.memory_info().rss,
        },
        "test_opened": phase == "test",
        "training_or_download": False,
        "claim_boundary": (
            "Non-causal per-token coefficient oracle on three one-layer interventions; "
            "not training, controller, full-48-layer TierFlow, latency, or deployment evidence."
        ),
        "output_artifact": str(output_path.relative_to(ROOT)).replace("\\", "/"),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    payload = run(args.phase)
    if payload["status"] == "blocked_artifact":
        REPORT.write_text(
            "# TierFlow persistent-set functional-span oracle\n\n"
            "Status: **blocked_artifact**.\n\n"
            f"Missing or invalid artifacts: `{json.dumps(payload['artifact_audit'], sort_keys=True)}`.\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "status": payload["status"],
        "gates": payload.get("gates"),
        "runtime": payload.get("runtime"),
        "output": payload["output_artifact"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
