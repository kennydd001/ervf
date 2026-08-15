from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from .metrics import mean_cosine_similarity, normalized_rmse


@dataclass(frozen=True)
class SyntheticConfig:
    tokens: int = 2048
    hidden_size: int = 128
    intermediate_size: int = 256
    experts: int = 16
    top_k: int = 2
    seed: int = 20260809


def _expert(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ gate) * (x @ up)) @ down


@torch.inference_mode()
def run_synthetic(config: SyntheticConfig, device: torch.device) -> dict[str, Any]:
    if not 0 < config.top_k <= config.experts:
        raise ValueError("top_k must be between 1 and experts")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    dtype = torch.float32
    h, m, e = config.hidden_size, config.intermediate_size, config.experts
    scale_h = h**-0.5
    scale_m = m**-0.5
    x = torch.randn(config.tokens, h, device=device, dtype=dtype)
    router = torch.randn(h, e, device=device, dtype=dtype) * scale_h
    gate = torch.randn(e, h, m, device=device, dtype=dtype) * scale_h
    up = torch.randn(e, h, m, device=device, dtype=dtype) * scale_h
    down = torch.randn(e, m, h, device=device, dtype=dtype) * scale_m

    route_logits = x @ router
    route_values, route_ids = route_logits.topk(config.top_k, dim=-1)
    route_weights = route_values.softmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    teacher = torch.zeros_like(x)
    contributions = torch.zeros(config.tokens, config.top_k, h, device=device)
    for slot in range(config.top_k):
        selected = route_ids[:, slot]
        for expert_id in range(config.experts):
            mask = selected == expert_id
            if mask.any():
                contributions[mask, slot] = _expert(
                    x[mask], gate[expert_id], up[expert_id], down[expert_id]
                )
        teacher += route_weights[:, slot : slot + 1] * contributions[:, slot]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started

    top1 = route_weights[:, :1] * contributions[:, 0]
    zero = torch.zeros_like(teacher)
    expert_parameters = config.experts * 3 * h * m
    active_parameters_per_token = config.top_k * 3 * h * m

    return {
        "config": asdict(config),
        "device": str(device),
        "teacher_seconds": elapsed,
        "teacher_tokens_per_second": config.tokens / elapsed,
        "peak_cuda_memory_mib": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 3)
            if device.type == "cuda"
            else None
        ),
        "route_histogram": torch.bincount(route_ids.flatten(), minlength=e).cpu().tolist(),
        "expert_parameters": expert_parameters,
        "active_expert_parameters_per_token": active_parameters_per_token,
        "baselines": {
            "exact_topk": {"nrmse": 0.0, "cosine": 1.0},
            "zero": {
                "nrmse": normalized_rmse(zero, teacher),
                "cosine": mean_cosine_similarity(zero, teacher),
            },
            "top1_unrenormalized": {
                "nrmse": normalized_rmse(top1, teacher),
                "cosine": mean_cosine_similarity(top1, teacher),
            },
        },
    }
