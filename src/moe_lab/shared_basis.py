from __future__ import annotations

import torch

from .trace import MoETrace


def fit_output_basis(
    trace: MoETrace, max_rank: int, device: torch.device | str, niter: int = 4
) -> torch.Tensor:
    """Fit a shared, uncentered basis to router-weighted expert outputs."""
    device = torch.device(device)
    outputs = trace.selected_expert_outputs.to(device=device, dtype=torch.float32)
    weights = trace.router_weights.to(device=device, dtype=torch.float32)
    weighted = (outputs * weights.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
    q = min(max_rank, weighted.shape[0], weighted.shape[1])
    _, _, basis = torch.pca_lowrank(weighted, q=q, center=False, niter=niter)
    return basis.contiguous()


def fit_expert_maps(
    trace: MoETrace,
    basis: torch.Tensor,
    ridge_relative: float,
    num_experts: int,
) -> tuple[torch.Tensor, list[int]]:
    """Fit weighted dual-ridge maps x -> shared-basis coefficients per expert."""
    device = basis.device
    x_all = trace.hidden_states.to(device=device, dtype=torch.float32)
    ids = trace.router_ids.to(device=device, dtype=torch.long)
    weights = trace.router_weights.to(device=device, dtype=torch.float32)
    outputs = trace.selected_expert_outputs.to(device=device, dtype=torch.float32)
    hidden, rank = x_all.shape[-1], basis.shape[-1]
    maps = torch.zeros(num_experts, hidden, rank, device=device)
    counts: list[int] = []
    for expert_id in range(num_experts):
        positions = (ids == expert_id).nonzero(as_tuple=False)
        counts.append(int(positions.shape[0]))
        if positions.numel() == 0:
            continue
        token_ids, slots = positions[:, 0], positions[:, 1]
        x = x_all[token_ids]
        coefficients = outputs[token_ids, slots] @ basis
        sample_weight = weights[token_ids, slots].clamp_min(0).sqrt().unsqueeze(-1)
        x_weighted = x * sample_weight
        coefficient_weighted = coefficients * sample_weight
        gram = x_weighted @ x_weighted.T
        scale = gram.diagonal().mean().clamp_min(1e-12)
        regularized = gram + ridge_relative * scale * torch.eye(
            gram.shape[0], device=device, dtype=gram.dtype
        )
        dual = torch.linalg.solve(regularized, coefficient_weighted)
        maps[expert_id] = x_weighted.T @ dual
    return maps, counts


def predict_routed(
    trace: MoETrace,
    basis: torch.Tensor,
    expert_maps: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    device = basis.device
    x = trace.hidden_states.to(device=device, dtype=torch.float32)
    ids = trace.router_ids.to(device=device, dtype=torch.long)
    weights = trace.router_weights.to(device=device, dtype=torch.float32)
    prediction = torch.zeros(x.shape[0], x.shape[1], device=device)
    basis_rank = basis[:, :rank]
    for expert_id in range(expert_maps.shape[0]):
        positions = (ids == expert_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_ids, slots = positions[:, 0], positions[:, 1]
        coefficients = x[token_ids] @ expert_maps[expert_id, :, :rank]
        output = coefficients @ basis_rank.T
        prediction.index_add_(
            0, token_ids, output * weights[token_ids, slots].unsqueeze(-1)
        )
    return prediction.cpu()


def oracle_projected_routed(
    trace: MoETrace, basis: torch.Tensor, rank: int
) -> torch.Tensor:
    device = basis.device
    outputs = trace.selected_expert_outputs.to(device=device, dtype=torch.float32)
    weights = trace.router_weights.to(device=device, dtype=torch.float32)
    basis_rank = basis[:, :rank]
    projected = (outputs @ basis_rank) @ basis_rank.T
    return (projected * weights.unsqueeze(-1)).sum(dim=1).cpu()


def shared_basis_parameter_count(hidden_size: int, num_experts: int, rank: int) -> int:
    return hidden_size * rank * (num_experts + 1)
