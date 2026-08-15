from __future__ import annotations

import torch
from torch import nn


def dense_router_features(
    router_ids: torch.Tensor, router_weights: torch.Tensor, num_experts: int
) -> torch.Tensor:
    features = torch.zeros(
        router_ids.shape[0],
        num_experts,
        device=router_weights.device,
        dtype=router_weights.dtype,
    )
    features.scatter_(1, router_ids.long(), router_weights)
    return features


class AggregateStudent(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        route_conditioned: bool,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.route_conditioned = route_conditioned
        input_size = hidden_size + (num_experts if route_conditioned else 0)
        self.gate_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_ids: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        inputs = hidden_states
        if self.route_conditioned:
            route = dense_router_features(
                router_ids, router_weights, self.num_experts
            ).to(hidden_states.dtype)
            inputs = torch.cat((hidden_states, route), dim=-1)
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ResidualBasisStudent(nn.Module):
    """One shared SwiGLU expert plus expert-specific low-rank linear residuals."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        adapter_rank: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.adapter_rank = adapter_rank
        self.base = AggregateStudent(
            hidden_size, intermediate_size, num_experts, route_conditioned=False
        )
        self.expert_in = nn.Parameter(
            torch.empty(num_experts, hidden_size, adapter_rank)
        )
        self.shared_out = nn.Linear(adapter_rank, hidden_size, bias=False)
        nn.init.normal_(self.expert_in, mean=0.0, std=0.002)
        nn.init.normal_(self.shared_out.weight, mean=0.0, std=0.002)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_ids: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base(hidden_states, router_ids, router_weights)
        coefficients = torch.zeros(
            hidden_states.shape[0],
            self.adapter_rank,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        for expert_id in range(self.num_experts):
            positions = (router_ids == expert_id).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids, slots = positions[:, 0], positions[:, 1]
            values = hidden_states[token_ids] @ self.expert_in[expert_id]
            selected_weights = router_weights[token_ids, slots].to(values.dtype)
            values = values * selected_weights.unsqueeze(-1)
            coefficients.index_add_(0, token_ids, values)
        return base + self.shared_out(coefficients)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
