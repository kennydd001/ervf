from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .deepseek_v2 import load_json
from .trace import MoETrace


@dataclass
class ProjectionWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass
class LoadedMoELayer:
    layer: int
    gate_weight: torch.Tensor
    experts: list[ProjectionWeights]
    shared: ProjectionWeights
    top_k: int
    routed_scaling_factor: float
    norm_topk_prob: bool

    @property
    def device(self) -> torch.device:
        return self.gate_weight.device

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(
            hidden_states.float(), self.gate_weight.float(), bias=None
        ).softmax(dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        else:
            weights = weights * self.routed_scaling_factor
        return ids, weights

    @staticmethod
    def expert_forward(x: torch.Tensor, weights: ProjectionWeights) -> torch.Tensor:
        return F.linear(
            F.silu(F.linear(x, weights.gate)) * F.linear(x, weights.up),
            weights.down,
        )

    @torch.inference_mode()
    def trace(self, hidden_states: torch.Tensor) -> MoETrace:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be [tokens, hidden] or [batch, seq, hidden]")
        x = hidden_states.to(device=self.device, dtype=self.gate_weight.dtype)
        router_ids, router_weights = self.route(x)
        tokens, hidden = x.shape
        selected = torch.empty(
            tokens,
            self.top_k,
            hidden,
            device=self.device,
            dtype=x.dtype,
        )
        for expert_id, expert in enumerate(self.experts):
            positions = (router_ids == expert_id).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids = positions[:, 0]
            slots = positions[:, 1]
            selected[token_ids, slots] = self.expert_forward(x[token_ids], expert)
        routed = (
            selected.float() * router_weights.unsqueeze(-1)
        ).sum(dim=1).to(x.dtype)
        shared = self.expert_forward(x, self.shared)
        return MoETrace(
            hidden_states=x.cpu(),
            router_ids=router_ids.to(torch.int16).cpu(),
            router_weights=router_weights.to(torch.float16).cpu(),
            selected_expert_outputs=selected.cpu(),
            routed_output=routed.cpu(),
            shared_output=shared.cpu(),
        )


def loaded_moe_from_official_module(
    module: torch.nn.Module, layer: int
) -> LoadedMoELayer:
    """Adapt the official DeepSeek MoE module without copying its parameters."""
    experts = [
        ProjectionWeights(
            gate=expert.gate_proj.weight,
            up=expert.up_proj.weight,
            down=expert.down_proj.weight,
        )
        for expert in module.experts
    ]
    shared = ProjectionWeights(
        gate=module.shared_experts.gate_proj.weight,
        up=module.shared_experts.up_proj.weight,
        down=module.shared_experts.down_proj.weight,
    )
    return LoadedMoELayer(
        layer=layer,
        gate_weight=module.gate.weight,
        experts=experts,
        shared=shared,
        top_k=int(module.num_experts_per_tok),
        routed_scaling_factor=float(module.config.routed_scaling_factor),
        norm_topk_prob=bool(module.config.norm_topk_prob),
    )


def _tensor_from_checkpoint(
    model_dir: Path, index: dict[str, Any], name: str, device: torch.device
) -> torch.Tensor:
    shard = model_dir / index["weight_map"][name]
    if not shard.is_file():
        raise FileNotFoundError(f"required shard is not present: {shard}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(device=device)


def _tensors_from_checkpoint(
    model_dir: Path,
    index: dict[str, Any],
    names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_tensor_names in by_shard.items():
        shard = model_dir / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"required shard is not present: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in shard_tensor_names:
                tensors[name] = handle.get_tensor(name).to(device=device)
    return tensors


def load_moe_layer(
    model_dir: Path, layer: int, device: torch.device | str
) -> LoadedMoELayer:
    device = torch.device(device)
    config = load_json(model_dir / "config.json")
    index = load_json(model_dir / "model.safetensors.index.json")
    prefix = f"model.layers.{layer}.mlp"
    names = [f"{prefix}.gate.weight"]
    for expert_id in range(int(config["n_routed_experts"])):
        expert_prefix = f"{prefix}.experts.{expert_id}"
        names.extend(
            [
                f"{expert_prefix}.gate_proj.weight",
                f"{expert_prefix}.up_proj.weight",
                f"{expert_prefix}.down_proj.weight",
            ]
        )
    shared_prefix = f"{prefix}.shared_experts"
    names.extend(
        [
            f"{shared_prefix}.gate_proj.weight",
            f"{shared_prefix}.up_proj.weight",
            f"{shared_prefix}.down_proj.weight",
        ]
    )
    tensors = _tensors_from_checkpoint(model_dir, index, names, device)
    gate_weight = tensors[f"{prefix}.gate.weight"]
    experts = []
    for expert_id in range(int(config["n_routed_experts"])):
        expert_prefix = f"{prefix}.experts.{expert_id}"
        experts.append(
            ProjectionWeights(
                gate=tensors[f"{expert_prefix}.gate_proj.weight"],
                up=tensors[f"{expert_prefix}.up_proj.weight"],
                down=tensors[f"{expert_prefix}.down_proj.weight"],
            )
        )
    shared = ProjectionWeights(
        gate=tensors[f"{shared_prefix}.gate_proj.weight"],
        up=tensors[f"{shared_prefix}.up_proj.weight"],
        down=tensors[f"{shared_prefix}.down_proj.weight"],
    )
    return LoadedMoELayer(
        layer=layer,
        gate_weight=gate_weight,
        experts=experts,
        shared=shared,
        top_k=int(config["num_experts_per_tok"]),
        routed_scaling_factor=float(config["routed_scaling_factor"]),
        norm_topk_prob=bool(config["norm_topk_prob"]),
    )


def load_token_embeddings(
    model_dir: Path, token_ids: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    index = load_json(model_dir / "model.safetensors.index.json")
    embedding = _tensor_from_checkpoint(
        model_dir, index, "model.embed_tokens.weight", torch.device("cpu")
    )
    selected = embedding[token_ids.to(torch.long).cpu()]
    return selected.to(device=torch.device(device))
