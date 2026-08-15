from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CapturedQwenMoeInvocations:
    """One official Qwen MoE forward, stored in token-major route order."""

    moe_input: torch.Tensor
    router_ids: torch.Tensor
    router_weights: torch.Tensor
    intermediate_z: torch.Tensor
    route_ids_exact: bool
    router_weight_maximum_absolute_error: float
    router_logits_maximum_absolute_error: float


def qwen_routes(block: torch.nn.Module, hidden_states: torch.Tensor):
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = block.gate(flat)
    weights = torch.softmax(logits, dim=1, dtype=torch.float)
    weights, ids = torch.topk(weights, block.top_k, dim=-1)
    if block.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return logits, ids, weights.to(flat.dtype)


class QwenMoeInvocationCapture:
    """Capture x/routes/z without replacing or approximating the official block."""

    def __init__(self, block: torch.nn.Module):
        self.block = block
        self._handles: list[Any] = []
        self._active = False
        self._flat_positions: dict[int, torch.Tensor] = {}
        self._z_chunks: dict[int, torch.Tensor] = {}
        self._moe_input: torch.Tensor | None = None
        self._router_logits: torch.Tensor | None = None
        self._router_ids: torch.Tensor | None = None
        self._router_weights: torch.Tensor | None = None
        self.result: CapturedQwenMoeInvocations | None = None

    def __enter__(self) -> "QwenMoeInvocationCapture":
        if self._handles:
            raise RuntimeError("capture hooks are already installed")
        self._handles.append(self.block.register_forward_pre_hook(self._block_pre_hook))
        for expert_index, expert in enumerate(self.block.experts):
            self._handles.append(
                expert.down_proj.register_forward_pre_hook(
                    self._make_down_pre_hook(expert_index)
                )
            )
        self._handles.append(self.block.register_forward_hook(self._block_post_hook))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def _block_pre_hook(self, _module, inputs) -> None:
        if self._active:
            raise RuntimeError("nested or repeated MoE capture before completion")
        hidden_states = inputs[0]
        logits, ids, weights = qwen_routes(self.block, hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        expert_mask = torch.nn.functional.one_hot(
            ids, num_classes=self.block.num_experts
        ).permute(2, 1, 0)
        self._flat_positions = {}
        for expert_index in range(self.block.num_experts):
            slot, token = torch.where(expert_mask[expert_index])
            self._flat_positions[expert_index] = (
                token * self.block.top_k + slot
            ).detach().cpu()
        self._z_chunks = {}
        self._moe_input = flat.detach().to("cpu").contiguous()
        self._router_logits = logits.detach()
        self._router_ids = ids.detach()
        self._router_weights = weights.detach()
        self.result = None
        self._active = True

    def _make_down_pre_hook(self, expert_index: int):
        def hook(_module, inputs) -> None:
            if not self._active:
                raise RuntimeError("expert down projection ran outside active capture")
            if expert_index in self._z_chunks:
                raise RuntimeError(f"expert {expert_index} ran twice in one capture")
            self._z_chunks[expert_index] = inputs[0].detach().to("cpu").contiguous()

        return hook

    def _block_post_hook(self, _module, _inputs, output) -> None:
        if not self._active:
            raise RuntimeError("MoE post-hook ran without a matching pre-hook")
        if self._moe_input is None or self._router_logits is None:
            raise RuntimeError("capture state is incomplete")
        if self._router_ids is None or self._router_weights is None:
            raise RuntimeError("capture routes are missing")

        official_logits = output[1]
        official_weights = torch.softmax(official_logits, dim=1, dtype=torch.float)
        official_weights, official_ids = torch.topk(
            official_weights, self.block.top_k, dim=-1
        )
        if self.block.norm_topk_prob:
            official_weights = official_weights / official_weights.sum(
                dim=-1, keepdim=True
            )
        official_weights = official_weights.to(self._router_weights.dtype)
        ids_exact = bool(torch.equal(self._router_ids, official_ids))
        weight_error = float(
            (self._router_weights.float() - official_weights.float()).abs().max().item()
        )
        logits_error = float(
            (self._router_logits.float() - official_logits.float()).abs().max().item()
        )

        routed_rows = self._router_ids.numel()
        intermediate_size = self.block.experts[0].intermediate_size
        z = torch.empty(
            (routed_rows, intermediate_size),
            dtype=self._moe_input.dtype,
            device="cpu",
        )
        filled = torch.zeros(routed_rows, dtype=torch.bool)
        for expert_index, positions in self._flat_positions.items():
            chunk = self._z_chunks.get(expert_index)
            if chunk is None:
                raise RuntimeError(f"missing z capture for expert {expert_index}")
            if chunk.shape[0] != positions.numel():
                raise RuntimeError(
                    f"expert {expert_index} z rows {chunk.shape[0]} != routes {positions.numel()}"
                )
            z.index_copy_(0, positions, chunk.to(dtype=z.dtype))
            filled[positions] = True
        if not bool(filled.all()):
            raise RuntimeError("one or more routed z rows were not captured")

        self.result = CapturedQwenMoeInvocations(
            moe_input=self._moe_input,
            router_ids=official_ids.detach().to(torch.int16).cpu().contiguous(),
            router_weights=official_weights.detach().float().cpu().contiguous(),
            intermediate_z=z,
            route_ids_exact=ids_exact,
            router_weight_maximum_absolute_error=weight_error,
            router_logits_maximum_absolute_error=logits_error,
        )
        self._active = False

