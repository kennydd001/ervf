from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


TRACE_SCHEMA_VERSION = 1


@dataclass
class MoETrace:
    hidden_states: torch.Tensor
    router_ids: torch.Tensor
    router_weights: torch.Tensor
    selected_expert_outputs: torch.Tensor
    routed_output: torch.Tensor
    shared_output: torch.Tensor

    @property
    def combined_output(self) -> torch.Tensor:
        return self.routed_output + self.shared_output

    def validate(self, atol: float = 2e-2, rtol: float = 2e-2) -> dict[str, Any]:
        if self.hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden]")
        tokens, hidden = self.hidden_states.shape
        if self.router_ids.ndim != 2 or self.router_ids.shape[0] != tokens:
            raise ValueError("router_ids must have shape [tokens, top_k]")
        if self.router_weights.shape != self.router_ids.shape:
            raise ValueError("router_weights must match router_ids")
        top_k = self.router_ids.shape[1]
        if self.selected_expert_outputs.shape != (tokens, top_k, hidden):
            raise ValueError(
                "selected_expert_outputs must have shape [tokens, top_k, hidden]"
            )
        if self.routed_output.shape != (tokens, hidden):
            raise ValueError("routed_output must have shape [tokens, hidden]")
        if self.shared_output.shape != (tokens, hidden):
            raise ValueError("shared_output must have shape [tokens, hidden]")
        reconstructed = (
            self.selected_expert_outputs.float()
            * self.router_weights.float().unsqueeze(-1)
        ).sum(dim=1)
        max_abs_error = float(
            (reconstructed - self.routed_output.float()).abs().max().item()
        )
        valid = torch.allclose(
            reconstructed, self.routed_output.float(), atol=atol, rtol=rtol
        )
        if not valid:
            raise ValueError(
                f"routed output does not match weighted expert sum; max error {max_abs_error}"
            )
        return {
            "tokens": tokens,
            "hidden_size": hidden,
            "top_k": top_k,
            "max_routed_reconstruction_abs_error": max_abs_error,
            "router_weight_sum_mean": float(
                self.router_weights.float().sum(dim=-1).mean().item()
            ),
            "router_weight_sum_min": float(
                self.router_weights.float().sum(dim=-1).min().item()
            ),
            "router_weight_sum_max": float(
                self.router_weights.float().sum(dim=-1).max().item()
            ),
        }

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": self.hidden_states.contiguous(),
            "router_ids": self.router_ids.contiguous(),
            "router_weights": self.router_weights.contiguous(),
            "selected_expert_outputs": self.selected_expert_outputs.contiguous(),
            "routed_output": self.routed_output.contiguous(),
            "shared_output": self.shared_output.contiguous(),
        }


def save_trace(
    trace: MoETrace, path: Path, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    validation = trace.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    text_metadata = {
        "schema_version": str(TRACE_SCHEMA_VERSION),
        "validation": json.dumps(validation, sort_keys=True),
    }
    if metadata:
        text_metadata["experiment"] = json.dumps(metadata, sort_keys=True)
    save_file(trace.tensors(), path, metadata=text_metadata)
    return validation


def load_trace(path: Path) -> MoETrace:
    tensors = load_file(path, device="cpu")
    trace = MoETrace(
        hidden_states=tensors["hidden_states"],
        router_ids=tensors["router_ids"],
        router_weights=tensors["router_weights"],
        selected_expert_outputs=tensors["selected_expert_outputs"],
        routed_output=tensors["routed_output"],
        shared_output=tensors["shared_output"],
    )
    trace.validate()
    return trace


def slice_trace(trace: MoETrace, tokens: int) -> MoETrace:
    if tokens <= 0 or tokens > trace.hidden_states.shape[0]:
        raise ValueError("tokens must be within the trace length")
    return MoETrace(
        hidden_states=trace.hidden_states[:tokens],
        router_ids=trace.router_ids[:tokens],
        router_weights=trace.router_weights[:tokens],
        selected_expert_outputs=trace.selected_expert_outputs[:tokens],
        routed_output=trace.routed_output[:tokens],
        shared_output=trace.shared_output[:tokens],
    )


def trace_baselines(trace: MoETrace) -> dict[str, torch.Tensor]:
    top1_slot = trace.router_weights.argmax(dim=-1)
    token_idx = torch.arange(trace.router_ids.shape[0])
    top1_unrenormalized = (
        trace.selected_expert_outputs[token_idx, top1_slot]
        * trace.router_weights[token_idx, top1_slot].unsqueeze(-1)
    )
    top1_renormalized = trace.selected_expert_outputs[token_idx, top1_slot]
    return {
        "zero": torch.zeros_like(trace.routed_output),
        "top1_unrenormalized": top1_unrenormalized,
        "top1_renormalized": top1_renormalized,
    }
