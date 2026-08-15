from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .deepseek_v2 import load_json


def checkpoint_state_for_prefix(model_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    """Load a module state dict while touching only its referenced shards."""
    index = load_json(model_dir / "model.safetensors.index.json")
    full_prefix = f"{prefix}."
    names = sorted(name for name in index["weight_map"] if name.startswith(full_prefix))
    if not names:
        raise KeyError(f"checkpoint contains no tensors below {prefix!r}")
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    state: dict[str, torch.Tensor] = {}
    for shard_name, shard_names in by_shard.items():
        shard = model_dir / shard_name
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in shard_names:
                state[name[len(full_prefix) :]] = handle.get_tensor(name)
    return state


def load_decoder_layer(
    model_dir: Path, layer_idx: int, device: torch.device | str
) -> tuple[torch.nn.Module, Any]:
    """Instantiate one official DeepSeek decoder layer and assign pinned weights."""
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    config._attn_implementation = "eager"
    decoder_class = get_class_from_dynamic_module(
        "modeling_deepseek.DeepseekV2DecoderLayer",
        str(model_dir),
        local_files_only=True,
    )
    layer = decoder_class(config, layer_idx)
    state = checkpoint_state_for_prefix(model_dir, f"model.layers.{layer_idx}")
    layer.load_state_dict(state, strict=True, assign=True)
    layer = layer.to(device=torch.device(device)).eval()
    return layer, config


@torch.inference_mode()
def run_layer_zero(
    layer_zero: torch.nn.Module, inputs_embeds: torch.Tensor
) -> torch.Tensor:
    batch, sequence, _ = inputs_embeds.shape
    position_ids = torch.arange(sequence, device=inputs_embeds.device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), inputs_embeds, 0
    )
    return layer_zero(
        inputs_embeds,
        attention_mask=causal_mask,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
    )[0]


@torch.inference_mode()
def layer_moe_input(
    decoder_layer: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Return the exact tensor entering a decoder layer's MoE block."""
    batch, sequence, _ = hidden_states.shape
    position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    residual = hidden_states
    normalized = decoder_layer.input_layernorm(hidden_states)
    attention_output = decoder_layer.self_attn(
        hidden_states=normalized,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )[0]
    post_attention = residual + attention_output
    return decoder_layer.post_attention_layernorm(post_attention)
