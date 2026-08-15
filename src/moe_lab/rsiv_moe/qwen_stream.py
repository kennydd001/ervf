from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer


INDEX_NAME = "model.safetensors.index.json"


def checkpoint_weight_map(model_dir: Path) -> dict[str, str]:
    payload = json.loads((model_dir / INDEX_NAME).read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"invalid or empty weight_map in {model_dir / INDEX_NAME}")
    return {str(key): str(value) for key, value in weight_map.items()}


def load_checkpoint_tensors(
    model_dir: Path, names: list[str], weight_map: dict[str, str] | None = None
) -> dict[str, torch.Tensor]:
    if weight_map is None:
        weight_map = checkpoint_weight_map(model_dir)
    missing = sorted(set(names) - set(weight_map))
    if missing:
        raise KeyError(f"checkpoint tensors are absent from index: {missing[:5]}")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_shard[weight_map[name]].append(name)
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(by_shard):
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
            for name in by_shard[shard_name]:
                tensors[name] = shard.get_tensor(name)
    return tensors


def prefixed_checkpoint_state(
    model_dir: Path, prefix: str, weight_map: dict[str, str] | None = None
) -> dict[str, torch.Tensor]:
    if weight_map is None:
        weight_map = checkpoint_weight_map(model_dir)
    names = sorted(name for name in weight_map if name.startswith(prefix))
    if not names:
        raise KeyError(f"no checkpoint tensors start with {prefix!r}")
    tensors = load_checkpoint_tensors(model_dir, names, weight_map)
    return {name.removeprefix(prefix): tensor for name, tensor in tensors.items()}


def load_qwen_decoder_layer(
    model_dir: Path,
    config: Qwen3MoeConfig,
    layer_index: int,
    device: torch.device | str,
    weight_map: dict[str, str] | None = None,
) -> Qwen3MoeDecoderLayer:
    if not 0 <= layer_index < config.num_hidden_layers:
        raise ValueError(f"layer {layer_index} outside [0, {config.num_hidden_layers})")
    prefix = f"model.layers.{layer_index}."
    state = prefixed_checkpoint_state(model_dir, prefix, weight_map)
    with init_empty_weights(include_buffers=True):
        layer = Qwen3MoeDecoderLayer(config, layer_index)
    incompatible = layer.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible layer state: {incompatible}")
    del state
    layer = layer.to(device=device)
    layer.eval()
    return layer


def load_token_embeddings(
    model_dir: Path,
    input_ids: torch.Tensor,
    device: torch.device | str,
    weight_map: dict[str, str] | None = None,
) -> torch.Tensor:
    name = "model.embed_tokens.weight"
    tensor = load_checkpoint_tensors(model_dir, [name], weight_map)[name]
    flat = input_ids.reshape(-1).to(device="cpu", dtype=torch.long)
    if int(flat.min()) < 0 or int(flat.max()) >= tensor.shape[0]:
        raise ValueError("input token outside embedding vocabulary")
    selected = tensor.index_select(0, flat).reshape(*input_ids.shape, tensor.shape[1])
    return selected.to(device=device).contiguous()

