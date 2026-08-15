from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_qwen_decoder_layer,
    load_token_embeddings,
    prefixed_checkpoint_state,
)


def tiny_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
    )


def write_tiny_checkpoint(path: Path):
    torch.manual_seed(31)
    config = tiny_config()
    layer = Qwen3MoeDecoderLayer(config, 0).eval()
    full_state = {
        f"model.layers.0.{name}": tensor.detach().contiguous()
        for name, tensor in layer.state_dict().items()
    }
    embedding = torch.randn(config.vocab_size, config.hidden_size)
    full_state["model.embed_tokens.weight"] = embedding
    names = sorted(full_state)
    split = len(names) // 2
    partitions = [names[:split], names[split:]]
    weight_map = {}
    for shard_index, partition in enumerate(partitions, start=1):
        shard_name = f"model-{shard_index:05d}-of-00002.safetensors"
        save_file({name: full_state[name] for name in partition}, path / shard_name)
        weight_map.update({name: shard_name for name in partition})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return layer, embedding


def test_streamed_layer_matches_original_state(tmp_path: Path) -> None:
    original, _embedding = write_tiny_checkpoint(tmp_path)
    weight_map = checkpoint_weight_map(tmp_path)
    loaded = load_qwen_decoder_layer(tmp_path, tiny_config(), 0, "cpu", weight_map)
    assert set(loaded.state_dict()) == set(original.state_dict())
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)


def test_prefixed_state_and_sparse_embedding_lookup(tmp_path: Path) -> None:
    _layer, embedding = write_tiny_checkpoint(tmp_path)
    weight_map = checkpoint_weight_map(tmp_path)
    state = prefixed_checkpoint_state(tmp_path, "model.layers.0.", weight_map)
    assert "mlp.gate.weight" in state
    ids = torch.tensor([[3, 8], [2, 63]])
    selected = load_token_embeddings(tmp_path, ids, "cpu", weight_map)
    torch.testing.assert_close(selected, embedding[ids], rtol=0, atol=0)


def test_missing_prefix_fails_closed(tmp_path: Path) -> None:
    write_tiny_checkpoint(tmp_path)
    try:
        prefixed_checkpoint_state(tmp_path, "model.layers.9.")
    except KeyError as exc:
        assert "model.layers.9" in str(exc)
    else:
        raise AssertionError("missing checkpoint prefix should fail")

