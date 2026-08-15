from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)

from moe_lab.rsiv_moe.qwen_capture import QwenMoeInvocationCapture, qwen_routes


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


def tiny_block() -> Qwen3MoeSparseMoeBlock:
    torch.manual_seed(29)
    config = tiny_config()
    return Qwen3MoeSparseMoeBlock(config).eval()


def test_capture_preserves_official_forward_and_token_major_z() -> None:
    block = tiny_block()
    hidden = torch.randn(2, 7, 16)
    with torch.inference_mode():
        expected_output, expected_logits = block(hidden)
        with QwenMoeInvocationCapture(block) as capture:
            actual_output, actual_logits = block(hidden)

    assert capture.result is not None
    result = capture.result
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    assert result.route_ids_exact
    assert result.router_weight_maximum_absolute_error == 0
    assert result.router_logits_maximum_absolute_error == 0

    flat = hidden.reshape(-1, hidden.shape[-1])
    _logits, ids, weights = qwen_routes(block, hidden)
    torch.testing.assert_close(result.moe_input, flat)
    torch.testing.assert_close(result.router_ids.long(), ids)
    torch.testing.assert_close(result.router_weights, weights.float())
    for token in range(flat.shape[0]):
        for slot in range(block.top_k):
            expert = block.experts[int(ids[token, slot])]
            expected_z = (
                expert.act_fn(expert.gate_proj(flat[token]))
                * expert.up_proj(flat[token])
            )
            actual_z = result.intermediate_z[token * block.top_k + slot]
            torch.testing.assert_close(actual_z, expected_z)


def test_capture_can_be_reused_for_sequential_forwards() -> None:
    block = tiny_block()
    with torch.inference_mode():
        with QwenMoeInvocationCapture(block) as first:
            block(torch.randn(1, 3, 16))
        with QwenMoeInvocationCapture(block) as second:
            block(torch.randn(1, 5, 16))
    assert first.result is not None
    assert second.result is not None
    assert first.result.moe_input.shape == (3, 16)
    assert second.result.intermediate_z.shape == (10, 8)


def test_capture_wraps_real_decoder_layer_call_used_by_streamer() -> None:
    config = tiny_config()
    config._attn_implementation = "sdpa"
    layer = Qwen3MoeDecoderLayer(config, 0).eval()
    hidden = torch.randn(2, 6, config.hidden_size)
    positions = torch.arange(6).unsqueeze(0)
    rotary = Qwen3MoeRotaryEmbedding(config)
    position_embeddings = rotary(hidden, positions)
    with torch.inference_mode(), QwenMoeInvocationCapture(layer.mlp) as capture:
        output = layer(
            hidden,
            attention_mask=None,
            position_ids=positions,
            use_cache=False,
            output_attentions=False,
            output_router_logits=False,
            cache_position=positions.squeeze(0),
            position_embeddings=position_embeddings,
        )[0]
    assert output.shape == hidden.shape
    assert capture.result is not None
    assert capture.result.moe_input.shape == (12, config.hidden_size)
    assert capture.result.intermediate_z.shape == (
        12 * config.num_experts_per_tok,
        config.moe_intermediate_size,
    )
