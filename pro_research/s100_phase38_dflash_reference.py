"""Auditable BF16-dequant reference for NVIDIA's official Nemotron DFlash.

The six-layer DFlash body is evaluated with checkpoint NVFP4 weights decoded to
BF16. The shared target LM head remains packed NVFP4 and uses the exact existing
LightningStream GEMV kernel. This is a correctness/reference path, not a claim
that dequantized BF16 is an optimized deployment format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from moe_lab.lightningstream_nemotron.loader import ShardIndex

HIDDEN = 2688
INTERMEDIATE = 6144
HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 2
KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
VOCAB = 131072
MASK_TOKEN_ID = 990
BLOCK_SIZE = 8
EPS = 1e-6


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Qwen3 RMSNorm: FP32 variance and BF16 output."""
    dtype = x.dtype
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return weight * normalized.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class YaRNRotary:
    """Transformers-compatible YaRN RoPE for the published DFlash config."""

    def __init__(
        self,
        device: torch.device,
        *,
        dim: int = HEAD_DIM,
        base: float = 10000.0,
        factor: float = 128.0,
        original_max_position_embeddings: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        def correction_dim(rotations: float) -> float:
            return (
                dim
                * math.log(original_max_position_embeddings / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(base))
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=device, dtype=torch.float32) - low)
            / (high - low),
            0.0,
            1.0,
        )
        pos_freqs = base ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        )
        inv_extrapolation = 1.0 / pos_freqs
        inv_interpolation = 1.0 / (factor * pos_freqs)
        extrapolation_factor = 1.0 - ramp
        self.inv_freq = (
            inv_interpolation * (1.0 - extrapolation_factor)
            + inv_extrapolation * extrapolation_factor
        )
        self.attention_factor = 0.1 * math.log(factor) + 1.0

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_factor).to(dtype=x.dtype)[:, None, :]
        sin = (emb.sin() * self.attention_factor).to(dtype=x.dtype)[:, None, :]
        return x * cos + rotate_half(x) * sin


def _dequant_nvfp4(
    tensors,
    prefix: str,
    device: torch.device,
    *,
    row_chunk: int = 128,
) -> torch.Tensor:
    """Decode one ModelOpt group-16 NVFP4 matrix directly into GPU BF16."""
    packed_cpu = tensors.get_tensor(f"{prefix}.weight")
    scales_cpu = tensors.get_tensor(f"{prefix}.weight_scale")
    global_scale = float(tensors.get_tensor(f"{prefix}.weight_scale_2").item())
    if packed_cpu.dtype != torch.uint8 or packed_cpu.ndim != 2:
        raise TypeError(f"{prefix}: expected packed U8 rank-2 weight")
    rows, packed_cols = packed_cpu.shape
    cols = packed_cols * 2
    if tuple(scales_cpu.shape) != (rows, cols // 16):
        raise ValueError(
            f"{prefix}: scale shape {tuple(scales_cpu.shape)} != {(rows, cols // 16)}"
        )

    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty((rows, cols), dtype=torch.bfloat16, device=device)
    for start in range(0, rows, row_chunk):
        end = min(rows, start + row_chunk)
        packed = packed_cpu[start:end].to(device=device)
        elements = torch.empty((end - start, cols), dtype=torch.float32, device=device)
        elements[:, 0::2] = e2m1[(packed & 0x0F).long()]
        elements[:, 1::2] = e2m1[((packed >> 4) & 0x0F).long()]
        block_scales = scales_cpu[start:end].to(device=device).float()
        expanded = torch.repeat_interleave(block_scales, 16, dim=1)
        output[start:end] = (elements * expanded * global_scale).to(torch.bfloat16)
    return output


def _load_bf16(tensors, name: str, device: torch.device) -> torch.Tensor:
    tensor = tensors.get_tensor(name)
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name}: expected BF16, got {tensor.dtype}")
    return tensor.to(device=device)


@dataclass
class DFlashLayer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


class OfficialDFlashReference:
    def __init__(
        self,
        snapshot: Path,
        required_token_ids: Iterable[int],
        *,
        device: str = "cuda",
    ) -> None:
        self.snapshot = Path(snapshot)
        self.device = torch.device(device)
        self.model_path = self.snapshot / "model.safetensors"
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.embedding_rows: dict[int, torch.Tensor] = {}
        self.layers: list[DFlashLayer] = []

        with safe_open(str(self.model_path), framework="pt", device="cpu") as tensors:
            embedding = tensors.get_slice("embed_tokens.weight")
            for token_id in sorted({int(x) for x in required_token_ids} | {MASK_TOKEN_ID}):
                if token_id < 0 or token_id >= VOCAB:
                    raise ValueError(f"token ID outside DFlash vocabulary: {token_id}")
                row = embedding[token_id:token_id + 1]
                self.embedding_rows[token_id] = row[0].to(device=self.device)

            self.fc = _dequant_nvfp4(tensors, "fc", self.device)
            self.hidden_norm = _load_bf16(tensors, "hidden_norm.weight", self.device)
            self.final_norm = _load_bf16(tensors, "norm.weight", self.device)

            for index in range(6):
                prefix = f"layers.{index}"
                attn = f"{prefix}.self_attn"
                mlp = f"{prefix}.mlp"
                self.layers.append(DFlashLayer(
                    input_norm=_load_bf16(tensors, f"{prefix}.input_layernorm.weight", self.device),
                    post_norm=_load_bf16(tensors, f"{prefix}.post_attention_layernorm.weight", self.device),
                    q_norm=_load_bf16(tensors, f"{attn}.q_norm.weight", self.device),
                    k_norm=_load_bf16(tensors, f"{attn}.k_norm.weight", self.device),
                    q_proj=_load_bf16(tensors, f"{attn}.q_proj.weight", self.device),
                    k_proj=_load_bf16(tensors, f"{attn}.k_proj.weight", self.device),
                    v_proj=_load_bf16(tensors, f"{attn}.v_proj.weight", self.device),
                    o_proj=_load_bf16(tensors, f"{attn}.o_proj.weight", self.device),
                    gate_proj=_dequant_nvfp4(tensors, f"{mlp}.gate_proj", self.device),
                    up_proj=_dequant_nvfp4(tensors, f"{mlp}.up_proj", self.device),
                    down_proj=_dequant_nvfp4(tensors, f"{mlp}.down_proj", self.device),
                ))
        self.rope = YaRNRotary(self.device)

    @torch.inference_mode()
    def project_target(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        if aux_hidden.shape[-2:] != (6, HIDDEN):
            raise ValueError(f"expected [..., 6, {HIDDEN}], got {tuple(aux_hidden.shape)}")
        flattened = aux_hidden.reshape(*aux_hidden.shape[:-2], 6 * HIDDEN)
        projected = F.linear(flattened.to(torch.bfloat16), self.fc)
        return rms_norm(projected, self.hidden_norm)

    @torch.inference_mode()
    def precompute_context(
        self, aux_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        projected = self.project_target(aux_hidden)
        positions = torch.arange(projected.shape[0], device=self.device, dtype=torch.long)
        kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            k = F.linear(projected, layer.k_proj).view(-1, NUM_KV_HEADS, HEAD_DIM)
            v = F.linear(projected, layer.v_proj).view(-1, NUM_KV_HEADS, HEAD_DIM)
            k = rms_norm(k, layer.k_norm)
            k = self.rope.apply(k, positions)
            kv.append((k, v))
        return projected, kv

    @torch.inference_mode()
    def project_and_kv_one(
        self, aux_hidden_row: torch.Tensor, position: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        projected = self.project_target(aux_hidden_row[None])[0]
        pos = torch.tensor([int(position)], device=self.device, dtype=torch.long)
        result: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            k = F.linear(projected, layer.k_proj).view(1, NUM_KV_HEADS, HEAD_DIM)
            v = F.linear(projected, layer.v_proj).view(1, NUM_KV_HEADS, HEAD_DIM)
            k = self.rope.apply(rms_norm(k, layer.k_norm), pos)
            result.append((k, v))
        return result

    def _embed_block(self, anchor_token_id: int) -> torch.Tensor:
        ids = [int(anchor_token_id)] + [MASK_TOKEN_ID] * (BLOCK_SIZE - 1)
        return torch.stack([self.embedding_rows[token_id] for token_id in ids])

    @torch.inference_mode()
    def forward_block(
        self,
        *,
        anchor_position: int,
        anchor_token_id: int,
        context_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if anchor_position < 0:
            raise ValueError("negative anchor position")
        positions = torch.arange(
            anchor_position, anchor_position + BLOCK_SIZE,
            device=self.device, dtype=torch.long,
        )
        hidden = self._embed_block(anchor_token_id)

        for layer_index, layer in enumerate(self.layers):
            residual = hidden
            normed = rms_norm(hidden, layer.input_norm)
            q = F.linear(normed, layer.q_proj).view(BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
            k_block = F.linear(normed, layer.k_proj).view(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            v_block = F.linear(normed, layer.v_proj).view(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            q = self.rope.apply(rms_norm(q, layer.q_norm), positions)
            k_block = self.rope.apply(rms_norm(k_block, layer.k_norm), positions)

            k_context, v_context = context_kv[layer_index]
            key = torch.cat((k_context[:anchor_position], k_block), dim=0)
            value = torch.cat((v_context[:anchor_position], v_block), dim=0)
            key = torch.repeat_interleave(key, KV_GROUPS, dim=1).transpose(0, 1)
            value = torch.repeat_interleave(value, KV_GROUPS, dim=1).transpose(0, 1)
            query = q.transpose(0, 1)
            scores = torch.matmul(query, key.transpose(-1, -2)) * (HEAD_DIM ** -0.5)
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attention = torch.matmul(probabilities, value).transpose(0, 1).reshape(
                BLOCK_SIZE, NUM_HEADS * HEAD_DIM
            )
            hidden = residual + F.linear(attention, layer.o_proj)

            residual = hidden
            normed = rms_norm(hidden, layer.post_norm)
            mlp = F.silu(F.linear(normed, layer.gate_proj)) * F.linear(normed, layer.up_proj)
            hidden = residual + F.linear(mlp, layer.down_proj)

        return rms_norm(hidden, self.final_norm)

    @property
    def resident_weight_bytes(self) -> int:
        tensors = [self.fc, self.hidden_norm, self.final_norm]
        for layer in self.layers:
            tensors.extend(vars(layer).values())
        tensors.extend(self.embedding_rows.values())
        return int(sum(x.numel() * x.element_size() for x in tensors))


class TargetNVFP4Head:
    """The target's packed NVFP4 LM head, evaluated without target residency."""

    def __init__(self, target_snapshot: Path) -> None:
        import cupy as cp
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        self.cp = cp
        self.index = ShardIndex(Path(target_snapshot))
        if self.index.quant_kind("lm_head") != "nvfp4":
            raise RuntimeError("Phase38 requires the official NVFP4 target LM head")
        self.codes = cp.asarray(self.index.read_raw("lm_head.weight"))
        self.scales = cp.asarray(self.index.read_raw("lm_head.weight_scale"))
        self.global_scale = self.index.get_scalar("lm_head.weight_scale_2")
        self.kernel = FusedNVFP4()
        self.logits = cp.empty((BLOCK_SIZE - 1, VOCAB), dtype=cp.float32)

    @torch.inference_mode()
    def predict(self, hidden: torch.Tensor) -> np.ndarray:
        cp = self.cp
        if tuple(hidden.shape) != (BLOCK_SIZE - 1, HIDDEN):
            raise ValueError(f"expected head input {(BLOCK_SIZE - 1, HIDDEN)}, got {tuple(hidden.shape)}")
        activations = hidden.to(dtype=torch.float32).contiguous()
        try:
            cupy_activations = cp.from_dlpack(activations)
        except Exception:
            cupy_activations = cp.fromDlpack(torch.utils.dlpack.to_dlpack(activations))
        for row in range(BLOCK_SIZE - 1):
            self.kernel.gemv_into(
                self.logits[row], self.codes, self.scales, cupy_activations[row],
                self.global_scale, VOCAB, HIDDEN,
            )
        cp.cuda.get_current_stream().synchronize()
        return cp.asnumpy(cp.argmax(self.logits, axis=1)).astype(np.int32, copy=False)

    @property
    def resident_weight_bytes(self) -> int:
        return int(self.codes.nbytes + self.scales.nbytes + 4)
