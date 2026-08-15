#!/usr/bin/env python3
"""PORT80B T0-R3 official layer-0 CPU reference runner.

`--phase smoke` is read-only and never executes the official forward.
`--phase reference --run-index {1,2}` is closed behind an exact acknowledgement
and an external immutable runner lock. CUDA and network access remain disabled.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any
import zlib

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("USE_HUB_KERNELS", "NO")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache
from transformers.models.qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDecoderLayer


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
RUN_DIR = ROOT / "reports" / "runs" / "streamq5_moe" / "port80b_t0r3_official_layer0"
SNAPSHOT = (Path.home() / ".cache" / "huggingface" / "hub" /
            "models--Qwen--Qwen3-Coder-Next" / "snapshots" /
            "a19358a7659bd1f564300250ee189120c49a562f")
SHARD = SNAPSHOT / "model-00001-of-00040.safetensors"
INDEX = SNAPSHOT / "model.safetensors.index.json"
PREREG = REPORTS / "PORT80B_T0R3_OFFICIAL_LAYER0_REFERENCE_PREREGISTRATION_2026-08-13.md"
P3_PREREG = REPORTS / "PORT80B_T0P3_OFFICIAL_LAYER0_PHYSICAL_PREREGISTRATION_2026-08-13.md"
PROMPT_LOCK = REPORTS / "port80b_t0r1_prompt_lock.json"
ENV_LOCK = REPORTS / "port80b_t0r1_reference_environment_lock.json"
RUNNER_LOCK = REPORTS / "port80b_t0r3_runner_lock.json"
RECORD_ARTIFACT = RUN_DIR / "layer0_513_real_q5_records.bin"
RECORD_MANIFEST = RUN_DIR / "layer0_513_real_q5_records_manifest.json"
ACK = "T0R3_OFFICIAL_LAYER0_CPU_REFERENCE_AFTER_INDEPENDENT_AUDIT"

REVISION = "a19358a7659bd1f564300250ee189120c49a562f"
SHARD_BYTES = 3_999_619_288
SHARD_SHA = "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a"
INDEX_SHA = "e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b"
R3_SHA = "0663b368a47a35a0c029ead2cc66d940c5ba91b3b3f83deb47d3ec8d06b9ff03"
P3_SHA = "b5b58eb66eeb9973edb552a5ed25ed636351040cab2be06fb3cb5ceda4187d21"
PROMPT_SHA = "f283da7e86adf915431459b08aac967d9c18c3de155699c369f5a55be20e5f34"
ENV_SHA = "eb31d4e0c1f6a806434ea8a20b6b00200781a89ed9f91e485aad0e3583c0f455"

GROUP = 128
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
PADDING_BYTES = 4_032
MATRIX_BYTES = 675_840
EXPERT_BYTES = 2_027_520
BANK_BYTES = 1_040_117_760
EXPERTS = 512
TOP_K = 10
HIDDEN = 2048
INTER = 512
MAX_RSS = 12 * 2**30
MIN_START_RAM = 8 * 2**30
MIN_RESERVE = 2 * 2**30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.float()).all().item())


def rss_guard(stage: str, peak: dict[str, int]) -> None:
    rss = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    peak["peak_rss_bytes"] = max(peak.get("peak_rss_bytes", 0), rss)
    peak["minimum_available_ram_bytes"] = min(peak.get("minimum_available_ram_bytes", available), available)
    if rss > MAX_RSS or available < MIN_RESERVE:
        raise MemoryError(f"resource gate at {stage}: rss={rss}, available={available}")


def locked_inputs() -> dict[str, Any]:
    wanted = {PREREG: R3_SHA, P3_PREREG: P3_SHA, PROMPT_LOCK: PROMPT_SHA, ENV_LOCK: ENV_SHA, INDEX: INDEX_SHA}
    observed = {str(path): sha256(path) if path.is_file() else None for path in wanted}
    if any(observed[str(path)] != expected for path, expected in wanted.items()):
        raise RuntimeError(f"immutable input mismatch: {observed}")
    if SHARD.stat().st_size != SHARD_BYTES or sha256(SHARD) != SHARD_SHA:
        raise RuntimeError("official shard size/hash mismatch")
    return observed


def expected_shapes() -> dict[str, list[int]]:
    result: dict[str, list[int]] = {
        "model.embed_tokens.weight": [151_936, HIDDEN],
        "model.layers.0.input_layernorm.weight": [HIDDEN],
        "model.layers.0.linear_attn.A_log": [32],
        "model.layers.0.linear_attn.conv1d.weight": [8192, 1, 4],
        "model.layers.0.linear_attn.dt_bias": [32],
        "model.layers.0.linear_attn.in_proj_ba.weight": [64, HIDDEN],
        "model.layers.0.linear_attn.in_proj_qkvz.weight": [12288, HIDDEN],
        "model.layers.0.linear_attn.norm.weight": [128],
        "model.layers.0.linear_attn.out_proj.weight": [HIDDEN, 4096],
        "model.layers.0.mlp.gate.weight": [EXPERTS, HIDDEN],
        "model.layers.0.mlp.shared_expert.gate_proj.weight": [INTER, HIDDEN],
        "model.layers.0.mlp.shared_expert.up_proj.weight": [INTER, HIDDEN],
        "model.layers.0.mlp.shared_expert.down_proj.weight": [HIDDEN, INTER],
        "model.layers.0.mlp.shared_expert_gate.weight": [1, HIDDEN],
        "model.layers.0.post_attention_layernorm.weight": [HIDDEN],
    }
    for expert in range(EXPERTS):
        base = f"model.layers.0.mlp.experts.{expert}"
        result[f"{base}.gate_proj.weight"] = [INTER, HIDDEN]
        result[f"{base}.up_proj.weight"] = [INTER, HIDDEN]
        result[f"{base}.down_proj.weight"] = [HIDDEN, INTER]
    return result


def inspect_safetensors_header() -> dict[str, Any]:
    expected = expected_shapes()
    weight_map = json.loads(INDEX.read_text(encoding="utf-8"))["weight_map"]
    expected_keys = {key for key, value in weight_map.items() if value == SHARD.name}
    with SHARD.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    entries = {key: value for key, value in header.items() if key != "__metadata__"}
    data_bytes = SHARD_BYTES - 8 - header_len
    intervals = []
    failures = []
    for key, meta in entries.items():
        begin, end = meta["data_offsets"]
        intervals.append((begin, end, key))
        if meta.get("dtype") != "BF16" or begin < 0 or end <= begin or end > data_bytes:
            failures.append(key)
        if key in expected and list(meta.get("shape", [])) != expected[key]:
            failures.append(key + ":shape")
    intervals.sort()
    overlap = any(intervals[i - 1][1] > intervals[i][0] for i in range(1, len(intervals)))
    result = {
        "header_bytes": header_len, "data_bytes": data_bytes, "tensor_entries": len(entries),
        "expected_entries": len(expected_keys), "all_bf16": all(v.get("dtype") == "BF16" for v in entries.values()),
        "keys_exact": set(entries) == expected_keys == set(expected), "offsets_nonoverlap": not overlap,
        "offsets_in_range": not failures, "shape_failures": failures[:20],
    }
    result["pass"] = (result["tensor_entries"] == 1567 and result["expected_entries"] == 1567
                      and result["all_bf16"] and result["keys_exact"]
                      and result["offsets_nonoverlap"] and result["offsets_in_range"])
    return result


def codec_sentinel() -> dict[str, Any]:
    signed = list(range(-15, 16)) + [-15]
    fields = [code + 15 for code in signed]
    packed = bytearray()
    for begin in range(0, 32, 8):
        word = sum(field << (5 * slot) for slot, field in enumerate(fields[begin:begin + 8]))
        packed.extend(word.to_bytes(5, "little"))
    decoded_fields = []
    for begin in range(0, 20, 5):
        word = int.from_bytes(packed[begin:begin + 5], "little")
        decoded_fields += [(word >> (5 * slot)) & 31 for slot in range(8)]
    invalid = bytearray(packed)
    invalid[:5] = ((int.from_bytes(invalid[:5], "little") & ~31) | 31).to_bytes(5, "little")
    invalid_detected = (int.from_bytes(invalid[:5], "little") & 31) == 31
    return {"pass": decoded_fields == fields and [x - 15 for x in decoded_fields] == signed and invalid_detected,
            "packed_hex": packed.hex(), "field_31_rejected": invalid_detected}


def make_layer(config: Qwen3NextConfig) -> Qwen3NextDecoderLayer:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            layer = Qwen3NextDecoderLayer(config, 0)
    finally:
        torch.set_default_dtype(previous)
    layer.to_empty(device="cpu")
    if any(parameter.dtype != torch.bfloat16 for parameter in layer.parameters()):
        raise RuntimeError("meta-to-empty materialized non-BF16 parameters")
    return layer.eval()


def load_official(config: Qwen3NextConfig, peak: dict[str, int]) -> tuple[Qwen3NextDecoderLayer, torch.Tensor, dict[str, str]]:
    layer = make_layer(config)
    identities: dict[str, str] = {}
    with torch.no_grad(), safe_open(SHARD, framework="pt", device="cpu") as source:
        for name, parameter in layer.named_parameters():
            if name == "mlp.experts.gate_up_proj":
                for expert in range(EXPERTS):
                    gate_name = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
                    up_name = f"model.layers.0.mlp.experts.{expert}.up_proj.weight"
                    gate = source.get_tensor(gate_name); up = source.get_tensor(up_name)
                    if list(gate.shape) != [INTER, HIDDEN] or list(up.shape) != [INTER, HIDDEN]:
                        raise RuntimeError("expert gate/up shape mismatch")
                    identities[gate_name] = tensor_sha(gate); identities[up_name] = tensor_sha(up)
                    parameter[expert, :INTER].copy_(gate)
                    parameter[expert, INTER:].copy_(up)
                    del gate, up
                    if expert % 32 == 0:
                        rss_guard(f"load_gate_up_{expert}", peak)
            elif name == "mlp.experts.down_proj":
                for expert in range(EXPERTS):
                    key = f"model.layers.0.mlp.experts.{expert}.down_proj.weight"
                    value = source.get_tensor(key)
                    identities[key] = tensor_sha(value); parameter[expert].copy_(value); del value
                    if expert % 32 == 0:
                        rss_guard(f"load_down_{expert}", peak)
            else:
                key = f"model.layers.0.{name}"
                value = source.get_tensor(key)
                if tuple(value.shape) != tuple(parameter.shape) or value.dtype != torch.bfloat16:
                    raise RuntimeError(f"parameter mismatch {key}")
                identities[key] = tensor_sha(value); parameter.copy_(value); del value
        embed = source.get_tensor("model.embed_tokens.weight")
        identities["model.embed_tokens.weight"] = tensor_sha(embed)
    return layer, embed, identities


def bf16_ordered(bits: np.ndarray) -> np.ndarray:
    raw = bits.astype(np.uint16).astype(np.int32)
    return np.where((raw & 0x8000) != 0, 0x8000 - (raw & 0x7FFF), 0x8000 + raw).astype(np.int32)


def max_bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    aa = a.detach().contiguous().view(torch.uint16).cpu().numpy()
    bb = b.detach().contiguous().view(torch.uint16).cpu().numpy()
    return int(np.max(np.abs(bf16_ordered(aa) - bf16_ordered(bb)))) if aa.size else 0


def cache_state(cache: DynamicCache, prompt: int, step: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    layer = cache.layers[0]
    conv = layer.conv_states[0].detach().cpu().contiguous()
    recurrent = layer.recurrent_states[0].detach().cpu().contiguous()
    tensors = {f"p{prompt}_s{step}_cache_conv": conv, f"p{prompt}_s{step}_cache_recurrent": recurrent}
    meta = {
        "prompt": prompt, "step": step, "layer": 0, "state_index": 0,
        "record_past": bool(layer.record_past), "has_previous_state": bool(layer.has_previous_state[0]),
        "conv": {"dtype": str(conv.dtype), "shape": list(conv.shape), "bytes": conv.numel() * conv.element_size(), "sha256": tensor_sha(conv)},
        "recurrent": {"dtype": str(recurrent.dtype), "shape": list(recurrent.shape), "bytes": recurrent.numel() * recurrent.element_size(), "sha256": tensor_sha(recurrent)},
    }
    if not finite(conv) or not finite(recurrent):
        raise RuntimeError("non-finite cache state")
    return meta, tensors


def capture_official(layer: Qwen3NextDecoderLayer, hidden: torch.Tensor, cache: DynamicCache) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    def save(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().cpu().contiguous()
        return hook
    for name, module in (("input_norm", layer.input_layernorm), ("gdn", layer.linear_attn),
                         ("post_norm", layer.post_attention_layernorm), ("router", layer.mlp.gate),
                         ("experts", layer.mlp.experts), ("shared", layer.mlp.shared_expert),
                         ("shared_gate", layer.mlp.shared_expert_gate)):
        hooks.append(module.register_forward_hook(save(name)))
    try:
        empty = torch.empty(0, dtype=torch.bfloat16)
        output = layer(hidden, position_embeddings=(empty, empty), attention_mask=None, past_key_values=cache)
    finally:
        for hook in hooks:
            hook.remove()
    return output.detach().cpu().contiguous(), captured


def router_artifacts(layer: Qwen3NextDecoderLayer, normed: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = normed.reshape(-1, HIDDEN)
    native_logits, native_weights, ids = layer.mlp.gate(flat)
    fp32_logits = F.linear(flat.float(), layer.mlp.gate.weight.float())
    probs = torch.softmax(fp32_logits, dim=-1)
    pre_weights, fp32_ids = torch.topk(probs, TOP_K, dim=-1)
    pre_weights = pre_weights / pre_weights.sum(dim=-1, keepdim=True)
    if not torch.equal(ids, fp32_ids) or not torch.equal(native_weights, pre_weights.to(torch.bfloat16)):
        raise RuntimeError("router recomputation mismatch")
    return {"router_logits_native_bf16": native_logits.cpu(), "router_logits_fp32": fp32_logits.cpu(),
            "router_probs_fp32": probs.cpu(), "router_ids": ids.cpu(),
            "router_weights_precast_fp32": pre_weights.cpu(), "router_weights_native_bf16": native_weights.cpu()}


def manual_moe(layer: Qwen3NextDecoderLayer, normed: torch.Tensor, routes: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = normed.reshape(-1, HIDDEN)
    ids = routes["router_ids"].to(torch.long)
    weights = routes["router_weights_native_bf16"]
    rows = x.shape[0]
    gate_raw = torch.zeros((rows, TOP_K, INTER), dtype=torch.bfloat16)
    up_raw = torch.zeros_like(gate_raw); down_raw = torch.zeros((rows, TOP_K, HIDDEN), dtype=torch.bfloat16)
    routed = torch.zeros_like(x)
    for expert in sorted(set(int(v) for v in ids.reshape(-1).tolist())):
        positions = torch.nonzero(ids == expert, as_tuple=False)
        token = positions[:, 0]; rank = positions[:, 1]
        fused = F.linear(x[token], layer.mlp.experts.gate_up_proj[expert])
        gate, up = fused.chunk(2, dim=-1)
        down = F.linear(F.silu(gate) * up, layer.mlp.experts.down_proj[expert])
        gate_raw[token, rank] = gate; up_raw[token, rank] = up; down_raw[token, rank] = down
        routed.index_add_(0, token, down * weights[token, rank, None])
    shared_raw = layer.mlp.shared_expert(x)
    shared_gate = torch.sigmoid(layer.mlp.shared_expert_gate(x))
    shared_gated = shared_raw * shared_gate
    return (routed + shared_gated).reshape(normed.shape), {
        "manual_gate": gate_raw, "manual_up": up_raw, "manual_down": down_raw,
        "manual_routed": routed, "manual_shared_raw": shared_raw,
        "manual_shared_gate": shared_gate, "manual_shared_gated": shared_gated,
    }


def quantize_matrix(value: torch.Tensor) -> tuple[bytes, bytes]:
    rows, columns = value.shape
    work = value.float().reshape(rows, columns // GROUP, GROUP)
    maximum = work.abs().amax(dim=-1, keepdim=True)
    temporary = torch.where(maximum > 0, maximum / 15, torch.ones_like(maximum))
    codes = torch.round(work / temporary).clamp(-15, 15).to(torch.int8).reshape(rows, columns)
    fields = (codes.numpy().astype(np.int16) + 15).astype(np.uint64).reshape(rows, columns // 8, 8)
    shifts = (np.arange(8, dtype=np.uint64) * 5).reshape(1, 1, 8)
    words = np.bitwise_or.reduce(fields << shifts, axis=-1)
    byte_shifts = (np.arange(5, dtype=np.uint64) * 8).reshape(1, 1, 5)
    packed = ((words[..., None] >> byte_shifts) & 255).astype(np.uint8).tobytes()
    scales = temporary.squeeze(-1).to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    if len(packed) != CODE_BYTES or len(scales) != SCALE_BYTES or b"\xff\xff\xff\xff\xff" in packed:
        raise RuntimeError("Q5 payload contract")
    return packed, scales


def matrix_record(value: torch.Tensor, expert: int, projection: int) -> tuple[bytes, dict[str, Any]]:
    rows, columns = value.shape
    packed, scales = quantize_matrix(value)
    crc = zlib.crc32(scales, zlib.crc32(packed)) & 0xFFFFFFFF
    header = struct.pack(HEADER_FORMAT, b"SQ5M", 1, 0, expert, projection, 5, rows, columns,
                         GROUP, len(packed), len(scales), crc, bytes(28))
    record = header + packed + scales + bytes(PADDING_BYTES)
    if len(record) != MATRIX_BYTES:
        raise RuntimeError("matrix record byte contract")
    return record, {"source_sha256": tensor_sha(value), "codes_sha256": hashlib.sha256(packed).hexdigest(),
                    "scales_sha256": hashlib.sha256(scales).hexdigest(), "record_sha256": hashlib.sha256(record).hexdigest(),
                    "crc32": crc, "rows": rows, "columns": columns}


def stream_records(layer: Qwen3NextDecoderLayer, run_index: int, peak: dict[str, int]) -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if run_index == 1 and (RECORD_ARTIFACT.exists() or RECORD_MANIFEST.exists()):
        raise FileExistsError("refusing to overwrite T0-R3 record artifact")
    if run_index == 2 and (not RECORD_ARTIFACT.is_file() or not RECORD_MANIFEST.is_file()):
        raise FileNotFoundError("run 2 requires immutable run-1 record artifact")
    partial = RECORD_ARTIFACT.with_suffix(".bin.inprogress")
    if partial.exists():
        raise FileExistsError("stale inprogress artifact")
    manifest: dict[str, Any] = {"records": [], "format": "SQ5M biased q+15 little-order 8/5"}
    digest = hashlib.sha256(); compared = 0
    existing = RECORD_ARTIFACT.open("rb") if run_index == 2 else None
    output = partial.open("xb", buffering=8 * 2**20) if run_index == 1 else None
    try:
        for expert in range(513):
            if expert < EXPERTS:
                fused = layer.mlp.experts.gate_up_proj[expert]
                matrices = (fused[:INTER], fused[INTER:], layer.mlp.experts.down_proj[expert])
                names = (f"model.layers.0.mlp.experts.{expert}.gate_proj.weight",
                         f"model.layers.0.mlp.experts.{expert}.up_proj.weight",
                         f"model.layers.0.mlp.experts.{expert}.down_proj.weight")
            else:
                shared = layer.mlp.shared_expert
                matrices = (shared.gate_proj.weight, shared.up_proj.weight, shared.down_proj.weight)
                names = tuple(f"model.layers.0.mlp.shared_expert.{kind}_proj.weight" for kind in ("gate", "up", "down"))
            entry = {"expert": expert, "shared": expert == EXPERTS, "projections": []}
            for projection, (value, source_name) in enumerate(zip(matrices, names)):
                record, meta = matrix_record(value, expert, projection)
                meta.update({"projection": projection, "source_key": source_name})
                entry["projections"].append(meta); digest.update(record)
                if output is not None:
                    output.write(record)
                else:
                    observed = existing.read(MATRIX_BYTES)
                    if observed != record:
                        raise RuntimeError(f"second-run record mismatch expert={expert} projection={projection}")
                    compared += len(record)
                del record
            manifest["records"].append(entry)
            if expert % 16 == 0:
                gc.collect(); rss_guard(f"record_{expert}", peak)
        if output is not None:
            output.flush(); os.fsync(output.fileno())
        elif existing.read(1):
            raise RuntimeError("run-1 record artifact has trailing bytes")
    finally:
        if output is not None: output.close()
        if existing is not None: existing.close()
    manifest.update({"bytes": BANK_BYTES, "sha256": digest.hexdigest(), "second_run_compared_bytes": compared})
    if run_index == 1:
        if partial.stat().st_size != BANK_BYTES:
            raise RuntimeError("record artifact size mismatch")
        os.replace(partial, RECORD_ARTIFACT)
        RECORD_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        prior = json.loads(RECORD_MANIFEST.read_text(encoding="utf-8"))
        if compared != BANK_BYTES or digest.hexdigest() != prior["sha256"] or sha256(RECORD_ARTIFACT) != prior["sha256"]:
            raise RuntimeError("second-run bank replay mismatch")
    return manifest


def reference(run_index: int) -> dict[str, Any]:
    if not RUNNER_LOCK.is_file():
        raise RuntimeError("immutable runner lock missing")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != lock.get("runner_sha256") or sha256(PREREG) != lock.get("t0r3_sha256"):
        raise RuntimeError("runner lock mismatch")
    if psutil.virtual_memory().available < MIN_START_RAM or torch.cuda.is_initialized():
        raise RuntimeError("CPU/resource start gate")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    peak: dict[str, int] = {}; rss_guard("start", peak)
    inputs = locked_inputs(); header = inspect_safetensors_header()
    if not header["pass"]: raise RuntimeError(f"safetensors header gate: {header}")
    config = Qwen3NextConfig.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    layer, embedding, identities = load_official(config, peak)
    prompt_lock = json.loads(PROMPT_LOCK.read_text(encoding="utf-8"))
    token_ids = torch.tensor([row["token_ids"] for row in prompt_lock["prompts"]], dtype=torch.long)
    hidden = F.embedding(token_ids, embedding).to(torch.bfloat16); del embedding; gc.collect(); rss_guard("embedding", peak)
    raw: dict[str, torch.Tensor] = {"token_ids": token_ids, "embedding": hidden.cpu()}
    cache_rows = []; whole_outputs = []; step_outputs = []; whole_routes = []; step_routes = []
    state_metrics = []
    manual_ulp = []
    with torch.inference_mode():
        for prompt in range(4):
            whole_cache = DynamicCache(config=config)
            whole, whole_cap = capture_official(layer, hidden[prompt:prompt + 1], whole_cache)
            whole_router = router_artifacts(layer, whole_cap["post_norm"])
            manual, manual_raw = manual_moe(layer, whole_cap["post_norm"], whole_router)
            residual = hidden[prompt:prompt + 1] + whole_cap["gdn"]
            manual_layer = residual + manual
            ulp = max_bf16_ulp(manual_layer, whole)
            if ulp > 1: raise RuntimeError(f"manual MoE ULP gate prompt {prompt}: {ulp}")
            manual_ulp.append(ulp)
            whole_outputs.append(whole); whole_routes.append(whole_router)
            for name, value in {**whole_cap, **whole_router, **manual_raw,
                                "whole_layer_output": whole, "manual_layer_output": manual_layer}.items():
                raw[f"p{prompt}_whole_{name}"] = value.cpu().contiguous()

            step_cache = DynamicCache(config=config); prompt_steps = []; route_parts: dict[str, list[torch.Tensor]] = {}
            for step in range(16):
                output, captured = capture_official(layer, hidden[prompt:prompt + 1, step:step + 1], step_cache)
                routes = router_artifacts(layer, captured["post_norm"])
                prompt_steps.append(output)
                for name, value in routes.items(): route_parts.setdefault(name, []).append(value)
                meta, tensors = cache_state(step_cache, prompt, step); cache_rows.append(meta); raw.update(tensors)
                raw[f"p{prompt}_s{step}_layer_output"] = output
            joined = torch.cat(prompt_steps, dim=1); step_outputs.append(joined)
            joined_routes = {name: torch.cat(values, dim=0) for name, values in route_parts.items()}; step_routes.append(joined_routes)
            held = slice(8, 16); diff = whole[:, held].float() - joined[:, held].float()
            max_abs = float(diff.abs().max()); rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(whole[:, held].float()))
            ids_equal = torch.equal(whole_router["router_ids"].reshape(1,16,TOP_K)[:,held], joined_routes["router_ids"].reshape(1,16,TOP_K)[:,held])
            weight_diff = float((whole_router["router_weights_native_bf16"].reshape(1,16,TOP_K)[:,held].float()
                                 - joined_routes["router_weights_native_bf16"].reshape(1,16,TOP_K)[:,held].float()).abs().max())
            state_metrics.append({"prompt": prompt, "heldout_max_abs": max_abs, "heldout_relative_l2": rel_l2,
                                  "heldout_top10_exact": ids_equal, "heldout_route_weight_max_abs": weight_diff})
            if max_abs > .02 or rel_l2 > 1e-3 or not ids_equal or weight_diff > 2e-3:
                raise RuntimeError(f"whole/token-step gate prompt {prompt}: {state_metrics[-1]}")
    record_manifest = stream_records(layer, run_index, peak)
    del layer; gc.collect(); rss_guard("after_records", peak)
    raw_path = RUN_DIR / f"t0r3_run_{run_index}_raw.safetensors"
    result_path = RUN_DIR / f"t0r3_run_{run_index}_result.json"
    if raw_path.exists() or result_path.exists(): raise FileExistsError("refusing to overwrite T0-R3 run outputs")
    save_file({name: value.contiguous() for name, value in raw.items()}, raw_path)
    result = {"kind": "port80b_t0r3_official_layer0_reference", "run_index": run_index,
              "status": "run1_complete_pending_clean_replay" if run_index == 1 else "t0r3_reference_candidate_pass",
              "inputs": inputs, "header": header, "source_tensor_sha256": identities,
              "state_equivalence": state_metrics, "manual_moe_max_bf16_ulp": manual_ulp,
              "cache_state_schema": cache_rows, "record_artifact": record_manifest,
              "raw_artifact": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
              "raw_artifact_sha256": sha256(raw_path), "resources": peak,
              "cuda_initialized_after": torch.cuda.is_initialized(),
              "claim_boundary": "Official CPU layer-0 reference only; no physical transport, full-depth logits or quality claim."}
    if run_index == 2:
        prior = json.loads((RUN_DIR / "t0r3_run_1_result.json").read_text(encoding="utf-8"))
        prior_raw = safe_open(RUN_DIR / "t0r3_run_1_raw.safetensors", framework="pt", device="cpu")
        current_raw = safe_open(raw_path, framework="pt", device="cpu")
        try:
            keys = list(prior_raw.keys())
            replay_equal = keys == list(current_raw.keys()) and all(torch.equal(prior_raw.get_tensor(k), current_raw.get_tensor(k)) for k in keys)
        finally:
            del prior_raw, current_raw
        result["clean_replay"] = {"raw_tensors_bitwise_equal": replay_equal,
                                  "record_sha_equal": prior["record_artifact"]["sha256"] == record_manifest["sha256"]}
        if not all(result["clean_replay"].values()): raise RuntimeError("clean-process replay mismatch")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def smoke() -> dict[str, Any]:
    inputs = locked_inputs(); header = inspect_safetensors_header(); codec = codec_sentinel()
    config = Qwen3NextConfig.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    layer = make_layer(config)
    parameters = sum(p.numel() for p in layer.parameters())
    meta_contract = all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in layer.parameters())
    del layer; gc.collect()
    return {"kind": "port80b_t0r3_cpu_smoke", "pass": header["pass"] and codec["pass"] and meta_contract,
            "inputs": inputs, "header": header, "codec": codec,
            "meta_to_cpu_bf16_parameter_elements": parameters, "meta_to_cpu_bf16_contract": meta_contract,
            "cuda_initialized": torch.cuda.is_initialized(),
            "physical_actions": {"network": False, "reference_forward": False, "bank_build": False,
                                 "gpu": False, "host_registration": False, "registry_edit": False}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "reference"), required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2))
    parser.add_argument("--acknowledge-reference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    started = time.perf_counter()
    if args.phase == "smoke":
        payload = smoke()
    else:
        if args.run_index is None or args.acknowledge_reference != ACK:
            raise SystemExit("reference phase requires run index and exact acknowledgement")
        payload = reference(args.run_index)
    payload["wall_seconds"] = time.perf_counter() - started
    print(json.dumps(payload if args.phase == "smoke" else {"status": payload["status"], "wall_seconds": payload["wall_seconds"]}, indent=2))

