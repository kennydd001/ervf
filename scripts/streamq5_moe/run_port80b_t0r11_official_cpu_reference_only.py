#!/usr/bin/env python3
"""PORT80B T0-R4 official layer-0 CPU reference runner.

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
import uuid
import subprocess
from datetime import datetime, timezone
from typing import Any
import zlib

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["USE_HUB_KERNELS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache
from transformers.models.qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next import modeling_qwen3_next as qwen_source
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDecoderLayer


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
RUN_DIR = ROOT / "reports" / "runs" / "streamq5_moe" / "port80b_t0r11_official_cpu_reference_only"
SNAPSHOT = (Path.home() / ".cache" / "huggingface" / "hub" /
            "models--Qwen--Qwen3-Coder-Next" / "snapshots" /
            "a19358a7659bd1f564300250ee189120c49a562f")
SHARD = SNAPSHOT / "model-00001-of-00040.safetensors"
INDEX = SNAPSHOT / "model.safetensors.index.json"
PREREG = REPORTS / "PORT80B_T0R4_OFFICIAL_LAYER0_REFERENCE_PREREGISTRATION_2026-08-13.md"
R7_PREREG = REPORTS / "PORT80B_T0R7_OFFICIAL_CPU_ROUTE_REPRO_PREREGISTRATION_2026-08-13.md"
R11_PREREG = REPORTS / "PORT80B_T0R11_OFFICIAL_CPU_REFERENCE_ONLY_PREREGISTRATION_2026-08-13.md"
PROMPT_GENERATION_LOCK = REPORTS / "port80b_t0r11_prompt_generation_lock.json"
PROMPT_LOCK = REPORTS / "port80b_t0r11_prompt_lock.json"
ENV_LOCK = REPORTS / "port80b_t0r1_reference_environment_lock.json"
DEPENDENCY_LOCK = REPORTS / "port80b_t0r4_dependency_execution_lock.json"
RUNNER_LOCK = REPORTS / "port80b_t0r11_runner_lock.json"
VERIFIER = ROOT / "scripts" / "streamq5_moe" / "verify_port80b_t0r11_official_cpu_reference_only.py"
VERIFIER_LOCK = REPORTS / "port80b_t0r11_verifier_lock.json"
INVOCATION_LEDGER = RUN_DIR / "invocation_ledger"
ACK = "T0R11_OFFICIAL_CPU_CAPTURE_AFTER_PREFLIGHT"

REVISION = "a19358a7659bd1f564300250ee189120c49a562f"
SHARD_BYTES = 3_999_619_288
SHARD_SHA = "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a"
INDEX_SHA = "e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b"
R4_SHA = "4c5da965e47e11e9ff36594e15387c48a1a2943aa0d9cbdc3b9b271418400a03"
DEPENDENCY_SHA = "1d08457aded09f139d25af84ba778d8e275ab5ff71967a3dc8b9a7452e6d2fae"
PROMPT_SHA = "11971c9d6a6ee26b0da55ce9fd2d162d967704f0811df0c1965dee69b1827be0"
R7_PREREG_SHA = "ccafeeb205efe67591518563443d4d9c802cae7bc809ecf35abcd47c7264dc98"
ENV_SHA = "eb31d4e0c1f6a806434ea8a20b6b00200781a89ed9f91e485aad0e3583c0f455"

GROUP = 128
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
CODE_BYTES = 655_360
SCALE_BYTES = 16_384
PADDING_BYTES = 4_032
MATRIX_BYTES = 675_840
EXPERT_BYTES = 2_027_520
EXPERTS = 512
TOP_K = 10
HIDDEN = 2048
INTER = 512
MAX_RSS = 12 * 2**30
MIN_START_RAM = 16 * 2**30
MIN_RESERVE = 2 * 2**30
MAX_PROJECTED_STEADY = int(10.5 * 2**30)
FAILURE_STATE: dict[str, Any] = {"stage": "not_started", "resources": {}}


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


def tensor_manifest(values: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    manifest = {}
    for name, value in sorted(values.items()):
        if not finite(value):
            raise RuntimeError(f"non-finite retained raw tensor: {name}")
        manifest[name] = {"semantic_key": name, "dtype": str(value.dtype), "shape": list(value.shape),
                          "bytes": value.numel() * value.element_size(), "sha256": tensor_sha(value)}
    return manifest


def runtime_dependencies() -> dict[str, Any]:
    lock = json.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))
    base = ROOT / ".venv-next-ref" / "Lib" / "site-packages" / "transformers"
    source_hashes = {name: sha256(base / name) for name in lock["transformers_sources"]}
    if source_hashes != lock["transformers_sources"]:
        raise RuntimeError("actual dependency source hash mismatch")
    import transformers, safetensors, tokenizers
    versions = {"transformers": transformers.__version__, "torch": torch.__version__,
                "safetensors": safetensors.__version__, "tokenizers": tokenizers.__version__, "numpy": np.__version__}
    expected = {"transformers": "5.15.0", "torch": "2.12.1+cu132", "safetensors": "0.8.0",
                "tokenizers": "0.22.2", "numpy": "2.2.6"}
    if versions != expected:
        raise RuntimeError(f"package version mismatch: {versions}")
    return {"source_sha256": source_hashes, "package_versions": versions}


def rss_guard(stage: str, peak: dict[str, int]) -> None:
    memory = psutil.Process().memory_info()
    rss = memory.rss
    windows_peak = getattr(memory, "peak_wset", rss)
    available = psutil.virtual_memory().available
    peak["peak_rss_bytes"] = max(peak.get("peak_rss_bytes", 0), rss)
    peak["windows_peak_working_set_bytes"] = max(peak.get("windows_peak_working_set_bytes", 0), windows_peak)
    peak["minimum_available_ram_bytes"] = min(peak.get("minimum_available_ram_bytes", available), available)
    FAILURE_STATE["stage"] = stage; FAILURE_STATE["resources"] = dict(peak)
    if rss > MAX_RSS or windows_peak > MAX_RSS or available < MIN_RESERVE:
        raise MemoryError(f"resource gate at {stage}: rss={rss}, peak_wset={windows_peak}, available={available}")


def verify_prompt_lock_before_shard() -> dict[str, Any]:
    if not PROMPT_LOCK.is_file() or sha256(PROMPT_LOCK) != PROMPT_SHA:
        raise RuntimeError("prompt lock mismatch")
    lock = json.loads(PROMPT_LOCK.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    rows = []
    for item in lock["prompts"]:
        ids = tokenizer(item["utf8_text"], add_special_tokens=False)["input_ids"][:16]
        packed = b"".join(int(value).to_bytes(4, "little") for value in ids)
        got = hashlib.sha256(packed).hexdigest()
        if ids != item["token_ids"] or got != item["token_ids_le_u32_sha256"]:
            raise RuntimeError(f"tokenizer replay mismatch: {item['domain']}")
        rows.append({"domain": item["domain"], "token_ids_le_u32_sha256": got})
    return {"prompt_lock_sha256": PROMPT_SHA, "rows": rows}


def locked_inputs() -> dict[str, Any]:
    prompt_replay = verify_prompt_lock_before_shard()
    wanted = {PREREG: R4_SHA, R7_PREREG: R7_PREREG_SHA, PROMPT_LOCK: PROMPT_SHA,
              ENV_LOCK: ENV_SHA, DEPENDENCY_LOCK: DEPENDENCY_SHA, INDEX: INDEX_SHA}
    observed = {str(path): sha256(path) if path.is_file() else None for path in wanted}
    if any(observed[str(path)] != expected for path, expected in wanted.items()):
        raise RuntimeError(f"immutable input mismatch: {observed}")
    if SHARD.stat().st_size != SHARD_BYTES or sha256(SHARD) != SHARD_SHA:
        raise RuntimeError("official shard size/hash mismatch")
    observed["prompt_replay"] = prompt_replay
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
    layer0_subset = {"model.embed_tokens.weight"} | {key for key in expected_keys if key.startswith("model.layers.0.")}
    extra = sorted(expected_keys - layer0_subset)
    result = {
        "header_bytes": header_len, "data_bytes": data_bytes, "tensor_entries": len(entries),
        "expected_entries": len(expected_keys), "all_bf16": all(v.get("dtype") == "BF16" for v in entries.values()),
        "keys_exact": set(entries) == expected_keys,
        "layer0_embedding_subset_exact": set(expected) == layer0_subset,
        "extra_shard1_keys": extra, "extra_shard1_key_count": len(extra),
        "offsets_nonoverlap": not overlap,
        "offsets_in_range": not failures, "shape_failures": failures[:20],
    }
    result["pass"] = (result["tensor_entries"] == 1567 and result["expected_entries"] == 1567
                      and result["all_bf16"] and result["keys_exact"] and result["layer0_embedding_subset_exact"]
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


def make_layer(config: Qwen3NextConfig, materialize: bool = True) -> Qwen3NextDecoderLayer:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            layer = Qwen3NextDecoderLayer(config, 0)
    finally:
        torch.set_default_dtype(previous)
    if materialize:
        layer.to_empty(device="cpu")
    if any(parameter.dtype != torch.bfloat16 for parameter in layer.parameters()):
        raise RuntimeError("meta-to-empty materialized non-BF16 parameters")
    return layer.eval()


def load_official(config: Qwen3NextConfig, peak: dict[str, int]) -> tuple[Qwen3NextDecoderLayer, torch.Tensor, dict[str, str]]:
    layer = make_layer(config, materialize=True)
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
                    if (tensor_sha(parameter[expert, :INTER]) != identities[gate_name]
                            or tensor_sha(parameter[expert, INTER:]) != identities[up_name]):
                        raise RuntimeError(f"packed gate/up byte mismatch expert={expert}")
                    del gate, up
                    if expert % 32 == 0:
                        rss_guard(f"load_gate_up_{expert}", peak)
            elif name == "mlp.experts.down_proj":
                for expert in range(EXPERTS):
                    key = f"model.layers.0.mlp.experts.{expert}.down_proj.weight"
                    value = source.get_tensor(key)
                    identities[key] = tensor_sha(value); parameter[expert].copy_(value)
                    if tensor_sha(parameter[expert]) != identities[key]:
                        raise RuntimeError(f"packed down byte mismatch expert={expert}")
                    del value
                    if expert % 32 == 0:
                        rss_guard(f"load_down_{expert}", peak)
            else:
                key = f"model.layers.0.{name}"
                value = source.get_tensor(key)
                if tuple(value.shape) != tuple(parameter.shape) or value.dtype != torch.bfloat16:
                    raise RuntimeError(f"parameter mismatch {key}")
                identities[key] = tensor_sha(value); parameter.copy_(value)
                if tensor_sha(parameter) != identities[key]:
                    raise RuntimeError(f"loaded parameter byte mismatch {key}")
                del value
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
    conv = layer.conv_states[0].detach().cpu().contiguous().clone()
    recurrent = layer.recurrent_states[0].detach().cpu().contiguous().clone()
    if list(conv.shape) != [1, 8192, 4] or conv.dtype != torch.bfloat16:
        raise RuntimeError(f"cache conv schema mismatch: {conv.dtype} {list(conv.shape)}")
    if list(recurrent.shape) != [1, 32, 128, 128] or recurrent.dtype != torch.float32:
        raise RuntimeError(f"cache recurrent schema mismatch: {recurrent.dtype} {list(recurrent.shape)}")
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
            captured[name] = value.detach().cpu().contiguous().clone()
        return hook
    def save_gate(_module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 3: raise RuntimeError("official gate tuple contract")
        captured["router_input"] = inputs[0].detach().cpu().contiguous().clone()
        for name,value in zip(("official_router_logits","official_router_weights","official_router_ids"),output):
            captured[name] = value.detach().cpu().contiguous().clone()
    for name, module in (("input_norm", layer.input_layernorm), ("gdn", layer.linear_attn),
                         ("post_norm", layer.post_attention_layernorm),
                         ("experts", layer.mlp.experts), ("shared", layer.mlp.shared_expert),
                         ("shared_gate", layer.mlp.shared_expert_gate)):
        hooks.append(module.register_forward_hook(save(name)))
    hooks.append(layer.mlp.gate.register_forward_hook(save_gate))
    try:
        empty = torch.empty(0, dtype=torch.bfloat16)
        output = layer(hidden, position_embeddings=(empty, empty), attention_mask=None, past_key_values=cache)
    finally:
        for hook in hooks:
            hook.remove()
    return output.detach().cpu().contiguous().clone(), captured


def router_artifacts(layer: Qwen3NextDecoderLayer, captured: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    native_logits = captured["official_router_logits"]; native_weights = captured["official_router_weights"]; ids = captured["official_router_ids"]
    second_logits, second_weights, second_ids = layer.mlp.gate(captured["router_input"])
    if not (torch.equal(native_logits, second_logits) and torch.equal(native_weights, second_weights) and torch.equal(ids, second_ids)):
        raise RuntimeError("direct official router tuple differs from diagnostic second call")
    fp32_logits = native_logits.float()  # exact widening; never an FP32 re-matmul
    probs = torch.softmax(fp32_logits, dim=-1)
    pre_weights, fp32_ids = torch.topk(probs, TOP_K, dim=-1)
    pre_weights = pre_weights / pre_weights.sum(dim=-1, keepdim=True)
    top11 = torch.topk(probs, TOP_K + 1, dim=-1).values
    margins = top11[:, 9] - top11[:, 10]
    boundary = top11[:, 9:10]
    boundary_tie_mask = probs == boundary
    selected_boundary_mask = torch.gather(boundary_tie_mask, 1, ids)
    top11_ids = torch.topk(probs, TOP_K + 1, dim=-1).indices
    top11_logits = torch.gather(native_logits, 1, top11_ids)
    fp32_sum_error = (pre_weights.sum(dim=-1) - 1).abs()
    bf16_sum_error = (native_weights.float().sum(dim=-1) - 1).abs()
    route_values_ok = bool((pre_weights > 0).all() and (native_weights > 0).all()
                           and torch.isfinite(pre_weights).all() and torch.isfinite(native_weights.float()).all()
                           and (pre_weights[:, :-1] >= pre_weights[:, 1:]).all()
                           and (native_weights[:, :-1] >= native_weights[:, 1:]).all()
                           and (ids >= 0).all() and (ids < EXPERTS).all()
                           and all(torch.unique(row).numel() == TOP_K for row in ids))
    if (not torch.equal(ids, fp32_ids) or not torch.equal(native_weights, pre_weights.to(torch.bfloat16))
            or float(fp32_sum_error.max()) > 2**-20
            or float(bf16_sum_error.max()) > 0.00390720367431640625 or not route_values_ok):
        raise RuntimeError("router recomputation mismatch")
    return {"router_logits_native_bf16": native_logits.cpu(), "router_logits_fp32": fp32_logits.cpu(),
            "router_probs_fp32": probs.cpu(), "router_ids": ids.cpu(),
            "router_top10_top11_margin_fp32": margins.cpu(),
            "router_boundary_tie_mask": boundary_tie_mask.cpu(),
            "router_selected_boundary_mask": selected_boundary_mask.cpu(),
            "router_top11_ids": top11_ids.cpu(), "router_top11_native_bf16_logits": top11_logits.cpu(),
            "router_weights_precast_fp32": pre_weights.cpu(), "router_weights_native_bf16": native_weights.cpu()}


def write_invocation_ledger(run_index: int) -> dict[str, Any]:
    INVOCATION_LEDGER.mkdir(parents=True, exist_ok=True)
    expected=["--phase","capture","--run-index",str(run_index),"--acknowledge-reference",ACK]
    if sys.argv[1:] != expected: raise RuntimeError("exact capture argv mismatch")
    process = psutil.Process(); created = datetime.now(timezone.utc).isoformat(); nonce = uuid.uuid4().hex
    entry = {"capture":run_index,"pid":os.getpid(),"parent_pid":os.getppid(),"process_create_time_unix":process.create_time(),
             "start_utc":created,"start_perf_counter_ns":time.perf_counter_ns(),"commandline":sys.argv,"nonce":nonce,
             "emitted_pre_model":True,"runner_sha256":sha256(Path(__file__))}
    path=INVOCATION_LEDGER/f"capture_{run_index}_pre_model.json"
    if path.exists(): raise FileExistsError("invocation ledger exists")
    if run_index==1:
        if any(INVOCATION_LEDGER.glob("capture_2_*")): raise RuntimeError("capture ordering violation")
    else:
        prior=json.loads((INVOCATION_LEDGER/"capture_1_pre_model.json").read_text())
        if entry["pid"] == prior["pid"] or entry["process_create_time_unix"] == prior["process_create_time_unix"] or entry["nonce"] == prior["nonce"]:
            raise RuntimeError("capture2 is not a distinct invocation")
        entry["capture1_scientific_outputs_read_permission"] = False
        entry["permitted_prior_inputs"] = [str(INVOCATION_LEDGER/"capture_1_pre_model.json")]
    with path.open("x",encoding="utf-8") as f: json.dump(entry,f,indent=2);f.write("\n")
    return {**entry,"path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":sha256(path)}


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



def execution_lock_check() -> dict[str, Any]:
    if not RUNNER_LOCK.is_file() or not VERIFIER.is_file() or not VERIFIER_LOCK.is_file():
        raise RuntimeError("R11 execution provenance file missing")
    lock = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    expected = {
        "runner_sha256": sha256(Path(__file__)), "verifier_sha256": sha256(VERIFIER),
        "verifier_lock_sha256": sha256(VERIFIER_LOCK), "t0r7_prereg_sha256": sha256(R7_PREREG),
        "t0r11_prereg_sha256": sha256(R11_PREREG), "prompt_generation_lock_sha256": sha256(PROMPT_GENERATION_LOCK),
        "prompt_generator_sha256": sha256(ROOT / json.loads(PROMPT_GENERATION_LOCK.read_text())["generator"]),
        "prompt_lock_sha256": sha256(PROMPT_LOCK), "t0r4_sha256": sha256(PREREG),
        "dependency_lock_sha256": sha256(DEPENDENCY_LOCK),
        "environment_lock_sha256": sha256(ENV_LOCK),
    }
    mismatched = {key: {"lock": lock.get(key), "actual": value} for key, value in expected.items() if lock.get(key) != value}
    if mismatched: raise RuntimeError(f"R11 execution lock mismatch: {mismatched}")
    generation=json.loads(PROMPT_GENERATION_LOCK.read_text());generator=ROOT/generation["generator"]
    if sha256(generator)!=generation["generator_sha256"] or generation["no_rejection"] is not True or generation["no_output_dependent_filtering"] is not True:
        raise RuntimeError("prompt generator source/protocol mismatch")
    completed=subprocess.run([sys.executable,str(generator)],capture_output=True,text=True,check=True,timeout=60)
    regenerated=json.loads(completed.stdout);canonical=json.loads(PROMPT_LOCK.read_text())
    if regenerated["prompts"] != canonical["prompts"] or regenerated["no_output_dependent_filtering"] is not True:
        raise RuntimeError("canonical prompt regeneration mismatch")
    return {"pass": True, "bindings": expected}


def premodel_target_guard(run_index: int) -> None:
    targets=[RUN_DIR/f"t0r11_capture_{run_index}_raw.safetensors",RUN_DIR/f"t0r11_capture_{run_index}_result.json",
             RUN_DIR/f"t0r11_capture_{run_index}_failure.json",INVOCATION_LEDGER/f"capture_{run_index}_pre_model.json"]
    if any(path.exists() for path in targets):raise FileExistsError(f"pre-model target exists: {[str(p) for p in targets if p.exists()]}")
    if run_index==2 and not (INVOCATION_LEDGER/"capture_1_pre_model.json").is_file():raise FileNotFoundError("capture2 requires capture1 ledger")

def ledger_schema_unit_test() -> dict[str, Any]:
    argv=["runner.py","--phase","capture","--run-index","1","--acknowledge-reference",ACK]
    required={"capture","pid","parent_pid","process_create_time_unix","start_utc","start_perf_counter_ns","commandline","nonce","emitted_pre_model","runner_sha256"}
    sample={"capture":1,"pid":2,"parent_pid":1,"process_create_time_unix":1.0,"start_utc":"x","start_perf_counter_ns":1,"commandline":argv,"nonce":"0"*32,"emitted_pre_model":True,"runner_sha256":"0"*64}
    return {"pass":set(sample)==required and argv[1:]==["--phase","capture","--run-index","1","--acknowledge-reference",ACK]}


def reference(run_index: int) -> dict[str, Any]:
    execution_lock_check()
    premodel_target_guard(run_index)
    invocation = write_invocation_ledger(run_index)
    if psutil.virtual_memory().available < MIN_START_RAM or torch.cuda.is_initialized():
        raise RuntimeError("CPU/resource start gate")
    wanted_affinity = json.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))["runtime"]["process_affinity"]
    process = psutil.Process(); process.cpu_affinity(wanted_affinity)
    if process.cpu_affinity() != wanted_affinity:
        raise RuntimeError("process affinity mismatch")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = True; torch.set_flush_denormal(False)
    denormal_preserved = bool(torch.tensor([1.0e-45], dtype=torch.float32).item() != 0.0)
    dependencies = runtime_dependencies()
    runtime_contract = {
        "affinity": process.cpu_affinity(), "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled, "flush_denormal": False,
        "flush_denormal_nonzero_subnormal_probe": denormal_preserved,
        "cpu_identity": os.environ.get("PROCESSOR_IDENTIFIER"),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "autocast_cpu_enabled": torch.is_autocast_enabled("cpu"),
        "cuda_initialized": torch.cuda.is_initialized(),
        "resolved_callables": {name: {"module": getattr(getattr(qwen_source, name), "__module__", None),
                                      "name": getattr(getattr(qwen_source, name), "__name__", None)}
                               for name in ("causal_conv1d_fn", "causal_conv1d_update",
                                            "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")},
        "rmsnorm_gated_callable": {"module": qwen_source.Qwen3NextRMSNormGated.forward.__module__,
                                    "name": qwen_source.Qwen3NextRMSNormGated.forward.__name__},
        "experts_callable": {"module": qwen_source.Qwen3NextExperts.forward.__module__,
                             "name": qwen_source.Qwen3NextExperts.forward.__name__},
    }
    if (runtime_contract["torch_threads"] != 1 or runtime_contract["torch_interop_threads"] != 1
            or not runtime_contract["deterministic_algorithms"] or runtime_contract["cuda_initialized"]
            or runtime_contract["autocast_cpu_enabled"] or not denormal_preserved):
        raise RuntimeError("runtime contract mismatch")
    dep_runtime = json.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))["runtime"]
    if (runtime_contract["affinity"] != dep_runtime["process_affinity"]
            or runtime_contract["cpu_identity"] != dep_runtime["cpu_identity"]
            or runtime_contract["torch_cpu_capability"] != dep_runtime["torch_cpu_capability"]):
        raise RuntimeError("runtime identity/capability mismatch")
    peak: dict[str, int] = {}; rss_guard("start", peak)
    inputs = locked_inputs(); header = inspect_safetensors_header()
    if not header["pass"]: raise RuntimeError(f"safetensors header gate: {header}")
    config = Qwen3NextConfig.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    projected = 3_919_393_152 + 3_919_393_152 + 1_040_117_760 + 288_358_400
    if projected > MAX_PROJECTED_STEADY:
        raise MemoryError("projected steady working set exceeds 10.5 GiB")
    layer, embedding, identities = load_official(config, peak)
    prompt_lock = json.loads(PROMPT_LOCK.read_text(encoding="utf-8"))
    token_ids = torch.tensor([row["token_ids"] for row in prompt_lock["prompts"]], dtype=torch.long)
    hidden = F.embedding(token_ids, embedding).to(torch.bfloat16); del embedding; gc.collect(); rss_guard("embedding", peak)
    raw: dict[str, torch.Tensor] = {"token_ids": token_ids, "embedding": hidden.cpu()}
    cache_rows = []; whole_outputs = []; whole_routes = []
    state_metrics = []
    manual_ulp = []
    with torch.inference_mode():
        if not torch.is_inference_mode_enabled() or torch.is_autocast_enabled("cpu"):
            raise RuntimeError("inference/autocast runtime contract mismatch")
        runtime_contract["inference_mode_inside_compute"] = torch.is_inference_mode_enabled()
        runtime_contract["autocast_cpu_inside_compute"] = torch.is_autocast_enabled("cpu")
        for prompt in range(4):
            whole_cache = DynamicCache(config=config)
            whole, whole_cap = capture_official(layer, hidden[prompt:prompt + 1], whole_cache)
            whole_router = router_artifacts(layer, whole_cap)
            manual, manual_raw = manual_moe(layer, whole_cap["post_norm"], whole_router)
            residual = hidden[prompt:prompt + 1] + whole_cap["gdn"]
            raw[f"p{prompt}_whole_post_attention_residual"] = residual.cpu().contiguous()
            manual_layer = residual + manual
            ulp = max_bf16_ulp(manual_layer, whole)
            if ulp > 1: raise RuntimeError(f"manual MoE ULP gate prompt {prompt}: {ulp}")
            manual_ulp.append(ulp)
            whole_outputs.append(whole); whole_routes.append(whole_router)
            whole_meta, whole_tensors = cache_state(whole_cache, prompt, 16)
            raw[f"p{prompt}_whole_cache_conv"] = whole_tensors[f"p{prompt}_s16_cache_conv"]
            raw[f"p{prompt}_whole_cache_recurrent"] = whole_tensors[f"p{prompt}_s16_cache_recurrent"]
            for name, value in {**whole_cap, **whole_router, **manual_raw,
                                "whole_layer_output": whole, "manual_layer_output": manual_layer}.items():
                raw[f"p{prompt}_whole_{name}"] = value.cpu().contiguous()

            prefix_equal = []
            for length in range(1, 17):
                prefix_cache = DynamicCache(config=config)
                output, _captured = capture_official(layer, hidden[prompt:prompt + 1, :length], prefix_cache)
                final = output[:, -1:].contiguous()
                equal = torch.equal(final, whole[:, length - 1:length])
                prefix_equal.append(equal)
                meta, tensors = cache_state(prefix_cache, prompt, length); cache_rows.append(meta); raw.update(tensors)
                raw[f"p{prompt}_prefix{length}_final_output"] = final
                if not equal: raise RuntimeError(f"fresh-cache prefix mismatch prompt={prompt} length={length}")
                if length==16 and (not torch.equal(raw[f"p{prompt}_whole_cache_conv"],tensors[f"p{prompt}_s16_cache_conv"])
                                   or not torch.equal(raw[f"p{prompt}_whole_cache_recurrent"],tensors[f"p{prompt}_s16_cache_recurrent"])):
                    raise RuntimeError(f"whole/prefix16 cache state mismatch prompt={prompt}")
            state_metrics.append({"prompt": prompt, "fresh_cache_prefixes": 16,
                                  "all_final_outputs_bitwise_equal": all(prefix_equal)})
    raw_path = RUN_DIR / f"t0r11_capture_{run_index}_raw.safetensors"
    result_path = RUN_DIR / f"t0r11_capture_{run_index}_result.json"
    raw_manifest = tensor_manifest(raw)
    min_margin = min(float(value["router_top10_top11_margin_fp32"].min()) for value in whole_routes)
    del layer; gc.collect(); rss_guard("after_reference", peak)
    save_file({name: value.contiguous() for name, value in raw.items()}, raw_path)
    rss_guard("after_raw_serialization", peak)
    result = {"kind": "port80b_t0r11_official_cpu_reference_only", "run_index": run_index,
              "status": "capture_complete_pending_independent_compare",
              "runner_sha256": sha256(Path(__file__)), "runner_lock_sha256": sha256(RUNNER_LOCK),
              "verifier_sha256": sha256(VERIFIER), "verifier_lock_sha256": sha256(VERIFIER_LOCK),
              "inputs": inputs, "header": header, "source_tensor_sha256": identities,
              "invocation": invocation,
              "state_equivalence": state_metrics, "manual_moe_max_bf16_ulp": manual_ulp,
              "cache_state_schema": cache_rows,
              "raw_tensor_manifest": raw_manifest, "minimum_top10_top11_margin_fp32": min_margin,
              "boundary_tie_classification": [{"prompt": p, "rows_with_boundary_tie": [int(row) for row in torch.nonzero(route["router_top10_top11_margin_fp32"] == 0, as_tuple=False).flatten()],
                                                "tie_count_per_row": [int(x) for x in route["router_boundary_tie_mask"].sum(dim=-1)]}
                                               for p, route in enumerate(whole_routes)],
              "raw_artifact": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
              "raw_artifact_sha256": sha256(raw_path),
              "resources": {**peak, "projected_steady_working_set_bytes": projected},
              "runtime_contract": runtime_contract, "actual_dependencies": dependencies,
              "cuda_initialized_after": torch.cuda.is_initialized(),
              "claim_boundary": "Official layer-0 CPU reference and same-backend clean-process reproducibility only. No Q5 bank/build/physical/P4 or cross-backend claim."}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def smoke() -> dict[str, Any]:
    inputs = locked_inputs(); header = inspect_safetensors_header()
    config = Qwen3NextConfig.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    layer = make_layer(config, materialize=False)
    parameters = sum(p.numel() for p in layer.parameters())
    meta_contract = all(p.device.type == "meta" and p.dtype == torch.bfloat16 for p in layer.parameters())
    del layer; gc.collect()
    return {"kind": "port80b_t0r11_cpu_smoke", "pass": header["pass"] and meta_contract,
            "inputs": inputs, "header": header,
            "meta_only_bf16_parameter_elements": parameters, "meta_only_bf16_contract": meta_contract,
            "cuda_initialized": torch.cuda.is_initialized(),
            "physical_actions": {"network": False, "reference_forward": False, "bank_build": False,
                                 "gpu": False, "host_registration": False, "registry_edit": False}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("lockcheck", "ledger-unit", "smoke", "capture"), required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2))
    parser.add_argument("--acknowledge-reference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    started = time.perf_counter()
    if args.phase == "ledger-unit":
        payload={"kind":"port80b_t0r11_ledger_unit",**ledger_schema_unit_test(),"physical_actions":{"model_loaded":False,"forward":False,"bank":False,"gpu":False}}
    elif args.phase == "lockcheck":
        payload = {"kind":"port80b_t0r11_exact_execution_lockcheck","pass":execution_lock_check()["pass"],
                   "physical_actions":{"model_loaded":False,"forward":False,"bank_build":False,"gpu":False}}
    elif args.phase == "smoke":
        payload = smoke()
    else:
        if args.run_index is None or args.acknowledge_reference != ACK:
            raise SystemExit("capture phase requires run index and exact acknowledgement")
        try:
            FAILURE_STATE["stage"] = "reference_entry"
            payload = reference(args.run_index)
        except BaseException as error:
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            failure = RUN_DIR / f"t0r11_capture_{args.run_index}_failure.json"
            if failure.exists():
                raise RuntimeError(f"refusing to overwrite failure evidence: {failure}") from error
            memory=psutil.Process().memory_info()
            evidence={"kind":"port80b_t0r11_official_cpu_reference_failure","status":"valid_negative_or_blocked_not_pass","run_index":args.run_index,
                        "stage":FAILURE_STATE["stage"],"error_type":type(error).__name__,"error":str(error),"runner_sha256":sha256(Path(__file__)),
                        "runner_lock_sha256":sha256(RUNNER_LOCK) if RUNNER_LOCK.is_file() else None,
                        "verifier_sha256":sha256(VERIFIER) if VERIFIER.is_file() else None,
                        "verifier_lock_sha256":sha256(VERIFIER_LOCK) if VERIFIER_LOCK.is_file() else None,
                        "resources":{**FAILURE_STATE["resources"], "final_rss_bytes":memory.rss,
                                     "final_windows_peak_working_set_bytes":getattr(memory,"peak_wset",memory.rss),
                                     "final_available_ram_bytes":psutil.virtual_memory().available},
                        "cuda_initialized":torch.cuda.is_initialized(),
                        "existing_paths":{"raw":(RUN_DIR/f"t0r11_capture_{args.run_index}_raw.safetensors").exists(),"result":(RUN_DIR/f"t0r11_capture_{args.run_index}_result.json").exists()},
                        "claim_boundary":"Failure evidence only; never a scientific pass."}
            with failure.open("x",encoding="utf-8") as handle: json.dump(evidence,handle,indent=2);handle.write("\n")
            print(json.dumps({"status":evidence["status"],"failure_evidence":str(failure)},indent=2));raise SystemExit(2)
    payload["wall_seconds"] = time.perf_counter() - started
    print(json.dumps(payload if args.phase in ("lockcheck","ledger-unit","smoke") else {"status": payload["status"], "wall_seconds": payload["wall_seconds"]}, indent=2))
