from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow.parquet as pq
import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_capture import QwenMoeInvocationCapture
from moe_lab.rsiv_moe.qwen_stream import (
    checkpoint_weight_map,
    load_qwen_decoder_layer,
    load_token_embeddings,
)


MODEL_ID = "Qwen/Qwen3-30B-A3B-Base"
MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LAYERS = (0, 23, 47)
CONTEXTS = 2
PREFIX_TOKENS = 1024
FUTURE_TOKENS = 128
CONTEXT_TOKENS = PREFIX_TOKENS + FUTURE_TOKENS
EXPECTED_WEIGHT_BYTES = 61_066_575_648
PROCESS_RSS_LIMIT = 32 * 2**30
CUDA_ALLOCATED_LIMIT = int(7.5 * 2**30)
ROUTER_TOLERANCE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream and capture preregistered P1C Qwen activations.")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_contexts(corpus_dir: Path, tokenizer, split: str) -> torch.Tensor:
    parquet = corpus_dir / f"{split}-00000-of-00001.parquet"
    table = pq.read_table(parquet, columns=["text"])
    joined = "\n\n".join(text for text in table["text"].to_pylist() if text and text.strip())
    requested = CONTEXTS * CONTEXT_TOKENS
    ids = tokenizer.encode(joined, add_special_tokens=False)[:requested]
    if len(ids) != requested:
        raise RuntimeError(f"not enough {split} tokens: {len(ids)} != {requested}")
    return torch.tensor(ids, dtype=torch.long).reshape(CONTEXTS, CONTEXT_TOKENS)


def validate_preregistered_config(config: Qwen3MoeConfig) -> None:
    expected = {
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "decoder_sparse_step": 1,
    }
    mismatches = {
        name: {"actual": getattr(config, name), "expected": value}
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if mismatches:
        raise RuntimeError(f"Qwen config differs from preregistration: {mismatches}")
    if config.mlp_only_layers:
        raise RuntimeError(f"expected every layer to be sparse, got mlp_only_layers={config.mlp_only_layers}")


def authorize_split(split: str) -> tuple[Path | None, str | None]:
    prereg = ROOT / "reports/rsiv_moe/P1C_QWEN3_30B_A3B_PREREGISTRATION.md"
    if not prereg.is_file():
        raise FileNotFoundError("P1C preregistration is missing")
    if split == "validation":
        return None, None
    lock = ROOT / "reports/rsiv_moe/p1c_qwen_validation_selection.json"
    if not lock.is_file():
        raise FileNotFoundError("test capture is forbidden before the validation lock")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    if payload.get("status") != "locked" or payload.get("test_open_authorized") is not True:
        raise RuntimeError("validation selection does not authorize the one-time test opening")
    return lock, sha256_file(lock)


def main() -> None:
    args = parse_args()
    split = args.split
    output = (args.output or Path(f"reports/runs/rsiv_moe/p1c_qwen_{split}.safetensors")).resolve()
    report_path = (args.report or Path(f"reports/rsiv_moe/p1c_qwen_{split}_capture.json")).resolve()
    if ROOT not in output.parents or ROOT not in report_path.parents:
        raise ValueError("capture artifacts must remain inside the project")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite P1C capture artifacts")
    lock_path, lock_sha = authorize_split(split)

    acquisition_path = ROOT / "reports/rsiv_moe/qwen_checkpoint_acquisition.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("status") != "complete_verified":
        raise RuntimeError("Qwen checkpoint acquisition is not complete and verified")
    if acquisition.get("revision") != MODEL_REVISION:
        raise RuntimeError("acquired Qwen revision differs from P1C")
    if acquisition.get("local_weight_bytes") != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError("acquired Qwen weight bytes differ from P1C")
    if acquisition.get("local_sha256_verified") is not True:
        raise RuntimeError("local Qwen shard SHA-256 verification is required")
    if transformers.__version__ != "4.51.3":
        raise RuntimeError(f"transformers version drift: {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1C capture requires CUDA")

    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model_dir = ROOT / "models/qwen3-30b-a3b-base"
    corpus_dir = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1"
    config = Qwen3MoeConfig.from_pretrained(model_dir, local_files_only=True)
    validate_preregistered_config(config)
    config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    input_ids = read_contexts(corpus_dir, tokenizer, split)
    input_ids_sha = sha256_tensor(input_ids)
    parquet_path = corpus_dir / f"{split}-00000-of-00001.parquet"
    parquet_sha = sha256_file(parquet_path)
    weight_map = checkpoint_weight_map(model_dir)
    hidden = load_token_embeddings(model_dir, input_ids, device, weight_map)
    if hidden.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 embeddings, got {hidden.dtype}")

    position_ids = torch.arange(CONTEXT_TOKENS, device=device).unsqueeze(0)
    rotary = Qwen3MoeRotaryEmbedding(config=config, device=device).to(device)
    with torch.inference_mode():
        position_embeddings = rotary(hidden, position_ids)

    tensors: dict[str, torch.Tensor] = {"input_ids": input_ids.contiguous()}
    controls: dict[str, Any] = {}
    captured_layers = set()
    for layer_index in range(config.num_hidden_layers):
        layer = load_qwen_decoder_layer(model_dir, config, layer_index, device, weight_map)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if layer_index in LAYERS:
            if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                raise RuntimeError(f"selected layer {layer_index} is not a sparse MoE block")
            with torch.inference_mode(), QwenMoeInvocationCapture(layer.mlp) as capture:
                hidden = layer(
                    hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    output_router_logits=False,
                    cache_position=position_ids.squeeze(0),
                    position_embeddings=position_embeddings,
                )[0]
            result = capture.result
            if result is None:
                raise RuntimeError(f"selected layer {layer_index} produced no capture")
            if not result.route_ids_exact:
                raise RuntimeError(f"selected layer {layer_index} route IDs differ")
            if result.router_weight_maximum_absolute_error > ROUTER_TOLERANCE:
                raise RuntimeError(f"selected layer {layer_index} router weights differ")
            if result.router_logits_maximum_absolute_error > ROUTER_TOLERANCE:
                raise RuntimeError(f"selected layer {layer_index} router logits differ")
            prefix = f"layer_{layer_index:02d}"
            tensors[f"{prefix}_moe_input"] = result.moe_input.contiguous()
            tensors[f"{prefix}_router_ids"] = result.router_ids.contiguous()
            tensors[f"{prefix}_router_weights"] = result.router_weights.contiguous()
            tensors[f"{prefix}_intermediate_z"] = result.intermediate_z.contiguous()
            counts = torch.bincount(result.router_ids.reshape(-1).long(), minlength=config.num_experts)
            controls[str(layer_index)] = {
                "route_ids_exact": result.route_ids_exact,
                "router_weight_maximum_absolute_error": result.router_weight_maximum_absolute_error,
                "router_logits_maximum_absolute_error": result.router_logits_maximum_absolute_error,
                "sum_expert_invocations": int(counts.sum()),
                "expected_expert_invocations": CONTEXTS * CONTEXT_TOKENS * config.num_experts_per_tok,
                "minimum_expert_invocations": int(counts.min()),
                "maximum_expert_invocations": int(counts.max()),
                "finite_x": bool(torch.isfinite(result.moe_input.float()).all()),
                "finite_z": bool(torch.isfinite(result.intermediate_z.float()).all()),
            }
            captured_layers.add(layer_index)
            print(f"P1C {split}: captured layer {layer_index}", flush=True)
        else:
            with torch.inference_mode():
                hidden = layer(
                    hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    output_router_logits=False,
                    cache_position=position_ids.squeeze(0),
                    position_embeddings=position_embeddings,
                )[0]
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        cuda_peak = int(torch.cuda.max_memory_allocated(device))
        if peak_rss > PROCESS_RSS_LIMIT:
            raise MemoryError(f"P1C RSS {peak_rss} exceeded {PROCESS_RSS_LIMIT}")
        if cuda_peak > CUDA_ALLOCATED_LIMIT:
            raise MemoryError(f"P1C CUDA allocation {cuda_peak} exceeded {CUDA_ALLOCATED_LIMIT}")
        print(f"P1C {split}: forwarded layer {layer_index + 1}/48", flush=True)

    if captured_layers != set(LAYERS):
        raise RuntimeError(f"captured layers {captured_layers} != {set(LAYERS)}")
    metadata = {
        "kind": "rsiv_moe_p1c_qwen_capture",
        "split": split,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "layers": json.dumps(LAYERS),
        "contexts": str(CONTEXTS),
        "context_tokens": str(CONTEXT_TOKENS),
        "prefix_tokens": str(PREFIX_TOKENS),
        "future_tokens": str(FUTURE_TOKENS),
        "validation_lock_sha256": lock_sha or "none",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata=metadata)
    capture_sha = sha256_file(output)
    report = {
        "kind": "rsiv_moe_p1c_qwen_capture",
        "status": "complete",
        "split": split,
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "transformers": transformers.__version__,
        "layers": list(LAYERS),
        "contexts": CONTEXTS,
        "context_tokens": CONTEXT_TOKENS,
        "prefix_tokens": PREFIX_TOKENS,
        "future_tokens": FUTURE_TOKENS,
        "input_ids_sha256": input_ids_sha,
        "source_parquet_sha256": parquet_sha,
        "capture": str(output.relative_to(ROOT)).replace("\\", "/"),
        "capture_sha256": capture_sha,
        "capture_bytes": output.stat().st_size,
        "validation_lock": None if lock_path is None else str(lock_path.relative_to(ROOT)).replace("\\", "/"),
        "validation_lock_sha256": lock_sha,
        "controls": controls,
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "safetensors": safetensors.__version__,
            "device": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_process_rss_bytes": int(peak_rss),
            "process_rss_limit_bytes": PROCESS_RSS_LIMIT,
            "cuda_allocated_limit_bytes": CUDA_ALLOCATED_LIMIT,
        },
        "test_analysis_performed": split == "test",
        "claim_boundary": "Teacher-state activation capture only; no quality, runtime or Eureka claim.",
    }
    write_json_once(report_path, report)
    print(json.dumps({"capture": str(output), "sha256": capture_sha, "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
