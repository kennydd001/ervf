from __future__ import annotations

import argparse
import gc
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
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from capture_routed_subspace_pilot import (
    DATASET_REVISION,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MODEL_REVISION,
    ROUTER_WEIGHT_TOLERANCE,
    TOP_K,
    absolute_under_reports,
    forward_layer_and_capture,
    git_state,
    selected_intermediates,
    sha256_file,
    write_json_once,
)
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import load_decoder_layer
from moe_lab.reporting import ROOT


LAYERS = (1, 13, 26)
SPLITS = ("validation", "test")
CONTEXTS_PER_SPLIT = 2
PREFIX_TOKENS = 1024
FUTURE_TOKENS = 128
CONTEXT_TOKENS = PREFIX_TOKENS + FUTURE_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture preregistered 1024->128 V2 prompt-specific RSIV activations."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/runs/rsiv_moe/p1b_v2_long_prefix.safetensors"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/rsiv_moe/p1b_v2_long_prefix_capture.json"),
    )
    return parser.parse_args()


def contexts(corpus_dir: Path, tokenizer: Tokenizer) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for split in SPLITS:
        table = pq.read_table(
            corpus_dir / f"{split}-00000-of-00001.parquet", columns=["text"]
        )
        joined = "\n\n".join(
            text for text in table["text"].to_pylist() if text and text.strip()
        )
        requested = CONTEXTS_PER_SPLIT * CONTEXT_TOKENS
        ids = tokenizer.encode(joined).ids[:requested]
        if len(ids) != requested:
            raise RuntimeError(f"not enough {split} tokens for P1B")
        result[split] = torch.tensor(ids, dtype=torch.long).view(
            CONTEXTS_PER_SPLIT, CONTEXT_TOKENS
        )
    return result


if __name__ == "__main__":
    args = parse_args()
    output = absolute_under_reports(args.output)
    report_path = absolute_under_reports(args.report)
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite P1B capture artifacts")
    preregistration = ROOT / "reports/rsiv_moe/P1B_V2_LONG_PREFIX_PREREGISTRATION.md"
    if not preregistration.is_file():
        raise FileNotFoundError("P1B preregistration is required before capture")
    if not torch.cuda.is_available():
        raise RuntimeError("P1B capture requires CUDA")

    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model_dir = ROOT / "models/deepseek-v2-lite"
    corpus_dir = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1"
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    split_ids = contexts(corpus_dir, tokenizer)
    input_ids = torch.cat([split_ids[split] for split in SPLITS], dim=0)
    split_context_codes = torch.cat(
        [
            torch.full((CONTEXTS_PER_SPLIT,), index, dtype=torch.int8)
            for index, _split in enumerate(SPLITS)
        ]
    )
    hidden = load_token_embeddings(model_dir, input_ids, device)

    layer_zero, _config = load_decoder_layer(model_dir, 0, device)
    batch, sequence, _ = hidden.shape
    positions = torch.arange(sequence, device=device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(None, (batch, sequence), hidden, 0)
    with torch.inference_mode():
        hidden = layer_zero(
            hidden,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            output_attentions=False,
        )[0]
    del layer_zero, mask
    gc.collect()
    torch.cuda.empty_cache()

    tensors: dict[str, torch.Tensor] = {
        "input_ids": input_ids.contiguous(),
        "split_context_codes": split_context_codes.contiguous(),
    }
    controls: dict[str, Any] = {}
    for layer_index in range(1, 27):
        layer, _config = load_decoder_layer(model_dir, layer_index, device)
        hidden, moe_input, official_ids, official_weights = forward_layer_and_capture(
            layer, hidden
        )
        if layer_index in LAYERS:
            flat_input = moe_input.reshape(-1, HIDDEN_SIZE)
            official_ids = official_ids.reshape(-1, TOP_K)
            official_weights = official_weights.reshape(-1, TOP_K)
            moe = loaded_moe_from_official_module(layer.mlp, layer=layer_index)
            route_ids, route_weights = moe.route(flat_input)
            ids_exact = bool(torch.equal(route_ids, official_ids))
            weight_error = float(
                (route_weights.float() - official_weights.float()).abs().max().item()
            )
            if not ids_exact or weight_error > ROUTER_WEIGHT_TOLERANCE:
                raise RuntimeError(
                    f"P1B layer {layer_index} route control failed: "
                    f"ids={ids_exact}, weight_error={weight_error}"
                )
            z, _manual = selected_intermediates(layer, moe_input, official_ids)
            prefix = f"layer_{layer_index:02d}"
            tensors[f"{prefix}_moe_input"] = flat_input.detach().cpu().contiguous()
            tensors[f"{prefix}_router_ids"] = official_ids.to(torch.int16).cpu().contiguous()
            tensors[f"{prefix}_router_weights"] = official_weights.float().cpu().contiguous()
            tensors[f"{prefix}_intermediate_z"] = z.contiguous()
            counts = torch.bincount(official_ids.reshape(-1).long().cpu(), minlength=64)
            controls[str(layer_index)] = {
                "route_ids_exact": ids_exact,
                "router_weight_maximum_absolute_error": weight_error,
                "sum_expert_invocations": int(counts.sum().item()),
                "expected_expert_invocations": int(flat_input.shape[0] * TOP_K),
                "minimum_expert_invocations": int(counts.min().item()),
                "maximum_expert_invocations": int(counts.max().item()),
                "finite_x": bool(torch.isfinite(flat_input.float()).all()),
                "finite_z": bool(torch.isfinite(z.float()).all()),
            }
            print(
                f"P1B captured layer={layer_index} tokens={flat_input.shape[0]} "
                f"route_weight_error={weight_error:.3e}",
                flush=True,
            )
            del moe, z, _manual, flat_input
        del layer, moe_input, official_ids, official_weights
        gc.collect()
        torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(f"P1B forwarded layer={layer_index}/26", flush=True)

    tokens_per_split = CONTEXTS_PER_SPLIT * CONTEXT_TOKENS
    offsets = {
        "validation": [0, tokens_per_split],
        "test": [tokens_per_split, 2 * tokens_per_split],
    }
    metadata = {
        "kind": "rsiv_moe_p1b_v2_long_prefix_capture",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "layers": json.dumps(LAYERS),
        "contexts_per_split": str(CONTEXTS_PER_SPLIT),
        "context_tokens": str(CONTEXT_TOKENS),
        "prefix_tokens": str(PREFIX_TOKENS),
        "future_tokens": str(FUTURE_TOKENS),
        "split_offsets": json.dumps(offsets, sort_keys=True),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata=metadata)
    capture_hash = sha256_file(output)
    result = {
        "kind": "rsiv_moe_p1b_v2_long_prefix_capture",
        "status": "complete",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "layers": list(LAYERS),
        "contexts_per_split": CONTEXTS_PER_SPLIT,
        "context_tokens": CONTEXT_TOKENS,
        "prefix_tokens": PREFIX_TOKENS,
        "future_tokens": FUTURE_TOKENS,
        "split_offsets": offsets,
        "capture": str(output.relative_to(ROOT)).replace("\\", "/"),
        "capture_sha256": capture_hash,
        "capture_bytes": output.stat().st_size,
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
            "declared_process_budget_bytes": 32 * 1024**3,
        },
        "repository": git_state(),
        "preregistration": str(preregistration.relative_to(ROOT)).replace("\\", "/"),
        "test_analysis_performed": False,
        "claim_boundary": "Long-prefix activation capture only; no test rank or runtime result.",
    }
    write_json_once(report_path, result)
    print(
        json.dumps(
            {
                "capture": str(output),
                "capture_sha256": capture_hash,
                "capture_bytes": output.stat().st_size,
                "controls": controls,
            },
            indent=2,
        )
    )

