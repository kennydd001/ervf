from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
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
import torch.nn.functional as F
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import load_decoder_layer
from moe_lab.reporting import ROOT


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LAYERS = (1, 13, 26)
BLOCK_SIZE = 128
SPLIT_BLOCKS = {"train": 8, "validation": 4, "test": 4}
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1408
TOP_K = 6
ROUTER_WEIGHT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture preregistered RSIV routed x/z activations on layers 1/13/26."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/runs/rsiv_moe/routed_subspace_pilot.safetensors"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/rsiv_moe/routed_subspace_capture.json"),
    )
    return parser.parse_args()


def absolute_under_reports(path: Path) -> Path:
    value = path if path.is_absolute() else ROOT / path
    value = value.resolve()
    if (ROOT / "reports").resolve() not in value.parents:
        raise ValueError("RSIV outputs must remain below reports/")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def token_blocks(
    corpus_dir: Path,
    tokenizer: Tokenizer,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for split, block_count in SPLIT_BLOCKS.items():
        table = pq.read_table(
            corpus_dir / f"{split}-00000-of-00001.parquet", columns=["text"]
        )
        joined = "\n\n".join(
            text for text in table["text"].to_pylist() if text and text.strip()
        )
        ids = tokenizer.encode(joined).ids[: block_count * BLOCK_SIZE]
        if len(ids) != block_count * BLOCK_SIZE:
            raise RuntimeError(f"not enough {split} tokens")
        result[split] = torch.tensor(ids, dtype=torch.long).view(
            block_count, BLOCK_SIZE
        )
    return result


@torch.inference_mode()
def forward_layer_and_capture(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence, _ = hidden_states.shape
    position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    captured_input: list[torch.Tensor] = []
    captured_route: list[tuple[torch.Tensor, torch.Tensor]] = []
    captured_output: list[torch.Tensor] = []

    def mlp_pre_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured_input.append(inputs[0].detach())

    def gate_hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        captured_route.append((output[0].detach(), output[1].detach()))

    def mlp_hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured_output.append(output.detach())

    handles = (
        layer.mlp.register_forward_pre_hook(mlp_pre_hook),
        layer.mlp.gate.register_forward_hook(gate_hook),
        layer.mlp.register_forward_hook(mlp_hook),
    )
    try:
        next_hidden = layer(
            hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )[0]
    finally:
        for handle in handles:
            handle.remove()
    if len(captured_input) != 1 or len(captured_route) != 1 or len(captured_output) != 1:
        raise RuntimeError("official MoE hooks did not fire exactly once")
    ids, weights = captured_route[0]
    return next_hidden, captured_input[0], ids, weights


@torch.inference_mode()
def selected_intermediates(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    moe = loaded_moe_from_official_module(layer.mlp, layer=-1)
    flat = inputs.reshape(-1, inputs.shape[-1])
    ids = expert_ids.reshape(flat.shape[0], TOP_K)
    z = torch.empty(
        flat.shape[0], TOP_K, INTERMEDIATE_SIZE, dtype=flat.dtype, device="cpu"
    )
    selected_y = torch.empty(
        flat.shape[0], TOP_K, HIDDEN_SIZE, dtype=flat.dtype, device="cpu"
    )
    for expert_id, expert in enumerate(moe.experts):
        positions = (ids == expert_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_ids, slots = positions[:, 0], positions[:, 1]
        x = flat[token_ids]
        gate = F.linear(x, expert.gate)
        up = F.linear(x, expert.up)
        local_z = F.silu(gate) * up
        z[token_ids.cpu(), slots.cpu()] = local_z.cpu()
        selected_y[token_ids.cpu(), slots.cpu()] = F.linear(
            local_z, expert.down
        ).cpu()
    if not torch.isfinite(z.float()).all() or not torch.isfinite(selected_y.float()).all():
        raise RuntimeError("non-finite routed expert capture")
    weights = moe.route(flat)[1].cpu().float()
    routed = (selected_y.float() * weights.unsqueeze(-1)).sum(1).to(flat.dtype)
    shared = moe.expert_forward(flat, moe.shared).cpu()
    manual_mlp = routed.cpu() + shared
    return z, {
        "manual_mlp": manual_mlp,
        "selected_y": selected_y,
    }


def regression(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference_f = reference.float().cpu()
    candidate_f = candidate.float().cpu()
    delta = candidate_f - reference_f
    denominator = float(torch.linalg.vector_norm(reference_f).item())
    return {
        "shape_exact": tuple(reference.shape) == tuple(candidate.shape),
        "bit_exact": bool(torch.equal(reference.cpu(), candidate.cpu())),
        "maximum_absolute_error": float(delta.abs().max().item()),
        "relative_l2": (
            float(torch.linalg.vector_norm(delta).item()) / denominator
            if denominator > 0.0
            else 0.0
        ),
    }


def git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "head_error": head.stderr.strip() if head.returncode else None,
    }


if __name__ == "__main__":
    args = parse_args()
    output = absolute_under_reports(args.output)
    report_path = absolute_under_reports(args.report)
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite RSIV capture or report")
    preregistration = ROOT / "reports/rsiv_moe/RSIV_MOE_PREREGISTRATION.md"
    if not preregistration.is_file():
        raise FileNotFoundError("P1 preregistration must exist before capture")
    if not torch.cuda.is_available():
        raise RuntimeError("the preregistered pilot capture requires CUDA")

    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model_dir = ROOT / "models/deepseek-v2-lite"
    corpus_dir = ROOT / "data/corpora/wikitext/wikitext-2-raw-v1"
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    split_ids = token_blocks(corpus_dir, tokenizer)
    ordered_splits = tuple(SPLIT_BLOCKS)
    input_ids = torch.cat([split_ids[name] for name in ordered_splits], dim=0)
    split_block_codes = torch.cat(
        [
            torch.full((SPLIT_BLOCKS[name],), index, dtype=torch.int8)
            for index, name in enumerate(ordered_splits)
        ]
    )
    hidden = load_token_embeddings(model_dir, input_ids, device)
    peak_rss = max(peak_rss, process.memory_info().rss)

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
    del layer_zero
    gc.collect()
    torch.cuda.empty_cache()

    tensors: dict[str, torch.Tensor] = {
        "input_ids": input_ids.contiguous(),
        "split_block_codes": split_block_codes.contiguous(),
    }
    layer_controls: dict[str, Any] = {}
    for layer_index in range(1, 27):
        layer, _config = load_decoder_layer(model_dir, layer_index, device)
        hidden, moe_input, official_ids, official_weights = forward_layer_and_capture(
            layer, hidden
        )
        if layer_index in LAYERS:
            flat_input = moe_input.reshape(-1, HIDDEN_SIZE)
            moe = loaded_moe_from_official_module(layer.mlp, layer=layer_index)
            recomputed_ids, recomputed_weights = moe.route(flat_input)
            route_ids_exact = bool(
                torch.equal(recomputed_ids, official_ids.reshape(-1, TOP_K))
            )
            route_weight_error = float(
                (
                    recomputed_weights.float()
                    - official_weights.reshape(-1, TOP_K).float()
                )
                .abs()
                .max()
                .item()
            )
            if not route_ids_exact or route_weight_error > ROUTER_WEIGHT_TOLERANCE:
                raise RuntimeError(
                    f"layer {layer_index} router control failed: "
                    f"ids={route_ids_exact}, weight_error={route_weight_error}"
                )
            z, manual = selected_intermediates(layer, moe_input, official_ids)
            with torch.inference_mode():
                official_mlp = layer.mlp(moe_input).detach().cpu()
            fallback_control = regression(
                official_mlp.reshape(-1, HIDDEN_SIZE), manual["manual_mlp"]
            )
            prefix = f"layer_{layer_index:02d}"
            tensors[f"{prefix}_moe_input"] = flat_input.detach().cpu().contiguous()
            tensors[f"{prefix}_router_ids"] = (
                official_ids.reshape(-1, TOP_K).to(torch.int16).cpu().contiguous()
            )
            tensors[f"{prefix}_router_weights"] = (
                official_weights.reshape(-1, TOP_K).float().cpu().contiguous()
            )
            tensors[f"{prefix}_intermediate_z"] = z.contiguous()
            counts = torch.bincount(
                official_ids.reshape(-1).long().cpu(), minlength=64
            )
            layer_controls[str(layer_index)] = {
                "route_ids_exact": route_ids_exact,
                "router_weight_maximum_absolute_error": route_weight_error,
                "sum_expert_invocations": int(counts.sum().item()),
                "expected_expert_invocations": int(flat_input.shape[0] * TOP_K),
                "minimum_expert_invocations": int(counts.min().item()),
                "maximum_expert_invocations": int(counts.max().item()),
                "manual_direct_fallback_regression": fallback_control,
            }
            print(
                f"captured layer={layer_index} tokens={flat_input.shape[0]} "
                f"route_weight_error={route_weight_error:.3e} "
                f"fallback_rel_l2={fallback_control['relative_l2']:.3e}",
                flush=True,
            )
            del moe, z, manual, official_mlp, flat_input
        del layer, moe_input, official_ids, official_weights
        gc.collect()
        torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(f"forwarded layer={layer_index}/26", flush=True)

    split_offsets: dict[str, list[int]] = {}
    offset = 0
    for name in ordered_splits:
        count = SPLIT_BLOCKS[name] * BLOCK_SIZE
        split_offsets[name] = [offset, offset + count]
        offset += count
    metadata = {
        "kind": "rsiv_moe_p1_routed_subspace_pilot_capture",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "layers": json.dumps(LAYERS),
        "block_size": str(BLOCK_SIZE),
        "split_blocks": json.dumps(SPLIT_BLOCKS, sort_keys=True),
        "split_offsets": json.dumps(split_offsets, sort_keys=True),
        "tensor_dtypes": "moe_input/intermediate_z preserve official BF16; router weights FP32",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata=metadata)
    capture_hash = sha256_file(output)
    result = {
        "kind": "rsiv_moe_p1_routed_subspace_capture",
        "status": "complete",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "layers": list(LAYERS),
        "block_size": BLOCK_SIZE,
        "split_blocks": SPLIT_BLOCKS,
        "split_offsets": split_offsets,
        "capture": str(output.relative_to(ROOT)).replace("\\", "/"),
        "capture_sha256": capture_hash,
        "capture_bytes": output.stat().st_size,
        "controls": layer_controls,
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
        "preregistration": "reports/rsiv_moe/RSIV_MOE_PREREGISTRATION.md",
        "claim_boundary": "Exact activation capture only; no rank, quality, byte-speed, or runtime verdict.",
    }
    write_json_once(report_path, result)
    print(json.dumps({"capture": str(output), "sha256": capture_hash, "controls": layer_controls}, indent=2))
