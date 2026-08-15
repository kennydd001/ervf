from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import safetensors
import torch
from safetensors.torch import load_file, save_file

from evaluate_crcq_oracle import (
    COMPONENT_RELATIVE,
    DATASET_REVISION,
    MODEL_REVISION,
    SEED,
    TRACE_TOKENS_PER_SPLIT,
    git_state,
    hardware_state,
    metadata,
    quantized_copy,
    regression_summary,
    sha256_file,
    write_json_once,
)
from moe_lab.craft_moe.qerc import (
    routed_output,
    weighted_error_decomposition,
)
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.reporting import ROOT


FULL_TOKENS_PER_SPLIT = 256
SMOKE_TOKENS = 32
LAYER = 26
HIDDEN_SIZE = 2048
ACTIVE_EXPERTS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture layer-26 BF16/Q3 co-routed errors and covariance."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--tokens-per-split", type=int, default=FULL_TOKENS_PER_SPLIT
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-components", type=Path)
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if not 1 <= args.tokens_per_split <= TRACE_TOKENS_PER_SPLIT:
        raise ValueError("tokens-per-split is outside the component trace")
    if args.stage == "smoke":
        if args.splits != ("validation",) or args.tokens_per_split > SMOKE_TOKENS:
            raise ValueError("smoke is at most 32 validation tokens")
    elif (
        args.tokens_per_split != FULL_TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.seed != SEED
    ):
        raise ValueError("full is fixed at 256 validation + 256 test and seed 20260810")
    if args.output_json is None:
        args.output_json = ROOT / (
            "reports/runs/craft_moe/qerc_covariance_layer26.json"
            if args.stage == "full"
            else "reports/runs/craft_moe/qerc_covariance_layer26_smoke.json"
        )
    elif not args.output_json.is_absolute():
        args.output_json = ROOT / args.output_json
    if args.output_components is None:
        args.output_components = ROOT / (
            "reports/runs/craft_moe/qerc_layer26_components.safetensors"
            if args.stage == "full"
            else "reports/runs/craft_moe/qerc_layer26_components_smoke.safetensors"
        )
    elif not args.output_components.is_absolute():
        args.output_components = ROOT / args.output_components
    args.output_json = args.output_json.resolve()
    args.output_components = args.output_components.resolve()
    reports = (ROOT / "reports").resolve()
    for path in (args.output_json, args.output_components):
        if reports not in path.parents:
            raise ValueError("outputs must be inside reports/")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    return args


def numeric_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().double().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("summary values must be non-empty and finite")
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


@torch.inference_mode()
def selected_bf16_q3_outputs(
    moe: LoadedMoELayer,
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    tokens, slots = expert_ids.shape
    bf16 = torch.empty(tokens, slots, HIDDEN_SIZE, dtype=inputs.dtype)
    q3 = torch.empty_like(bf16)
    touched = 0
    for expert_id, expert in enumerate(moe.experts):
        positions = (expert_ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            touched += 1
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            x = inputs[token_indices].to(moe.device)
            bf16[token_indices, slot_indices] = moe.expert_forward(x, expert).cpu()
            quant3 = quantized_copy(expert, 3)
            q3[token_indices, slot_indices] = moe.expert_forward(x, quant3).cpu()
            del quant3
        if expert_id % 8 == 7:
            print(f"qerc_capture_experts={expert_id + 1}/64", flush=True)
    if not torch.isfinite(bf16.float()).all() or not torch.isfinite(q3.float()).all():
        raise RuntimeError("non-finite captured expert output")
    return bf16, q3, {"experts_touched": touched}


def decomposition_record(
    bf16: torch.Tensor, q3: torch.Tensor, weights: torch.Tensor
) -> dict[str, Any]:
    decomposition = weighted_error_decomposition(bf16, q3, weights)
    return {
        "diagonal_energy": {
            "aggregate": numeric_summary(decomposition["diagonal_energy"]),
            "raw": decomposition["diagonal_energy"].tolist(),
            "sum": decomposition["diagonal_energy_sum"],
        },
        "aggregate_energy": {
            "aggregate": numeric_summary(decomposition["aggregate_energy"]),
            "raw": decomposition["aggregate_energy"].tolist(),
            "sum": decomposition["aggregate_energy_sum"],
        },
        "cross_term": {
            "aggregate": numeric_summary(decomposition["cross_term"]),
            "raw": decomposition["cross_term"].tolist(),
            "sum": decomposition["cross_term_sum"],
        },
        "token_cancellation_fraction": {
            "aggregate": numeric_summary(
                decomposition["token_cancellation_fraction"]
            ),
            "raw": decomposition["token_cancellation_fraction"].tolist(),
        },
        "global_cancellation_fraction_ratio_of_sums": decomposition[
            "global_cancellation_fraction"
        ],
    }


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("QERC capture requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    device = torch.device("cuda")
    model_dir = ROOT / "models/deepseek-v2-lite"
    component_path = ROOT / COMPONENT_RELATIVE
    preregistration = ROOT / "reports/craft_moe/H6_QERC_LAYER26_PREREGISTRATION.md"
    for path in (model_dir, component_path, preregistration):
        if not path.exists():
            raise FileNotFoundError(path)

    phase = time.perf_counter()
    input_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in (
            component_path,
            preregistration,
            model_dir / "config.json",
            model_dir / "model.safetensors.index.json",
        )
    }
    repository = git_state()
    initial_hardware = hardware_state()
    component_all = load_file(component_path, device="cpu")
    trace_indices: dict[str, list[int]] = {}
    indices: list[int] = []
    split_slices: dict[str, slice] = {}
    offset = 0
    for split in args.splits:
        base = 0 if split == "validation" else TRACE_TOKENS_PER_SPLIT
        chosen = list(range(base, base + args.tokens_per_split))
        trace_indices[split] = chosen
        indices.extend(chosen)
        split_slices[split] = slice(offset, offset + args.tokens_per_split)
        offset += args.tokens_per_split
    index = torch.tensor(indices, dtype=torch.long)
    needed = (
        "teacher",
        "moe_input",
        "router_ids",
        "router_weights",
        "selected_quant3",
    )
    components = {
        key: component_all[key].index_select(0, index) for key in needed
    }
    del component_all
    timings["load_inputs_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    moe = load_moe_layer(model_dir, LAYER, device)
    route_ids, route_weights = moe.route(components["moe_input"].to(device))
    route_control = {
        "slot_order_ids_exact": bool(
            torch.equal(route_ids.cpu(), components["router_ids"].long())
        ),
        "router_weight_maximum_absolute_error": float(
            (
                route_weights.cpu().float()
                - components["router_weights"].float()
            ).abs().max().item()
        ),
    }
    bf16, q3, capture = selected_bf16_q3_outputs(
        moe, components["moe_input"], components["router_ids"].long()
    )
    q3_trace_regression = regression_summary(components["selected_quant3"], q3)
    natural_routed = routed_output(bf16, components["router_weights"])
    q3_routed = routed_output(q3, components["router_weights"])
    exact_control_hidden = (
        components["teacher"].float()
        + natural_routed.float()
        - natural_routed.float()
    ).to(components["teacher"].dtype)
    exact_delta_bit_exact = torch.equal(exact_control_hidden, components["teacher"])
    if not exact_delta_bit_exact:
        raise RuntimeError("QERC exact teacher delta control failed")
    del moe, route_ids, route_weights
    gc.collect()
    torch.cuda.empty_cache()
    timings["capture_bf16_q3_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    results = {
        split: decomposition_record(
            bf16[split_slices[split]],
            q3[split_slices[split]],
            components["router_weights"][split_slices[split]],
        )
        for split in args.splits
    }
    timings["covariance_decomposition_seconds"] = time.perf_counter() - phase

    args.output_components.parent.mkdir(parents=True, exist_ok=True)
    phase = time.perf_counter()
    save_file(
        {
            "bf16_selected": bf16.contiguous(),
            "q3_selected": q3.contiguous(),
            "natural_routed": natural_routed.contiguous(),
            "q3_routed": q3_routed.contiguous(),
            "teacher": components["teacher"].contiguous(),
            "expert_ids": components["router_ids"].to(torch.int16).contiguous(),
            "router_weights": components["router_weights"].contiguous(),
            "trace_indices": index.contiguous(),
        },
        args.output_components,
        metadata={
            "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "layer": str(LAYER),
            "stage": args.stage,
            "splits": ",".join(args.splits),
            "tokens_per_split": str(args.tokens_per_split),
            "source_component": str(component_path.resolve()),
        },
    )
    component_hash = sha256_file(args.output_components)
    timings["write_component_artifact_seconds"] = time.perf_counter() - phase
    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()

    report = {
        "schema_version": 1,
        "kind": "craft_moe_h6_qerc_layer26_covariance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H6_QERC_PHASE_A",
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "layer": LAYER,
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
        },
        "definition": {
            "slot_error": "p_s * (Q3_expert_s(x) - BF16_expert_s(x))",
            "diagonal": "sum_s ||slot_error_s||^2",
            "aggregate": "||sum_s slot_error_s||^2",
            "cross": "aggregate - diagonal",
            "cancellation_fraction": "(diagonal - aggregate) / diagonal",
        },
        "results": results,
        "controls": {
            "route_recomputation": route_control,
            "official_teacher_delta_bit_exact": exact_delta_bit_exact,
            "stored_q3_batch_shape_regression_diagnostic": q3_trace_regression,
            "capture": capture,
        },
        "component_artifact": {
            "path": str(args.output_components.resolve()),
            "bytes": args.output_components.stat().st_size,
            "sha256": component_hash,
        },
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "safetensors": safetensors.__version__,
                "psutil": psutil.__version__,
            },
            "inputs": {
                "sha256": input_hashes,
                "component_metadata": metadata(component_path),
                "trace_indices": trace_indices,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "phase-A late-layer covariance only; no gain optimization or full-depth claim",
            "BF16/Q3 expert GEMMs are recomputed in the same batch, while comparison to the older stored Q3 tensor is diagnostic only",
            "cross terms describe natural error interaction and do not by themselves prove an optimizable same-byte scheme",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}")
    print(f"components={args.output_components}")
    for split, values in results.items():
        print(
            f"{split}_cancellation_fraction="
            f"{values['global_cancellation_fraction_ratio_of_sums']:.6f}"
        )


if __name__ == "__main__":
    main()
