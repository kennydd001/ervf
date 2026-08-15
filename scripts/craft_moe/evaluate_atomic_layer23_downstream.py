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
import pyarrow
import safetensors
import torch
import tokenizers

from evaluate_atomic_oracle import (
    ACTIVE_EXPERTS,
    ATOMS_PER_EXPERT,
    BOOTSTRAP_RESAMPLES,
    CANDIDATE_BATCH,
    FRACTIONS,
    HIDDEN_SIZE,
    POLICY_BATCH,
    accounting_summary,
    block_bootstrap_mean,
    exact_activations_and_outputs,
    numeric_summary,
    packed_mask_record,
    policy_id,
    reconstruct_policy_masks,
)
from evaluate_crcq_layer23_downstream import (
    forward_layer,
    forward_with_router,
    layer_components,
    split_shaped_ids,
)
from evaluate_crcq_oracle import (
    BLOCK_SIZE,
    DATASET_REVISION,
    MODEL_REVISION,
    SEED,
    evaluate_hidden,
    git_state,
    hardware_state,
    make_teacher_reference,
    nullable,
    regression_summary,
    sequence_blocks,
    sha256_file,
    write_json_once,
)
from moe_lab.craft_moe.atomic import (
    delta_patched_hidden,
    global_topk_mask,
    relative_routed_l2,
    support_known_accounting,
)
from moe_lab.metrics import topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT


FULL_TOKENS_PER_SPLIT = 256
SMOKE_TOKENS = 32
INTERVENTION_LAYER = 23
TAIL_LAYERS = (24, 25, 26)
METHOD = "global_contribution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact layer-23 atomic intervention through layers 24-26."
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
    parser.add_argument("--candidate-batch", type=int, default=CANDIDATE_BATCH)
    parser.add_argument("--policy-batch", type=int, default=POLICY_BATCH)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.splits = tuple(dict.fromkeys(args.splits))
    if args.stage == "smoke":
        if (
            not 1 <= args.tokens_per_split <= SMOKE_TOKENS
            or args.splits != ("validation",)
        ):
            raise ValueError("smoke is limited to at most 32 validation tokens")
    elif (
        args.tokens_per_split != FULL_TOKENS_PER_SPLIT
        or args.splits != ("validation", "test")
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
        or args.candidate_batch != CANDIDATE_BATCH
        or args.policy_batch != POLICY_BATCH
    ):
        raise ValueError("the preregistered full layer-23 configuration is immutable")
    if args.bootstrap_resamples < 1 or args.candidate_batch < 1 or args.policy_batch < 1:
        raise ValueError("bootstrap and batch sizes must be positive")
    if args.output_json is None:
        relative = (
            Path("reports/craft_moe/atomic_layer23_downstream.json")
            if args.stage == "full"
            else Path("reports/runs/craft_moe/atomic_layer23_smoke.json")
        )
        args.output_json = ROOT / relative
    elif not args.output_json.is_absolute():
        args.output_json = ROOT / args.output_json
    args.output_json = args.output_json.resolve()
    if (ROOT / "reports").resolve() not in args.output_json.parents:
        raise ValueError("output-json must be inside reports/")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output_json}")
    return args


def build_masks(
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    selected_down_norms: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    contribution = (
        activations.float().abs()
        * router_weights.float().abs().unsqueeze(-1)
        * selected_down_norms.float()
    )
    specifications = [
        {
            "id": policy_id("exact_all_atoms", 1.0),
            "method": "exact_all_atoms",
            "requested_fraction": 1.0,
        }
    ]
    masks = [torch.ones_like(activations, dtype=torch.bool)]
    for fraction in FRACTIONS[1:]:
        specifications.append(
            {
                "id": policy_id(METHOD, fraction),
                "method": METHOD,
                "requested_fraction": fraction,
            }
        )
        masks.append(global_topk_mask(contribution, fraction))
    return specifications, masks


def split_policy_record(
    *,
    candidate_routed: torch.Tensor,
    original_routed: torch.Tensor,
    mask: torch.Tensor,
    final_hidden: torch.Tensor,
    reference: Any,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    candidate_batch: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    local = relative_routed_l2(original_routed, candidate_routed)
    counts = mask.sum(dim=2).to(torch.int64)
    accounting = support_known_accounting(
        counts,
        atoms_per_expert=ATOMS_PER_EXPERT,
        hidden_size=HIDDEN_SIZE,
    )
    return {
        "local_routed_relative_l2": {
            "aggregate": numeric_summary(local),
            "bootstrap_95_mean": block_bootstrap_mean(
                local, reference.blocks, bootstrap_resamples, seed
            ),
            "raw": local.tolist(),
        },
        "full_model": evaluate_hidden(
            final_hidden,
            reference,
            norm_weight,
            lm_head,
            candidate_batch,
            bootstrap_resamples,
            seed,
        ),
        "support": packed_mask_record(mask),
        "accounting": accounting_summary(accounting),
    }


def record_at(split_result: dict[str, Any], fraction: float) -> dict[str, Any]:
    selected_id = (
        policy_id("exact_all_atoms", 1.0)
        if fraction == 1.0
        else policy_id(METHOD, fraction)
    )
    return split_result["policies"][selected_id]


def adjudicate(results: dict[str, Any], stage: str) -> tuple[str, dict[str, Any]]:
    if stage == "smoke":
        return "smoke_passed_not_adjudicated", {
            "adjudicated": False,
            "reason": "fixed validation and test windows are required",
        }
    split_gates: dict[str, Any] = {}
    for split in ("validation", "test"):
        primary = record_at(results[split], 0.25)["full_model"]["aggregate"]
        moonshot = record_at(results[split], 0.10)["full_model"]["aggregate"]
        control = record_at(results[split], 1.0)
        control_metrics = control["full_model"]
        control_exact = (
            max(control_metrics["raw"]["teacher_to_candidate_kl"]) == 0.0
            and all(control_metrics["raw"]["top1_agreement"])
            and control_metrics["aggregate"]["cross_entropy_delta"] == 0.0
            and control["local_routed_relative_l2"]["aggregate"]["maximum"] == 0.0
        )
        criteria = {
            "relative_ce_increase_lt_0_02": primary[
                "relative_cross_entropy_delta"
            ]
            < 0.02,
            "mean_kl_le_0_01": primary["teacher_to_candidate_kl"] <= 0.01,
            "top1_agreement_ge_0_95": primary["top1_agreement"] >= 0.95,
            "exact_control": control_exact,
        }
        hard = (
            primary["relative_cross_entropy_delta"] >= 0.02
            or primary["teacher_to_candidate_kl"] > 0.02
            or primary["top1_agreement"] < 0.90
            or not control_exact
        )
        split_gates[split] = {
            "criteria": criteria,
            "passed": all(criteria.values()),
            "hard_falsification": hard,
            "primary_25pct": primary,
            "moonshot_10pct_relative_ce_lt_0_03": moonshot[
                "relative_cross_entropy_delta"
            ]
            < 0.03,
            "moonshot_10pct": moonshot,
        }
    primary_pass = all(item["passed"] for item in split_gates.values())
    moonshot_pass = all(
        item["moonshot_10pct_relative_ce_lt_0_03"]
        for item in split_gates.values()
    )
    hard = any(item["hard_falsification"] for item in split_gates.values())
    if primary_pass:
        verdict = "downstream_positive_opens_spread_layers"
    elif hard:
        verdict = "downstream_falsified"
    else:
        verdict = "inconclusive"
    return verdict, {
        "adjudicated": True,
        "primary_downstream_passed": primary_pass,
        "moonshot_passed": moonshot_pass,
        "hard_falsification": hard,
        "splits": split_gates,
    }


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("the exact layer-23 atomic intervention requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    model_dir = ROOT / "models/deepseek-v2-lite"
    authorization = ROOT / "reports/craft_moe/atomic_oracle.json"
    preregistration = ROOT / "reports/craft_moe/H3_ATOMIC_LAYER23_PREREGISTRATION.md"
    for path in (model_dir, authorization, preregistration):
        if not path.exists():
            raise FileNotFoundError(path)
    with authorization.open("r", encoding="utf-8") as handle:
        authorization_result = json.load(handle)
    if not authorization_result["gates"][
        "primary_global_25pct_relative_ce_lt_2pct_both_splits"
    ]:
        raise RuntimeError("layer-26 H3 gate did not authorize layer 23")

    initial_hardware = hardware_state()
    repository = git_state()
    disk_before = psutil.disk_usage(str(ROOT))
    input_hashes = {
        str(preregistration.resolve()): sha256_file(preregistration),
        str(authorization.resolve()): sha256_file(authorization),
        str((model_dir / "config.json").resolve()): sha256_file(
            model_dir / "config.json"
        ),
        str((model_dir / "model.safetensors.index.json").resolve()): sha256_file(
            model_dir / "model.safetensors.index.json"
        ),
    }
    split_ids = {
        split: split_shaped_ids(model_dir, split, args.tokens_per_split)
        for split in args.splits
    }
    input_ids = torch.cat(tuple(split_ids.values()), dim=0)
    sequence = input_ids.shape[1]
    blocks_per_split = input_ids.shape[0] // len(args.splits)
    total_blocks = input_ids.shape[0]
    total_tokens = input_ids.numel()

    phase = time.perf_counter()
    hidden = load_token_embeddings(model_dir, input_ids, device)
    for layer_index in range(INTERVENTION_LAYER):
        layer, _ = load_decoder_layer(model_dir, layer_index, device)
        hidden = forward_layer(layer, hidden)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"atomic_prefix_layer={layer_index:02d}", flush=True)
    timings["exact_prefix_layers_0_to_22_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    layer23, _ = load_decoder_layer(model_dir, INTERVENTION_LAYER, device)
    official_teacher23, official_ids, official_weights = forward_with_router(
        layer23, hidden
    )
    _, moe_input = layer_components(layer23, hidden)
    moe = loaded_moe_from_official_module(layer23.mlp, layer=INTERVENTION_LAYER)
    flat_input = moe_input.reshape(-1, moe_input.shape[-1])
    natural_ids, natural_weights = moe.route(flat_input)
    routing_control = {
        "slot_order_ids_exact": bool(torch.equal(natural_ids, official_ids)),
        "set_ids_exact": bool(
            torch.equal(
                natural_ids.sort(dim=1).values,
                official_ids.sort(dim=1).values,
            )
        ),
        "router_weight_max_absolute_error": float(
            (natural_weights.float() - official_weights.float()).abs().max().item()
        ),
    }
    if not routing_control["slot_order_ids_exact"]:
        raise RuntimeError("official and adapted layer-23 route slot order differs")
    activations, selected_outputs, down_norm_bank = exact_activations_and_outputs(
        moe, flat_input.cpu(), natural_ids.cpu()
    )
    selected_down_norms = down_norm_bank[natural_ids.cpu()]
    specifications, masks = build_masks(
        activations, natural_weights.cpu(), selected_down_norms
    )
    direct_routed = (
        selected_outputs.float() * natural_weights.cpu().float().unsqueeze(-1)
    ).sum(dim=1).to(selected_outputs.dtype)
    exact_routed = reconstruct_policy_masks(
        moe,
        activations,
        natural_ids.cpu(),
        natural_weights.cpu(),
        [masks[0]],
        1,
    )[0]
    decomposition_regression = regression_summary(direct_routed, exact_routed)
    decomposition_bit_exact = torch.equal(direct_routed, exact_routed)
    if (
        decomposition_regression["nrmse"] > 1e-4
        or decomposition_regression["maximum_absolute_error"] > 0.01
    ):
        raise RuntimeError(
            "layer-23 full atom decomposition exceeded BF16 regression tolerance: "
            f"{decomposition_regression}"
        )
    sparse_routed = reconstruct_policy_masks(
        moe,
        activations,
        natural_ids.cpu(),
        natural_weights.cpu(),
        masks[1:],
        args.policy_batch,
    )
    all_routed = torch.cat((exact_routed.unsqueeze(0), sparse_routed), dim=0)
    teacher_flat = official_teacher23.reshape(-1, HIDDEN_SIZE).cpu()
    candidate23 = torch.stack(
        [
            delta_patched_hidden(teacher_flat, exact_routed, all_routed[index])
            for index in range(len(specifications))
        ]
    ).reshape(len(specifications), *official_teacher23.shape)
    if not torch.equal(candidate23[0].cpu(), official_teacher23.cpu()):
        raise RuntimeError("layer-23 exact delta control is not bit-exact")
    timings["layer23_components_selectors_and_reconstruction_seconds"] = (
        time.perf_counter() - phase
    )
    del down_norm_bank, selected_down_norms, selected_outputs, direct_routed
    del layer23, moe, hidden, moe_input, flat_input, sparse_routed
    gc.collect()
    torch.cuda.empty_cache()

    phase = time.perf_counter()
    combined = torch.cat(
        [official_teacher23, *(candidate23[index].to(device) for index in range(len(specifications)))],
        dim=0,
    )
    downstream_layers: list[dict[str, Any]] = []
    for layer_index in TAIL_LAYERS:
        layer, _ = load_decoder_layer(model_dir, layer_index, device)
        combined, router_ids, router_weights = forward_with_router(layer, combined)
        teacher_router_ids = router_ids[:total_tokens]
        teacher_router_weights = router_weights[:total_tokens]
        teacher_hidden = combined[:total_blocks]
        policy_rows: dict[str, Any] = {}
        for policy_index, specification in enumerate(specifications, start=1):
            token_start = policy_index * total_tokens
            token_stop = token_start + total_tokens
            block_start = policy_index * total_blocks
            block_stop = block_start + total_blocks
            candidate_hidden = combined[block_start:block_stop]
            split_rows: dict[str, Any] = {}
            flat_offset = 0
            for split in args.splits:
                local_block_start = flat_offset // sequence
                local_block_stop = local_block_start + blocks_per_split
                token_slice = slice(flat_offset, flat_offset + args.tokens_per_split)
                split_rows[split] = {
                    "hidden": regression_summary(
                        teacher_hidden[local_block_start:local_block_stop].cpu(),
                        candidate_hidden[local_block_start:local_block_stop].cpu(),
                    ),
                    "router_top6_overlap": topk_overlap(
                        router_ids[token_start:token_stop][token_slice].cpu(),
                        teacher_router_ids[token_slice].cpu(),
                    ),
                    "router_weight": regression_summary(
                        teacher_router_weights[token_slice].cpu(),
                        router_weights[token_start:token_stop][token_slice].cpu(),
                    ),
                }
                flat_offset += args.tokens_per_split
            policy_rows[specification["id"]] = split_rows
        downstream_layers.append({"layer": layer_index, "policies": policy_rows})
        print(f"atomic_exact_tail_layer={layer_index}", flush=True)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    timings["exact_tail_layers_24_to_26_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    results: dict[str, Any] = {}
    block_offset = 0
    token_offset = 0
    for split_index, split in enumerate(args.splits):
        block_slice = slice(block_offset, block_offset + blocks_per_split)
        token_slice = slice(token_offset, token_offset + args.tokens_per_split)
        teacher_final = combined[block_slice].reshape(-1, HIDDEN_SIZE).cpu()
        reference = make_teacher_reference(
            teacher_final,
            split_ids[split].reshape(-1),
            sequence_blocks(args.tokens_per_split),
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        split_result: dict[str, Any] = {
            "teacher_reference": {
                "token_ids": split_ids[split].reshape(-1).tolist(),
                "true_token_nll": nullable(reference.true_token_nll),
                "sequence_blocks": [list(block) for block in reference.blocks],
            },
            "policies": {},
        }
        for policy_index, specification in enumerate(specifications, start=1):
            policy_block_start = policy_index * total_blocks + block_offset
            policy_block_stop = policy_block_start + blocks_per_split
            final_hidden = combined[policy_block_start:policy_block_stop].reshape(
                -1, HIDDEN_SIZE
            ).cpu()
            record = split_policy_record(
                candidate_routed=all_routed[policy_index - 1, token_slice],
                original_routed=all_routed[0, token_slice],
                mask=masks[policy_index - 1][token_slice],
                final_hidden=final_hidden,
                reference=reference,
                norm_weight=norm_weight,
                lm_head=lm_head,
                candidate_batch=args.candidate_batch,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed + split_index,
            )
            record["method"] = specification["method"]
            record["requested_fraction"] = specification["requested_fraction"]
            split_result["policies"][specification["id"]] = record
            print(
                f"atomic_layer23_evaluated[{split}]={policy_index}/{len(specifications)}",
                flush=True,
            )
        split_result["curve_index"] = [
            {
                "requested_fraction": fraction,
                "policy_id": (
                    policy_id("exact_all_atoms", 1.0)
                    if fraction == 1.0
                    else policy_id(METHOD, fraction)
                ),
            }
            for fraction in FRACTIONS
        ]
        results[split] = split_result
        block_offset += blocks_per_split
        token_offset += args.tokens_per_split
    timings["final_projection_and_metrics_seconds"] = time.perf_counter() - phase

    verdict, gates = adjudicate(results, args.stage)
    timings["total_compute_seconds"] = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    disk_after = psutil.disk_usage(str(ROOT))
    report = {
        "schema_version": 1,
        "kind": "craft_moe_exact_atomic_layer23_downstream",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H3_ATOMIC_LAYER23_EXACT_TAIL",
        "verdict": verdict,
        "spread_layer_domain_eligible": verdict
        == "downstream_positive_opens_spread_layers",
        "preregistration": str(preregistration.resolve()),
        "authorization_result": str(authorization.resolve()),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "intervention_layer": INTERVENTION_LAYER,
            "exact_tail_layers": list(TAIL_LAYERS),
        },
        "dataset": {
            "name": "WikiText-2-raw-v1",
            "revision": DATASET_REVISION,
            "windows": {
                split: f"first {args.tokens_per_split} tokens"
                for split in args.splits
            },
            "sequence_length": sequence,
            "blocks_per_split": blocks_per_split,
        },
        "configuration": {
            "tokens_per_split": args.tokens_per_split,
            "splits": list(args.splits),
            "fractions": list(FRACTIONS),
            "method": METHOD,
            "candidate_batch": args.candidate_batch,
            "policy_batch": args.policy_batch,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "active_routed_experts": ACTIVE_EXPERTS,
            "atoms_per_expert": ATOMS_PER_EXPERT,
            "shared_experts": "exact via official teacher delta patch",
            "router_weights_renormalized": False,
            "counterfactual_patch": (
                "BF16(official_teacher23 + sparse_routed - manual_full_routed)"
            ),
            "ties": "stable original expert-slot/neuron order",
            "quality_evaluation_implementation": (
                "dense BF16 GEMM with zero-masked activations; not sparse runtime"
            ),
        },
        "policy_order_after_teacher": specifications,
        "routing_control": routing_control,
        "decomposition_control": {
            "separate_direct_gemm_bit_exact": decomposition_bit_exact,
            "regression": decomposition_regression,
            "fixed_tolerance": {
                "nrmse_le": 1e-4,
                "maximum_absolute_error_le": 0.01,
            },
        },
        "downstream_layers": downstream_layers,
        "results": results,
        "gates": gates,
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "input_hashes": input_hashes,
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "safetensors": safetensors.__version__,
                "tokenizers": tokenizers.__version__,
                "pyarrow": pyarrow.__version__,
                "psutil": psutil.__version__,
            },
            "disk": {
                "free_bytes_before": disk_before.free,
                "free_bytes_after_compute": disk_after.free,
            },
            "initial_hardware": initial_hardware,
            "final_hardware": final_hardware,
        },
        "timings": timings,
        "limitations": [
            "exact-activation oracle support, not a deployable early selector",
            "only layer 23 with exact tail, not a full-depth multi-layer intervention",
            "256-token existing windows are exploratory, not confirmation",
            "two sequence blocks per split make bootstrap intervals coarse",
            "analytical bytes/MAC/pages are not packed-runtime wall-clock measurements",
            "task accuracy and autoregressive stability are not yet tested",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    print(f"gates={json.dumps(gates, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
