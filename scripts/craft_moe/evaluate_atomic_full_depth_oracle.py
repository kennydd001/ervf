from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
import torch.nn.functional as F
import tokenizers
from safetensors.torch import save_file
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from evaluate_atomic_oracle import (
    ACTIVE_EXPERTS,
    ATOMS_PER_EXPERT,
    BOOTSTRAP_RESAMPLES,
    CANDIDATE_BATCH,
    FRACTIONS,
    HIDDEN_SIZE,
    accounting_summary,
    exact_activations_and_outputs,
    numeric_summary,
    policy_id,
)
from evaluate_atomic_spread_oracle import (
    DOMAINS,
    FULL_TOKENS_PER_DOMAIN,
    METHOD,
    build_domains,
    fraction_policy_id,
)
from evaluate_crcq_layer23_downstream import forward_layer
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
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT


SMOKE_TOKENS = 32
MOE_LAYERS = tuple(range(1, 27))
SUPPORT_ROW_BYTES = ACTIVE_EXPERTS * ATOMS_PER_EXPERT // 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simultaneous 26-layer exact atomic oracle."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAINS, default=DOMAINS
    )
    parser.add_argument(
        "--tokens-per-domain", type=int, default=FULL_TOKENS_PER_DOMAIN
    )
    parser.add_argument("--candidate-batch", type=int, default=CANDIDATE_BATCH)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-supports", type=Path)
    return parser.parse_args()


def checked_args(args: argparse.Namespace) -> argparse.Namespace:
    args.domains = tuple(dict.fromkeys(args.domains))
    if args.stage == "smoke":
        if len(args.domains) != 1 or not 1 <= args.tokens_per_domain <= SMOKE_TOKENS:
            raise ValueError("smoke requires one domain and at most 32 tokens")
    elif (
        args.domains != DOMAINS
        or args.tokens_per_domain != FULL_TOKENS_PER_DOMAIN
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
        or args.candidate_batch != CANDIDATE_BATCH
    ):
        raise ValueError("the preregistered full-depth configuration is immutable")
    if args.bootstrap_resamples < 1 or args.candidate_batch < 1:
        raise ValueError("bootstrap and candidate batch must be positive")
    if args.output_json is None:
        relative = (
            Path("reports/craft_moe/atomic_full_depth_oracle.json")
            if args.stage == "full"
            else Path("reports/runs/craft_moe/atomic_full_depth_smoke.json")
        )
        args.output_json = ROOT / relative
    elif not args.output_json.is_absolute():
        args.output_json = ROOT / args.output_json
    if args.output_supports is None:
        relative = (
            Path("reports/craft_moe/atomic_full_depth_supports.safetensors")
            if args.stage == "full"
            else Path("reports/runs/craft_moe/atomic_full_depth_smoke_supports.safetensors")
        )
        args.output_supports = ROOT / relative
    elif not args.output_supports.is_absolute():
        args.output_supports = ROOT / args.output_supports
    args.output_json = args.output_json.resolve()
    args.output_supports = args.output_supports.resolve()
    reports = (ROOT / "reports").resolve()
    for path in (args.output_json, args.output_supports):
        if reports not in path.parents:
            raise ValueError("all outputs must be inside reports/")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    return args


def policy_specifications() -> list[dict[str, Any]]:
    return [
        {
            "id": fraction_policy_id(fraction),
            "method": "exact_all_atoms" if fraction == 1.0 else METHOD,
            "requested_fraction": fraction,
        }
        for fraction in FRACTIONS
    ]


@torch.inference_mode()
def forward_with_moe_capture(
    layer: torch.nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence, _ = hidden_states.shape
    positions = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    captured_route: list[tuple[torch.Tensor, torch.Tensor]] = []
    captured_input: list[torch.Tensor] = []

    def gate_hook(_module, _inputs, output):
        captured_route.append((output[0].detach(), output[1].detach()))

    def mlp_pre_hook(_module, inputs):
        captured_input.append(inputs[0].detach())

    gate_handle = layer.mlp.gate.register_forward_hook(gate_hook)
    mlp_handle = layer.mlp.register_forward_pre_hook(mlp_pre_hook)
    try:
        output = layer(
            hidden_states,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            output_attentions=False,
        )[0]
    finally:
        gate_handle.remove()
        mlp_handle.remove()
    if len(captured_route) != 1 or len(captured_input) != 1:
        raise RuntimeError("MoE hooks did not capture exactly one call")
    return output, captured_route[0][0], captured_route[0][1], captured_input[0]


@torch.inference_mode()
def reconstruct_tokenwise_mask(
    moe,
    activations: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.shape != activations.shape or mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean and match activations")
    tokens, active, atoms = activations.shape
    if expert_ids.shape != (tokens, active) or router_weights.shape != (
        tokens,
        active,
    ):
        raise ValueError("route tensors must match activations")
    selected = torch.empty(tokens, active, HIDDEN_SIZE, dtype=activations.dtype)
    for expert_index, expert in enumerate(moe.experts):
        positions = (expert_ids == expert_index).nonzero(as_tuple=False)
        if not positions.numel():
            continue
        token_indices = positions[:, 0]
        slot_indices = positions[:, 1]
        exact = activations[token_indices, slot_indices].to(moe.device)
        selected_mask = mask[token_indices, slot_indices].to(moe.device)
        sparse = torch.where(
            selected_mask,
            exact,
            torch.zeros((), dtype=exact.dtype, device=exact.device),
        )
        output = F.linear(sparse.reshape(positions.shape[0], atoms), expert.down)
        selected[token_indices, slot_indices] = output.cpu()
    if not torch.isfinite(selected.float()).all():
        raise RuntimeError("non-finite tokenwise sparse expert output")
    return (
        selected.float() * router_weights.float().unsqueeze(-1)
    ).sum(dim=1).to(activations.dtype)


def packed_support(mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    flattened = mask.detach().cpu().numpy().reshape(mask.shape[0], -1)
    packed = np.packbits(flattened, axis=1, bitorder="little")
    payload = packed.tobytes(order="C")
    tensor = torch.from_numpy(packed.copy())
    return tensor, {
        "encoding": "numpy.packbits(bitorder=little, flattened expert-slot/neuron)",
        "logical_shape": list(mask.shape),
        "packed_shape": list(tensor.shape),
        "packed_row_bytes": int(tensor.shape[1]),
        "sha256_packed": hashlib.sha256(payload).hexdigest(),
    }


def route_overlap_raw(
    candidate_ids: torch.Tensor, teacher_ids: torch.Tensor
) -> torch.Tensor:
    if candidate_ids.shape != teacher_ids.shape:
        raise ValueError("candidate and teacher route IDs must match")
    return (
        (candidate_ids.unsqueeze(2) == teacher_ids.unsqueeze(1))
        .any(dim=2)
        .sum(dim=1)
        .float()
        / candidate_ids.shape[1]
    )


def build_layer_masks(
    activations: torch.Tensor,
    router_weights: torch.Tensor,
    selected_down_norms: torch.Tensor,
    tokens_per_policy: int,
    specifications: list[dict[str, Any]],
) -> torch.Tensor:
    masks = torch.empty_like(activations, dtype=torch.bool)
    for policy_index, specification in enumerate(specifications):
        selected = slice(
            policy_index * tokens_per_policy,
            (policy_index + 1) * tokens_per_policy,
        )
        fraction = specification["requested_fraction"]
        if fraction == 1.0:
            masks[selected] = True
            continue
        score = (
            activations[selected].float().abs()
            * router_weights[selected].float().abs().unsqueeze(-1)
            * selected_down_norms[selected].float()
        )
        masks[selected] = global_topk_mask(score, fraction)
    return masks


def layer_domain_diagnostic(
    *,
    full_routed: torch.Tensor,
    sparse_routed: torch.Tensor,
    mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    teacher_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    candidate_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> dict[str, Any]:
    local = relative_routed_l2(full_routed, sparse_routed)
    overlap = route_overlap_raw(candidate_ids, teacher_ids)
    accounting = support_known_accounting(
        mask.sum(dim=2).to(torch.int64),
        atoms_per_expert=ATOMS_PER_EXPERT,
        hidden_size=HIDDEN_SIZE,
    )
    return {
        "local_routed_relative_l2": {
            "aggregate": numeric_summary(local),
            "raw": local.tolist(),
        },
        "route_top6_overlap": {
            "aggregate": numeric_summary(overlap),
            "raw": overlap.tolist(),
        },
        "router_weight_slot_order": regression_summary(
            teacher_weights, candidate_weights
        ),
        "hidden_after_patch": regression_summary(teacher_hidden, candidate_hidden),
        "accounting": accounting_summary(accounting),
    }


def aggregate_modelwide_accounting(
    layer_records: dict[str, Any],
    specifications: list[dict[str, Any]],
    domains: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    additive = (
        "retained_atoms",
        "ideal_weight_bytes",
        "ideal_macs",
        "tensor_local_pages_4k",
        "tensor_local_page_bytes",
    )
    for specification in specifications:
        policy = specification["id"]
        result[policy] = {}
        for domain in domains:
            raw_by_layer = [
                layer_records[str(layer)]["policies"][policy]["domains"][domain][
                    "accounting"
                ]["raw"]
                for layer in MOE_LAYERS
            ]
            summed = {
                key: np.asarray(
                    [record[key] for record in raw_by_layer], dtype=np.float64
                ).sum(axis=0)
                for key in additive
            }
            full = layer_records["1"]["policies"][policy]["domains"][domain][
                "accounting"
            ]
            full_atoms = full["full_routed_atoms"] * len(MOE_LAYERS)
            full_bytes = full["full_ideal_weight_bytes"] * len(MOE_LAYERS)
            full_macs = full["full_ideal_macs"] * len(MOE_LAYERS)
            full_pages = full["full_tensor_local_pages_4k"] * len(MOE_LAYERS)
            full_page_bytes = (
                full["full_tensor_local_page_bytes"] * len(MOE_LAYERS)
            )
            fractions = {
                "retained_atom_fraction": summed["retained_atoms"] / full_atoms,
                "ideal_weight_byte_fraction": summed["ideal_weight_bytes"]
                / full_bytes,
                "ideal_mac_fraction": summed["ideal_macs"] / full_macs,
                "tensor_local_page_byte_fraction": summed[
                    "tensor_local_page_bytes"
                ]
                / full_page_bytes,
            }
            result[policy][domain] = {
                "full_routed_atoms": full_atoms,
                "full_ideal_weight_bytes": full_bytes,
                "full_ideal_macs": full_macs,
                "full_tensor_local_pages_4k": full_pages,
                "full_tensor_local_page_bytes": full_page_bytes,
                "aggregate": {
                    **{
                        key: numeric_summary(torch.from_numpy(values))
                        for key, values in summed.items()
                    },
                    **{
                        key: numeric_summary(torch.from_numpy(values))
                        for key, values in fractions.items()
                    },
                },
                "raw": {
                    **{key: values.tolist() for key, values in summed.items()},
                    **{key: values.tolist() for key, values in fractions.items()},
                },
            }
    return result


def adjudicate(
    final_results: dict[str, Any], domains: tuple[str, ...], stage: str
) -> tuple[str, dict[str, Any]]:
    if stage == "smoke":
        return "smoke_passed_not_adjudicated", {
            "adjudicated": False,
            "reason": "all four preregistered domains are required",
        }
    domain_gates: dict[str, Any] = {}
    for domain in domains:
        primary = final_results[domain]["policies"][fraction_policy_id(0.25)]
        moonshot = final_results[domain]["policies"][fraction_policy_id(0.10)]
        control = final_results[domain]["policies"][fraction_policy_id(1.0)]
        primary_aggregate = primary["aggregate"]
        moonshot_aggregate = moonshot["aggregate"]
        control_exact = (
            max(control["raw"]["teacher_to_candidate_kl"]) == 0.0
            and all(control["raw"]["top1_agreement"])
            and control["aggregate"]["cross_entropy_delta"] == 0.0
        )
        primary_criteria = {
            "relative_ce_increase_lt_0_02": primary_aggregate[
                "relative_cross_entropy_delta"
            ]
            < 0.02,
            "mean_kl_le_0_03": primary_aggregate["teacher_to_candidate_kl"]
            <= 0.03,
            "top1_agreement_ge_0_90": primary_aggregate["top1_agreement"] >= 0.90,
            "exact_control": control_exact,
        }
        moonshot_criteria = {
            "relative_ce_increase_lt_0_03": moonshot_aggregate[
                "relative_cross_entropy_delta"
            ]
            < 0.03,
            "mean_kl_le_0_05_safety": moonshot_aggregate[
                "teacher_to_candidate_kl"
            ]
            <= 0.05,
            "top1_agreement_ge_0_85_safety": moonshot_aggregate[
                "top1_agreement"
            ]
            >= 0.85,
        }
        hard = (
            primary_aggregate["relative_cross_entropy_delta"] >= 0.02
            or primary_aggregate["teacher_to_candidate_kl"] > 0.10
            or primary_aggregate["top1_agreement"] < 0.75
            or not control_exact
        )
        domain_gates[domain] = {
            "primary_criteria": primary_criteria,
            "primary_passed": all(primary_criteria.values()),
            "primary_25pct": primary_aggregate,
            "moonshot_criteria": moonshot_criteria,
            "moonshot_ce_passed": moonshot_criteria[
                "relative_ce_increase_lt_0_03"
            ],
            "moonshot_all_safety_diagnostics_passed": all(
                moonshot_criteria.values()
            ),
            "moonshot_10pct": moonshot_aggregate,
            "hard_falsification": hard,
        }
    primary = all(row["primary_passed"] for row in domain_gates.values())
    moonshot_ce = all(row["moonshot_ce_passed"] for row in domain_gates.values())
    moonshot_safe = all(
        row["moonshot_all_safety_diagnostics_passed"]
        for row in domain_gates.values()
    )
    hard = any(row["hard_falsification"] for row in domain_gates.values())
    if primary:
        verdict = "full_depth_positive_opens_candidate_validation"
    elif hard:
        verdict = "full_depth_falsified"
    else:
        verdict = "inconclusive"
    return verdict, {
        "adjudicated": True,
        "primary_25pct_all_domains_passed": primary,
        "moonshot_10pct_ce_all_domains_passed": moonshot_ce,
        "moonshot_10pct_all_safety_diagnostics_passed": moonshot_safe,
        "hard_falsification": hard,
        "domains": domain_gates,
    }


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("the simultaneous full-depth oracle requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models/deepseek-v2-lite"
    authorization = ROOT / "reports/craft_moe/atomic_spread_oracle.json"
    preregistration = ROOT / "reports/craft_moe/H3_ATOMIC_FULL_DEPTH_PREREGISTRATION.md"
    for path in (model_dir, authorization, preregistration):
        if not path.exists():
            raise FileNotFoundError(path)
    with authorization.open("r", encoding="utf-8") as handle:
        authorization_result = json.load(handle)
    if not authorization_result.get("simultaneous_full_depth_eligible", False):
        raise RuntimeError("spread result did not authorize full depth")

    initial_hardware = hardware_state()
    repository = git_state()
    disk_before = psutil.disk_usage(str(ROOT))
    domain_ids, corpus_manifest = build_domains(
        model_dir, args.domains, args.tokens_per_domain
    )
    input_ids = torch.cat(tuple(domain_ids[domain] for domain in args.domains), dim=0)
    sequence = input_ids.shape[1]
    blocks_per_domain = input_ids.shape[0] // len(args.domains)
    total_blocks = input_ids.shape[0]
    tokens_per_policy = input_ids.numel()
    specifications = policy_specifications()
    policies = len(specifications)

    teacher_embeddings = load_token_embeddings(model_dir, input_ids, device)
    combined = torch.cat([teacher_embeddings] * (policies + 1), dim=0)
    layer0, _ = load_decoder_layer(model_dir, 0, device)
    combined = forward_layer(layer0, combined)
    del layer0, teacher_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    support_tensors: dict[str, torch.Tensor] = {}
    support_manifest: dict[str, Any] = {}
    layer_records: dict[str, Any] = {}
    exact_control_by_layer: dict[str, bool] = {}
    layer_timings: dict[str, Any] = {}
    for layer_index in MOE_LAYERS:
        layer_started = time.perf_counter()
        layer, _ = load_decoder_layer(model_dir, layer_index, device)
        official_next, official_ids, official_weights, moe_input = (
            forward_with_moe_capture(layer, combined)
        )
        teacher_ids = official_ids[:tokens_per_policy]
        teacher_weights = official_weights[:tokens_per_policy]
        candidate_ids = official_ids[tokens_per_policy:]
        candidate_weights = official_weights[tokens_per_policy:]
        candidate_input = moe_input.reshape(-1, HIDDEN_SIZE)[tokens_per_policy:]
        moe = loaded_moe_from_official_module(layer.mlp, layer=layer_index)
        recomputed_ids, recomputed_weights = moe.route(candidate_input)
        route_control = {
            "slot_order_ids_exact": bool(torch.equal(recomputed_ids, candidate_ids)),
            "set_ids_exact": bool(
                torch.equal(
                    recomputed_ids.sort(dim=1).values,
                    candidate_ids.sort(dim=1).values,
                )
            ),
            "router_weight_max_absolute_error": float(
                (recomputed_weights.float() - candidate_weights.float())
                .abs()
                .max()
                .item()
            ),
        }
        if not route_control["slot_order_ids_exact"]:
            raise RuntimeError(f"layer {layer_index} captured/recomputed routes differ")

        phase = time.perf_counter()
        activations, selected_outputs, down_norm_bank = exact_activations_and_outputs(
            moe, candidate_input.cpu(), candidate_ids.cpu()
        )
        selected_down_norms = down_norm_bank[candidate_ids.cpu()]
        masks = build_layer_masks(
            activations,
            candidate_weights.cpu(),
            selected_down_norms,
            tokens_per_policy,
            specifications,
        )
        full_routed = (
            selected_outputs.float()
            * candidate_weights.cpu().float().unsqueeze(-1)
        ).sum(dim=1).to(selected_outputs.dtype)
        sparse_routed = reconstruct_tokenwise_mask(
            moe,
            activations,
            candidate_ids.cpu(),
            candidate_weights.cpu(),
            masks,
        )
        sparse_routed = torch.cat(
            (full_routed[:tokens_per_policy], sparse_routed[tokens_per_policy:]),
            dim=0,
        )
        layer_compute_seconds = time.perf_counter() - phase

        teacher_next = official_next[:total_blocks]
        candidate_official = official_next[total_blocks:].reshape(
            policies, tokens_per_policy, HIDDEN_SIZE
        ).cpu()
        next_candidates = torch.stack(
            [
                delta_patched_hidden(
                    candidate_official[policy_index],
                    full_routed[
                        policy_index
                        * tokens_per_policy : (policy_index + 1)
                        * tokens_per_policy
                    ],
                    sparse_routed[
                        policy_index
                        * tokens_per_policy : (policy_index + 1)
                        * tokens_per_policy
                    ],
                )
                for policy_index in range(policies)
            ]
        ).reshape(policies, *teacher_next.shape)
        control_exact = torch.equal(next_candidates[0].cpu(), teacher_next.cpu())
        exact_control_by_layer[str(layer_index)] = control_exact
        if not control_exact:
            raise RuntimeError(f"100% exact control diverged at layer {layer_index}")

        policies_record: dict[str, Any] = {}
        for policy_index, specification in enumerate(specifications):
            policy = specification["id"]
            policy_slice = slice(
                policy_index * tokens_per_policy,
                (policy_index + 1) * tokens_per_policy,
            )
            packed, packed_metadata = packed_support(masks[policy_slice])
            support_key = f"layer_{layer_index:02d}__{policy}"
            if packed.shape[1] != SUPPORT_ROW_BYTES:
                raise RuntimeError("unexpected packed support row width")
            support_tensors[support_key] = packed
            support_manifest[support_key] = packed_metadata
            domain_rows: dict[str, Any] = {}
            for domain_index, domain in enumerate(args.domains):
                token_start = domain_index * args.tokens_per_domain
                token_stop = token_start + args.tokens_per_domain
                token_slice = slice(token_start, token_stop)
                block_start = domain_index * blocks_per_domain
                block_stop = block_start + blocks_per_domain
                domain_rows[domain] = layer_domain_diagnostic(
                    full_routed=full_routed[policy_slice][token_slice],
                    sparse_routed=sparse_routed[policy_slice][token_slice],
                    mask=masks[policy_slice][token_slice],
                    candidate_ids=candidate_ids[policy_slice][token_slice].cpu(),
                    teacher_ids=teacher_ids[token_slice].cpu(),
                    candidate_weights=candidate_weights[policy_slice][
                        token_slice
                    ].cpu(),
                    teacher_weights=teacher_weights[token_slice].cpu(),
                    candidate_hidden=next_candidates[policy_index][
                        block_start:block_stop
                    ].cpu(),
                    teacher_hidden=teacher_next[block_start:block_stop].cpu(),
                )
                domain_rows[domain]["support"] = {
                    "artifact_key": support_key,
                    "artifact_row_slice": [token_start, token_stop],
                    **packed_metadata,
                }
            policies_record[policy] = {
                "requested_fraction": specification["requested_fraction"],
                "domains": domain_rows,
            }
        layer_records[str(layer_index)] = {
            "route_control": route_control,
            "exact_control": control_exact,
            "policies": policies_record,
        }
        combined = torch.cat(
            [teacher_next, *(next_candidates[index].to(device) for index in range(policies))],
            dim=0,
        )
        layer_timings[str(layer_index)] = {
            "exact_activation_mask_and_reconstruction_seconds": layer_compute_seconds,
            "total_layer_seconds": time.perf_counter() - layer_started,
        }
        print(f"full_depth_atomic_layer={layer_index:02d}/26", flush=True)
        del layer, moe, official_next, moe_input, candidate_input
        del activations, selected_outputs, down_norm_bank, selected_down_norms
        del masks, full_routed, sparse_routed, candidate_official, next_candidates
        gc.collect()
        torch.cuda.empty_cache()

    phase = time.perf_counter()
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    final_results: dict[str, Any] = {}
    for domain_index, domain in enumerate(args.domains):
        block_start = domain_index * blocks_per_domain
        block_stop = block_start + blocks_per_domain
        teacher_final = combined[block_start:block_stop].reshape(
            -1, HIDDEN_SIZE
        ).cpu()
        reference = make_teacher_reference(
            teacher_final,
            domain_ids[domain].reshape(-1),
            sequence_blocks(args.tokens_per_domain),
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        rows: dict[str, Any] = {}
        for policy_index, specification in enumerate(specifications, start=1):
            policy_start = policy_index * total_blocks + block_start
            policy_stop = policy_start + blocks_per_domain
            final_hidden = combined[policy_start:policy_stop].reshape(
                -1, HIDDEN_SIZE
            ).cpu()
            rows[specification["id"]] = evaluate_hidden(
                final_hidden,
                reference,
                norm_weight,
                lm_head,
                args.candidate_batch,
                args.bootstrap_resamples,
                args.seed + domain_index,
            )
            rows[specification["id"]]["requested_fraction"] = specification[
                "requested_fraction"
            ]
            print(
                f"full_depth_evaluated[{domain}]={policy_index}/{policies}",
                flush=True,
            )
        final_results[domain] = {
            "teacher_reference": {
                "token_ids": domain_ids[domain].reshape(-1).tolist(),
                "true_token_nll": nullable(reference.true_token_nll),
                "sequence_blocks": [list(block) for block in reference.blocks],
            },
            "policies": rows,
            "curve_index": [
                {
                    "requested_fraction": specification["requested_fraction"],
                    "policy_id": specification["id"],
                }
                for specification in specifications
            ],
        }
    final_projection_seconds = time.perf_counter() - phase
    modelwide_accounting = aggregate_modelwide_accounting(
        layer_records, specifications, args.domains
    )
    verdict, gates = adjudicate(final_results, args.domains, args.stage)

    args.output_supports.parent.mkdir(parents=True, exist_ok=True)
    temporary_supports = args.output_supports.with_suffix(
        args.output_supports.suffix + ".tmp"
    )
    save_file(
        support_tensors,
        str(temporary_supports),
        metadata={
            "kind": "craft_moe_atomic_full_depth_packed_supports",
            "model_revision": MODEL_REVISION,
            "encoding": "numpy.packbits bitorder=little",
        },
    )
    os.replace(temporary_supports, args.output_supports)
    supports_sha256 = sha256_file(args.output_supports)

    total_seconds = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    disk_after = psutil.disk_usage(str(ROOT))
    report = {
        "schema_version": 1,
        "kind": "craft_moe_exact_atomic_simultaneous_full_depth",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H3_ATOMIC_SIMULTANEOUS_FULL_DEPTH",
        "verdict": verdict,
        "candidate_validation_eligible": verdict
        == "full_depth_positive_opens_candidate_validation",
        "preregistration": str(preregistration.resolve()),
        "authorization_result": str(authorization.resolve()),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "dense_layer": 0,
            "simultaneously_intervened_moe_layers": list(MOE_LAYERS),
        },
        "dataset": {
            "wikitext_revision": DATASET_REVISION,
            "domains": list(args.domains),
            "tokens_per_domain": args.tokens_per_domain,
            "block_size": BLOCK_SIZE,
            "blocks_per_domain": blocks_per_domain,
            "corpus_manifest": corpus_manifest,
        },
        "configuration": {
            "policy_order_after_teacher": specifications,
            "method": METHOD,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "active_routed_experts": ACTIVE_EXPERTS,
            "atoms_per_expert": ATOMS_PER_EXPERT,
            "support_follows_each_candidate_path": True,
            "shared_experts": "exact via per-layer official-full routed delta patch",
            "router_weights_renormalized": False,
            "counterfactual_patch": (
                "BF16(official_full_candidate_next + sparse_routed_candidate "
                "- manual_full_routed_candidate)"
            ),
            "ties": "stable original expert-slot/neuron order",
            "quality_evaluation_implementation": (
                "dense BF16 GEMM with zero-masked activations; not sparse runtime"
            ),
        },
        "exact_control_by_layer": exact_control_by_layer,
        "layers": layer_records,
        "modelwide_accounting": modelwide_accounting,
        "final": final_results,
        "gates": gates,
        "support_artifact": {
            "path": str(args.output_supports),
            "sha256": supports_sha256,
            "size_bytes": args.output_supports.stat().st_size,
            "tensor_count": len(support_tensors),
            "manifest": support_manifest,
        },
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": repository,
            "input_hashes": {
                str(preregistration.resolve()): sha256_file(preregistration),
                str(authorization.resolve()): sha256_file(authorization),
                str((model_dir / "config.json").resolve()): sha256_file(
                    model_dir / "config.json"
                ),
                str((model_dir / "model.safetensors.index.json").resolve()): sha256_file(
                    model_dir / "model.safetensors.index.json"
                ),
            },
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
        "timings": {
            "total_compute_and_support_write_seconds": total_seconds,
            "final_projection_and_metrics_seconds": final_projection_seconds,
            "per_layer": layer_timings,
        },
        "limitations": [
            "support is selected from each candidate's exact activations and is not deployable early",
            "local domains are transfer checks and WikiText test was previously opened",
            "256 tokens per domain and two blocks are exploratory, not confirmation",
            "the tile64 hardware gate remains failed from the immutable layer26 screen",
            "analytical accounting and dense masked evaluation are not packed sparse runtime",
            "task accuracy, autoregressive stability, second-model replication, and confirmation remain open",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"support_result={args.output_supports}", flush=True)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    print(f"gates={json.dumps(gates, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
