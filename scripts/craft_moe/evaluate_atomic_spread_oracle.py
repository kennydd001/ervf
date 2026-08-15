from __future__ import annotations

import argparse
import gc
import hashlib
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
import pyarrow.parquet as pq
import safetensors
import torch
import tokenizers
from tokenizers import Tokenizer

from evaluate_atomic_layer23_downstream import (
    METHOD,
    build_masks,
    split_policy_record,
)
from evaluate_atomic_oracle import (
    ACTIVE_EXPERTS,
    ATOMS_PER_EXPERT,
    BOOTSTRAP_RESAMPLES,
    CANDIDATE_BATCH,
    FRACTIONS,
    HIDDEN_SIZE,
    POLICY_BATCH,
    exact_activations_and_outputs,
    policy_id,
    reconstruct_policy_masks,
)
from evaluate_crcq_layer23_downstream import (
    forward_layer,
    forward_with_router,
    layer_components,
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
    regression_summary,
    sequence_blocks,
    sha256_file,
    write_json_once,
)
from moe_lab.craft_moe.atomic import delta_patched_hidden
from moe_lab.metrics import topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT


FULL_TOKENS_PER_DOMAIN = 256
SMOKE_TOKENS = 32
LAYERS = (1, 13, 26)
DOMAINS = (
    "wikitext_validation",
    "wikitext_test",
    "local_instruction",
    "local_code",
)
ATTACHMENTS = (
    Path(
        "C:/Users/de_do/.codex/attachments/35a0d2b1-5831-4a76-afe7-97aa4662286d/pasted-text.txt"
    ),
    Path(
        "C:/Users/de_do/.codex/attachments/7bdae3b8-a66e-43e0-8f2f-8dd1cee42181/pasted-text.txt"
    ),
    Path(
        "C:/Users/de_do/.codex/attachments/49e95e1b-41bd-45ec-8317-318bdb2af4d2/pasted-text.txt"
    ),
)
CRAFT_EXCLUSIONS = (
    "scripts/craft_moe/",
    "src/moe_lab/craft_moe/",
    "tests/craft_moe/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact atomic spread-layer and domain oracle."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument(
        "--domains", choices=DOMAINS, nargs="+", default=DOMAINS
    )
    parser.add_argument(
        "--tokens-per-domain", type=int, default=FULL_TOKENS_PER_DOMAIN
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
    args.layers = tuple(dict.fromkeys(args.layers))
    args.domains = tuple(dict.fromkeys(args.domains))
    if any(layer not in LAYERS for layer in args.layers):
        raise ValueError("layers must be selected from the preregistered 1, 13, 26")
    if args.stage == "smoke":
        if (
            len(args.layers) != 1
            or len(args.domains) != 1
            or not 1 <= args.tokens_per_domain <= SMOKE_TOKENS
        ):
            raise ValueError("smoke requires one layer, one domain, and <=32 tokens")
    elif (
        args.layers != LAYERS
        or args.domains != DOMAINS
        or args.tokens_per_domain != FULL_TOKENS_PER_DOMAIN
        or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or args.seed != SEED
        or args.candidate_batch != CANDIDATE_BATCH
        or args.policy_batch != POLICY_BATCH
    ):
        raise ValueError("the preregistered full spread configuration is immutable")
    if args.bootstrap_resamples < 1 or args.candidate_batch < 1 or args.policy_batch < 1:
        raise ValueError("bootstrap and batch sizes must be positive")
    if args.output_json is None:
        relative = (
            Path("reports/craft_moe/atomic_spread_oracle.json")
            if args.stage == "full"
            else Path(
                "reports/runs/craft_moe/"
                f"atomic_spread_smoke_layer{args.layers[0]}_{args.domains[0]}.json"
            )
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def token_blocks(tokenizer: Tokenizer, text: str, tokens: int) -> torch.Tensor:
    ids = tokenizer.encode(text).ids[:tokens]
    if len(ids) != tokens:
        raise RuntimeError(f"corpus yielded only {len(ids)} of {tokens} tokens")
    sequence = BLOCK_SIZE if tokens % BLOCK_SIZE == 0 else tokens
    return torch.tensor(ids, dtype=torch.long).view(-1, sequence)


def wikitext_text(split: str) -> tuple[str, Path]:
    parquet = (
        ROOT
        / "data/corpora/wikitext/wikitext-2-raw-v1"
        / f"{split}-00000-of-00001.parquet"
    )
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    return "\n\n".join(text for text in texts if text and text.strip()), parquet


def fixed_code_files() -> list[Path]:
    files = []
    for folder in (ROOT / "scripts", ROOT / "src", ROOT / "tests"):
        for path in folder.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if any(relative.startswith(prefix) for prefix in CRAFT_EXCLUSIONS):
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def build_domains(
    model_dir: Path, requested: tuple[str, ...], tokens: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    texts: dict[str, str] = {}
    manifest: dict[str, Any] = {
        "tokenizer": {
            "path": str(tokenizer_path.resolve()),
            "sha256": sha256_file(tokenizer_path),
        },
        "domains": {},
    }
    for domain in requested:
        if domain.startswith("wikitext_"):
            split = domain.removeprefix("wikitext_")
            text, path = wikitext_text(split)
            sources = [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            ]
            description = f"first tokens of pinned WikiText-2 {split}"
        elif domain == "local_instruction":
            missing = [path for path in ATTACHMENTS if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing)
            text = "\n\n".join(path.read_text(encoding="utf-8") for path in ATTACHMENTS)
            sources = [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in ATTACHMENTS
            ]
            description = "three user-supplied Dutch research/instruction attachments"
        else:
            code_files = fixed_code_files()
            if not code_files:
                raise RuntimeError("fixed local code corpus is empty")
            text = "\n\n".join(
                f"# {path.relative_to(ROOT).as_posix()}\n"
                f"{path.read_text(encoding='utf-8')}"
                for path in code_files
            )
            sources = [
                {
                    "path": str(path.relative_to(ROOT).as_posix()),
                    "sha256": sha256_file(path),
                }
                for path in code_files
            ]
            description = (
                "lexicographic Python source excluding all craft_moe directories"
            )
        texts[domain] = text
        manifest["domains"][domain] = {
            "description": description,
            "sources": sources,
            "concatenated_utf8_sha256": sha256_bytes(text.encode("utf-8")),
        }
    ids = {
        domain: token_blocks(tokenizer, texts[domain], tokens)
        for domain in requested
    }
    for domain, value in ids.items():
        manifest["domains"][domain]["token_ids_sha256"] = sha256_bytes(
            value.numpy().tobytes(order="C")
        )
        manifest["domains"][domain]["token_ids"] = value.reshape(-1).tolist()
    return ids, manifest


def fraction_policy_id(fraction: float) -> str:
    return (
        policy_id("exact_all_atoms", 1.0)
        if fraction == 1.0
        else policy_id(METHOD, fraction)
    )


def run_intervention_layer(
    *,
    layer_index: int,
    input_ids: torch.Tensor,
    domain_ids: dict[str, torch.Tensor],
    domains: tuple[str, ...],
    tokens_per_domain: int,
    model_dir: Path,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    layer_started = time.perf_counter()
    layer_timings: dict[str, float] = {}
    sequence = input_ids.shape[1]
    blocks_per_domain = input_ids.shape[0] // len(domains)
    total_blocks = input_ids.shape[0]
    total_tokens = input_ids.numel()

    phase = time.perf_counter()
    hidden = load_token_embeddings(model_dir, input_ids, device)
    for prefix_index in range(layer_index):
        layer, _ = load_decoder_layer(model_dir, prefix_index, device)
        hidden = forward_layer(layer, hidden)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"spread_layer={layer_index} prefix={prefix_index:02d}", flush=True
        )
    layer_timings["exact_prefix_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    layer, _ = load_decoder_layer(model_dir, layer_index, device)
    official_teacher, official_ids, official_weights = forward_with_router(
        layer, hidden
    )
    _, moe_input = layer_components(layer, hidden)
    moe = loaded_moe_from_official_module(layer.mlp, layer=layer_index)
    flat_input = moe_input.reshape(-1, HIDDEN_SIZE)
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
        raise RuntimeError(f"layer {layer_index} route slot order differs")
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
        or decomposition_regression["maximum_absolute_error"] > 0.02
    ):
        raise RuntimeError(
            f"layer {layer_index} atom decomposition regression failed: "
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
    teacher_flat = official_teacher.reshape(-1, HIDDEN_SIZE).cpu()
    candidates = torch.stack(
        [
            delta_patched_hidden(teacher_flat, exact_routed, all_routed[index])
            for index in range(len(specifications))
        ]
    ).reshape(len(specifications), *official_teacher.shape)
    if not torch.equal(candidates[0].cpu(), official_teacher.cpu()):
        raise RuntimeError(f"layer {layer_index} exact delta is not bit-exact")
    layer_timings["intervention_seconds"] = time.perf_counter() - phase
    del down_norm_bank, selected_down_norms, selected_outputs, direct_routed
    del layer, moe, hidden, moe_input, flat_input, activations, sparse_routed
    gc.collect()
    torch.cuda.empty_cache()

    phase = time.perf_counter()
    combined = torch.cat(
        [official_teacher, *(candidates[index].to(device) for index in range(len(specifications)))],
        dim=0,
    )
    downstream: list[dict[str, Any]] = []
    for tail_index in range(layer_index + 1, 27):
        tail, _ = load_decoder_layer(model_dir, tail_index, device)
        combined, router_ids, router_weights = forward_with_router(tail, combined)
        teacher_ids = router_ids[:total_tokens]
        teacher_weights = router_weights[:total_tokens]
        teacher_hidden = combined[:total_blocks]
        policy_rows: dict[str, Any] = {}
        for policy_index, specification in enumerate(specifications, start=1):
            token_start = policy_index * total_tokens
            token_stop = token_start + total_tokens
            block_start = policy_index * total_blocks
            block_stop = block_start + total_blocks
            candidate_hidden = combined[block_start:block_stop]
            domain_rows: dict[str, Any] = {}
            token_offset = 0
            for domain_index, domain in enumerate(domains):
                block_start_local = domain_index * blocks_per_domain
                block_stop_local = block_start_local + blocks_per_domain
                token_slice = slice(token_offset, token_offset + tokens_per_domain)
                domain_rows[domain] = {
                    "hidden": regression_summary(
                        teacher_hidden[block_start_local:block_stop_local].cpu(),
                        candidate_hidden[block_start_local:block_stop_local].cpu(),
                    ),
                    "router_top6_overlap": topk_overlap(
                        router_ids[token_start:token_stop][token_slice].cpu(),
                        teacher_ids[token_slice].cpu(),
                    ),
                    "router_weight": regression_summary(
                        teacher_weights[token_slice].cpu(),
                        router_weights[token_start:token_stop][token_slice].cpu(),
                    ),
                }
                token_offset += tokens_per_domain
            policy_rows[specification["id"]] = domain_rows
        downstream.append({"layer": tail_index, "policies": policy_rows})
        print(
            f"spread_layer={layer_index} exact_tail={tail_index:02d}", flush=True
        )
        del tail
        gc.collect()
        torch.cuda.empty_cache()
    layer_timings["exact_tail_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    results: dict[str, Any] = {}
    for domain_index, domain in enumerate(domains):
        block_offset = domain_index * blocks_per_domain
        block_slice = slice(block_offset, block_offset + blocks_per_domain)
        token_offset = domain_index * tokens_per_domain
        token_slice = slice(token_offset, token_offset + tokens_per_domain)
        teacher_final = combined[block_slice].reshape(-1, HIDDEN_SIZE).cpu()
        reference = make_teacher_reference(
            teacher_final,
            domain_ids[domain].reshape(-1),
            sequence_blocks(tokens_per_domain),
            norm_weight,
            lm_head,
            args.candidate_batch,
        )
        domain_result: dict[str, Any] = {
            "teacher_reference": {
                "teacher_cross_entropy": float(
                    reference.true_token_nll[torch.isfinite(reference.true_token_nll)]
                    .double()
                    .mean()
                    .item()
                ),
                "sequence_blocks": [list(block) for block in reference.blocks],
            },
            "policies": {},
        }
        for policy_index, specification in enumerate(specifications, start=1):
            policy_block_start = policy_index * total_blocks + block_offset
            policy_block_stop = policy_block_start + blocks_per_domain
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
                seed=args.seed + domain_index,
            )
            record["method"] = specification["method"]
            record["requested_fraction"] = specification["requested_fraction"]
            domain_result["policies"][specification["id"]] = record
            print(
                f"spread_evaluated[layer={layer_index},domain={domain}]="
                f"{policy_index}/{len(specifications)}",
                flush=True,
            )
        domain_result["curve_index"] = [
            {
                "requested_fraction": fraction,
                "policy_id": fraction_policy_id(fraction),
            }
            for fraction in FRACTIONS
        ]
        results[domain] = domain_result
    layer_timings["final_projection_and_metrics_seconds"] = (
        time.perf_counter() - phase
    )
    layer_timings["total_layer_seconds"] = time.perf_counter() - layer_started
    del combined, official_teacher, candidates
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "layer": layer_index,
        "policy_order_after_teacher": specifications,
        "routing_control": routing_control,
        "decomposition_control": {
            "separate_direct_gemm_bit_exact": decomposition_bit_exact,
            "regression": decomposition_regression,
            "fixed_tolerance": {
                "nrmse_le": 1e-4,
                "maximum_absolute_error_le": 0.02,
            },
        },
        "downstream_layers": downstream,
        "results": results,
        "timings": layer_timings,
    }


def adjudicate(
    layer_results: dict[str, Any],
    layers: tuple[int, ...],
    domains: tuple[str, ...],
    stage: str,
) -> tuple[str, dict[str, Any]]:
    if stage == "smoke":
        return "smoke_passed_not_adjudicated", {
            "adjudicated": False,
            "reason": "all preregistered layers and domains are required",
        }
    cells: dict[str, Any] = {}
    for layer in layers:
        layer_result = layer_results[str(layer)]
        for domain in domains:
            result = layer_result["results"][domain]
            primary = result["policies"][fraction_policy_id(0.25)]["full_model"]
            moonshot = result["policies"][fraction_policy_id(0.10)]["full_model"]
            control = result["policies"][fraction_policy_id(1.0)]
            control_metrics = control["full_model"]
            control_exact = (
                max(control_metrics["raw"]["teacher_to_candidate_kl"]) == 0.0
                and all(control_metrics["raw"]["top1_agreement"])
                and control_metrics["aggregate"]["cross_entropy_delta"] == 0.0
                and control["local_routed_relative_l2"]["aggregate"]["maximum"]
                == 0.0
            )
            aggregate = primary["aggregate"]
            criteria = {
                "relative_ce_increase_lt_0_02": aggregate[
                    "relative_cross_entropy_delta"
                ]
                < 0.02,
                "mean_kl_le_0_01": aggregate["teacher_to_candidate_kl"] <= 0.01,
                "top1_agreement_ge_0_95": aggregate["top1_agreement"] >= 0.95,
                "exact_control": control_exact,
            }
            hard = (
                aggregate["relative_cross_entropy_delta"] >= 0.02
                or aggregate["teacher_to_candidate_kl"] > 0.02
                or aggregate["top1_agreement"] < 0.90
                or not control_exact
            )
            key = f"layer{layer}:{domain}"
            cells[key] = {
                "layer": layer,
                "domain": domain,
                "criteria": criteria,
                "passed": all(criteria.values()),
                "hard_falsification": hard,
                "primary_25pct": aggregate,
                "moonshot_10pct": moonshot["aggregate"],
                "moonshot_relative_ce_lt_0_03": moonshot["aggregate"][
                    "relative_cross_entropy_delta"
                ]
                < 0.03,
            }
    primary = all(cell["passed"] for cell in cells.values())
    moonshot = all(cell["moonshot_relative_ce_lt_0_03"] for cell in cells.values())
    hard = any(cell["hard_falsification"] for cell in cells.values())
    if primary:
        verdict = "spread_positive_opens_simultaneous_full_depth"
    elif hard:
        verdict = "spread_falsified"
    else:
        verdict = "inconclusive"
    return verdict, {
        "adjudicated": True,
        "primary_all_12_cells_passed": primary,
        "moonshot_all_12_cells_passed": moonshot,
        "hard_falsification": hard,
        "cells": cells,
    }


def main() -> None:
    args = checked_args(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("the exact atomic spread oracle requires CUDA")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models/deepseek-v2-lite"
    authorization = ROOT / "reports/craft_moe/atomic_layer23_downstream.json"
    preregistration = ROOT / "reports/craft_moe/H3_ATOMIC_SPREAD_PREREGISTRATION.md"
    for path in (model_dir, authorization, preregistration):
        if not path.exists():
            raise FileNotFoundError(path)
    with authorization.open("r", encoding="utf-8") as handle:
        authorization_result = json.load(handle)
    if not authorization_result.get("spread_layer_domain_eligible", False):
        raise RuntimeError("layer-23 result did not authorize spread execution")

    initial_hardware = hardware_state()
    repository = git_state()
    disk_before = psutil.disk_usage(str(ROOT))
    domain_ids, corpus_manifest = build_domains(
        model_dir, args.domains, args.tokens_per_domain
    )
    input_ids = torch.cat(tuple(domain_ids[domain] for domain in args.domains), dim=0)
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)

    layer_results: dict[str, Any] = {}
    for layer in args.layers:
        layer_results[str(layer)] = run_intervention_layer(
            layer_index=layer,
            input_ids=input_ids,
            domain_ids=domain_ids,
            domains=args.domains,
            tokens_per_domain=args.tokens_per_domain,
            model_dir=model_dir,
            norm_weight=norm_weight,
            lm_head=lm_head,
            args=args,
            device=device,
        )

    verdict, gates = adjudicate(
        layer_results, args.layers, args.domains, args.stage
    )
    total_seconds = time.perf_counter() - started
    final_hardware = hardware_state()
    final_hardware["cuda"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final_hardware["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    disk_after = psutil.disk_usage(str(ROOT))
    report = {
        "schema_version": 1,
        "kind": "craft_moe_exact_atomic_spread_layers_domains",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "stage": args.stage,
        "experiment": "H3_ATOMIC_SPREAD_LAYERS_DOMAINS",
        "verdict": verdict,
        "simultaneous_full_depth_eligible": verdict
        == "spread_positive_opens_simultaneous_full_depth",
        "preregistration": str(preregistration.resolve()),
        "authorization_result": str(authorization.resolve()),
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": MODEL_REVISION,
            "intervention_layers": list(args.layers),
            "each_layer_evaluated_independently_to_full_depth": True,
        },
        "dataset": {
            "wikitext_revision": DATASET_REVISION,
            "domains": list(args.domains),
            "tokens_per_domain": args.tokens_per_domain,
            "block_size": BLOCK_SIZE,
            "blocks_per_domain": args.tokens_per_domain
            // (BLOCK_SIZE if args.tokens_per_domain % BLOCK_SIZE == 0 else args.tokens_per_domain),
            "corpus_manifest": corpus_manifest,
        },
        "configuration": {
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
                "BF16(official_teacher_L + sparse_routed_L - manual_full_routed_L)"
            ),
            "ties": "stable original expert-slot/neuron order",
            "quality_evaluation_implementation": (
                "dense BF16 GEMM with zero-masked activations; not sparse runtime"
            ),
        },
        "layers": layer_results,
        "gates": gates,
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
            "total_compute_seconds": total_seconds,
            "per_layer": {
                layer: result["timings"] for layer, result in layer_results.items()
            },
        },
        "limitations": [
            "each layer is intervened independently; this is not simultaneous full depth",
            "exact-activation oracle support is not an early deployable selector",
            "local instruction/code are transfer checks, not held-out confirmation",
            "WikiText test was already opened in earlier exploratory stages",
            "two blocks per cell make bootstrap intervals coarse",
            "analytical support accounting is not a packed sparse runtime measurement",
            "task accuracy and autoregressive stability are not yet tested",
        ],
    }
    write_json_once(args.output_json, report)
    print(f"result={args.output_json}", flush=True)
    print(f"verdict={verdict}", flush=True)
    print(f"gates={json.dumps(gates, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
