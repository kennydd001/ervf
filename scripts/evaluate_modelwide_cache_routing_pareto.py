from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from estimate_layer26_observability import behavioral_metrics, forward_layer
from evaluate_modelwide_cache_aware_bottom1 import (
    bootstrap_cache_intervals,
    bootstrap_quality_intervals,
)
from moe_lab.aggregate_student import dense_router_features
from moe_lab.cache_routing import CacheRoutingPolicy, parse_policy, route_batch
from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


DEFAULT_POLICIES = (
    "original",  # numerical control: must reproduce the official model
    "max_rank:j5:7",  # previous conservative baseline
    "max_rank:j2:7",
    "max_rank:j2:8",
    "max_rank:j2:10",
    "max_rank:j2:12",
    "cumsum:j2:0.1",
    "cumsum:j2:0.2",
    "cumsum:j2:0.3",
    "cumsum:j2:0.4",
    "cumsum:j2:0.5",
    "cache_prior:j2:0.05",
    "cache_prior:j2:0.1",
    "cache_prior:j2:0.2",
    "cache_prior:j2:0.3",
    "cache_prior:j2:0.5",
)
MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-per-split", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="skip this many tokenizer IDs independently at the start of each split",
    )
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument(
        "--policy",
        action="append",
        help="repeat METHOD:jN:VALUE; defaults to a broad paper-baseline sweep",
    )
    parser.add_argument(
        "--corpus-preset",
        choices=("wikitext", "local_diverse"),
        default="wikitext",
    )
    parser.add_argument("--validation-kl-limit", type=float, default=0.01)
    parser.add_argument("--validation-relative-ce-limit", type=float, default=0.005)
    parser.add_argument(
        "--report-name", default="modelwide_cache_routing_pareto.json"
    )
    return parser.parse_args()


def _token_blocks(
    tokenizer: Tokenizer,
    text: str,
    blocks: int,
    block_size: int,
    token_offset: int = 0,
) -> torch.Tensor:
    if token_offset < 0:
        raise ValueError("token-offset must be non-negative")
    required = blocks * block_size
    stop = token_offset + required
    ids = tokenizer.encode(text).ids
    if len(ids) < stop:
        raise RuntimeError(
            f"corpus has {len(ids)} tokens, fewer than requested stop {stop}"
        )
    return torch.tensor(ids[token_offset:stop], dtype=torch.long).view(
        blocks, block_size
    )


def wikitext_blocks(
    model_dir: Path,
    split: str,
    blocks: int,
    block_size: int,
    token_offset: int = 0,
) -> torch.Tensor:
    parquet = (
        ROOT
        / "data"
        / "corpora"
        / "wikitext"
        / "wikitext-2-raw-v1"
        / f"{split}-00000-of-00001.parquet"
    )
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return _token_blocks(tokenizer, joined, blocks, block_size, token_offset)


def local_diverse_blocks(
    model_dir: Path, blocks: int, block_size: int, token_offset: int = 0
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    attachment_paths = (
        Path("C:/Users/de_do/.codex/attachments/35a0d2b1-5831-4a76-afe7-97aa4662286d/pasted-text.txt"),
        Path("C:/Users/de_do/.codex/attachments/7bdae3b8-a66e-43e0-8f2f-8dd1cee42181/pasted-text.txt"),
        Path("C:/Users/de_do/.codex/attachments/49e95e1b-41bd-45ec-8317-318bdb2af4d2/pasted-text.txt"),
    )
    instruction_text = "\n\n".join(
        path.read_text(encoding="utf-8") for path in attachment_paths
    )
    code_paths = sorted(
        path
        for folder in (ROOT / "scripts", ROOT / "src", ROOT / "tests")
        for path in folder.rglob("*.py")
    )
    code_text = "\n\n".join(
        f"# {path.relative_to(ROOT)}\n{path.read_text(encoding='utf-8')}"
        for path in code_paths
    )
    return {
        "validation": _token_blocks(
            tokenizer, instruction_text, blocks, block_size, token_offset
        ),
        "test": _token_blocks(
            tokenizer, code_text, blocks, block_size, token_offset
        ),
    }, {
        "validation": "three user-supplied Dutch research/instruction attachments",
        "test": "Python source under workspace scripts, src, and tests",
    }


def _policy_weights(probabilities: torch.Tensor, ids: torch.Tensor, layer) -> torch.Tensor:
    moe = loaded_moe_from_official_module(layer.mlp, layer=0)
    weights = probabilities.gather(1, ids)
    if moe.norm_topk_prob:
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    return weights * moe.routed_scaling_factor


@torch.inference_mode()
def forward_all_policies(
    layer,
    teacher_input: torch.Tensor,
    student_inputs: dict[str, torch.Tensor],
    policies: tuple[CacheRoutingPolicy, ...],
    capacity: int,
    validation_blocks: int,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    names = tuple(policy.name for policy in policies)
    batch = teacher_input.shape[0]
    sequence = teacher_input.shape[1]
    moe = loaded_moe_from_official_module(layer.mlp, layer=0)

    captured_teacher_inputs: list[torch.Tensor] = []

    def capture_teacher_input(_module, inputs):
        captured_teacher_inputs.append(inputs[0].detach())

    handle = layer.mlp.gate.register_forward_pre_hook(capture_teacher_input)
    try:
        official_teacher = forward_layer(layer, teacher_input)
    finally:
        handle.remove()
    if len(captured_teacher_inputs) != 1:
        raise RuntimeError("expected exactly one teacher gate invocation")
    teacher_moe_input = captured_teacher_inputs[0]
    teacher_flat = teacher_moe_input.reshape(-1, teacher_moe_input.shape[-1])
    teacher_logits = F.linear(teacher_flat.float(), moe.gate_weight.float())
    teacher_probabilities = teacher_logits.softmax(-1)
    teacher_ranked_probabilities, teacher_ranked_ids = teacher_probabilities.sort(
        dim=-1, descending=True, stable=True
    )
    calibration_tokens = validation_blocks * sequence
    delta_average = float(
        (
            teacher_logits[:calibration_tokens].amax(-1)
            - teacher_logits[:calibration_tokens].amin(-1)
        )
        .mean()
        .item()
    )
    original = CacheRoutingPolicy("original")
    teacher_route, strict_stats = route_batch(
        teacher_ranked_ids.view(batch, sequence, -1),
        teacher_ranked_probabilities.view(batch, sequence, -1),
        teacher_logits.view(batch, sequence, -1),
        original,
        capacity,
        delta_average,
        moe.top_k,
    )
    teacher_route = teacher_route.to(teacher_logits.device)
    teacher_weights = _policy_weights(
        teacher_probabilities, teacher_route, layer
    )
    outputs: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for policy in policies:
        name = policy.name
        policy_input = student_inputs[name]
        captured: dict[str, object] = {}
        original_gate_forward = layer.mlp.gate.forward

        def cache_aware_gate(hidden_states):
            flat_input = hidden_states.reshape(-1, hidden_states.shape[-1])
            raw_logits = F.linear(flat_input.float(), moe.gate_weight.float())
            probabilities = raw_logits.softmax(-1)
            ranked_probabilities, ranked_ids = probabilities.sort(
                dim=-1, descending=True, stable=True
            )
            route_for_cache, stats = route_batch(
                ranked_ids.view(batch, sequence, -1),
                ranked_probabilities.view(batch, sequence, -1),
                raw_logits.view(batch, sequence, -1),
                policy,
                capacity,
                delta_average,
                moe.top_k,
            )
            if policy.method == "original":
                chosen, weights, auxiliary_loss = original_gate_forward(hidden_states)
            else:
                chosen = route_for_cache.to(raw_logits.device)
                weights = _policy_weights(probabilities, chosen, layer)
                auxiliary_loss = None
            captured.update(
                {
                    "ids": chosen.detach(),
                    "weights": weights.detach(),
                    "stats": stats,
                }
            )
            return chosen, weights, auxiliary_loss

        layer.mlp.gate.forward = cache_aware_gate
        try:
            output = forward_layer(layer, policy_input)
        finally:
            layer.mlp.gate.forward = original_gate_forward
        if set(captured) != {"ids", "weights", "stats"}:
            raise RuntimeError(f"policy gate was not captured for {name}")
        chosen = captured["ids"]
        weights = captured["weights"]
        stats = captured["stats"]
        outputs[name] = output
        diagnostics[name] = {
            "hidden": regression_metrics(output.cpu(), official_teacher.cpu()),
            "router_topk_overlap": topk_overlap(
                chosen.cpu(), teacher_route.cpu()
            ),
            "router_weight_nrmse": regression_metrics(
                dense_router_features(chosen.cpu(), weights.cpu(), 64),
                dense_router_features(
                    teacher_route.cpu(), teacher_weights.cpu(), 64
                ),
            )["nrmse"],
            "cache": stats,
        }
    control = regression_metrics(outputs["original"].cpu(), official_teacher.cpu())
    layer_metadata = {
        "delta_average_from_validation_teacher": delta_average,
        "strict_cache": strict_stats,
        "official_gate_control": control,
    }
    return official_teacher, outputs, layer_metadata, diagnostics


def _blank_totals(blocks: int, block_size: int) -> dict[str, object]:
    return {
        "adaptive_expert_loads": 0,
        "substituted_tokens": 0.0,
        "route_overlap_sum": 0.0,
        "per_block": [
            {
                "adaptive_expert_loads": 0,
                "substituted_tokens": 0.0,
                "route_overlap_sum": 0.0,
                "token_expert_decisions": block_size * 26,
            }
            for _ in range(blocks)
        ],
    }


def _pareto_front(rows: dict[str, dict[str, float]]) -> list[str]:
    front = []
    for name, row in rows.items():
        dominated = any(
            other != name
            and candidate["cache_miss_fraction"] <= row["cache_miss_fraction"]
            and candidate["teacher_to_candidate_kl"]
            <= row["teacher_to_candidate_kl"]
            and (
                candidate["cache_miss_fraction"] < row["cache_miss_fraction"]
                or candidate["teacher_to_candidate_kl"]
                < row["teacher_to_candidate_kl"]
            )
            for other, candidate in rows.items()
        )
        if not dominated:
            front.append(name)
    return sorted(front, key=lambda name: rows[name]["cache_miss_fraction"])


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("model-wide cache routing evaluation requires CUDA")
    if args.blocks_per_split < 1 or args.block_size < 2:
        raise ValueError("blocks-per-split must be positive and block-size at least two")
    if args.token_offset < 0:
        raise ValueError("token-offset must be non-negative")
    specifications = tuple(args.policy or DEFAULT_POLICIES)
    policies = tuple(parse_policy(specification) for specification in specifications)
    if not any(policy.method == "original" for policy in policies):
        policies = (CacheRoutingPolicy("original"), *policies)
    names = tuple(policy.name for policy in policies)
    if len(set(names)) != len(names):
        raise ValueError("policy specifications must map to unique names")
    if not 1 <= args.capacity <= 64:
        raise ValueError("capacity must be in [1, 64]")

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    if args.corpus_preset == "wikitext":
        split_ids = {
            split: wikitext_blocks(
                model_dir,
                split,
                args.blocks_per_split,
                args.block_size,
                args.token_offset,
            )
            for split in ("validation", "test")
        }
        corpus_sources = {
            "validation": "pinned WikiText-2 raw validation",
            "test": "pinned WikiText-2 raw test",
        }
    else:
        split_ids, corpus_sources = local_diverse_blocks(
            model_dir,
            args.blocks_per_split,
            args.block_size,
            args.token_offset,
        )
    input_ids = torch.cat((split_ids["validation"], split_ids["test"]), dim=0)
    total_blocks = input_ids.shape[0]
    embeddings = load_token_embeddings(model_dir, input_ids, device)
    layer0, _ = load_decoder_layer(model_dir, 0, device)
    teacher = forward_layer(layer0, embeddings)
    students = {name: teacher.clone() for name in names}
    del layer0, embeddings
    gc.collect()
    torch.cuda.empty_cache()

    strict_total = {
        "expert_loads": 0,
        "per_block": [0 for _ in range(total_blocks)],
    }
    totals = {
        name: _blank_totals(total_blocks, args.block_size) for name in names
    }
    layer_reports = []
    for layer_index in range(1, 27):
        layer, _ = load_decoder_layer(model_dir, layer_index, device)
        teacher, students, metadata, diagnostics = forward_all_policies(
            layer,
            teacher,
            students,
            policies,
            args.capacity,
            args.blocks_per_split,
        )
        strict = metadata["strict_cache"]
        strict_total["expert_loads"] += strict["expert_loads"]
        for block_index, block_stats in enumerate(strict["per_block"]):
            strict_total["per_block"][block_index] += block_stats["expert_loads"]
        for name in names:
            stats = diagnostics[name]["cache"]
            totals[name]["adaptive_expert_loads"] += stats["expert_loads"]
            totals[name]["substituted_tokens"] += (
                stats["substituted_token_fraction"] * input_ids.numel()
            )
            totals[name]["route_overlap_sum"] += (
                stats["original_route_overlap"] * input_ids.numel()
            )
            for block_index, block_stats in enumerate(stats["per_block"]):
                target = totals[name]["per_block"][block_index]
                target["adaptive_expert_loads"] += block_stats["expert_loads"]
                target["substituted_tokens"] += (
                    block_stats["substituted_token_fraction"] * args.block_size
                )
                target["route_overlap_sum"] += (
                    block_stats["original_route_overlap"] * args.block_size
                )
        layer_reports.append(
            {"layer": layer_index, **metadata, "policies": diagnostics}
        )
        compact = " ".join(
            f"{name}:m{diagnostics[name]['cache']['cache_miss_fraction']:.3f}"
            for name in names
        )
        print(f"layer={layer_index:02d} {compact}", flush=True)
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    token_layer_count = input_ids.numel() * 26
    for name in names:
        total = totals[name]
        total["strict_expert_loads"] = strict_total["expert_loads"]
        total["expert_load_reduction_fraction"] = 1.0 - (
            total["adaptive_expert_loads"] / strict_total["expert_loads"]
        )
        total["cache_miss_fraction"] = total["adaptive_expert_loads"] / (
            token_layer_count * 6
        )
        total["strict_cache_miss_fraction"] = strict_total["expert_loads"] / (
            token_layer_count * 6
        )
        total["substituted_token_fraction"] = (
            total.pop("substituted_tokens") / token_layer_count
        )
        total["original_route_overlap"] = (
            total.pop("route_overlap_sum") / token_layer_count
        )
        for block_index, block_stats in enumerate(total["per_block"]):
            strict_loads = strict_total["per_block"][block_index]
            block_stats["strict_expert_loads"] = strict_loads
            block_stats["expert_load_reduction_fraction"] = 1.0 - (
                block_stats["adaptive_expert_loads"] / strict_loads
            )
            block_stats["cache_miss_fraction"] = (
                block_stats["adaptive_expert_loads"]
                / block_stats["token_expert_decisions"]
                / 6
            )
            block_stats["substituted_token_fraction"] = (
                block_stats.pop("substituted_tokens")
                / block_stats["token_expert_decisions"]
            )
            block_stats["original_route_overlap"] = (
                block_stats.pop("route_overlap_sum")
                / block_stats["token_expert_decisions"]
            )

    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(
        device
    )
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    final: dict[str, dict[str, object]] = {}
    for split_index, split in enumerate(("validation", "test")):
        block_start = split_index * args.blocks_per_split
        block_stop = (split_index + 1) * args.blocks_per_split
        teacher_split = teacher[block_start:block_stop].cpu()
        final[split] = {}
        for policy_index, name in enumerate(names):
            candidate_split = students[name][block_start:block_stop].cpu()
            aggregate = behavioral_metrics(
                teacher_split,
                candidate_split,
                split_ids[split],
                norm_weight,
                lm_head,
            )
            per_block = [
                behavioral_metrics(
                    teacher_split[index : index + 1],
                    candidate_split[index : index + 1],
                    split_ids[split][index : index + 1],
                    norm_weight,
                    lm_head,
                    block_batch=1,
                )
                for index in range(args.blocks_per_split)
            ]
            cache_blocks = totals[name]["per_block"][block_start:block_stop]
            final[split][name] = {
                **aggregate,
                "cache_miss_fraction": sum(
                    row["adaptive_expert_loads"] for row in cache_blocks
                )
                / (args.blocks_per_split * args.block_size * 26 * 6),
                "expert_load_reduction_fraction": 1.0
                - sum(row["adaptive_expert_loads"] for row in cache_blocks)
                / sum(row["strict_expert_loads"] for row in cache_blocks),
                "per_block": per_block,
                "bootstrap_95_percent_intervals": bootstrap_quality_intervals(
                    per_block, seed=20260810 + split_index * 1000 + policy_index
                ),
                "cache_bootstrap_95_percent_intervals": bootstrap_cache_intervals(
                    [
                        {
                            **row,
                            "substituted_tokens": row[
                                "substituted_token_fraction"
                            ]
                            * row["token_expert_decisions"],
                        }
                        for row in cache_blocks
                    ],
                    seed=20260811 + split_index * 1000 + policy_index,
                ),
            }

    validation_rows = final["validation"]
    pareto = _pareto_front(validation_rows)
    eligible = [
        name
        for name in names
        if validation_rows[name]["teacher_to_candidate_kl"]
        <= args.validation_kl_limit
        and validation_rows[name]["relative_cross_entropy_delta"]
        <= args.validation_relative_ce_limit
    ]
    chosen = (
        min(eligible, key=lambda name: validation_rows[name]["cache_miss_fraction"])
        if eligible
        else None
    )
    max_control_error = max(
        row["official_gate_control"]["max_abs_error"]
        for row in layer_reports
    )
    report = {
        "status": "complete",
        "experiment": "modelwide_cache_routing_paper_baseline_pareto",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "corpus_preset": args.corpus_preset,
        "corpus_sources": corpus_sources,
        "blocks_per_split": args.blocks_per_split,
        "block_size": args.block_size,
        "token_offset_per_split": args.token_offset,
        "capacity_per_layer": args.capacity,
        "policies": {
            policy.name: {
                "method": policy.method,
                "top_j": policy.top_j,
                "parameter": policy.parameter,
            }
            for policy in policies
        },
        "implementation": {
            "cache": "independent empty LRU per sequence block and MoE layer",
            "parallel_route_eviction_order": "higher original router weight becomes older and is evicted first",
            "cache_prior_delta": "per-layer mean raw-logit range calibrated only on teacher validation tokens",
            "selection_logits": "reranked logits choose IDs; unmodified router probabilities weight selected expert outputs",
            "execution": "gate IDs are injected into the complete official DeepSeek decoder-layer forward and official moe_infer kernel",
            "strict_baseline": "official unmodified teacher top-6 routes with an independent LRU cache",
            "cache_caveat": "expert-granularity simulator; no byte transfers, prefetch, concurrency, or kernel latency modeled",
        },
        "numerical_validation": {
            "official_original_control_max_abs_across_layers": max_control_error,
            "per_layer_metrics_in_layer_reports": True,
        },
        "total_cache_statistics": totals,
        "validation_pareto_front": pareto,
        "validation_selection": {
            "kl_limit": args.validation_kl_limit,
            "relative_ce_limit": args.validation_relative_ce_limit,
            "eligible": eligible,
            "chosen": chosen,
            "test_not_used_for_selection": True,
        },
        "layer_reports": layer_reports,
        "final": final,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        args.report_name, envelope("cache_routing_pareto", report)
    )
    print(path)
    print(
        {
            "validation_pareto_front": pareto,
            "chosen": chosen,
            "validation": {
                name: {
                    "miss": validation_rows[name]["cache_miss_fraction"],
                    "load_reduction": validation_rows[name][
                        "expert_load_reduction_fraction"
                    ],
                    "kl": validation_rows[name]["teacher_to_candidate_kl"],
                    "relative_ce": validation_rows[name][
                        "relative_cross_entropy_delta"
                    ],
                }
                for name in names
            },
        }
    )
