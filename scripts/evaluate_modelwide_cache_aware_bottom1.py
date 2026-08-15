from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from estimate_layer26_observability import behavioral_metrics, corpus_blocks, forward_layer
from evaluate_layer23_route_equivalence_downstream import forward_with_router
from evaluate_layer26_dynamic_precision_oracle import layer_components
from moe_lab.aggregate_student import dense_router_features
from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


BLOCK_SIZE = 128
CAPACITIES = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-per-split", type=int, default=2)
    parser.add_argument(
        "--capacity",
        type=int,
        action="append",
        choices=CAPACITIES,
        help="repeat to evaluate selected cache capacities",
    )
    parser.add_argument(
        "--report-name", default="modelwide_cache_aware_bottom1.json"
    )
    parser.add_argument(
        "--corpus-preset",
        choices=("wikitext", "local_diverse"),
        default="wikitext",
    )
    return parser.parse_args()


def token_blocks_from_text(
    tokenizer: Tokenizer, text: str, blocks: int
) -> torch.Tensor:
    ids = tokenizer.encode(text).ids[: blocks * BLOCK_SIZE]
    if len(ids) < blocks * BLOCK_SIZE:
        raise RuntimeError(
            f"corpus has {len(ids)} tokens, fewer than requested {blocks * BLOCK_SIZE}"
        )
    return torch.tensor(ids, dtype=torch.long).view(blocks, BLOCK_SIZE)


def local_diverse_blocks(model_dir, blocks: int):
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    attachment_paths = (
        "C:/Users/de_do/.codex/attachments/35a0d2b1-5831-4a76-afe7-97aa4662286d/pasted-text.txt",
        "C:/Users/de_do/.codex/attachments/7bdae3b8-a66e-43e0-8f2f-8dd1cee42181/pasted-text.txt",
        "C:/Users/de_do/.codex/attachments/49e95e1b-41bd-45ec-8317-318bdb2af4d2/pasted-text.txt",
    )
    instruction_text = "\n\n".join(
        Path(path).read_text(encoding="utf-8") for path in attachment_paths
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
        "validation": token_blocks_from_text(tokenizer, instruction_text, blocks),
        "test": token_blocks_from_text(tokenizer, code_text, blocks),
    }, {
        "validation": "three user-supplied Dutch research/instruction attachments",
        "test": "Python source under workspace scripts, src, and tests",
    }


def touch_route(cache: list[int], route: list[int], capacity: int) -> int:
    misses = 0
    for expert in route:
        if expert in cache:
            cache.remove(expert)
        else:
            misses += 1
        cache.append(expert)
        if len(cache) > capacity:
            cache.pop(0)
    return misses


def cache_aware_positions(
    top7_ids: torch.Tensor,
    batch: int,
    sequence: int,
    capacity: int,
    strict_reference_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    ids = top7_ids.view(batch, sequence, 7).cpu()
    strict_ids = (
        ids[..., :6]
        if strict_reference_ids is None
        else strict_reference_ids.view(batch, sequence, 6).cpu()
    )
    positions = torch.arange(6).view(1, 1, 6).expand(batch, sequence, 6).clone()
    strict_loads = 0
    adaptive_loads = 0
    substitutions = 0
    per_block = []
    for block in range(batch):
        strict_cache: list[int] = []
        adaptive_cache: list[int] = []
        block_strict_loads = 0
        block_adaptive_loads = 0
        block_substitutions = 0
        for token in range(sequence):
            original = ids[block, token, :6].tolist()
            strict_route = strict_ids[block, token].tolist()
            misses = touch_route(strict_cache, strict_route, capacity)
            strict_loads += misses
            block_strict_loads += misses
            alternative_route = original.copy()
            alternative = int(ids[block, token, 6].item())
            alternative_route[-1] = alternative
            original_misses = touch_route(list(adaptive_cache), original, capacity)
            alternative_misses = touch_route(
                list(adaptive_cache), alternative_route, capacity
            )
            if alternative_misses < original_misses:
                chosen = alternative_route
                positions[block, token, -1] = 6
                substitutions += 1
                block_substitutions += 1
            else:
                chosen = original
            misses = touch_route(adaptive_cache, chosen, capacity)
            adaptive_loads += misses
            block_adaptive_loads += misses
        per_block.append(
            {
                "strict_expert_loads": block_strict_loads,
                "adaptive_expert_loads": block_adaptive_loads,
                "expert_load_reduction_fraction": (
                    1.0 - block_adaptive_loads / block_strict_loads
                ),
                "substituted_token_fraction": block_substitutions / sequence,
            }
        )
    tokens = batch * sequence
    return positions.view(tokens, 6).to(top7_ids.device), {
        "strict_expert_loads": strict_loads,
        "adaptive_expert_loads": adaptive_loads,
        "expert_load_reduction_fraction": 1.0 - adaptive_loads / strict_loads,
        "substituted_token_fraction": substitutions / tokens,
        "per_block": per_block,
    }


def _percentile_interval(values: torch.Tensor) -> list[float]:
    return [
        float(torch.quantile(values, 0.025).item()),
        float(torch.quantile(values, 0.975).item()),
    ]


def bootstrap_quality_intervals(
    rows: list[dict[str, float]], seed: int, resamples: int = 10_000
) -> dict[str, object]:
    values = torch.tensor(
        [
            [
                row["teacher_to_candidate_kl"],
                row["top1_agreement"],
                row["teacher_cross_entropy"],
                row["candidate_cross_entropy"],
            ]
            for row in rows
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0, len(rows), (resamples, len(rows)), generator=generator
    )
    sampled = values[indices].mean(dim=1)
    relative_ce = (sampled[:, 3] - sampled[:, 2]) / sampled[:, 2]
    ce_delta = sampled[:, 3] - sampled[:, 2]
    return {
        "method": "sequence-block bootstrap with replacement",
        "confidence_level": 0.95,
        "resamples": resamples,
        "sampling_units": len(rows),
        "teacher_to_candidate_kl": _percentile_interval(sampled[:, 0]),
        "top1_agreement": _percentile_interval(sampled[:, 1]),
        "cross_entropy_delta": _percentile_interval(ce_delta),
        "relative_cross_entropy_delta": _percentile_interval(relative_ce),
    }


def bootstrap_cache_intervals(
    rows: list[dict[str, float]], seed: int, resamples: int = 10_000
) -> dict[str, object]:
    values = torch.tensor(
        [
            [
                row["strict_expert_loads"],
                row["adaptive_expert_loads"],
                row["substituted_tokens"],
                row["token_expert_decisions"],
            ]
            for row in rows
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0, len(rows), (resamples, len(rows)), generator=generator
    )
    sampled = values[indices].sum(dim=1)
    reduction = 1.0 - sampled[:, 1] / sampled[:, 0]
    substitution = sampled[:, 2] / sampled[:, 3]
    return {
        "method": "sequence-block bootstrap with replacement",
        "confidence_level": 0.95,
        "resamples": resamples,
        "sampling_units": len(rows),
        "expert_load_reduction_fraction": _percentile_interval(reduction),
        "substituted_token_fraction": _percentile_interval(substitution),
    }


@torch.inference_mode()
def forward_cache_aware(
    layer,
    hidden_states,
    capacity: int,
    strict_reference_ids: torch.Tensor | None = None,
):
    post_attention, moe_input = layer_components(layer, hidden_states)
    moe = loaded_moe_from_official_module(layer.mlp, layer=0)
    flat_input = moe_input.reshape(-1, moe_input.shape[-1])
    scores = F.linear(flat_input.float(), moe.gate_weight.float()).softmax(-1)
    top_weights, top_ids = torch.topk(scores, 7, dim=-1, sorted=True)
    positions, cache_stats = cache_aware_positions(
        top_ids,
        hidden_states.shape[0],
        hidden_states.shape[1],
        capacity,
        strict_reference_ids,
    )
    router_ids = top_ids.gather(1, positions)
    router_weights = top_weights.gather(1, positions)
    if moe.norm_topk_prob:
        router_weights = router_weights / router_weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-20)
    else:
        router_weights = router_weights * moe.routed_scaling_factor
    selected = torch.empty(
        flat_input.shape[0],
        6,
        flat_input.shape[-1],
        dtype=flat_input.dtype,
        device=flat_input.device,
    )
    for expert_id, expert in enumerate(moe.experts):
        locations = (router_ids == expert_id).nonzero(as_tuple=False)
        if locations.numel():
            token_indices = locations[:, 0]
            slots = locations[:, 1]
            selected[token_indices, slots] = moe.expert_forward(
                flat_input[token_indices], expert
            )
    routed = (
        selected.float() * router_weights.unsqueeze(-1)
    ).sum(1).to(flat_input.dtype)
    shared = moe.expert_forward(flat_input, moe.shared)
    output = post_attention.reshape(-1, post_attention.shape[-1]) + (routed + shared)
    return output.view_as(hidden_states), router_ids, router_weights, cache_stats


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("model-wide cache policy requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    capacities = tuple(args.capacity) if args.capacity else CAPACITIES
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    if args.corpus_preset == "wikitext":
        split_ids = {
            split: corpus_blocks(model_dir, split, args.blocks_per_split)
            for split in ("validation", "test")
        }
        corpus_sources = {
            "validation": "pinned WikiText-2 raw validation",
            "test": "pinned WikiText-2 raw test",
        }
    else:
        split_ids, corpus_sources = local_diverse_blocks(
            model_dir, args.blocks_per_split
        )
    input_ids = torch.cat((split_ids["validation"], split_ids["test"]), dim=0)
    embeddings = load_token_embeddings(model_dir, input_ids, device)
    layer0, _ = load_decoder_layer(model_dir, 0, device)
    teacher = forward_layer(layer0, embeddings)
    students = {capacity: teacher.clone() for capacity in capacities}
    del layer0, embeddings
    gc.collect()
    torch.cuda.empty_cache()

    total_cache_stats = {
        capacity: {
            "strict_expert_loads": 0,
            "adaptive_expert_loads": 0,
            "substituted_tokens": 0.0,
            "per_block": [
                {
                    "strict_expert_loads": 0,
                    "adaptive_expert_loads": 0,
                    "substituted_tokens": 0.0,
                    "token_expert_decisions": BLOCK_SIZE * 26,
                }
                for _ in range(input_ids.shape[0])
            ],
        }
        for capacity in capacities
    }
    layer_reports = []
    for layer_idx in range(1, 27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        teacher, teacher_ids, teacher_weights = forward_with_router(layer, teacher)
        capacity_rows = {}
        for capacity in capacities:
            student, student_ids, student_weights, cache_stats = forward_cache_aware(
                layer, students[capacity], capacity, teacher_ids
            )
            students[capacity] = student
            total_cache_stats[capacity]["strict_expert_loads"] += cache_stats[
                "strict_expert_loads"
            ]
            total_cache_stats[capacity]["adaptive_expert_loads"] += cache_stats[
                "adaptive_expert_loads"
            ]
            total_cache_stats[capacity]["substituted_tokens"] += (
                cache_stats["substituted_token_fraction"] * input_ids.numel()
            )
            for block_index, block_stats in enumerate(cache_stats["per_block"]):
                total_block = total_cache_stats[capacity]["per_block"][block_index]
                total_block["strict_expert_loads"] += block_stats[
                    "strict_expert_loads"
                ]
                total_block["adaptive_expert_loads"] += block_stats[
                    "adaptive_expert_loads"
                ]
                total_block["substituted_tokens"] += (
                    block_stats["substituted_token_fraction"] * BLOCK_SIZE
                )
            capacity_rows[str(capacity)] = {
                "hidden": regression_metrics(student.cpu(), teacher.cpu()),
                "router_topk_overlap": topk_overlap(
                    student_ids.cpu(), teacher_ids.cpu()
                ),
                "router_weight_nrmse": regression_metrics(
                    dense_router_features(
                        student_ids.cpu(), student_weights.cpu(), 64
                    ),
                    dense_router_features(
                        teacher_ids.cpu(), teacher_weights.cpu(), 64
                    ),
                )["nrmse"],
                "cache": cache_stats,
            }
        layer_reports.append({"layer": layer_idx, "capacities": capacity_rows})
        print(
            f"layer={layer_idx:02d} "
            + " ".join(
                f"c{capacity}:load-{capacity_rows[str(capacity)]['cache']['expert_load_reduction_fraction']:.3f}"
                for capacity in capacities
            ),
            flush=True,
        )
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    total_tokens_across_moe_layers = input_ids.numel() * 26
    for capacity in capacities:
        stats = total_cache_stats[capacity]
        stats["expert_load_reduction_fraction"] = 1.0 - (
            stats["adaptive_expert_loads"] / stats["strict_expert_loads"]
        )
        stats["substituted_token_fraction"] = (
            stats.pop("substituted_tokens") / total_tokens_across_moe_layers
        )
        for block_stats in stats["per_block"]:
            block_stats["expert_load_reduction_fraction"] = 1.0 - (
                block_stats["adaptive_expert_loads"]
                / block_stats["strict_expert_loads"]
            )
            block_stats["substituted_token_fraction"] = (
                block_stats["substituted_tokens"]
                / block_stats["token_expert_decisions"]
            )

    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    final = {}
    for split_index, split in enumerate(("validation", "test")):
        block_start = split_index * args.blocks_per_split
        block_stop = (split_index + 1) * args.blocks_per_split
        teacher_split = teacher[block_start:block_stop].cpu()
        final[split] = {}
        for capacity in capacities:
            aggregate_metrics = behavioral_metrics(
                teacher_split,
                students[capacity][block_start:block_stop].cpu(),
                split_ids[split],
                norm_weight,
                lm_head,
            )
            block_metrics = [
                behavioral_metrics(
                    teacher_split[block_index : block_index + 1],
                    students[capacity][
                        block_start + block_index : block_start + block_index + 1
                    ].cpu(),
                    split_ids[split][block_index : block_index + 1],
                    norm_weight,
                    lm_head,
                    block_batch=1,
                )
                for block_index in range(args.blocks_per_split)
            ]
            cache_blocks = total_cache_stats[capacity]["per_block"][
                block_start:block_stop
            ]
            final[split][str(capacity)] = {
                **aggregate_metrics,
                "per_block": block_metrics,
                "bootstrap_95_percent_intervals": bootstrap_quality_intervals(
                    block_metrics, seed=20260809 + split_index * 100 + capacity
                ),
                "cache_bootstrap_95_percent_intervals": bootstrap_cache_intervals(
                    cache_blocks, seed=20260810 + split_index * 100 + capacity
                ),
            }

    report = {
        "status": "complete",
        "experiment": "modelwide_cache_aware_bottom1_substitution",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "blocks_per_split": args.blocks_per_split,
        "corpus_preset": args.corpus_preset,
        "corpus_sources": corpus_sources,
        "policy": "compare exact within-token LRU misses for original top6 versus top5+rank7; substitute only when the latter has fewer misses",
        "strict_cache_baseline": "unmodified teacher top6 routes with an independent LRU cache",
        "cache_caveat": "expert-granularity simulator; no bytes, prefetch, concurrency, or kernel latency modeled",
        "total_cache_statistics": {
            str(capacity): stats for capacity, stats in total_cache_stats.items()
        },
        "layer_reports": layer_reports,
        "final": final,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        args.report_name,
        envelope("route_equivalent_cache", report),
    )
    print(path)
    print(total_cache_stats)
    print(final)
