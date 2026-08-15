from __future__ import annotations

import argparse
import itertools
import math
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from moe_lab.behavioral import rmsnorm
from moe_lab.metrics import regression_metrics
from moe_lab.moe_layer import loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


TOKENS_PER_SPLIT = 256
THRESHOLDS = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-split", type=int, default=TOKENS_PER_SPLIT)
    parser.add_argument(
        "--artifact-name", default="layer26_route_equivalence.safetensors"
    )
    parser.add_argument("--report-name", default="layer26_route_equivalence.json")
    return parser.parse_args()


@torch.inference_mode()
def top_candidate_outputs(layer, inputs: torch.Tensor):
    moe = loaded_moe_from_official_module(layer.mlp, layer=26)
    scores = F.linear(inputs.float(), moe.gate_weight.float()).softmax(-1)
    top_weights, top_ids = torch.topk(scores, 12, dim=-1, sorted=True)
    if moe.norm_topk_prob:
        raise RuntimeError("experiment assumes the pinned unnormalized router")
    top_weights = top_weights * moe.routed_scaling_factor
    outputs = torch.empty(
        inputs.shape[0], 12, inputs.shape[-1], dtype=inputs.dtype, device=inputs.device
    )
    for expert_id, expert in enumerate(moe.experts):
        positions = (top_ids == expert_id).nonzero(as_tuple=False)
        if positions.numel():
            token_indices = positions[:, 0]
            slots = positions[:, 1]
            outputs[token_indices, slots] = moe.expert_forward(
                inputs[token_indices], expert
            )
    return top_ids, top_weights, outputs


@torch.inference_mode()
def exact_subset_kl(
    reference_hidden: torch.Tensor,
    post_attention: torch.Tensor,
    shared: torch.Tensor,
    outputs: torch.Tensor,
    weights: torch.Tensor,
    subsets: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
) -> torch.Tensor:
    damage = torch.empty(reference_hidden.shape[0], subsets.shape[0])
    for token in range(reference_hidden.shape[0]):
        selected_outputs = outputs[token, subsets]
        selected_weights = weights[token, subsets]
        routed = (selected_outputs.float() * selected_weights.unsqueeze(-1)).sum(1)
        candidates = post_attention[token].unsqueeze(0) + (
            routed.to(outputs.dtype) + shared[token].unsqueeze(0)
        )
        reference_logits = F.linear(
            rmsnorm(reference_hidden[token : token + 1], norm_weight), lm_head
        ).float()
        candidate_logits = F.linear(rmsnorm(candidates, norm_weight), lm_head).float()
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
        damage[token] = (
            reference_log_probs.exp()
            * (reference_log_probs - candidate_log_probs)
        ).sum(-1).cpu()
        if token % 16 == 15:
            print(f"route_tokens={token + 1}/{reference_hidden.shape[0]}", flush=True)
    return damage


if __name__ == "__main__":
    args = parse_args()
    if not 1 <= args.tokens_per_split <= 1024:
        raise ValueError("tokens-per-split must be in [1, 1024]")
    if not torch.cuda.is_available():
        raise RuntimeError("route-equivalence experiment requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    components_all = load_file(
        ROOT / "data" / "traces" / "layer26_dynamic_precision_components.safetensors",
        device="cpu",
    )
    indices = torch.cat(
        (
            torch.arange(args.tokens_per_split),
            torch.arange(1024, 1024 + args.tokens_per_split),
        )
    )
    components = {
        key: value[indices].to(device) for key, value in components_all.items()
    }
    layer, _ = load_decoder_layer(model_dir, 26, device)
    top_ids, top_weights, outputs = top_candidate_outputs(
        layer, components["moe_input"]
    )
    subsets = torch.tensor(
        list(itertools.combinations(range(12), 6)), dtype=torch.long, device=device
    )
    original_subset = torch.arange(6, device=device)
    original_index = int(
        (subsets == original_subset).all(1).nonzero(as_tuple=False).item()
    )
    original_outputs = outputs[:, :6]
    original_weights = top_weights[:, :6]
    original_routed = (
        original_outputs.float() * original_weights.unsqueeze(-1)
    ).sum(1).to(outputs.dtype)
    manual_reference = components["post_attention"] + (
        original_routed + components["shared"]
    )
    reconstruction = regression_metrics(
        manual_reference.cpu(), components["teacher"].cpu()
    )
    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    damage = exact_subset_kl(
        manual_reference,
        components["post_attention"],
        components["shared"],
        outputs,
        top_weights,
        subsets,
        norm_weight,
        lm_head,
    )
    artifact_path = ROOT / "data" / "traces" / args.artifact_name
    save_file(
        {
            "subset_kl": damage.contiguous(),
            "top12_expert_ids": top_ids.cpu().contiguous(),
            "top12_router_weights": top_weights.cpu().contiguous(),
            "subsets": subsets.cpu().contiguous(),
        },
        artifact_path,
        metadata={
            "splits": (
                f"validation:first{args.tokens_per_split},"
                f"test:first{args.tokens_per_split}"
            ),
            "reference": "manual exact original top6 in same expert-output batch",
        },
    )
    original_damage = damage[:, original_index]
    intersections = (subsets < 6).sum(1).cpu()
    jaccard = intersections.float() / (12 - intersections).float()
    alternative_mask = torch.ones(subsets.shape[0], dtype=torch.bool)
    alternative_mask[original_index] = False
    split_results = {}
    for split_index, split in enumerate(("validation", "test")):
        sl = slice(
            split_index * args.tokens_per_split,
            (split_index + 1) * args.tokens_per_split,
        )
        split_damage = damage[sl]
        threshold_rows = {}
        for threshold in THRESHOLDS:
            counts = (split_damage <= threshold).sum(1)
            alternatives = (counts - 1).clamp_min(0)
            threshold_rows[str(threshold)] = {
                "tokens_with_at_least_one_alternative_fraction": float(
                    (alternatives > 0).float().mean().item()
                ),
                "mean_alternative_count": float(alternatives.float().mean().item()),
                "median_alternative_count": float(alternatives.float().median().item()),
                "mean_route_equivalence_entropy_bits_including_original": float(
                    counts.float().clamp_min(1).log2().mean().item()
                ),
            }
        best_alternative = split_damage[:, alternative_mask].min(1).values
        low_overlap = jaccard <= 0.5
        best_low_overlap = split_damage[:, low_overlap].min(1).values
        disjoint_index = int(
            (subsets == torch.arange(6, 12, device=device))
            .all(1)
            .nonzero(as_tuple=False)
            .item()
        )
        disjoint = split_damage[:, disjoint_index]
        split_results[split] = {
            "thresholds": threshold_rows,
            "best_alternative_kl": {
                "mean": float(best_alternative.mean().item()),
                "median": float(best_alternative.median().item()),
                "p95": float(torch.quantile(best_alternative, 0.95).item()),
            },
            "best_jaccard_at_most_0_5_kl": {
                "mean": float(best_low_overlap.mean().item()),
                "median": float(best_low_overlap.median().item()),
                "p95": float(torch.quantile(best_low_overlap, 0.95).item()),
            },
            "fully_disjoint_route_kl": {
                "mean": float(disjoint.mean().item()),
                "median": float(disjoint.median().item()),
                "p95": float(torch.quantile(disjoint, 0.95).item()),
            },
        }

    report = {
        "status": "complete",
        "experiment": "layer26_top12_choose6_route_equivalence",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "tokens_per_split": args.tokens_per_split,
        "candidate_route_count": subsets.shape[0],
        "weights": "original raw softmax router weights; no top-k renormalization and no coefficient refit",
        "reference": "manual reconstruction using original top-6 from the same top-12 expert-output batch",
        "manual_reference_vs_official_teacher": reconstruction,
        "original_subset_numerical_kl": {
            "maximum": float(original_damage.max().item()),
            "mean": float(original_damage.mean().item()),
        },
        "artifact": str(artifact_path.resolve()),
        "results": split_results,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        args.report_name,
        envelope("route_equivalence", report),
    )
    print(path)
    for split in ("validation", "test"):
        print(split, split_results[split])
