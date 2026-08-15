from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from estimate_layer26_observability import behavioral_metrics, corpus_blocks, forward_layer
from evaluate_layer23_route_equivalence_downstream import forward_with_router
from evaluate_layer26_dynamic_precision_oracle import layer_components
from moe_lab.aggregate_student import dense_router_features
from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.reporting import ROOT, envelope, write_json


BLOCK_SIZE = 128
POLICIES = {
    "replace_bottom_one_all_moe_layers": {
        "positions": (0, 1, 2, 3, 4, 6),
        "layers": set(range(1, 27)),
    },
    "replace_bottom_two_all_moe_layers": {
        "positions": (0, 1, 2, 3, 6, 7),
        "layers": set(range(1, 27)),
    },
    "replace_bottom_two_layers4_to23": {
        "positions": (0, 1, 2, 3, 6, 7),
        "layers": set(range(4, 24)),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-per-split", type=int, default=2)
    parser.add_argument(
        "--policy",
        action="append",
        choices=tuple(POLICIES),
        help="repeat to evaluate a subset; omit to evaluate every policy",
    )
    parser.add_argument("--report-name", default="modelwide_rank_substitution.json")
    return parser.parse_args()


@torch.inference_mode()
def forward_rank_substitution(layer, hidden_states, positions):
    post_attention, moe_input = layer_components(layer, hidden_states)
    moe = loaded_moe_from_official_module(layer.mlp, layer=0)
    flat_input = moe_input.reshape(-1, moe_input.shape[-1])
    scores = F.linear(flat_input.float(), moe.gate_weight.float()).softmax(-1)
    candidate_count = max(positions) + 1
    candidate_weights, candidate_ids = torch.topk(
        scores, candidate_count, dim=-1, sorted=True
    )
    chosen_positions = torch.tensor(positions, device=flat_input.device)
    router_ids = candidate_ids[:, chosen_positions]
    router_weights = candidate_weights[:, chosen_positions]
    if moe.norm_topk_prob:
        router_weights = router_weights / router_weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-20)
    else:
        router_weights = router_weights * moe.routed_scaling_factor
    selected = torch.empty(
        flat_input.shape[0],
        len(positions),
        flat_input.shape[-1],
        dtype=flat_input.dtype,
        device=flat_input.device,
    )
    for expert_id, expert in enumerate(moe.experts):
        selected_positions = (router_ids == expert_id).nonzero(as_tuple=False)
        if selected_positions.numel():
            token_indices = selected_positions[:, 0]
            slots = selected_positions[:, 1]
            selected[token_indices, slots] = moe.expert_forward(
                flat_input[token_indices], expert
            )
    routed = (
        selected.float() * router_weights.unsqueeze(-1)
    ).sum(1).to(flat_input.dtype)
    shared = moe.expert_forward(flat_input, moe.shared)
    output = post_attention.reshape(-1, post_attention.shape[-1]) + (routed + shared)
    return output.view_as(hidden_states), router_ids, router_weights


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("model-wide route substitution requires CUDA")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    policies_to_run = (
        {name: POLICIES[name] for name in args.policy}
        if args.policy
        else POLICIES
    )
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    split_ids = {
        split: corpus_blocks(model_dir, split, args.blocks_per_split)
        for split in ("validation", "test")
    }
    input_ids = torch.cat((split_ids["validation"], split_ids["test"]), dim=0)
    embeddings = load_token_embeddings(model_dir, input_ids, device)
    layer0, _ = load_decoder_layer(model_dir, 0, device)
    teacher = forward_layer(layer0, embeddings)
    students = {name: teacher.clone() for name in policies_to_run}
    del layer0, embeddings
    gc.collect()
    torch.cuda.empty_cache()

    layer_reports = []
    for layer_idx in range(1, 27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        teacher, teacher_ids, teacher_weights = forward_with_router(layer, teacher)
        policies = {}
        for name, policy in policies_to_run.items():
            if layer_idx in policy["layers"]:
                student, student_ids, student_weights = forward_rank_substitution(
                    layer, students[name], policy["positions"]
                )
            else:
                student, student_ids, student_weights = forward_with_router(
                    layer, students[name]
                )
            students[name] = student
            policies[name] = {
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
            }
        layer_reports.append({"layer": layer_idx, "policies": policies})
        print(
            f"layer={layer_idx:02d} "
            + " ".join(
                f"{name}={policies[name]['hidden']['nrmse']:.4f}"
                for name in policies_to_run
            ),
            flush=True,
        )
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    final = {}
    for split_index, split in enumerate(("validation", "test")):
        block_start = split_index * args.blocks_per_split
        block_stop = (split_index + 1) * args.blocks_per_split
        teacher_split = teacher[block_start:block_stop].cpu()
        final[split] = {}
        for name in policies_to_run:
            candidate = students[name][block_start:block_stop].cpu()
            final[split][name] = behavioral_metrics(
                teacher_split,
                candidate,
                split_ids[split],
                norm_weight,
                lm_head,
            )

    report = {
        "status": "complete",
        "experiment": "modelwide_fixed_router_rank_substitution",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "blocks_per_split": args.blocks_per_split,
        "policies": {
            name: {
                "selected_top_candidate_positions_zero_based": list(policy["positions"]),
                "modified_layers": sorted(policy["layers"]),
                "teacher_free": True,
                "router_weights": "original raw softmax weights; no renormalization",
            }
            for name, policy in policies_to_run.items()
        },
        "layer_reports": layer_reports,
        "final": final,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        args.report_name,
        envelope("route_substitution", report),
    )
    print(path)
    print(final)
