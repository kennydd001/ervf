from __future__ import annotations

import gc

import torch

from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_, packed_quantized_bytes
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import load_trace


PARAMETERS_PER_EXPERT = 8_650_752
ROUTED_EXPERT_PARAMETERS = 64 * PARAMETERS_PER_EXPERT
ORIGINAL_BF16_BYTES = ROUTED_EXPERT_PARAMETERS * 2
ROWS_PER_EXPERT = 1408 + 1408 + 2048
POLICIES = (
    {"hot_bits": 3, "cold_bits": 1, "hot_experts": 24},
    {"hot_bits": 3, "cold_bits": 1, "hot_experts": 32},
    {"hot_bits": 4, "cold_bits": 1, "hot_experts": 16},
    {"hot_bits": 4, "cold_bits": 1, "hot_experts": 21},
    {"hot_bits": 8, "cold_bits": 1, "hot_experts": 9},
)


def expert_importance(trace) -> torch.Tensor:
    importance = torch.zeros(64, dtype=torch.float64)
    importance.scatter_add_(
        0,
        trace.router_ids.long().reshape(-1),
        trace.router_weights.double().reshape(-1),
    )
    return importance


def quantize_policy(
    layer: LoadedMoELayer, hot_ids: set[int], hot_bits: int, cold_bits: int
) -> tuple[int, int]:
    total_weight_bits = 0
    scale_count = 0
    for expert_id, expert in enumerate(layer.experts):
        bits = hot_bits if expert_id in hot_ids else cold_bits
        for weight in (expert.gate, expert.up, expert.down):
            scale_count += fake_quantize_symmetric_per_row_(weight, bits)
            total_weight_bits += weight.numel() * bits
    return total_weight_bits, scale_count


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("mixed quantization baseline requires CUDA")
    device = torch.device("cuda")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    traces = {
        split: load_trace(ROOT / "data" / "traces" / f"wikitext_{split}_layer_1.safetensors")
        for split in ("train", "validation", "test")
    }
    importance = expert_importance(traces["train"])
    ordering = torch.argsort(importance, descending=True).tolist()
    validation_rows = []
    for policy in POLICIES:
        hot_ids = set(ordering[: policy["hot_experts"]])
        layer = load_moe_layer(model_dir, 1, device)
        total_bits, scale_count = quantize_policy(
            layer, hot_ids, policy["hot_bits"], policy["cold_bits"]
        )
        student = layer.trace(traces["validation"].hidden_states)
        storage_bytes = (total_bits + 7) // 8 + scale_count * 2
        row = {
            **policy,
            "hot_expert_ids": sorted(hot_ids),
            "hot_train_router_mass_fraction": float(
                importance[list(hot_ids)].sum() / importance.sum()
            ),
            "scale_count": scale_count,
            "packed_storage_bytes": storage_bytes,
            "compression_ratio_vs_bf16_routed_bank": ORIGINAL_BF16_BYTES / storage_bytes,
            "validation": {
                **regression_metrics(student.routed_output, traces["validation"].routed_output),
                "router_topk_overlap": topk_overlap(
                    student.router_ids, traces["validation"].router_ids
                ),
            },
        }
        validation_rows.append(row)
        print(
            f"{policy} ratio={row['compression_ratio_vs_bf16_routed_bank']:.3f}x "
            f"mass={row['hot_train_router_mass_fraction']:.3f} "
            f"val={row['validation']['nrmse']:.6f}"
        )
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    eligible = [
        row for row in validation_rows if row["compression_ratio_vs_bf16_routed_bank"] >= 7.9
    ]
    selected = min(eligible, key=lambda row: row["validation"]["nrmse"])
    hot_ids = set(selected["hot_expert_ids"])
    layer = load_moe_layer(model_dir, 1, device)
    quantize_policy(layer, hot_ids, selected["hot_bits"], selected["cold_bits"])
    student = layer.trace(traces["test"].hidden_states)
    selected["test"] = {
        **regression_metrics(student.routed_output, traces["test"].routed_output),
        "router_topk_overlap": topk_overlap(student.router_ids, traces["test"].router_ids),
    }
    report = {
        "status": "complete",
        "method": "train_frequency_selected_mixed_precision_per_row_quantization",
        "selection": "best validation NRMSE among predeclared policies at >=7.9x",
        "scope": "routed experts only; router and shared experts remain exact",
        "execution": "fake quantization dequantized to BF16; packed storage estimate only",
        "expert_importance": importance.tolist(),
        "validation_policies": validation_rows,
        "selected_policy": selected,
    }
    path = write_json("mixed_weight_quantization_layer1.json", envelope("compression_baseline", report))
    print("selected", selected)
    print(path)
