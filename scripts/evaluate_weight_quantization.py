from __future__ import annotations

import argparse
import gc

import torch

from moe_lab.metrics import regression_metrics, topk_overlap
from moe_lab.moe_layer import LoadedMoELayer, load_moe_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_, packed_quantized_bytes
from moe_lab.reporting import ROOT, envelope, write_json
from moe_lab.trace import load_trace


ROUTED_EXPERT_PARAMETERS = 64 * 8_650_752
ORIGINAL_BF16_BYTES = ROUTED_EXPERT_PARAMETERS * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2])
    return parser.parse_args()


def quantize_routed_experts(layer: LoadedMoELayer, bits: int) -> int:
    scale_count = 0
    for expert in layer.experts:
        scale_count += fake_quantize_symmetric_per_row_(expert.gate, bits)
        scale_count += fake_quantize_symmetric_per_row_(expert.up, bits)
        scale_count += fake_quantize_symmetric_per_row_(expert.down, bits)
    return scale_count


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("quantization baseline requires CUDA")
    device = torch.device("cuda")
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    traces = {
        split: load_trace(ROOT / "data" / "traces" / f"wikitext_{split}_layer_1.safetensors")
        for split in ("validation", "test")
    }
    rows = []
    for bits in args.bits:
        layer = load_moe_layer(model_dir, 1, device)
        scale_count = quantize_routed_experts(layer, bits)
        split_metrics = {}
        for split, teacher in traces.items():
            student = layer.trace(teacher.hidden_states)
            split_metrics[split] = {
                **regression_metrics(student.routed_output, teacher.routed_output),
                "router_topk_overlap": topk_overlap(student.router_ids, teacher.router_ids),
            }
        storage_bytes = packed_quantized_bytes(
            ROUTED_EXPERT_PARAMETERS, bits, scale_count
        )
        row = {
            "bits": bits,
            "scheme": "symmetric_per_output_row_fake_quantization",
            "scale_dtype": "bf16",
            "scale_count": scale_count,
            "packed_storage_bytes": storage_bytes,
            "compression_ratio_vs_bf16_routed_bank": ORIGINAL_BF16_BYTES / storage_bytes,
            "validation": split_metrics["validation"],
            "test": split_metrics["test"],
        }
        rows.append(row)
        print(
            f"bits={bits} ratio={row['compression_ratio_vs_bf16_routed_bank']:.3f}x "
            f"val={row['validation']['nrmse']:.6f} test={row['test']['nrmse']:.6f}"
        )
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "status": "complete",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "layer": 1,
        "teacher_routed_expert_parameters": ROUTED_EXPERT_PARAMETERS,
        "teacher_bf16_bytes": ORIGINAL_BF16_BYTES,
        "scope": "routed experts only; router and shared experts remain exact",
        "execution": "fake quantized weights dequantized to BF16; storage estimate only, not a packed-kernel latency benchmark",
        "results": rows,
    }
    path = write_json("weight_quantization_layer1.json", envelope("compression_baseline", report))
    print(path)
