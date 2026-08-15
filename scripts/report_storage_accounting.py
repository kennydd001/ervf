from __future__ import annotations

import math

from safetensors import safe_open

from moe_lab.reporting import ROOT, envelope, write_json


DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


if __name__ == "__main__":
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    total_bytes = 0
    routed_bytes = 0
    routed_parameters = 0
    routed_rows = 0
    routed_tensors = 0
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                view = handle.get_slice(name)
                shape = view.get_shape()
                elements = math.prod(shape)
                dtype = str(view.get_dtype())
                tensor_bytes = elements * DTYPE_BYTES[dtype]
                total_bytes += tensor_bytes
                if ".mlp.experts." in name:
                    routed_bytes += tensor_bytes
                    routed_parameters += elements
                    routed_rows += shape[0]
                    routed_tensors += 1
    non_routed_bytes = total_bytes - routed_bytes
    mixed_weight_bytes = (routed_parameters * 2 + 7) // 8
    mixed_scale_bytes = routed_rows * 2
    mixed_routed_bytes = mixed_weight_bytes + mixed_scale_bytes
    mixed_total_bytes = non_routed_bytes + mixed_routed_bytes
    uniform = {}
    for bits in (4, 3):
        weight_bytes = (routed_parameters * bits + 7) // 8
        quantized_routed_bytes = weight_bytes + mixed_scale_bytes
        quantized_total_bytes = non_routed_bytes + quantized_routed_bytes
        uniform[str(bits)] = {
            "routed_bytes": quantized_routed_bytes,
            "routed_compression_ratio": routed_bytes / quantized_routed_bytes,
            "full_checkpoint_bytes": quantized_total_bytes,
            "full_checkpoint_gib": quantized_total_bytes / (1024**3),
            "full_checkpoint_compression_ratio": total_bytes / quantized_total_bytes,
        }
    parameters_per_layer = routed_parameters // 26
    edge_weight_bits = parameters_per_layer * ((20 * 4) + (6 * 8))
    edge_routed_bytes = (edge_weight_bits + 7) // 8 + mixed_scale_bytes
    edge_total_bytes = non_routed_bytes + edge_routed_bytes
    report = {
        "status": "complete",
        "source": "exact safetensor headers",
        "checkpoint_tensor_bytes": total_bytes,
        "routed_expert_tensor_count": routed_tensors,
        "routed_expert_parameters": routed_parameters,
        "routed_expert_bf16_bytes": routed_bytes,
        "non_routed_checkpoint_bytes": non_routed_bytes,
        "routed_fraction_of_checkpoint": routed_bytes / total_bytes,
        "mixed_policy": "per layer: 32 experts at 3-bit and 32 experts at 1-bit, one BF16 scale per output row",
        "mixed_routed_weight_bytes": mixed_weight_bytes,
        "mixed_routed_scale_bytes": mixed_scale_bytes,
        "mixed_routed_total_bytes": mixed_routed_bytes,
        "mixed_routed_compression_ratio": routed_bytes / mixed_routed_bytes,
        "mixed_full_checkpoint_bytes": mixed_total_bytes,
        "mixed_full_checkpoint_gib": mixed_total_bytes / (1024**3),
        "mixed_full_checkpoint_compression_ratio": total_bytes / mixed_total_bytes,
        "uniform_per_row": uniform,
        "edge_policy_4bit_middle_8bit_first_last_three": {
            "routed_bytes": edge_routed_bytes,
            "routed_compression_ratio": routed_bytes / edge_routed_bytes,
            "full_checkpoint_bytes": edge_total_bytes,
            "full_checkpoint_gib": edge_total_bytes / (1024**3),
            "full_checkpoint_compression_ratio": total_bytes / edge_total_bytes,
        },
    }
    path = write_json("storage_accounting.json", envelope("storage_accounting", report))
    print(path)
    print(report)
