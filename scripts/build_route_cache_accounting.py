from __future__ import annotations

import json
from pathlib import Path

from moe_lab.quantization import packed_quantized_bytes
from moe_lab.reporting import ROOT, envelope, write_json


BLOCK_SIZE = 128
REPORTS = {
    "wikitext": (
        ROOT
        / "reports"
        / "baseline"
        / "modelwide_cache_aware_bottom1_teacher_lru_capacity32_wikitext_1024_ci.json"
    ),
    "instructions_and_code": (
        ROOT
        / "reports"
        / "baseline"
        / "modelwide_cache_aware_bottom1_teacher_lru_capacity32_diverse_1024_ci.json"
    ),
}


def gib(value: int | float) -> float:
    return value / (1024**3)


def mib(value: int | float) -> float:
    return value / (1024**2)


def corpus_accounting(path: Path, bytes_per_expert: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    stats = payload["total_cache_statistics"]["32"]
    tokens = payload["blocks_per_split"] * 2 * BLOCK_SIZE
    strict_loads = stats["strict_expert_loads"]
    adaptive_loads = stats["adaptive_expert_loads"]
    saved_loads = strict_loads - adaptive_loads
    return {
        "source_report": str(path.relative_to(ROOT)),
        "tokens": tokens,
        "strict_expert_loads": strict_loads,
        "adaptive_expert_loads": adaptive_loads,
        "saved_expert_loads": saved_loads,
        "expert_load_reduction_fraction": stats[
            "expert_load_reduction_fraction"
        ],
        "strict_int4_routed_io_mib_per_token": mib(
            strict_loads * bytes_per_expert / tokens
        ),
        "adaptive_int4_routed_io_mib_per_token": mib(
            adaptive_loads * bytes_per_expert / tokens
        ),
        "saved_int4_routed_io_mib_per_token": mib(
            saved_loads * bytes_per_expert / tokens
        ),
        "projected_int4_routed_bandwidth_gib_per_second_at_10_tokens_per_second": {
            "strict": gib(strict_loads * bytes_per_expert / tokens * 10),
            "adaptive": gib(adaptive_loads * bytes_per_expert / tokens * 10),
            "saved": gib(saved_loads * bytes_per_expert / tokens * 10),
        },
        "quality": payload["final"],
    }


if __name__ == "__main__":
    config = json.loads(
        (ROOT / "models" / "deepseek-v2-lite" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    index = json.loads(
        (
            ROOT
            / "models"
            / "deepseek-v2-lite"
            / "model.safetensors.index.json"
        ).read_text(encoding="utf-8")
    )
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    routed_experts = config["n_routed_experts"]
    moe_layers = config["num_hidden_layers"] - config["first_k_dense_replace"]
    expert_parameters = 3 * hidden * intermediate
    expert_scale_rows = 2 * intermediate + hidden
    bf16_expert_bytes = expert_parameters * 2
    int4_expert_bytes = packed_quantized_bytes(
        expert_parameters, 4, expert_scale_rows
    )
    total_routed_parameters = moe_layers * routed_experts * expert_parameters
    routed_bf16_bytes = total_routed_parameters * 2
    routed_int4_bytes = moe_layers * routed_experts * int4_expert_bytes
    checkpoint_tensor_bytes = index["metadata"]["total_size"]
    nonrouted_bf16_bytes = checkpoint_tensor_bytes - routed_bf16_bytes
    cache_capacity = 32
    cache_experts_total = cache_capacity * moe_layers
    report = {
        "status": "complete",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "architecture": {
            "moe_layers": moe_layers,
            "routed_experts_per_layer": routed_experts,
            "active_routed_experts_per_token_per_layer": config[
                "num_experts_per_tok"
            ],
            "parameters_per_routed_expert": expert_parameters,
            "bf16_bytes_per_routed_expert": bf16_expert_bytes,
            "int4_bytes_per_routed_expert": int4_expert_bytes,
            "int4_scheme": "packed 4-bit values plus one BF16 scale per output row for gate, up, and down projections",
        },
        "storage": {
            "checkpoint_tensor_bytes": checkpoint_tensor_bytes,
            "checkpoint_tensor_gib": gib(checkpoint_tensor_bytes),
            "routed_experts_bf16_gib": gib(routed_bf16_bytes),
            "nonrouted_checkpoint_bf16_gib": gib(nonrouted_bf16_bytes),
            "routed_experts_int4_gib": gib(routed_int4_bytes),
            "checkpoint_with_nonrouted_bf16_and_routed_int4_gib": gib(
                nonrouted_bf16_bytes + routed_int4_bytes
            ),
        },
        "capacity_32_per_layer_cache": {
            "experts_resident_across_layers": cache_experts_total,
            "bf16_cache_gib": gib(cache_experts_total * bf16_expert_bytes),
            "int4_cache_gib": gib(cache_experts_total * int4_expert_bytes),
            "nonrouted_bf16_plus_int4_cache_gib": gib(
                nonrouted_bf16_bytes + cache_experts_total * int4_expert_bytes
            ),
        },
        "corpora": {
            name: corpus_accounting(path, int4_expert_bytes)
            for name, path in REPORTS.items()
        },
        "interpretation_limits": [
            "The evaluator measured exact BF16 model quality and expert-level LRU load counts; it did not execute packed int4 kernels.",
            "The int4 byte and bandwidth figures are deterministic accounting projections, not measured wall-clock throughput.",
            "Capacity 32 means 32 routed experts reserved independently for each of 26 MoE layers, not 32 experts globally.",
            "The bandwidth figures cover routed-expert cache fills only and exclude attention, shared experts, compute, prefetch overlap, metadata, and KV-cache traffic.",
            "Combining this routing policy with quantized weights requires a separate end-to-end quality validation.",
        ],
    }
    output = write_json(
        "route_cache_storage_io_accounting.json",
        envelope("route_cache_accounting", report),
    )
    print(output)
    print(json.dumps(report, indent=2))
