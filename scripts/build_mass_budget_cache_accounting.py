from __future__ import annotations

import json
from pathlib import Path

from moe_lab.quantization import packed_quantized_bytes
from moe_lab.reporting import ROOT, envelope, write_json


COMPARISONS = {
    "wikitext_offset4096_128": {
        "report": ROOT
        / "reports"
        / "baseline"
        / "preregistered_wikitext_offset4096_mass_budget_confirmation.json",
        "fixed_policy": "cache_prior_j2_lambda0p0275",
        "mass_budget_policy": "mass_budget_j2_delta0p004",
    },
    "wikitext_128": {
        "report": ROOT
        / "reports"
        / "baseline"
        / "matched8_wikitext_mass_budget_vs_cache_prior.json",
        "fixed_policy": "cache_prior_j2_lambda0p085",
        "mass_budget_policy": "mass_budget_j2_delta0p016",
    },
    "instructions_code_128": {
        "report": ROOT
        / "reports"
        / "baseline"
        / "matched8_diverse_mass_budget_vs_cache_prior.json",
        "fixed_policy": "cache_prior_j2_lambda0p095",
        "mass_budget_policy": "mass_budget_j2_delta0p018",
    },
    "wikitext_1024": {
        "report": ROOT
        / "reports"
        / "baseline"
        / "matched_context1024_wikitext_mass_budget_vs_cache_prior.json",
        "fixed_policy": "cache_prior_j2_lambda0p095",
        "mass_budget_policy": "mass_budget_j2_delta0p018",
    },
}


def mib(value: int | float) -> float:
    return value / (1024**2)


def split_loads(
    stats: dict[str, object], blocks_per_split: int, split: str
) -> tuple[int, int]:
    blocks = stats["per_block"]
    if not isinstance(blocks, list):
        raise TypeError("per_block must be a list")
    if split == "validation":
        selected = blocks[:blocks_per_split]
    elif split == "test":
        selected = blocks[-blocks_per_split:]
    else:
        raise ValueError(f"unknown split: {split}")
    strict = sum(int(block["strict_expert_loads"]) for block in selected)
    adaptive = sum(int(block["adaptive_expert_loads"]) for block in selected)
    return strict, adaptive


def policy_accounting(
    payload: dict[str, object], policy: str, bytes_per_expert: int
) -> dict[str, object]:
    block_size = int(payload["block_size"])
    blocks_per_split = int(payload["blocks_per_split"])
    tokens_per_split = block_size * blocks_per_split
    stats = payload["total_cache_statistics"][policy]
    result: dict[str, object] = {}
    for split in ("validation", "test"):
        strict, adaptive = split_loads(stats, blocks_per_split, split)
        saved = strict - adaptive
        quality = payload["final"][split][policy]
        result[split] = {
            "tokens": tokens_per_split,
            "strict_expert_loads": strict,
            "adaptive_expert_loads": adaptive,
            "saved_expert_loads": saved,
            "expert_load_reduction_fraction": saved / strict,
            "strict_projected_int4_routed_io_mib_per_token": mib(
                strict * bytes_per_expert / tokens_per_split
            ),
            "adaptive_projected_int4_routed_io_mib_per_token": mib(
                adaptive * bytes_per_expert / tokens_per_split
            ),
            "saved_projected_int4_routed_io_mib_per_token": mib(
                saved * bytes_per_expert / tokens_per_split
            ),
            "teacher_to_candidate_kl": quality["teacher_to_candidate_kl"],
            "relative_cross_entropy_delta": quality[
                "relative_cross_entropy_delta"
            ],
            "top1_agreement": quality["top1_agreement"],
        }
    return result


def comparison_accounting(
    spec: dict[str, object], bytes_per_expert: int
) -> dict[str, object]:
    path = spec["report"]
    if not isinstance(path, Path):
        raise TypeError("report must be a Path")
    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    fixed_policy = str(spec["fixed_policy"])
    mass_policy = str(spec["mass_budget_policy"])
    fixed = policy_accounting(payload, fixed_policy, bytes_per_expert)
    mass = policy_accounting(payload, mass_policy, bytes_per_expert)
    test_fixed = fixed["test"]
    test_mass = mass["test"]
    fixed_kl = float(test_fixed["teacher_to_candidate_kl"])
    mass_kl = float(test_mass["teacher_to_candidate_kl"])
    return {
        "source_report": str(path.relative_to(ROOT)),
        "block_size": payload["block_size"],
        "blocks_per_split": payload["blocks_per_split"],
        "fixed_cache_prior": {
            "policy": fixed_policy,
            **fixed,
        },
        "mass_budget": {
            "policy": mass_policy,
            **mass,
        },
        "test_mass_budget_minus_fixed": {
            "expert_load_reduction_percentage_points": 100
            * (
                float(test_mass["expert_load_reduction_fraction"])
                - float(test_fixed["expert_load_reduction_fraction"])
            ),
            "teacher_to_candidate_kl": mass_kl - fixed_kl,
            "relative_kl_change": (mass_kl / fixed_kl) - 1,
            "saved_projected_int4_routed_io_mib_per_token": float(
                test_mass["saved_projected_int4_routed_io_mib_per_token"]
            )
            - float(
                test_fixed["saved_projected_int4_routed_io_mib_per_token"]
            ),
        },
    }


if __name__ == "__main__":
    config = json.loads(
        (ROOT / "models" / "deepseek-v2-lite" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    expert_parameters = 3 * hidden * intermediate
    expert_scale_rows = 2 * intermediate + hidden
    int4_expert_bytes = packed_quantized_bytes(
        expert_parameters, 4, expert_scale_rows
    )
    report = {
        "status": "complete",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "int4_accounting": {
            "parameters_per_routed_expert": expert_parameters,
            "bytes_per_routed_expert": int4_expert_bytes,
            "mib_per_routed_expert": mib(int4_expert_bytes),
            "scheme": "packed 4-bit values plus one BF16 scale per output row for gate, up, and down projections",
        },
        "comparisons": {
            name: comparison_accounting(spec, int4_expert_bytes)
            for name, spec in COMPARISONS.items()
        },
        "interpretation_limits": [
            "The source evaluators executed BF16 weights and measured expert-level LRU loads; they did not execute packed-int4 kernels.",
            "All byte values are deterministic routed-weight I/O projections, not wall-clock, throughput, or energy measurements.",
            "The projection excludes attention, shared experts, compute, prefetch overlap, metadata, and KV-cache traffic.",
            "The matched policies are close Pareto comparisons, not identical-load constraints.",
        ],
    }
    output = write_json(
        "mass_budget_cache_accounting.json",
        envelope("mass_budget_cache_accounting", report),
    )
    print(output)
    print(json.dumps(report, indent=2))
