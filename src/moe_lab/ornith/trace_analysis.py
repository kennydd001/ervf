"""Real llama.cpp trace parsing and expert-cache replay for Ornith."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from math import inf
from typing import Any, Mapping, Sequence


TOP_K = 8
N_EXPERTS = 256


@dataclass(frozen=True)
class OrnithTrace:
    tokens: tuple[int, ...]
    routes: Mapping[int, tuple[tuple[int, ...], ...]]
    weights: Mapping[int, tuple[tuple[float, ...], ...]]
    result_norm: tuple[tuple[float, ...], ...]


def _layer_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def parse_llama_trace(payload: Mapping[str, Any]) -> OrnithTrace:
    """Validate and reshape the compact trace emitted by llama_ornith_trace."""

    tokens = tuple(int(value) for value in payload.get("tokens", ()))
    if not tokens:
        raise ValueError("trace has no tokens")

    routes: dict[int, tuple[tuple[int, ...], ...]] = {}
    weights: dict[int, tuple[tuple[float, ...], ...]] = {}
    result_norm: tuple[tuple[float, ...], ...] = ()
    for tensor in payload.get("tensors", ()):
        name = str(tensor["name"])
        shape = tuple(int(value) for value in tensor["shape"])
        values = tensor["values"]
        layer = _layer_index(name, "ffn_moe_topk-")
        if layer is not None:
            if shape != (TOP_K, len(tokens), 1, 1):
                raise ValueError(f"{name}: expected {(TOP_K, len(tokens), 1, 1)}, got {shape}")
            rows = tuple(
                tuple(int(value) for value in values[token * TOP_K:(token + 1) * TOP_K])
                for token in range(len(tokens))
            )
            if any(len(set(row)) != TOP_K for row in rows):
                raise ValueError(f"{name}: duplicate expert within a token")
            if any(expert < 0 or expert >= N_EXPERTS for row in rows for expert in row):
                raise ValueError(f"{name}: expert ID out of range")
            routes[layer] = rows
            continue

        layer = _layer_index(name, "ffn_moe_weights_norm-")
        if layer is not None:
            if shape != (TOP_K, len(tokens), 1, 1):
                raise ValueError(f"{name}: expected {(TOP_K, len(tokens), 1, 1)}, got {shape}")
            weights[layer] = tuple(
                tuple(float(value) for value in values[token * TOP_K:(token + 1) * TOP_K])
                for token in range(len(tokens))
            )
            continue

        if name == "result_norm":
            if shape[0] != 2048 or shape[2:] != (1, 1):
                raise ValueError(f"result_norm: unsupported shape {shape}")
            result_norm = tuple(
                tuple(float(value) for value in values[token * 2048:(token + 1) * 2048])
                for token in range(shape[1])
            )

    if sorted(routes) != list(range(40)):
        raise ValueError(f"expected route layers 0..39, got {sorted(routes)}")
    if weights and sorted(weights) != list(range(40)):
        raise ValueError(f"expected weight layers 0..39, got {sorted(weights)}")
    return OrnithTrace(tokens=tokens, routes=routes, weights=weights, result_norm=result_norm)


def replay_expert_cache(
    routes: Sequence[Sequence[int]],
    slots: int = 52,
    policy: str = "lru",
) -> dict[str, Any]:
    """Replay one layer with atomic per-token lookups and post-token eviction."""

    normalized = tuple(tuple(int(expert) for expert in row) for row in routes)
    if slots < TOP_K:
        raise ValueError(f"slots must be at least {TOP_K}")
    if policy not in {"lru", "belady"}:
        raise ValueError(f"unsupported policy {policy}")
    if any(len(row) != TOP_K or len(set(row)) != TOP_K for row in normalized):
        raise ValueError("every route row must contain eight unique experts")

    future: dict[int, list[int]] = defaultdict(list)
    for token, row in enumerate(normalized):
        for expert in row:
            future[expert].append(token)

    resident: set[int] = set()
    last_used: dict[int, int] = {}
    hits = 0
    misses = 0
    evictions = 0
    per_token = []
    all_miss_experts: set[int] = set()
    for token, row in enumerate(normalized):
        selected = set(row)
        token_hits = tuple(expert for expert in row if expert in resident)
        token_misses = tuple(expert for expert in row if expert not in resident)
        hits += len(token_hits)
        misses += len(token_misses)
        all_miss_experts.update(token_misses)

        resident.update(selected)
        for expert in selected:
            last_used[expert] = token
        token_evicted = []
        while len(resident) > slots:
            if policy == "lru":
                victim = min(resident, key=lambda expert: (last_used[expert], expert))
            else:
                def next_use(expert: int) -> float:
                    positions = future.get(expert, ())
                    index = bisect_right(positions, token)
                    return positions[index] if index < len(positions) else inf
                victim = max(resident, key=lambda expert: (next_use(expert), expert))
            resident.remove(victim)
            last_used.pop(victim, None)
            token_evicted.append(victim)
            evictions += 1
        per_token.append({
            "token": token,
            "hits": list(token_hits),
            "misses": list(token_misses),
            "evicted": token_evicted,
            "resident": len(resident),
        })

    assignments = len(normalized) * TOP_K
    return {
        "policy": policy,
        "slots": slots,
        "tokens": len(normalized),
        "assignments": assignments,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / assignments if assignments else 0.0,
        "miss_rate": misses / assignments if assignments else 0.0,
        "unique_miss_experts": len(all_miss_experts),
        "evictions": evictions,
        "final_resident": sorted(resident),
        "per_token": per_token,
    }


def replay_trace(trace: OrnithTrace, slots: int = 52, policy: str = "lru") -> dict[str, Any]:
    layers = {
        str(layer): replay_expert_cache(trace.routes[layer], slots=slots, policy=policy)
        for layer in sorted(trace.routes)
    }
    assignments = sum(row["assignments"] for row in layers.values())
    hits = sum(row["hits"] for row in layers.values())
    misses = sum(row["misses"] for row in layers.values())
    evictions = sum(row["evictions"] for row in layers.values())
    return {
        "policy": policy,
        "slots_per_layer": slots,
        "tokens": len(trace.tokens),
        "layers": layers,
        "summary": {
            "assignments": assignments,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / assignments,
            "miss_rate": misses / assignments,
            "evictions": evictions,
        },
    }


def summarize_h4_miss_groups(
    replay: Mapping[str, Any],
    block_size: int = 4,
    warmup_tokens: int = 16,
) -> dict[str, Any]:
    """Summarize unique transported experts for aligned verifier blocks."""

    if block_size <= 0 or warmup_tokens < 0:
        raise ValueError("block_size must be positive and warmup_tokens non-negative")
    n_tokens = int(replay["tokens"])
    if n_tokens % block_size:
        raise ValueError("trace token count must be divisible by block_size")
    if warmup_tokens % block_size:
        raise ValueError("warmup_tokens must be divisible by block_size")

    blocks = []
    histogram = {str(count): 0 for count in range(TOP_K * block_size + 1)}
    for begin in range(0, n_tokens, block_size):
        layer_counts = []
        for layer in sorted(replay["layers"], key=int):
            per_token = replay["layers"][layer]["per_token"]
            misses = {
                int(expert)
                for row in per_token[begin:begin + block_size]
                for expert in row["misses"]
            }
            count = len(misses)
            layer_counts.append(count)
            histogram[str(count)] += 1
        blocks.append({
            "begin_token": begin,
            "unique_miss_groups_by_layer": layer_counts,
            "sum_unique_miss_groups": sum(layer_counts),
            "max_unique_miss_groups": max(layer_counts),
            "zero_miss_layers": sum(count == 0 for count in layer_counts),
        })

    warm_blocks = [row for row in blocks if row["begin_token"] >= warmup_tokens]
    if not warm_blocks:
        raise ValueError("warmup excludes all blocks")
    warm_counts = [
        count
        for block in warm_blocks
        for count in block["unique_miss_groups_by_layer"]
    ]
    sorted_counts = sorted(warm_counts)

    def percentile(fraction: float) -> float:
        index = int(round((len(sorted_counts) - 1) * fraction))
        return float(sorted_counts[index])

    return {
        "block_size": block_size,
        "warmup_tokens": warmup_tokens,
        "blocks": blocks,
        "histogram_all_layer_blocks": histogram,
        "warm": {
            "blocks": len(warm_blocks),
            "layer_blocks": len(warm_counts),
            "mean_unique_miss_groups_per_layer_h4": sum(warm_counts) / len(warm_counts),
            "p50_unique_miss_groups_per_layer_h4": percentile(0.50),
            "p95_unique_miss_groups_per_layer_h4": percentile(0.95),
            "max_unique_miss_groups_per_layer_h4": max(warm_counts),
            "mean_zero_miss_layers_per_h4": (
                sum(row["zero_miss_layers"] for row in warm_blocks) / len(warm_blocks)
            ),
            "mean_unique_miss_groups_all_layers_h4": (
                sum(row["sum_unique_miss_groups"] for row in warm_blocks) / len(warm_blocks)
            ),
        },
    }
