from __future__ import annotations

import time

import torch
from safetensors.torch import load_file

from moe_lab.reporting import ROOT, envelope, write_json


CAPACITIES = (6, 8, 12, 16, 24, 32)
THRESHOLDS = (1e-3, 3e-3)
TOKENS_PER_SPLIT = 256
BLOCK_SIZE = 128


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


def simulate_strict(ids: torch.Tensor, capacity: int) -> dict[str, float]:
    loads = 0
    tokens_with_load = 0
    for block_start in range(0, ids.shape[0], BLOCK_SIZE):
        cache: list[int] = []
        for token in range(block_start, block_start + BLOCK_SIZE):
            route = ids[token, :6].tolist()
            misses = touch_route(cache, route, capacity)
            loads += misses
            tokens_with_load += int(misses > 0)
    return {
        "expert_loads": loads,
        "expert_load_fraction": loads / (ids.shape[0] * 6),
        "tokens_with_any_load_fraction": tokens_with_load / ids.shape[0],
    }


def simulate_equivalent(
    ids: torch.Tensor,
    subset_kl: torch.Tensor,
    subsets: torch.Tensor,
    capacity: int,
    threshold: float,
    policy: str,
) -> dict[str, float]:
    loads = 0
    tokens_with_load = 0
    reroutes = 0
    selected_kl = []
    selected_jaccard = []
    original = torch.arange(6)
    original_index = int((subsets == original).all(1).nonzero(as_tuple=False).item())
    for block_start in range(0, ids.shape[0], BLOCK_SIZE):
        cache: list[int] = []
        for token in range(block_start, block_start + BLOCK_SIZE):
            eligible = (subset_kl[token] <= threshold).nonzero(
                as_tuple=False
            ).squeeze(1)
            candidates = []
            for subset_index in eligible.tolist():
                positions = subsets[subset_index].tolist()
                route = ids[token, positions].tolist()
                misses = touch_route(list(cache), route, capacity)
                candidates.append((misses, float(subset_kl[token, subset_index]), subset_index, route))
            if policy == "zero_miss_or_original":
                zero_miss = [candidate for candidate in candidates if candidate[0] == 0]
                if zero_miss:
                    chosen = min(zero_miss, key=lambda candidate: candidate[1])
                else:
                    route = ids[token, :6].tolist()
                    misses = touch_route(list(cache), route, capacity)
                    chosen = (
                        misses,
                        float(subset_kl[token, original_index]),
                        original_index,
                        route,
                    )
            elif policy == "minimum_miss_equivalent":
                chosen = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
            else:
                raise ValueError(policy)
            misses, kl, subset_index, route = chosen
            observed_misses = touch_route(cache, route, capacity)
            if observed_misses != misses:
                raise RuntimeError("LRU miss accounting mismatch")
            loads += misses
            tokens_with_load += int(misses > 0)
            reroutes += int(subset_index != original_index)
            intersection = int((subsets[subset_index] < 6).sum().item())
            selected_jaccard.append(intersection / (12 - intersection))
            selected_kl.append(kl)
    values = torch.tensor(selected_kl)
    return {
        "expert_loads": loads,
        "expert_load_fraction": loads / (ids.shape[0] * 6),
        "tokens_with_any_load_fraction": tokens_with_load / ids.shape[0],
        "rerouted_token_fraction": reroutes / ids.shape[0],
        "selected_route_kl_mean": float(values.mean().item()),
        "selected_route_kl_p95": float(torch.quantile(values, 0.95).item()),
        "selected_route_jaccard_mean": sum(selected_jaccard) / len(selected_jaccard),
    }


if __name__ == "__main__":
    started = time.perf_counter()
    artifact = load_file(
        ROOT / "data" / "traces" / "layer26_route_equivalence.safetensors",
        device="cpu",
    )
    results = {}
    for split_index, split in enumerate(("validation", "test")):
        sl = slice(
            split_index * TOKENS_PER_SPLIT,
            (split_index + 1) * TOKENS_PER_SPLIT,
        )
        ids = artifact["top12_expert_ids"][sl].long()
        subset_kl = artifact["subset_kl"][sl].float()
        subsets = artifact["subsets"].long()
        rows = []
        for capacity in CAPACITIES:
            strict = simulate_strict(ids, capacity)
            for threshold in THRESHOLDS:
                for policy in ("zero_miss_or_original", "minimum_miss_equivalent"):
                    adaptive = simulate_equivalent(
                        ids, subset_kl, subsets, capacity, threshold, policy
                    )
                    rows.append(
                        {
                            "capacity_experts": capacity,
                            "kl_threshold": threshold,
                            "policy": policy,
                            "strict": strict,
                            "equivalent": adaptive,
                            "expert_load_reduction_fraction": 1.0
                            - adaptive["expert_loads"] / strict["expert_loads"],
                        }
                    )
        results[split] = rows

    report = {
        "status": "complete",
        "experiment": "layer26_route_equivalent_lru_cache_simulation",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "cache_scope": "one layer; reset at each 128-token block; expert-granularity LRU",
        "cost": "one load per missing full expert; no byte overlap or prefetch modeled",
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_route_equivalent_lru_cache.json",
        envelope("route_equivalent_cache", report),
    )
    print(path)
    for row in results["test"]:
        if row["policy"] == "minimum_miss_equivalent":
            print(
                f"capacity={row['capacity_experts']} eps={row['kl_threshold']} "
                f"load_reduction={row['expert_load_reduction_fraction']:.3f} "
                f"reroute={row['equivalent']['rerouted_token_fraction']:.3f} "
                f"KL={row['equivalent']['selected_route_kl_mean']:.6f}"
            )
