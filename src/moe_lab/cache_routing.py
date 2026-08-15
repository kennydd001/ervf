from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheRoutingPolicy:
    method: str
    top_j: int = 0
    parameter: float = 0.0

    @property
    def name(self) -> str:
        if self.method == "original":
            return "original"
        parameter = f"{self.parameter:g}".replace(".", "p")
        label = {
            "max_rank": "m",
            "cumsum": "p",
            "cache_prior": "lambda",
            "mass_budget": "delta",
        }[self.method]
        return f"{self.method}_j{self.top_j}_{label}{parameter}"


def parse_policy(specification: str) -> CacheRoutingPolicy:
    parts = specification.lower().split(":")
    if parts == ["original"]:
        return CacheRoutingPolicy("original")
    if len(parts) != 3 or not parts[1].startswith("j"):
        raise ValueError(
            "policy must be original or METHOD:jN:VALUE, for example "
            "cache_prior:j2:0.5"
        )
    method = parts[0]
    if method not in {"max_rank", "cumsum", "cache_prior", "mass_budget"}:
        raise ValueError(f"unsupported policy method: {method}")
    top_j = int(parts[1][1:])
    parameter = float(parts[2])
    if top_j < 0:
        raise ValueError("top_j must be non-negative")
    if method == "max_rank" and parameter < 1:
        raise ValueError("max-rank M must be at least one")
    if method in {"cumsum", "cache_prior", "mass_budget"} and not 0 <= parameter <= 1:
        raise ValueError(f"{method} parameter must be in [0, 1]")
    return CacheRoutingPolicy(method, top_j, parameter)


def touch_route(
    cache: list[int], route: list[int], capacity: int
) -> int:
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


def _promoted_ranking(
    ranked_ids: list[int], cache: set[int], top_j: int, max_rank: int
) -> list[int]:
    protected = ranked_ids[:top_j]
    promoted = [
        expert
        for expert in ranked_ids[:max_rank]
        if expert in cache and expert not in protected
    ]
    used = set(protected) | set(promoted)
    return protected + promoted + [
        expert for expert in ranked_ids if expert not in used
    ]


def select_route(
    ranked_ids: list[int],
    ranked_probabilities: list[float],
    raw_logits_by_expert: list[float],
    cache: set[int],
    policy: CacheRoutingPolicy,
    delta_average: float,
    top_k: int,
) -> list[int]:
    if len(ranked_ids) != len(ranked_probabilities):
        raise ValueError("ranked ids and probabilities must have equal length")
    if top_k > len(ranked_ids):
        raise ValueError("top_k exceeds expert count")
    if policy.top_j > top_k:
        raise ValueError("top_j cannot exceed top_k")
    if policy.method == "original":
        return ranked_ids[:top_k]
    if policy.method == "max_rank":
        max_rank = min(int(policy.parameter), len(ranked_ids))
        return _promoted_ranking(
            ranked_ids, cache, policy.top_j, max_rank
        )[:top_k]
    if policy.method == "cumsum":
        cumulative = 0.0
        max_rank = 0
        while max_rank < len(ranked_ids) and cumulative < policy.parameter:
            cumulative += ranked_probabilities[max_rank]
            max_rank += 1
        return _promoted_ranking(
            ranked_ids, cache, policy.top_j, max_rank
        )[:top_k]
    if policy.method == "cache_prior":
        protected = set(ranked_ids[: policy.top_j])
        boosted = cache | protected
        original_rank = {
            expert: rank for rank, expert in enumerate(ranked_ids)
        }
        reranked = sorted(
            ranked_ids,
            key=lambda expert: (
                -(
                    raw_logits_by_expert[expert]
                    + policy.parameter
                    * delta_average
                    * (expert in boosted)
                ),
                original_rank[expert],
            ),
        )
        return reranked[:top_k]
    if policy.method == "mass_budget":
        probability_by_expert = dict(
            zip(ranked_ids, ranked_probabilities, strict=True)
        )
        original = ranked_ids[:top_k]
        original_mass = sum(probability_by_expert[expert] for expert in original)
        best_route = original
        best_key = (sum(expert not in cache for expert in original), 0.0, 0.0)
        protected = set(ranked_ids[: policy.top_j])
        boosted = cache | protected
        original_rank = {expert: rank for rank, expert in enumerate(ranked_ids)}
        for lambda_value in (
            0.025,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.75,
            1.00,
        ):
            reranked = sorted(
                ranked_ids,
                key=lambda expert: (
                    -(
                        raw_logits_by_expert[expert]
                        + lambda_value * delta_average * (expert in boosted)
                    ),
                    original_rank[expert],
                ),
            )
            route = reranked[:top_k]
            selected_mass = sum(probability_by_expert[expert] for expert in route)
            mass_loss = max(0.0, original_mass - selected_mass)
            if mass_loss > policy.parameter + 1e-12:
                continue
            key = (
                sum(expert not in cache for expert in route),
                mass_loss,
                lambda_value,
            )
            if key < best_key:
                best_route = route
                best_key = key
        return best_route
    raise ValueError(f"unsupported policy method: {policy.method}")


def route_batch(
    ranked_ids: torch.Tensor,
    ranked_probabilities: torch.Tensor,
    raw_logits: torch.Tensor,
    policy: CacheRoutingPolicy,
    capacity: int,
    delta_average: float,
    top_k: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    if ranked_ids.ndim != 3:
        raise ValueError("ranked_ids must have shape [batch, sequence, experts]")
    ids = ranked_ids.cpu()
    probabilities = ranked_probabilities.cpu()
    logits = raw_logits.cpu()
    batch, sequence, _ = ids.shape
    chosen = torch.empty(batch, sequence, top_k, dtype=torch.long)
    total_loads = 0
    substitutions = 0
    overlap_sum = 0.0
    probability_mass_loss_sum = 0.0
    per_block = []
    for block in range(batch):
        cache: list[int] = []
        block_loads = 0
        block_substitutions = 0
        block_overlap = 0.0
        block_probability_mass_loss = 0.0
        for token in range(sequence):
            token_ids = ids[block, token].tolist()
            token_probabilities = probabilities[block, token].tolist()
            route = select_route(
                token_ids,
                token_probabilities,
                logits[block, token].tolist(),
                set(cache),
                policy,
                delta_average,
                top_k,
            )
            chosen[block, token] = torch.tensor(route)
            original = token_ids[:top_k]
            overlap = len(set(route) & set(original)) / top_k
            changed = route != original
            substitutions += int(changed)
            block_substitutions += int(changed)
            overlap_sum += overlap
            block_overlap += overlap
            probability_by_expert = dict(
                zip(token_ids, token_probabilities, strict=True)
            )
            probability_mass_loss = max(
                0.0,
                sum(probability_by_expert[expert] for expert in original)
                - sum(probability_by_expert[expert] for expert in route),
            )
            probability_mass_loss_sum += probability_mass_loss
            block_probability_mass_loss += probability_mass_loss
            original_rank = {
                expert: rank for rank, expert in enumerate(token_ids)
            }
            touch_order = sorted(route, key=original_rank.__getitem__)
            misses = touch_route(cache, touch_order, capacity)
            total_loads += misses
            block_loads += misses
        per_block.append(
            {
                "expert_loads": block_loads,
                "cache_miss_fraction": block_loads / (sequence * top_k),
                "substituted_token_fraction": block_substitutions / sequence,
                "original_route_overlap": block_overlap / sequence,
                "mean_original_probability_mass_loss": (
                    block_probability_mass_loss / sequence
                ),
            }
        )
    tokens = batch * sequence
    return chosen.view(tokens, top_k), {
        "expert_loads": total_loads,
        "cache_miss_fraction": total_loads / (tokens * top_k),
        "substituted_token_fraction": substitutions / tokens,
        "original_route_overlap": overlap_sum / tokens,
        "mean_original_probability_mass_loss": probability_mass_loss_sum / tokens,
        "per_block": per_block,
    }
