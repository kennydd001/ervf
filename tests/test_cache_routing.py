import torch

from moe_lab.cache_routing import (
    CacheRoutingPolicy,
    parse_policy,
    select_route,
    touch_route,
)


def test_parse_policy_has_stable_name() -> None:
    policy = parse_policy("cache_prior:j2:0.5")
    assert policy == CacheRoutingPolicy("cache_prior", 2, 0.5)
    assert policy.name == "cache_prior_j2_lambda0p5"
    assert parse_policy("mass_budget:j2:0.01").name == "mass_budget_j2_delta0p01"


def test_max_rank_j5_m7_replaces_only_bottom_expert() -> None:
    ranked = list(range(8))
    probabilities = [0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04]
    logits = [float(8 - expert) for expert in ranked]
    policy = CacheRoutingPolicy("max_rank", 5, 7)
    route = select_route(
        ranked, probabilities, logits, {6}, policy, 8.0, 6
    )
    assert route == [0, 1, 2, 3, 4, 6]
    unchanged = select_route(
        ranked, probabilities, logits, {5}, policy, 8.0, 6
    )
    assert unchanged == [0, 1, 2, 3, 4, 5]


def test_cumsum_promotes_only_inside_probability_nucleus() -> None:
    ranked = list(range(6))
    probabilities = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04]
    logits = [float(6 - expert) for expert in ranked]
    policy = CacheRoutingPolicy("cumsum", 1, 0.75)
    route = select_route(
        ranked, probabilities, logits, {2, 4}, policy, 6.0, 2
    )
    assert route == [0, 2]


def test_cache_prior_preserves_protected_top_experts() -> None:
    ranked = [0, 1, 2, 3]
    probabilities = [0.4, 0.3, 0.2, 0.1]
    logits = [4.0, 3.0, 2.0, 1.0]
    policy = CacheRoutingPolicy("cache_prior", 2, 1.0)
    route = select_route(
        ranked, probabilities, logits, {2, 3}, policy, 4.0, 2
    )
    assert set(route) == {0, 1}


def test_touch_route_is_lru() -> None:
    cache = [0, 1, 2]
    assert touch_route(cache, [1, 3], capacity=3) == 1
    assert cache == [2, 1, 3]


def test_mass_budget_rejects_excessive_router_mass_loss() -> None:
    ids = [0, 1, 2, 3, 4, 5, 6, 7]
    probabilities = [0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.04, 0.01]
    logits = [2.0, 1.8, 1.5, 1.3, 1.1, 0.9, 0.85, 0.8]
    strict = select_route(
        ids,
        probabilities,
        logits,
        {6, 7},
        CacheRoutingPolicy("mass_budget", top_j=2, parameter=0.0),
        delta_average=2.0,
        top_k=6,
    )
    flexible = select_route(
        ids,
        probabilities,
        logits,
        {6, 7},
        CacheRoutingPolicy("mass_budget", top_j=2, parameter=0.2),
        delta_average=2.0,
        top_k=6,
    )
    assert strict == ids[:6]
    assert len(set(flexible) & {6, 7}) > 0
    selected_mass = sum(probabilities[ids.index(expert)] for expert in flexible)
    assert sum(probabilities[:6]) - selected_mass <= 0.2
