from __future__ import annotations

import pytest

from moe_lab.ornith.h4_plan import build_h4_route_plan, choose_miss_transport


def _routes():
    return (
        (1, 2, 3, 4, 5, 6, 7, 8),
        (1, 2, 9, 10, 11, 12, 13, 14),
        (1, 15, 16, 17, 18, 19, 20, 21),
        (1, 2, 22, 23, 24, 25, 26, 27),
    )


def test_route_plan_buckets_multiplicity_and_preserves_all_route_indices():
    routes = _routes()
    plan = build_h4_route_plan(routes, {expert: expert + 100 for expert in range(28)})
    assert plan.miss_transport == "none"
    assert plan.hot_assignments == 32
    assert plan.miss_assignments == 0
    by_m = {bucket.multiplicity: bucket for bucket in plan.hot_buckets}
    assert by_m[4].expert_ids == (1,)
    assert by_m[3].expert_ids == (2,)
    assert by_m[4].input_ids == (0, 1, 2, 3)
    assert sorted(index for bucket in plan.hot_buckets for index in bucket.route_indices) == list(range(32))


def test_route_plan_splits_hits_and_misses_and_selects_bulk_stage():
    routes = _routes()
    plan = build_h4_route_plan(routes, {1: 7, 2: 9})
    assert plan.hot_assignments == 7
    assert plan.miss_assignments == 25
    assert plan.unique_hot_experts == 2
    assert plan.unique_miss_experts == 25
    assert plan.miss_transport == "bulk_stage"


def test_one_unique_miss_selects_direct_uva():
    routes = _routes()
    resident = {expert: expert for row in routes for expert in row if expert != 27}
    plan = build_h4_route_plan(routes, resident)
    assert plan.unique_miss_experts == 1
    assert plan.miss_transport == "direct_uva"


def test_invalid_routes_rejected():
    routes = list(_routes())
    routes[0] = (1, 1, 3, 4, 5, 6, 7, 8)
    with pytest.raises(ValueError, match="duplicate"):
        build_h4_route_plan(routes, {})
    with pytest.raises(ValueError, match="non-negative"):
        choose_miss_transport(-1)
