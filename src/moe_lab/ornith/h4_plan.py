"""Route bucketing and transport policy for the Ornith H4 ERVF executor.

The GPU kernels consume one bucket per route multiplicity.  This module keeps
planning independent of CUDA so route order, cache behavior, and reduction
indices can be tested without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


H4_POSITIONS = 4
ORNITH_TOP_K = 8


@dataclass(frozen=True)
class RouteBucket:
    """Uniform-multiplicity groups for one cache residency class."""

    multiplicity: int
    expert_ids: tuple[int, ...]
    cache_slots: tuple[int, ...]
    input_ids: tuple[int, ...]
    route_indices: tuple[int, ...]

    @property
    def groups(self) -> int:
        return len(self.expert_ids)

    @property
    def assignments(self) -> int:
        return self.groups * self.multiplicity


@dataclass(frozen=True)
class RoutePlan:
    """Stable H4 plan split into device-cache hits and host misses."""

    routes: tuple[tuple[int, ...], ...]
    hot_buckets: tuple[RouteBucket, ...]
    miss_buckets: tuple[RouteBucket, ...]
    miss_transport: str

    @property
    def hot_assignments(self) -> int:
        return sum(bucket.assignments for bucket in self.hot_buckets)

    @property
    def miss_assignments(self) -> int:
        return sum(bucket.assignments for bucket in self.miss_buckets)

    @property
    def unique_hot_experts(self) -> int:
        return sum(bucket.groups for bucket in self.hot_buckets)

    @property
    def unique_miss_experts(self) -> int:
        return sum(bucket.groups for bucket in self.miss_buckets)


def choose_miss_transport(unique_miss_experts: int) -> str:
    """Select the Phase62 transport arm.

    Direct mapped-host execution is measured best for one unique miss. Bulk
    staging wins by four misses. Counts two and three conservatively use the
    staged arm until separately measured.
    """

    count = int(unique_miss_experts)
    if count < 0:
        raise ValueError("unique_miss_experts must be non-negative")
    if count == 0:
        return "none"
    if count == 1:
        return "direct_uva"
    return "bulk_stage"


def _bucketize(groups: list[tuple[int, int, list[tuple[int, int]]]]) -> tuple[RouteBucket, ...]:
    buckets = []
    for multiplicity in range(1, H4_POSITIONS + 1):
        selected = [row for row in groups if len(row[2]) == multiplicity]
        if not selected:
            continue
        experts = tuple(row[0] for row in selected)
        slots = tuple(row[1] for row in selected)
        inputs = tuple(token for row in selected for token, _route in row[2])
        route_indices = tuple(route for row in selected for _token, route in row[2])
        buckets.append(RouteBucket(
            multiplicity=multiplicity,
            expert_ids=experts,
            cache_slots=slots,
            input_ids=inputs,
            route_indices=route_indices,
        ))
    return tuple(buckets)


def build_h4_route_plan(
    routes: Sequence[Sequence[int]],
    cache_slots: Mapping[int, int],
) -> RoutePlan:
    """Build stable expert-major buckets from four token-major top-8 routes.

    ``route_indices`` maps every expert-major output back to its original
    flattened token-major route slot, preserving the authoritative reduction
    order after parallel expert execution.
    """

    normalized = tuple(tuple(int(expert) for expert in row) for row in routes)
    if len(normalized) != H4_POSITIONS:
        raise ValueError(f"expected {H4_POSITIONS} route rows, got {len(normalized)}")
    if any(len(row) != ORNITH_TOP_K for row in normalized):
        raise ValueError(f"every route row must contain {ORNITH_TOP_K} experts")
    for token, row in enumerate(normalized):
        if any(expert < 0 for expert in row):
            raise ValueError(f"negative expert ID in token {token}")
        if len(set(row)) != len(row):
            raise ValueError(f"duplicate expert within token {token}")

    grouped: dict[int, list[tuple[int, int]]] = {}
    for token, row in enumerate(normalized):
        for route_slot, expert in enumerate(row):
            flattened = token * ORNITH_TOP_K + route_slot
            grouped.setdefault(expert, []).append((token, flattened))

    hot = []
    misses = []
    for expert, assignments in grouped.items():
        if len(assignments) > H4_POSITIONS:
            raise AssertionError("one expert cannot occur more than once per token")
        if expert in cache_slots:
            slot = int(cache_slots[expert])
            if slot < 0:
                raise ValueError(f"negative cache slot for expert {expert}")
            hot.append((expert, slot, assignments))
        else:
            misses.append((expert, -1, assignments))

    hot_buckets = _bucketize(hot)
    miss_buckets = _bucketize(misses)
    plan = RoutePlan(
        routes=normalized,
        hot_buckets=hot_buckets,
        miss_buckets=miss_buckets,
        miss_transport=choose_miss_transport(len(misses)),
    )
    if plan.hot_assignments + plan.miss_assignments != H4_POSITIONS * ORNITH_TOP_K:
        raise AssertionError("route planner lost assignments")
    return plan
