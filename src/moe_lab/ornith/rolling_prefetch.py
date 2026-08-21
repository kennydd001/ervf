"""Transactional rolling expert prefetch for the Ornith H4 verifier.

The target router is authoritative.  A drafter or route predictor may stage
weights early, but staged weights live outside the persistent 52-slot expert
cache until the target has exposed its real routes.  Prediction mistakes can
therefore waste transport without changing model arithmetic or cache metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .h4_plan import RoutePlan, build_h4_route_plan


ORNITH_LAYERS = 40
ORNITH_EXPERTS = 256
DEFAULT_CACHE_SLOTS = 52
DEFAULT_RING_DEPTH = 2
SEGMENT_BYTES = (
    ("gate_codes", 524_288),
    ("gate_scales", 65_536),
    ("up_codes", 524_288),
    ("up_scales", 65_536),
    ("down_codes", 524_288),
    ("down_scales", 65_536),
)
EXPERT_BYTES = sum(size for _name, size in SEGMENT_BYTES)


Routes = tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class LayerCacheSnapshot:
    """Immutable metadata for one persistent expert cache."""

    slot_to_expert: tuple[int | None, ...]
    last_used: tuple[tuple[int, int], ...]
    clock: int

    @property
    def expert_to_slot(self) -> dict[int, int]:
        return {
            expert: slot
            for slot, expert in enumerate(self.slot_to_expert)
            if expert is not None
        }


@dataclass(frozen=True)
class LayerPrefetchTask:
    """One layer's segmented copy into a temporary ring entry."""

    request_generation: int
    schedule_epoch: int
    block_id: int
    task_index: int
    layer: int
    ring_slot: int
    experts: tuple[int, ...]

    @property
    def groups(self) -> int:
        return len(self.experts)

    @property
    def bytes(self) -> int:
        return self.groups * EXPERT_BYTES

    @property
    def segments(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, self.groups * bytes_per_expert)
            for name, bytes_per_expert in SEGMENT_BYTES
        )


@dataclass(frozen=True)
class PreparedBlock:
    """A predicted H4 block and its tentative cache transition."""

    request_id: str
    request_generation: int
    schedule_epoch: int
    block_id: int
    predicted_routes: Routes
    cache_before: tuple[LayerCacheSnapshot, ...]
    cache_after: tuple[LayerCacheSnapshot, ...]
    tasks: tuple[LayerPrefetchTask, ...]


@dataclass(frozen=True)
class ExecutionLayerPlan:
    """Authoritative target routes split over resident, staged and cold data."""

    layer: int
    resident_plan: RoutePlan
    combined_plan: RoutePlan
    staged_experts: tuple[int, ...]
    staged_hits: tuple[int, ...]
    uncovered_experts: tuple[int, ...]
    false_prefetch_experts: tuple[int, ...]


@dataclass(frozen=True)
class BlockAdjudication:
    """Result of committing an authoritative prefix of a prepared block."""

    block_id: int
    committed_positions: int
    compared_assignments: int
    exact_assignments: int
    staged_hits: int
    uncovered_misses: int
    false_prefetches: int
    invalidated_block_ids: tuple[int, ...]
    requires_copy_barrier: bool

    @property
    def route_accuracy(self) -> float:
        if not self.compared_assignments:
            return 1.0
        return self.exact_assignments / self.compared_assignments


class _LayerCache:
    def __init__(self, slots: int) -> None:
        self.slot_to_expert: list[int | None] = [None] * slots
        self.last_used: dict[int, int] = {}
        self.clock = 0

    @classmethod
    def from_snapshot(cls, snapshot: LayerCacheSnapshot) -> _LayerCache:
        result = cls(len(snapshot.slot_to_expert))
        result.slot_to_expert[:] = snapshot.slot_to_expert
        result.last_used = dict(snapshot.last_used)
        result.clock = snapshot.clock
        return result

    @property
    def expert_to_slot(self) -> dict[int, int]:
        return {
            expert: slot
            for slot, expert in enumerate(self.slot_to_expert)
            if expert is not None
        }

    def snapshot(self) -> LayerCacheSnapshot:
        return LayerCacheSnapshot(
            slot_to_expert=tuple(self.slot_to_expert),
            last_used=tuple(sorted(self.last_used.items())),
            clock=self.clock,
        )

    def process_rows(self, rows: Sequence[Sequence[int]]) -> tuple[int, ...]:
        """Apply atomic token lookups and deterministic post-lookup LRU fills."""

        misses: list[int] = []
        for row in rows:
            selected = set(row)
            mapping = self.expert_to_slot
            token_misses = [expert for expert in row if expert not in mapping]
            for expert in token_misses:
                if expert in self.expert_to_slot:
                    continue
                try:
                    slot = self.slot_to_expert.index(None)
                except ValueError:
                    candidates = [
                        resident
                        for resident in self.last_used
                        if resident not in selected
                    ]
                    if not candidates:
                        candidates = list(self.last_used)
                    victim = min(
                        candidates,
                        key=lambda value: (self.last_used[value], value),
                    )
                    slot = self.expert_to_slot[victim]
                    self.slot_to_expert[slot] = None
                    del self.last_used[victim]
                self.slot_to_expert[slot] = expert
                misses.append(expert)
            self.clock += 1
            for expert in selected:
                self.last_used[expert] = self.clock
        return tuple(dict.fromkeys(misses))


def _normalize_routes(
    routes: Sequence[Sequence[Sequence[int]]],
    *,
    layers: int,
    experts: int,
) -> Routes:
    normalized = tuple(
        tuple(tuple(int(expert) for expert in row) for row in layer)
        for layer in routes
    )
    if len(normalized) != layers:
        raise ValueError(f"expected {layers} layers, got {len(normalized)}")
    for layer, rows in enumerate(normalized):
        if len(rows) != 4:
            raise ValueError(f"layer {layer}: expected four H4 rows")
        for token, row in enumerate(rows):
            if len(row) != 8 or len(set(row)) != 8:
                raise ValueError(
                    f"layer {layer} token {token}: expected eight unique experts"
                )
            if any(expert < 0 or expert >= experts for expert in row):
                raise ValueError(f"layer {layer} token {token}: expert out of range")
    return normalized


def build_execution_layer_plan(
    authoritative_routes: Sequence[Sequence[int]],
    cache_before: LayerCacheSnapshot,
    staged_experts: Sequence[int],
    *,
    layer: int = 0,
) -> ExecutionLayerPlan:
    """Adjudicate one dynamic staging task against real target H4 routes."""

    rows = tuple(tuple(int(expert) for expert in row) for row in authoritative_routes)
    if len(rows) != 4 or any(len(row) != 8 or len(set(row)) != 8 for row in rows):
        raise ValueError("authoritative_routes must contain four top-8 rows")
    staged = tuple(dict.fromkeys(int(expert) for expert in staged_experts))
    if len(staged) != len(tuple(staged_experts)):
        raise ValueError("staged_experts must be unique")
    if any(expert < 0 for row in rows for expert in row) or any(expert < 0 for expert in staged):
        raise ValueError("expert IDs must be non-negative")
    resident = cache_before.expert_to_slot
    replay = _LayerCache.from_snapshot(cache_before)
    true_misses = replay.process_rows(rows)
    true_miss_set = set(true_misses)
    staged_set = set(staged)
    staged_hits = tuple(expert for expert in true_misses if expert in staged_set)
    uncovered = tuple(expert for expert in true_misses if expert not in staged_set)
    false_prefetch = tuple(expert for expert in staged if expert not in true_miss_set)
    # A start-resident expert can be evicted and miss again inside H4. Treat
    # every replayed miss as cold for the block plan; a staged copy may then
    # serve all of that expert's assignments without changing arithmetic.
    effective_resident = {
        expert: slot for expert, slot in resident.items() if expert not in true_miss_set
    }
    staged_slots = {
        expert: len(cache_before.slot_to_expert) + index
        for index, expert in enumerate(staged_hits)
    }
    combined = dict(effective_resident)
    combined.update(staged_slots)
    return ExecutionLayerPlan(
        layer=layer,
        resident_plan=build_h4_route_plan(rows, effective_resident),
        combined_plan=build_h4_route_plan(rows, combined),
        staged_experts=staged,
        staged_hits=staged_hits,
        uncovered_experts=uncovered,
        false_prefetch_experts=false_prefetch,
    )


class RollingPrefetchController:
    """Manage speculative H4 prefetch without speculatively mutating cache state."""

    def __init__(
        self,
        *,
        layers: int = ORNITH_LAYERS,
        experts: int = ORNITH_EXPERTS,
        cache_slots: int = DEFAULT_CACHE_SLOTS,
        ring_depth: int = DEFAULT_RING_DEPTH,
    ) -> None:
        if layers <= 0 or experts < 8:
            raise ValueError("layers must be positive and experts at least eight")
        if cache_slots < 8 or cache_slots > experts:
            raise ValueError("cache_slots must be in [8, experts]")
        if ring_depth <= 0:
            raise ValueError("ring_depth must be positive")
        self.layers = int(layers)
        self.experts = int(experts)
        self.cache_slots = int(cache_slots)
        self.ring_depth = int(ring_depth)
        self._generation = 0
        self._schedule_epoch = 0
        self._request_id: str | None = None
        self._next_block_id = 0
        self._next_task_index = 0
        self._committed = tuple(_LayerCache(cache_slots) for _ in range(layers))
        self._pending: list[PreparedBlock] = []

    @property
    def request_generation(self) -> int:
        return self._generation

    @property
    def schedule_epoch(self) -> int:
        return self._schedule_epoch

    @property
    def pending_block_ids(self) -> tuple[int, ...]:
        return tuple(block.block_id for block in self._pending)

    def reset_request(self, request_id: str) -> None:
        request = str(request_id)
        if not request:
            raise ValueError("request_id must be non-empty")
        self._generation += 1
        self._schedule_epoch += 1
        self._request_id = request
        self._next_block_id = 0
        self._next_task_index = 0
        self._committed = tuple(
            _LayerCache(self.cache_slots) for _ in range(self.layers)
        )
        self._pending.clear()

    def cache_snapshot(self) -> tuple[LayerCacheSnapshot, ...]:
        return tuple(cache.snapshot() for cache in self._committed)

    def cache_slots_for_layer(self, layer: int) -> dict[int, int]:
        return self._committed[layer].expert_to_slot

    def _require_request(self) -> str:
        if self._request_id is None:
            raise RuntimeError("reset_request must be called before preparing a block")
        return self._request_id

    def _block(self, block_id: int) -> PreparedBlock:
        for block in self._pending:
            if block.block_id == block_id:
                if block.request_generation != self._generation:
                    break
                return block
        raise KeyError(f"block {block_id} is not pending in the current request")

    def prepare_block(
        self,
        predicted_routes: Sequence[Sequence[Sequence[int]]],
    ) -> PreparedBlock:
        request_id = self._require_request()
        routes = _normalize_routes(
            predicted_routes, layers=self.layers, experts=self.experts
        )
        if self._pending:
            before = self._pending[-1].cache_after
        else:
            before = self.cache_snapshot()
        caches = tuple(_LayerCache.from_snapshot(snapshot) for snapshot in before)
        tasks = []
        for layer, rows in enumerate(routes):
            misses = caches[layer].process_rows(rows)
            task_index = self._next_task_index
            self._next_task_index += 1
            tasks.append(LayerPrefetchTask(
                request_generation=self._generation,
                schedule_epoch=self._schedule_epoch,
                block_id=self._next_block_id,
                task_index=task_index,
                layer=layer,
                ring_slot=task_index % self.ring_depth,
                experts=misses,
            ))
        block = PreparedBlock(
            request_id=request_id,
            request_generation=self._generation,
            schedule_epoch=self._schedule_epoch,
            block_id=self._next_block_id,
            predicted_routes=routes,
            cache_before=before,
            cache_after=tuple(cache.snapshot() for cache in caches),
            tasks=tuple(tasks),
        )
        self._next_block_id += 1
        self._pending.append(block)
        return block

    def plan_layer(
        self,
        block_id: int,
        layer: int,
        authoritative_routes: Sequence[Sequence[int]],
    ) -> ExecutionLayerPlan:
        block = self._block(block_id)
        if layer < 0 or layer >= self.layers:
            raise IndexError(layer)
        rows = _normalize_routes(
            tuple(
                authoritative_routes if index == layer else block.predicted_routes[index]
                for index in range(self.layers)
            ),
            layers=self.layers,
            experts=self.experts,
        )[layer]
        task = block.tasks[layer]
        return build_execution_layer_plan(
            rows,
            block.cache_before[layer],
            task.experts,
            layer=layer,
        )

    def adjudicate(
        self,
        block_id: int,
        authoritative_routes: Sequence[Sequence[Sequence[int]]],
        *,
        committed_positions: int = 4,
    ) -> BlockAdjudication:
        if not self._pending or self._pending[0].block_id != block_id:
            raise RuntimeError("blocks must be adjudicated in preparation order")
        if committed_positions < 0 or committed_positions > 4:
            raise ValueError("committed_positions must be in [0, 4]")
        block = self._block(block_id)
        actual = _normalize_routes(
            authoritative_routes, layers=self.layers, experts=self.experts
        )
        compared = self.layers * 4 * 8
        exact = sum(
            predicted == observed
            for predicted_layer, actual_layer in zip(block.predicted_routes, actual)
            for predicted_row, actual_row in zip(predicted_layer, actual_layer)
            for predicted, observed in zip(predicted_row, actual_row)
        )
        plans = [self.plan_layer(block_id, layer, actual[layer]) for layer in range(self.layers)]
        staged_hits = sum(len(plan.staged_hits) for plan in plans)
        uncovered = sum(len(plan.uncovered_experts) for plan in plans)
        false_prefetches = sum(len(plan.false_prefetch_experts) for plan in plans)

        committed = tuple(
            _LayerCache.from_snapshot(snapshot) for snapshot in self.cache_snapshot()
        )
        for layer in range(self.layers):
            committed[layer].process_rows(actual[layer][:committed_positions])
        self._committed = committed

        self._pending.pop(0)
        full_exact = committed_positions == 4 and exact == compared
        invalidated: tuple[int, ...] = ()
        if full_exact:
            if tuple(cache.snapshot() for cache in committed) != block.cache_after:
                raise AssertionError("exact prediction produced a different cache transition")
        elif self._pending:
            invalidated = tuple(candidate.block_id for candidate in self._pending)
            self._pending.clear()
            self._schedule_epoch += 1
        return BlockAdjudication(
            block_id=block_id,
            committed_positions=committed_positions,
            compared_assignments=compared,
            exact_assignments=exact,
            staged_hits=staged_hits,
            uncovered_misses=uncovered,
            false_prefetches=false_prefetches,
            invalidated_block_ids=invalidated,
            requires_copy_barrier=bool(invalidated),
        )

    def abort(self, block_id: int) -> BlockAdjudication:
        block = self._block(block_id)
        return self.adjudicate(
            block_id,
            block.predicted_routes,
            committed_positions=0,
        )
