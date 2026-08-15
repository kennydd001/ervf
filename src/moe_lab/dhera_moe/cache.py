from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal


CacheEvent = Literal["base", "primary_hit", "victim_hit", "miss"]
ExpertKey = tuple[int, int]


@dataclass
class BudgetedVictimCache:
    """Fixed per-layer primary cache with one shared LRU victim cache."""

    base: frozenset[ExpertKey]
    layers: int = 48
    victim_capacity: int = 8
    primary: list[ExpertKey | None] = field(init=False)
    victim: OrderedDict[ExpertKey, None] = field(init=False)

    def __post_init__(self) -> None:
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.victim_capacity < 0:
            raise ValueError("victim_capacity cannot be negative")
        self.reset()

    def reset(self) -> None:
        self.primary = [None] * self.layers
        self.victim = OrderedDict()

    def _insert_victim(self, key: ExpertKey | None) -> None:
        if key is None or self.victim_capacity == 0:
            return
        self.victim.pop(key, None)
        self.victim[key] = None
        if len(self.victim) > self.victim_capacity:
            self.victim.popitem(last=False)

    def access(self, layer: int, expert: int) -> CacheEvent:
        if not 0 <= layer < self.layers:
            raise IndexError("layer outside cache")
        key = (layer, expert)
        if key in self.base:
            return "base"
        if self.primary[layer] == key:
            return "primary_hit"
        if key in self.victim:
            self.victim.pop(key)
            old_primary = self.primary[layer]
            self.primary[layer] = key
            self._insert_victim(old_primary)
            return "victim_hit"

        old_primary = self.primary[layer]
        self.primary[layer] = key
        self._insert_victim(old_primary)
        return "miss"

    def access_route(self, layer: int, experts: Iterable[int]) -> list[CacheEvent]:
        return [self.access(layer, expert) for expert in experts]
