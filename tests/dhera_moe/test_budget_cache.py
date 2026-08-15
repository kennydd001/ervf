from __future__ import annotations

import pytest

from moe_lab.dhera_moe.cache import BudgetedVictimCache


def test_base_access_never_populates_cache() -> None:
    cache = BudgetedVictimCache(frozenset({(0, 7)}), layers=2, victim_capacity=1)
    assert cache.access(0, 7) == "base"
    assert cache.primary == [None, None]
    assert not cache.victim


def test_primary_miss_then_hit() -> None:
    cache = BudgetedVictimCache(frozenset(), layers=1, victim_capacity=1)
    assert cache.access(0, 4) == "miss"
    assert cache.access(0, 4) == "primary_hit"


def test_victim_hit_swaps_with_primary() -> None:
    cache = BudgetedVictimCache(frozenset(), layers=1, victim_capacity=2)
    assert cache.access(0, 4) == "miss"
    assert cache.access(0, 5) == "miss"
    assert list(cache.victim) == [(0, 4)]
    assert cache.access(0, 4) == "victim_hit"
    assert cache.primary == [(0, 4)]
    assert list(cache.victim) == [(0, 5)]


def test_global_victim_lru_evicts_oldest() -> None:
    cache = BudgetedVictimCache(frozenset(), layers=3, victim_capacity=2)
    for layer in range(3):
        assert cache.access(layer, 0) == "miss"
        assert cache.access(layer, 1) == "miss"
    assert list(cache.victim) == [(1, 0), (2, 0)]
    assert cache.access(0, 0) == "miss"


def test_reset_clears_context_state() -> None:
    cache = BudgetedVictimCache(frozenset(), layers=1, victim_capacity=1)
    cache.access(0, 1)
    cache.access(0, 2)
    cache.reset()
    assert cache.primary == [None]
    assert not cache.victim


def test_invalid_configuration_and_layer() -> None:
    with pytest.raises(ValueError):
        BudgetedVictimCache(frozenset(), layers=0)
    with pytest.raises(ValueError):
        BudgetedVictimCache(frozenset(), victim_capacity=-1)
    with pytest.raises(IndexError):
        BudgetedVictimCache(frozenset(), layers=1).access(1, 0)
