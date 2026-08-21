from __future__ import annotations

import pytest

from moe_lab.ornith.page_cache import (
    ORNITH_LOGICAL_CACHE_SLOTS,
    OrnithPageCache,
    PhysicalPageLRU,
)


def test_default_has_52_logical_slots_and_configurable_staging_slots():
    cache = PhysicalPageLRU(staging_slots=3)
    snapshot = cache.snapshot()

    assert len(snapshot.logical_to_physical) == ORNITH_LOGICAL_CACHE_SLOTS == 52
    assert len(snapshot.staging_to_physical) == 3
    assert isinstance(cache, OrnithPageCache)


def test_miss_promotes_by_page_table_swap_without_payload_copy():
    cache = PhysicalPageLRU(logical_slots=2, staging_slots=1)
    first = cache.plan([("page-a",)])
    cache.commit(first)
    before = cache.snapshot()

    plan = cache.plan([("page-b",)])
    promotion = plan.promotions[0]

    assert promotion.source == "external"
    assert promotion.logical_slot == 1
    assert promotion.staging_slot == 0
    assert promotion.evicted_page is None
    assert promotion.page_table_swap is True
    assert promotion.payload_copy is False
    assert plan.payload_copy_bytes == 0
    assert plan.after.logical_to_physical == ("page-a", "page-b")
    assert before.logical_to_physical == ("page-a", None)

    cache.commit(plan)
    assert cache.snapshot().logical_to_physical == ("page-a", "page-b")


def test_full_logical_slot_swaps_old_physical_page_into_staging():
    cache = PhysicalPageLRU(logical_slots=1, staging_slots=1)
    cache.commit(cache.plan([("old",)]))

    plan = cache.plan([("new",)])
    promotion = plan.promotions[0]

    assert promotion.logical_slot == 0
    assert promotion.staging_slot == 0
    assert promotion.evicted_page == "old"
    assert plan.after.logical_to_physical == ("new",)
    assert plan.after.staging_to_physical == ("old",)
    cache.commit(plan)
    cache.assert_invariants()
    assert cache.snapshot().physical_to_location == {
        "new": ("logical", 0),
        "old": ("staging", 0),
    }


def test_lru_is_stable_and_ties_choose_lowest_slot():
    cache = PhysicalPageLRU(logical_slots=2, staging_slots=1)
    cache.commit(cache.plan([("a", "b")]))
    cache.commit(cache.plan([("b",)]))

    plan = cache.plan([("c",)])
    assert plan.promotions[0].evicted_page == "a"
    assert plan.promotions[0].logical_slot == 0

    cache.commit(plan)
    assert cache.snapshot().logical_to_physical == ("c", "b")


def test_partial_h4_ignores_padded_rows_entirely():
    cache = PhysicalPageLRU(logical_slots=4, staging_slots=2)
    rows = (("active-0",), ("active-1",), ("pad-0",), ("pad-1",))

    plan = cache.plan_h4(rows, valid_rows=2)

    assert plan.valid_rows == 2
    assert plan.padded_rows == (("pad-0",), ("pad-1",))
    assert [access.row for access in plan.accesses] == [0, 1]
    assert plan.misses == ("active-0", "active-1")
    assert "pad-0" not in plan.after.physical_pages
    assert "pad-1" not in plan.after.physical_pages

    cache.commit(plan)
    assert cache.snapshot().logical_to_physical[:2] == ("active-0", "active-1")
    assert "pad-0" not in cache.snapshot().physical_pages


def test_abort_has_no_state_mutation_and_commit_rejects_stale_plan():
    cache = PhysicalPageLRU(logical_slots=2, staging_slots=1)
    before = cache.snapshot()
    plan = cache.plan_h4((("x",), (), ("padded",), ()), valid_rows=1)

    assert cache.abort(plan) == before
    assert cache.snapshot() == before

    cache.commit(plan)
    with pytest.raises(ValueError, match="stale"):
        cache.commit(plan)


def test_physical_page_bijection_is_preserved_after_repeated_swaps():
    cache = PhysicalPageLRU(logical_slots=3, staging_slots=2)
    for page in ("a", "b", "c", "d", "e", "f", "b", "a"):
        plan = cache.plan([(page,)])
        cache.commit(plan)
        snapshot = cache.snapshot()
        pages = snapshot.physical_pages
        assert len(pages) == len(set(pages))
        assert len(pages) <= 5
        cache.assert_invariants()


def test_snapshots_are_deterministic_for_same_sequence():
    rows = (("a", "b"), ("c",), ("d",), ("e",))
    left = PhysicalPageLRU(logical_slots=3, staging_slots=2)
    right = PhysicalPageLRU(logical_slots=3, staging_slots=2)

    left.commit(left.plan_h4(rows, valid_rows=3))
    right.commit(right.plan_h4(rows, valid_rows=3))

    assert left.snapshot() == right.snapshot()
    assert left.snapshot().physical_pages == right.snapshot().physical_pages


@pytest.mark.parametrize("bad_slots", [0, -1])
def test_staging_requires_at_least_one_page_table_slot(bad_slots):
    with pytest.raises(ValueError, match="staging_slots"):
        PhysicalPageLRU(staging_slots=bad_slots)
