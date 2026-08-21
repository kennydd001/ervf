"""Small transactional physical-page LRU cache for Ornith.

The cache models page-table metadata, rather than copying expert payloads.
On a miss a physical page is made visible in a staging slot and promotion is
represented by swapping page-table entries with the selected logical slot.
Consequently a promotion has no D2D payload-copy operation associated with it.

The public API deliberately has no speculative acceptance/rejection state
machine.  A caller creates an immutable plan, commits it atomically, or
aborts it without changing cache state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence


ORNITH_LOGICAL_CACHE_SLOTS = 52
H4_ROWS = 4

PageId = Hashable
OperationKind = Literal["hit", "promote"]
PromotionSource = Literal["staging", "external"]


@dataclass(frozen=True)
class PageCacheSnapshot:
    """Deterministic, immutable page-table and LRU metadata snapshot."""

    logical_to_physical: tuple[PageId | None, ...]
    staging_to_physical: tuple[PageId | None, ...]
    logical_last_used: tuple[int | None, ...]
    staging_last_used: tuple[int | None, ...]
    clock: int
    epoch: int

    @property
    def physical_pages(self) -> tuple[PageId, ...]:
        """All mapped physical pages in deterministic slot order."""

        return tuple(
            page
            for page in (*self.logical_to_physical, *self.staging_to_physical)
            if page is not None
        )

    @property
    def logical_page_to_slot(self) -> dict[PageId, int]:
        return {
            page: slot
            for slot, page in enumerate(self.logical_to_physical)
            if page is not None
        }

    @property
    def physical_to_location(self) -> dict[PageId, tuple[str, int]]:
        """The unique logical/staging location of every mapped page."""

        locations: dict[PageId, tuple[str, int]] = {}
        for slot, page in enumerate(self.logical_to_physical):
            if page is not None:
                locations[page] = ("logical", slot)
        for slot, page in enumerate(self.staging_to_physical):
            if page is not None:
                locations[page] = ("staging", slot)
        return locations


@dataclass(frozen=True)
class PagePromotion:
    """One physical-page promotion into a logical cache slot.

    ``page_table_swap`` and ``payload_copy`` are explicit so callers can
    assert the intended transport primitive in tests and instrumentation.
    """

    page: PageId
    logical_slot: int
    staging_slot: int
    source: PromotionSource
    evicted_page: PageId | None
    page_table_swap: bool = True
    payload_copy: bool = False


@dataclass(frozen=True)
class PageAccess:
    """The planned result of one active H4 page access."""

    row: int
    position: int
    page: PageId
    kind: OperationKind
    logical_slot: int
    staging_slot: int | None
    promotion_index: int | None


@dataclass(frozen=True)
class PageCachePlan:
    """Immutable transaction produced from one cache snapshot."""

    before: PageCacheSnapshot
    after: PageCacheSnapshot
    rows: tuple[tuple[PageId | None, ...], ...]
    valid_rows: int
    accesses: tuple[PageAccess, ...]
    promotions: tuple[PagePromotion, ...]
    misses: tuple[PageId, ...]

    @property
    def padded_rows(self) -> tuple[tuple[PageId | None, ...], ...]:
        return self.rows[self.valid_rows :]

    @property
    def payload_copy_bytes(self) -> int:
        """Always zero: promotion changes mappings, never payload bytes."""

        return 0


@dataclass
class _MutableState:
    logical: list[PageId | None]
    staging: list[PageId | None]
    logical_age: list[int | None]
    staging_age: list[int | None]
    clock: int
    epoch: int


class PhysicalPageLRU:
    """A deterministic 52-slot logical LRU backed by physical page mappings.

    ``staging_slots`` controls the number of temporary physical-page mappings
    available while a miss is promoted.  Logical and staging arrays together
    form a bijection over all resident physical pages: no page can appear in
    two slots, and every non-empty mapping has exactly one location.
    """

    def __init__(
        self,
        *,
        logical_slots: int = ORNITH_LOGICAL_CACHE_SLOTS,
        staging_slots: int = 2,
    ) -> None:
        if logical_slots <= 0:
            raise ValueError("logical_slots must be positive")
        if staging_slots <= 0:
            raise ValueError("staging_slots must be positive")
        self.logical_slots = int(logical_slots)
        self.staging_slots = int(staging_slots)
        self._state = _MutableState(
            logical=[None] * self.logical_slots,
            staging=[None] * self.staging_slots,
            logical_age=[None] * self.logical_slots,
            staging_age=[None] * self.staging_slots,
            clock=0,
            epoch=0,
        )
        self.assert_invariants()

    def snapshot(self) -> PageCacheSnapshot:
        """Return a deterministic immutable snapshot of current state."""

        state = self._state
        return PageCacheSnapshot(
            logical_to_physical=tuple(state.logical),
            staging_to_physical=tuple(state.staging),
            logical_last_used=tuple(state.logical_age),
            staging_last_used=tuple(state.staging_age),
            clock=state.clock,
            epoch=state.epoch,
        )

    def assert_invariants(self) -> None:
        """Raise ``AssertionError`` if mappings or LRU metadata are invalid."""

        state = self._state
        if len(state.logical) != self.logical_slots:
            raise AssertionError("logical page table has the wrong size")
        if len(state.staging) != self.staging_slots:
            raise AssertionError("staging page table has the wrong size")
        if len(state.logical_age) != self.logical_slots:
            raise AssertionError("logical LRU table has the wrong size")
        if len(state.staging_age) != self.staging_slots:
            raise AssertionError("staging LRU table has the wrong size")
        pages = [page for page in (*state.logical, *state.staging) if page is not None]
        try:
            if len(set(pages)) != len(pages):
                raise AssertionError("a physical page is mapped more than once")
        except TypeError as exc:
            raise AssertionError("physical page IDs must be hashable") from exc
        if len(self.snapshot().physical_to_location) != len(pages):
            raise AssertionError("physical page mapping is not bijective")
        for page, age in zip(state.logical, state.logical_age):
            if (page is None) != (age is None):
                raise AssertionError("logical page and LRU metadata disagree")
        for page, age in zip(state.staging, state.staging_age):
            if (page is None) != (age is None):
                raise AssertionError("staging page and LRU metadata disagree")
        if state.clock < 0 or state.epoch < 0:
            raise AssertionError("clock and epoch must be non-negative")
        ages = [age for age in (*state.logical_age, *state.staging_age) if age is not None]
        if any(age <= 0 or age > state.clock for age in ages):
            raise AssertionError("LRU age is outside the clock range")

    def plan_h4(
        self,
        rows: Sequence[Sequence[PageId | None]],
        *,
        valid_rows: int = H4_ROWS,
    ) -> PageCachePlan:
        """Plan a four-row H4 transaction.

        Only the first ``valid_rows`` rows are active.  Remaining rows are
        treated as padding and are copied into the plan for observability, but
        never looked up, staged, promoted, or committed.
        """

        if len(rows) != H4_ROWS:
            raise ValueError(f"expected exactly {H4_ROWS} H4 rows")
        if valid_rows < 0 or valid_rows > H4_ROWS:
            raise ValueError("valid_rows must be in [0, 4]")
        normalized = tuple(tuple(row) for row in rows)
        return self._plan(normalized, valid_rows=valid_rows)

    def plan(
        self,
        rows: Sequence[Sequence[PageId | None]],
        *,
        valid_rows: int | None = None,
    ) -> PageCachePlan:
        """Plan a generic sequence, with optional trailing padded rows."""

        normalized = tuple(tuple(row) for row in rows)
        active_rows = len(normalized) if valid_rows is None else int(valid_rows)
        if active_rows < 0 or active_rows > len(normalized):
            raise ValueError("valid_rows must be within the supplied rows")
        return self._plan(normalized, valid_rows=active_rows)

    def _plan(
        self,
        rows: tuple[tuple[PageId | None, ...], ...],
        *,
        valid_rows: int,
    ) -> PageCachePlan:
        state = _MutableState(
            logical=list(self._state.logical),
            staging=list(self._state.staging),
            logical_age=list(self._state.logical_age),
            staging_age=list(self._state.staging_age),
            clock=self._state.clock,
            epoch=self._state.epoch,
        )
        accesses: list[PageAccess] = []
        promotions: list[PagePromotion] = []
        misses: list[PageId] = []

        for row_index, row in enumerate(rows[:valid_rows]):
            for position, page in enumerate(row):
                self._validate_page_id(page)
                logical_slot = self._find(state.logical, page)
                if logical_slot is not None:
                    self._touch_logical(state, logical_slot)
                    accesses.append(PageAccess(
                        row=row_index,
                        position=position,
                        page=page,  # type: ignore[arg-type]
                        kind="hit",
                        logical_slot=logical_slot,
                        staging_slot=None,
                        promotion_index=None,
                    ))
                    continue

                misses.append(page)  # type: ignore[arg-type]
                staging_slot = self._find(state.staging, page)
                source: PromotionSource
                if staging_slot is None:
                    staging_slot = self._choose_staging_slot(state)
                    state.staging[staging_slot] = page
                    state.staging_age[staging_slot] = state.clock + 1
                    source = "external"
                else:
                    source = "staging"

                logical_slot = self._choose_logical_slot(state)
                evicted = state.logical[logical_slot]
                state.clock += 1
                state.logical[logical_slot] = page
                state.logical_age[logical_slot] = state.clock
                state.staging[staging_slot] = evicted
                state.staging_age[staging_slot] = state.clock if evicted is not None else None
                promotion_index = len(promotions)
                promotions.append(PagePromotion(
                    page=page,  # type: ignore[arg-type]
                    logical_slot=logical_slot,
                    staging_slot=staging_slot,
                    source=source,
                    evicted_page=evicted,
                ))
                accesses.append(PageAccess(
                    row=row_index,
                    position=position,
                    page=page,  # type: ignore[arg-type]
                    kind="promote",
                    logical_slot=logical_slot,
                    staging_slot=staging_slot,
                    promotion_index=promotion_index,
                ))
                self._assert_state(state)

        after = self._snapshot_of(state, epoch=state.epoch + 1)
        return PageCachePlan(
            before=self.snapshot(),
            after=after,
            rows=rows,
            valid_rows=valid_rows,
            accesses=tuple(accesses),
            promotions=tuple(promotions),
            misses=tuple(misses),
        )

    def commit(self, plan: PageCachePlan) -> PageCacheSnapshot:
        """Atomically commit a plan produced from the current snapshot."""

        if plan.before != self.snapshot():
            raise ValueError("stale page-cache plan")
        self._state = _MutableState(
            logical=list(plan.after.logical_to_physical),
            staging=list(plan.after.staging_to_physical),
            logical_age=list(plan.after.logical_last_used),
            staging_age=list(plan.after.staging_last_used),
            clock=plan.after.clock,
            epoch=plan.after.epoch,
        )
        self.assert_invariants()
        return self.snapshot()

    def abort(self, plan: PageCachePlan) -> PageCacheSnapshot:
        """Abort a plan without changing state, even if it contains misses."""

        if plan.before != self.snapshot():
            raise ValueError("cannot abort a plan from a different cache state")
        return self.snapshot()

    @staticmethod
    def _find(table: Sequence[PageId | None], page: PageId | None) -> int | None:
        if page is None:
            return None
        try:
            return table.index(page)
        except ValueError:
            return None

    @staticmethod
    def _validate_page_id(page: PageId | None) -> None:
        if page is None:
            raise ValueError("active page IDs may not be None")
        try:
            hash(page)
        except TypeError as exc:
            raise TypeError("physical page IDs must be hashable") from exc

    @staticmethod
    def _touch_logical(state: _MutableState, slot: int) -> None:
        state.clock += 1
        state.logical_age[slot] = state.clock

    def _choose_logical_slot(self, state: _MutableState) -> int:
        for slot, page in enumerate(state.logical):
            if page is None:
                return slot
        return min(
            range(self.logical_slots),
            key=lambda slot: (state.logical_age[slot], slot),  # type: ignore[arg-type]
        )

    def _choose_staging_slot(self, state: _MutableState) -> int:
        for slot, page in enumerate(state.staging):
            if page is None:
                return slot
        return min(
            range(self.staging_slots),
            key=lambda slot: (state.staging_age[slot], slot),  # type: ignore[arg-type]
        )

    @staticmethod
    def _snapshot_of(state: _MutableState, *, epoch: int) -> PageCacheSnapshot:
        return PageCacheSnapshot(
            logical_to_physical=tuple(state.logical),
            staging_to_physical=tuple(state.staging),
            logical_last_used=tuple(state.logical_age),
            staging_last_used=tuple(state.staging_age),
            clock=state.clock,
            epoch=epoch,
        )

    def _assert_state(self, state: _MutableState) -> None:
        original = self._state
        try:
            self._state = state
            self.assert_invariants()
        finally:
            self._state = original


OrnithPageCache = PhysicalPageLRU

__all__ = [
    "H4_ROWS",
    "ORNITH_LOGICAL_CACHE_SLOTS",
    "OrnithPageCache",
    "PageAccess",
    "PageCachePlan",
    "PageCacheSnapshot",
    "PagePromotion",
    "PhysicalPageLRU",
]
