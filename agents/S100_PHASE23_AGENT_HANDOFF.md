# Phase23 handoff

Flatten route_ref = token*6 + route_slot and preserve 0..23 order.

`cache_assign_h4` is the existing device LRU algorithm with loop bound 24 and
updates the same slot_of/expert_of/last_used/state2 tables as V6.

Groups are created in first-occurrence route order. Since top-k has unique ids
per token and H=4, expert multiplicity M is at most 4.

Up:
M1/M2/M3/M4 use ERVF virtual-thread reduction order. Repeated experts reuse one
weight stream across M activation rows.

Down:
Only PCIe gather uses the union of nonzero columns. Every route keeps its own
panel masks and ascending panel list for arithmetic, preserving per-row order.

Accumulation:
shared expert first, then route slots s=0..5 via fmaf, no atomics/reordering.

Phase22 graph infrastructure is reused; only MoE changes.
