# S100 Phase 34 preregistration — temporal sparse-DOWN panel cache

Frozen before route/panel tracing.

## New mechanism

The H8 sparse DOWN path currently copies every activated 32 KiB code panel
from mapped host memory into a transient device mirror. The mirror is discarded
at the next layer even when the same `(layer, expert, panel)` is used in the
next verification window.

Phase34 first measures the exact temporal key stream, then permits an
associative persistent device panel cache only if the signal is large enough.
Scales are already resident and are not counted as panel payload.

## Frozen diagnostic

- Exact Phase25/Phase32-equivalent H8 routes at context 1024.
- 16 consecutive canonical H8 windows.
- Record every `(layer, expert, nonzero-panel)` transfer after route grouping
  and mask union.
- Simulate exact global LRU capacities of 32, 64, 96 and 128 MiB with 32 KiB
  payload per entry.
- Report cold misses, temporal hits, hit rate, bytes avoided and per-layer
  previous-window intersection.

This synchronized trace is not throughput.

## Implementation gate

Open the real cache build only when all tokens remain exact and at least one
capacity <=128 MiB has:

- steady-state hit rate >=20%;
- at least 64 MiB less host traffic over the measured 16-window trace; and
- no key aliasing or duplicate-transfer accounting errors.

The first implementation must be write-through on a miss, exact on every hit,
and fail closed to the existing gather path. It may not alter route masks,
scales, FMA order or rejection semantics.
