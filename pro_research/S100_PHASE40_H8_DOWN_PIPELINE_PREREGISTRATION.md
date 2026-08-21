# S100 Phase40 — exact H8 sparse-down transfer/compute pipeline

Frozen before Phase40 GPU timing.

## Question

Phase32's exact `dense_m8` H8 verifier overlaps its shared-expert branch, but
still launches the 48-group resident-scale sparse-down gather and all 48 routed
down projections monolithically. Phase25 profiling attributed about 22.8 ms/H8
to down gather and 6.8 ms/H8 to sparse-down compute/reduce. Phase27 proved that
disjoint group ranges can pipeline these stages for H4. This experiment ports
that exact mechanism to H8.

## Frozen arms

- `BASE_A`: unmodified Phase32 `dense_m8`, context 1024, fresh process.
- `PIPELINE_B3`: the same graph with 48 group slots split into three fixed
  ranges `(0,16)`, `(16,32)`, `(32,48)`. Gather uses `grid.y=4` on a dedicated
  stream; the main stream executes each range's route/chunk down projection as
  soon as its gather event is ready.
- `BASE_B`: unmodified Phase32 `dense_m8`, fresh process after the candidate.

Every active route/chunk executes the same sparse-down body and writes the same
partial index. Reduction and slot-0-to-slot-5 accumulation remain unchanged.
Group ranges write disjoint mirror regions. No arithmetic, cache capacity,
resident-plane set, model state, or output quality changes are permitted.

Protocol: canonical context 1024, four warmup H8 windows and sixteen measured
H8 windows per arm, synchronous replay/readback, fresh process per arm.

## Frozen gates

- `G40-C1`: all three arms produce every canonical token exactly.
- `G40-R1`: both new kernels report zero local-memory bytes.
- `G40-D1`: baseline median drift is at most 5% of the baseline midpoint.
- `G40-P1`: candidate median is at least 3% below the baseline midpoint.
- `G40-P2`: candidate median is at most 120 ms/H8 (strong milestone, reported
  independently and not required for correctness).

Passing C1, R1, D1, and P1 opens thermal/state promotion. It does not establish
speculative or end-to-end throughput. The S100 perfect-draft ceiling remains
80 ms/H8.

