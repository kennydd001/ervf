# S100 phase 6 research state

Date: 2026-08-17
Canonical tracked parent: `pro-research@c839060bf8e03b34c12401b123f788399db807e5`
Runtime parent: completed phase-5 worktree.

## What phase 5 actually found

Phase 5 selected no candidate because selection required every **strict**
validation gate. That is not the same as saying every nonzero arm failed the
frozen final fidelity gates.

Two global down-activation thresholds passed every official validation gate:

- alpha 0.0003;
- alpha 0.0010.

They missed only the additional exploratory strict p95-KL threshold of 0.060.
The frozen final heldout p95-KL gate remains 0.080. Heldout was never read, so
these arms can be frozen now and evaluated honestly.

The smallest four-layer K5 portfolio was also close: its validation p95 KL was
0.0823. It remains a confirmatory candidate under unchanged final gates, not a
promoted result.

## First-principles kernel hypothesis

V18 currently:

1. computes routed ReLU2 activations;
2. scans nonzero columns;
3. copies selected down-projection code columns from mapped host memory into a
   VRAM mirror;
4. reads the same codes again from VRAM in the down GEMV;
5. reduces chunk partials into a temporary contribution tensor;
6. launches another kernel to apply route weights.

Phase 6 tests three exact changes:

- `BALLOT_FUSED`: replace per-value atomic mask construction with warp ballots,
  and fuse partial reduction with route-weight accumulation;
- `DIRECT`: eliminate the mirror copy and let the masked GEMV read the mapped
  host code bytes directly while scales remain resident in VRAM;
- `DIRECT_OPT`: compose both exact changes.

The direct kernel reads the same expert, panel, code byte, resident scale byte
and activation in the same panel/column order. Only the location and timing of
the code-byte read changes. Full token parity and a destructive routing control
are mandatory before timing can be promoted.

## Phase-6 quality frontier

The fixed validation grid is frozen before phase-6 execution:

Threshold-only:

- 0.0003, 0.0010, 0.0015, 0.0020, 0.0025.

Static K portfolios, derived from phase-5 calibration only:

- K1: layer 40 -> K5;
- K2: layers 40, 34 -> K5;
- K3: layers 40, 34, 49 -> K5;
- K4: layers 40, 34, 49, 47 -> K5.

Compositions:

- alpha 0.0010 + K1;
- alpha 0.0010 + K2;
- alpha 0.0015 + K1.

Every validation-official-pass arm is frozen and sent to the untouched 5,120
heldout targets. Final gates are unchanged. No strict gate is retrospectively
renamed as a final gate.

## Success definitions

- Quality-green improvement: heldout `v18_fidelity_candidate` plus valid fresh
  timing against QFAST.
- S100-single: <=10.000 ms/useful token plus heldout quality green.
- Exact backend improvement: exact QFAST token parity plus valid fresh timing.

Component projections, direct-kernel microseconds and validation quality do not
count as S100.
