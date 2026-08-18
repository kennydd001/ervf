# S100 Phase 12C preregistration

Date frozen: 2026-08-18

## Inputs

- quality/performance parent: current Phase-12 worktree;
- Phase-12A exact block floor;
- Phase-12B route-union census;
- real checkpoint matrices and expert records;
- no drafter and no heldout quality selection.

## Dense ERVF-M

B={2,4,8}. For every enumerated real matrix:

1. generate deterministic FP32 activation rows;
2. run B independent current ERVF calls;
3. run one ERVF-M call;
4. require bit identity for every output element.

Timing streams all enumerated real matrices in one fixed order. Total distinct
weight bytes must exceed four times device L2. No per-matrix warm-L2 speedup is
promotable.

Gates:
- B=2 useful-row speedup >=1.75;
- B=4 >=3.20;
- B=8 >=5.50.

## Grouped MoE

Use 48 real `(layer, expert)` records selected deterministically from the frozen
route trace. This rotates more than four L2 capacities for routed-up and
routed-down independently.

Routed-up:
- one real row-major NVFP4 expert matrix;
- M={1,2,3,4,6,8} activation rows;
- exact against M independent production ERVF calls.

Routed-down:
- one real panel-major NVFP4 down record;
- one real extracted H-SCALE plane;
- deterministic 9% nonzero ReLU2-like activation masks;
- exact per-row panel order, chunk assignment and reduction;
- exact against M independent H-SCALE down calls.

The Phase-12B row-count histogram weights B=4 and B=8 economics.

Grouped B=4 gate:
- exact all M;
- M=1 candidate penalty <=15%;
- weighted routed-up+down useful speedup >=1.20.

## Decision

`INTEGRATED_VERIFIER_BUILD_OPEN=true` only if:

- dense B=4 gate passes;
- grouped B=4 gate passes;
- both streams satisfy cold rotation and exactness.

The economics file may project a perfect-draft cycle, but projections are
explicitly non-claims.
