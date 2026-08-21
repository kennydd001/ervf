# S100 Phase44 — perfect H4 expert-prefetch oracle

## Frozen parent

`codex/s100-phase31-critical-path@1046da1` is the only performance parent.
Its context-1024 thermal median is 63.53125 ms/H4 (62.961 target-only
tok/s), with exact state, logits and token IDs.

## Question

Can perfect knowledge of the 24 routed experts in every MoE layer remove at
least 2.0 ms/H4 by starting the existing UP-code, UP-scale and H-SCALE-plane
transfers before their ordinary post-router position?

This is an oracle ceiling, not an implementable predictor.  The real routers
and all model arithmetic still execute.  Their 24 IDs are compared on device
against the frozen oracle routes at every MoE layer.  Any mismatch closes the
run.

## Mechanism

1. Capture the true per-layer routes for the exact frozen calibration window.
2. Build a fresh Phase31 parent graph.
3. Enqueue cache assignment and the ordinary UP/plane fetches on a dedicated
   prefetch stream using one of the frozen schedules below.
4. Continue the ordinary verifier on the main graph stream.
5. At each MoE layer, wait only for that layer's prefetch event, then run the
   unchanged authoritative router, grouping, UP, scan, sparse DOWN, reduction
   and accumulation arithmetic.

The cache capacity, LRU assignment kernel, source bytes, destination cache
layout and H-SCALE planes are unchanged.  Only transfer start time changes.

Frozen schedules:

- `layer_now`: immediately before the authoritative router in that layer;
- `moe_l1`: one MoE layer ahead, staggered through the graph;
- `moe_l2`: two MoE layers ahead, staggered through the graph;
- `block_all`: all 23 layer transfers at H4 graph entry, as a contention
  control and absolute full-block lookahead experiment.

## Frozen screen

- context: 1024
- warmup: 8 H4 blocks
- measured: 16 H4 blocks per arm
- order: parent A, oracle candidate, parent B
- exact generated IDs required for every block
- zero route-oracle mismatches required
- baseline drift must be at most 5%

The research route opens only when the bootstrap lower-95 saving is at least
2.0 ms/H4.  A mean saving of at least 4.0 ms/H4 is classified as a structural
breakthrough.  Below 2.0 ms/H4, learned expert-prefetch is closed against this
parent; predictor training is not authorized by this experiment.

## Claim boundary

Exact target-only H4 verifier oracle on the local RTX PRO 2000 Blackwell
Laptop GPU.  The oracle has perfect future-route knowledge and includes no
predictor, drafter, rollback, rejection or extra-prefetch-byte cost.
