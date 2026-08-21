# S100 Phase72 — Ornith cross-H4 rolling prefetch preregistration

Date: 2026-08-21

## Question

Does a continuous perfect-lookahead copy ring remove Phase71's repeated H4
pipeline cold start and reduce exposed LRU-52 transport below the 1.443 ms/H4
residual required for 65 tok/s?

## Frozen parent and change

Phase71 is the only timing parent. Expert size, Phase70 route trace, 52-slot
LRU/Belady replays, pinned rotating source, destination ring, dedicated copy
stream, calibrated per-layer/head waits and 28 warm H4 blocks are unchanged.

Phase71 reset the copy pipeline at every H4. Phase72 flattens all 28x40 layer
dependencies into one rolling sequence. `future = current + lead` can cross an
H4 boundary, so the next block's layer-0 transfer may run under the current
block's late layers and final head envelope. Destination reuse still waits for
the corresponding main-stream consumption event.

Frozen lookahead depths are 2, 4, 8 and 16 layer dependencies. Lead 1 is closed
by Phase71 because a single reusable buffer serializes copy and consumption.

## Measurement

- One warmup epoch and five measured full-trace epochs per arm.
- Report raw p50/H4 and `exposed = rolling - compute_only`.
- Normalize the final latency to the measured component floor as
  `60.095487602 + exposed`; do not claim the slightly lower proxy baseline.
- Report both LRU-52 and unattainable Belady-52.

## Gates

1. Source working set is at least 4x L2 and calibrated waits are within 5%.
2. Every rolling arm is at least 2% faster than Phase71 serial for its policy.
3. At least one Belady arm has floor-normalized latency <= 61.538462 ms/H4.
4. At least one LRU arm has floor-normalized latency <= 61.538462 ms/H4.
5. The selected LRU arm is no slower than the selected Belady arm by more than
   0.75 ms/H4 exposed tail; larger divergence indicates replacement quality,
   not H4 reset overhead, remains dominant.

A green result proves only that an ideal continuous DMA schedule fits the
measured component envelope. It authorizes a real route predictor/segmented
copy integration; it is not an end-to-end throughput claim.
