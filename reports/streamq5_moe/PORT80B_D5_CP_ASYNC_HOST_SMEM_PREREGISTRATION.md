# PORT80B-D5 — mapped-host `cp.async` to SMEM preregistration

**Frozen before physical execution:** 2026-08-12

D3R showed that scalar ordinary SM loads from mapped host memory are exact but
far too slow. D4R showed that native copy batching is exact but remains above
45 ms. D5 is the final locally executable DirectPath mechanism: one persistent
grid moves 16-byte vectors from mapped host records directly to 4-KiB shared-
memory tiles with CUDA pipeline `cp.async`, then writes the tile to the output
oracle buffer. A future Q5 kernel would consume the SMEM tile instead of writing
it back; this test deliberately keeps the full output for correctness.

This is an asynchronous global-to-SMEM probe, not a claim that descriptor-based
TMA tensor maps or DAK's complete kernel are implemented.

## Frozen protocol

- immutable P0 bank; same 307-expert/layer (60%, 27.826-GiB) read-only mapped
  registration as D2–D4R;
- 480 selected records/token, 495 exact 4-KiB tiles/record, 237,600 tiles and
  973,209,600 bytes/token;
- 256 threads/block; each warp-coalesced lane copies one 16-byte vector;
- block schedules 256/512/1024/2048;
- 4 warm-ups and 16 validation samples/schedule, rotating and reversed;
- select lowest validation p50, ties by fewer blocks;
- exact full-buffer structural verification before timing;
- test opens only when mismatch count is zero and validation p50 <=65 ms;
- once-only 120-token test, no retune.

## Gates

Mechanism pass: zero mismatches, 120 finite samples, p95 <=65 ms, >=15 GB/s,
48 registration ranges, clean unregister and no CUDA/runner error.

Strong transport pass: p95 <=45 ms and >=21.627 GB/s. A pass remains 60%-bank
only because D2 already proved full-bank registration fails on this 64-GiB
system.

## Claim boundary

Synthetic byte movement only: no Q5 multiply/reduction, TMA tensor descriptor,
real 80B weights, quality, dense shell, tokens/s or endurance. `cp.async` and
mapped host memory are prior art.
