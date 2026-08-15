# PORT80B-D6 — exact mapped-host Q5 fusion preregistration

**Frozen before physical execution:** 2026-08-12

D5 is the first strong DirectPath byte pass: 43.708-ms p95 and 22.266 GB/s on
the 60%-bank registered path. D6 replaces its full HBM output copy by the
actual synthetic-80B Q5 computation. Each block pipelines 32 quantized rows
from mapped host records into SMEM and immediately executes the frozen ERVF
width-8 reduction. No complete expert or dequantized matrix is staged in HBM.

## Frozen semantics and shape

- 48 layers, ten routed experts/layer, hidden 2048, intermediate 512;
- gate/up/down matrices each contain 1,048,576 weights, 655,360 Q5 code bytes
  and 16,384 BF16 scale bytes in the immutable P0 record layout;
- official synthetic N4B-R arithmetic: exact Q5 decode, BF16 weight rounding,
  width-8 ERVF virtual-256 reduction, BF16 output cast, then canonical
  two-stage BF16 SwiGLU before down;
- deterministic FP32 input from seed `120826`;
- mapped source routes restricted to the registered 307-expert/layer prefix;
- a resident-HBM width-8 kernel with the same source arithmetic is the oracle;
- compare gate, up and down outputs for all ten experts and all 48 layers:
  1,474,560 float32 values per arm, bitwise, with SHA-256 digests.

The remote kernels use 256 threads, 32 rows/block and fixed 16-byte
global-to-SMEM `cp.async` operations (8-byte only for the compact down-scale
row). This is one locked schedule inherited from D5 and N4B-R; no tuning.

## Timing

- exactness opens timing;
- 5 warm-ups;
- 24 validation samples; test opens at validation p50 <=65 ms;
- 120 once-only test samples;
- one sample is the complete 48-layer ten-expert plane:
  mapped-host gate/up, SwiGLU and mapped-host down for every layer;
- no post-validation changes.

## Gates

Primary exact-host-expert pass:

- 0 bit differences and equal full-output digests;
- 120 finite test samples;
- test p95 <=65.0 ms;
- effective remote Q5 payload bandwidth >=15.0 GB/s at p95;
- test p95 + frozen dense-shell conservative p95 28.077227 ms <=100 ms;
- 48 registration ranges, clean unregister and no CUDA/runner error.

Strong plane gate: p95 <=55 ms and projected total <=90 ms. The bandwidth
metric divides the exact 973,209,600 remote payload bytes by full Q5 plane time;
it therefore includes computation and is deliberately conservative.

## Claim boundary

This is the exact synthetic Q5 active expert plane on a 60%-registered bank.
It is not a full-bank result, real Qwen3-Coder-Next checkpoint, natural routing,
quality test, physical dense shell, end-to-end decode, tokens/s or endurance.
`cp.async`, mapped memory and Q5 GEMV are prior art; only the exact fused local
combination is tested here.
