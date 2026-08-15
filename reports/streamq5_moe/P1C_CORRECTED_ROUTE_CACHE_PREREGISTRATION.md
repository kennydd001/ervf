# STREAMQ5-MoE P1C - corrected-semantics route-cache confirmation

Locked on 2026-08-12 after the independently verified P0C physical-scale
quality pass and before opening any P1C route output.

## Purpose and fixed candidate

P1A passed for the original fake-quantized implementation, but its routes may
not be reused for the corrected physical semantics. P1C repeats route capture
with codes selected against FP32 max-absolute scales, scales rounded to BF16
before dequantization, and BF16 materialized weights. Q5 experts and the INT8
trunk use this same rule. No cache-policy or capacity change is permitted.

## Fresh route data

Five new 1,024-token contexts are locked before routing: general, code, math,
multilingual, and instruction. Every aligned 128-token chunk must be disjoint
from CORETAIL P2, STREAMQ4 P0, original STREAMQ5 P0, STREAMQ5 P0C, and the P1A
route inputs. Routes cover the official top-8 at all 48 layers.

Per domain, tokens 0-511 are calibration, 512-767 validation, and 768-1023 the
once-only test. Validation must pass before test opens.

## Frozen cache and accounting

The physical record remains 1,011,712 bytes per matrix and 3,035,136 bytes per
expert. The full bank is 18,647,875,584 bytes (17.3671875 GiB). The cache has
exactly 1,910 slots: 40 per layer for layers 0-37 and 39 for layers 38-47.

For each `(domain, layer)`, select 32 static experts by descending calibration
frequency with ascending expert-ID ties. The dynamic exact LRU has 8 slots in
layers 0-37 and 7 in layers 38-47; it starts empty for every domain and split.
Official top-8 order is retained, static hits never enter LRU, dynamic hits
move to MRU, and every miss transfers one full expert record.

## Gates

At the locked measured pinned-H2D bandwidth of 26.158915272090432 GB/s, both
validation and test must satisfy:

- aggregate and every-domain mean projected dynamic H2D <=25 ms/token;
- aggregate and every-domain p95 projected dynamic H2D <=35 ms/token;
- conservative static domain preload <=250 ms;
- resident allocation <=7.9599609375 GiB;
- bank <=17.45 GiB and <=24 GiB pinned-host ceiling;
- valid routes, fixed hashes, and an independent verifier pass.

P1C proves only corrected-candidate routes, cache simulation, and byte
accounting. It does not prove physical packing, measured transfer of Q5
records, a Q5 compute kernel, overlap, or end-to-end wall-clock.
