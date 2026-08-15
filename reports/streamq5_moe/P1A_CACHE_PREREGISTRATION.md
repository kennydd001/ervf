# STREAMQ5-MoE P1A - physical accounting and fresh route-cache preregistration

Locked on 2026-08-12 after the independently verified P0 quality pass and
before opening any P1A route output.

## Fixed physical accounting

Each gate/up/down matrix has 1,572,864 weights. Codes are packed as an exact
little-order 5-bit row stream (983,040 bytes), followed by 12,288 BF16
group-128 scales (24,576 bytes) and a 64-byte header. Every matrix record is
aligned to 4,096 bytes, yielding 1,011,712 bytes per matrix and 3,035,136 bytes
per expert. All headers, padding, scales, indices, and staging count.

The complete 6,144-expert host bank is therefore 18,647,875,584 bytes
(17.3671875 GiB) before any optional file-level manifest. It must remain below
17.45 GiB and fit within a 24-GiB pinned-host-bank ceiling.

GPU accounting is fixed as:

- INT8 trunk: 1.4352550506591797 GiB;
- 4K BF16 KV cache: 0.375 GiB;
- runtime/staging reserve: 0.75 GiB;
- remaining expert cache: 5.39970588684082 GiB on 7.9599609375 GiB VRAM.

Exactly 1,910 aligned expert records fit. Capacity is partitioned by layer:
40 slots in layers 0-37 and 39 slots in layers 38-47.

## Fresh route data and policy

Five new 1,024-token contexts are locked before routing: general, code, math,
multilingual, and instruction. They must be exact-context disjunct from all
CORETAIL, STREAMQ4, and STREAMQ5 P0 decision contexts.

Routes are captured from the exact fake-quantized Q5+INT8 candidate at all 48
layers. Within each domain, tokens 0-511 are calibration, 512-767 validation,
and 768-1023 once-only test.

The cache policy is fixed per `(domain, layer)`:

- 32 static experts selected by descending calibration frequency, expert ID
  ascending for ties;
- a dynamic exact LRU of 8 slots in layers 0-37 and 7 slots in layers 38-47;
- the dynamic tier starts empty for every evaluated domain/split;
- top-8 requests are processed in official slot order; static hits do not
  enter LRU, dynamic hits move to MRU, misses transfer the full expert record
  and enter LRU;
- a domain switch conservatively preloads all 32 x 48 static records.

No capacity, static/dynamic split, admission, or eviction sweep is allowed.

## Gates

Using the independently measured pinned H2D bandwidth of 26.158915272090432
GB/s, both validation and test must satisfy:

- mean projected dynamic H2D <=25 ms/token;
- p95 projected dynamic H2D <=35 ms/token;
- conservative full static domain preload <=250 ms;
- resident formula <=7.9599609375 GiB;
- all routes valid, all hashes correct, and an independent verifier passes.

Validation must pass before test opens. P1A is accounting and route-cache
evidence only. It does not prove physical packing, a Q5 kernel, overlap, or
end-to-end tokens per second.
