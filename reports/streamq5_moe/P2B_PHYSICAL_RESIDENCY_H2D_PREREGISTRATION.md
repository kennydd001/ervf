# STREAMQ5-MoE P2B - actual residency and fragmented-H2D preregistration

Locked on 2026-08-12 after the independently verified P2A kernel pass and
before opening any P2B route or transfer output.

## VRAM correction and fixed capacity

The CUDA-only diagnostic context reported 7,385,120,768 free bytes on the
8,546,484,224-byte RTX PRO 2000 GPU. The earlier nominal 1,910-slot cache does
not physically coexist with the 1,541,093,376-byte INT8 trunk and 402,653,184-
byte KV allocation under that observed footprint.

P2B fixes a stricter 1,640-slot cache (4,977,623,040 bytes), leaving at least
384 MiB free after simultaneous cache, trunk, and KV allocation. Each layer
retains 32 static slots. Layers 0-7 receive three dynamic LRU slots and layers
8-47 receive two, totaling exactly 1,640 slots. No capacity or policy sweep is
allowed after route output.

## Fresh routes and policy

Capture corrected P0C physical-semantics routes for five new 1,024-token
contexts, disjoint in aligned 128-token blocks from every prior quality and
route decision set. Tokens 0-511 calibrate the 32 static experts per
`(domain, layer)`; tokens 512-767 are validation and 768-1023 once-only test.
Static ordering, exact LRU behavior, and official top-8 request order remain
identical to P1C.

## Actual physical test

- allocate and populate all 18,647,875,584 physical bank bytes in CUDA-pinned
  host memory, preserving every layer/expert record byte;
- simultaneously allocate and touch the 1,640-slot device cache, INT8-trunk
  bytes, and KV bytes; retain >=384 MiB free for runtime scratch;
- preload all static records for each domain with real fragmented async H2D;
- issue one full 3,035,136-byte physical expert-record copy on every simulated
  miss and reuse the actual layer slots under the frozen LRU;
- record CUDA-event and host-wall latency per token, exact bytes/misses, and
  deterministic sampled source/destination integrity.

## Gates

Validation must pass before test. Both splits require:

- aggregate and every-domain mean host-wall fragmented H2D <=25 ms/token;
- aggregate and every-domain p95 host-wall fragmented H2D <=35 ms/token;
- every-domain static preload host-wall <=250 ms;
- actual misses equal an independent policy simulation exactly;
- full bank pinned, all device allocations co-resident, >=384 MiB scratch
  free, and all sampled transfers byte-exact.

P2B proves actual host pinning, device residency, cache miss traffic, and H2D
latency. It does not yet combine expert compute, trunk/attention, or a complete
token loop.
