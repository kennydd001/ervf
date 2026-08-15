# STREAMQ5-MoE P2C - static/dynamic rebalance preregistration

Locked on 2026-08-12 after P2B validation closed at 35.447 ms aggregate p95
and before any P2C selection output or P2B/P2C test-token access. The P2B test
partition (tokens 768-1023) remains unopened.

## Hypothesis

At the physically proven 1,640-slot capacity, retaining 32 static slots leaves
only two or three dynamic slots per layer and fails under instruction-domain
route drift. Fewer calibration-static slots and a larger exact LRU can reduce
validation misses without changing physical memory.

## Validation-only selection

Using only calibration tokens 0-511 and validation tokens 512-767 from the
already locked fresh P2B routes, evaluate uniform static counts
`[8, 12, 16, 20, 24, 28, 32]`. Each layer still has 35 total slots in layers
0-7 and 34 in layers 8-47; dynamic capacity is the remainder. Static experts
use descending calibration frequency and ascending expert-ID ties. Dynamic
LRU semantics and official top-8 ordering are unchanged.

Select the candidate with the smallest worst-domain p95 miss count; ties use
smallest aggregate mean misses, then the larger static count. No other count,
per-layer tuning, admission rule, or test observation is allowed.

## Confirmatory physical gates

After selection, rerun actual full-bank pinning, simultaneous cache+trunk+KV
residency, sampled integrity, fragmented H2D, and host-wall timing on validation.
Only a full validation pass opens the once-only test. Both splits retain P2B's
mean <=25 ms, p95 <=35 ms, every-domain, preload <=250 ms, exact simulation,
co-residency, >=384 MiB scratch, and byte-integrity gates.

P2C changes only the static/dynamic partition inside the already physical
1,640-slot cache. It proves no combined compute or end-to-end token loop.
