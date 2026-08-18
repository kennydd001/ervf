# Phase 8 overnight preregistration

Default wall budget: 8 hours.

## A — inventory / capability

Record Intel OpenCL device, driver, work-group limit and extensions relevant to
subgroups, integer dot product, command buffers, external memory and external
semaphores.

## B — route/cache

Run 8192 causal tokens with the live QFAST runtime. Preserve actual need[] miss
counts, route overlap, hot-set coverage and LRU capacity curves.

## C — real snapshots

Capture offsets 0, 1, 4 and 16. At every offset export all 23 live MoE layers.
Use the runtime's discovered hidden/intermediate/top-k dimensions. No stale model
shape constants may drive the export.

## D — Arc real NVFP4

For every snapshot use the existing independent NumPy decoder and Intel OpenCL
kernel. N={1,2,4,6}; strict and fast-math; local={64,128,256}. Strict gates:
finite, cosine >=0.999, NRMSE <=0.02. N=6 is the primary route.

## E — RTX direct reference

On the same offsets and all layers measure the current RTX path. Report
`down_only` and `scale_fetch_serial_plus_down` separately. The former is the
primary comparator because H-SCALE fetch can overlap prior work in production;
the latter is a conservative bound.

## F — full-bank cold-residency control

Export one complete real routed-down bank from the middle MoE layer. Preload it once
into an Intel-GPU buffer, rotate random distinct six-expert sets across the full bank,
and evict device caches with a 128 MiB scrub before each timed kernel. Compare cold
random-route latency with a repeated warm actual route and use the ratio as a
conservative pressure correction in final adjudication.

## G — bridge / contention

Repeat exact-size pinned CUDA/OpenCL bridge measurements and BASE/ARC_LOAD/BASE
QFAST windows throughout the campaign.

## H — adjudication

Use all snapshot/layer medians rather than the best run. Report median, p95,
first/last drift and an all-layer token-0 sum. Promotion thresholds are fixed in
the research plan.
