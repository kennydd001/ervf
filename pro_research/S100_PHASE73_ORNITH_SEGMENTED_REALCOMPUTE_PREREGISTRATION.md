# S100 Phase73 — Ornith segmented rolling prefetch under real compute

Date: 2026-08-21

## Question

Does Phase72's cross-H4 breakthrough survive six real projection transfers and
concurrent execution of the bandwidth-heavy real Ornith routed-expert kernel?

## Frozen parent and inputs

- Phase72 continuous 28x40 rolling schedule and Phase70 LRU/Belady miss counts.
- Real Pottokao layer-20 experts 0..31 and Phase59 exact bulk32 SwiGLU kernel.
- Each expert transfer is split into the exact checkpoint records:
  gate/up/down code planes of 524,288 bytes each and scale planes of 65,536
  bytes each, totaling the unchanged 1,769,472 bytes per expert.
- Six independent pinned rotating sources and six GPU destination segments per
  ring slot; total source working set remains at least 4x L2.

## Frozen compute envelope

Every simulated layer runs the real hot Phase59 bulk32 gate, up, SwiGLU and
down path on disjoint resident weights. A calibrated one-SM wait fills only the
remaining time to Phase72's 1.463017 ms per-layer envelope. The final head
remains a calibrated 1.574816 ms wait. Thus real routed-weight VRAM traffic
contends with H2D writes while the total no-copy envelope remains tied to the
60.095488 ms/H4 component floor.

The real hot output before and after rolling copies must be bit-identical and
finite. Copy destinations are deliberately disjoint from the hot control
weights; this experiment measures contention and scheduling, not cache-slot
addressing correctness already covered by Phases60/69.

## Arms and measurement

- LRU-52 and Belady-52.
- Continuous rolling leads 2, 4 and 8.
- One warmup and five measured full-trace epochs per arm.
- Report exposed tail relative to the same-session real-compute baseline and
  floor-normalized latency `60.095488 + exposed`.

## Gates

1. Exact six-segment byte sum, >=4x-L2 rotating source, and no ring overflow.
2. Hot output is bit-identical before/after overlap, repeat-exact and finite.
3. Bulk32 hot latency is within 20% of Phase59's 0.560960 ms reference, and the
   same-session no-copy envelope is within 5% of 60.095488 ms/H4.
4. At least one Belady arm remains <=61.538462 ms/H4 floor-normalized.
5. At least one LRU arm remains <=61.538462 ms/H4 floor-normalized.

A green result authorizes integration into the custom executor. It still does
not include all real attention/support kernels, route prediction, segmented
cache-slot scatter, draft rejection or HTTP/runtime overhead.
