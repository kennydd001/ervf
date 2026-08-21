# S100 Phase71 — Ornith real-trace copy-engine prefetch oracle

Date: 2026-08-21

## Question

Can perfect layer-ahead knowledge hide the real Phase70 expert-miss bytes under
the already measured 60.095 ms/H4 all-hot compute envelope, leaving no more
than the 1.443 ms/H4 residual required for 65 tok/s?

## Frozen inputs

- Phase70 fixed 128-token route trace, aligned into 28 warm H4 blocks after 16
  warmup tokens.
- Both 52-slot policies are replayed unchanged: implementable LRU and
  future-aware Belady as an optimistic lower bound.
- One complete Ornith NVFP4 expert is frozen at 1,769,472 bytes, measured from
  the real gate/up/down codes and scales in Phase62.
- Per layer/H4 copy size is that byte count times the exact Phase70 unique miss
  group count. Zero-miss layers issue no copy.
- Source memory is pinned and rotated over a working set at least four times
  the measured GPU L2. Destination buffers are real GPU allocations.

## Frozen schedules

Each H4 block contains 40 layer dependencies. A dedicated non-blocking stream
performs H2D copies; the main stream waits only at the consuming layer.
Destination buffers are recycled only after a main-stream consumption event.

- `serial`: copy immediately before every layer's compute envelope;
- `lead1`, `lead2`, `lead4`, `lead8`: ring-buffered perfect prefetch with that
  many layer slots of lookahead;
- `compute_only`: no transfer, preserving the same layer/head envelope.

The measured component floor is represented by 40 calibrated CUDA wait
kernels totaling `60.095487602 - 1.574815989` ms plus a final calibrated
1.574815989 ms head wait. A wait kernel occupies one SM block and deliberately
does not contend for VRAM bandwidth. This is an optimistic copy-engine ceiling,
not an end-to-end implementation or proof that real kernels overlap equally.

## Measurement

- One timing sample is a complete epoch over all 28 warm H4 blocks; report
  milliseconds per H4.
- 1 warmup epoch and 5 measured epochs per arm.
- GPU events enclose the main stream; the final event is downstream of every
  layer copy dependency.
- Report serial transfer increment, overlap exposed tail, speedup and effective
  tok/s for both policies.

## Gates

1. Pinned source working set is at least 4x L2 and every requested copy fits a
   destination ring slot.
2. The calibrated layer and head waits are each within 5% of their targets.
3. Every lookahead arm is no slower than serial by more than 2%.
4. At least one LRU lookahead arm has an exposed p50 tail no greater than
   1.443 ms/H4.
5. The same threshold must also pass for Belady; if Belady fails, prefetch-only
   research is closed structurally.

A green oracle only authorizes integration with real kernels and segmented
expert copies. It does not authorize a 65 tok/s end-to-end claim. A failed
threshold is `measured_fail`; compile/runtime errors are `technical_failure`.
