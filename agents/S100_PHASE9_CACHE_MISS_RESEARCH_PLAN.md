# S100 phase 9 — cache oracle + heterogeneous expert-miss racing

Date: 2026-08-18

## Frozen evidence entering phase 9

- Phase-8 Arc routed-down engine: NO-GO. The Arc implementation was numerically excellent and thermally stable, but raw all-layer routed-down was about 3x slower than the current RTX downflow before bridge cost.
- Arc contention with QFAST is modest (~1.6% median), so the iGPU can still be useful for sparse/occasional work.
- The repaired 8192-token route census measured only ~0.600 up-cache misses per MoE layer/token: ~90.0% of routed slots already hit the RTX cache.
- Phase-6 `thr_0003` is heldout-green and saved ~0.474 ms/token in a fresh A/B.

## New premise

Do not move an entire MoE sublayer to the Arc. Keep the common path on RTX and attack only rare cache misses. For every miss there are three physically different choices:

1. stage the ~2.806 MB NVFP4 up record across PCIe, then run RTX ERVF;
2. let the RTX ERVF kernel stream the mapped pinned host record directly;
3. let Arc compute the missed routed-up projection in UMA and return only the ~7.4 KB ReLU2 activation.

At the same time, use the long route trace to establish an oracle upper bound on cache replacement, capacity reallocation and previous-token predictive prefetch.

## Claim boundary

All cache/capacity changes are exact placement changes and require token parity. Arc/DirectHost microbenchmarks are component evidence only. S100-single remains <=10.000 ms/useful token with frozen model-quality gates green.
