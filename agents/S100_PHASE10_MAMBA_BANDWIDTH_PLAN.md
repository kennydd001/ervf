# S100 phase 10 — dense bandwidth + sparse panel-cache sidecar

Date: 2026-08-18
Parent: `agent/s100-phase9-repair-hardware`

## Measured parent

Phase 9 fully closes the up-projection expert miss/cache route as a primary S100 lever. The current repaired line is about 18.06 ms/token (~55.4 tok/s). Belady cuts the modeled miss fraction roughly in half, but the total current serial miss-fetch budget is only about 1.51 ms/token, so even a perfect cache policy is capped at a few percent wall-clock gain.

The parallel branch `agent/s100-phase6-direct-down@fc6093d` contains separate evidence that must not be conflated with phase 9:

- `thr_0020` is full heldout-fidelity green at 18.368 ms/token in that measurement era;
- a full routed-down-record cache was never benchmarked because every arm failed VRAM admission;
- its 64..320-record locality curve is weak per MiB;
- a `(layer, expert, panel)` sparse code cache was preregistered but has not yet been hardware-measured.

## Phase 10A — exact routed-down panel cache

This remains open because phase 9 cached UP expert records, while this experiment targets the DOWN sparse-code PCIe path after H-SCALE.

Each down panel contains 16 FP4 code columns of 1,344 B each; scale metadata already lives in resident H-SCALE planes. Candidate cache keys are `(layer, expert, panel)` and selection value is measured avoided host-code bytes, weighted by `popcount(panel_mask) * 1344`.

Frozen physical budgets: 8, 16, 24, 32, 40, 48 MiB plus maps. The candidate must preserve routing, top-k, panel masks, scale-plane reads, reduction chunks and accumulation order. Promotion requires exact token parity, sabotage-control sensitivity and fresh-process A/C/C/B with >=0.15 ms/token saving.

Important correction: the parallel panel profile was designed around `alpha=0.002`, while the repaired phase-9 line uses `alpha=0.0003`. These produce different active-column masks. Therefore panel selections must be reprofiled for each threshold parent; selection from `thr_0020` must never be reused silently for `thr_0003`.

Before panel timing, run a same-era fresh comparison of `thr_0003` and `thr_0020` under the repaired runtime. The faster full-fidelity-green parent becomes the primary panel-cache parent. If practical, retain a second small profile for the other threshold to test whether stronger activation pruning creates enough panel locality to compensate for its own quality/performance trade-off.

## Phase 10B — Mamba dense bandwidth

This remains the main expected S100 lever. Mamba accounts for ~892 MB/token across 23 layers and about 43.6% of the modeled resident VRAM traffic.

Current generalized ERVF uses 16 physical lanes per output row. Phase 10B tests exact 32-lane/warp-per-row ERVF mappings, strip-mined virtual accumulators, full-warp coalesced FP8 reads, decode strategy and PTX cache-policy variants. Old microbench results above the measured DRAM roofline are treated as warm-L2 evidence only.

Every microbenchmark must rotate >=4x the assumed 32 MiB L2 footprint using real checkpoint matrices before pointer reuse. A kernel is not promoted from microbench alone: bit equality plus fresh-process integrated A/C/C/B is mandatory.

## Priority

1. Run `thr_0003` vs `thr_0020` same-era parent A/B.
2. Run exact 8..48 MiB down-panel cache profiling/timing; close or retain it on evidence.
3. In parallel, build cold-L2 Mamba FP8 kernel geometry sweep.
4. Integrate the best exact Mamba kernel on the winning threshold parent.
5. Only then move to grouped SM120 MoE if the remaining gap is still material.

## Claim boundary

No component projection counts as S100. S100-single requires <=10.000 ms/useful token end-to-end under the frozen quality gates.
