
# S100 phase 8 decision — static hot routed-down records

Date: 2026-08-17

## Frozen parent

The phase-7 quality/performance parent is:

- QFAST attention-Q NVFP4 CEIL;
- global routed-down relative threshold alpha 0.0020;
- 18.3680 ms/token;
- 54.4425 tok/s;
- 5,120 heldout target tokens;
- top1 agreement 0.979883;
- target recall in top5 1.0;
- mean CE delta 0.010934;
- mean coarse KL 0.008712.

S100-single remains false.

## Closed phase-7 backend

The exact packed sparse mirror was bit-identical and stable, but improved full
latency by only 0.01655 ms. It is below the frozen 0.15 ms adoption gate and is
closed.

## First-principles phase-8 hypothesis

The current routed-down path still copies every selected code column over PCIe
for every routed expert, even when the same `(layer, expert)` is repeatedly
selected.

A dynamic full-record cache is unattractive: a miss would copy a complete
2,806,272-byte down record, while the current sparse miss usually copies only
the active columns.

Phase 8 instead uses a static, calibration-frozen cache:

1. profile teacher-forced route IDs on `_01` calibration prompts;
2. rank all `(layer, expert)` pairs by calibration frequency;
3. freeze the top 64, 128, 192, 256 and 320 pairs;
4. preload those complete down records before graph capture;
5. on a static hit, skip sparse PCIe gather and read the record from VRAM;
6. on a miss, execute the unchanged sparse gather and masked GEMV.

No full-record transfer occurs in the timed decode loop. Static-cache misses are
therefore no worse than the legacy path, apart from one expert-map lookup.

## Why the cache is exact

The hybrid down kernel preserves:

- selected expert IDs;
- panel list and panel masks;
- panel-to-reduction-chunk assignment;
- low-to-high mask-bit iteration;
- resident scale plane;
- E2M1 lookup;
- FMA order;
- chunk reduction;
- route-weight accumulation.

Only the code-byte source changes from sparse mirror to preloaded full record on
a cache hit. Smoke token parity, deterministic repeat and a destructive route
control are required.

## Expected ceiling

The cache attacks only routed-down sparse code traffic. It cannot remove routed
up-projection, routing, shared expert, Mamba, attention or lm_head time.

A substantial exact win would validate locality and justify more elaborate
column/subrecord caching. A null result closes static full-record residency and
moves the primary path to grouped MoE and model-structure recovery.
