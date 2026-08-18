# S100 phase 9 preregistration

## A — route trace and cache oracle

Collect 8192 measured causal tokens from the frozen 40-prompt set. Every context
is limited to <=224 measured tokens after 32 warm-up tokens. Save warm-up routes
as simulation prefix, but exclude them from reported miss rates.

Split whole sessions by parity into train/test; learned frequency/transition
statistics may use train only.

Evaluate per layer:

- production LRU at capacities 32..128 step 2;
- static train-frequency cache;
- Belady offline oracle at the production capacity map;
- train-optimized capacity maps under total slot budgets 1656, 1784, 1912 and
  2035 slots, evaluated on test;
- previous-route Markov prefetch with global staging budgets 4, 8 and 12 expert
  records/token.

Simulation is valid only if production-map LRU miss fraction is within 1.5
percentage points of the measured device-cache miss fraction.

## B — exact capacity-map full A/B

Compose every candidate with QFAST + heldout-green `alpha=0.0003`.
Each candidate is compared in four fresh processes:

BASE_A -> CAND_A -> CAND_B -> BASE_B.

Require token parity, finite outputs, >=765 timing samples, <=1 ms base and
candidate drift and <=7987 MiB VRAM.

## C — real routed-up miss engines

Capture real QFAST normalized hidden states and routed expert up records from
high-miss layers. Reorder each sample with actual misses first, but preserve the
original IDs and `need[]` metadata.

For N={1,2,3} distinct experts compare:

- RTX production-like SM-side `cache_fetch` + batched ERVF up;
- RTX direct mapped-host ERVF using the same production ERVF arithmetic;
- Intel Arc OpenCL NVFP4 row-major routed-up kernel over the same records.

RTX DirectHost must be bit-identical to staged RTX. Arc must have cosine >=.999
and NRMSE <=.02 against the staged RTX output.

Arc economics include measured CUDA D2H of one 2688-float hidden vector plus H2D
of N x 1856 floats. Python/OpenCL event time and host wall time are both kept;
only wall+bridge can promote an Arc latency claim.

## D — decision

- `CACHE_PROMOTE`: an exact capacity map gives >=0.15 ms/token valid fresh gain.
- `DIRECTHOST_PROMOTE`: direct host up is exact and >=10% faster than staged miss
  path on median real samples.
- `ARC_MISS_PROMOTE`: Arc wall+bridge is >=10% faster than staged RTX miss path
  for at least N=1 and N=2, with correctness green.
- `PREFETCH_RESEARCH`: simulation removes >=30% of demand misses with <=8
  prefetch records/token and >=40% prefetch precision.

These may compose, but no composed end-to-end claim is made in phase 9.
