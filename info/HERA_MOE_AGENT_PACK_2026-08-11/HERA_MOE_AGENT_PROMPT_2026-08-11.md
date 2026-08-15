# Agent prompt — HERA-MoE

Open a new independent registry named `HERA_MOE`.

Do not modify the closed FLEQ, E2GQ, EFCQ, CRAFT or RSIV registries.

## Mechanistic hypothesis

`HERA = Hot Entropy-Resident, Rare-Exact`.

Do not force the entire expert bank below 2 bpp. Instead:

- experts with >=128 natural routed calibration rows form a hot tier;
- quantize the hot tier with exact 2-bit GPTQ and lossless entropy coding;
- keep every undercovered expert in BF16 in pinned/mapped host RAM;
- keep non-expert weights INT4 in VRAM;
- asynchronously prefetch exact cold experts after router logits are known.

From the locked P0 counts:

- hot experts: 4449;
- cold experts: 1695;
- cold parameter fraction: 0.275878906250;
- cold invocation fraction: 0.003811359406;
- projected hot bank: 4.718477 GiB;
- projected non-expert INT4: 0.717628 GiB;
- projected resident weights: 5.436104 GiB;
- cold BF16 bank: 14.897461 GiB;
- projected total weights across tiers: 20.333565 GiB.

These are projections, not runtime measurements.

## P0 — preregister and audit the tier

Independently reproduce all counts from the original 48 layer artifacts.

Before selecting the hot set, lock calibration corpora from:

- general text;
- code;
- math/reasoning;
- multilingual;
- instruction/chat.

For every expert collect:

- invocation count;
- sum of original router probability;
- sum of squared router probability;
- top-k margin statistics;
- per-token activation indicator.

Report:

- mean, p95 and p99 cold expert calls per token;
- maximum layer cold-call fraction;
- Jaccard/union growth of hot sets across domains;
- size of the frozen hot union.

Do not use test data to set the hot set or threshold.

P0 memory gate:

- projected hot entropy pack + non-expert INT4 <=5.75 GiB;
- projected exact cold BF16 bank + all mapped backing weights <=24 GiB host RAM;
- no uncounted sidecar.

If the multidomain hot union exceeds the VRAM gate, close the static-tier
hypothesis before model-quality tuning.

## P1 — full-model quality without a custom runtime

Generate natural-routed GPTQ only for the frozen hot tier. Keep the cold tier
exact BF16.

Compare:

1. full BF16 teacher;
2. hot true fixed-width 2-bit GPTQ + cold BF16;
3. hot entropy-decoded exact GPTQ + cold BF16.

The two GPTQ paths must reconstruct the same code/scales. Entropy coding may
not add quality loss.

Evaluate full depth on locked validation/test corpora and independent
512-token rollouts.

Primary gate:

- relative CE delta <=2%;
- no domain/task hard failure;
- stable router behavior and no rollout collapse.

If relative CE is >2% and <=10%, exactly one predeclared repair is permitted:

- rank-8 INT4 correction over hot matrices;
- model-wise discrepancy objective;
- all repair bytes counted;
- no rank or objective sweep after test.

If CE >10%, close HERA quality.

## P2 — actual entropy pack

Build a random-access hot-tier file.

Count:

- code stream;
- scales;
- coder tables;
- row/chunk offsets;
- expert index;
- padding/alignment.

Gates:

- actual hot pack <=4.95 GiB;
- code and scale bit identity;
- deterministic decode;
- no full BF16 materialization.

## P3 — cold tier

Store cold BF16 experts in pinned or memory-mapped host RAM.

Preregister exactly three cold-cache capacities:

- 0 GiB;
- 0.5 GiB;
- 1.0 GiB.

Prefetch after router logits and overlap with hot-expert compute.

Measure:

- H2D bytes;
- cold expert calls/token p50/p95/p99;
- transfer latency;
- cache hit rate;
- latency spikes under every domain.

## P4 — fused runtime

Mandatory baseline: true uint2 hot pack + exact cold BF16 tier.

Measure:

- peak VRAM;
- process RSS;
- file-backed page cache separately;
- p50/p95/p99 token latency;
- tokens/s at 1K and 4K context;
- energy if available.

Final gates:

- <=8 GiB VRAM;
- <=32 GiB process RAM;
- >=10 batch-1 decode tok/s;
- <=2% relative CE delta;
- stable 512-token rollouts.

## P5 — second family

No broad claim before replication on another MoE family.

## Research integrity and novelty

- Mixed precision, expert offloading, caching and entropy coding are prior art.
- Do not claim those mechanisms as new.
- The potential contribution is the measured modern-Qwen intersection of
  natural calibration coverage, exact entropy rate, heterogeneous residency,
  full-model quality and real laptop throughput.
- Preserve all failures append-only.
