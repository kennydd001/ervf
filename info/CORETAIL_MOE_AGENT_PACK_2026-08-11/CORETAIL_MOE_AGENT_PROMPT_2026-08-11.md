# Agent prompt — CORETAIL-MoE

Open a new independent registry named `CORETAIL_MOE`.

## Immutable closures

- CRAFT remains `closed_no_eureka`.
- FLEQ GSQ remains falsified.
- HERA static `count>=128` multidomain expert-ID tier remains
  `static_tier_negative`.
- Do not change HERA thresholds, gates or reports.
- Do not describe CORETAIL as a rescued HERA result.

## Mechanistic hypothesis

The HERA union contains 6,081/6,144 layer-expert pairs. Expert identity is
therefore the wrong primary tiering axis.

For every existing GPTQ code q in {-2,-1,0,+1}, use the exact identity:

```text
t = max(q, -1)
e = 1[q == -2]
q = t - e
```

Keep a universal ternary core for every expert resident. Store or stream
only the exact sparse extreme tail for routed experts. Natural routing and
all GPTQ codes remain unchanged.

Read first:

- `CORETAIL_MOE_EUREKA_HYPOTHESIS_2026-08-11.md`
- `CORETAIL_MOE_CALCULATIONS_2026-08-11.json`
- final HERA P0 audit, verification, preregistration and addenda;
- E2GQ audit and FLEQ closure;
- QMoE, SliceMoE, SpQR/SqueezeLLM, MoEpic and dtANS sparse-MVM sources.

## P0 — full-bank format census

No model-quality run before P0 closes.

For all 48×128 experts and gate/up/down matrices:

1. Reconstruct the exact GPTQ codes and BF16 group-128 scales.
2. Build an actual row-random-access core:
   - one fixed nonzero bitmap;
   - a sign stream for nonzeros;
   - row/block offsets;
   - raw or losslessly compressed BF16 scales.
3. Build an actual extreme tail:
   - extreme flag only among negative core entries;
   - chunked rANS/dtANS or enumerative coding;
   - row/block offsets for selected-expert access.
4. Count every byte:
   - headers;
   - tables;
   - alignment;
   - offsets;
   - index;
   - checksums;
   - fallback blocks.

Primary P0 gates:

- actual resident core <= 5.95 GiB;
- actual complete tail <= 0.90 GiB;
- bit-exact recovery for every code and every BF16 scale;
- `core + INT4 trunk + BF16 4K KV + 0.75 GiB mandatory runtime reserve`
  <= reported physical VRAM;
- no matrix or expert can silently use a fixed-width fallback without its
  bytes counting.

If P0 fails, close CORETAIL.

## P1 — exact fused kernel

Implement and compare:

- BF16 reference;
- true fixed-width uint2 GPTQ;
- E2GQ entropy-GPTQ reference;
- CORETAIL exact.

Test gate/up/down separately on layers 0, 24 and 47.

The kernel must not materialize full dequantized matrices.

Measure:

- decoded weights/s;
- effective compressed bytes/s;
- output error;
- p50/p95/p99 latency;
- GPU peak memory;
- host-to-device tail traffic;
- thermal clocks.

P1 gates:

- exact same GPTQ code and scale semantics;
- no additional quality error beyond the preregistered accumulation tolerance;
- >=27.2 billion routed weight applications/s for a 1.5x safety margin over
  the 10-token/s target;
- actual selected-tail traffic and decode fit the full-token 100-ms budget.

## P2 — isolate model quality

Run full-depth teacher-forced and held-out evaluation for:

1. BF16 teacher;
2. GPTQ experts + BF16 trunk;
3. BF16 experts + INT4 trunk;
4. GPTQ experts + INT4 trunk;
5. GPTQ experts + INT8 trunk.

Freeze all datasets and thresholds before opening test.

Decision:

- relative CE <=2%: continue;
- >2% and <=10%: authorize exactly one repair;
- >10%: close the quality line.

## P3 — one repair only

Pre-register exactly one rank-8 INT4 model-wise discrepancy correction.
No rank sweep, no objective sweep and no test-driven bit allocation.

All correction bytes and metadata count toward the memory budget.

## P4 — physical residency variants

Test both without choosing from test quality:

### A. ALLCORE_Q4TRUNK

- all expert cores resident;
- INT4 trunk resident;
- BF16 or INT8 KV;
- tail in pinned host RAM and copied only for selected experts.

### B. GUCORE_FULLTAIL_Q8TRUNK_STREAMDOWN

- gate/up cores resident;
- full tail resident;
- INT8 trunk resident;
- selected down-core streamed while gate/up executes.

Selection is based only on physical memory, bandwidth and latency measured
before test-quality outputs are opened.

## P5 — end-to-end gates

- peak VRAM <=8.0 GiB;
- process RAM <=32 GiB;
- relative CE degradation <=2%;
- >=512-token independent rollouts;
- batch-1 decode >=10 tokens/s at 1K and 4K context;
- p95 token latency reported;
- no broad claim before a second MoE family.

## Integrity

- Every phase gets a timestamped preregistration.
- Keep every failed attempt append-only.
- The HERA top-k tie lesson is mandatory: intercept the exact official
  `topk` result; never recompute tied BF16 routes as the authoritative path.
- Entropy coding, sparse residuals, bit-sliced caching and expert splitting
  are prior art. Do not claim them broadly.
