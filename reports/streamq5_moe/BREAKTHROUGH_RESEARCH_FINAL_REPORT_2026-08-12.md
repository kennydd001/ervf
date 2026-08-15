# Breakthrough research — final synthesis

## Verdict

**The new material produced one important correction, one real compiler
component result and two useful feasibility bounds, but not an industrial LLM
breakthrough yet.** Every locally runnable hypothesis from `pro1.txt` and the
`BREAKTHROUGH_NEXT_PHASE_PACK_2026-08-12` has either been executed, bounded or
explicitly separated as a future training/real-model project.

## 1. GaugePack: premise invalidated, corrected variants fail

The historical P9B “50% pruning quality pass” was a no-op. Expressions such as
`weight[boolean_mask].zero_()` changed an advanced-indexing copy, not the
parameter. P9B and the first group-balanced rerun therefore measured the
ordinary Q5 baseline. An independent checkpoint mutation audit proved unchanged
weight hashes and bitexact expert output; the corrected broadcast mask changed
the weights as intended.

Correct full-depth validation results:

| candidate | relative CE | top-1 agreement | verdict |
|---|---:|---:|---|
| frozen P9B mask, 50% kept | +47.804% | 60.866% | closed |
| 64/128 per original group, 50% kept | +47.186% | 60.866% | closed |
| new 25% pruning, 75% kept | +22.846% | 74.803% | closed |

All intended masks changed hundreds of millions of nonzero elements per layer
and left zero masked nonzeros. Test splits stayed sealed. The GaugePack layout
projects to a 0.502363 byte ratio (about 8.725 GiB for Qwen30), but without a
quality-valid source operator that is not a runnable breakthrough. Codec and
kernel work correctly stopped before a misleading speed claim.

## 2. ERGV: exact compiler prototype and physical autotuning pass

`ExactReductionIR` now represents logical accumulators, ordered add edges,
round/cast points, FMA policy and physical schedules. C0/C1 established:

- 7/7 CPU tests and 20/20 graph-isomorphic Q8/Q5 schedules;
- 2,680/2,680 CPU bitchecks and 6/6 corrupt semantic mutations rejected;
- generated width-16 Q8/Q5 CUDA: 115,496 outputs, zero bit differences.

C2 then searched generated widths on the real resident Q8/Q5 banks. All widths
were exact. Against uniform manual P7, generated selections achieved:

| bank | p50 ratio | p95 ratio | p50 gain |
|---|---:|---:|---:|
| Q8 | 0.8599 | 0.8575 | 16.30% |
| Q5 | 0.9271 | 0.9289 | 7.87% |

Against the later hand-tuned N1C graph, ratios were essentially parity
(Q8 0.9980/1.00035; Q5 1.00007/0.99983). Independent verification passed
63/63 checks over 960 raw events. This is a real compiler/autotuning component
result—not a new end-to-end speedup beyond N1C. A second GPU, second model,
matched public kernels and end-to-end integration remain open.

## 3. TierFlow: traffic target feasible only with large route substitution

On held-out real Qwen30 P4D traces, the optimistic `r=1` route-edit oracle
achieved 4.1577× fewer critical expert bytes and 8× fewer worst-case new loads.
Independent verification passed 20/20 checks. However, it retained only 67.97%
mean route overlap, substituted 32.03% of router outputs, and exactly matched
only 8.99% of top-8 sets. General and multilingual also missed a stricter
per-domain 4× gate.

This proves arithmetic traffic feasibility, not quality. TierFlow is rational
only as a separately budgeted small-model training study with the original
≤1% quality gate; no trained or deployable controller exists yet.

## 4. PORT80B: exact full-size host bank is stable, transfer gate fails

The physical P0 built a non-sparse, uncompressed, Q5-compatible bank of exactly
49,925,652,480 bytes (46.496887 GiB), verified its full SHA256 and 132 sampled
records, and ran uninterrupted for 3,600.093 seconds.

| gate/metric | result |
|---|---:|
| physical bank and full hash | pass |
| 3 × 10,000 primary tokens | pass |
| one-hour endurance | pass |
| peak process commit | 6.075 GB, pass |
| CUDA/thermal/runner errors | none, pass |
| zero-cache H2D p95 | **73.544 ms**, fail vs ≤45 ms |
| post-warmup system Page Reads/s | p50 0, max 7,759, fail vs exact zero |

The bank can therefore be hosted on the current 64-GB machine far more stably
than the analytical risk suggested. But the zero-cache active-set path misses
its transfer target by 1.634×. Because private commit is low and paging is
episodic, 96 GB RAM is allowed for a controlled comparison but is not proven to
solve the bottleneck; mmap→pinned copy and 480 record transfers/dispatches per
token may dominate. Real 80B weights, quality and decode remain untested.

## 5. Industry/novelty boundary

Primary-source research found extensive prior art in pruning, sparse+quantized
kernels, weight repacking and reproducible reductions. The defensible ERGV
boundary is preservation of a chosen source reduction DAG under a changed
physical topology, not “first deterministic reduction.” GaugePack would have
needed the narrow combination of frozen codes, original BF16 scales/group IDs
and logical leaf positions, but its quality premise failed. A search miss is
not novelty proof.

Public 8-GB Qwen3-Coder-Next reports vary widely (roughly 1.75–1.80 tok/s in an
experimental FP8 streamer and 20–21 tok/s in a host-heavy Q4 llama.cpp report)
and are not protocol-equivalent. The synthetic N4B-R projection and P0 host
test must not be compared as real tokens/s.

## What is proved and what comes next

Proved:

1. the old P9B/GaugePack foundation was invalid and corrected 25–50% pruning
   fails quality;
2. a restricted exact-reduction compiler can mechanically generate bitexact
   kernels and recover the best known local projection schedules;
3. TierFlow's traffic arithmetic is possible but demands major route changes;
4. a real 46.5-GiB aligned bank survives one hour on 64 GB RAM, while its
   zero-cache transfer path is too slow under the frozen protocol.

Highest-value next sequence:

1. profile PORT80B P0 into page-resident mmap→pinned copy, batched H2D and
   per-record dispatch; test batching before buying RAM;
2. only if the isolated page-in component remains material, repeat on 96 GB;
3. continue ERGV on a second GPU/model and matched public kernels;
4. open TierFlow training only with a fixed small-model compute budget;
5. begin real Qwen3-Coder-Next conversion only after the transfer path projects
   below the 100-ms full-decode gate.

No current result justifies the claim “industrial LLM breakthrough.” The
strongest new result is the verified ERGV compiler component; the most decisive
engineering result is that PORT80B is transfer-limited rather than simply
impossible to host.

## Main artifacts

- `BREAKTHROUGH_PHASE_REGISTRY_2026-08-12.yaml`
- `GAUGEPACK_P9D1_P9B_MUTATION_AUDIT_2026-08-12.md`
- `P9BR_CORRECTED_STRUCTURED_WANDA_REPORT_2026-08-12.md`
- `P9ER_GROUP_BALANCED_CORRECTED_REPORT_2026-08-12.md`
- `P9F_QUARTER_PRUNING_REPORT_2026-08-12.md`
- `ERGV_C2_PERFORMANCE_AUTOTUNER_REPORT_2026-08-12.md`
- `TIERFLOW_F0_REPORT_2026-08-12.md`
- `PORT80B_P0_PHYSICAL_HOST_BANK_REPORT_2026-08-12.md`
- `INDUSTRY_PRIOR_ART_GAUGEPACK_ERGV_2026-08-12.md`
