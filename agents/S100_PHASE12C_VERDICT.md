# S100 phase 12C verdict — ERVF-M + grouped MoE microkernels

Date: 2026-08-18 · Branch: agent/s100-phase12c-hardware (commit 1bd38af) ·
Results: `pro_research/results/s100_phase12c/` (DENSE / GROUPED_MOE /
ECONOMICS / SUMMARY JSON).

## Outcome

`INTEGRATED_VERIFIER_BUILD_OPEN = False`. Instrumentation complete, zero
technical failures, both streams bit-exact with cold rotation >=4x L2. The
gates failed on measured performance, not on correctness.

One pack bug was found and fixed from the real error JSON: the grouped bench
passed a missing `globals_dev` key to `down_masked_sres`; the kernel indexes
`globals[id*2+0]`, so each standalone record now carries a 2-float device
array `[g_down, g_up]` (eid=0). That was a technical fix, not a result change.

## Dense ERVF-M (140 real matrices, 1.56 GB rotation = 46.6x L2)

| B | exact | baseline M=1 | ERVF-M | useful-row speedup | gate |
|---|---|---:|---:|---:|---:|
| 2 | yes | 15.53 ms | 13.38 ms | 1.161x | 1.75x |
| 4 | yes | 28.70 ms | 22.59 ms | 1.271x | 3.20x |
| 8 | yes | 57.01 ms | 39.63 ms | 1.439x | 5.50x |

The candidate shares weight loads across rows yet scales far below linear:
the per-byte efficiency of the multi-row kernel is much lower than the
existing GEMV, so row scaling saturates near ~1.4x. The required margin
(3.2x at B=4) is not within tuning distance of a kernel that gets 1.27x.

## Grouped MoE (48 real (layer, expert) records from the phase-9 trace)

Exact at every M in {1,2,3,4,6,8}; up and down rotations 4.01x L2 each.

- M=1 candidate penalty: **26.0%** (gate: <=15%)
- Weighted B=4 up+down speedup under the real phase-12B row histogram:
  **0.942x** — i.e. 5.8% slower than the current M=1 path (gate: >=1.20x)
- Weighted B=8: 0.989x

The phase-12B caveat decided the outcome: at 1.43 rows/expert (B=4) the
grouping overhead exceeds the reuse win. Grouped MoE only pays when blocks
produce genuinely deep expert groups, which this model's routing does not.

## Economics projection (explicitly a non-claim, per preregistration)

Substituting the measured microkernel savings into the phase-12A B=4 floor
(70.99 ms): projected cycle 60.94 ms -> 65.6 perfect-draft tok/s, vs the
28 ms gate. Projection gate: fail.

## Consequence per the preregistered decision rules

- The layer-major integrated verifier build does **not** open.
- The breakthrough document's kill criterion is met in substance: after a
  true grouped/weight-shared measurement the B=4 verifier remains far above
  35 ms. **Block-ERVF on this 30B parent closes as the primary S100 route.**
- Remaining honest routes: (a) a fundamentally better multi-row kernel
  generation than this ERVF-M design (nothing in these numbers supports
  spending that effort next), or (b) the already-recorded parallel route:
  NVIDIA's quality-trained Elastic 12B-A2 derivative under the same ERVF
  runtime — materially fewer bytes per token, then block verification on
  top. That is a different model and must be claimed as such.

## What phase 12 established, positively

- Block verification is semantically exact on this hybrid (12A).
- Real route reuse exists (12B) but is shallow, and shallow reuse does not
  pay for grouping machinery in CUDA (12C grouped).
- Weight-shared multi-row dense kernels are bit-exact buildable, but this
  kernel generation does not convert reuse into bandwidth (12C dense).
