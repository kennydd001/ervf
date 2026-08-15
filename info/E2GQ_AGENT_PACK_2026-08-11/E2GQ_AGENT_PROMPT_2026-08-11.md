# Agent prompt — E2GQ-MoE

You are continuing a rigorously preregistered model-compression project.

## Immutable history

- `CRAFT_MOE`, `RSIV_MOE`, and the GSQ-specific `FLEQ_MOE_P1_GSQ_PTQ`
  hypothesis remain closed.
- Do not reinterpret the GSQ failure.
- Do not alter or overwrite any existing artifact.
- Create a new independent registry named `E2GQ_MOE`.

## New positive precondition

An independent audit of the 16 locked P1 GPTQ experts found:

- GPTQ codebook: `{-2,-1,0,+1}` with group-128 BF16 scales.
- Aggregate code entropy: `1.7828648913739187 bpp`.
- Raw scale cost: `0.125 bpp`.
- Ideal exact total: `1.9078648913739187 bpp`.
- All 16 experts and all 48 matrices are individually below 2 bpp.
- A conservative projected layout remains below 1.945 bpp on every expert.

This does not improve GPTQ quality. It changes only the exact representation.

Read first:

- `E2GQ_EUREKA_REPORT_2026-08-11.md`
- `E2GQ_ENTROPY_AUDIT_2026-08-11.json`
- `audit_e2gq_entropy.py`
- the original FLEQ reports and preregistrations.

## P0 — full-bank entropy census

Before building a runtime:

1. Stream all 48×128 routed experts.
2. Produce exact GPTQ code histograms at matrix, row and group level.
3. Audit BF16 scale bit patterns:
   - raw entropy;
   - XOR/delta entropy by row and adjacent group;
   - exact lossless compressibility.
4. Compute actual lower bounds and upper bounds including:
   - coder tables;
   - row/chunk offsets;
   - alignment;
   - expert index;
   - scale storage.
5. Lock validation/test prompts before any model-quality evaluation.

Primary P0 gate:

- at least 99% of routed parameters can be represented in an actual planned
  format at `<=1.98 bpp` including every metadata byte;
- no matrix may silently fall back to fixed 2-bit without its bytes counting.

## P1 — bit-exact pack

Implement two formats, without model-quality tuning:

### A. Reference rANS/dtANS pack

- Static or per-matrix four-symbol model.
- Chunked parallel random access suitable for fused MVM.
- Codes and BF16 scales decode bit-exactly.

### B. Zero/Sign/Extreme (ZSE) pack

For each code q:

- zero/nonzero mask;
- one sign bit only for nonzeros;
- an entropy-coded extreme flag only among negative values;
- raw or losslessly compressed BF16 scales.

Also test the exact decomposition:

```text
t = max(q, -1)
e = 1[q == -2]
q = t - e
```

Primary P1 gates:

- actual file size `<=1.98 bpp`, not Shannon-only accounting;
- 100% code and scale bit identity;
- deterministic decode;
- row/chunk random access;
- no uncounted sidecar.

Hard stop if actual rate is above 2.0 bpp.

## P2 — three-layer semantic equivalence

Use layers 0, 24 and 47.

Compare:

1. BF16 teacher;
2. fixed-width GPTQ;
3. entropy-packed GPTQ decoded through the reference path.

The fixed and entropy-packed GPTQ paths must reproduce the same quantized
weights. Report hidden error, router overlap, final KL/CE and exact decoder
differences. Entropy coding must add no model-quality loss beyond numerical
kernel accumulation.

## P3 — full-model GPTQ

Run full-depth teacher-forced evaluation, standard tasks and independent
512-token rollouts.

Quality decision:

- relative CE delta `<=2%`: proceed directly to runtime;
- `>2% and <=10%`: one repair experiment is authorized;
- `>10%`: close the quality hypothesis.

## P4 — one entropy-funded repair only

Pre-register exactly one repair before opening held-out data:

- rank-8 INT4 correction per 768×2048 matrix;
- model-wise activation discrepancy loss;
- all base codes remain exact GPTQ;
- actual base + correction + all metadata must remain `<=2.0 bpp`.

Do not test rank 16, alternative ranks, multiple objectives or post-hoc
bit allocations after viewing test. The purpose is to test whether the
measured `0.092135109 bpp` reserve can buy back full-model quality.

## P5 — direct decoder and runtime

Mandatory baselines:

- BF16;
- true fixed uint2 GPTQ;
- entropy-packed exact GPTQ;
- ternary-core + extreme-tail exact GPTQ.

Implement fused decode+MVM. Do not materialize full BF16 weights.

Measure:

- actual VRAM and host RAM;
- actual bytes transferred;
- decode throughput in weights/s;
- layer and full-model p50/p95/p99 latency;
- batch-1 decode tokens/s;
- energy if available;
- 1K and 4K contexts.

Final gates:

- routed effective rate `<=2.0 bpp`;
- relative CE loss `<=2%`;
- stable 512-token rollouts;
- peak VRAM `<=8.0 GiB`;
- process RAM `<=32 GiB`;
- batch-1 decode `>=10 tok/s`.

## P6 — second family

A broad claim requires a second MoE family. No exception.

## Research integrity

- Every phase gets a preregistration before opening held-out outputs.
- All failed attempts remain append-only.
- Report theoretical entropy, projected rate and actual file rate separately.
- Entropy coding is prior art; do not claim otherwise.
- The potentially interesting contribution is the measured modern-Qwen
  rate/quality/runtime intersection and an entropy-funded exact correction,
  not generic compression.
