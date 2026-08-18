# S100 Phase 13B — activation-subspace census

Date: 2026-08-18

## Verdict

**Discovery screen measured; SR-ERVF is not promoted.**

The exact Phase‑12C checkpoint was run autoregressively on 10 `_01`
calibration prompts and 10 `_02` validation prompts, 64 generated tokens per
prompt. Hooks captured Mamba input/output, attention input, routed-MoE input
and final-norm activations. An origin SVD was fit from at most 1,024 pooled
calibration rows per family and evaluated on disjoint validation rows.

The signal is mixed rather than a clean breakthrough. At rank 128, for
example, the pooled Mamba-out family has about 92.1% projected dense-byte
reduction, but its validation residual energy is about 84.5%; with the
50%-fallback gate the projected expected byte reduction is about 44.0%.
Those are byte/residual projections only, not model output fidelity.

The required `W U Uᵀx` output NRMSE, top-1/top-5 effects, official validation
quality and end-to-end timing were not measured. Therefore the result is not a
promotion or a no-go theorem; the next implementation step would require an
isolated output-reconstruction test before any runtime change.
