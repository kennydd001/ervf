# S100 Phase 13E — expert shared-basis census

Date: 2026-08-18

This discovery screen uses the exact Phase-12C checkpoint and all 128 routed
experts at layers 1, 27, and 51 for both `up_proj` and `down_proj`. It fits
low-rank shared expert structure on a deterministic 4,096-feature sample of
the FP8 code plane. Per-expert weight-scale and secondary-scale bytes remain
in the byte budget and are not factorized.

The census is evidence about expert-axis redundancy only. It does not decode
FP8 values, measure activation/output reconstruction, include sparse residual
indices, benchmark a shared-basis kernel, or run official quality validation.
The independent verifier must be green, but the promotion flag remains false.

Across all six matrices, rank 4 leaves about 0.474–0.478 reconstruction NRMSE
while its idealized BF16-basis byte reduction is 83.3%. Rank 32 improves the
NRMSE only to about 0.402–0.405 and still has an idealized 44.4% reduction;
rank 64 reaches about 0.315–0.318 NRMSE but removes essentially 0% after
BF16 basis/coefficient storage and untouched scales. This is insufficient
evidence for a shared expert representation without decoded-output and
validation tests.
