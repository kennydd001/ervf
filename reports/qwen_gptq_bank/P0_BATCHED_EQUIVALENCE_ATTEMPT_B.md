# Batched GPTQ equivalence — attempt B lock

Locked after attempt A failed (368 code mismatches and 33 BF16 scale-bit mismatches over 30 matrices) and before opening attempt-B outputs.

Attempt B removes both identified numerical batching dimensions:

- gate and up are quantized as separate projections;
- the pinned per-channel MSE `find_params` and hard quantize calls execute separately for every expert, preserving the official tensor shapes and reduction kernels;
- Hessian construction, Cholesky, sequential column error propagation, and block updates already execute expert-by-expert with the pinned 2-D operations.

The same locked 10 experts and 30 matrices are reused. The pass rule remains zero integer-code mismatches and zero BF16 scale-bit mismatches. Failure forbids this producer and triggers the official per-matrix fallback.
