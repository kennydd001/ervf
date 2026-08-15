# HET-NEXT L0 PH0X-R2 — exploratory real-projection repair preregistration

Date: 2026-08-13. This is an exploratory, one-attempt repair revision. It is not a formal PH0 pass.

## Immutable inputs and prior evidence

- Same official Qwen3-Coder-Next layer-0 expert-50 gate projection, same official D2 p0/token-15 post-norm input, same Q5 record, and same width-8 reduction DAG as PH0X.
- Prior PH0X result SHA-256: `bf10932ad5e67bcb356e49184f57261e1d3453b099b48bba502988eb5743c3c0`.
- CPU/OpenCL diagnostic SHA-256: `18b0540a55e2c02fa82db82724994bd3139875884dc3e9ed8c755fa0ee487b54`.
- The diagnostic proves the sole Intel compiler error was ambiguous `atomic_inc` overload resolution. R2 changes only its pointer type to explicit `volatile __global unsigned int*`; launch, arithmetic, record, input, and thresholds are unchanged.

## Frozen control erratum

The original PH0-R3 full-array SHA fixtures for the synthetic q8-to-q7 witness were transcription errors. Independent pre-run recomputation retained the already frozen scalar words and produced:

- activation SHA-256 `2498a04e393ec5eb0ec88b7f098523dd5f3a1cbaf9803fa7ace4b4776c17f561`;
- original output SHA-256 `98fac647d0adc50536d5b397b1974ac237ec14a818ac4ec287760dbab312400b`;
- mutated output SHA-256 `3571cfc8dbc22de68d5b216fa5766b3bc0036e745062dda3d645fc1b1c019910`;
- exactly one changed word, row 0: `0x3894` to `0x3882`.

R2 requires all eight rejection controls plus this ninth synthetic sensitivity witness to pass.

## Execution and outcome

One clean sequential attempt only: CPU source/codec/oracle, Intel host-USM compile+kernel, then NVIDIA CUDA compile+kernel only if Intel completes. No retry or retuning. Create-new output directory. Exact 512 BF16 words and uint32 row counters are retained for both devices.

A positive exploratory outcome requires both devices bitwise equal to the independent CPU output, all 512 counters exactly one, output sentinels overwritten, all nine controls positive, cleanup rows positive, and distinct PCI identities.

Claim boundary: one real projection and one known natural activation only. No full expert, MoE, layer, model, quality-generalization, cohabitation, concurrency, timing, performance, deployment, novelty, or breakthrough claim.
