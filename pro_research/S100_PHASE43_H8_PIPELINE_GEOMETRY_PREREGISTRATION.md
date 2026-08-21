# S100 Phase43 — full-pipeline batch geometry

Frozen before Phase43 GPU timing.

Phase42 B3 was exact and spill-free, improving its paired baseline midpoint by
2.57% (125.610 to 122.383 ms/H8), just below the frozen 3% gate. B3 launches
eight routed-UP kernels and one scan kernel three times per MoE layer.

Phase43 changes only the number of fixed global group ranges:

- `GLOBAL_B2`: `(0,24)`, `(24,48)` — 16 UP plus two scan launches/layer;
- `GLOBAL_B4`: `(0,12)`, `(12,24)`, `(24,36)`, `(36,48)` — 32 UP plus four
  scan launches/layer, but finer transfer/compute overlap.

All explicit-g0 kernels, streams, `gather_y=4`, route/chunk arithmetic,
reduction, cache policy and resident planes are otherwise identical to Phase42.

Protocol: fresh `BASE_A`, B2, B4, `BASE_B`; canonical context 1024; four
warmup plus sixteen measured H8 windows. Every arm must be token-exact,
baseline drift must be <=5%, and all kernels retain zero local-memory bytes.
The fastest geometry opens thermal promotion only if it improves the baseline
midpoint by >=3%. <=120 ms/H8 is reported as a strong milestone.

