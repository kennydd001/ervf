# Co-route physical ordering — independent verification

## Verdict

**independent_verification_pass — 17/17 checks passed.**

The verifier independently reloaded all 48 route tensors, rebuilt every
learned order from learn-only rows, and brute-force enumerated all 128
contiguous partitions of every evaluated top-8 route.

Verified `validation` metrics:

- mean intervals: `4.650456`;
- p95 intervals: `7.000`;
- payload inflation: `1.066530x`;
- trace gate: `fail`.

No GPU or physical-bank work is authorized by this verifier.

## Artifacts

- verifier: `scripts/streamq5_moe/verify_co_route_physical_ordering_trace.py`
- machine-readable verification: `reports/streamq5_moe/co_route_physical_ordering_trace_independent_verification.json`
