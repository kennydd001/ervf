# TierFlow-F0 independent verification

## Verdict

**independent_verification_pass — 20/20 checks passed.**

The verifier independently reloaded all 48 raw P4D safetensors, checked their
locked hashes and route invariants, reimplemented the oracle without importing
the experiment runner, recomputed all validation budgets and the one selected
held-out test, and verified byte/rate arithmetic and gates.

## Independently recomputed held-out result

- selected edit budget: `1`;
- critical-byte reduction: **4.157684x**;
- worst-case new-load reduction: **8.0x**;
- mean route overlap: **67.97%**;
- router-output substitution: **32.03%**;
- traffic gates: **pass**.

This verification is limited to frozen-route traffic feasibility. LM quality,
causal-controller performance, measured latency and a second memory hierarchy
remain untested.

## Artifacts

- verifier: `scripts/streamq5_moe/verify_tierflow_f0_independent.py`
- machine-readable result: `reports/streamq5_moe/tierflow_f0_independent_verification.json`
