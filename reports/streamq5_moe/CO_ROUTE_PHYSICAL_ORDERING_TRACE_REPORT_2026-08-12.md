# Co-route-aware physical expert ordering — trace report

## Verdict

**FAIL on validation.** Test remained closed because the validation trace gate failed.

The learned physical ordering changes no route or model value. On the evaluated
partition its exact one-extra-record interval cover measured:

| metric | learned order | identity order |
|---|---:|---:|
| mean intervals / token / layer | 4.650456 | 7.231201 |
| p95 intervals | 7.000 | 8.000 |
| p99 intervals | 8.000 | 8.000 |
| payload inflation | 1.066530x | 1.043650x |
| coverage errors | 0 | 0 |

## Frozen gates

| gate | result |
|---|---|
| aggregate_p95_intervals_at_most_2 | fail |
| aggregate_mean_intervals_at_most_1_5 | fail |
| every_domain_p95_intervals_at_most_3 | fail |
| aggregate_payload_inflation_at_most_1_10 | pass |
| exact_coverage | pass |
| valid_permutations | pass |
| learn_only_provenance | pass |
| trace_gate_pass | fail |

## Split and claim boundary

Learn `[0,512)`, validation `[512,768)` and test `[768,1024)` are strictly
disjoint. The latter windows were previously used by TierFlow-F0, so this is
not a fresh-dataset confirmation. This result contains no GPU execution,
physical relayout, transfer, latency, model-quality, or 80B evidence.

## Artifacts

- preregistration: `reports/streamq5_moe/CO_ROUTE_PHYSICAL_ORDERING_TRACE_PREREGISTRATION_2026-08-12.md`
- runner: `scripts/streamq5_moe/run_co_route_physical_ordering_trace.py`
- raw result: `reports/streamq5_moe/co_route_physical_ordering_trace_validation.json`
