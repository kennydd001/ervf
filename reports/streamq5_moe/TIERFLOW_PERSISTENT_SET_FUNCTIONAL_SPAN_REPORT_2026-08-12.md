# TierFlow persistent-set functional-span oracle

## Verdict

**validation_negative_test_closed**. Test remains closed.

| sentinel layer | mean routed rel-L2 | p95 rel-L2 | mean downstream KL | relative CE | top-1 agreement |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.512516 | 0.865522 | 0.044850 | 1.642% | 92.784% |
| 24 | 0.407422 | 0.913767 | 0.005821 | -0.194% | 97.333% |
| 47 | 0.220883 | 0.590669 | 0.043767 | 3.283% | 93.725% |

Traffic replication: `4.148102x`
critical bytes and `8.0x`
worst-case new loads.

## Frozen gates

| gate | result |
|---|---|
| traffic_reduction_at_least_4x | pass |
| worst_case_new_load_reduction_at_least_8x | pass |
| all_natural_routes_match_capture | pass |
| manual_sentinel_natural_bitexact | pass |
| all_finite | pass |
| layer_0_mean_relative_l2_le_0_05 | fail |
| layer_0_p95_relative_l2_le_0_10 | fail |
| layer_0_mean_kl_le_0_001 | fail |
| layer_0_relative_ce_le_0_01 | fail |
| layer_0_top1_ge_0_99 | fail |
| layer_0_every_domain_relative_ce_le_0_02 | fail |
| layer_0_simplex_and_kkt | pass |
| layer_24_mean_relative_l2_le_0_05 | fail |
| layer_24_p95_relative_l2_le_0_10 | fail |
| layer_24_mean_kl_le_0_001 | fail |
| layer_24_relative_ce_le_0_01 | pass |
| layer_24_top1_ge_0_99 | fail |
| layer_24_every_domain_relative_ce_le_0_02 | pass |
| layer_24_simplex_and_kkt | pass |
| layer_47_mean_relative_l2_le_0_05 | fail |
| layer_47_p95_relative_l2_le_0_10 | fail |
| layer_47_mean_kl_le_0_001 | fail |
| layer_47_relative_ce_le_0_01 | fail |
| layer_47_top1_ge_0_99 | fail |
| layer_47_every_domain_relative_ce_le_0_02 | fail |
| layer_47_simplex_and_kkt | pass |
| overall_pass | fail |

## Claim boundary

This is a per-token non-causal coefficient oracle on three single-layer
interventions. It is not a trained TierFlow controller, full-48-layer
intervention, runtime, latency or deployment result. Validation/test inputs are
strictly disjoint but were previously used for P4D/TierFlow route analysis.

## Artifacts

- preregistration: `reports/streamq5_moe/TIERFLOW_PERSISTENT_SET_FUNCTIONAL_SPAN_PREREGISTRATION_2026-08-12.md`
- runner: `scripts/streamq5_moe/run_tierflow_persistent_set_functional_span.py`
- raw result: `reports/streamq5_moe/tierflow_persistent_set_functional_span_validation.json`
