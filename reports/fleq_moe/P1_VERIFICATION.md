# FLEQ-MoE P1 — onafhankelijke verificatie

Uitkomst: **PASS (18/18)**.

De verifier heeft de lockselectie, 32 rapporthashes, 32 artifacthashes, alle verbeteringsformules, codebereiken, bitbudgetten en de vooraf vastgelegde gate onafhankelijk gecontroleerd. Voor de eerste geselecteerde expert van beide lagen zijn bovendien alle held-out metrics van 2-bit GPTQ/GSQ en ternary RTN/GSQ opnieuw uit de opgeslagen gewichten berekend.

De bewezen conclusie is begrensd: P1 is `smoke_negative`, P2 is niet geautoriseerd en er is geen Eureka-claim. Dit bewijst geen algemene onmogelijkheid van low-entropy MoE-quantisatie.

## Controles

- `verdict_is_smoke_negative`: `True`
- `p2_is_not_authorized`: `True`
- `no_eureka_claim`: `True`
- `all_execution_controls_pass`: `True`
- `two_bit_gate_fails`: `True`
- `selection_lock_hash_matches`: `True`
- `all_three_addenda_hash_match`: `True`
- `selected_experts_match_lock`: `True`
- `all_32_report_hashes_match`: `True`
- `all_32_artifact_hashes_match`: `True`
- `all_improvement_formulas_match`: `True`
- `all_code_ranges_match`: `True`
- `primary_has_zero_of_16_improvements`: `True`
- `primary_has_16_of_16_p95_regressions`: `True`
- `two_bit_effective_bpp_is_2_125`: `True`
- `ternary_bounds_match`: `True`
- `attempt_004_preserved`: `True`
- `eight_anchor_metric_sets_recompute`: `True`
