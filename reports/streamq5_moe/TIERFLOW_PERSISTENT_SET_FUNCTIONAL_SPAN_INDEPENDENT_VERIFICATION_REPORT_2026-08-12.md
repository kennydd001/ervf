# TierFlow persistent-set functional-span — independent verification

- Status: **PASS**
- Source validation SHA-256: `4c85d0c5d9ce9e29c1f68a750502364d7a59f2d53831173f275b5f2113330615`
- Checks passed: **16/16**
- Recomputed validation gate: **FAIL**
- Test artifact exists: **False**

## Recomputed sentinel metrics

| layer | mean rel-L2 | p95 rel-L2 | mean KL | relative CE | top-1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.512516 | 0.865522 | 0.044850 | 1.642% | 92.784% |
| 24 | 0.407422 | 0.913767 | 0.005821 | -0.194% | 97.333% |
| 47 | 0.220883 | 0.590669 | 0.043767 | 3.283% | 93.725% |

## Check ledger

| check | pass |
|---|:---:|
| all_locked_input_hashes | yes |
| canonical_input_hash | yes |
| canonical_capture_hash | yes |
| artifact_audit_passed | yes |
| strict_validation_partition | yes |
| sentinel_contract | yes |
| raw_shapes_match_preregistration | yes |
| raw_values_finite_and_feasible | yes |
| oracle_summaries_recomputed | yes |
| baseline_statistics_recomputed | yes |
| downstream_statistics_recomputed | yes |
| traffic_recomputed_from_locked_f0 | yes |
| all_gate_decisions_recomputed | yes |
| validation_failed | yes |
| test_partition_remained_closed | yes |
| no_training_or_download | yes |

The verifier does not rerun the model. It independently recomputes every published statistic and hard gate from the frozen per-token arrays, verifies the locked local artefacts, and confirms that validation failure prevented test access.
