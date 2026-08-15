# PORT80B-D9 — independent CPU verification

Verdict: **capacity_bridge_strong_pass_independently_verified**. Checks: **16/16**.

| case | validation wall p50 ms | test wall p50 ms | test wall p95 ms |
|---|---:|---:|---:|
| all_hot | 48.318 | 48.496 | 49.116 |
| mixed_5_hot_5_cold | 62.190 | 61.901 | 68.670 |
| all_cold_tail | 81.916 | 80.950 | 88.136 |

Wrong-expert and wrong-layer controls detected 3 and 150 byte mismatches. All positive images had zero mismatches; all outputs were bitexact. All 48 registered ranges unregistered cleanly.

RAM was 52.887 GB before and 52.778 GB immediately after registration, but 3.123 GB after the run and clean unregister. Registration therefore did not initially fault most pages; the timed workload first-touched the mapped bank and file-backed pages remained resident/standby. This is not evidence of failed CUDA cleanup, but it also does not prove prompt OS reclamation or stable endurance headroom.

Capacity caveat: The sweep is cumulative, not five independent cold-start trials. Available RAM fell sharply and was not restored after clean unregisters, consistent with persistent residency/cache/OS effects. Thus 499 is the largest clean point observed in this sequence, not a stable monotone capacity guarantee or endurance result.

Claim boundary: CPU-only verification of frozen D9 evidence; no GPU rerun, registration, bank sweep, model, quality or endurance claim.
