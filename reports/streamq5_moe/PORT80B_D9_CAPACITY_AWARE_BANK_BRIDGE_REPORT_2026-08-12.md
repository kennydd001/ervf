# PORT80B-D9 — capacity-aware 499+13 bank bridge report

Verdict: **capacity_bridge_strong_pass**. Primary pass: **True**. Strong pass: **True**. Clean unregister: **True**.

| case | validation wall p50 ms | test wall p50 ms | test wall p95 ms |
|---|---:|---:|---:|
| all_hot | 48.318149987608194 | 48.49565000040457 | 49.1163649916416 |
| mixed_5_hot_5_cold | 62.19029999920167 | 61.900600005174056 | 68.6695049967966 |
| all_cold_tail | 81.91599999554455 | 80.95004998904187 | 88.13604997849325 |

The differentiated header oracle includes positive all-hot/mixed/all-cold checks and deliberate wrong-expert/wrong-layer controls. Pass/fail timing is inclusive wall time; CUDA events are diagnostic.

Claim boundary: Exact synthetic 499+13 active-plane bridge only; no 512/full-bank registration, stable capacity, real checkpoint, natural traffic, quality, physical dense shell, end-to-end tok/s or endurance claim.
