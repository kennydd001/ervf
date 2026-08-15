# HERA-MoE P0 — multidomain-tieraudit

Uitkomst: **static_tier_negative**. De vaste multidomainunion bevat **6,081** hot en **63** cold laag-expertparen.

De geprojecteerde resident weights zijn **7.167 GiB** tegenover de vooraf vastgelegde 5,75-GiB-gate; cold BF16 is **0.554 GiB**.

| Domein | Hot per domein | Gem. cold calls/token | p95 | p99 |
|---|---:|---:|---:|---:|
| general | 4453 | 0.012 | 0 | 0 |
| code | 4168 | 0.031 | 0 | 1 |
| math | 4823 | 0.047 | 0 | 1 |
| multilingual | 4320 | 0.018 | 0 | 1 |
| instruction | 4957 | 0.072 | 0 | 2 |

General reproduceert de gesloten E2GQ-counts exact: `False`. Alle routehooks en logits zijn exact: `False`.

Dit is uitsluitend een router/tierbesluit. Er is geen GPTQ, kwaliteitsmeting, werkelijk entropybestand, cold-transferbenchmark of tokens/s-resultaat geopend.
