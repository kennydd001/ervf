# HERA-MoE P0 — multidomain-tieraudit

Uitkomst: **static_tier_negative**. De vaste multidomainunion bevat **6,081** hot en **63** cold laag-expertparen.

De geprojecteerde resident weights zijn **7.167 GiB** tegenover de vooraf vastgelegde 5,75-GiB-gate; cold BF16 is **0.554 GiB**.

| Domein | Hot per domein | Gem. cold calls/token | p95 | p99 |
|---|---:|---:|---:|---:|
| general | 4449 | 0.014 | 0 | 0 |
| code | 4173 | 0.030 | 0 | 1 |
| math | 4823 | 0.049 | 0 | 1 |
| multilingual | 4317 | 0.019 | 0 | 1 |
| instruction | 4951 | 0.071 | 0 | 2 |

Alle officiële routecalls en logits zijn exact onderschept: `True`. De historische E2GQ-counts zijn niet exact reproduceerbaar door tied BF16-topk (`L1=0`); dit is diagnostiek en geen gate-input.

Dit is uitsluitend een router/tierbesluit. Er is geen GPTQ, kwaliteitsmeting, werkelijk entropybestand, cold-transferbenchmark of tokens/s-resultaat geopend.
