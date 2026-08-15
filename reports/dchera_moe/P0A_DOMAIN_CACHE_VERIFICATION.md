# DCHERA-MoE P0A — onafhankelijke verificatie

Uitkomst: **p0a_exploratory_negative_verified**; **18/18** controles slagen.

| Domein | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|:---:|
| general | 18.669 | 54 | 153 | PASS |
| code | 24.570 | 63 | 567 | FAIL |
| math | 43.059 | 126 | 261 | PASS |
| multilingual | 21.593 | 72 | 216 | PASS |
| instruction | 59.233 | 225 | 666 | FAIL |

General, math en multilingual passeren. Code en instruction falen de staartgates, zodat P0B en P1 gesloten blijven.

De conclusie geldt voor de vaste bekende-domeinbasis en niet voor alle mogelijke contextadaptieve caches.
