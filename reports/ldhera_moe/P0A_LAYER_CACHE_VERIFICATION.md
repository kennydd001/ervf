# LDHERA-MoE P0A — onafhankelijke verificatie

Uitkomst: **p0a_exploratory_negative_verified**; **20/20** controles slagen.

| Domein | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|:---:|
| general | 19.274 | 63 | 162 | PASS |
| code | 26.867 | 63 | 657 | FAIL |
| math | 44.000 | 126 | 270 | PASS |
| multilingual | 23.115 | 81 | 234 | PASS |
| instruction | 62.670 | 243 | 711 | FAIL |

De training-misscurves, exacte DP-allocaties en validation-LRU zijn onafhankelijk gereproduceerd.
Code en instruction falen; P0B en P1 blijven gesloten.
