# DCHERA-MoE P0A — domeingeconditioneerde cache

Voorlopige exploratieve uitkomst: **p0a_exploratory_negative_pending_verification**.

| Domein | Base-calls | Cold hitrate | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| general | 99.259% | 44.846% | 18.669 | 54 | 153 | PASS |
| code | 99.033% | 40.034% | 24.570 | 63 | 567 | FAIL |
| math | 98.258% | 36.012% | 43.059 | 126 | 261 | PASS |
| multilingual | 99.034% | 48.935% | 21.593 | 72 | 216 | PASS |
| instruction | 97.477% | 37.282% | 59.233 | 225 | 666 | FAIL |

Iedere 1.024-tokencontext draagt conservatief een volledige basewissel van 4648.182 MiB. Resident: 5.749055 GiB; actieve host-cold: 16.382812 GiB.

De routes waren al geopend voor DHERA. Ook bij een positieve uitkomst is dit geen bevestiging: P0B moet nieuwe routes gebruiken.
