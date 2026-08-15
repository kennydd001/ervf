# LDHERA-MoE P0A — training-geleerde laagcache

Voorlopige exploratieve uitkomst: **p0a_exploratory_negative_pending_verification**.

| Domein | Cold hitrate | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|---:|:---:|
| general | 42.484% | 19.274 | 63 | 162 | PASS |
| code | 33.156% | 26.867 | 63 | 657 | FAIL |
| math | 34.448% | 44.000 | 126 | 270 | PASS |
| multilingual | 44.377% | 23.115 | 81 | 234 | PASS |
| instruction | 33.340% | 62.670 | 243 | 711 | FAIL |

Alle allocaties zijn vóór validation uit trainingroutes gelockt; één volledige basewissel per context is meegerekend.
P0A gebruikt geopende routes en kan alleen verse P0B openen.
