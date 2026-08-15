# DHERA-MoE P0 — vaste budgetcachetrace

Voorlopige uitkomst: **cache_trace_negative_pending_independent_verification**. Onafhankelijke verificatie is nog vereist.

De geprojecteerde resident weights zijn **5.749055 GiB**; de exacte cold bank in host-RAM is **16.382812 GiB**.

| Domein | Base-calls | Cold hitrate | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| general | 95.470% | 41.664% | 91.336 | 279 | 432 | FAIL |
| code | 86.554% | 21.427% | 365.123 | 909 | 1296 | FAIL |
| math | 87.930% | 19.670% | 335.077 | 828 | 1053 | FAIL |
| multilingual | 90.516% | 29.726% | 230.323 | 738 | 1080 | FAIL |
| instruction | 91.117% | 33.179% | 205.151 | 558 | 828 | FAIL |

De cachepolicy, basis van 4.280 experts, 56 slots, contextreset, verkeersgates en out-of-sample inputsets zijn niet gesweept.

Dit is een routesimulatie. Zij bewijst geen echte PCIe-latency of overlap, geen entropy-packgrootte, geen modelkwaliteit en geen 10 tokens/s.
