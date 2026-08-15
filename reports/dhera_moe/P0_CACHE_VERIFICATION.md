# DHERA-MoE P0 — onafhankelijke cacheverificatie

Uitkomst: **cache_trace_negative_verified**. Alle **22/22** controles slagen.

| Domein | Gem. MiB/token | p95 | p99 | Verkeersgate |
|---|---:|---:|---:|:---:|
| general | 91.336 | 279 | 432 | FAIL |
| code | 365.123 | 909 | 1296 | FAIL |
| math | 335.077 | 828 | 1053 | FAIL |
| multilingual | 230.323 | 738 | 1080 | FAIL |
| instruction | 205.151 | 558 | 828 | FAIL |

De geheugengate slaagt op **5.749055 GiB** resident en **16.382812 GiB** host-cold. De verkeersgate faalt voor ieder domein; P1 blijft gesloten.

Deze conclusie falsifieert uitsluitend de vooraf geregistreerde 4.280-base + 48-primary + 8-victimpolicy. Zij bewijst niet dat iedere dynamische of domeingeconditioneerde cache onmogelijk is.
