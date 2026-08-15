# DHERA-MoE P0 — eindbesluit

**Besluit: `cache_trace_negative_verified`.**

De ene vooraf geregistreerde budgetcache past in het geheugenbudget, maar niet
in het verkeersbudget. De onafhankelijke verifier reproduceert alle 22
controles.

| Domein | Gem. MiB/token | Limiet | Factor boven limiet | p95 / limiet | p99 / limiet |
|---|---:|---:|---:|---:|---:|
| general | 91,336 | 64 | 1,427× | 279 / 144 | 432 / 288 |
| code | 365,123 | 64 | 5,705× | 909 / 144 | 1296 / 288 |
| math | 335,077 | 64 | 5,236× | 828 / 144 | 1053 / 288 |
| multilingual | 230,323 | 64 | 3,599× | 738 / 144 | 1080 / 288 |
| instruction | 205,151 | 64 | 3,206× | 558 / 144 | 828 / 288 |

Het mechanische probleem is zichtbaar ondanks hoge base-coverage: general
heeft 95,470% base-calls, maar de resterende cold calls halen slechts 41,664%
cache-hitrate. Code heeft 86,554% base-calls en slechts 21,427% cold-hitrate.
Met 384 expertcalls per token stapelen kleine cold-fracties daardoor snel op.

De geheugengate slaagt: 5,749055 GiB resident en 16,382812 GiB exact-cold in
host-RAM. Omdat elk domein de vooraf geregistreerde transfergates faalt, blijft
de echte transfermicrobenchmark P1 gesloten.

## Claimgrens

Gefalsificeerd is uitsluitend de training-geaggregeerde globale basis van
4.280 experts met 48 primary- en 8 globale victimslots, contextreset iedere
1.024 tokens en onveranderde officiële routes. Dit resultaat bewijst niet dat
een vooraf gekozen domeingeconditioneerde basis onmogelijk is.
