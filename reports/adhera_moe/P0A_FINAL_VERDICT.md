# ADHERA-MoE P0A — eindbesluit

**Besluit: `p0a_exploratory_negative_verified`.** Alle 17 onafhankelijke
controles slagen.

| Domein | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|:---:|
| general | 27,988 | 81 | 189 | PASS |
| code | 22,576 | 45 | 405 | FAIL |
| math | 73,758 | 198 | 369 | FAIL |
| multilingual | 29,524 | 90 | 234 | PASS |
| instruction | 72,267 | 261 | 666 | FAIL |

De policy rekent twee volledige geprojecteerde basewissels per context mee.
Code verbetert ten opzichte van de vaste domeinbasis, maar de p99 blijft te
hoog. Math en instruction tonen dat routes uit de eerste 64 tokens geen
betrouwbare proxy zijn voor de resterende 960 tokens.

Er is geen warmuplengte- of selectorsweep uitgevoerd. Omdat slechts twee van
de vijf domeinen slagen, worden P0B en P1 niet geopend. Dit sluit deze ene
causale selector; het is geen algemeen onmogelijkheidsbewijs voor adaptieve
caches.
