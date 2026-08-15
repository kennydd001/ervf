# DCHERA-MoE P0A — eindbesluit

**Besluit: `p0a_exploratory_negative_verified`.** Alle 18 onafhankelijke
controles slagen.

| Domein | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|:---:|
| general | 18,669 | 54 | 153 | PASS |
| code | 24,570 | 63 | 567 | FAIL |
| math | 43,059 | 126 | 261 | PASS |
| multilingual | 21,593 | 72 | 216 | PASS |
| instruction | 59,233 | 225 | 666 | FAIL |

De cijfers omvatten bij iedere 1.024-tokencontext een volledige geprojecteerde
basewissel. Domeinconditionering is dus veel sterker dan de globale basis, maar
de code- en instructionstaarten doorbreken de vooraf vastgelegde eis dat ieder
domein moet slagen. P0B en de transferbenchmark blijven gesloten.

Dit was bovendien een exploratie op reeds geopende routes. De uitkomst sluit de
vaste bekende-domeinbasis; zij sluit causale adaptatie binnen een context niet.
