# ADHERA-MoE P0A — onafhankelijke verificatie

Uitkomst: **p0a_exploratory_negative_verified**; **17/17** controles slagen.

| Domein | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|:---:|
| general | 27.988 | 81 | 189 | PASS |
| code | 22.576 | 45 | 405 | FAIL |
| math | 73.758 | 198 | 369 | FAIL |
| multilingual | 29.524 | 90 | 234 | PASS |
| instruction | 72.267 | 261 | 666 | FAIL |

De 64-tokenwarmup redt de code-p99 niet en verslechtert math en instruction.
P0B en P1 blijven gesloten.
