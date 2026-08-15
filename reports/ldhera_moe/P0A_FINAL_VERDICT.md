# LDHERA-MoE P0A — eindbesluit

**Besluit: `p0a_exploratory_negative_verified`.** Alle 20 onafhankelijke
controles slagen, inclusief de training-misscurves en exacte DP-allocaties.

| Domein | Gem. MiB/token | p95 | p99 | Gate |
|---|---:|---:|---:|:---:|
| general | 19,274 | 63 | 162 | PASS |
| code | 26,867 | 63 | 657 | FAIL |
| math | 44,000 | 126 | 270 | PASS |
| multilingual | 23,115 | 81 | 234 | PASS |
| instruction | 62,670 | 243 | 711 | FAIL |

De allocation gebruikt exact 56 laaglokale slots en is uitsluitend op
HERA-trainingroutes geoptimaliseerd. Eén volledige basewissel per context is
meegerekend. De gemiddelden zijn meestal goed, maar optimalisatie van totale
training-misses generaliseert niet naar de vooraf begrensde validationstaarten.
P0B en P1 blijven gesloten.
