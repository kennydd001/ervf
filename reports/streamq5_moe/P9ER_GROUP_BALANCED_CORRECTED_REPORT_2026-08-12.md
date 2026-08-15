# P9E-R — group-balanced 64/128-pruning gesloten

De regelmatige variant uit `pro1.txt` hield met echte in-place mutatie exact 64
van 128 neuronen in elk van de zes oorspronkelijke down-Q5-groepen. Alle 48
lagen veranderden werkelijk en alle gemaskeerde posities waren daarna nul.

| validation-metriek | resultaat | poort |
|---|---:|---:|
| relatieve CE-toename | **+47,186%** | ≤2,5% |
| top-1-overeenkomst | **60,866%** | ≥90% |
| eind-hidden relatieve L2 | **0,4889** | diagnostisch |
| groepsbalans | exact 64/128, 6/6 groepen | vereist |

Status: **validation_closed**; de test-split bleef dicht. Regelmatige
groepsbezetting repareert de kwaliteitsinstorting dus niet. GaugePack wordt op
50% pruning voor zowel het oorspronkelijke arbitrary masker als deze
group-balanced variant gesloten.
