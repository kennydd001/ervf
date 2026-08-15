# P9E1 — group-balanced structured-Wanda preregistration

Datum: 2026-08-12. Vastgelegd voordat enig P9E1-resultaat is geopend.

## Hypothese

P9B hield de globaal beste 384 van 768 expertkanalen en passeerde de
full-depth kwaliteitspoort. GaugePack kan eenvoudiger en sneller worden als
iedere oorspronkelijke down-quantisatiegroep van 128 exact 64 kanalen houdt.
P9E1 test daarom dezelfde activatie-RMS maal down-kolomnormscore als P9B, maar
selecteert onafhankelijk de beste 64 kanalen in elk van de zes oorspronkelijke
groepen. Ties worden nog steeds door de oorspronkelijke neuronindex gebroken.

## Bevroren semantiek en data

- dezelfde vijf domeinen, calibratievensters, validation/test-contexten en
  teacher als P9B;
- exact 384 unieke kanalen per expert: 64 uit elk interval
  `[0,128)`, ..., `[640,768)`;
- niet-overlevende gate/up-rijen en down-kolommen worden vóór de bestaande
  Q5/group-128-quantisatie op nul gezet;
- trunk en LM-head blijven op de bestaande INT8-semantiek;
- validation schrijft nieuwe P9E1-maskers; test hergebruikt uitsluitend die
  verzegelde maskers.

## Poorten

Validation opent test bij eindige uitvoer, alle 48 lagen, exact 64/128 per
groep, relatieve CE-toename `<=2,5%` en top-1-overeenkomst `>=90%`.
Definitieve pass vereist validation én test relatieve CE `<=2,0%` en top-1
`>=90%`.

Een kwaliteitspass bewijst alleen dat de group-balanced selectie de P9B-
kwaliteitspoort houdt. Fysieke bytes, exacte GaugePack-decodering, kernelwinst
en end-to-end snelheid blijven afzonderlijke poorten.
