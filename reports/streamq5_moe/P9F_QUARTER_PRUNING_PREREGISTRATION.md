# P9F — corrected 25% structured-Wanda preregistration

Datum: 2026-08-12. Vastgelegd na de correcte P9B-R/P9E-R validationfails en
voordat P9F-maskers of resultaten bestaan.

## Hypothese

De P9B-score kan bij 50% pruning te agressief zijn zonder waardeloos te zijn.
P9F herhaalt de oorspronkelijke calibratie en score, maar behoudt exact de
beste 576 van 768 kanalen per expert (25% pruning). Dit is een nieuw model en
erft geen P9B-kwaliteitclaim.

## Bevroren semantiek

- dezelfde vijf domeinen, eerste tien 128-token-calibratievensters en verse
  P0C validation/test-contexten;
- score per expertkanaal: `RMS(SwiGLU-activatie) * ||down[:,j]||_2`, met
  oorspronkelijke index als tie-break;
- validation schrijft exact één verzegeld `(48,128,576)`-maskerartefact;
- echte in-place gate/up-rij- en down-kolommaskering vóór bestaande Q5-
  quantisatie; iedere laag moet niet-nulle gewichten wijzigen en exact nul
  gemaskeerde waarden overhouden;
- trunk en head blijven op bestaande INT8-semantiek.

## Poorten

Validation opent test bij eindige uitvoer, alle 48 lagen, effectieve mutatie,
exact 576 unieke indices per expert, relatieve CE `<=2,5%` en top-1 `>=90%`.
Definitieve pass vereist op validation én test relatieve CE `<=2,0%` en top-1
`>=90%`.

Een pass bewijst uitsluitend de full-depth kwaliteit van 25% structured
pruning. Fysieke compactie, exacte codec, kernelsnelheid en runtime blijven
afzonderlijke poorten.
