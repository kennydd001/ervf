# P9E-R — corrected group-balanced pruning preregistration

Datum: 2026-08-12. Vastgelegd na P9B-R validation-close en vóór P9E-R-output.

## Vraag

De door `pro1.txt` voorgestelde regelmatige variant houdt exact 64 kanalen in
ieder oorspronkelijk down-Q5-interval van 128. De eerdere P9E1-run gebruikte
dezelfde foutieve no-op-helper als P9B en levert geen kwaliteitsevidence.
P9E-R test de reeds verzegelde P9E1-maskers met echte in-place mutatie.

## Bevroren kandidaat en poorten

- exact het bestaande P9E1-maskerartefact en dezelfde vijf domeinen/contexten;
- exact 64/128 per oorspronkelijke groep, zes groepen en 384/768 totaal;
- echte in-place gate/up-rij- en down-kolommaskering vóór de bestaande Q5-
  quantisatie; nul resterende gemaskeerde waarden en effectieve mutatie op alle
  48 lagen zijn verplicht;
- validation: relatieve CE `<=2,5%`, top-1 `>=90%`; test wordt alleen dan
  geopend en vereist validation én test CE `<=2,0%`, top-1 `>=90%`.

Een pass bewijst alleen kwaliteit van dit nieuwe regelmatige pruningmodel.
Fysieke GaugePack-bytes, codecexactheid, kernel- en runtimewinst blijven apart.
