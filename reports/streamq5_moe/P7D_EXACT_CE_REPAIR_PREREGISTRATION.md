# STREAMQ5-MoE P7D — exacte CE-retentiereparatie

Datum: 2026-08-12. Status bij vastlegging: geen P7D-output geopend.

## Aanleiding

P7C was vooraf verplicht iedere validation- en test-CE exact met P6B te
vergelijken. De bestaande P6-evaluator bewaarde echter alleen CE-gemiddelden,
voorspellingen en missreeksen, niet de 1.270 individuele CE-waarden. Dat is een
bewijsretentiegat: gelijke gemiddelden zijn niet voldoende voor de letterlijk
vastgelegde poort.

## Reparatie

- De ongewijzigde P6B-baseline en geselecteerde 16-lane P7C-ERVF-runtime worden
  ieder één keer nieuw geladen.
- Binnen iedere load worden de volledige bestaande validation- en testsplits
  doorlopen.
- De enige evaluatorwijziging is het bewaren van alle `domain_ce`- en `all_ce`-
  arrays in het JSON-artifact.
- De inputs, routes, cache, kernels, stopwatch en modelsemantiek blijven gelijk.
- Timings uit deze reparatierun worden niet gebruikt voor de snelheidsclaim.

## Poorten

P7D sluit het gat alleen als voor validation én test alle volgende structuren
exact gelijk zijn tussen baseline en ERVF:

1. alle 1.270 CE-floats;
2. alle voorspellingen en hun SHA-256;
3. alle expertmissen;
4. alle KV-digests;
5. alle uitgangen zijn eindig en beide runs tellen exact 1.270 labels.

Bij enig verschil blijft P7C methodologisch onbewezen, ongeacht de snelheid.
