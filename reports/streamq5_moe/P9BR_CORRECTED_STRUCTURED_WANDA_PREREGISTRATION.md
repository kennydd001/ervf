# P9B-R — corrected structured-Wanda preregistration

Datum: 2026-08-12. Vastgelegd nadat de P9B/P9E-maskgevoeligheidsaudit de fout
identificeerde en voordat enig P9B-R-resultaat werd geopend.

## Reden voor replicatie

De oude helper gebruikte `weight[boolean_mask].zero_()`. PyTorch advanced
indexing retourneert daar een kopie; de gewichtsparameter werd niet gewijzigd.
P9B rapporteerde daardoor de gewone Q5-baseline en bewijst geen pruning. De
P9B-maskers zelf blijven bruikbaar: ze zijn vóór de foutieve mutatie uit de
ongewijzigde calibratie-forward en de vastgelegde score afgeleid.

## Bevroren kandidaat

- gebruik uitsluitend het bestaande P9B-maskerartefact met SHA-256
  `a238950c97c52f7b5c12eaae76a1ef21f8c4ad4fabd4724bc13f9ed17801dab1`;
- per expert blijven exact 384/768 kanalen behouden;
- gate/up krijgen een in-place rijmasker via vermenigvuldiging met een
  `[768,1]`-booleanmasker; down krijgt hetzelfde via `[1,768]`;
- vóór Q5-quantisatie moet iedere gemaskeerde positie exact nul zijn en ten
  minste één oorspronkelijk niet-nul element moet aantoonbaar zijn gewijzigd;
- daarna blijven de bestaande Q5/group-128 expert- en INT8 trunk/headsemantiek
  en dezelfde vijf domeinen/10×128 contexten per split gelden.

## Poorten

Validation opent test uitsluitend bij: correcte maskerhash, 48 lagen, echte
mutatie op iedere laag, nul resterende gemaskeerde waarden, eindige uitvoer,
relatieve CE-toename `<=2,5%` en top-1-overeenkomst `>=90%`.
Definitieve pass vereist op validation én test relatieve CE `<=2,0%` en top-1
`>=90%`.

Een pass herstelt alleen de P9B-kwaliteitspremisse. GaugePack-bytes,
codecexactheid, kernelsnelheid en end-to-end winst blijven afzonderlijk
onbewezen. Een validationfail sluit GaugePack op dit vaste 50%-masker.
