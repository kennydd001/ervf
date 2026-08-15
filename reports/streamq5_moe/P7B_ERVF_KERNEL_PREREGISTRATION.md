# STREAMQ5-MoE P7B — Exact-Reduction Virtual Fusion (ERVF)

Datum: 2026-08-12. Status bij vastlegging: geen P7B-output geopend.

## Hypothese

P7A classificeerde zowel Q8 als Q5 vooraf als
`row_geometry_reduction_or_launch_dominant`. De bestaande kernel reserveert één
256-thread block per outputrij en voert acht blockbrede synchronisatiestappen
uit. ERVF laat één subwarp meerdere rijen verwerken, maar emuleert per rij alle
256 oorspronkelijke virtuele threads en exact dezelfde binaire FP32-optelboom.
Daardoor kan de launch/reductie-overhead dalen zonder de BF16-uitvoerbitjes te
veranderen.

## Vastgelegde varianten en selectie

- Subwarpbreedtes 8, 16 en 32 worden getest; een block blijft 256 threads.
- Iedere lane houdt respectievelijk 32, 16 of 8 virtuele threadaccumulatoren.
- De stappen van stride 128 tot en met de subwarpbreedte gebeuren lane-lokaal;
  de resterende stappen gebruiken width-begrensde warp shuffles.
- Q8 gebruikt alle 241 deviceprojecties. Q5 gebruikt experts 0–7 van alle 48
  lagen, zonder H2D in het getimede gebied.
- Eerst wordt over de volledige workload bitgelijkheid tegen de ongewijzigde
  P6B-kernel geëist. Een niet-bitgelijke variant wordt uitgesloten.
- Validation: 5 warmups en 30 metingen. Per bank wordt de bitgelijke variant
  met laagste p50 gekozen.
- Test: verse timingreeks met 10 warmups en 120 metingen van baseline en de
  vooraf door validation gekozen variant.

## Primaire poorten

ERVF slaagt voor een bank alleen als:

1. alle geteste BF16-uitvoerbitjes identiek zijn aan de P6B-kernel;
2. `test_variant_p50 / test_baseline_p50 <= 0,90`;
3. `test_variant_p95 / test_baseline_p95 <= 0,95`;
4. alle tijden en uitgangen eindig zijn.

De kernelhypothese heet bevestigd als Q5 én Q8 slagen. Een smallere uitkomst
wordt per bank gerapporteerd en niet opgewaardeerd tot end-to-end winst.

## Grenzen

Deze test bewijst nog geen volledige tok/s-winst. Bij een pass volgt een aparte
P7C-integratie in de strikte P6B-runtime, met exact dezelfde kwaliteits- en
geheugenpoorten. De naam ERVF is een lokale werknaam, geen claim van
wereldwijde nieuwheid of afwezigheid van verwant eerder werk.
