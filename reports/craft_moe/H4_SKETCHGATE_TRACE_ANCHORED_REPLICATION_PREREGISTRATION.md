# H4 SketchGate — preregistratie trace-anchored replicatie

Vastgelegd op `2026-08-10T11:20:59.3729942Z`, na de eerste volledige H4-run
maar vóór inspectie van de hieronder vastgezette vensters. De eerste run blijft
ongewijzigd bewaard in `sketchgate.json`: zij miste de inhoudelijke gates ruim,
maar haar extra Q3/Q4-herberekeningscontrol faalde ook. Een read-only diagnose
met de oorspronkelijke 2.048-tokenbatch bleef buiten de vooraf vastgelegde
tolerantie (Q3 NRMSE `0,002743`, Q4 `0,002300`). Die tolerantie wordt niet
achteraf verruimd.

## Doel en vaste vensters

Deze afzonderlijke replicatie bepaalt of de inhoudelijke H4-negatief standhoudt
wanneer de bevroren componenttrace zelf het Q3/Q4-meetanker is:

- WikiText-validatie trace-indices `256..511` voor configuratiekeuze;
- WikiText-test trace-indices `1280..1535`, overeenkomend met splitposities
  `256..511`, éénmalig na validatiekeuze;
- 256 tokens per split, 128-token sequence blocks, 10.000× bootstrap;
- dezelfde model- en datasetrevisions, laag 26, natuurlijke top-6 en
  ongenormaliseerde routergewichten.

De vensters vallen binnen een eerder geaggregeerd 1.024-tokenbaseline-artifact,
maar hun H4-scores, attributie, high-damage-labels en losse resultaten zijn nog
niet bekeken. Dit is daarom een onafhankelijke exploratieve H4-replicatie, geen
verse confirmatieclaim.

## Trace-anchored kandidaat

De opgeslagen `selected_quant3` en `selected_quant4` zijn de gezaghebbende
all-Q3/all-Q4 expertoutputs. Iedere kandidaat wordt in de originele laagvorm
opgebouwd:

`candidate = BF16(post_attention + BF16(candidate_routed + exact_shared))`.

Hiermee zijn all-Q3 en all-Q4 exact dezelfde getallen die de eerdere oracle
gebruikte. De officiële original-control is het opgeslagen teacher-hidden zelf
en moet volledige-vocabulaire KL `0`, CE-delta `0` en top-1 `1` geven. Routes
moeten exact herberekenen. Nieuwe gate/up/down-hybriden en downsketches worden
uit dezelfde lokale gewichten berekend; hun herberekende all-Q3/Q4-output blijft
als diagnostiek zichtbaar maar is geen controlvervanging.

## Ongewijzigde hypothese en gates

Alle inhoudelijke regels uit
`H4_SKETCHGATE_LAYER26_PREREGISTRATION.md` blijven ongewijzigd:

- fase A: gate, up, down, gate+up en alle matrices; alle 64 maskers; exact 25%;
- down-attributie minstens 70% op beide splits;
- Gaussian/Rademacher, `r={4,8,16,32,64}`, seeds `20260810..20260814`;
- int8 `u` met FP16-schaal, geen calibratie, score met routergewicht²;
- validatiekeuze over alle vijf seeds; primaire bank seed `20260810`;
- recovery minstens 80% en high-damage-FN maximaal 1% voor alle vijf seeds op
  beide splits;
- metadata minder dan 0,1 effectieve bit;
- sketchcompute minder dan 10% van vermeden-transfer-tijd in hetzelfde expliciet
  gelabelde batch-1 hardwaremodel;
- exact 25% upgrades en dezelfde random/router/non-deployable controles.

Output is append-only
`reports/craft_moe/sketchgate_trace_anchored_replication.json`. Alleen als alle
gates slagen, opent spreadonderzoek. Een nieuwe negatieve uitkomst met sluitende
controls falsificeert H4; test mag niets aan score, selectie of grenzen wijzigen.
