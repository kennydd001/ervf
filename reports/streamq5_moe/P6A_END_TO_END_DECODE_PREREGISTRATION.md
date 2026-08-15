# STREAMQ5-MoE P6A — fysieke end-to-end decode-preregistratie

Datum: 2026-08-12. Status bij vastlegging: geen P6-bank, P6-runtime-output of
P6-beslissing geopend.

## Hypothese

Een fysiek Q5-expert/Q8-trunk Qwen3-30B-A3B-model kan op de aanwezige 8-GB-GPU
een echte batch-1 autoregressieve decoder uitvoeren met live router-top-8,
gewogen MoE-reductie, RMSNorm, RoPE, causale GQA-attention, BF16-KV-mutatie,
residuals, LM-head, greedy sampling en autoregressieve feedback. De primaire
doelpoort is minimaal 10 gegenereerde tokens per seconde (`mean <= 100 ms`) bij
een ononderbroken rollout van 512 tokens.

## Semantische reparatie vóór de proef

P5A koos INT8-codes met de reeds naar BF16 afgeronde schaal. P0C koos codes met
de FP32-maxabs-schaal en rondde alleen de opgeslagen dequantisatieschaal naar
BF16. P5A blijft als projectieplane-resultaat bestaan, maar mag daarom niet als
letterlijk identieke P0C-kwaliteitssemantiek worden gebruikt.

P6 bouwt vóór uitvoering een nieuwe immutable Q8-bank volgens exact P0C:

1. `scale_fp32 = max(abs(group)) / 127`;
2. `code = round_nearest_even(weight / scale_fp32)`, begrensd tot `[-127,127]`;
3. alleen de persistente schaal wordt BF16;
4. fysiek gewicht bij decode is `BF16(code * BF16_scale)`.

De bank omvat q/k/v/o/router voor 48 lagen, LM-head én de host-residente
embedding. Alle input/post-attention/q/k/finale RMSNormgewichten worden als
oorspronkelijke BF16-bits opgeslagen. De bestaande P1D-expertbank heeft reeds
dezelfde P0C-codekeuze.

## Vastgelegde modelsemantiek

- Qwen3-MoE: 48 lagen, hidden 2048, 32 queryheads, 4 KV-heads, head-dim 128;
- RoPE default, theta 1.000.000, rotate-half over twee helften van 64;
- RMSNorm epsilon `1e-6`;
- attention-softmax in FP32, 8 queryheads per KV-head;
- router-softmax over 128 experts in FP32, top-8, geselecteerde kansen opnieuw
  genormaliseerd en daarna naar BF16 afgerond;
- experts worden in oplopende expert-ID-volgorde gewogen/opgeteld, conform de
  referentie-implementatie;
- KV wordt werkelijk per laag en positie als BF16 geschreven en gelezen;
- tussenresultaten worden op de BF16-modelgrenzen naar BF16 afgerond;
- LM-head blijft permanent device-resident; embeddinglookup is fysiek Q8 op de
  host en levert een BF16-afgeronde activatie;
- sampling voor de rollout is deterministisch greedy; EOS stopt de meting niet,
  zodat exact 512 decode-stappen worden gemeten.

## Cachebeleid

De expertcache blijft exact 4.977.623.040 bytes. Per laag zijn 20 statische
slots plus 15 dynamische slots in lagen 0–7 en 14 in lagen 8–47 beschikbaar.
Statische sets worden uitsluitend uit de reeds gesloten P4D-calibratiesplit
gekozen. De kwaliteitsproef gebruikt de vooraf bekende domeinset van zijn
corpus; de vrije rollout gebruikt de `general`-set. Misses worden causaal in
dezelfde laag gekopieerd en kunnen alleen met reeds aanwezige expertcompute
overlappen; geen toekomstige routes worden gebruikt.

## Fasen, data en poorten

1. Bankbouw en onafhankelijke verificatie: alle bytes/aantallen/hashes exact;
   15 vooraf vaste Q8-records plus embedding en alle normrecords exact opnieuw
   afgeleid uit het checkpoint.
2. Smoke: CUDA-microcontroles voor Q8, Q5, RMSNorm, RoPE/KV, attention,
   gewogen reductie en logits-statistiek; daarna één volledige context van 8
   tokens. Geen beslissende validation/test-output wordt geopend bij een fail.
3. Validation: de gesloten P0C-validationdata, 5 domeinen × 2 contexten × 128
   tokens (1.270 next-tokenlabels). Test opent alleen als alle poorten slagen.
4. Test: de disjuncte P0C-testdata met dezelfde omvang. Bij pass volgt binnen
   dezelfde gesloten evaluator een rollout van exact 512 nieuwe tokens na het
   vaste UTF-8-prompt `The future of efficient artificial intelligence is`.

Beslissende poorten per validation/test:

- alle 1.270 labels verwerkt; alle activaties, logits, CE's en tijden eindig;
- relatieve aggregate next-token-CE versus de reeds vastgelegde BF16-teacher
  `<= 0,02`; geen herkalibratie op validation/test;
- gemiddelde end-to-end tokenwandtijd `<= 100 ms`, p95 `<= 150 ms`;
- trunkbank, expertcache en een 4.096-token BF16-KV-cache tegelijk resident,
  met minimaal 192 MiB vrije device-scratch na vaste allocaties;
- live router: exact acht unieke experts per laag/token en gewichten sommen
  binnen `2e-2` van één na BF16-afronding;
- gemeten expertmissbytes zijn exact `misses * 3.035.136`;
- iedere verwerkte positie muteert K en V in alle 48 lagen; geen toekomstige
  KV-posities worden gelezen;
- onafhankelijke outputverificatie herberekent tellingen, hashes, CE,
  percentielen, throughput, route-invarianten en rolloutfeedback.

Voor de 512-tokenrollout gelden aanvullend: exact 512 token-ID's, iedere stap
gebruikt de vorige argmax als volgende embedding, gemiddelde `<= 100 ms`, p95
`<= 150 ms`, eindige logits en decodeerbare tekst. Kwaliteit wordt primair door
de gesloten teacher-forced CE-proef beslist, niet door subjectieve tekstevaluatie.

## Beslissingsbetekenis

Een volledige pass bewijst op deze machine voor deze artefacten zowel
full-depth fysieke kwaliteit als de feitelijke end-to-end batch-1-decodesnelheid
van de volledige custom dataplane. Een fail sluit alleen de gefaalde poort; hij
mag niet als Eureka-pass worden geherinterpreteerd. Externe generalisatie naar
andere GPU's, contextlengten boven 4.096, batches groter dan één of andere
modellen blijft buiten de claim.
