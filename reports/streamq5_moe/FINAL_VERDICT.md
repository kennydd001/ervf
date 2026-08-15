# STREAMQ5-MoE - finaal onderzoeksverdict

Datum: 2026-08-12

## Verdict

**Strikte fysieke full-model decode-Eureka: bewezen en onafhankelijk
geverifieerd.**

De nieuwe constructie gebruikt exact RTN-Q5 experts met fysieke BF16-schalen,
een INT8-trunkkwaliteitskandidaat en een domeingekalibreerde cache van 1.640
expertslots. De doorslaggevende systeemvondst is niet alleen Q5, maar de
herverdeling van de fysiek beschikbare slots: **20 static plus 15/14 exact-LRU
dynamic**. De oorspronkelijke 32-static policy faalde op echte VRAM-residentie
en H2D-p95; de vooraf geregistreerde validation-only herverdeling werd daarna
op een ongeopende test bevestigd.

P6B voert nu ook attention, q/k-normalisatie, RoPE, echte BF16-KV-mutatie,
live routertop-8 met gewichten, residuals, hostembedding, finale norm, LM-head,
sampling en autoregressieve feedback in dezelfde fysieke decode-loop uit. Op de
gesloten test waren de relatieve CE-stijging **+0,0480%**, de end-to-end mean/p95
**49,927/58,187 ms** en de 512-tokenrollout **15,867 tok/s**. De onafhankelijke
eindcontrole slaagde **120/120**. De claim blijft lokaal: deze artefacten, deze
8-GB-GPU, batch 1 en maximaal 4.096 contexttokens.

## Beslissend bewijs

### Kwaliteit met exacte fysieke schaalvolgorde

- validation cross-entropy versus BF16: **+1.4778%**;
- once-only test: **-0.4782%** (datasetfluctuatie, geen kwaliteitswinstclaim);
- vooraf vastgelegde eindgate: beide <=+2%;
- onafhankelijke audit: **60/60**.

### Volledige fysieke bank

- **18,647,875,584 bytes (17.3671875 GiB)**;
- 48 lagen, 6,144 experts en 18,432 matrixrecords;
- 28,991,029,248 Q5-codes en 226,492,416 BF16-schalen;
- ieder record opnieuw gedecodeerd en uit immutable BF16-brongewichten
  herberekend: **0 code-, schaal-, CRC-, header-, padding- of hashfouten**;
- onafhankelijke audit: **18/18**.

### Directe fysieke Q5-kernel

- 72/72 gelockte fysieke matrices correct;
- p50: **64.767 Gweight/s**;
- conservatieve summed-p95: **36.658 Gweight/s**;
- geprojecteerde volledige expertcompute-p95: **49.429 ms/token**;
- onafhankelijke audit: **15/15**.

### Werkelijke residentie en gefragmenteerde H2D

De actuele CUDA-context had 7,385,120,768 vrije bytes. Daarom werd de eerdere
1,910-slotboekhouding verworpen en de fysieke cache vooraf op 1,640 slots
vastgezet. Werkelijk gelijktijdig gealloceerd en aangeraakt:

- expertcache: 4,977,623,040 bytes;
- INT8-trunkruimte: 1,541,093,376 bytes;
- KV-ruimte: 402,653,184 bytes;
- over na allocatie: **462,422,016 bytes (440.99 MiB)**.

De 32-static validatie faalde terecht met 35.447 ms p95. De validation-only
selectie koos 20 static; de test bleef daarbij ongeopend. Met de vaste nieuwe
policy maten echte afzonderlijke recordkopieën:

- validation mean/p95: **9.198/20.664 ms**;
- once-only test mean/p95: **9.565/20.230 ms**;
- exacte simulatieovereenkomst en transferintegriteit;
- onafhankelijke audit: **19/19**.

### Geïntegreerde fysieke expert-dataplane

Per token werden 48 lagen maal top-8 experts uitgevoerd: fysieke cachemissen,
fused gate/up Q5 GEMV, FP32 SwiGLU, down Q5 GEMV en acht-expertreductie, op één
seriële CUDA-stream zonder overlapcredit.

- validation mean/p95: **28.013/40.526 ms/token**;
- once-only test mean/p95: **28.101/39.456 ms/token**;
- testmaximum: 108.110 ms; er is dus geen absolute max-latentieclaim;
- testcachehit: circa **81.60%** van 384 expertverzoeken per token;
- alle outputs eindig; alle domeinen onder 60/75-ms mean/p95-gates;
- onafhankelijke missreconstructie plus 96/96 volledige H2D-records
  byte-exact; audit: **20/20**.

## Wat nu bewezen is

1. De exacte fysieke BF16-schaalsemantiek voldoet aan de kwaliteitspoort.
2. Een volledige 17.37-GiB Q5-expertbank is reproduceerbaar en bit-exact.
3. De bank kan volledig pinned in host-RAM staan.
4. De fysieke expertcache past werkelijk naast trunk- en KV-byteallocaties op
   de lokale 8-GiB GPU met meer dan 384 MiB scratch over.
5. De fysieke Q5-kernel is correct en snel genoeg voor de expertbudgetgate.
6. Een 20-static/15-14-dynamic cache houdt echte H2D ruim binnen het budget.
7. De complete routed-expert data plane draait fysiek rond 28 ms gemiddeld en
   39-41 ms p95 op verse validation/testroutes.

## Wat nog niet bewezen is

- een fysiek INT8-trunkformaat en de echte trunk-/attention-/routerkernels;
- echte KV-cachemutatie en sequentiële autoregressieve routefeedback;
- embeddings, normalization, LM-head en sampling in dezelfde runtime;
- volledige end-to-end modelkwaliteit vanuit de fysieke bankdecoder;
- volledige-model tok/s en thermische steady-state over een lange decode.

De volgende legitieme fase is daarom geen nieuwe expertcompressievariant, maar
een **full-model decode-integratie** die de bewezen expert-dataplane ongewijzigd
behoudt en uitsluitend de ontbrekende trunk/attention/KV/head-stappen toevoegt.

## Addendum na P3A: causale overlap en fysieke trunk

Dit addendum vervangt de eerdere passage dat fysiek trunkformaat,
router/trunkkernels en LM-head nog volledig onbewezen waren.

P4A testte causale same-layer overlap zonder toekomstige route-informatie.
Serial en async hadden exact dezelfde LRU-misses, fysieke recordbytes en
bit-identieke eindstates:

- validation async mean/p95: **19.772/22.641 ms**;
- once-only test async mean/p95: **20.374/23.185 ms**;
- testspeedup: **1.402x**;
- de preregistreerde 20.0-ms meanpoort faalde nipt met 0.374 ms; alle andere
  p95-, domein-, byte-, correctheids- en residentiepoorten slaagden.

De aangeleverde launchanalyse was niet correct: P3A had al vier launches per
laag, dus 192 per token en niet 768. CUDA Graph gaf 17.243 versus 17.113 ms p50
en faalde de 10%-winstpoort. Vroeg rendez-vous, seriele gate/up-fusie en een
pinned metadatarecord faalden eveneens op afzonderlijke validations; hun tests
bleven dicht. Post-P3A-verificatie: **24/24**.

P5A bouwde vervolgens alle echte INT8 q/k/v/o-projecties, alle 48 routers en de
volledige LM-head:

- 1,229,717,504 codes en 9,607,168 BF16-schalen;
- 241 records, **1,248,931,840 bytes (1.163 GiB)**;
- 15/15 volledig herberekende bankrecords, inclusief de LM-head;
- co-resident met de 4,977,623,040-byte expertcache en 402,653,184-byte KV;
- vrije VRAM: **753,926,144 bytes**;
- validation mean/p95: **15.235/17.531 ms**;
- once-only test mean/p95: **15.360/17.029 ms**;
- onafhankelijke kernelaudit: **16/16**.

De gemeten expert- en projectiemeans tellen op tot **35.734 ms**, een
componentbudget van 27.98 tok/s met nog 64.266 ms tot de 10-tok/s-grens. Dit is
geen full-modelmeting: attention score/softmax/value-reductie, RoPE, echte
KV-mutatie, norms/residuals, gewogen routing, embeddingfetch, sampling en
autoregressieve feedback blijven open.

Het geconsolideerde bewijs en de correctie op de andere ontleding staan in
`reports/streamq5_moe/DATAPLANE_ONTLEDING_VERDICT_2026-08-12.md`.

## Definitief addendum: P6B strikte end-to-end-replicatie

De secties hierboven bewaren de bewijsstand tot en met P5A. De toenmalige lijst
“Wat nog niet bewezen is” is met P6B voor de lokale claim afgesloten; alleen de
expliciete generalisatiegrenzen hieronder blijven open.

Tijdens integratie bleek de P5A-codekeuze niet letterlijk P0C-identiek: in een
vooraf geselecteerde sample verschilde 4,60694% van de codes doordat P5A vóór
codekeuze de schaal al naar BF16 afrondde. P6 bouwde daarom een nieuwe immutable
bank met P0C's exacte FP32-codekeuze, 242 matrixrecords, 193 originele
BF16-normrecords en 1.564.958.720 totale Q8-bytes inclusief hostembedding. De
onafhankelijke bankcontrole slaagde volledig.

P6A voerde de complete decoder uit, maar startte de stopwatch direct na de
fysieke hostembeddinglookup. P6B werd vóór nieuwe outputs gesloten met exact één
wijziging: de stopwatch start vóór embedding. Alle inhoudelijke outputs van P6A
en P6B bleven exact gelijk.

| fase | kwaliteit | mean / p95 | throughput |
|---|---:|---:|---:|
| validation, 1.270 labels | +1,782862% CE | 50,112 / 59,352 ms | 19,955 tok/s |
| test, 1.270 labels | **+0,048026% CE** | **49,927 / 58,187 ms** | **20,029 tok/s** |
| rollout, 512 tokens | vaste greedy feedback | **63,024 / 74,936 ms** | **15,867 tok/s** |

De rollout-p99/max waren 80,519/94,918 ms. De volledige Q5-expertbank stond
pinned in host-RAM; 4.977.623.040 bytes expertcache, 1.248.931.840 bytes
Q8-trunk/head en 402.653.184 bytes KV stonden gelijktijdig op de GPU, met
751.828.992 bytes vrij na vaste allocaties. Alle routes hadden acht unieke
experts, de missbyteboekhouding was exact en alle 85.824 verwachte
laag/positie-KV-writes vonden plaats.

Definitieve status: **`p6b_strict_end_to_end_eureka_pass`**, onafhankelijk
geverifieerd met **120/120** controles. Niet bewezen zijn andere hardware,
batch>1, context>4.096, andere modellen, urenlange thermische steady-state of
universele kwaliteit. Het volledige rapport staat in
`reports/streamq5_moe/P6B_STRICT_END_TO_END_FINAL_REPORT_2026-08-12.md`.
