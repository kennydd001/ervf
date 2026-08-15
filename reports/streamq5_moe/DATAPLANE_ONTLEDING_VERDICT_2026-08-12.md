# Verdict over `DATAPLANE_ONTLEDING_2026-08-12.md`

Datum: 2026-08-12. Status: fysiek getest, splits bewaakt en onafhankelijk
geverifieerd.

## Uitkomst in één zin

De ontleding vond de juiste hoofdhefboom—causale H2D/compute-overlap—maar
overschatte launchoverhead en claimde ten onrechte dat transfer volledig
verdwijnt. De losse componentmetingen waren 20,374 ms voor de expertplane en
15,360 ms voor de projectieplane. P6B heeft de ontbrekende integratie inmiddels
streng end-to-end gemeten: 49,927 ms mean op de test en 63,024 ms mean over een
512-token autoregressieve rollout, met hostembedding binnen de stopwatch.

## Wat overeind blijft

De seriële tweetermenregressie is sterk. In de nieuwe gelijktijdige P4A-control
op de eenmalige test is:

```text
serial wall_ms = 19,0930 + 0,134019 × misses     R² = 0,817
```

Dat reproduceert de eerder afgeleide missprijs van 0,133401 ms vrijwel exact.
De externe meanvoorspelling van 19,06 ms voor async kwam uit op 20,374 ms,
6,90% afwijking en dus binnen de vooraf toegestane 15%-band.

P4A gebruikte geen toekomstinformatie: alleen misskopieën van de huidige laag
gingen op stream 2, terwijl de hits van diezelfde laag werden berekend. Alle
fysieke recordbytes, LRU-misses en eindstates waren exact gelijk aan serial.

| split | serial mean / p95 | causal async mean / p95 | speedup |
|---|---:|---:|---:|
| validation | 26,865 / 38,124 ms | **19,772 / 22,641 ms** | 1,359× |
| test | 28,563 / 39,823 ms | **20,374 / 23,185 ms** | **1,402×** |

De vooraf gekozen testpoort was mean <= 20,0 ms. P4A mist die eerlijk met
0,374 ms (1,87%) en heeft daarom status `closed`; p95 en alle domein-, byte-,
correctheids- en residentiepoorten slaagden. De grens is niet achteraf verlegd.

## Wat de ontleding onjuist voorspelde

### 1. Niet 768 maar 192 launches

P3A groepeerde de acht experts al in één gate/up-kernel en één down-kernel.
Samen met SwiGLU en reductie zijn dat vier launches per laag, dus
48 × 4 = **192 launches/token**. P3B verving exact die keten door CUDA Graph:

| pad | validation p50 |
|---|---:|
| eager | 17,113 ms |
| graph | 17,243 ms |
| ratio | **1,0076** |

De no-opcontrolereeks werd wel circa 9× goedkoper. Dispatch kan dus verdwijnen,
maar de echte Q5-kerneltijd niet. De preregisterde >=10%-winst faalde en de
test bleef dicht.

### 2. Transfer verdwijnt niet volledig

Na P4A-overlap is de testregressie:

```text
async wall_ms = 18,5601 + 0,025672 × misses      R² = 0,444
```

De misshelling daalt 80,8%, maar is niet nul. Kopie/compute concurreren om
resources, sommige lagen hebben onvoldoende hits en de gesplitste keten voegt
launches toe. De async staart krimpt sterk (p95 39,823 → 23,185 ms), maar
`p95=max=intercept` wordt niet waargenomen.

### 3. De voorgestelde “gegroepeerde GEMV” bestond grotendeels al

Drie vervolgvarianten zijn op afzonderlijke routecaptures gefalsificeerd:

| variant | validation mean / p95 | reden |
|---|---:|---|
| P4B: rendezvous na gate/up | 26,199 / 36,570 ms | overlapvenster te kort |
| P4C: gate+up+SwiGLU per blok | 24,766 / 33,177 ms | gate/up-parallelisme verloren |
| P4D: één pinned metadatarecord | 24,443 / 31,614 ms | mini-DMA-overhead groter dan vier kleine updates |

Alle varianten waren bit-exact en fysiek; hun tests bleven na validation-falen
ongeopend. De beste implementatie blijft P4A.

### 4. Lookahead-probe is nog geen gratis vervolgstap

De bestaande routecaptures bevatten alleen top-8 expert-ID's, geen
2048-dimensionale hidden states of routerlogits. Een lineaire probe kan dus
niet uit die bestanden worden getraind. Bovendien is laag `l+1` causaal
afhankelijk van de expertuitvoer en attentiontransformatie van laag `l`;
lookahead is speculatie en recall alleen is onvoldoende—onnodige prefetchbytes
en cachevervuiling moeten eveneens worden gemeten. Deze hypothese blijft open,
niet bewezen.

### 5. De exacte embeddingwinst is 96 slots in deze fysieke layout

P5A verving de placeholder van 1.541.093.376 bytes door een echte trunkbank
zonder embedding van 1.248.931.840 bytes. Dat maakt 292.161.536 bytes vrij,
oftewel **96 volledige expertrecords**, niet 102. De gemeten vrije VRAM steeg
van 462.422.016 naar 753.926.144 bytes.

## Nieuwe eigen hypothese: de echte trunkterm

P5A bouwde alle werkelijke INT8 projectiematrices:

- 48 × q/k/v/o attentionprojecties;
- 48 routers;
- volledige 151.936 × 2.048 LM-head;
- 1.229.717.504 codes + 9.607.168 BF16-schalen;
- 241 records, 1.248.931.840 bytes (1,163 GiB).

De onafhankelijke bankaudit herberekende 15/15 records, 350.486.528 gewichten,
inclusief de volledige LM-head: nul code-, schaal- of hashverschillen.

De fysieke Q8-kernel liep co-resident naast de 4.977.623.040-byte expertcache
en 402.653.184-byte KV-reservering, met 753.926.144 bytes vrij:

| split | host mean | host p95 | p50 throughput |
|---|---:|---:|---:|
| validation (120) | 15,235 ms | 17,531 ms | 83,85 Gweight/s |
| test (360) | **15,360 ms** | **17,029 ms** | **81,12 Gweight/s** |

Alle P5A-poorten slaagden; onafhankelijke verificatie: 16/16 checks.

## Wat nu werkelijk bewezen is

| component | test mean | test p95 | status |
|---|---:|---:|---|
| causal fysieke expertplane | 20,374 ms | 23,185 ms | 20-ms meanpoort nipt fail; verder correct |
| fysieke INT8 q/k/v/o/router/head-GEMV's | 15,360 ms | 17,029 ms | pass |
| som van gemeten componenten | **35,734 ms** | **<=40,215 ms** | componentbudget |

De meansom correspondeert met 27,98 componenttokens/s en laat **64,266 ms**
over tot de 100-ms/10-tok/s-grens. Die marge is reëel maar geen tok/s-claim,
omdat de twee componenten nog niet in één echte decoderloop zijn geïntegreerd.

## Open bewijs vóór “full-model Eureka”

Nog fysiek te implementeren en gezamenlijk te meten:

1. RMSNorm, q/k-normalisatie, RoPE en residuals;
2. KV-write en de echte 4k-KV-read;
3. attention score, causal mask, softmax en value-reductie;
4. echte routertop-k, routergewichten en gewogen expertreductie;
5. host-embeddingfetch, finale norm, sampling;
6. autoregressieve routefeedback over ten minste 512 gegenereerde tokens;
7. circa 10× grotere kwaliteitsset met gepaarde block-bootstrap.

Tot die integratie is het correcte verdict:

> **Eureka voor de fysieke expert- en projectieplanes; full-model decode nog
> niet bewezen.** De 10-tok/s-hypothese heeft nu een gemeten componentmarge,
> geen loutere rooflineprojectie.

## Verificatie

- post-P3A/P4A-D: 24/24 onafhankelijke checks;
- P5A bank: 15/15 volledige recordherberekeningen;
- P5A kernel: 16/16 onafhankelijke checks;
- mislukte P3B/P4B/P4C/P4D-runs en gesloten tests zijn bewaard.

## Definitieve P6B-resolutie van de open punten

De eerdere sectie “Open bewijs” bewaart de stand vóór P6. P6/P6B heeft punten
1–6 fysiek geïmplementeerd en gezamenlijk gemeten. Een circa tienmaal grotere
bootstrap blijft een zinvolle generalisatiestudie, maar was geen poort in de
vooraf gesloten lokale P6B-claim.

De runtime omvat Q8-hostembedding, alle RMSNorms, q/k/v/o, q/k-normalisatie,
RoPE, causale GQA, echte BF16-KV-mutatie, live routertop-8,
BF16-routegewichten, fysieke Q5-cachemisses, gewogen expertreductie, residuals,
finale norm, volledige LM-head, argmax en autoregressieve feedback.

| fase | relatieve CE | mean | p95 | tok/s |
|---|---:|---:|---:|---:|
| validation | +1,782862% | 50,112 ms | 59,352 ms | 19,955 |
| test | +0,048026% | 49,927 ms | 58,187 ms | 20,029 |
| 512-tokenrollout | n.v.t. | 63,024 ms | 74,936 ms | 15,867 |

De P5A-bank bleek niet exact P0C-semantisch in zijn codekeuze; P6 verving hem
door een nieuwe, onafhankelijk geverifieerde bank. P6A sloot embeddinglookup
per ongeluk uit van timing; de vooraf gesloten P6B-replicatie corrigeerde alleen
die stopwatchgrens en reproduceerde alle inhoudelijke outputs exact.

> **Actueel verdict: lokale strikte full-model decode-Eureka bewezen.** Test:
> +0,0480% CE en 49,927/58,187 ms mean/p95. Rollout: 512 tokens en 15,867 tok/s.
> Onafhankelijke eindcontrole: 120/120. Geen claim voor andere hardware,
> batch>1, context>4.096 of universele kwaliteit.

Aanvullende verificatie:

- P6 exacte bank: 16 geselecteerde Q8-records en 193/193 normen exact;
- P6B strict end-to-end: 120/120 onafhankelijke checks;
- P6A/P6B: identieke CE, missreeksen, predictionhashes en 512 rollouttokens.
