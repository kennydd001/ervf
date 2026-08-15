# STREAMQ5-MoE P6B — definitief strikt end-to-end-rapport

Datum: 2026-08-12

## Eindbeslissing

**Eureka binnen de vooraf vastgelegde lokale claim: geslaagd en onafhankelijk
geverifieerd.** De volledige custom Qwen3-30B-A3B-dataplane draait fysiek op de
aanwezige 8-GB-GPU met Q5-experts uit pinned host-RAM, een resident Q8-trunk,
een echte BF16-KV-cache, live top-8-routing, gewogen expertreductie en volledige
autoregressieve feedback.

De strikte testsplit behaalde 20,03 tokenstappen/s en een relatieve
next-token-CE-stijging van slechts 0,0480% versus de BF16-teacher. De aansluitende
512-tokenrollout behaalde 15,87 tok/s. Alle vooraf geregistreerde poorten
slaagden; de onafhankelijke eindcontrole slaagde 120/120.

## Waarom P6 de eerdere bewijsgrens doorbreekt

P0C tot en met P5A bewezen kwaliteit, expertbank, cache/H2D, expertcompute en
trunkprojecties nog als afzonderlijke delen. P6 voert nu werkelijk per token en
per laag uit:

1. fysieke Q8-hostembedding en BF16-afronding;
2. input-RMSNorm;
3. Q8 q/k/v-GEMV, q/k-RMSNorm en default Qwen-RoPE;
4. BF16-KV-write, causale 32-query/4-KV GQA-score, FP32-softmax en value-reductie;
5. Q8-o-projectie en residual;
6. post-attention-RMSNorm en Q8-router;
7. FP32-softmax over 128 experts, live top-8 en BF16-routegewichten;
8. causale static/dynamic expertcache, echte Q5-H2D-misses, gate/up/SwiGLU/down;
9. gewogen expertoptelling in referentievolgorde en residual;
10. finale RMSNorm, volledige Q8-LM-head, logsumexp/argmax, sampling en feedback.

Er worden geen toekomstige routes gebruikt. Iedere laag beslist pas na zijn
echte routeroutput welke expertrecords nodig zijn.

## Noodzakelijke semantische correctie

De P5A-bank bleek bij de eindintegratie niet letterlijk P0C-identiek: P5A koos
codes met de reeds naar BF16 afgeronde schaal, terwijl P0C en de geldige
P1D-expertbank codes met de FP32-maxabs-schaal kiezen en alleen de persistente
dequantisatieschaal naar BF16 afronden. In de vooraf geselecteerde P5A-sample
verschilde daardoor 4,60694% van de codes.

P6 herschreef P5A niet, maar bouwde een nieuwe immutable bank met exact de
P0C-volgorde. De onafhankelijke controle leidde zestien grote Q8-records,
inclusief embedding en volledige LM-head, opnieuw af uit het checkpoint en
controleerde alle 193 normrecords. Alles was exact.

| fysieke P6-bank | aantal/bytes |
|---|---:|
| device q/k/v/o/router/head-records | 241 |
| hostembedding-records | 1 |
| matrixgewichten/codes | 1.540.882.432 |
| device Q8-bank | 1.248.931.840 bytes |
| host Q8-embedding | 316.026.880 bytes |
| totale Q8-bank | 1.564.958.720 bytes |
| originele BF16-normbank | 421.888 bytes, 193 records |

## Strikte P6B-metingen

P6A doorliep alle poorten, maar de stopwatch startte direct na de fysieke
hostembeddinglookup. P6B werd daarom vóór nieuwe outputs als een één-regelige
replicatie gesloten: de stopwatch start vóór embeddinglookup; data, kernels,
gewichten, numeriek gedrag, poorten en rollout bleven onveranderd.

| beslissende fase | labels/tokens | kwaliteit vs teacher | mean | p95 | throughput |
|---|---:|---:|---:|---:|---:|
| validation | 1.270 labels | +1,782862% CE | 50,112 ms | 59,352 ms | 19,955 tok/s |
| test | 1.270 labels | **+0,048026% CE** | **49,927 ms** | **58,187 ms** | **20,029 tok/s** |
| greedy rollout | 512 tokens | teacher-forcing n.v.t. | **63,024 ms** | **74,936 ms** | **15,867 tok/s** |

Test-CE was 2,260959 tegenover 2,259874 voor de BF16-teacher. De
rollout-p99/max waren 80,519/94,918 ms en bleven dus zelfs onder de 100-ms
doelgrens, hoewel alleen mean en p95 vooraf beslissend waren.

P6A en P6B reproduceerden exact:

- validation- en test-CE;
- alle cachemissen en voorspellinghashes;
- alle 512 gegenereerde token-ID's;
- rollouthash `15d81584c8633ae6f628e0302787eaa9af89a28066b6a86646e0ad05a76583bb`.

De gegenereerde tekst is coherent en begint met “here, and it's called the AI
Chip.” Tekstkwaliteit was nadrukkelijk geen subjectieve beslispoort; de gesloten
teacher-forced CE-proef beslist kwaliteit.

## Fysieke residentie en runtime-invarianten

| allocatie/controle | gemeten waarde |
|---|---:|
| totale GPU-VRAM | 8.546.484.224 bytes |
| expertcache | 4.977.623.040 bytes |
| resident Q8-trunk/head | 1.248.931.840 bytes |
| 4.096-token BF16-KV | 402.653.184 bytes |
| vrije VRAM na vaste allocaties | 751.828.992 bytes |
| vrije VRAM na scratch | 749.731.840 bytes |
| pinned Q5-expertbank | 18.647.875.584 bytes |
| pinned volledige P6-Q8-bank | 1.564.958.720 bytes |

Over test plus prefill/rollout werden 172.213 echte expertcachemisses en exact
522.689.875.968 missbytes geboekt. Alle routes hadden acht unieke experts; de
grootste afwijking van de som van BF16-routegewichten tot één was 0,002930.
Er waren exact 85.824 laag/positie-KV-writes en alle gecontroleerde contexten
hadden K én V in alle lagen gemuteerd.

## Verificatieketen

- P6-bank: 16 geselecteerde Q8-records plus 193/193 normrecords exact;
- P6B-eindverificatie: **120/120** controles;
- alle bron-, lock-, evaluator-, route-, bank- en outputhashes bewaard;
- validation/test-statistieken opnieuw uit de ruwe tijd-, miss- en
  predictionarrays berekend;
- CE gewogen uit domeinrecords herberekend en tegen de immutable P0C-teacher
  gecontroleerd;
- rollouttokenhash, tokenisatie, decodering en autoregressieve feedbackketen
  opnieuw gecontroleerd;
- P6A/P6B-replicatie exact op alle inhoudelijke outputs.

## Exacte claim en grenzen

Bewezen is dat deze artefacten op deze NVIDIA RTX PRO 2000 Blackwell Generation
Laptop GPU een volledige batch-1 custom decode uitvoeren die op twee vooraf
gesloten 1.270-labelsplits de 2%-kwaliteitsgate haalt en in een aansluitende
512-tokenrollout ruim boven 10 tok/s blijft.

Niet bewezen zijn universele modelkwaliteit, andere hardware, batchgroottes
groter dan één, contexten boven 4.096 tokens, thermisch gedrag over uren, of
productierobuustheid. De domeingeconditioneerde statische cache veronderstelt
een vooraf gekozen domeinklasse; de dynamische LRU zelf gebruikt uitsluitend
causale live routes. Die grenzen verkleinen de claim, maar veranderen de lokale
Eureka-uitkomst niet.

## Reproduceerbare artefacten

- preregistratie: `P6B_STRICT_END_TO_END_REPLICATION_PREREGISTRATION.md`;
- input/evaluatorlocks: `p6b_strict_end_to_end_*_lock.json`;
- smoke/validation/test: `p6b_strict_end_to_end_{smoke,validation,test}.json`;
- onafhankelijke verificatie: `p6b_end_to_end_verification.json`;
- evaluator: `scripts/streamq5_moe/run_p6b_strict_end_to_end_decode.py`;
- verifier: `scripts/streamq5_moe/verify_p6b_end_to_end.py`.
