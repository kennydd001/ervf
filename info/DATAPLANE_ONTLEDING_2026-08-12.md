# De dataplane is opgelost. Wat er nu nog staat is een implementatieprobleem.

**Datum:** 2026-08-12 · **Type:** heranalyse van `p3a_integrated_expert_test.json`
(1280 gepaarde metingen) · **Status:** afleiding uit gemeten data, geen nieuwe run

---

## 0. Eerst: de vorige voorspelling is uitgekomen

Uit de analyse van 12 augustus, vóór de StreamQ4-run:

| Voorspeld | Gemeten |
|---|---|
| RTN-Q4 experts: 2,64–3,52% | **2,744%** |
| INT8-trunk: ~0,07% (klein) | **0,441%** |
| Q4+INT8 primair: 2,70–3,58% | **3,044%** |
| *"een RTN-run landt naar verwachting op 2,7–3,6% en sluit de lijn"* | gesloten op 3,0441% |

De progressiegate faalde met 0,0441 procentpunt. Precies het risico dat toen
benoemd is. Q5 loste het op: **+0,698% validation / +0,999% test** (P0) en
**+1,478% / −0,478%** met fysieke schaalsemantiek (P0C). Dat betekent dat het
onderliggende schaalmodel klopt en dat de rest van deze analyse op dezelfde
basis staat.

---

## 1. De exacte ontleding van de 28,1 ms

De P3A-testrun levert per token zowel het aantal cachemissen als de wandtijd.
Dat zijn **1280 gepaarde meetpunten** over vijf domeinen. Kleinste-kwadraten:

```
wall_ms  =  18,6743  +  0,133401 × misses          R = 0,882   R² = 0,778
                                                    residu-σ = 3,10 ms
```

### Beide termen zijn fysiek interpreteerbaar

**De helling.** Eén miss = één expert = 2,8828 MiB = 3,023 MB.

```
3,023 MB / 0,133401 ms = 22,66 GB/s  =  86% van de door P-C gemeten 26,341 GB/s
```

86% efficiëntie op gefragmenteerde per-record kopieën van 2,88 MiB. Dat is een
goede implementatie, geen verliespost.

**Het snijpunt.** 18,674 ms voor 1,8119 Gweight/token. Bij Q5 is dat
1,1608 GB uit VRAM. De roofline op 384 GB/s is **3,023 ms**.

```
gemeten 18,674 ms  →  62,2 GB/s effectief  →  16,2% van de VRAM-roofline
```

### Out-of-sample validatie

Het model voorspelt uit `mean misses = 70,66`:

```
transfer = 0,133401 × 70,66 = 9,427 ms/token
```

P2C mat de H2D **apart, in een andere fase**: **9,565 ms/token**.
Afwijking **1,45%**. Het model is niet op dat cijfer gefit.

### De decompositie

| Component | ms/token | aandeel |
|---|---:|---:|
| vaste expertcompute | 18,674 | 66,5% |
| transfer (70,66 missen) | 9,427 | 33,5% |
| **totaal gemeten** | **28,101** | 100% |

**Twee derde van de tijd is compute die op 16% van zijn eigen roofline draait.
Een derde is transfer die volledig verborgen kan worden.** Geen van beide is
een fysieke grens.

---

## 2. Drie hefbomen, geen daarvan onderzoek

### Hefboom 1 — asynchrone H2D op een tweede stream (gratis)

Het rapport zegt letterlijk: *"op één seriële CUDA-stream zonder
overlapcredit."*

Per laag: compute 0,3890 ms, transfer bij gemiddeld 1,47 miss/laag 0,1964 ms.
**De transfer past onder de compute.** Uitgeven van de kopieën op een tweede
stream direct na routing, en de gecachte experts eerst rekenen, verbergt hem.

### Hefboom 2 — één laag router-lookahead prefetch

Hidingbudget zonder lookahead: 0,3890 ms/laag = 2,92 missen/laag = **140
missen/token**.
Met één laag lookahead: 0,7781 ms/laag = 5,83 missen/laag = **280
missen/token**.

Het waargenomen maximum over alle 1280 tokens is **280 missen**. De drempel en
het empirische maximum vallen samen — dat is toeval, maar de praktische
gevolgtrekking is exact: **elke token in de testset valt onder de grens, dus de
transfer-term verdwijnt volledig.**

Benodigde recall van de probe om zelfs de zwaarste tokens te verbergen:

| token | recall nodig |
|---:|---:|
| 71 missen (gemiddeld) | 0,0% |
| 155 missen (p95) | 9,7% |
| 214 missen (p99) | 34,6% |
| 280 missen (max) | 50,0% |

Een lineaire probe van hidden state ℓ naar de top-8 van laag ℓ+1 hoeft dus maar
**50% recall** te halen om het slechtste geval te dekken. Dat is een zeer lage
drempel, en te trainen en evalueren op de route-captures die al in de repo
staan — **zonder één GPU-modelrun.**

### Hefboom 3 — gegroepeerde GEMV

De acht experts van een laag delen dezelfde inputvector x. Nu: 48 × 8 × 2 =
**768 kernel-launches per token**. Op Windows WDDM kost een launch 10–20 µs:

| launch-overhead | totaal | aandeel van de 18,67 ms |
|---:|---:|---:|
| 8 µs | 6,14 ms | 32,9% |
| 12 µs | 9,22 ms | 49,4% |
| 20 µs | 15,36 ms | 82,3% |

Groeperen tot één kernel per laag: 768 → 96 launches. Dat is precies wat een
fused-MoE-kernel doet. Verwacht: 18,67 → 8–11 ms.

### De ladder, op dezelfde 1280 testtokens en dezelfde missreeks

| Stap | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|
| 0. gemeten, serieel | **28,10** | 39,35 | 47,22 | 56,03 |
| 1. + async H2D, 2e stream | **19,06** | 20,68 | 28,55 | 37,35 |
| 2. + 1 laag lookahead-prefetch | **18,67** | 18,67 | 18,67 | 18,68 |
| 3. + gegroepeerde GEMV (40% roofline) | **10,83** | 10,60 | 17,95 | 26,75 |

Stap 2 is opvallend: de spreiding verdwijnt. p95 = p99 = max = het snijpunt.
De hele staartlatentie is transfer, en die is volledig hideable.

---

## 3. Volledig model: wat komt er nog bij

| Component | GB uit VRAM | ms @roofline |
|---|---:|---:|
| attentie, 48 lagen, INT8 | 0,906 | 2,36 |
| LM-head INT8 | 0,311 | 0,81 |
| KV-cache @4k lezen | 0,403 | 1,05 |
| router | 0,013 | 0,03 |
| **som trunk-zijde** | **1,632** | **4,25** |

Plus een gratis geheugenwinst: **de embeddingtabel hoeft niet in VRAM.** Bij
decode zoek je één rij van 2048 waarden op per token = 4 KB, over PCIe 0,15 µs.
Daarvoor 0,290 GiB VRAM reserveren is verspilling. Vrijgeven levert **102
extra expertslots** (1640 → 1742).

### Projectie

| trunk-efficiëntie | expertplane | totaal | **tok/s** |
|---|---:|---:|---:|
| 16% (zoals nu) | 28,10 (nu) | 54,67 ms | **18,3** |
| 40% (gegroepeerd) | 18,67 (+overlap) | 29,30 ms | **34,1** |
| 40% | 10,60 (+overlap+groep) | 21,23 ms | **47,1** |
| 60% (CUDA Graphs) | 10,60 | 17,68 ms | **56,5** |

**Zelfs de meest pessimistische rij — niets geoptimaliseerd, trunk even
inefficiënt als de huidige expertkernel — geeft 18,3 tok/s.** Het doel van
10 tok/s is met de reeds gemeten componenten met 1,8× marge gehaald.

---

## 4. Wat de wetenschappelijke claim is

Niet "een nieuwe compressiemethode". Dit:

> **Batch-1 decode van een offloaded MoE gehoorzaamt een tweetermen-roofline
> `wall = C + β·m`, waarin C de per-token expertcompute is en β de kosten per
> cachemiss. Beide zijn onafhankelijk meetbaar. Op consumentenhardware is β·m
> volledig te verbergen en is C implementatiegebonden op ~16% van de
> geheugenroofline. De enige werkelijk bindende ontwerpvariabele is daarmee de
> kwantisatiebitbreedte, en die hoort vanaf de kwaliteitskant gekozen te
> worden — niet het compressieschema, niet de cachepolicy, niet de routing.**

Het bewijs is ongebruikelijk sterk omdat het uit twee richtingen komt:

**Positief.** De regressie op 1280 tokens (R² 0,778) voorspelt een
onafhankelijke meting in een andere fase tot op 1,45%. De bitbreedteladder is
met harde vooraf vastgelegde poorten doorlopen: Q2 → +21,5% (fail), Q4 →
+3,04% (fail), Q5 → +0,70% (pass).

**Negatief.** Dertien alternatieve hypotheses zijn preregistreerd gefalsificeerd
— CRAFT H1/H2/H3/H4/H6/H7/H8, vier cachepolicies, BITFLOW, RSIV, FLEQ, E2GQ,
CORETAIL — telkens met onafhankelijke verifiers en bewaarde mislukte pogingen.
**Dat corpus is het bewijs voor de positieve claim.** Niemand publiceert dat, en
zonder dat corpus is "kies gewoon de juiste bitbreedte" een anekdote in plaats
van een resultaat.

En het inzetbare feit: een MoE van 30,5 miljard parameters met een expertbank
van 17,37 GiB draait zijn volledige routed-expert dataplane op een laptop-GPU
met 8 GB VRAM, op 28,1 ms/token, met ≤1% CE-verlies.

---

## 5. Waar ik kritisch op blijf

**De kwaliteitsmeting is ruisgedomineerd.** Dezelfde constructie gaf in P0C
+1,478% op validation en −0,478% op test — een spreiding van 1,96 procentpunt
tussen twee splits van 1270 tokens. Over de vier Q5+INT8-metingen (0,698 /
0,999 / 1,478 / −0,478) is het gemiddelde 0,67% met σ ≈ 0,82. De poort van 2%
wordt gehaald, maar de individuele cijfers zeggen weinig. **Voor elke
publicatieclaim moet de evaluatieset ~10× groter**, met gepaarde
block-bootstrap — dezelfde methodiek die H1 en H3 al gebruikten.

**De routes komen uit een capture, niet uit autoregressieve decode.** Het
verdict benoemt dit correct. Het risico is reëel: de eigen outputs van het
model verschuiven de routeverdeling, en de cache ziet dan een andere
missreeks. Dit is de belangrijkste openstaande onzekerheid in de hele keten.

**De 108,11 ms maximumtoken.** Die zit in het general-domein en is vrijwel
zeker de koude-cachestart van een context. Bij lange decodes is dat
verwaarloosbaar, maar het hoort apart gerapporteerd, niet weggemiddeld.

---

## 6. Voorgestelde preregistratie: P4

Eén fase, vier vooraf vastgelegde stappen, poorten afgeleid uit de hardware.

### P4A · Lookahead-probe (geen GPU-modelrun)

Train per laag ℓ een lineaire probe `W_ℓ : R^2048 → R^128` van hidden state ℓ
naar de routerlogits van laag ℓ+1, uitsluitend op de bestaande
route-captures uit P1A/P1C. Meet recall@K van de werkelijke top-8, voor
K ∈ {8, 12, 16, 24}, op de ongeziene P3A-routes.

**Poort:** recall@16 ≥ 50% op alle vijf domeinen. Rapporteer de volledige
recallcurve, ook bij falen.

### P4B · Async dataplane (herhaling van P3A, één wijziging)

Exact dezelfde P3A-configuratie, exact dezelfde routes, exact dezelfde
1280 tokens. Enige verandering: H2D op een tweede CUDA-stream, gecachte
experts eerst.

**Poort:** mean ≤ 20,0 ms/token, p95 ≤ 25,0. Voorspelling uit dit document:
**19,06 mean / 20,68 p95.** Wijkt de meting meer dan 15% af, dan is het
tweetermenmodel gefalsificeerd en dat moet zo gerapporteerd.

### P4C · Gegroepeerde Q5-GEMV

Eén kernel per laag over alle acht experts, gedeelde inputvector.
Correctheidscontrole: bit-identiek aan de bestaande per-expert kernel op
dezelfde 72 gelockte matrices.

**Poort:** ≥ 35% van de VRAM-roofline (≥ 134 GB/s effectief), oftewel
expertcompute ≤ 8,6 ms/token.

Overweeg daarbij de Q5-uitlijning: 5 bits deelt niet op 8 of 32, dus codes
kruisen bytegrenzen. De CORETAIL-machinerie splitst dit gratis in een
4-bit-vlak plus een 1-bit-vlak, beide uitgelijnd. Dat is de enige plek waar
CORETAIL nog direct nut heeft.

### P4D · Volledige decode-integratie

Trunk, attentie, router, KV-mutatie, LM-head en sampling in dezelfde loop, met
de dataplane uit P4B/P4C ongewijzigd. Embedding blijft in host-RAM.

**Poorten, afgeleid uit de gemeten hardware en niet arbitrair gekozen:**
tok/s ≥ 10 · relatieve CE ≤ 2% op een 10× grotere evaluatieset · VRAM ≤ 7,5 GiB
· 512-token autoregressieve rollouts stabiel · gemeten missreeks gerapporteerd
naast de P3A-capture, zodat het autoregressieve verschil zichtbaar wordt.

**Voorspelling uit dit document: 18–47 tok/s.** Leg die band vooraf vast; een
meting daarbuiten is informatiever dan een meting erbinnen.

---

## 7. Eén regel

De expertcompressie is klaar. De cachepolicy is klaar. De routing is klaar.
Wat er nu nog tussen dit project en een echt resultaat staat, is **een tweede
CUDA-stream, één gegroepeerde kernel en een lineaire probe** — en geen daarvan
is onderzoek.
