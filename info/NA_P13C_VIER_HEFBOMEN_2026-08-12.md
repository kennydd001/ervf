# Na P13C: vier hefbomen, waarvan de grootste uit jullie eigen mislukking komt

**Datum:** 2026-08-12 · **Basis:** P13C 69,862 ms / 14,235 tok/s @4K onder 32 GiB
**Type:** heranalyse van gemeten data · **Status:** afleiding, geen nieuwe run

---

## 0. Mijn vier voorspellingen, gescoord

| Voorspelling | Uitkomst |
|---|---|
| **H-C contextmuur is de grootste blinde vlek** | **JUIST, en veel sterker dan ik dacht.** Ik schatte 2,64 ms attentie @4K op roofline; gemeten was de originele attentieplane **96,626 ms** |
| **H-E exactheidsbewarende transformatie als methode** | **BEVESTIGD.** EVT-PM is precies dat: 0 bitverschillen, 7,47× sneller |
| **De +0,048% CE is een steekproeftrekking** | **JUIST.** P16A op 12.700 labels: **+1,4517%**, 95%-BI [+1,1542, +1,7619] |
| **H-A: de glue is 45% en het grootste doel** | **PREMISSE ONJUIST.** Mijn residu-term "glue" bij ctx=128 was grotendeels attentie |
| **CPU-misscompute is een hefboom** | **GEFALSIFICEERD.** P11A: CPU p50 7,447 ms vs GPU all-cold 1,122 ms = 6,64× trager |

### De les uit de glue-fout

Ik rekende `G = totaal − E − P − T = 14,961 ms` bij ctx=128 en noemde dat
launch-overhead. De P7A-launchmeting (7,05 µs × ~772 = 5,44 ms) leek dat te
bevestigen. Maar de attentie zat óók in dat residu, en die is
**contextafhankelijk**: bij ctx=128 is ze verwaarloosbaar, bij 4K is ze 96,6 ms.

> **Een residu-term nooit een naam geven zonder hem apart te meten.** Ik heb een
> restpost een oorzaak toegedicht en er een hypothese op gebouwd. De juiste
> volgorde was: eerst de attentie isoleren, dán het residu benoemen.

Dat P8C hem administratief `superseded` heeft verklaard in plaats van hem toch
te draaien, is correct.

---

## 1. De ontleding van P13C

| Component | ms | % |
|---|---:|---:|
| **venster-penalty** (mapped → pinned → H2D, 8 vensters) | **24,12** | **34,5%** |
| attentie EVT-PM @4K | 12,93 | 18,5% |
| rest (norms, RoPE, router, residuals, KV-write, sampling) | 16,37 | 23,4% |
| projecties Q8 | 8,82 | 12,6% |
| experts Q5 ERVF | 7,61 | 10,9% |

De venster-penalty is afgeleid: P7C (63 GiB, volledig gepinned, ctx 128) is
33,208 ms; P13C is 69,862; de attentiegroei van ctx 128 → 4K is ~12,5 ms; er
blijft 24,1 ms over, en de enige structurele wijziging is dat de bank niet meer
volledig gepinned is.

**Onder 32 GiB is transfer opnieuw de dominante term.** Het project heeft nu twee
configuraties met tegengestelde bottlenecks — dat is op zichzelf een resultaat.

---

## 2. Hefboom 1 — vensterdiepte (grootste, goedkoopste)

Acht pinned vensters van 3.035.136 B = 24.281.088 B totaal.

```
kosten per miss:  341 µs onder 32 GiB  vs  25,7 µs volledig gepinned  = 13×
de keten:         memcpy 63 µs (@48,1 GB/s gemeten STREAM) + H2D 115 µs = 178 µs serieel
```

Bij 1,47 miss per laag en acht vensters is er nauwelijks pipeline-diepte. Er is
geen reden voor acht: **64 vensters is 194 MB pinned** — verwaarloosbaar binnen
32 GiB.

Belangrijker nog: een **prefetch-thread** die op de routeruitkomst van laag ℓ al
stageert haalt de memcpy volledig van het kritieke pad. Dan blijft alleen de H2D
over, en die was in P7C al causaal overlapt.

**Verwacht: 69,86 → 50,6 ms → 19,8 tok/s.** Kosten: enkele dagen.

---

## 3. Hefboom 2 — P9B fysiek maken, en de bank halveert

Dit is de belangrijkste regel in dit document.

**Wat er gemeten is.** P9B: vrije 384-van-768-kanaalselectie passeert full-depth
kwaliteit (+1,478% validation, −0,478% test). P9C: dicht hergroeperen naar nieuwe
Q5-groepen faalt met **+48,03% CE**.

**De diagnose.** `down` is `[2048, 768]` met Q5-groepen van 128 **langs de
reductiedimensie**. Kanalen wegnemen breekt die groepen; hergroeperen geeft nieuwe
schalen over andere kanaalcombinaties. Vandaar de catastrofe. Jullie conclusie —
"kanaalselectie en quantisatiegroepindeling zijn niet onafhankelijk" — is exact
juist en is een echte, publiceerbare bevinding.

**De oplossing: hergroepeer niet.**

Bewaar de oorspronkelijke groepsidentiteit. Groep *g* houdt zijn eigen
BF16-schaal en zijn overlevende *k_g* codes. Groepen worden variabel lang:

| overlevende kanalen | bits per groep | fractie |
|---:|---:|---:|
| 128 van 128 | 5×128 + 16 = 656 | 100% |
| 64 van 128 | 5×64 + 16 = 336 | **51,2%** |

Het masker is gedeeld over alle 2048 outputrijen, dus de layout blijft uniform:
één variabele stride per groep plus een 768-bits masker per expert (96 B).
**Bit-exact, want geen enkele code of schaal verandert.** Precies dezelfde
bewijsvorm als ERVF en EVT-PM.

**Drie gevolgen:**

1. expertbank **17,367 → 8,684 GiB**
2. E **7,614 → 3,807 ms**
3. **en de grote: 8,68 GiB + 1,56 GiB Q8 past volledig gepinned onder 32 GiB.**
   De hele vensterketen wordt overbodig en de 24,1 ms penalty verdwijnt.

Hefboom 2 maakt hefboom 1 dus overbodig in plaats van er bovenop te komen.

---

## 4. Hefboom 3 — attentie staat op 8,1% van de bandbreedte

| | doorvoer | % van piek |
|---|---:|---:|
| originele attentieplane @4K | 4,17 GB/s | 1,1% |
| EVT-PM @4K | 31,14 GB/s | **8,1%** |
| raw scan (P7A) | 361,32 GB/s | 94% |

EVT-PM is een uitstekende 7,47× — maar hij vertrekt van 1,1%. FlashDecoding-
kernels halen routineus 60–80%.

| doel | attentietijd @4K | besparing |
|---:|---:|---:|
| 30% van piek | 3,50 ms | 9,43 ms |
| 40% | 2,62 ms | 10,31 ms |
| 60% | 1,75 ms | 11,18 ms |

**Vermoedelijke oorzaak.** De naam zegt het: *Probability Materialization*
schrijft de softmax-probabilities weg en leest ze terug, dus K en V worden in
gescheiden passes gelezen. Een **twee-pass exacte variant** — pass 1 scores plus
de exacte som in vaste reductievolgorde, pass 2 de V-reductie — leest K en V
precies eenmaal en blijft bit-exact. Daarnaast: KV-layout (contigu per kop) en
gevectoriseerde loads, dezelfde twee dingen die ERVF bij de GEMV opleverden.

---

## 5. Hefboom 4 — de iGPU is nooit als rekeneenheid getest

`p11b` enumereerde vier devices. In diezelfde microbenchmark, batch 8:

| device | latency |
|---|---:|
| **GPU.0 — Intel Arc Pro 140T iGPU** | **1,315 ms** |
| NPU — Intel AI Boost | 2,149 ms |
| CPU — Core Ultra 9 285H | 2,371 ms |

**De iGPU won, maar was niet het onderwerp van de test.** 75,5 MB FP16 in
1,315 ms = **57,4 GB/s**. De CPU-Q5-kernel uit P11A haalde 3,3 GB/s effectief —
18× slechter. P11A falsificeerde dus niet "de CPU-bandbreedte", maar één
CPU-kernel.

**Waarom dit er nu toe doet.** De iGPU leest de gemapte bank rechtstreeks uit
DDR5, **zonder PCIe**. Onder de 32-GiB-configuratie kost een miss nu 341 µs; via
de iGPU zou dat 3,035 MB / 57,4 GB/s = **53 µs** zijn — **6,5× sneller**, en het
resultaat dat naar de dGPU moet is 2048 floats = 8 KB.

Dit is geen vervanging van hefboom 2, maar het is de enige route die transfer
volledig van PCIe afhaalt, en hij is met OpenVINO/SYCL/Level Zero bereikbaar op
deze machine. Dat er een tweede GPU met 32 GB gedeeld geheugen in de laptop zit
is in de hele campagne nooit als rekencapaciteit behandeld.

---

## 6. De ladder op de 32-GiB/4K-configuratie

| Stap | totaal | tok/s |
|---|---:|---:|
| gemeten P13C | 69,86 ms | **14,24** |
| + 64 vensters + prefetch-thread | 50,56 ms | 19,8 |
| + groepsbehoudende 50% pruning (bank 8,68 GiB, volledig gepinned) | 41,93 ms | 23,9 |
| + attentie naar 40% van piek | 31,62 ms | 31,6 |
| + attentie naar 60% van piek | 30,75 ms | **32,5** |

Alle vier zijn bit-exact of kwaliteitsgevalideerd. Geen van vier vraagt nieuwe
kwaliteitsevaluatie behalve de pruning, en die is in P9B al full-depth
gevalideerd.

---

## 7. Vier voorgestelde preregistraties

### P18A · Groepsbehoudende compacte pruningbank (prioriteit 1)

**Hypothese.** Een fysieke bank die de P9B-maskering toepast **zonder
hergroepering** — elke Q5-groep behoudt zijn oorspronkelijke BF16-schaal en zijn
overlevende codes, met een variabele stride per groep — reproduceert de
P9B-logische-nulmaskeroutput **bit-exact**, halveert de bank tot 8,68 GiB en
halveert de expertkerneltijd.

**Poorten.** (1) 0 bitverschillen tegen de P9B-nulmaskervariant over alle
expertoutputs; (2) bank ≤ 9,0 GiB; (3) volledige bank + Q8-trunk pinbaar onder
een 32-GiB Job Object; (4) E ≤ 4,2 ms geïsoleerd; (5) CE identiek aan P9B.

**Waarom dit eerst.** Het is de enige stap die twee termen tegelijk wegneemt: E
halveert én de 24,1 ms venster-penalty verdwijnt omdat de bank weer volledig
gepinned past. En de kwaliteitskant is al gedaan.

**Val om te vermijden.** Als het masker per expert verschilt, verschilt de stride
per expert — dat is prima, maar het offsettabel moet in de kernel als constante
per (laag, expert) beschikbaar zijn, niet per aanroep berekend.

### P18B · Twee-pass exacte attentie

**Hypothese.** Een variant die K en V elk precies eenmaal leest — pass 1 scores
plus exacte som in de vastgelegde reductievolgorde, pass 2 de V-reductie — is
bit-exact tegen EVT-PM en haalt ≥ 30% van de piekbandbreedte.

**Poorten.** 0 bitverschillen op scores én attentieoutputs bij ctx 128, 512, 1024
en 4096 (dezelfde eis die P13B haalde); attentieplane ≤ 4,4 ms @4K.
**Voorspelling: 3,5 ms bij 30% van piek.**

### P18C · iGPU-misscompute

**Hypothese.** Een Level-Zero/SYCL Q5-expertkernel op de Arc Pro 140T, lezend uit
de gemapte bank, produceert bit-identieke BF16-expertoutputs en haalt onder de
32-GiB-configuratie een lagere kosten-per-miss dan de huidige
mapped→venster→H2D-keten.

**Poorten.** (1) bit-identiek aan de dGPU-Q5-kernel op dezelfde 72 gelockte
matrices; (2) kosten per miss ≤ 150 µs (nu 341); (3) geen PCIe-verkeer voor
expertgewichten. **Claimgrens:** dit vervangt hefboom 2 niet, het is een tweede,
onafhankelijke route.

**Eerlijke waarschuwing.** De iGPU deelt de DDR5-bus met de CPU en met de
staging van de dGPU. Contentie moet apart gemeten worden; het 57,4-GB/s-getal
komt uit een microbenchmark zonder concurrent verkeer.

### P18D · De configuratie-fasegrens als resultaat

**Hypothese.** tok/s als functie van het hostgeheugenbudget vertoont een scherpe
knik op het punt waar de expertbank niet meer volledig pinbaar is.

**Meting.** Draai dezelfde runtime onder Job-Object-limieten van 20, 24, 28, 32,
40, 48 en 63 GiB, met de bank op 17,37 GiB en (na P18A) op 8,68 GiB. Rapporteer
tok/s, kosten per miss en peak commit.

**Waarom dit waardevol is.** Het is de enige meting die de twee tegengestelde
kostenstructuren in één curve vat, en hij verklaart alle eerdere verdicten in het
project in één beeld. Voor een publicatie is dat sterker dan nog een
optimalisatie: het maakt van "wij hebben dit op onze laptop gehaald" een
uitspraak over wanneer offloading welk regime binnengaat.

---

## 8. Wat ik niet zou doen

- **De volledige fysieke DeepSeek-replicatie eerst.** P14A heeft de kwaliteit al
  aangetoond (+0,716% / +1,493%). De fysieke replicatie kost twee weken en meet
  vooral opnieuw wat P18A/P18B op Qwen goedkoper uitwijzen. Doe hem ná P18A, want
  dan repliceer je een halve bank in plaats van een hele.
- **Nog een quantisatievariant.** P9A mixed Q4/Q5 levert 5% bytes; P18A levert
  50% en is bit-exact.
- **De CPU nog eens proberen.** P11A is een eerlijk negatief resultaat. De iGPU
  is een andere vraag, geen herhaling.
- **Een bredere nieuwheidsclaim.** `breakthrough_claim_allowed: false` is de juiste
  status en jullie eigen P17-lijst van vijf ontbrekende voorwaarden klopt.

---

## 9. Waar het project nu staat

Drie bit-exacte systeemtransformaties op rij — ERVF (2,39× experts), EVT-PM
(7,47× attentie), Pinned-Window Streaming (32-GiB-capaciteit) — elk met nul
bitverschillen bewezen. Plus een kwaliteitsclaim die nu op 12.700 labels staat
met een 95%-BI dat de 2%-poort haalt, en een replicatie op een tweede
MoE-familie.

De rode draad is inmiddels duidelijk genoeg om als methode te formuleren:

> **Elke winst in dit project kwam uit een exactheidsbewarende transformatie —
> een herschikking van geometrie, layout of staging die de numerieke semantiek
> aantoonbaar niet verandert. Geen enkele kwam uit een kwaliteitsoffer.**

Dat is de zeldzame kant. Het maakt prestatiewerk goedkoop, want elke stap wordt
geverifieerd met een bitvergelijking in plaats van een volledige
kwaliteitsherhaling. P18A is de vierde in die reeks en de eerste die de
representatie zelf halveert zonder één bit te veranderen.
