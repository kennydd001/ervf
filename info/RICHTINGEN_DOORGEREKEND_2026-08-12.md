# Wat nu: elke richting doorgerekend tegen de gemeten kostenverdeling

**Datum:** 2026-08-12 · **Basis:** P6B-meting E 18,560 | P 15,360 | G 14,193 | β·m 1,814 ms
**Hardware:** RTX PRO 2000 Blackwell Laptop (384 GB/s, 7,96 GiB) · Intel Arrow Lake-HX
(Family 6 Model 197), 16 fysieke cores, DDR5-6400 dual channel · PCIe 5.0 ×8 (26,341 GB/s gemeten)

---

## 0. De vraag die alles ordent

Elke richting hieronder wordt beoordeeld op één ding: **welke van de vier gemeten
termen raakt hij, en hoe groot is die term?**

| Term | ms | % |
|---|---:|---:|
| E — expertcompute | 18,560 | 37,2% |
| P — projecties | 15,360 | 30,8% |
| G — glue | 14,193 | 28,4% |
| β·m — transfer | 1,814 | 3,6% |

---

## 1. CPU — de grootste onbeproefde hefboom

**De verrassing.** De GPU-kernel doet één expert in **48,3 µs** (18,560 ms / 384).
De CPU moet daarvoor 3,02 MB uit DDR5 lezen:

| DDR5 haalbaar | CPU per expert | verhouding |
|---:|---:|---|
| 60 GB/s | 50,4 µs | 0,96× — gelijk |
| 70 GB/s | 43,2 µs | **1,12× sneller** |
| 80 GB/s | 37,8 µs | **1,28× sneller** |

**De CPU is vandaag per expert sneller dan de GPU.** Niet omdat de CPU snel is,
maar omdat de GPU-kernel op 16,3% van zijn roofline draait en de CPU op ~80% van
de zijne.

### Bandbreedte-proportionele splitsing

Beide banken staan **al** pinned in host-RAM (18,65 GB experts + 1,56 GB trunk).
De CPU kan er dus direct uit rekenen. Optimale CPU-fractie
f = BW_cpu / (BW_cpu + η·BW_gpu):

| GPU-efficiëntie | f_CPU | E+P nu | E+P gesplitst | winst |
|---:|---:|---:|---:|---:|
| 12,6% (huidig totaal) | 0,591 | 49,80 ms | 20,35 ms | **2,45×** |
| 16,3% (expertkernel) | 0,528 | 38,50 ms | 18,17 ms | **2,12×** |
| 30% | 0,378 | 20,92 ms | 13,01 ms | 1,61× |
| 60% | 0,233 | 10,46 ms | 8,02 ms | 1,30× |

### Het elegante deel

Bij een 60%-kernel is de optimale CPU-fractie **0,233**. De huidige missfractie is
70,66/384 = **0,184**. Die liggen vrijwel op elkaar.

**De regel "cachemiss → reken op de CPU" ís de bandbreedte-optimale policy.** Geen
nieuwe scheduler nodig — de bestaande cachelogica selecteert al bijna precies de
juiste verzameling.

En het elimineert de transferterm volledig: in plaats van 3,02 MB per gemiste
expert stuur je 768 floats terug = 3 KB. **984× minder transfer.** De PCIe-bus
komt vrij, de 4,98 GiB expertcache mag krimpen, en β·m gaat van 1,814 naar ~0,01 ms.

### Risico's die gemeten moeten worden

- DDR5-doorvoer met 16 cores tegelijk (STREAM-benchmark, 5 minuten). Ik reken met
  70 GB/s van 102,4 piek; als het 45 blijkt, halveert de winst.
- Contentie: de PCIe-DMA en de CPU-GEMV delen dezelfde geheugenbus.
- Eén core moet vrij blijven voor de CUDA-driverthread; reken met 14 van 16.

---

## 2. Meer RAM — nu geen beperking, straks dé poort

Nu in gebruik: 18,93 GiB pinned van 63,4 GiB. **RAM is op dit moment geen
bottleneck** en meer RAM versnelt Qwen3-30B-A3B met nul procent.

Maar het is wel de enige harde grens voor grotere modellen:

| bpp | expertparameters die passen (55 GiB pinbaar) | ×huidig (29,0 B) |
|---|---:|---:|
| Q5 (+0,9% CE) | 92,2 B | 3,2× |
| Q4-GPTQ (~+1,8%) | 114,5 B | 3,9× |
| Q3 (~+9,6%) | 151,2 B | 5,2× |

Met **128 GiB** (twee SODIMM-modules, ~€300 op een Arrow Lake-HX-laptop) wordt dat
214 B bij Q5 en 266 B bij Q4. Dat is precies de sprong die de 235B-klasse opent —
zie §6.

---

## 3. NPU — eerlijk antwoord: geen snelheidshefboom

Een NPU deelt **dezelfde DDR5-bus** als de CPU. Batch-1 decode is 100%
bandbreedtegebonden, niet TOPS-gebonden.

```
1 expert = 3,02 MB lezen en 9,44 MFLOP rekenen
arithmetic intensity = 3,12 FLOP/byte
een NPU van 50 TOPS zou 16 TB/s nodig hebben om verzadigd te raken
                        = 160× de DDR5-bus
```

Een NPU voegt dus **geen enkele byte per seconde toe**. Wat hij wel doet:
- **batterij** — dezelfde bytes lezen tegen een fractie van het CPU-vermogen;
- **prefill** — dát is compute-gebonden en daar telt TOPS wel;
- **cores vrijhouden** voor de driverthread.

Verdict: geen prioriteit voor decode-snelheid. Wel interessant als het doel
"lang draaien op accu" wordt.

---

## 4. Nog compacter — ik moet mezelf corrigeren

Ik zei vorige keer: *"compressie helpt de snelheid niet, want transfer is 3,6%."*
Dat was te kort door de bocht. Compressie raakt **twee** termen, en de tweede is
de grootste:

| | GB/token uit VRAM | E bij 60%-kernel |
|---|---:|---:|
| Q5 (huidig) | 1,1608 | 5,04 ms |
| Q4 | 0,9343 | 4,06 ms (−19,5%) |
| Q3 | 0,7078 | 3,07 ms (−39,0%) |

**Bits verlagen versnelt de computeterm evenredig.** Dat is een echte hefboom.

Maar er is een scherpe grens, en CORETAIL is het tegenvoorbeeld: 2,125 → 1,994 bpp
is −6,2% bytes, terwijl de CORETAIL-kernel **30,7 Gweight/s** haalde tegen 97,6
voor de Q5-kernel. **6% minder bytes voor 3,2× minder doorvoer.**

> **Regel: verliesloze codering hoort nooit op het hete pad.** Alleen in de
> opslag-/capaciteitslaag, waar bytes wél de enige valuta zijn.

Bitbreedte verlagen mag dus wel, entropie-codering niet. En de kwaliteitsmarge is
smal: Q5 kost ~+0,9% tegen een poort van 2%; Q4-GPTQ zou ~+1,8% kosten en de hele
marge opeten voor 8% snelheid. Niet doen zolang de kernels op 16% draaien —
daar ligt 3–4×, niet 8%.

---

## 5. Oud onderzoek herwaarderen

Herwaardeer alles tegen **E 37,2% | P 30,8% | G 28,4% | T 3,6%**:

| Waarde | Item | Raakt |
|---|---|---|
| **HOOG** | H3 atomaire sparsity 50% (CE +0,04% op V2-Lite) | **E** — halveert bytes per expert |
| MIDDEL | Layer-aware gemengde precisie (CRAFT 3,24×) | E + capaciteit |
| LAAG, wel nuttig | CORETAIL lossless codec | alleen capaciteit |
| DOOD | Mass-Budget cache-routing (−14% loads) | T — 0,25 ms van 49,93 |
| DOOD | HERA/DHERA/DCHERA/ADHERA/LDHERA | T |
| DOOD | E2GQ entropy coding | T, en decode kost meer dan het oplevert |
| DOOD | H1/H2/H4/H6/H7/H8, RSIV, FLEQ, BITFLOW | T |

### De nieuwe lezing van H3 — dit is de belangrijkste regel in dit document

H3 faalde op de **gather**: per-token atoomselectie raakt elke 4-KiB-pagina
(P(tile van 64 bevat geen geselecteerd neuron) = 0,75⁶⁴ = 10⁻⁸), en spectrale
permutatie haalde 5,0×/8,3× tegen poort 1,20×.

Dat was een **byte**-probleem onder het oude kostenmodel. Onder het nieuwe model
is de juiste variant een andere:

> **Statische per-expert pruning** (Wanda- of SparseGPT-stijl, activatie-aware),
> offline gecompacteerd. Dan is er **geen gather** — de expert is fysiek half zo
> groot. Dat halveert E én de bank tegelijk.

Dit is nooit getest, omdat H3 altijd per-token dynamisch was. De gemeten reden
waarom H3 faalde verdwijnt volledig zodra de selectie statisch wordt.

E 18,56 → 9,28 ms · bank 17,37 → 8,68 GiB · totaal 40,65 ms → **24,6 tok/s**,
en dat vóór alle andere verbeteringen.

---

## 6. Grotere modellen — de capaciteitsformule

Drie randvoorwaarden, alle drie gemeten:

```
(1) hostRAM  : expertparams × bpp_e/8  ≤  55 GiB pinbaar (128 GiB → 118)
(2) VRAM     : LM-head + attentie + KV + workspace  ≤  7,96 GiB
(3) snelheid : actieve bytes / (η·384 + 70)  ≤  100 ms
```

### Randvoorwaarde (3) is verrassend ruim

| kernel-efficiëntie | gecombineerde BW | budget @10 tok/s | actieve params @Q5 |
|---:|---:|---:|---:|
| 12,6% (nu) | 118,4 GB/s | 11,84 GB/token | **18,5 B** (4,9× huidig) |
| 30% | 185,2 GB/s | 18,52 GB/token | 28,9 B |
| 60% | 300,4 GB/s | 30,04 GB/token | **46,9 B** (12,5× huidig) |

### Randvoorwaarde (2) is de echte muur

| Model | trunk@Q8 | trunk@Q4 |
|---|---|---|
| Qwen3-30B-A3B (h=2048) | 2,53 GiB ✓ | 1,96 GiB ✓ |
| ~100B-klasse (h=3072) | 3,62 GiB ✓ | 2,70 GiB ✓ |
| Qwen3-235B-A22B (h=4096, 94 lagen) | 8,66 GiB ✗ | **5,25 GiB ✓** |

*(LM-head + attentie + KV@4k + 1 GiB workspace)*

### Wat dat concreet betekent

**Vandaag, 63 GiB RAM:** de ~100B/10B-klasse past ruim. <cite index="9-1">Solar Open is bijvoorbeeld ~102,6 B totaal met 10 B actief</cite> — precies de vorm waar deze machine voor gebouwd is. Ook Llama 4 Scout (109B/17B) en de gpt-oss-120b-klasse.

**Met 128 GiB RAM (~€300):** Qwen3-235B-A22B wordt haalbaar. 227 B expertparameters
bij Q4 = 109 GiB bank, trunk@Q4 = 5,25 GiB in VRAM, actief 11,12 GB/token.
Bij de huidige kernels: 94 ms → **10,6 tok/s**. Na de kernelfixes: **~27 tok/s**.

De echte frontier — GLM-5.2 (744B/40B), DeepSeek V4-Pro (1,6T/49B), Qwen3.5
(397B/17B) — past ook met 128 GiB niet. Daarvoor is 256 GiB nodig, en dat past
niet in een laptop.

**De sweet spot van deze machine is ~100–235 B totaal met ≤ 20 B actief en
hidden ≤ 4096.**

---

## 7. Speculative decoding — ik trek mijn eigen advies in

Ik heb dit twee keer voorgesteld. De gemeten kostenverdeling zegt nu dat het voor
dít model niet werkt, en dat hoort in het rapport.

De reden: bij een verificatiepass over s posities moet je de **unie** van hun
experts lezen. Bij top-8 van 128 is die unie bijna lineair:

| draftdiepte | unieke experts/laag | expertwerk per token |
|---:|---:|---:|
| s=2 | 15,5 van 16 | ×0,969 |
| s=4 | 29,1 van 32 | ×0,910 |
| s=8 | 51,6 van 64 | ×0,807 |

P en G amortiseren wél (die zijn per pass), maar E — de grootste term — niet.
Nettoresultaat met een 0,6B-drafter, s=4, 2,5 geaccepteerd:
**1,10× nu, en negatief zodra de kernels gefixt zijn** (dan is E weer dominant en
de drafterkosten blijven).

> **Algemene regel die hieruit volgt:** de waarde van speculative decoding voor
> MoE-decode schaalt met K/Nₑ. Qwen3-30B-A3B zit op 8/128 = 6,25% — vrijwel het
> slechtst mogelijke geval. Bij 8 experts top-2 (25%) zou het wél lonen.

Dat is een nuttig negatief resultaat op zich, en het volgt direct uit de eigen
metingen.

---

## 8. De volledige stapeling, conservatief gerekend

| Stap | totaal | tok/s test | tok/s rollout |
|---|---:|---:|---:|
| gemeten P6B | 49,93 ms | **20,0** | **15,9** |
| + CUDA Graph op de glue | 41,41 ms | 24,1 | 19,1 |
| + CPU/GPU-bandbreedtesplitsing | 23,86 ms | 41,9 | 33,2 |
| + gefuseerde dequant-kernels (40%) | 16,46 ms | 60,7 | 48,1 |
| + statische 50% pruning (H3 herzien) | 13,87 ms | **72,1** | **57,1** |

Geen enkele stap is onderzoek. Alle vijf zijn implementatie of hergebruik van
werk dat al in de repo staat.

---

## 9. Aanbevolen volgorde

| # | Stap | Kosten | Verwachte winst | Waarom eerst |
|---|---|---|---|---|
| 1 | **P7A kernel-roofline-diagnose** | 20 min | — | beslist of stap 3 3–4× of 1× is |
| 2 | **STREAM-benchmark DDR5 met 16 cores** | 5 min | — | beslist of stap 4 2,1× of 1,3× is |
| 3 | **Piecewise CUDA Graph op de glue** | 1–2 dagen | 1,21× | grootste winst per uur werk |
| 4 | **CPU/GPU-bandbreedtesplitsing** | 3–5 dagen | 1,74× | grootste absolute winst |
| 5 | **Gefuseerde dequant-matvec** | 1–2 weken | 1,45× | hangt af van stap 1 |
| 6 | **Statische per-expert pruning** | 1 week | 1,19× + halve bank | opent ook capaciteit |
| 7 | **10× grotere kwaliteitsset** | 1 run | — | maakt het publiceerbaar |
| 8 | **Groter model (100B-klasse)** | 1 week | — | het eigenlijke doel |

De twee diagnoses bovenaan kosten samen 25 minuten en bepalen of de rest 3× of
10× oplevert. Die zou ik morgenochtend doen.

---

## 10. Wat ik niet zou doen

- **Nog een compressievariant.** Zeven van de dertien gesloten registries waren dat.
- **Verliesloze codering op het hete pad.** CORETAIL: −6% bytes voor 3,2× doorvoer.
- **Speculative decoding op dit model.** K/Nₑ = 6,25% is het slechtste geval.
- **Een NPU-integratie voor snelheid.** 3,12 FLOP/byte; er is geen TOPS-probleem.
- **Nog een cachepolicy.** T is 3,6% en gaat naar 0,01% zodra de CPU meerekent.
- **Q4 om snelheid te winnen.** 8% sneller voor de hele kwaliteitsmarge, terwijl
  de kernels 3–4× overhouden.
