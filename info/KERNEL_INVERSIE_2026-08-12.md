# De inversie: transfer is 3,6% geworden. De kernels zijn nu alles.

**Datum:** 2026-08-12 · **Type:** heranalyse van P4A/P5A/P6B · **Status:** afleiding
uit gemeten data, geen nieuwe run

---

## 0. Eerst mijn eigen score, inclusief de fouten

| Voorspelling | Uitkomst |
|---|---|
| async overlap → 19,06 ms | **20,374 ms** — 6,9% af, binnen de vooraf gestelde 15%-band ✓ |
| full model 18–47 tok/s | **20,03 tok/s** ✓ |
| embedding vrij → 102 slots | **96 slots** — bijna ✓ |
| kwaliteitsmeting is ruisgedomineerd | **bevestigd, sterker dan gedacht** ✓ |
| *"768 launches per token"* | **fout — het waren er 192.** De experts waren al gegroepeerd |
| *"1 laag lookahead laat de transfer volledig verdwijnen"* | **fout in de sterke vorm.** De misshelling daalt 80,8%, niet 100% |
| *"de probe is gratis te trainen op bestaande captures"* | **fout.** De captures bevatten alleen top-8-ID's, geen hidden states |

Het P3B/P4A-verdict corrigeert die drie punten terecht en met bewijs. Ik neem ze
over. De vierde en vijfde correctie zijn belangrijker dan ze lijken, want ze
sturen de conclusie hieronder.

---

## 1. Het viertermenmodel, volledig uit gemeten data

P6B test: 49,927 ms/token. Twee componenten zijn afzonderlijk gemeten
(P4A: 20,374 ms; P5A: 15,360 ms). Het verschil is de rest van de decode.

| Term | ms/token | aandeel |
|---|---:|---:|
| **E** — expertcompute, Q5 GEMV | 18,560 | 37,2% |
| **P** — projecties, Q8 q/k/v/o/router/LM-head | 15,360 | 30,8% |
| **G** — glue: norms, RoPE, attention, softmax, top-k, residuals, sampling | 14,193 | 28,4% |
| **β·m** — resterende transfer na causale overlap | 1,814 | 3,6% |
| **som** | **49,927** | **100%** |

E en β·m komen uit de P4A-testregressie `wall = 18,5601 + 0,025672 × misses`
bij 70,66 missen. G is het residu tegen de gemeten P6B-integratie.

---

## 2. De inversie

| | serieel (P3A) | na causale overlap (P4A) |
|---|---:|---:|
| prijs per miss | 0,133401 ms | **0,025672 ms** (−80,8%) |
| transfer per token | 9,427 ms | **1,814 ms** |
| aandeel van de totale tokentijd | 33,5% van de expertplane | **3,6% van de hele token** |

**De term die de hele offloadingliteratuur optimaliseert is 3,6% van de kosten
geworden.**

En dat is precies wat de dertien gefalsificeerde hypotheses in dit project
verklaart. Kijk waar ze allemaal op zaten:

| Hypothese | Optimaliseerde | Uitkomst |
|---|---|---|
| CRAFT H1 route-coreset + CRCQ | transfer | gefalsificeerd |
| CRAFT H2 block-coalescing | transfer | 19,65% vs poort 40% |
| CRAFT H3 atomaire sparsity | transfer | modelbreed gefalsificeerd |
| CRAFT H4 SketchGate | transfer | 22,7% high-damage miss |
| CRAFT H6 QERC | transfer | kruisterm ≈ nul |
| CRAFT H7 route-coreset oracle | transfer | cardinaliteitsstaart |
| CRAFT H8 cache span | transfer | gesloten |
| DHERA/DCHERA/ADHERA/LDHERA | transfer | 4× gesloten op verkeerspoorten |
| HERA statische tiering | transfer | unie te groot |
| RSIV / GhostWeights | transfer | terminaal gefalsificeerd |
| FLEQ / GSQ 2-bit | transfer | gefalsificeerd |
| E2GQ entropy coding | transfer | coverage-negatief |
| CORETAIL lossless codec | transfer | kwaliteit −, codec + |

**Dertien preregistreerde falsificaties, allemaal aan dezelfde kant van de
vergelijking.** Niet omdat de ideeën slecht waren, maar omdat de term die ze
aanvielen na één correcte overlapimplementatie 3,6% waard is.

Dat is het wetenschappelijke resultaat. Niet een methode — een gemeten
kostenverdeling die de premisse van het veld omdraait, gedragen door het
grootste preregistreerde negatieve corpus in dit deelgebied.

---

## 3. Waar de 96,4% nu zit: de kernels

| | Gweight/token | GB uit VRAM | roofline @384 GB/s | gemeten | % roofline | effectief |
|---|---:|---:|---:|---:|---:|---:|
| experts Q5 | 1,8119 | 1,1608 | 3,023 ms | 18,560 ms | **16,3%** | 62,5 GB/s |
| projecties Q8 | 1,2297 | 1,2489 | 3,252 ms | 15,360 ms | **21,2%** | 81,3 GB/s |
| **samen** | 3,0416 | 2,4097 | **6,275 ms** | 33,920 ms | | |

Fysieke ondergrens compute + projectie: **6,275 ms → 159 tok/s.**
Gemeten: 49,927 ms → 20,03 tok/s.

**Het model draait op 12,6% van zijn eigen hardwareroofline.**
Gepubliceerde matvec-kernels van deze klasse (Marlin, machete, llama.cpp mmvq)
halen 60–75%.

### De deductie die de oorzaak vastpint

P3B heeft al bewezen dat dit géén dispatch is: eager 17,113 ms versus graph
17,243 ms, ratio 1,0076, terwijl de no-opcontrole 9× goedkoper werd. De 18,56 ms
is echte kerneltijd.

En dan dit: **de Q8-projecties halen ook maar 21,2%.** Q8 vereist nul
bitmanipulatie — één byte laden, één convert. Als zelfs die kernel op een vijfde
van de roofline blijft, dan is de 5-bit-uitpakking níét de oorzaak. Het probleem
is gedeeld tussen beide kernels en zit dus in de GEMV-structuur zelf.

De meest waarschijnlijke gedeelde oorzaak: **beide kernels dequantiseren naar een
BF16-scratchbuffer en roepen daarna een standaard BF16-GEMV aan.** Dat is de
klassieke "fake quantization"-implementatie. Ze verklaart waarom beide kernels op
16–21% blijven ongeacht bitbreedte, want het extra verkeer is 2 bytes per gewicht
in beide gevallen:

| | Q-bytes gelezen | BF16 geschreven | BF16 teruggelezen | totaal per gewicht |
|---|---:|---:|---:|---:|
| Q5 | 0,641 B | 2 B | 2 B | 4,641 B (7,2×) |
| Q8 | 1,016 B | 2 B | 2 B | 5,016 B (4,9×) |

De fix is standaard: **gefuseerde dequant-in-register matvec** — uitpakken in
registers, gedequantiseerde gewichten nooit naar geheugen schrijven.

---

## 4. De verbeterladder

| Stap | G | E | P | β·m | totaal | tok/s test | tok/s rollout |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemeten P6B | 14,19 | 18,56 | 15,36 | 1,81 | 49,93 | **20,0** | **15,9** |
| A. CUDA Graph op de **glue** | 5,68 | 18,56 | 15,36 | 1,81 | 41,41 | 24,1 | 19,1 |
| B. + kernels naar 40% roofline | 5,68 | 7,56 | 8,13 | 1,81 | 23,18 | 43,1 | 34,2 |
| C. + kernels naar 60% roofline | 5,68 | 5,04 | 5,42 | 1,81 | 17,95 | **55,7** | **44,1** |

Stap A is niet hetzelfde als P3B. P3B legde de graph op de **expertplane** — 192
launches, compute-gebonden, en vond terecht 1,0076×. De glue is een andere
populatie: per laag ongeveer input-RMSNorm, q/k-norm, RoPE, KV-write,
attention-score, softmax, value-reductie, residual, post-attention-norm,
routersoftmax, top-k, gewogen reductie, residual — allemaal kernels van enkele
kilobytes, met vrijwel nul rekenwerk. **Dat is precies de populatie waarvoor P3B
de 9× dispatchreductie heeft gemeten.** De graph moet dus op de glue, niet op de
experts.

Voorbehoud: graphs eisen statische shapes. De expert-H2D is data-afhankelijk en
blijft eager; de glue is shape-statisch mits KV op een vaste maximumlengte wordt
gepadded met masker, of per lengtebucket wordt gecaptured. Dat is standaard
praktijk (vLLM doet exact dit als piecewise CUDA graphs).

---

## 5. De rollout-straf is een tweede aanwijzing voor dezelfde oorzaak

Rollout 63,024 ms tegen test 49,927 ms: **+13,097 ms (+26,2%).**

Dat kan niet van cachemisses komen. Zelfs bij 100% miss (384 van 384) is de
expertplane 28,418 ms, dus maximaal 8,044 ms extra. Er blijft **minstens
5,053 ms** over die geen miss is. De groeiende KV is het ook niet: bij 512
context is dat 50,3 MB = 0,131 ms.

De overblijvende verklaring is vormafhankelijke her-dispatch: bij elke
rollout-stap verandert de contextlengte, dus verandert de attention-shape,
dus vervalt elke kernelcaching en autotuning. **Gebucketteerde CUDA Graphs
lossen exact dat op** — en het is dezelfde ingreep als stap A.

---

## 6. Waar ik hard op blijf: de +0,048% mag niet naar buiten

Dezelfde constructie gaf:

| split | relatieve CE |
|---|---:|
| P6B validation | **+1,782862%** |
| P6B test | **+0,048026%** |

Een spreiding van **1,73 procentpunt** tussen twee splits van 1270 labels, voor
identieke gewichten en identieke kernels. Tel de vier eerdere Q5+INT8-metingen
erbij (+0,698 / +0,999 / +1,478 / −0,478) en het beeld is eenduidig: de
meetruis op 1270 labels is σ ≈ 0,8–1,0 procentpunt.

De eerlijke uitspraak is niet "+0,048%" maar **"ongeveer +0,9%, met een
standaardfout van dezelfde orde"**. Dat haalt de 2%-poort nog steeds, maar
het getal +0,048% is een steekproeftrekking, geen eigenschap van het systeem.
Dit is het enige punt in de hele keten waar de verder uitstekende
bewijsdiscipline een niet-houdbare precisie suggereert.

**Vóór enige externe claim: ~10× grotere evaluatieset met gepaarde
block-bootstrap** — dezelfde methodiek die CRAFT H1 en H3 al gebruikten. Dat is
één run en het maakt het verschil tussen een anekdote en een resultaat.

---

## 7. Voorgestelde preregistraties

### P7A · De kernel-roofline-diagnose (20 minuten, beslissend)

Drie microbenchmarks op exact de bestaande banken:

1. **Pure lees**: kernel die alleen de Q5-expertbytes leest en XOR-reduceert.
   Geen unpack, geen MAC.
2. Idem op de Q8-projectiebank.
3. De bestaande kernels, met een expliciete telling van gealloceerde
   scratchbytes per aanroep.

**Discriminatie:**
- pure lees ≥ 300 GB/s **en** scratch > 0 → dequant-to-scratch bevestigd,
  fix = gefuseerde matvec, verwachte winst 3–4×;
- pure lees ≈ 62–81 GB/s → het probleem is de geheugenlayout/coalescing,
  fix = herindeling van de bank;
- pure lees ≥ 300 GB/s **en** scratch = 0 → beide hypotheses gefalsificeerd,
  de kosten zitten in de MAC-lus en de vraag is open.

Poort: het rapport moet zeggen wélke van de drie het is, met gemeten GB/s.

### P7B · Piecewise CUDA Graph op de glue

Capture alles behalve de expert-H2D en de expertkernels: norms, q/k-norm, RoPE,
KV-write, attention, softmax, residuals, routersoftmax, top-k, finale norm,
LM-head, argmax. KV gepadded op vaste buckets (128/256/512/1024/2048/4096).

**Poort:** G ≤ 8,0 ms op test (van 14,193). Voorspelling uit dit document:
**5,68 ms.** Meet daarnaast de rollout apart — de voorspelling is dat de
26,2%-straf grotendeels verdwijnt. Wijkt het meer dan 25% af, dan is de
glue-diagnose gefalsificeerd en dat moet zo gerapporteerd.

### P7C · Gefuseerde dequant-matvec (alleen bij een P7A-bevestiging)

Eén kernel per laag over de acht experts, uitpakken in registers, nooit
dequantiseerde gewichten naar geheugen. Correctheid: bit-identiek aan de
bestaande kernel op dezelfde 72 gelockte matrices.

**Poort:** ≥ 40% van de VRAM-roofline op beide banken, oftewel E ≤ 7,6 ms en
P ≤ 8,2 ms. Voorspelling voor het totaal: **23,2 ms → 43 tok/s test,
34 tok/s rollout.**

### P7D · De kwaliteitsclaim vastzetten

10× grotere evaluatieset, gepaarde block-bootstrap, 95%-BI op de relatieve CE.
Geen nieuwe constructie, geen nieuwe kernels — alleen de bestaande P6B-runtime
op meer data.

**Poort:** bovengrens van het 95%-BI ≤ 2%. Dit is de enige stap die van
"lokale Eureka" een publiceerbaar resultaat maakt.

---

## 8. Wat dit project nu is

Een werkend 30,5B-parameter MoE met een expertbank van 17,37 GiB, draaiend op
een laptop-GPU met 8 GB VRAM, volledig autoregressief, op 15,87 tok/s over 512
gegenereerde tokens, met een CE-verlies van ongeveer een procent. Onafhankelijk
geverifieerd, 120/120.

En daarnaast een gemeten kostenverdeling die zegt: **na de juiste bitbreedte en
causale overlap is transfer 3,6% van de kosten, en de resterende 96,4% is
on-device rekentijd op 12,6% van de eigen hardwareroofline.** Er ligt dus nog een
factor 8 tot de fysica en een factor 2,8 tot wat gepubliceerde kernels halen.

Dat is geen nieuwe hypothese. Het is de vaststelling dat het onderzoek klaar is
en het optimalisatiewerk begint.
