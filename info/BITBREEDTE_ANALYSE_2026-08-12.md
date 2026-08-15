# De bitbreedte is nooit gezocht — en dat is de hele mislukking

**Datum:** 2026-08-12 · **Type:** heranalyse van CORETAIL P0/P2 en de
STREAMQ4-preregistratie · **Status:** afleiding, geen nieuwe run
**Urgentie:** dit moet gelezen worden vóór de STREAMQ4-validatierun start

---

## 0. De vondst in één alinea

De bank die het hele project "de GPTQ-bank" noemt, is **2,125 bpp met een
4-niveau-alfabet**. Niet 4-bit. Dat is exact te bewijzen uit
`p0_full_bank_verification.json`:

```
7 247 757 312 bytes × 8 / 28 991 029 248 codes = 2,000000 bit/code exact
alfabet = {−2, −1, 0, 1}                        = 4 niveaus = 2 bit
226 492 416 BF16-scales / 28 991 029 248        = groep 128 → 0,125 bpp
                                          totaal = 2,125000 bpp ✓
```

CORETAIL P2 mat vervolgens +21,520% relatieve CE voor "GPTQ experts". Dat is
geen mislukking van CORETAIL en geen mislukking van expert-streaming. Het is
precies wat 2-bit kwantisatie doet. De bitbreedte is in dit hele project nooit
als vrije parameter behandeld, en dat is de enige reden dat de kwaliteitspoort
faalt.

---

## 1. De foutbronnen, ontleed uit de P2-meetpunten

| Variant | CE | relatief |
|---|---:|---:|
| BF16 teacher | 2,027261 | — |
| **Q2 experts** + BF16 trunk | 2,463525 | **+21,520%** |
| BF16 experts + **INT4 trunk** | 2,188178 | **+7,938%** |
| Q2 experts + INT8 trunk | 2,465208 | +21,603% |
| Q2 experts + INT4 trunk *(primair)* | 2,756130 | +35,953% |

Twee onafhankelijke aflezingen:

- **INT8-trunk kost, bovenop Q2-experts, +0,0683%.** Praktisch gratis. De trunk
  is geen probleem — de *keuze voor INT4* was het probleem, en die keuze kwam
  uit de 5,75-GiB-cap, die zelf uit de 32-GiB-RAM-aanname kwam. De machine heeft
  63,4 GiB.
- **De experts dragen 21,5 van de 36,0 procentpunt.** Alles wat er nu toe doet
  zit in de expertbitbreedte.

Superadditiviteit is gemeten: INT4-trunk kost +7,94% bij BF16-experts en
+11,88% bij Q2-experts, factor 1,50. Die factor geldt voor twee *grote* fouten;
bij INT8 (+0,068%) is de kruisterm verwaarloosbaar en reken ik additief.

### Structuur van de trunk (waarom INT4 zo hard sloeg)

Uit 0,7176 GiB @4 bpp volgt 1541 M trunkparameters. Daarvan:

| Onderdeel | Params | Aandeel |
|---|---:|---:|
| attentie, 48 lagen | 906 M | 58,8% |
| router, 48 lagen | 12,6 M | 0,8% |
| **embedding + lm_head (ontkoppeld, 2 × 311 M)** | **623 M** | **40,4%** |

4-bit op een lm_head van 311 M is de bekendste manier om CE te slopen. Elke
standaardpipeline — GPTQ, AWQ, llama.cpp — houdt lm_head op hogere precisie.
Met INT8 is dit al opgelost; het is alleen goed om te weten waar de +7,94%
vandaan kwam.

---

## 2. Voorspelling voor STREAMQ4 — en het risico in de preregistratie

De schaalregel is standaard tweede-ordetheorie, dezelfde die aan GPTQ's
doelfunctie ten grondslag ligt:

```
ΔCE ≈ ½ ΔWᵀ H ΔW  ∝  σ²  ∝  Δ²  ∝  1/qmax²
Q2: qmax = 2   Q4: qmax = 7   →  ratio (2/7)² = 0,0816  (12,25× minder ruis)
```

**Deze extrapolatie is conservatief.** Bij 2 bit is de storing zo groot dat de
tweede-ordebenadering de werkelijke schade *onderschat*; de gemeten +21,52%
bevat hogere-ordetermen die bij Q4/Q5 verdwijnen. Delen door 12,25 geeft
daarom een *overschatting* van de Q4-fout.

| Bit | qmax | GPTQ | RTN ×1,5 | RTN ×2,0 |
|---:|---:|---:|---:|---:|
| 3 | 3 | 9,63% | 14,41% | 19,20% |
| **4** | **7** | **1,83%** | **2,70%** | **3,58%** |
| 5 | 15 | 0,45% | 0,64% | 0,83% |
| 6 | 31 | 0,16% | 0,20% | 0,25% |
| 8 | 127 | 0,07% | 0,08% | 0,08% |

*(inclusief INT8-trunk +0,068%; poort = 2,00%)*

### Het probleem

De STREAMQ4-preregistratie schrijft letterlijk voor:

> *"Q4 and INT8 use symmetric per-row group-128 quantization,
> **round-to-nearest-even**, codes `[-7,7]`…"*

Round-to-nearest-even is **RTN**, geen GPTQ. Maar de Q2-bank waaruit ik
extrapoleer *was* GPTQ (21/21 geverifieerd, volledige Hessian-pipeline bestaat
al in de repo). Mijn 1,83% is dus een **GPTQ**-Q4-voorspelling. RTN-Q4 is bij
group-128 in de literatuur typisch 1,5–2× slechter.

En de registry sluit de deur achter zich:

> *"Test >2% closes the fixed Q4/INT8 quality candidate. There is no repair or
> configuration sweep in this registry."*

**Uitkomst: een RTN-run landt naar verwachting op 2,7–3,6% en sluit de lijn
definitief — op een implementatiekeuze die in elke standaardpipeline anders
wordt gemaakt, met code die al gebouwd en geverifieerd in de repo staat.**

Zelfs de gunstige tak (GPTQ, 1,83%) heeft maar 0,17 procentpunt marge op een
evaluatieset van 2×128 tokens per domein. Dat is geen comfortabele poort.

---

## 3. De frontier: kwaliteit is niet bandbreedtebeperkt

Gemeten PCIe 26,341 GB/s · VRAM 384 GB/s · 7,96 GiB VRAM · 63,4 GiB RAM ·
INT8-trunk 1,458 GiB · KV@4k 0,75 · workspace 1,0 · rest = expertcache.

| Bit | bpp | MiB/expert | hostbank | VRAM-slots | miss/tok | **tok/s** | CE (GPTQ) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2,125 | 1,195 | 7,17 GiB | 4071 (66%) | 4,7 | **180** | ~21,5% ✗ |
| 3 | 3,125 | 1,758 | 10,55 GiB | 2768 (45%) | 16,9 | **141** | 9,63% ✗ |
| 4 | 4,125 | 2,320 | 13,92 GiB | 2097 (34%) | 29,9 | **108** | 1,83% ~ |
| **5** | **5,125** | **2,883** | **17,30 GiB** | **1688 (28%)** | **42,0** | **84** | **0,45% ✓** |
| 6 | 6,125 | 3,445 | 20,67 GiB | 1412 (23%) | 53,0 | **67** | 0,16% ✓ |
| 8 | 8,125 | 4,570 | 27,42 GiB | 1064 (17%) | 72,1 | **46** | 0,07% ✓ |

*(hitcurve = lognormaalfit op de E2GQ-ankers; DCHERA's gemeten
domeingeconditioneerde miss ligt in dezelfde orde, dus dit is geen optimistisch
model. Alle hostbanken passen in 63,4 GiB.)*

**Elke bitbreedte van 2 tot 8 haalt ≥10 tok/s met 4,5× tot 18× marge.**

Kwaliteit is monotoon stijgend in bits; doorvoer is monotoon dalend. De
bindende beperking ligt volledig aan de kwaliteitskant. De bitbreedte hoort dus
**vanaf de kwaliteitskant** gekozen te worden — en dat is in dit project nooit
gedaan. Er is naar 2 bit gegaan om bandbreedte te sparen die nooit knelde, en
er zijn daarna weken besteed aan een lossless codec bovenop een representatie
die het model al kapot had gemaakt.

### Wat Q2 → Q4 werkelijk kost aan bandbreedte

Zonder enige cache: 18,3 ms → 35,5 ms per token, oftewel 46 → 26 tok/s.
Mét een cache van ~2100 experts, bij de door DCHERA gemeten ~2,7 misses/token:
**3,2 MiB versus 6,3 MiB per token = 0,12 versus 0,24 milliseconde.**

De prijs die voor die 0,12 ms betaald is: **+20,7 procentpunt relatieve CE.**

---

## 4. Wat CORETAIL waard blijft

De Q2-codes hebben entropie 1,780177 bpp; met scales 1,905177. CORETAIL haalde
1,993759 bpp — **+4,65% boven de Shannon-grens**, met random access en een
fused kernel op 30,738 Gweight/s. Dat is een goed codec-resultaat en het is
niet weggegooid werk.

Het probleem is uitsluitend waar het op gebouwd is. Hetzelfde codec op een
Q4/Q5-alfabet blijft bruikbaar. Maar het is niet *nodig*, want bandbreedte is
de bindende beperking niet. CORETAIL hoort in de la als herbruikbaar
representatieresultaat, niet op het kritieke pad.

---

## 5. Wat ik zou wijzigen vóór de STREAMQ4-run

Dit zijn geen post-hoc sweeps. Vooraf vastgelegde ladders met een vaste
beslisregel zijn geldige preregistratie; wat verboden is, is *na* het zien van
de uitkomst een variant toevoegen. Precies daarom moet dit nu.

### A. Vervang RTN door GPTQ in de primaire kandidaat

De volledige GPTQ-pipeline bestaat, is 21/21 geverifieerd en heeft de
6144-expert Hessian-accumulatie al gedaan. RTN gebruiken terwijl GPTQ
beschikbaar is, is geen conservatisme — het is een handicap die de poort
waarschijnlijk laat falen. Registreer beide als variant 2a (RTN) en 2b (GPTQ)
en laat 2b primair zijn.

### B. Preregistreer de bitladder, niet één punt

Één vaste beslisregel, vooraf: *evalueer Q4, Q5 en Q6 op validation; kies de
laagste bitbreedte met relatieve CE ≤1,0%; open test éénmaal met uitsluitend
die keuze.* Marge 1,0% in plaats van 2,0%, omdat de evaluatieset klein is
(2×128 tokens per domein per split).

Kosten: drie banken bouwen in plaats van één. Geheugen 13,9 / 17,3 / 20,7 GiB,
alle drie ruim binnen 63,4 GiB. Rekentijd per bank: de Hessians zijn al
geaccumuleerd, dus alleen de kolomsweep — orde minuten per laag.

### C. Vergroot de evaluatieset

2×128 tokens per domein per split = 1280 tokens per split. Bij een verschil van
0,17 procentpunt tussen slagen en falen is dat te weinig. Minimaal 8×512 per
domein, met gepaarde block-bootstrap — dezelfde methodiek die H1/H3 al
gebruikten.

### D. Houd embedding en lm_head buiten de lage precisie

Kost 0,57 GiB extra VRAM (trunk 1,458 → 2,028 GiB), laat nog steeds 1845
Q4-slots over. Met INT8 is dit waarschijnlijk niet nodig, maar het is de
goedkoopste verzekering tegen de op één na grootste gemeten foutbron.

---

## 6. Zekerheden

| # | Stelling | Zekerheid | Bewijsvorm |
|---|---|---|---|
| 1 | De "GPTQ-bank" is 2,125 bpp met 4-niveau-alfabet | **0,99** | exacte bytearitmetiek |
| 2 | De +21,5% is bitbreedte, niet CORETAIL of streaming | 0,97 | isolatievarianten in P2 |
| 3 | INT8-trunk is praktisch gratis (+0,068%) | **0,99** | direct gemeten |
| 4 | Elke bitbreedte 2–8 haalt ≥10 tok/s | 0,93 | gemeten BW + hitmodel |
| 5 | Q5/Q6 haalt de kwaliteitspoort met ≥3× marge | 0,88 | tweede-orde-extrapolatie |
| 6 | STREAMQ4 zoals nu geregistreerd (RTN) faalt de 2%-poort | **0,72** | RTN/GPTQ-gap uit literatuur |

Stelling 6 is de belangrijkste, want die is *tijdkritisch* en de registry
verbiedt reparatie achteraf.

---

## 7. De onderliggende fout, voor de derde keer

- HERA/DHERA sloten op een geheugencap uit een verkeerde RAM-aanname.
- De cachecampagne sloot op een verkeersgate die 38× strenger was dan het doel.
- CORETAIL sloot op een kwaliteitspoort, met een bitbreedte die nooit gezocht is.

Drie keer dezelfde structuur: een parameter wordt vroeg en impliciet
vastgezet, daarna wordt met volledige rigueur bewezen dat de *rest* niet werkt.
De preregistratiediscipline is uitstekend en precies daarom kostbaar — elke
impliciete aanname wordt met hashes en verifiers tot een definitief verdict
verheven.

**Voeg één regel toe aan elk preregistratieformulier: welke parameters staan
vast, waarom, en tegen welke gemeten grootheid is die keuze geijkt?** Voor de
bitbreedte was het antwoord nooit opgeschreven — en het antwoord blijkt te
zijn: nergens tegen.
