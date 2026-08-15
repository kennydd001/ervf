# N1–N5 — vijf eigen hypotheses, en een correctie op mijn eigen conclusie

Datum: 2026-08-15
Verdict: **Twee van de vijf slaan aan, en ze zijn groot. 23,7% van een token is kernel-uitgifte en geen rekenwerk (N1). De gather in het `down`-pad is 8,19 ms met een drift van 0,040 — 67% van dat pad (N2). Attention is voor ~100% byte-gebonden en draait op 47,2 GB/s waar het apparaat 338,4 haalt (N4). En de roofline van één token is 6,05 ms bij ctx 0 en 8,43 ms bij 262K — waarmee ik mijn eerdere uitspraak dat 50 tok/s "buiten de fysica" lag moet terugnemen (N5). N3, de exacte ReLU²-prefilter, faalt: de expertmatrices hebben geen laag-rang-structuur.**
Terminal state: `n1_n5_issue_overhead_and_attention_efficiency_are_the_gap`
Preregistratie: `N1_N5_OWN_HYPOTHESES_PREREGISTRATION_2026-08-15.md`

---

## N5 eerst — de roofline, want die herschikt alles

Gemeten streaming-leesbandbreedte van dit apparaat: **338,4 GB/s** (256 MiB,
`float4`-loads, eigen kernel — geen datasheet).

| | bytes die een correcte forward moet lezen | vloer | plafond |
|---|---:|---:|---:|
| ctx 0 | 1.953,3 MiB | **6,05 ms** | **165,2 tok/s** |
| ctx 262.100 | 2.721,2 MiB | **8,43 ms** | **118,6 tok/s** |

Wat dat betekent voor de doelen:

| doel | ms/token | verdict |
|---|---:|---|
| 1000 tok/s | 1,00 | **uitgesloten** — 6× onder de bytevloer |
| 100 tok/s | 10,00 | niet uitgesloten; vraagt 60–84% van de piekbandbreedte op élke byte |
| 50 tok/s | 20,00 | **niet uitgesloten**; vraagt 30–42% |

**Ik moet een eerdere uitspraak terugnemen.** Ik schreef dat 50 tok/s "buiten de
gemeten fysica van dit model op deze GPU" ligt. Dat was te sterk. De bytevloer
laat 50 tok/s toe. Wat de metingen uitsluiten is het bereiken ervan langs de
mechanismen die getest zijn — speculatie, bytereductie, certificering — niet het
doel zelf.

De runtime draait nu op **16,8% van de roofline** bij ctx 0 (36,05 tegen 6,05 ms)
en **15,5%** bij 262K. 50 tok/s vraagt 30–42%. Het gat is dus efficiëntie, geen
natuurwet — en N1 en N4 wijzen precies aan waar het zit.

---

## N1 — bijna een kwart van een token is uitgifte, geen rekenwerk

Dezelfde kernelreeks, dezelfde argumenten, dezelfde bytes, één keer eager en één
keer als CUDA-graph (routes bevroren, want capture verbiedt synchronisatie).

| | ms |
|---|---:|
| eager | 36,714 |
| graph-replay | **28,023** |
| **verwijderbaar** | **23,7%** |

Dat is de **bovengrens van élk ontwerp dat werk van de host naar de GPU
verplaatst**: megakernel, device-side routing, persistente kernels, graph-based
decoding. S9 schatte de launch-overhead op 13% van de MoE-term; over het hele
token blijkt het 23,7%, want er komt host-werk en per-kernel-latency bij.

Toegepast op de bevroren basislijn: 36,05 → 27,5 ms bij ctx 0 (≈36 tok/s), en
54,28 → 41,4 ms bij 262K (≈24 tok/s) — als het volledig realiseerbaar zou zijn.
Dat is het niet zonder de routes op device te krijgen, en V1 heeft laten zien dat
de voor de hand liggende manier daarvoor te duur is. Maar de **prijs** is nu
gemeten en hij is veel groter dan eerder gedacht.

## N2 — de gather is het `down`-pad

In-lus replicatie, gebracketeerd:

| gerepliceerd | marginaal | lokale drift | |
|---|---:|---:|:--:|
| `panel_scan` | +0,305 ms | 4,669 | onder de ruisvloer |
| **`gather_down_sparse`** | **+8,192 ms** | **0,040** | ✅ scherp |
| masked GEMV + reductie | +12,429 ms | 4,825 | ruis |
| het hele `down_masked_into` | +12,649 ms | 3,216 | |

**G-N2-1 gehaald**: `scan` + `gather` zijn 67,2% van het hele down-pad.

Het scherpste getal is de gather: **8,192 ms met een drift van 0,040** — een
verhouding van 205:1, de best opgeloste marginaal van de hele sessie. Die gather
haalt ~35 MB per token uit mapped host, wat neerkomt op ~4,3 GB/s effectief tegen
de 25,05 GB/s die S5 voor dezelfde kernel geïsoleerd mat. Zesmaal slechter in de
lus dan alleen — precies S8's les, opnieuw.

Dit verklaart ook S11 achteraf: die haalde alle gathers weg en verloor toch 4,8%,
dus de extra misskosten waren daar ~10,8 ms. De gather is duur, maar hem
vervangen door volledige records is duurder.

## N3 — de exacte ReLU²-prefilter faalt, en waarom is leerzaam

91% van de ReLU²-uitgangen is nul en wordt volledig berekend. Een rang-`r`
benadering plus een sound residualgrens zou die rijen bewijsbaar kunnen
overslaan, bit-identiek.

| rang | gecertificeerd | grens/\|ŷ\| | spectrale energie |
|---:|---:|---:|---:|
| 8 | 0,01% | 59,92 | 7,8% |
| 16 | 0,01% | 57,72 | 10,7% |
| 32 | 0,01% | 54,72 | 15,1% |
| 64 | 0,01% | 50,92 | **21,8%** |

**G-N3-S1 gehaald** (nul valse certificaten), **G-N3-R1 gefaald** met een factor
3000.

De laatste kolom is de verklaring: **rang 64 vangt maar 21,8% van de spectrale
energie** van een matrix van rang 1856. Deze expertgewichten zijn vrijwel
vol-rang. Er is geen laag-rang-structuur om op te steunen, dus elke
laag-rang-schatting laat een residual over die de grens onbruikbaar maakt.

Dat is een onafhankelijke bevestiging van waarom de RSIV-lijn met laag-rang
surrogaten weerlegd werd, en het sluit die deur nu ook voor het *exacte* gebruik
— niet als vervanging maar als bewijs.

## N4 — attention is byte-gebonden, en 7× van zijn roofline

| context | ms | KV-bytes | effectief |
|---:|---:|---:|---:|
| 32.768 | 1,640 | 96,0 MiB | 61,4 GB/s |
| 65.536 | 4,583 | 192,0 MiB | 43,9 |
| 131.072 | 9,043 | 384,0 MiB | 44,5 |
| 196.608 | 12,930 | 576,0 MiB | 46,7 |
| 262.100 | 17,047 | 767,9 MiB | 47,2 |

Fit: **21,48 ms per GB plus −0,033 ms vast, R² = 0,9964**. De vaste term is
praktisch nul: attention is **volledig byte-gebonden**.

Twee gevolgen, en het tweede is groter dan het eerste:

1. **Halvering van de KV-bytes halveert de tijd**: 17,047 → 8,615 ms bij 262K,
   een besparing van 8,43 ms. Een FP4-KV zou dat opleveren — met een
   kwaliteitspoort die hier niet gehaald of geclaimd wordt.
2. **De kernel draait op 47,2 GB/s waar het apparaat 338,4 haalt** — een factor
   **7,2**. Dat is geen bytesprobleem maar een kernelprobleem. Zou attention de
   helft van de roofline halen, dan zakt hij van 17,0 naar 4,8 ms zónder één byte
   te besparen en zonder enige semantiekwijziging.

---

## Waar het gat zit, nu voor het eerst opgeteld

| post | gemeten | status |
|---|---:|---|
| uitgifte-overhead (N1) | 23,7% van het token | bovengrens, mechanisme bekend |
| attention onder roofline (N4) | 47,2 van 338,4 GB/s | 7,2× ruimte |
| GEMV onder roofline (Y2-R1) | 81,4 van 338,4 GB/s | 4,2× ruimte |
| gather in de lus (N2) | 8,19 ms, 4,3 GB/s effectief | 6× slechter dan geïsoleerd |

Alle vier zijn **efficiëntie**, geen bytes en geen semantiek. En de roofline zegt
dat er een factor 6 aan efficiëntie tussen de huidige stand en de bytevloer zit.

## Poorten

| poort | uitkomst |
|---|:--|
| G-N1-1 verwijderbaar | 23,7% gerapporteerd |
| G-N1-2 zelfde kernelreeks | ✅ capture geslaagd, bevroren routes |
| G-N2-1 scan+gather ≥ 30% | ✅ 67,2% |
| G-N3-S1 soundness | ✅ nul valse certificaten |
| G-N3-R1 ≥ 30% gecertificeerd | ❌ 0,01% |
| G-N4-2 fit R² ≥ 0,98 | ✅ 0,9964 |
| G-N5-1/2 vloer en bandbreedte | ✅ gemeten, 338,4 GB/s |

Verifier **36/36**, `VERIFIED`. Protected 0 modified / 0 removed.

## Claim boundary

N1 herhaalt een vastgelegde kernelreeks met **bevroren routes**; dat is na de
eerste token semantisch onjuist en wordt uitsluitend als tijd-oracle gebruikt
voor de vraag hoeveel van een token uitgifte is in plaats van rekenwerk. N2's
marginalen zijn in-lus ondergrenzen en tellen niet op tot het token; `panel_scan`
ligt onder zijn eigen ruisvloer en krijgt geen waarde. N4 timet de
attention-kernel alleen, op dezelfde KV-allocatie bij verschillende diepten, dus
het meet de helling van de kernel en niet het token. N5's vloer telt de bytes die
een correcte forward moet lezen en deelt door de **gemeten** streaming-bandbreedte
van dit apparaat; het is een **harde bovengrens** op tokens per seconde voor elke
implementatie die de semantiek behoudt, geen haalbaar cijfer en geen meting van
deze runtime. N3 is een numerieke oracle; de gecertificeerde rijen zijn bewijsbaar
nul, en de niet-gecertificeerde worden exact berekend, dus de uitvoer zou
bit-identiek blijven.

## Artefacten

`N1_N5_OWN_HYPOTHESES_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/n1n2n4n5_ceilings.py` · `n1n2n4n5_ceilings.json` ·
`scripts/lightningstream_nemotron/n3_relu2_prefilter_oracle.py` ·
`n3_relu2_prefilter_oracle.json` ·
`scripts/lightningstream_nemotron/n1_n5_independent_verify.py` ·
`n1_n5_independent_verification.json` · `protected_verification_after_n5.json`
