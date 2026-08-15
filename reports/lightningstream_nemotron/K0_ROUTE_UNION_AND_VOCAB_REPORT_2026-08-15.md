# K0/K1/K2 — Kimi's P0 gemeten, en wat het wel en niet beslist

Datum: 2026-08-15
Verdict: **Kimi's correctie op mijn drempel is terecht; mijn 12 was fout, de pariteit is 18,683. Maar de gemeten unie bij B=5 is 19,880 — nét erboven. P0 komt daarmee negatief uit op zijn eigen criterium: 6,384 expert-records per uitgestoten token tegen 6 autoregressief, en 1,62× zoveel cache-misses. De LM-kop blijkt 51% van de draft-keten; het actieve vocabulaire snijdt die weg (−44,9%) maar haalt de recall-poort niet.**
Terminal state: `k0_union_above_parity_p0_negative_but_not_time_decisive`
Preregistratie: `K0_ROUTE_UNION_AND_VOCAB_PREREGISTRATION_2026-08-15.md`

## 0. Eerst: Kimi had gelijk over mijn drempel

Mijn S10-A-rapport stelde voor een vijf-token-unie boven ~12 als negatief te
behandelen. Dat was fout gerekend. Bij een unie van 12 en 3,114 uitgestoten
tokens is dat 3,85 records per token tegen 6 — een verbetering van 36%. De
juiste pariteit is die van Kimi:

```
U* = top_k × (A+1) = 6 × 3,1139 = 18,683 unieke experts per laag
```

Die grens is overgenomen en er is verder geen enkele drempel verzonnen: de
AR-basislijn van 6 records per uitgestoten token ís het criterium.

## 1. K0 — de census, op de officiële routes

`step(capture_routes=…)` levert de routes die de runtime zélf gebruikt, geen
herberekende top-k. Drie bevroren domeinprompts, 120 greedy gegenereerde tokens
elk, ≥ 300 vensters per `B`. Verifier 62/62.

| B | D | gem. unie | uitgestoten | records/uitgestoten | vs. AR 6 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 10,06 | 1,786 | **5,631** | −6,2% |
| 3 | 2 | 13,63 | 2,378 | **5,734** | −4,4% |
| 5 | 4 | **19,88** | 3,114 | **6,384** | **+6,4%** |
| 7 | 6 | 25,23 | ongemeten | — | — |
| 9 | 8 | 30,01 | ongemeten | — | — |
| 13 | 12 | 38,24 | ongemeten | — | — |

**G-K0-1 gefaald:** 6,384 ≥ 6. De unie bij B=5 ligt 6,4% boven pariteit.

Voor `D > 4` is `A` niet gemeten, dus daar staat geen quotiënt. Wat wél volgt uit
de unie alleen: pariteit bij B=7 vraagt 4,21 uitgestoten tokens, bij B=9 5,00,
bij B=13 6,37. De gemeten acceptatieladder is 0,786 / 0,753 / 0,728 / 0,710;
bleef die op ~0,71 hangen, dan komt D=8 uit op ~3,66 uitgestoten — ruim onder de
5,00 die pariteit vraagt. **Dat is een extrapolatie van een gemeten ladder, geen
meting**, maar de richting is structureel: de unie groeit met ~2 à 3,6 experts
per extra token terwijl de uitgestoten tokens geometrisch afvlakken. Dieper
drafting maakt het slechter, niet beter. Het byte-optimum ligt bij **D=1**, en
dat is 6,2% — geen pad naar 2,7×.

### Cache-replay

De per-laag-LRU opnieuw afgespeeld over dezelfde routes, in AR-orde en in
ronde-orde (een ronde vraagt de unie van 5 tokens in één keer op):

| capacity | AR misses/token | ronde-B5 misses/uitgestoten | ratio |
|---:|---:|---:|---:|
| 32 | 51,81 | 84,73 | 1,635 |
| 48 | 37,83 | 61,71 | 1,631 |
| 56 | 32,45 | 52,88 | 1,630 |
| 60 | 30,23 | 49,24 | 1,629 |
| 64 | 28,15 | 45,74 | 1,625 |
| **72** | **24,63** | **39,94** | **1,622** |

**G-K0-2 gefaald** bij elke capacity: ronde-orde kost ~1,62× zoveel misses per
uitgestoten token. De hitrate zakt van 0,822 naar 0,731 omdat een ronde 19,9
unieke experts per laag ineens opvraagt en er maar 3,1 token uitkomt.

*Meetnotitie.* De runner deelde het totaal aantal misses door de gemiddelde
**serie**lengte in plaats van door tokens × prompts, waardoor zijn
`*_per_emitted`-velden precies een factor 3 (het aantal prompts) te hoog staan.
De poort is een verhouding en blijft daardoor ongewijzigd; de tabel hierboven
komt uit de verifier, die het onafhankelijk herberekent en tegen de hitrate
kruiscontroleert (24,63 = 23 × 6 × (1 − 0,8215) ✓). De ruwe JSON behoudt het
onjuiste veld met deze notitie erbij.

## 2. K1 — waar de draft-keten zijn tijd laat

Gebracketeerde replicatie, S12-R1-protocol, globale drift **0,192 ms** over elf
armen — de stabielste meting van de hele lijn.

| gerepliceerd | marginaal per keten van 4 drafts | lokale drift |
|---|---:|---:|
| **`lm_head`** | **+10,508 ms** | 0,365 |
| 6 experts | +3,546 | 0,392 |
| attention | +1,262 | 0,265 |
| shared expert | +0,824 | 0,329 |
| `eh_proj` | +0,733 | 0,101 |
| som | 16,87 | van een basislijn van 20,66 ms |

**De LM-kop is ~51% van de draft-keten.** De experts zijn 17%. Dat herschikt
Kimi's H1 volledig: kwantiseren van de MTP-experts (H1 punten 1–3) valt de
kleinste post aan; het actieve vocabulaire (punt 4) valt de grootste aan.

Marginalen zijn ondergrenzen en worden niet naar aandelen of tok/s omgerekend;
de 51% hierboven is de verhouding van twee gemeten grootheden binnen dezelfde
meting, geen toerekening van het geheel.

## 3. K2 — het actieve vocabulaire, één variabele

Per commit-positie de top-`N` rijen van de logits die de **backbone zelf** al
berekend heeft, en de vier drafts projecteren alleen daarover. Backbone-verificatie
ongewijzigd, dus dit kan de uitvoer alleen via acceptatie raken.

| `N` | recall doel-token | gepoolde `A` | keten p50 | Δ keten |
|---|---:|---:|---:|---:|
| vol (131.072) | — | **2,1139** | 18,83 ms | — |
| 4.096 | 0,9319 | 2,0028 | **10,52 ms** | **−44,9%** |
| 2.048 | 0,8889 | 1,9278 | 10,55 ms | −44,8% |
| 1.024 | 0,8201 | 1,8139 | 10,62 ms | −44,4% |

| poort | vereist | 4.096 | |
|---|---|---:|:--:|
| G-K2-T1 keten | −30% | −44,9% | ✅ |
| G-K2-R1 recall | ≥ 99,5% | 93,2% | ❌ |
| G-K2-A1 acceptatie | ≥ 2,064 | 2,003 | ❌ |

**Geen enkele `N` haalt alle drie.** De tijdwinst is precies wat K1 voorspelde:
18,83 → 10,52 is 8,31 ms voor vier drafts, oftewel 2,08 ms per `lm_head` — tegen
de 2,106 ms die S8 los mat. Twee onafhankelijke methoden, hetzelfde getal.

De keten-tijd is verder vlak over `N` (10,42–10,75 ms), dus onder 4.096 gaan
kost alleen acceptatie en levert niets op.

Wat hier eerlijk bij hoort: de ruil bij `N` = 4.096 is 5,3% acceptatie voor 44,9%
kortere draft. In doorvoer klinkt dat gunstig. De poorten waren vooraf
vastgelegd en worden niet herschreven omdat de ruil aantrekkelijk oogt — maar
het is de selector die faalt, niet het idee. Top-`N` van de huidige
backbone-logits is de simpelste denkbare selector; Kimi's 99,5% vraagt er een
betere (bijvoorbeeld unie over meerdere posities, of aanvullen met
prompt-/n-gram-continuaties). Dat is een eigen preregistratie waard.

## 4. Wat dit voor de 50 tok/s betekent

De gemeten doorvoer is **onveranderd**, want er is niets gebouwd:
27,574 @ctx0 · 25,523 @32K · 21,794 @128K · 18,358 @262K.

Twee dingen zijn nu wél bekend, en ze wijzen tegengesteld.

**Tegen.** Kimi's eigen P0-criterium komt negatief uit: bij D=4 kost een ronde
6,4% méér expert-records per uitgestoten token dan gewoon decoderen, en 1,62×
zoveel cache-misses. Dieper drafting verergert dat. Alleen D=1 en D=2 zitten aan
de goede kant, met 6,2% respectievelijk 4,4%.

**Vóór, en dit is de kern.** P0 meet **records**, en S12 heeft gemeten dat de
MoE-term niet expert-load-gebonden is: van de 39,523 ms zijn de per-expert
marginalen (`down` 7,478 + `up` 4,756) samen 12,23 ms, en `shared` 3,30 ms. Als
de overige ~24 ms per *laag* is en niet per *expert*, dan kost een B=5-sweep die
3,31× zoveel experts laadt: 12,23 × 3,31 + 24 ≈ 68 ms voor 3,114 tokens = **21,8
ms per token tegen 39,5 nu**. Schaalt de hele term daarentegen met het aantal
experts, dan wordt het 131 ms = 42 ms per token, dus slechter.

Die twee lezingen verschillen een factor twee en **alleen een meting scheidt
ze**. Dit is dezelfde les als S8 en S11: byte-boekhouding voorspelt op deze
runtime geen tijd. P0 was Kimi's eigen eerste fase en komt negatief uit, maar
P0 meet niet de grootheid die hier beslist.

### De optimistische bovengrens, uitdrukkelijk aritmetiek

Neem van elke term de **gunstigste** lezing die met gemeten componenten
verenigbaar is, bij 262K: MoE 21,8 · attention 18,634/3,114 = 5,98 (KV één keer
gelezen voor 5 queries) · Mamba ~3,5 (projecties één keer per sweep) · `lm_head`
0,68 · compacte draft 10,52/3,114 = 3,38. Samen **≈ 35,4 ms per uitgestoten
token**, tegen 54,28 ms nu.

Dat is **geen meting en geen voorspelling**. Het is wat je krijgt als je elke
ongemeten term zijn theoretische beste waarde geeft en de hele P1-verifier
(ReplaySSM-Mamba, één GQA-sweep, expert-major MoE-GEMM, gebatchte LM-kop)
werkend bouwt. De 20 ms per token die 50 tok/s vraagt, zit daar niet in — ook
niet bij benadering, en ook niet bij 128K.

## 5. Wat er níét getest is, en waarom

P1 (exacte B-token-verifier), P3–P8 en H4–H8 zijn bouw- of trainingsprojecten,
geen metingen. Ze zijn hier niet "getest" en er wordt niets over beweerd. De
eerstvolgende meting die het meest oplevert is **niet** de hele P1-verifier maar
een afgebakend stuk ervan: de **tijd** van één MoE-laag over de unie van vijf
token-routes met expert-major groepering, tegen 5× het huidige per-token-pad.
Dat scheidt de twee lezingen in §4 en is een fractie van het werk.

## 6. Claim boundary

Route-unies en LRU-replays zijn geteld op de **officiële** routes die deze
runtime tijdens echte greedy generatie uitzendt, niet op een herberekende top-k.
De LRU-replay is een simulatie over die routes, geen getimede meting. Acceptatie
en recall zijn gemeten; keten-tijden zijn componentmetingen met alle 128
MTP-experts device-resident en worden niet naar tokens per seconde omgerekend.
Er is geen speculatieve lus gebouwd, dus niets hiervan is een doorvoerresultaat.
`A` voor D > 4 is niet gemeten; daar staat alleen de unie. De rekensom in §4 is
aritmetiek op gemeten componenten met best-case aannames op ongemeten termen, en
is expliciet geen resultaat.

## 7. Artefacten

`K0_ROUTE_UNION_AND_VOCAB_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/k0_route_union_census.py` ·
`k0_route_union_census.json` ·
`scripts/lightningstream_nemotron/k0_independent_verify.py` ·
`k0_independent_verification.json` ·
`src/moe_lab/lightningstream_nemotron/mtp.py` (actief vocabulaire) ·
`protected_verification_after_k0.json`
