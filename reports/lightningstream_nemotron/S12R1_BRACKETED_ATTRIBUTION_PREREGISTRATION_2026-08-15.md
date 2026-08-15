# S12-R1 — gebracketeerde in-lus attributie: preregistratie-addendum

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.** Geschreven ná S12's resultaat, dat volledig
en ongewijzigd blijft staan in `s12_in_loop_attribution.json`.

## 1. Waarom een herhaling nodig is

S12 haalde G-S12-C1 (identiteit) en G-S12-S1 (sanity) maar **faalde op
G-S12-D1**: `base2 − base1` was 5,057 ms bij 262100, groter dan de kleinste
gerapporteerde marginale kosten (3,898 ms). Bij ctx 0 was de drift 2,331 ms.

De drift is bovendien **eenzijdig**: `base2` is in beide contexten trager dan
`base1`. Dat is geen ruis maar een trend, en er is een voor de hand liggende
oorzaak: de vijf probe-armen dóén meer werk dan de basislijn en verwarmen de GPU,
en `base2` draaide na alle vijf. Ter vergelijking: S11's drift tussen twee
identieke armen was 0,042 ms bij dezelfde contextdiepte, dus de meetlus zelf is
stabiel als alle armen even zwaar zijn.

**De poort wordt niet verruimd.** S12's marginalen bij 262100 blijven
gerapporteerd zoals ze zijn, met `up`, `shared` en `accum` onder de ruisvloer.
Wat hier verandert is het *ontwerp*, niet het criterium.

## 2. De wijziging: één variabele, het meetschema

Elke probe-arm wordt **omsloten** door een basislijn-arm:

```
base0 · up · base1 · down · base2 · router · base3 · shared · base4 · accum · base5
```

- marginale kosten van P = `p50(P) − ½ · ( p50(base vóór P) + p50(base ná P) )`
- lokale drift bij P = `| p50(base ná P) − p50(base vóór P) |`

Een lineaire trend in de tijd valt daarmee weg uit elke marginale waarde, en elke
probe krijgt zijn **eigen** ruisvloer in plaats van één globale.

Verder identiek aan S12: dezelfde subklasse-probe, dezelfde componenten,
capacity 70, contexten 0 en 262.100, hetzelfde warm-up- en sampleprotocol,
`runtime.py` onaangeraakt. Nieuw is alleen dat de GPU-temperatuur bij elke
sweep-rij wordt vastgelegd, zodat de thermische verklaring gedocumenteerd is en
niet aangenomen.

## 3. Poorten

- **G-S12R-C1 — semantiek.** Generatie bit-identiek in alle elf armen.
- **G-S12R-D1 — lokale drift.** Een marginale waarde wordt alleen als waarde
  gerapporteerd als zij groter is dan haar **eigen** lokale drift. Marginalen
  die dat niet halen heten "onder de ruisvloer" en krijgen geen getal
  toegekend in de conclusie.
- **G-S12R-S1 — sanity.** De som van de gerapporteerde marginalen mag de gemeten
  MoE-term uit S8 (39,523 ms bij 262K) niet overschrijden.
- **G-S12R-T1 — thermisch.** De globale drift `|base5 − base0|` wordt
  gerapporteerd naast de temperatuur bij de eerste en laatste basislijn. Als de
  globale drift groter is dan de grootste marginale waarde, is ook dit schema
  ontoereikend en wordt de fase als niet-conclusief afgesloten in plaats van
  geherinterpreteerd.

## 4. Wat ongewijzigd blijft

Marginalen blijven **ondergrenzen** (warme L2, meer ruimte voor de copy-stream),
worden niet omgerekend naar aandelen of naar tok/s, en de niet-gedekte rest
krijgt geen naam.

## 5. Artefacten

`scripts/lightningstream_nemotron/s12r1_bracketed_attribution.py` ·
`s12r1_bracketed_attribution.json` ·
`scripts/lightningstream_nemotron/s12r1_independent_verify.py` ·
`s12r1_independent_verification.json`

## 6. Claim boundary

Geen meting, geen resultaat.
