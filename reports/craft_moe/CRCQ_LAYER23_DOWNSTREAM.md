# H1 CRCQ — laag 23 met exacte lagen 24–26

Datum: 2026-08-10  
Status: `downstream_falsified`  
Machineleesbaar resultaat: `reports/craft_moe/crcq_layer23_downstream.json`

## Vraag en vaste methode

Na de positieve volledige laag-26-oracle zijn op laag 23 opnieuw alle 59.136
route×Q3/Q4-kandidaten per token geëvalueerd. Selectie gebruikte uitsluitend
lokale routed-output-MSE tegen natural BF16. De gekozen sequencepolicies zijn
daarna zonder verdere interventie door de officiële lagen 24, 25 en 26
uitgevoerd. Alleen finale KL/CE/top-1 bepaalde de downstreamgate.

## Lokale oracle

| Split | Natural minimum | Joint minimum | Joint gem. bit | Relatieve reductie | All-Q3 route-gap closure |
|---|---:|---:|---:|---:|---:|
| Validatie, 256 | 85,156% | **75,521%** | 3,7552 | 11,31% | 0,16% |
| Test, 256 | 80,534% | **70,508%** | 3,7051 | 12,45% | 1,12% |

De vereiste ≤15%-gate faalt dus zeer ruim en beide waarden overschrijden de
vooraf vastgelegde 25%-falsificatiegrens. De blockbootstrap bevestigt dat dit
geen grensgeval is: joint `74,414–77,018%` op validatie en
`68,164–73,438%` op test. Routevrijheid sluit vrijwel niets van de lokale
Q3→Q4-MSE-gap; het gunstige finale-KL-landschap van laag 26 is geen lokaal
Euclidisch compressielandschap op laag 23.

## Exacte downstreammetrics

| Policy | Validatie-KL | Test-KL | Val. rel. CE | Test rel. CE |
|---|---:|---:|---:|---:|
| Natural all-Q4 | 0,002843 | 0,003971 | +0,499% | −0,047% |
| Natural lokaal minimum | 0,002834 | 0,003766 | +0,426% | −0,045% |
| Joint lokaal minimum | 0,002417 | 0,003938 | −0,063% | −0,282% |
| Natural bij 15% | 0,005233 | **0,005011** | +0,678% | +0,150% |
| Joint bij 15% | **0,004976** | 0,005088 | +0,278% | +0,001% |

Met 70–76% upgrades blijft joint kwalitatief goed, maar dat is geen nuttige
compressie. Bij het beoogde 15%-budget wint joint 4,91% KL op validatie en
verliest 1,52% op test. Het voordeel reproduceert dus niet eens in richting.

De routerdynamiek blijft op zichzelf vrij stabiel: voor joint-15% is de
top-6-overlap door lagen 24–26 ongeveer 95,5–97,0% op validatie en
96,2–97,7% op test. Dat redt de mislukte byte-/bitgate niet.

## Controles

- De apart door de volledige tail gevoerde BF16-deltapatchcontrol geeft exact
  KL `0`, top-1 `1` en CE-delta `0` op beide splits.
- Top-6-ID-sets en routergewichten van de top-12-constructie matchen de
  officiële natural route exact.
- De lokale gekozen schedule reproduceert de DP binnen `1e-7`.
- 52/52 tests slagen.
- De berekening duurde 82,59 s vóór serialisatie en gebruikte circa 2,00 GB
  piek-CUDA-allocatie. Het 963,6 MiB JSON bevat beide volledige lokale
  damage-cubes, policies, routermetrics en alle finale tokenmetrics.

## Stop/go

**Stop H1.** De joint route×bit-vrijheid is een sterk en volledig bewezen
laat-laag-oracleplafond, maar zij overleeft de vooraf vereiste eerdere
interventie niet bij een bruikbaar budget. Er wordt daarom geen learned
CRCQ-selector, 1.024-token-kandidaatvalidatie of confirmatory CRCQ-window
geopend. Het positieve laag-26-resultaat en deze negatieve downstreamgrens
blijven beide staan.

**Go naar de volgende onafhankelijke P0-richting H3 (atomic expert oracle).**

