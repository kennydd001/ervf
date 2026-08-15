# H10 Numerical Reduction-Order Compensation

## Definitief oordeel

**H10 is op de vooraf vastgelegde held-out stopregel hard gefalsificeerd.** De
uitsluitend op validatie gekozen vaste Q3-reductie sluit `1,487%` van de
Q3→Q4-KL-kloof op validatie en slechts `0,829%` op test. De vereiste sterke
gate was minstens 20% op beide; minder dan 10% op test was een harde stop.

De gekozen configuratie is BF16 sequentiële accumulatie met slotorde
`[3,5,1,4,0,2]`. Zij verlaagt de Q3-KL heel licht, maar verslechtert test-top-1
van `91,41%` naar `91,02%`. Beschermde BF16-operands die naar FP32 worden
gepromoveerd zijn over alle 720 ordes exact invariant. De beste op validatie
gekozen FP32-orde sluit op test exact `0%` van de kloof. Er volgt daarom geen
reducerbenchmark, laag-23-interventie, spread of full-depthproef.

## Vaste orde versus Q3/Q4

| Metric | Validatie | Test | Gate |
|---|---:|---:|---:|
| Q3-reference KL | 0,0128913 | 0,0385477 | — |
| Q4-reference KL | 0,00262941 | 0,00415433 | target |
| Q3→Q4-KL-kloof | 0,0102619 | 0,0343934 | positief |
| vaste Q3-KL | 0,0127387 | 0,0382625 | ≤Q3 |
| vaste gapclosure | **1,487%** | **0,829%** | ≥20% |
| 95%-blockbootstrap | 0,924–2,261% | 0,672–0,886% | — |
| FP32-controlclosure | ≈0% | 0% | hard ≥10% |
| per-token lokale MSE-oracleclosure | 0,831% | 1,183% | diagnostisch |
| Q3-reference relatieve CE | +1,451% | +2,247% | — |
| vaste Q3 relatieve CE | +1,385% | +2,216% | — |
| Q3-reference top-1 | 94,14% | 91,41% | — |
| vaste Q3 top-1 | 94,14% | 91,02% | — |

De gepaarde 10.000× bootstrap gebruikt de twee vooraf vaste 128-tokenblokken.
Geen enkele resample haalt 10%, ook niet voor de per-token lokale MSE-oracle;
met slechts twee sampling units blijven de intervallen beschrijvend. De
test-hard-stop rust op de vooraf geregistreerde puntschatting, niet op de CI.

Dezelfde vaste orde verhoogt Q4-KL licht van `0,002629` naar `0,002714` op
validatie en van `0,004154` naar `0,004186` op test. Er is dus geen verborgen
Q4-robustheid die het negatieve Q3-resultaat relativeert.

## Volledige 720×8×2-sweep

Alle 720 lexicografische ordes zijn voor Q3 en Q4, beide splits en acht vaste
schema's geëvalueerd: sequentieel/boom met FP32-, BF16- en FP16-accumulatie plus
BF16-operands beschermd in FP32.

Op Q3-validatie levert BF16-sequentieel 360 verschillende MSE-profielen en
maximale routed spreiding `0,5` na de finale BF16-cast. De beste vaste orde
verlaagt routed MSE echter slechts `0,0175%`. Op test kan een achteraf gekozen
BF16-orde lokaal `0,480%` MSE reduceren, maar die testkeuze is uitsluitend een
diagnostiek en mag de validatieorde niet vervangen.

FP32-sequentieel heeft 316/318 MSE-profielen op validatie/test, maar haar beste
lokale MSE-winst is minder dan `1e-7` relatief en de exacte KL-closure is nul.
De paperachtige beschermde modus—BF16-termen exact gepromoveerd naar FP32—heeft
voor Q3 én Q4 op beide splits precies één profiel en maximale outputspreiding
`0,0`. Dit reproduceert de relevante numerieke compatibiliteitscontrol, maar
levert geen compensatiemechanisme.

De per-token oracle kiest vooral BF16-sequentieel (196/256 validatietokens),
maar zijn exacte KL-closure blijft slechts `0,831%` validatie en `1,183%` test.
Omdat selectie op routed MSE en niet op alle 720 vocabulaire-KL's gebeurde, is
dit geen absoluut KL-oracle. Het is wel extra negatief bewijs: zelfs de veel
vrijere tokenafhankelijke numerieke keuze blijft twee ordes van grootte onder
de 20%-doelstelling.

## Controls, accounting en prior art

- officiële expert-ID's en slotvolgorde zijn bitexact; maximale
  routergewichtfout in de full run is `0,0`;
- BF16, Q3 en Q4 zijn in dezelfde 512-tokenbatch uit dezelfde gewichten
  herberekend en alle outputs zijn finite;
- de officiële teacher-delta-original-control is vóór en na capture bitexact;
- vergelijking met de oudere opgeslagen Q3/Q4-tensors blijft expliciet een
  batchvormregressiediagnostiek (`NRMSE 0,00347/0,00257`), geen control;
- alle kandidaten gebruiken exact dezelfde zes termen, nul extra weightbytes
  en nul metadata; alleen addorde/rounding verandert.

Een fysieke reducerbenchmark was alleen toegestaan na positieve inhoudelijke
gates en is dus bewust niet uitgevoerd. “Vijf additions” betekent niet
automatisch gelijke throughput; er is geen latency- of snelheidsclaim.

De nabije primaire studie [From Expert Reduction to Behavioral Divergence
(arXiv:2607.28097)](https://arxiv.org/abs/2607.28097) toont gecontroleerde
causale divergentie en definieert accumulatorsemantiek als compatibiliteits-
contract. Zij claimt geen systematische kwaliteitswinst. H10 voegt hier een
negatief held-out V2-Lite-resultaat aan toe en maakt geen noveltyclaim.

## Artefacten en reproduceerbaarheid

- hoofdresultaat: `reduction_order.json`, 1.130.069 bytes, SHA-256
  `59497a28a0bacf791bb47cf1ef13f216caac6fb4bc6a1d17da874ad089f370ef`;
- same-batch capture: `reduction_order_capture.safetensors`, 46.161.040 bytes,
  SHA-256
  `c2cfcedf1e147ee7fec5a73adece667fb45bea9a4ba88a03582284890e0c5079`;
- volledige per-order/per-token-MSE: `reduction_order_raw.safetensors`,
  27.802.080 bytes, SHA-256
  `b47ac1cc872e43f8c43bb0a2d8d162faf48cd45ec201a15fd0d278421f97e7c9`;
- append-only gepaarde closure-audit: 1.854.185 bytes, SHA-256
  `7735559957f79ea24a99b0959f33f4922c1c9e9bfbdc822ff746c90b6c73cb39`.

De full run duurde `15,81 s`. JSON bevat exacte KL/CE/top-1-tokenseries en
bootstrap-CI's; het safetensors-rawartifact bevat alle 8×720×256 MSE-series per
bitbreedte/split, oracleselecties en routed oracleoutputs. Commands, hashes,
libraryversies, hardwarestaat, trace-indices en de dirty repository zonder
commit zijn vastgelegd. De geslaagde 32-token-smoke blijft apart behouden.

**Stop H10:** geen throughputbenchmark, laag 23, spread of full-depth. Daarmee
zijn alle niet-geblokkeerde technische hypothesen uit het huidige CRAFT-pakket
gescreend; het programma gaat nu naar een onafhankelijke reproduceerbaarheids-
en eindverdictaudit, niet naar post-hoc varianten.
