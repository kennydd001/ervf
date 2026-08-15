# H1 CRCQ — preregistratie laag-23-interventie met exacte modeltail

Vastgelegd nadat de volledige laag-26-oracle positief was en vóór uitvoering
van laag-23-code of laag-23-metriekinspectie.

## Vraag en afbakening

Een volledige finale-KL-enumeratie van 59.136 kandidaten voor ieder laag-23-
token is door causale interacties in lagen 24–26 geen onafhankelijke
per-tokenoptimalisatie. Deze run claimt dat daarom niet. Zij meet een vaste,
exhaustieve **lokale routed-outputoracle** en voert de eenmaal gekozen sequence-
schedule daarna exact door de echte lagen 24, 25 en 26.

De lokale selector gebruikt geen tokenlabels of finale test-KL. Alleen de
finale tailmetrics bepalen het downstreamoordeel.

## Vaste input en constructie

- dezelfde gepinde DeepSeek-V2-Lite- en WikiText-commits;
- laag 23, eerst een smoke op 32 validatietokens, daarna 2×128 validatie- en
  2×128 bestaande testtokens;
- exacte officiële prefixlagen 0–22 en officiële laag-23-teacher;
- per token top-12, alle 924 routes en alle 64 Q3/Q4-maskers;
- originele, ongenormaliseerde routergewichten;
- dezelfde symmetric per-row Q3/Q4-quantizer;
- kandidaatpatch:
  `BF16(official_teacher23 + candidate_routed - natural_BF16_routed)`;
- daarna officiële lagen 24–26 zonder quantisatie of routeoverride.

De natural-BF16-patch moet exact zijn. Alle 59.136 lokale route-maskerfouten
worden bewaard. Lokale schade is gemiddelde kwadratische routed-outputfout
tegen de natuurlijke BF16-routed output. De globale DP gebruikt als doel
`1,01×` de lokale schade van natural all-Q4.

## Vooraf vastgelegde policies

In één gedeelde exacte-tailbatch:

1. officiële BF16-teacher/control;
2. natural all-Q3;
3. natural all-Q4;
4. natural-route minimum-DP bij lokaal all-Q4-doel;
5. full joint minimum-DP bij lokaal all-Q4-doel;
6. natural-route beste schedule binnen het werkelijke joint-minimumbudget;
7. natural-route beste schedule binnen 15% upgrades;
8. full joint beste schedule binnen 15% upgrades.

Per downstreamlaag worden hidden NRMSE, router-top-6-overlap en
routergewicht-NRMSE gerapporteerd. Finale metrics zijn volledige-vocabulaire
teacher→candidate-KL, CE, top-1 en gepaarde sequence-block-bootstrapintervallen.

## Gates

`downstream_positive` vereist op validatie én de vaste testreplicatie:

1. lokaal full-joint minimum-upgradepercentage ≤15%;
2. finale KL van `joint minimum-DP` ≤`1,10×` finale natural-all-Q4-KL;
3. absolute relatieve CE-delta van `joint minimum-DP` <2%;
4. bij een budget van 15% heeft joint finale KL niet hoger dan natural;
5. de BF16-control is exact.

De 10%-KL-marge is vooraf ruimer dan de lokale 1%-doelwaarde omdat de selector
geen finale-KL gebruikt. Rapporteer ook joint versus natural bij exact hetzelfde
werkelijke joint-minimumbudget; dit is een secundaire dominantiecheck.

`downstream_falsified` wanneer op een split de joint-minimum-KL meer dan 25%
boven natural all-Q4 ligt, de absolute relatieve CE-delta minstens 2% is, of de
lokale minimum-upgradefractie boven 25% ligt. Andere gatepatronen zijn
`inconclusive`.

Alleen `downstream_positive` mag een grotere 1.024-token-kandidaatvalidatie en
daarna een nieuw confirmatory venster openen. Ook een positieve uitkomst blijft
een teacher-oracle zonder goedkope selector of packed-runtimeclaim.

