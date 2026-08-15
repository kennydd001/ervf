# H1 CRCQ — preregistratie volledige 59.136-route-maskeroracle

Vastgelegd na de vooraf toegestane sterke top-32-screen en vóór uitvoering van
de volledige route×bitmeting. De screenresultaten of gates worden niet
gewijzigd.

## Vaste uitbreiding

- dezelfde model-/datasetcommits, laag 26, eerste 256 validatie- en eerste 256
  bestaande testtokens;
- dezelfde opnieuw berekende BF16/Q3/Q4-outputbatch, Q3/Q4-quantizer,
  ongenormaliseerde routergewichten en officiële-teacher-deltapatch;
- alle `924×64=59.136` route-maskerkandidaten per token;
- exact volledige-vocabulaire teacher→candidate-KL in batches van 128;
- dezelfde globale DP en hetzelfde kwaliteitsdoel:
  `1,01×` gemiddelde natuurlijke all-Q4-KL per split;
- geen aanpassing op basis van de bestaande testresultaten.

De natuurlijke BF16-control moet opnieuw exact zijn. De natuurlijke Q3/Q4-KL
en natuurlijke minimum-upgradefractie moeten binnen `1e-7` respectievelijk één
discrete upgrade overeenkomen met de top-32-screen; een grotere afwijking stopt
de run als technische fout.

## Gates

De volledige oracle is `full_oracle_positive` wanneer:

1. de minimum-upgradefractie hoogstens 15% is op validatie én test; en
2. zij niet slechter is dan de overeenkomstige top-32-fractie; en
3. de direct opnieuw geëvalueerde gekozen schedule-KL het DP-resultaat binnen
   `1e-6` reproduceert.

Rapporteer daarnaast de verbetering van full versus top-32, de routewissel-
frequentie, BF16-KL van gekozen routes, CE/top-1 en block-bootstrapintervallen.
Een testinterval dat 15% kruist blijft expliciet als onzekerheid staan.

Alleen bij `full_oracle_positive` wordt H1 op laag 23 geopend met exacte lagen
24–26. De volledige zoekruimte blijft een teacher-oracle en rechtvaardigt geen
deployment- of snelheidsclaim.

