# H1 CRCQ — top-32 exacte-KL-oraclescreen

Datum: 2026-08-10  
Status: `strong_positive` screen  
Machineleesbaar resultaat: `reports/craft_moe/crcq_oracle.json`

## Hypothese en vaste screen

H1 test of alternatieve top-12-kies-6-routes als een foutcorrigerende codebook
voor Q3/Q4-quantisatie werken. Voor ieder token zijn eerst alle 924 all-Q3-
routes met exacte volledige-vocabulaire-KL gemeten. Daarna zijn op de 32 beste
routes alle 64 Q3/Q4-maskers exact geëvalueerd. De natuurlijke route werd
altijd opgenomen. Routergewichten bleven origineel en ongenormaliseerd.

De sterke preregistratiegate was een minimum van hoogstens 15% Q4-upgrades
(gemiddeld hoogstens 3,15 actieve bit) bij `1,01×` de natuurlijke all-Q4-KL,
met dezelfde richting op validatie en de vaste testreplicatie.

## Resultaat

| Split | Natural 3→4 | Joint route+bit | Gem. bit | Relatieve upgradereductie | All-Q3 gap closure |
|---|---:|---:|---:|---:|---:|
| Validatie, 256 | 20,313% | **11,263%** | **3,1126** | 44,55% | 51,65% |
| Test, 256 | 22,461% | **14,128%** | **3,1413** | 37,10% | 25,15% |

Dezelfde vooraf vastgelegde joint-gate slaagt dus op beide splits. De route-only
all-Q3-claim is minder stabiel: zij sluit op validatie meer dan de helft van de
Q3→Q4-gap, maar op test slechts een kwart. Het sterkste signaal is de
**interactie** van routekeuze en bitkeuze, niet rerouting alleen.

Bij het minimum-Q4-budget gebruikt de joint oracle een alternatieve route voor
87,89% van de validatie- en 82,81% van de testtokens. De gekozen routes in BF16
hebben gemiddelde KL `0,001586` en `0,001706`; routeverschuiving en
quantisatieresidual worden dus gezamenlijk geoptimaliseerd.

| Eindmetric bij Q4-kwaliteitsbudget | Validatie | Test |
|---|---:|---:|
| Joint teacher→candidate-KL | 0,002700 | 0,004255 |
| Natural-DP teacher→candidate-KL | 0,002697 | 0,004254 |
| Joint relatieve CE-delta | +0,439% | +0,134% |
| Natural-DP relatieve CE-delta | +0,355% | +0,208% |
| Joint top-1-overeenkomst | 98,05% | 98,44% |

De gepaarde blockbootstrap voor de joint upgradefractie is
`11,00–11,91%` op validatie en `12,50–17,12%` op test. Het testinterval kruist
15%; de puntgate is vooraf vastgelegd en slaagt, maar dit is nadrukkelijk nog
geen confirmatory bewijs.

## Controles en reproduceerbaarheid

- De natuurlijke BF16-route geeft op beide splits exact KL `0`, top-1 `1` en
  CE-delta `0`.
- Alle 51 tests slagen, inclusief brute-forcevergelijkingen voor route×bit-
  constructie en de globale budget-DP.
- De volledige run duurde 80,24 s; maximaal circa 1,40 GB CUDA-allocatie.
- Het 55,9 MiB JSON bevat alle 924 Stage-A-KL's, alle `32×64` Stage-C-KL's,
  routes, maskers, DP-curves, tokenmetrics, bronhashes, commandoregel,
  gitstatus en hardware.
- De opnieuw berekende Q3/Q4-output wijkt circa `0,0026` NRMSE af van de oude
  trace door expertbatch/reductienumeriek. Alle H1-kandidaten en baselines zijn
  binnen één nieuwe, gedeelde batch berekend; dit blijft als beperking en
  controlemeter gerapporteerd.

## Stop/go

**Go voor H1.** De preregistratie staat nu de volledige
`924×64=59.136`-kandidatenruimte toe. Deze top-32-uitkomst bewijst een sterk
oracleplafond, maar nog geen goedkope selector, packed runtime, downstream-
stabiliteit of algemene Eureka. De volledige enumeratie krijgt een apart,
vooraf vastgelegd resultaat en mag deze screen niet overschrijven.

