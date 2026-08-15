# H3 Exact Atomic Expert Oracle — gelijktijdig full-depth

## Definitief H3-oordeel

**De oorspronkelijke H3-primary is gefalsificeerd.** Wanneer 25% van de
routed atomen gelijktijdig in alle 26 MoE-lagen wordt behouden en iedere
kandidaat haar eigen routepad volgt, overschrijdt WikiText-test de vaste
CE-grens: `+2,1129%` tegenover `<2%`. Lokale instructies blijven qua CE onder
de grens (`+1,4991%`) maar missen de vaste KL-gate met `0,03505 > 0,03`.

De laag-26-, laag-23- en spreadresultaten blijven geldige positieve
oracleplafonds; zij mochten de vooraf geregistreerde modelbrede gate echter
niet vervangen. Er volgt daarom geen atomic-indexpredictor of packed kernel uit
H3.

## Modelbrede curve

| Retentie in alle 26 lagen | Wiki-val KL / rel. CE / top-1 | Wiki-test KL / rel. CE / top-1 | Instructie KL / rel. CE / top-1 | Code KL / rel. CE / top-1 |
|---:|---|---|---|---|
| 100% | 0 / 0,000% / 100,00% | 0 / 0,000% / 100,00% | 0 / 0,000% / 100,00% | 0 / 0,000% / 100,00% |
| 75% | 0,00134 / +0,172% / 98,44% | 0,00129 / −0,216% / 99,22% | 0,00245 / −0,045% / 97,27% | 0,00099 / +0,167% / 99,22% |
| 50% | 0,00350 / +0,483% / 98,05% | 0,00317 / +0,407% / 97,66% | 0,00692 / +0,110% / 95,31% | 0,00252 / +0,135% / 97,27% |
| **35%** | 0,01197 / +1,162% / 96,09% | 0,01049 / +0,939% / 94,92% | 0,01791 / +0,424% / 93,36% | 0,00749 / +0,316% / 95,31% |
| **25%** | **0,02942 / +1,923% / 92,97%** | **0,02630 / +2,113% / 90,23%** | **0,03505 / +1,499% / 90,62%** | **0,01538 / +0,033% / 94,14%** |
| 15% | 0,10350 / +6,558% / 83,59% | 0,08497 / +4,911% / 85,94% | 0,08548 / +3,245% / 86,33% | 0,04121 / +0,905% / 92,58% |
| **10%** | **0,19460 / +10,759% / 80,47%** | **0,13684 / +6,964% / 80,86%** | **0,14994 / +5,057% / 80,47%** | **0,06294 / +1,289% / 89,84%** |
| 5% | 0,33362 / +17,490% / 78,12% | 0,27099 / +12,939% / 77,73% | 0,26523 / +8,457% / 75,78% | 0,13576 / +4,610% / 82,42% |

De 25%-CE-bootstrap is `+0,698%–+3,079%` op Wiki-validatie,
`+1,127%–+2,965%` op test, `+0,556%–+2,407%` op instructies en
`−0,563%–+0,611%` op code. Met twee blokken zijn deze intervallen grof, maar ze
tonen geen robuuste marge onder 2%.

Uniform 35% is het eerste vaste curvepunt dat op alle vier domeinen de
modelbrede kwaliteitsgrenzen haalt. Dat is een bruikbaar diagnostisch plafond,
geen geslaagde H3-claim: het is `2,86×` ideale atomreductie in plaats van de
vereiste `4×`, en de huidige tensor-lokale paginadruk blijft `56,67%`.

## Waarom losse lagen misleidden

De 25%-lokale routed fout is per laag vaak slechts ongeveer `0,09–0,18`, maar
route-overlap daalt gaandeweg en hidden-afwijkingen accumuleren. Op
Wiki-validatie is de gemiddelde top-6-overlap van de 25%-policy na laag 13
`93,55%`; na laag 25 `93,10%`. De finale fout is dus een trajecteffect dat
noch laag-26 noch drie afzonderlijke spread-lagen konden uitsluiten.

Elke fractionele policy volgde in deze proef haar eigen officiële hidden
states, routes, routergewichten en exacte supports. Er is geen teacher-route
afgedwongen na de eerste afwijking.

## Controles en raw bewijs

- de 100%-policy bleef na elk van de 26 MoE-lagen bitexact en finaal KL/CE `0`;
- alle captured en opnieuw berekende routes waren exact uitgelijnd;
- 208 packed supporttensors (`26×8`) zijn lossless opgeslagen;
- supportartifact: `224.942.720` bytes, SHA-256
  `cf09d9096be20efedac1eca429e6d452c372211e8ff5c6b69ed72dff17288258`;
- raw JSON: `80.563.484` bytes, SHA-256
  `bc681cea5c03d2407686fe438ef63b8cff67950fa70296c5f779ba786d3bb60c`;
- totale compute plus supportwrite `113,05 s`; piek toegewezen VRAM
  `2.815.363.584` bytes;
- ideale 25%-BF16 atom-bytes/MACs zijn analytisch 25%, maar huidige
  tensor-lokale pagina-accounting is ongeveer 50%; geen runtime is gemeten.

## Stop/go

**Stop H3:** geen predictor, atomic index, tilekernel, candidate-validation of
confirmatie voor de 25%-uniforme atompolicy. De tile-64-hardwaregate was al
gefaald en blijft negatief.

**Go onderzoeksprogramma:** ga conform de vaste P0-volgorde door naar H4
Residual Syndrome Sketch/SketchGate. Een toekomstige, afzonderlijk benoemde
laag-adaptieve rate-allocationoracle mag alleen met een nieuwe preregistratie
en mag dit gefaalde H3-resultaat nooit overschrijven.
