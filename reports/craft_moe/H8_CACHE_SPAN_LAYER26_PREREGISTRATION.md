# H8 Cache-Span Reconstruction — preregistratie laag 26

Vastgelegd op `2026-08-10T12:05:29.4785656Z` vóór nieuwe H8-code, vóór capture
van alle expertoutputs en vóór inspectie van spanfit- of kwaliteitsresultaten.

## Hypothese

Wanneer een door Mass-Budget gekozen expert niet in de LRU-cache staat, kan de
gewogen bijdrage van één of meer van die misses worden gereconstrueerd uit een
kleine lineaire span van reeds berekende geselecteerde expertoutputs en ten
hoogste vier extra outputs van resident gebleven experts. Een oracle moet zo
minstens 50% van de Mass-Budget-loads kunnen vermijden bij lokale
teacher→candidate-KL hoogstens `0,001`, gemiddeld hoogstens twee extra resident
expertcomputaties per vermeden load en geprojecteerde computetijd kleiner dan
de vermeden transfer.

## Vaste model-, data- en cachebasis

- DeepSeek-V2-Lite revision
  `604d5664dddd88a0433dbae533b7fe9472482de0`, laag 26;
- WikiText-2-raw-v1 revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- eerste 256 validatie- en eerste 256 testtokens uit de bestaande
  layer-26-componenttrace, in twee onafhankelijke blokken van 128;
- ongenormaliseerde top-6-routergewichten en alle 64 echte BF16-expertoutputs
  op exact dezelfde MoE-input;
- sterke systeembaseline `mass_budget:j2:0.004`, expert-LRU-capaciteit 32,
  lege cache per 128-tokenblok en dezelfde route-touchvolgorde als de bestaande
  baseline;
- vaste laag-26-logitrang `5.425435543060303`, afkomstig uit de eerdere
  onafhankelijke validation-teachercalibratie; geen herkalibratie op H8-data.

De validation-split kiest exact één globale spanconfiguratie. Alleen daarna
wordt dezelfde configuratie op test geopend. Test kiest geen solver,
basisfamilie, coefficientbound, miss-subset of hyperparameter; de per-token
oracle mag wel de ware outputsignatuur gebruiken, want deze fase meet een
plafond en is niet deploybaar.

## Exacte control en kwaliteitsanker

Alle kandidaatstates worden aan de officiële teacher verankerd als
`teacher + BF16(candidate_routed - natural_routed_same_batch)`. De
`original`-control gebruikt aan beide kanten exact dezelfde natural routed
tensor en moet bitexact de teacher teruggeven, met KL/CE nul en top-1 één.
Router-ID's, routergewichten en Mass-Budget-missboekhouding worden tevens
onafhankelijk herberekend en opgeslagen.

Primaire kwaliteit is volledige-vocabulaire teacher→candidate-KL na de finale
norm en LM-head. Target-output-NRMSE/cosine en aggregate-routed-MSE zijn alleen
mechanistische diagnostiek.

## Optimistische ghost-cache-oracle

De primaire screen bevriest voor ieder token de cache **vóór** de load zoals
die in de exacte Mass-Budget-baseline bestond. Een vermeden load wordt voor
latere tokens toch gratis als resident behandeld. Dit kan in een echte cache
niet en maakt de screen bewust optimistisch: een negatieve uitkomst is sterker;
een positieve uitkomst vereist nog een causale cacheproef.

Voor elk token worden alle niet-lege subsets van de actuele misses doorlopen.
Voor een te reconstrueren subset is het target de exacte gewogen som van haar
expertoutputs. De overige geselecteerde experts blijven exact en worden niet
als extra compute geteld. Er zijn vaste basisfamilies:

1. `zero_fill`: geen span, alleen een drop-ablation;
2. `resident_selected`: alleen geselecteerde cachehits;
3. `available_selected`: alle geselecteerde experts die niet worden
   gereconstrueerd — dus bij één miss exact de andere vijf;
4. beide vorige families met een geneste OMP-volgorde van maximaal vier
   niet-geselecteerde cache-residenten.

OMP kiest telkens de cacheoutput met de grootste genormaliseerde absolute
correlatie met het actuele residual. Per vermeden load zijn nooit meer dan twee
extra residentoutputs toegestaan; de absolute cap blijft vier per token.

Vaste coefficientmethoden zijn:

- ridge met `alpha = 1e-4 * mean(diag(BᵀB))`;
- exacte NNLS (`c >= 0`);
- bounded least squares met `-1 <= c <= 1`.

Voor iedere globale configuratie en ieder mogelijk aantal vermeden loads kiest
de oracle per token de miss-subset/basislengte met laagste aggregate-target-MSE;
ties kiezen minder extra compute en daarna lexicografisch de expert-ID's. Pas
die ene kandidaat per avoid-count krijgt de dure exacte vocabulaire-KL. Het
grootste avoid-count met token-KL `<=0,001` en zonder niet-finite waarden wint.
Als geen reconstructie slaagt, blijft de exacte Mass-Budget-output staan.

De validation-selectie maximaliseert totale missreductie onder gemiddelde KL
`<=0,001` en extra-compute/load `<=2`; ties kiezen achtereenvolgens minder
extra compute, lagere KL, de basisvolgorde hierboven en `ridge`, `nnls`,
`bounded`. Deze ene configuratie wordt onveranderd op test gebruikt.

## Gates, ablaties en stop/go

De layer-26-oracle is alleen positief wanneer op **validatie én test**:

1. minstens 50% van de Mass-Budget-misses wordt vermeden;
2. gemiddelde volledige-vocabulaire KL `<=0,001`;
3. hoogstens twee extra resident expertforwards per vermeden load;
4. een batch-1-microbenchmarkmodel geeft totale extra resident compute kleiner
   dan de vermeden packed-int4-experttransfer;
5. de missreductie is minstens 10 procentpunt hoger dan `zero_fill`, zodat het
   resultaat daadwerkelijk door de span en niet alleen door lage routermassa
   wordt gedragen;
6. original-, route-, cache- en finite-controls slagen.

Harde falsificatie: de optimistische ghost-cache-primary haalt op test minder
dan 40% missreductie, test-KL is groter dan `0,001`, de span-uplift is niet
positief, of een exacte control faalt. Een uitkomst tussen 40% en 50% is
inconclusief negatief en opent geen predictor.

Alleen een positieve screen opent een nieuwe preregistratie voor een causale
cache (vermeden experts worden niet ingevoegd), minimaal 1.024
validationtokens, een kleine coefficient-/acceptatiepredictor zonder ware
targetoutputs, laag 23 plus exacte tail, spread en uiteindelijk full-depth CE
`<2%`. Zonder die fasen is H8 geen Eureka en geen runtimeclaim.

## Pre-full controlamendement na smoke-stop

De eerste 32-token-smoke stopte vóór outputcapture en vóór enige spanfit omdat
de expert-ID's en slotvolgorde exact waren, maar opnieuw berekende
routergewichten maximaal `4,9173832e-7` afweken van de oudere opgeslagen
componenttrace. Dit is dezelfde bekende GEMM-batchvormcategorie waarvoor de
QERC-preregistratie al een afzonderlijke regressiediagnostiek gebruikte. Vóór
herstart van de smoke en vóór de full data-inspectie wordt daarom vastgelegd:
expert-ID's/slotvolgorde en de onafhankelijke Mass-Budget-routes/loadcounts
blijven bitexact vereist; routergewichten krijgen uitsluitend voor vergelijking
met de oudere trace een absolute tolerantie `1e-6`, waarbij de werkelijke
maximumfout raw wordt gerapporteerd. De nieuwe run gebruikt haar intern
herberekende gewichten consequent voor baseline én kandidaten. Geen
inhoudelijke gate, spanmethode of testselectie is gewijzigd.
