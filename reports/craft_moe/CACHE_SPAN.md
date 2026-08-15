# H8 Cache-Span Reconstruction

## Definitief oordeel

**H8 eindigt als een vooraf geregistreerde, inconclusief negatieve
layer-26-screen.** De optimistische ghost-cache-oracle vermijdt `41,35%` van de
Mass-Budget-loads op validatie en `48,54%` op de eenmaal geopende testset. De
gate vereiste minstens 50% op beide splits. De kwaliteit, compute-count en het
microbenchmarkmodel slagen, maar de kernclaim en de mechanistische
span-ablation niet. Daarom is geen coefficientpredictor getraind en zijn een
causale cache, laag 23, spread en full-depth niet geopend.

Dit is geen harde `<40%`-falsificatie op test: het testpunt ligt er boven en de
twee-blockinterval kruist 50%. Het resultaat is desondanks onvoldoende om H8
door te laten. Validatie ligt volledig rond 41%, en vrijwel de hele
testreductie blijkt zonder reconstructiespan haalbaar door de gemiste bijdrage
op oraclebasis simpelweg op nul te zetten.

## Primaire resultaten en zero-fill-ablation

Validatie koos zonder testinzage één globale configuratie:
`resident_selected_no_cached_bounded`. Zij reconstrueert de gewogen som van de
gekozen misses uit geselecteerde cachehits, met coefficients begrensd tot
`[-1,1]`. Er zijn geen extra cache-expertforwards gebruikt.

| Metric | Validatie | Test | Gate |
|---|---:|---:|---:|
| Mass-Budget-loads | 416 | 445 | — |
| vermeden loads, primary | 172 | 216 | — |
| primary missreductie | **41,346%** | **48,539%** | ≥50% op beide |
| 95%-blockbootstrap | 40,807–41,969% | 45,078–51,190% | — |
| vermeden loads, zero-fill | 166 | 215 | — |
| zero-fill missreductie | 39,904% | 48,315% | ablation |
| echte span-uplift | **+1,442 pp** | **+0,225 pp** | ≥+10 pp |
| 95%-upliftinterval | +0,897–+2,073 pp | 0–+0,397 pp | — |
| gemiddelde teacher→candidate-KL | 0,000251 | 0,000278 | ≤0,001 |
| relatieve CE-delta | −0,0848% | −0,0441% | diagnostisch |
| extra resident forwards / vermeden load | 0 | 0 | ≤2 |

De span voegt dus slechts zes vermeden loads toe op validatie en één op test.
Dat verschil is veel kleiner dan de vooraf vastgelegde tien procentpunt die
nodig was om het mechanisme aan de lineaire span toe te schrijven. De sterke
zero-fill-uitkomst is een oraclekeuze van welke lage-impact misses mogen worden
gedropt; ze is geen deploybare selector en mag niet als nieuwe methodeclaim
worden opgevoerd.

De exacte Mass-Budget-baseline zelf heeft gemiddelde lokale KL
`0,0001247/0,0002077` en top-1 `100%/99,61%` op validatie/test. De primary blijft
onder de KL-grens, maar verdubbelt ongeveer de lokale KL. Negatieve CE-punten
op deze kleine vensters vormen geen kwaliteitsverbeteringsclaim.

## Waarom dit een optimistisch plafond is

Voor elke miss-subset kende de oracle de echte ontbrekende outputvector. Hij
mocht de beste subset en basislengte per token kiezen na exacte
volledige-vocabulaire-KL, en de cachetoestand bleef die van de baseline alsof
een vermeden expert later toch gratis resident was. Per avoid-count werd vóór
KL de kandidaat met de laagste aggregate-target-MSE gekozen. Naast zero-fill
zijn twaalf vooraf vaste combinaties van resident/available selected outputs,
OMP met maximaal vier cache-experts, ridge, NNLS en bounded least squares
doorgerekend.

Een inzetbaar systeem heeft geen ware targetoutput, mag een niet-geladen expert
niet als ghost-resident behandelen en moet coefficients plus acceptatie uit
router-/cachefeatures voorspellen. Omdat zelfs het optimistische plafond de
gate mist en nauwelijks boven zero-fill uitkomt, zou predictortraining de
bewijsstand alleen verzwakken. Het meegeleverde trainingsscript rapporteert
daarom gecontroleerd `not_opened` en schrijft geen model.

## Exacte controls en accounting

- officiële router-ID's en slotvolgorde zijn bitexact; op de full run is ook de
  maximale routergewichtfout exact `0,0`;
- sorted top-6-expertsets en dense routergewichten zijn exact gelijk;
- Mass-Budget-routes en loadcounts sluiten via een onafhankelijk codepad:
  `416/416` validatie en `445/445` test;
- de LRU-missmaskers volgen de echte sequentiële intra-token-touchvolgorde,
  inclusief een regressietest voor een vroege miss die een latere hit verdringt;
- de officiële teacher-delta-original-control is vóór en na capture bitexact;
- alle 64×512 expertoutputs en alle kandidaatstates zijn finite.

De eerste smoke stopte vóór artifactschrijven op een bekende
batchvormafwijking van `4,917e-7` in routergewichten. Een expliciet pre-full
amendement legde tolerantie `1e-6` vast zonder inhoudelijke gatewijziging. De
herhaalde smoke sloot; de full run had uiteindelijk fout `0,0`.

De hardwarediagnostiek vergelijkt één packed-int4-expertload van `4.325.376`
bytes met een resident BF16-expertforward en een tienvector-spancombinatie.
Omdat de gekozen primary nul extra forwards gebruikt, kost het geprojecteerde
spanwerk `0,02257 ms` tegenover `0,17302 ms` transfer, ratio `13,04%`; die gate
slaagt. Dit is een batch-1-microbenchmarkmodel, geen packed kernel, causale
cache of wall-clock-speedup.

## Bootstrap, artefacten en reproduceerbaarheid

De gepaarde 10.000× blockbootstrap gebruikt de twee vooraf vaste 128-tokenblokken
per split. Met slechts twee sampling units zijn de intervallen beschrijvend;
ze mogen de gefaalde puntschattinggates niet vervangen. Alle blocktotalen
reconciliëren exact met de hoofd-JSON.

- hoofdresultaat: `cache_span.json`, 9.384.788 bytes, SHA-256
  `60b39e4aa5717221c561ad5cbb286b412c04c40d2ab34b5f6435e43a1d5b63bc`;
- lossless capture: `cache_span_layer26_capture.safetensors`, 140.587.256
  bytes, SHA-256
  `260c241b4513a45e9c7165df996847a2b7ef4481f3d52c2420c3ba76eb771a93`;
- append-only blockbootstrapaudit: 1.951.907 bytes, SHA-256
  `b051af8a2d383d7f56423f440bb30e3a867ebb5f01294d643a7b7709f117229a`;
- behouden geslaagde smoke-JSON/capture: 1.076.347/8.787.840 bytes, hashes
  `bbd5f1ae72f07856cd5787b0e78df5264fa105c66620aaa658a30ebf5b480e17`
  en `56bbce748dfc1eb225bb329ecaa35172a3d3c0ba6817df673fe671ac2edb0a10`.

De full run duurde `46,53 s`. JSON bevat alle oraclekandidaten, coefficients,
per-tokenbesluiten, volledige KL/CE/top-1-series, twee-blockbootstrapseries,
commands, libraryversies, hardwarestaat, trace-indices, inputhashes en de dirty
repository zonder commit.

**Stop H8:** geen predictor, causale cache, laag-23-interventie, spread of
full-depth. **Go onderzoeksprogramma:** vervolg met de nog onafhankelijke H10
Reduction-Order-hypothese; H5/H9 en packed atomic runtime blijven door de
full-depth H3-falsificatie geblokkeerd.
