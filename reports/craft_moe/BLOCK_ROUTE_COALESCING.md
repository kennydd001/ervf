# H2 Speculative Route Coalescing Oracle

## Definitief oordeel

**H2 is op laag 26 hard gefalsificeerd.** Voor de vooraf vastgelegde primaire
8-tokenblokken, lokale KL-drempel `0,001` en slatecap 32 reduceert de exacte ILP
de natuurlijke expert-unie slechts `19,65%` op validatie en `20,24%` op test.
De gate was minstens 40%; beide uitkomsten liggen ook onder de vooraf bepaalde
25%-harde grens. Additioneel tegenover per-token Mass-Budget is de reductie
`17,75%/18,24%`, niet de vereiste 25%.

De lokale KL is veilig en alle exacte solvercontrols sluiten. Het probleem is
dus geen kwaliteitsoverschrijding of zwakke heuristic: het combinatorische
expert-unieplafond zelf is te laag. Er volgt geen laag-23-interventie,
bitplane-/atomtile-uitbreiding of speculative runtime uit H2.

## Primaire exacte ILP

| Metric | Validatie | Test | Gate |
|---|---:|---:|---:|
| natural totale union | 865 | 899 | — |
| exacte ILP-union | 695 | 717 | — |
| reductie versus natural | **19,65%** | **20,24%** | ≥40% |
| 95%-blockbootstrap | 17,37–21,98% | 18,34–22,06% | — |
| Mass-Budget-union | 845 | 877 | — |
| extra reductie versus Mass-Budget | **17,75%** | **18,24%** | ≥25% |
| 95%-blockbootstrap | 15,55–20,05% | 16,37–20,04% | — |
| gemiddelde lokale KL | 0,000183 | 0,000200 | ≤0,001 |

Mass-Budget `δ=0,004` bespaart zelf slechts `2,31%/2,45%` union versus natural.
Marginal-union greedy haalt `10,87%/12,01%`; eligible-set pruning
`3,35%/2,22%`. Beam-1024 vindt in alle 64 primaire blocks exact dezelfde
unioncount als de ILP (`695/717` totaal), zodat de negatieve oracle niet door
een slechte benadering wordt veroorzaakt.

De exacte schedule wijzigt circa `67,6%/68,0%` van de routes en blijft toch
maar rond 20% unionreductie. De routeslates hebben gemiddeld 21,52 validatie-
en 20,19 testkandidaten; gebrek aan één alternatief per token verklaart het
plafond dus niet volledig. De alternatieven delen eenvoudigweg onvoldoende één
kleine globale expertset.

## Sweep en fixed-cache-diagnostiek

Bij de ruimste diagnostische cel (drempel `0,003`, cap 64, 16-tokenblok) haalt
beam `29,04%` validatie en `30,41%` test bij gemiddelde lokale KL
`0,000465/0,000500`. Meer toekomstcontext, ruimere slates en drie keer de
primaire per-route-KL-grens brengen de union dus nog steeds niet bij 40%. Dit
is een beamdiagnostiek, geen vervanging van de gepreregistreerde exacte primary.

Met de vooraf gekalibreerde 32 hot experts als gratis cache reduceert de exacte
cache-aware ILP de **cold** union `35,64%/37,01%`. Dat is nuttige
cache-accounting, maar verandert de lege-cache-primary niet en blijft ook onder
40%.

## Exacte controls en audit

- de natuurlijke subset is exact één keer aanwezig en heeft maximale numerieke
  reconstructie-KL `9,67×10⁻⁶`;
- 1.280 ILP's eindigen allemaal met HiGHS-status `Optimal`;
- beam versus exact sluit op iedere primaire block;
- de gekozen unioncount is gelijk aan de ILP-objective binnen
  `1,35×10⁻¹²` absoluut;
- maximale gerapporteerde relatieve MIP-gap is slechts `4,10×10⁻¹⁶`.

De eerste JSON-adjudicator eiste per ongeluk floating `mip_gap == 0.0` en zette
daardoor vier machine-epsilon-uitkomsten als control-fout. Het hoofdartifact
blijft ongewijzigd. Een append-only audit v2 past de vooraf bedoelde criteria
toe—status optimal en objective/union-sluiting—en laat alle 1.280 records
slagen. Een eerste auditbestand met een verkeerde empty-cache/cold-unionkolom
blijft eveneens bewaard; audit v2 corrigeert alleen die boekhouding. Het harde
inhoudelijke verdict verandert in geen van de drie adjudicaties.

## Accounting, beperkingen en stop/go

Een routed expert bevat `8.650.752` gewichten: `17.301.504` BF16-bytes of
`4.325.376` packed-int4-bytes. De unionreducties vertalen daarom exact naar
dezelfde geprojecteerde expertbytefracties, maar dit is geen gemeten transfer,
cachekernel, latency of speculative accepted-tokenmeting.

- hoofd-JSON: `75.858.594` bytes, SHA-256
  `63e80464823a7c696230e9e4d87c4d889a583901296fca50d005e70e9ba9a09d`;
- beslissende control-audit v2: `32.851` bytes, SHA-256
  `79bddd85cafe9fdb420ac716f8934e05ccdeb156d1cf727cd4d24e9112543942`;
- behouden foutieve audit v1: `441.847` bytes, SHA-256
  `11be8316cdb583ce73ef327325734e2886d0667f41d0c54c85ed5a2167899083`;
- totale sweeptijd `380,84 s`; alle slates, gekozen routes, unionmasks,
  blockmetrics, bootstrapseries en solverdiagnostiek staan in de JSON.

De proef veronderstelt bekende toekomstige routes en gebruikt exacte teacher-KL
voor eligibility; zelfs het oracle is dus niet deploybaar. Er is geen
noveltyclaim vanwege nabije EcoSpec/AcceptMoE/EdgeXpert-prior art.

**Stop H2:** geen laag 23, Q3/Q4-/atomtile-uitbreiding of echte speculative
runtime. **Go onderzoeksprogramma:** de P0-rij is nu uitgeput; vervolg met de
onafhankelijke P1-hypothese H6 Quantization-Error Route Cancellation (QERC).
