# Prior art en gevolgen voor onze hypothese

Laatst bijgewerkt: 2026-08-10. Dit document onderscheidt gepubliceerde claims
van eigen metingen. Alleen waar een lokaal JSON-rapport expliciet wordt
genoemd, is een resultaat door ons gereproduceerd; overige paperclaims zijn
literatuurbevindingen.

## Zeer nabije methoden

### MergeMoE — expert-output merging

Paper: <https://arxiv.org/abs/2510.14436>

MergeMoE formuleert compressie expliciet in termen van het benaderen van
gewogen expertoutputs en gebruikt echte input samples plus least squares om
compressiematrices te bepalen. Dit overlapt rechtstreeks met het idee om niet
de losse expertgewichten maar de uiteindelijke expertfunctie te benaderen.

Gevolg: “expertoutputs in plaats van weights comprimeren” is geen nieuwe claim.
Onze aggregate surrogate moet minimaal tegen MergeMoE worden vergeleken.

### Sub-MoE — activation-weighted shared subspaces

Paper: <https://arxiv.org/abs/2506.23266>

Sub-MoE clustert experts op outputgelijkenis en zoekt gedeelde subruimten met
activation weighting. Dit ligt zeer dicht bij een activation-manifold/shared-
basis-hypothese.

Gevolg: activation-aware shared bases zijn eveneens prior art. Eventuele nieuwe
waarde moet uit de hogere compressiefactor, progressieve hardwaregestuurde
residuals, tail-latency of een aantoonbaar beter Paretofront komen.

### MoBE — basisexperts

Paper: <https://arxiv.org/abs/2508.05257>
Code: <https://github.com/inclusionAI/MoBE>

MoBE factoriseert expertmatrices in een expert-specifiek deel en een lineaire
combinatie van gedeelde basismatrices. Het paper bevat expliciet resultaten op
DeepSeek-V2-Lite-Chat en rapporteert voor grotere modellen grofweg 24–30%
parameterreductie met ongeveer 1–2% accuratesseverlies.

Gevolg: MoBE is een verplichte weight-space-baseline. Het resultaat is ook een
waarschuwing: kwaliteit behouden bij 4×–8× reductie is veel agressiever dan het
gebied waarin MoBE zijn sterkste claims doet.

### PuzzleMoE — fijnmazig mergen plus quantisatie

Paper: <https://arxiv.org/abs/2511.04805>

PuzzleMoE rapporteert 50% expertcompressie met behouden kwaliteit en, samen met
quantisatie, circa 4,8× totale compressie met ongeveer 1–1,7% verlies op de
geteste MoE-modellen. De snelheidswinst is gekoppeld aan speciale bit-packed
kernels, niet alleen aan een kleiner checkpoint.

Gevolg: 4× is geen passende “uitvinding”-gate meer; het is een relevante
reproductiebaseline. Onze 8× gate blijft zinvol als onderscheidend technisch
doel, mits dezelfde kwaliteit en werkelijk gemeten runtimebytes worden gehaald.

### QMoE — sub-1-bit quantisatie

Paper/code: <https://arxiv.org/abs/2310.16795>

QMoE rapporteert 0,8 bit/parameter en ongeveer 20× compressie voor een
1,6T SwitchTransformer met geringe accuratesseafname en eigen GPU-kernels. De
demonstratie gebruikt nog steeds vier A6000- of acht 3090-GPU's en is dus geen
bewijs voor onze laptopdoelstelling of voor modernere DeepSeek-architecturen.

Gevolg: een zeer hoge opslagcompressie is niet principieel uitgesloten. We
moeten quantisatie als serieuze baseline behandelen en mogen een eventuele
function-spacewinst niet vergelijken met alleen BF16.

## Negatieve prior

### Geometric Asymmetry in MoE Specialization

Paper: <https://arxiv.org/abs/2605.16349>

Deze analyse rapporteert lage cross-expert Jacobian-alignment ondanks deels
overlappende representatiesubruimten. Dat is precies het patroon waarbij een
universele kleine surrogate onvoldoende kan zijn, zelfs als PCA op activaties
een lage dimensie lijkt te tonen.

Gevolg: we meten niet alleen activation PCA. We meten ook expert-output- en
Jacobian/subspace-overlap, plus held-out en autoregressieve fouten. Anders kunnen
we een lage representationele rank verwarren met gedeelde expertfuncties.

## Bijgestelde onderzoeksvraag

De oorspronkelijke brede claim was te nieuw geformuleerd. De verdedigbare vraag
wordt:

> Kan een hardware-aware combinatie van bestaande output-merging, gedeelde
> bases, quantisatie en conditionele residuals op DeepSeek-V2-Lite een gemeten
> 8× expert-byte-reductie behalen zonder meer dan de vooraf vastgelegde
> kwaliteitsdaling en zonder onacceptabele routerdrift of tail latency?

Mogelijke uitkomsten:

- Minder dan 4×: bestaande methoden zijn waarschijnlijk al sterker; onze route
  heeft geen praktisch vervolg richting V4 Flash.
- 4×–8×: technisch nuttig, maar waarschijnlijk een combinatie/engineering-
  bijdrage en geen fundamenteel nieuwe compressor.
- Minimaal 8× met stabiele rollouts: vervolg naar V4 Flash is gerechtvaardigd;
  originaliteit vereist dan nog een formele novelty-audit.
- Hoge lokale reconstructiescore maar instabiele rollouts: de hypothese faalt
  door distributie- en routerdrift, ook als de losse laagtest “goed” lijkt.

## Verplichte experimentele baselines

1. BF16 exacte top-6.
2. Top-1, zowel met oorspronkelijke als opnieuw genormaliseerde routerweight.
3. Eenvoudige quantisatie bij gelijk bytebudget.
4. Shared mean / expert merging.
5. MoBE-achtige gedeelde bases.
6. MergeMoE/Sub-MoE-achtige activation-aware output/subspacevariant.
7. Onze route-conditioned aggregate/progressive-residualvariant.

## 2026-08-10 — Novelty-audit van route-equivalentie en cachekeuze

Deze aanvulling volgt op de lokale route-equivalentie-experimenten. Anders dan
de eerdere secties zijn de hieronder genoemde papers rechtstreeks vergeleken
met de uiteindelijk uitgevoerde policy.

### Cache-Conditional Experts — onze praktische policy is een grensgeval

Paper: <https://arxiv.org/abs/2412.00099>  
TMLR-versie: <https://openreview.net/forum?id=ul4W26KEKz>

Skliar et al. introduceren trainingvrije cache-aware routing en evalueren die
expliciet op DeepSeek-V2-Lite. Hun `Max Rank`-baseline bewaart de eerste `J`
routerkeuzes en mag gecachete experts tot rang `M` promoveren. Voor DeepSeek
top-6 is onze regel — top-5 behouden en rang 6 alleen door gecachete rang 7
vervangen wanneer dat een LRU-miss voorkomt — praktisch het geval `J=5, M=7`,
met als kleine implementatienuance dat wij de exacte within-token LRU-volgorde
simuleren. Hun sterkere Cache-Prior manipuleert uitsluitend de rangschikking;
de oorspronkelijke routerweights blijven voor aggregatie behouden.

Gevolg: cache-aware bottom-rankvervanging is geen nieuwe uitvinding. Onze
metingen zijn een onafhankelijke, streng gepinde reproductie/variant.

### MoE-ERAS — residency-aware selectie bestond al in 2024

Paper: <https://openreview.net/forum?id=o43eHjPEMO>

MoE-ERAS gebruikt thresholding en biasing om bij expertselectie zowel kwaliteit
als residentie mee te nemen. Het rapporteert minder swaps en tot 21,2% lagere
latency bovenop caching en quantisatie.

Gevolg: ook de brede claim “de runtime mag een resident expert verkiezen” is
bezet.

### BuddyMoE — offline vervangingsparen en runtime substitution

Paper: <https://arxiv.org/abs/2511.10054>

BuddyMoE profileert co-activatie en outputgelijkenis, bouwt offline buddy-lijsten
en vervangt een gemiste expert door een resident buddy onder token- en
residency-gates. Het rapporteert tot 10% throughputwinst met verwaarloosbaar
kwaliteitsverlies.

Gevolg: een offline tabel met “veilige” expertparen plus een cachebeslissing is
evenmin voldoende nieuwheid voor een volgende variant.

### SERE — functionele rerouting van secundaire experts

Paper: <https://arxiv.org/abs/2602.07616>

SERE berekent per laag een activatiegebaseerde expert-similaritymatrix, bewaart
primaire en kritieke experts en reroutet secundaire experts naar vergelijkbare
primaire experts. Het paper rapporteert op DeepSeekV2 1,2–1,6× versnelling in
batch decoding; de DeepSeek-V2-Lite-prefill behoudt in de gerapporteerde
batchconfiguraties alle geactiveerde experts als primair.

Gevolg: functionele expertvervanging en activatiesimilariteit zijn bezet. Een
eventuele eigen bijdrage moet aantoonbaar iets leveren wat deze kalibratiematrix
niet levert.

### Counterfactual Routing Analysis — alternatieve routes zijn al direct getest

Paper: <https://arxiv.org/abs/2605.07260>

Yoon et al. vergelijken per token de standaardroute met 32 sampled
equal-compute routes uit de router-top-32 en draaien daarna het resterende model
door. Zij gebruiken de kans op het gerealiseerde volgende token als utility en
rapporteren hetzelfde kwalitatieve patroon op DeepSeek-V2-Lite: vooral op
fragiele tokens bestaan betere alternatieven.

Onze meting verschilt nog wel: alle 924 top-6-subsets uit de top-12 worden
uitputtend geëvalueerd, met teacher-to-route-KL over de volledige
vocabulaireverdeling en een telling/entropie van alle routes onder een vaste
KL-grens. Dat is een scherpere diagnostiek, maar op zichzelf geen bewezen nieuw
inferencesysteem en geen veilige brede novelty-claim.

### SliceMoE — dynamische bit-slices plus cache zijn eveneens bezet

Paper: <https://arxiv.org/abs/2512.12990>

SliceMoE combineert Dynamic Bit-Sliced Caching, Matryoshka-quantisatie en
cachewarmup. Het gebruikt een unified cache over lagen en rapporteert op
DeepSeek-V2-Lite tot 1,81× decodeversnelling en 2,37× energiereductie bij
near-high-bit accuracy.

Gevolg: de brede combinatie van dynamische precisie, caching en DeepSeek-V2-Lite
is prior art. Onze 2→4- en 3→4-bit-oracles zijn diagnostische resultaten, geen
nieuwe systeemclaim.

### CARE — tokenadaptieve cumulatieve probability mass is ook prior art

Paper: <https://arxiv.org/abs/2607.26052>

CARE (`Confidence-Adaptive Routing of Experts`) activeert in MoE-LoRA experts
in dalende routervolgorde tot een cumulatieve-massadrempel is bereikt en
kalibreert het gemiddelde computebudget. Het verandert daarmee het aantal
actieve experts en behandelt geen expert-weightcache of LRU-missminimalisatie.

Gevolg: CARE is geen directe implementatie van onze vaste-top-6-
Mass-Budgetpolicy, maar blokkeert wel iedere brede claim dat tokenconfidence of
cumulatieve routermass voor adaptieve expertselectie nieuw zou zijn.

### Mass-Budget Cache-Prior — incrementele kandidaat, geen brede nieuwheidsclaim

Eigen implementatie en metingen:
`reports/MASS_BUDGET_EUREKA_2026-08-10.md`.

De nieuwe policy bewaart top-2, genereert een vaste slate van reeds bekende
Cache-Prior-routes en kiest de route met de minste actuele LRU-misses onder een
expliciete bovengrens op het verlies aan originele geselecteerde top-6-
probability mass. De originele routerkansen blijven de expertoutputs wegen.

De dichtstbijzijnde methode blijft Cache-Conditional Experts. Vooral hun
`Cumsum`-baseline is belangrijk: die kiest voor ieder token al een dynamische
maximumrang op basis van cumulatieve routerprobabiliteit. Hun Cache-Prior
genereert de routefamilie die wij als kandidaatenslate gebruiken. Appendix E
meldt bovendien dat een geleerd cache-MLP de vaste prior niet wist te
overtreffen.

Een gerichte zoekactie op combinaties van “cache-aware MoE routing”,
“probability mass”, “per-token constraint” en “expert cache” vond geen primaire
bron met exact onze constraint plus miss-minimalisatie. Dat is slechts
negatief zoekbewijs. Confidence-adaptive cumulative-mass-routing bestaat ook
buiten expertcaching, en MoE-ERAS bezet de brede residency-aware claim.

Gevolg: verdedigbaar is hoogstens “een trainingvrije incrementele
selectieregel die in onze DeepSeek-V2-Lite-evaluatie het vaste-λ-front
verbetert”. Niet verdedigbaar zijn claims dat probability-mass routing,
cache-aware routing of route-optimalisatie zelf nieuw zijn.

## Novelty-oordeel

- **Niet nieuw:** cache-aware expertpromotie, bottom-rankvervanging,
  residency-aware routing, buddy-experts, functionele rerouting en dynamische
  bit-sliced caching.
- **Mogelijk incrementieel nieuw als meetprotocol:** uitputtende top-12-choose-6
  route-equivalence entropy onder volledige-distributie-KL, zonder
  hernormalisatie van DeepSeek-routerweights.
- **Nog niet bewezen nieuw of praktisch:** een runtime die die KL-certificaten
  zonder teacher en zonder alle alternatieven uit te voeren kan voorspellen.
- **Mogelijk incrementieel nieuw en empirisch nuttig:** een vaste Cache-Prior-
  slate minimaliseren op misses onder een geselecteerde-mass-lossbudget; geen
  fundamentele of patentclaim zonder bredere search.

Dit is een gerichte primaire-literatuurcontrole, geen patentonderzoek en geen
garantie dat het meetprotocol nergens eerder voorkomt. Er mag daarom geen
fundamentele nieuwheidsclaim worden gemaakt.
