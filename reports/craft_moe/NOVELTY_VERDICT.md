# CRAFT-MoE noveltyverdict

## Uitkomst

**Geen verdedigbare brede nieuwheidsclaim en geen Eureka.** De audit vond veel directe en nabije prior art. De twee smalle, niet exact gevonden doorsneden zijn bovendien technisch gefalsificeerd; de volledige route–bit–atom–cache-stack is nooit dependency-vrij geïmplementeerd.

| Claim | Label | Technische status |
|---|---|---|
| CU1: Joint alternative-route and quantization-bit selection | `possibly novel intersection` | `falsified_downstream` |
| CU2: Blockwise route coalescing over explicit equivalence classes | `close/overlapping` | `hard_falsified` |
| CU3: Randomized residual syndrome for precision acquisition | `possibly novel intersection` | `falsified` |
| CU4: Joint route–bit–atom–cache optimizer | `not searched sufficiently` | `not_implemented_dependencies_falsified` |
| CU5: Custom kernel or layout for CRAFT-MoE | `clearly prior art` | `not_implemented` |

## Waarom dit sluitend genoeg is voor het projectbesluit

- CU1 is alleen lokaal positief en faalt de verplichte eerdere-laagtest hard.
- CU2 heeft met exacte optimale ILP's een te laag block-unionplafond.
- CU3 haalt één recoverymetric maar faalt veiligheid, attributie en hardwaremodel.
- CU4 heeft geen uitvoerbare kandidaat omdat de componentgates faalden.
- CU5 bestaat niet in deze repository; packed runtime bleef geblokkeerd.

De correcte bijdrageformulering is daarom: een streng gepinde, preregistered negatieve-resultatenstudie met exacte oracleplafonds en reproduceerbare falsificatie. Projected bytes of microbenchmarks zijn geen end-to-end speedup.

## Beperkingen

- The literature cutoff is 2026-08-10; future and unindexed work is absent.
- The search is targeted rather than a systematic-review database export.
- No citation graph, full-text similarity corpus, non-English database, or thesis repository was exhaustively searched.
- The patent pass is a limited keyword search and supports no legal conclusion.
- Several 2026 sources are preprints; their reported results were not independently reproduced here.
- A possibly novel intersection label is negative search evidence only and never a novelty determination.

Stop/go: **stop de huidige CRAFT-hypothesefamilie en download V4 Flash niet**. Een vervolg vereist een nieuwe, onafhankelijk gemotiveerde hypothese met nieuwe preregistratie; post-hoc varianten van de gesloten mechanismen mogen dit verdict niet vervangen.
