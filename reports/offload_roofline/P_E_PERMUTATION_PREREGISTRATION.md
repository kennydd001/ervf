# P-E preregistratie — spectraal gepermuteerde tile-64

Deze test gebruikt bestaande, eerder geopende DeepSeek-V2-Lite-laag-26-traces
en is daarom exploratief. De nieuwe permutatie en de afgescheiden
calibratie/evaluatiesplit zijn vóór uitvoering vastgelegd.

## Input en scheiding

- Laag 26, 64 routed experts, 1.408 SwiGLU-neuronen per expert, natuurlijke
  top-6 routes en routergewichten.
- Calibratie: uitsluitend validation trace-indices 256–1023 (768 tokens).
- Evaluatie: validation 0–255 en test 0–255. Geen evaluatiemasker of
  evaluatiemetric mag de permutatie beïnvloeden.
- De exacte bijdragescore blijft
  `|p_e · a_j| · ||down_column_j||₂`, gelijk aan de oorspronkelijke H3-screen.

## Permutatie-algoritme

1. Bouw op calibratie het globale 25%-neuronoraclemasker.
2. Groepeer maskerrijen per werkelijke expert-ID.
3. Partitioneer de 1.408 neuronkolommen per expert deterministisch in 22
   groepen van exact 64 via recursieve gebalanceerde spectrale splitsing:
   centreer de binaire co-selectiefeatures, projecteer op de eerste rechter
   hoofdcomponent, stable-sort op projectie met origineel neuron-ID als tie en
   splits exact volgens het aantal resterende 64-groepen.
4. Bij nul variantie gebruikt die node de oorspronkelijke neuronvolgorde.
5. De concatenatie van de 22 groepen is de fysieke neuronpermutatie voor die
   expert. Alle 64 permutaties moeten bijectief zijn.

## Kandidaten en gates

Op beide evaluatiesplits worden exact drie 25%-kandidaten met gelijk
atom-/tilebudget gemeten: globale neuronoracle, oorspronkelijke tile-64 en
gepermuteerde tile-64. Selectie blijft per token een globale top-ceil(25% ×
132 tiles) = 33 volledige expertlokale tiles.

- Primaire gate: gepermuteerde tile-64 gemiddelde teacher→candidate-KL is op
  validation én test ≤1,20× de globale neuronoracle-KL.
- De oude tile-64 wordt exact gereproduceerd binnen absolute metric-tolerantie
  `5e-6`; anders is de nieuwe run ongeldig.
- De volledige 100%-permutatiereconstructie moet volgens de aangeleverde claim
  bit-identiek zijn. Daarnaast rapporteren we NRMSE en maximum absolute fout,
  zodat een eventuele floating-point-reductieordeafwijking zichtbaar blijft.
- CE, top-1, routed-L2, retained counts, hashes en piekgeheugen worden altijd
  gerapporteerd. Geen runtimeclaim: evaluatie gebruikt dense zero-masked GEMM.

De officiële P-E slaagt alleen als zowel de KL-gate, oude-baselinereproductie,
bijectiviteit als de geëiste bit-exacte reconstructiecontrole slagen.
