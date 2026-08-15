# Z1 — TreeSweep Oracle A: het plafond van de boomverifier

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Aanleiding: `NEMOTRON_TREESWEEP_200_AGENT_PACK`, Oracle A / fase P1 / poort P2C.

## 1. De poort van het pack, en waarom één meting hem beslist

TreeSweep-200 zet twee goedkope oracles vóór elke bouw en sluit 200 tok/s als er
één faalt. De sterke poort is:

```
max over N ≤ 64 van  A_oracle(N) / T_v(N)  ≥ 250 tok/s
```

`A_oracle(N)` is de best mogelijke opbrengst van een boom met `N` knopen, en die
is **hoogstens `N`**: een boom van `N` geverifieerde posities kan er nooit meer
dan `N` committen. Daarmee geldt, ongeacht welke drafter of topologie ooit
gebouwd wordt:

```
A_oracle(N) / T_v(N)  ≤  N / T_v(N)
```

X1 heeft `T_v` voor de dominante term al gemeten en vond hem **lineair in het
aantal posities**: het sequentiële MoE-pad schaalde 4,959× bij B=5, en de
expert-major sweep kwam op dezelfde tijd uit (verhouding 1,0017). Als
`T_moe(N) = c·N`, dan is `N / T_moe(N) = 1/c` — **een constante die niet van `N`
afhangt**. Geen enkele boomgrootte, topologie of dekkingsverbetering kan daar
overheen.

Deze fase meet `c` over een groter bereik dan X1 deed, zodat de lineariteit niet
uit vijf punten geëxtrapoleerd hoeft te worden.

## 2. Meetopzet

`N ∈ {1, 2, 4, 8, 16, 32}`. Alleen het **sequentiële** pad, want X1 heeft
gemeten dat expert-major er niet onder komt (1,0017 bij B=5) — de goedkoopste
bekende implementatie is dus de sequentiële, en die is de eerlijke `T_v`.

Alle 23 MoE-lagen, echte hidden states en echte officiële routes uit een echte
greedy generatie, elke expert van het blok resident, korte context. Gebracketeerde
basislijnen; een punt telt alleen als het buiten de lokale drift ligt.

Gerapporteerd: `T_moe(N)`, de lineaire fit `T = c·N + d`, de unie per laag, en
het plafond `1/c` in tokens per seconde.

## 3. Poorten

- **G-Z1-L1 — lineariteit.** De fit `T = c·N + d` moet `R² ≥ 0,99` halen én de
  gemeten `T(32)/T(1)` moet binnen 10% van 32 liggen. Haalt hij dat niet, dan is
  het plafond niet uit een enkele constante af te leiden en wordt er geen
  plafond gerapporteerd.
- **G-Z1-P2C — de poort van het pack, overgenomen zoals hij is.** `1/c ≥ 250`
  tok/s houdt het 200-tok/s-programma open. Daaronder is het gesloten voor dit
  model op deze GPU, **ongeacht de drafter**, want `1/c` is een bovengrens die
  perfecte dekking en nul draftkosten al veronderstelt.
- **G-Z1-S1 — sanity.** `T_moe(1)` moet binnen 10% liggen van X1's 22,454 ms,
  anders meet deze fase iets anders dan X1.

## 4. Wat dit niet doet

Geen boom gebouwd, geen Mamba-tree-scan, geen topologie-bewuste GQA, geen
drafter. Die zijn pas zinvol als de poort open is. De meting dekt alleen de
MoE-term; Mamba (sequentieel per knoop), attention, LM-kop en draftkosten komen
er in elke echte implementatie **bovenop**, dus het gerapporteerde plafond is een
**bovengrens** en geen schatting.

## 5. Artefacten

`scripts/lightningstream_nemotron/z1_tree_verifier_ceiling.py` ·
`z1_tree_verifier_ceiling.json` ·
`scripts/lightningstream_nemotron/z1_independent_verify.py` ·
`z1_independent_verification.json` · rapport met claim boundary.

## 6. Claim boundary van dit document

Geen meting, geen resultaat.
