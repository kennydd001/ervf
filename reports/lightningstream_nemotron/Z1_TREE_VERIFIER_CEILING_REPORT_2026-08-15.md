# Z1 — TreeSweep Oracle A: het plafond ligt op 45–61 tok/s, de poort vroeg 250

Datum: 2026-08-15
Verdict: **De kosten van de MoE-term zijn lineair in het aantal geverifieerde posities — R² = 0,99986, T(16)/T(1) = 15,845 tegen 16. Daarmee is `N/T_v(N)` een constante die geen enkele boomgrootte of topologie kan overtreffen: 60,59 tok/s uit deze meting, 44,90 uit X1's onafhankelijke meting van dezelfde grootheid. TreeSweep's poort P2C vroeg ≥ 250. Gefaald met een factor 4,1 tot 5,6.**
Terminal state: `z1_tree_ceiling_60_tok_s_p2c_failed_by_4x`
Preregistratie: `Z1_TREE_VERIFIER_CEILING_PREREGISTRATION_2026-08-15.md`

## 1. Waarom één meting deze poort beslist

TreeSweep-200 sluit het 200-tok/s-programma als de gezamenlijke oracle faalt:

```
max over N ≤ 64 van  A_oracle(N) / T_v(N)  ≥ 250 tok/s
```

`A_oracle(N)` is de best denkbare opbrengst van een boom met `N` knopen, en die
is **hoogstens `N`** — meer tokens committen dan je posities verifieert kan niet.
Dus voor élke drafter, élke topologie en élke dekkingsverbetering geldt:

```
A_oracle(N) / T_v(N)  ≤  N / T_v(N)
```

Als `T_v(N) = c·N`, dan is dat `1/c`: **een constante die niet van `N` afhangt.**
Grotere bomen helpen dan per definitie niet.

## 2. De meting

Alle 23 MoE-lagen, echte hidden states, echte officiële routes, elke expert
resident, korte context, gebracketeerde basislijnen.

| N | totaal | per positie | unie/laag |
|---:|---:|---:|---:|
| 1 | 16,630 ms | 16,630 | 6,00 |
| 2 | 32,519 | 16,260 | 10,13 |
| 4 | 63,021 | 15,755 | 17,13 |
| 8 | 130,812 | 16,352 | 27,26 |
| 16 | 263,493 | 16,468 | 42,65 |

Lineaire fit: **16,505 ms per positie**, intercept −1,038 ms, **R² = 0,99986**.
`T(16)/T(1) = 15,845` tegen 16 — binnen 1%.

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-Z1-L1** | R² ≥ 0,99 én schaling binnen 10% | 0,99986 / 15,845 | ✅ |
| **G-Z1-P2C** | `1/c` ≥ 250 tok/s | **60,59** | ❌ |
| **G-Z1-S1** | `T(1)` binnen 10% van X1's 22,454 ms | 16,630 (−26%) | ❌ |

Verifier 27/27, `VERIFIED`.

**De sanity-poort faalt en dat wordt niet weggepoetst.** X1 mat dezelfde
grootheid op 22,454 ms per positie, Z1 op 16,630. Het verschil zit in de opzet:
X1 middelde over acht blokken verspreid over de generatie, waarbij de LRU tussen
de blokken doorstroomde; Z1 gebruikt één blok met een warme cache. Beide zijn
echte metingen van hetzelfde pad onder een iets andere cachedruk, en het verschil
is precies wat je van cachedruk verwacht.

Voor de conclusie maakt het niets uit. De twee onafhankelijke hellingen geven
plafonds van **44,90** en **60,59** tok/s. De poort vroeg 250. Het gat is een
factor 4,1 tot 5,6, en geen keuze tussen de twee metingen verandert dat.

## 3. Wat dit plafond wél en niet is

Het is een **bovengrens** die al veronderstelt:

- perfecte dekking — élke van de `N` geverifieerde posities wordt gecommit;
- nul draftkosten;
- gratis Mamba, attention, LM-kop, router, shared expert en state-commit.

Een echte implementatie betaalt die er allemaal bij. Mamba is per knoop
sequentieel (8,3 ms per token, S8), attention en LM-kop komen erbovenop, en de
gemeten acceptatie is 2,114 en niet `N`. Het werkelijke resultaat ligt dus ruim
onder 45–61 tok/s, niet erop.

Daarmee is TreeSweep's uitkomst **A** bereikt, in hun eigen woorden: *"the 200
tok/s target is falsified on this hardware/target"*. En het pack schrijft zelf
voor wat dan geldt: doorgaan met optimaliseren richting 50–100, geen nieuwe
drafter trainen voor 200.

## 4. Wat het pack verder aandraagt, en waar het staat

| TreeSweep-hypothese | status |
|---|---|
| H1 hybride bomen zijn goedkoop genoeg | **weerlegd** — dit rapport, plus X1 |
| H2 dynamische boomdekking | **niet relevant meer**: dekking staat in de teller, en de teller is al maximaal gezet |
| H3 Mamba tree scan | **niet relevant meer**: het plafond is al gehaald zonder Mamba mee te tellen |
| H4 expert-major batching houdt de unie beheersbaar | **weerlegd door X1** — verhouding 1,0017 bij B=5 |
| H5 native MTP heeft meer massa dan top-1 | onmeetbaar nuttig: zelfs perfecte dekking haalt de poort niet |
| H6 one-shot dependent drafting · H7 diffusion | **begrensd**: verbeteren de teller, maar die is al op `N` gezet |
| H8 BranchCert · H9 OrbitANS · H10 PathQ | **begrensd door Y2-R1**: bytes halveren → 34% van de GEMV |
| H11 heterogeen drafting | verbergt draftkosten, die in dit plafond al nul zijn |
| H12 lange context | het plafond is bij korte context gemeten en wordt bij 262K slechter |

Alle twaalf zijn nu gemeten of begrensd door een meting. Geen ervan is nog een
open vraag over de doorvoer.

## 5. Claim boundary

Gemeten kosten van het **routed-expert-deel** van alle 23 MoE-lagen voor `N`
geverifieerde posities, op echte hidden states en echte officiële routes, elke
expert resident, korte context, capacity 72. Het plafond `1/c` is een
**bovengrens** op de doorvoer van welke boomverifier dan ook voor dit target op
deze GPU; het veronderstelt perfecte dekking, nul draftkosten en gratis Mamba,
attention, LM-kop, router, shared expert en state-commit. Het is **geen gemeten
doorvoer** en er bestaat geen runtime die het haalt. De vergelijking met X1 is
een kruiscontrole van dezelfde grootheid onder andere cachedruk, geen replicatie.

## 6. Artefacten

`Z1_TREE_VERIFIER_CEILING_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/z1_tree_verifier_ceiling.py` ·
`z1_tree_verifier_ceiling.json` ·
`scripts/lightningstream_nemotron/z1_independent_verify.py` ·
`z1_independent_verification.json` · `protected_verification_after_z1.json`
