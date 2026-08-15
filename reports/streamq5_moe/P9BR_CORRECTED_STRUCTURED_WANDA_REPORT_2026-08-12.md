# P9B-R — gecorrigeerde structured-Wanda gesloten

## Uitkomst

De oorspronkelijke P9B-kwaliteitspass was ongeldig: boolean advanced indexing
maakte tijdelijke kopieën, zodat `.zero_()` de expertgewichten niet muteerde.
P9B evalueerde daardoor feitelijk de bestaande ongesnoeide Q5-baseline.

P9B-R heeft hetzelfde verzegelde masker met echte in-place broadcastmaskers
opnieuw geëvalueerd. Iedere laag wijzigde circa 302 miljoen oorspronkelijk
niet-nulle gewichtselementen en hield daarna exact nul niet-nulle waarden op
gemaskeerde posities over.

| validation-metriek | resultaat | poort |
|---|---:|---:|
| relatieve CE-toename | **+47,804%** | ≤2,5% |
| top-1-overeenkomst | **60,866%** | ≥90% |
| eind-hidden relatieve L2 | **0,4871** | diagnostisch |
| effectieve mutatie | 48/48 lagen | vereist |

Status: **validation_closed**. De test-split is niet geopend.

## Consequentie

GaugePack kan de kwaliteit van P9B niet erven, omdat P9B geen werkelijk
gepruned model testte. De compacte layoutprojectie van ongeveer 0,502× is dus
alleen een byteberekening en geen uitvoerbare kandidaat. Codec-, kernel-,
full-bank- en end-to-endpoorten voor dit vaste masker zijn gesloten.

Een andere selectie, lagere sparsiteit of opnieuw getraind model is een nieuw
kwaliteitsonderzoek en mag niet als reparatie van P9B worden gepresenteerd.
