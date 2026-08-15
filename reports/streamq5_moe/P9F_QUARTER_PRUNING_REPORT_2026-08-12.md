# P9F — 25% structured-Wanda gesloten

Na de 50%-fails testte P9F een eigen, minder agressieve hypothese: per expert
de beste 576/768 kanalen behouden volgens opnieuw berekende
activatie-RMS×down-normscores. De maskering muteerde op iedere laag werkelijk
circa 151 miljoen niet-nulle gewichten en liet nul gemaskeerde waarden over.

| validation-metriek | resultaat | poort |
|---|---:|---:|
| relatieve CE-toename | **+22,846%** | ≤2,5% |
| top-1-overeenkomst | **74,803%** | ≥90% |
| eind-hidden relatieve L2 | **0,3749** | diagnostisch |
| effectieve mutatie | 48/48 lagen | vereist |

Status: **validation_closed**; test bleef dicht. Ook 25% globale
expertkanaalpruning is dus ruim buiten de kwaliteitsgrens. Een nog kleinere of
laagselectieve variant is een nieuw, waarschijnlijk incrementeel onderzoek en
kan niet de geprojecteerde 2× GaugePack-doorbraak leveren.
