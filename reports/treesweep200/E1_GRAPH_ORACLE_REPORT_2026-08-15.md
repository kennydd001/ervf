# E1 fase 1 — N1-oracle gereproduceerd, en ERVF maakt E1 waardevoller

Datum: 2026-08-15 · Registry `TREESWEEP200`
Verdict: **G-E1-R1 gehaald: 22,2% verwijderbaar tegen N1's 23,7%, een afwijking van 1,5 procentpunt binnen de tolerantie van 5. Nieuw en niet triviaal: met ERVF aan stijgt het aandeel naar 27,0% — de uitgifte-overhead is per-launch en verandert niet als de kernels sneller worden, dus ERVF vergroot juist wat E1 kan opleveren.**
Terminal state: `e1_phase1_oracle_reproduced_ervf_raises_the_prize`

## Meting

CUDA-graph-replay van één tokenreeks met **bevroren routes** (capture verbiedt
synchronisatie), tegen eager uitvoering van exact dezelfde reeks.

| arm | eager | graph | verwijderbaar | absoluut |
|---|---:|---:|---:|---:|
| baseline | 36,204 ms | 28,152 ms | **22,2%** | 8,052 ms |
| **met ERVF** | 33,037 ms | 24,112 ms | **27,0%** | **8,925 ms** |

| poort | eis | gemeten | |
|---|---|---:|:--:|
| **G-E1-R1** reproductie van N1 | 23,7% +- 5pp | **22,2%** (dev 1,5pp) | ✅ |

## Wat het nieuwe getal zegt

De **absolute** uitgifte-overhead beweegt nauwelijks: 8,05 ms zonder ERVF, 8,93
met. Dat is precies wat je verwacht — die kosten zitten per kernel-launch en per
host-stap, niet in hoe snel een kernel zijn bytes verwerkt. Maar het **aandeel**
stijgt van 22,2% naar 27,0%, omdat ERVF de rekentijd eromheen kleiner maakte.

Daaruit volgt iets dat voor de planning telt: **ERVF en graph-residentie zijn
complementair, en ERVF verhoogt de opbrengst van E1 in plaats van hem te
verlagen.** Elke volgende kernelversnelling doet dat opnieuw.

Het gecombineerde plafond op deze diepte: 33,037 − 8,925 = **24,112 ms**, oftewel
ongeveer 41,5 tok/s bij ctx 64 — als E1 volledig realiseerbaar zou zijn.

## Wat er nog tussen zit

Dat "als" is de hele fase 2. Een graph over een echte token vraagt dat de routes
op device blijven, en V1 heeft de voor de hand liggende weg daarheen gesloten:
de GEMV die zijn codes uit mapped host leest haalt 6,7 GB/s tegen 85,9 vanaf
device, 1,42x te duur tegen de sync die het bespaart. Het alternatief —
device-gestuurde indirectie waarbij de gecoalesceerde gather de miss afhandelt en
de kernels hun expert-pointer uit device-geheugen lezen — is ongebouwd.

Deze meting geeft dat ontwerp zijn budget: **8,9 ms per token**, en het moet die
verdienen zonder de miss-afhandeling duurder te maken dan NERVF-4 liet zien.

## Claim boundary

Graph-replay met **bevroren routes** is na de eerste token semantisch onjuist en
wordt hier uitsluitend gebruikt als tijd-oracle voor de vraag hoeveel van een
token uitgifte is in plaats van rekenwerk. Het is een **bovengrens** op wat een
device-resident ontwerp kan terugwinnen, geen haalbaar runtime-cijfer; er bestaat
geen graph-resident runtime. Gemeten bij ctx 64, capacity 72, 24 samples per arm.

## Artefacten

`scripts/treesweep200/e1_graph_oracle.py` · `E1_GRAPH_ORACLE.json`
