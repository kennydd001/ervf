# E2 — gatherloze downflow: weerlegd, met de reden waarom

Datum: 2026-08-15 · Registry `TREESWEEP200`
Verdict: **Het mechanisme is gebouwd, exact, en conclusief slechter op alle drie
de contextdiepten: −5,70 / −7,56 / −7,38 ms per token. De gather van 8,19 ms
verdient zichzelf terug. E2 is weerlegd, niet ongebouwd.**
Terminal state: `e2_gatherless_downflow_falsified_by_pcie_access_pattern`

## Waarom dit rapport naar NERVF-4 wijst

E2 en NERVF-4 zijn **hetzelfde experiment**. De treesweep200-lijn schreef het op
als "gatherloze downflow" (agent 20), de NERVF-lijn als fase 4 van de
ERVF-replicatie. Beide zetten exact één variabele om: `gather_from_host = False`
in `down_masked_into`, zodat de masked down-GEMV rechtstreeks van host leest in
plaats van via een device-gather.

Het is één keer gemeten, met de strengere opzet van de twee (drie armen
`gather_a / gatherless / gather_b`, drie contextdiepten tot 262100, exactheid
tegen het bevroren anker). Ik meet het niet nog eens over voor de registry;
ik neem het resultaat over en noteer waar het vandaan komt.

Bron: `reports/nervf_nemotron/nervf4_gatherless_ab.json` ·
`reports/nervf_nemotron/NERVF_4_REPORT_2026-08-15.md`

## Uitkomst

| context | MoE-blok speedup | tokenwinst | conclusief |
|---:|---:|---:|:--:|
| 0 | 0,785× | **−5,698 ms** | ✅ |
| 131072 | 0,717× | **−7,555 ms** | ✅ |
| 262100 | 0,725× | **−7,380 ms** | ✅ |

| poort | uitslag |
|---|:--|
| `exact_support_and_weight_semantics` | ✅ — alle drie de armen identiek aan het anker |
| `gather_time_reduction_ge_80pct` | ❌ — vereist +6,55 ms, gemeten **−5,99 ms** |
| `down_path_speedup_ge_1_8x` | ❌ — gemeten 0,72 tot 0,79× |
| `no_hidden_duplicate_bank` | ✅ — geen extra bank |

De exactheidspoort haalt het wél. Dat is belangrijk: de weerlegging is een
**prestatie**-weerlegging, niet een correctheidsfout. Het idee werkt, het is
alleen langzamer.

## De reden

De masked down-GEMV leest per thread losse bytes uit een sterk strided patroon.
Over PCIe haalt dat **6,7 GB/s**; hetzelfde patroon vanaf device haalt **85,9
GB/s** (V1, dezelfde meting die eerder de host-read-route voor device-side
routing sloot). De device-gather kost 8,19 ms maar zet het strided patroon om in
een gecoalesceerde lees, en die 12,8× betere doorvoer betaalt de gather ruim
terug.

Dit is dezelfde bevinding als V1, in een tweede vorm: **op deze machine is de
PCIe-bandbreedte voor strided toegang de bindende beperking, niet het aantal
kernellanceringen of de gekopieerde bytes.** Elk toekomstig ontwerp dat de
gather wil weglaten moet eerst het toegangspatroon aanpakken, niet de gather.

## Wat dit voor E6 betekent

E2 was in het oorspronkelijke plan een van de vier componenten van de
E6-integratie. Die post valt weg. De E6-run bevat daarom alleen E4(v4) + E5(ERVF)
+ D1, en de eindpoort van ≥50 tok/s blijft daarmee buiten bereik in dit regime —
zie `E6_INTEGRATED_REPORT_2026-08-15.md`.

## Claim boundary

Overgenomen meting, geen nieuwe run. Drie armen, drie contextdiepten, 64 tokens
per arm per diepte, capacity 72, één modelload, exactheid tegen het bevroren
V35-anker. Tokentijden zijn end-to-end wandtijd inclusief synchronisatie. De
6,7-tegen-85,9-GB/s-cijfers komen uit V1 en zijn microbenchmarks van het
toegangspatroon, geen tokentijden.

## Artefacten

`scripts/nervf_nemotron/nervf4_gatherless_ab.py` ·
`reports/nervf_nemotron/nervf4_gatherless_ab.json`
