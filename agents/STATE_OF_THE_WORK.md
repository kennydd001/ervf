# Waar we staan — Nemotron 3.5 Lightning 30B-A3B NVFP4 op 8 GiB

Bijgewerkt: 2026-08-15 · lees dit vóór je iets aanraakt

## In één alinea

Een 30B MoE-model draait causaal op een 8 GiB laptop-GPU door de experts vanaf
host te streamen. Een token kostte **41,98 ms**; hij kost nu **37,49 ms**, en de
uitvoer is **bit-identiek** aan wat de langzame versie produceerde. Die winst is
niet opgeteld uit componentmetingen maar fysiek gemeten in een A/B over 3 × 512
tokens. De snelste bewezen stack staat sinds fase A1 **als default aan**.

## Hoeveel tok/s?

| regime | tok/s | bron |
|---|---:|---|
| kort, ctx 0, vaste diepte | **29,5** | NERVF-3 |
| 512-token rollout, context groeit mee | **26,7** | E6 |
| ctx 262100 | **19,6** | NERVF-3 |
| *roofline-plafond van deze machine* | *165 (ctx0) / 119 (lang)* | Y-lijn |
| *plafond mét graph-residentie (oracle)* | *~41,5 bij ctx 64* | E1 fase 1 |

De runtime draait op ongeveer **17% van zijn roofline**. Dat is de kern van de
zaak: er is nog veel hoofdruimte, maar niet oneindig veel.

**Wat vaststaat over de doelen:** 50 en 100 tok/s zijn fysiek *niet* uitgesloten.
**1000 tok/s is fysiek uitgesloten** — de gemeten streaming-leesbandbreedte van
338,4 GB/s legt een harde ondergrens van 6,05 ms per token bij ctx 0.

## Wat bewezen is

- **ERVF** (Exact-Reduction Virtual Fusion) — subwarps van 16 lanes per rij, elk
  met eigen virtuele accumulatoren, die de reductieboom van de referentiekernel
  exact reconstrueren. **1,936× sneller, 0 van 72 numerieke verschillen** op
  vier breedtes. Dit is een replicatie van de eerdere Qwen3-30B-doorbraak op een
  architecturaal ander model, andere quantisatie, andere shape — en hij koos
  dezelfde breedte 16.
- **v4-attentiekernel** — bitexact, −17,8%.
- **D1, accumulatie in routevolgorde** — maakt de runtime run-to-run
  deterministisch. Zie hieronder; dit was de belangrijkste vondst.
- **Geïntegreerd (E6)** — 41,98 → 37,49 ms per token, exact, VRAM ongewijzigd.
- **Geadopteerd (A1)** — alle drie staan default aan en zijn als default
  geverifieerd.

## Wat weerlegd is — niet opnieuw proberen zonder nieuw idee

| idee | waarom dood |
|---|---|
| **Speculative decoding / MTP** | Langs drie onafhankelijke paden dicht: X1-ratio 1,0017, Z1-lineariteit R²=0,99986, K0/S13 route-unie boven pariteit. |
| **Gatherloze downflow** (E2 = NERVF-4) | −5,7 tot −7,4 ms per token. Strided host-reads halen 6,7 GB/s over PCIe tegen 85,9 vanaf device. De gather van 8,19 ms verdient zichzelf terug. |
| **Device-side routing via host-read** | Zelfde muur (V1). |
| **Byte-reductie / betere compressie** | Begrensd: halveren scheelt 34,2% (Y2-R1); OrbitANS-plafond 7,23% (O1). |
| **1000 tok/s** | Roofline sluit het uit. |

## De belangrijkste vondst van deze ronde

`_moe_cached` telde de zes routed experts op in **hit-dan-miss-volgorde**. Welke
expert een hit is hangt van de LRU-staat af, dus twee runs met verschillende
cachegeschiedenis telden in verschillende volgorde op — en FP-optelling is niet
associatief. **Twee armen met identieke configuratie divergeerden.**

Vier eerdere fasen haalden hun exactheidspoort zonder dit te zien, omdat ze
allemaal over 2 × 64 tokens vergeleken binnen één proces. De les zit nu in
werkregel 8: **bouw een controle-arm die moet falen.**

Opgelost door rekenvolgorde en optelvolgorde te scheiden: rekenen blijft
hit-eerst (de latencywinst blijft), maar de zes bijdragen gaan naar aparte
buffers en worden na de lus in routevolgorde opgeteld. Kosten: 64 KB, nul extra
kernels.

## Twee ankers, niet door elkaar halen

- `reports/treesweep200/V35_GENERATION_ANCHOR.json` — hit-dan-miss-volgorde.
  Referentie voor elke meting van **vóór** 2026-08-15 A1.
- `reports/treesweep200/V36_DETERMINISTIC_ANCHOR.json` — routevolgorde.
  Referentie voor **al het werk hierna**.

Ze zijn **niet bit-vergelijkbaar**. Kies op de datum van je meting.

## De enige echt open richting

**E1 fase 2 — graph-resident token.** Fase 1 is af en heeft het budget gemeten:
een CUDA-graph die de hele tokenlus uitgeeft haalt er **8,9 ms per token** uit,
en dat getal **stijgt** met ERVF aan (van 22,2% naar 27,0% van de tokentijd),
omdat uitgifte-overhead per-launch is en niet meebeweegt met snellere kernels.

De blokkade: graph-capture verbiedt synchronisatie, en de huidige lus leest de
routes terug naar de host om te beslissen welke experts gehaald worden. Dat moet
device-side. De voor de hand liggende variant is al gesloten (V1). Wie dit
oppakt moet dus een écht nieuw ontwerp meenemen, geen herhaling.

## Waar alles staat

| wat | waar |
|---|---|
| runtime | `src/moe_lab/lightningstream_nemotron/runtime.py` |
| NVFP4-kernels + ERVF | `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py` |
| overige GPU-kernels | `src/moe_lab/lightningstream_nemotron/gpu_kernels.py` |
| E-lijn (roofline) | `reports/treesweep200/` + `scripts/treesweep200/` |
| ERVF-replicatie | `reports/nervf_nemotron/` + `scripts/nervf_nemotron/` |
| registry met alle 31 experimenten | `reports/treesweep200/EXPERIMENT_REGISTRY.yaml` |
| eindrapport ERVF | `reports/nervf_nemotron/NERVF_NEMOTRON_FINAL_REPORT.md` |
| python | `./.venv-nemotron/Scripts/python.exe` |
