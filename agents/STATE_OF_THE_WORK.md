# Waar we staan — Nemotron 3.5 Lightning 30B-A3B NVFP4 op 8 GiB

Bijgewerkt: 2026-08-15 (na E1 fase 2.1) · lees dit vóór je iets aanraakt

## In één alinea

Een 30B MoE-model draait causaal op een 8 GiB laptop-GPU door de experts vanaf
host te streamen. Een token kostte **41,98 ms**; de geadopteerde stack kost
**37,49 ms** (bit-identiek, fysiek gemeten over 3 × 512 tokens). Daar bovenop
komt nu **E1 fase 2.1** (device-resident routing + device-LRU, alle poorten
PASS, verifier 14/14): in hetzelfde meetregime 41,540 → **36,998 ms per
token**, nog eens −4,542 ms, eveneens met pariteit. Nog niet als default
geadopteerd — opt-in via `rt.device_cache = True`.

## Hoeveel tok/s?

| regime | tok/s | bron |
|---|---:|---|
| E1-2.1-arm, contexts_max=4096 | **27,0** | E1F21 A/B (36,998 ms) |
| 512-token rollout, context groeit mee | **26,7** | E6 (zonder 2.1) |
| kort, ctx 0, vaste diepte | **29,5** | NERVF-3 |
| ctx 262100 | **19,6** | NERVF-3 (vóór 2.1; niet opnieuw gemeten) |
| *roofline-plafond van deze machine* | *165 (ctx0) / 119 (lang)* | Y-lijn |
| *plafond mét volledige graph-residentie (oracle)* | *~41,5 bij ctx 64* | E1 fase 1 |

De runtime draait op ongeveer **17% van zijn roofline**. Dat is de kern van de
zaak: er is nog veel hoofdruimte, maar niet oneindig veel.

**Wat vaststaat over de doelen:** 50 en 100 tok/s zijn fysiek *niet*
uitgesloten, maar 50+ vraagt méér dan graph-residentie alleen (plafond ~41,5
bij ctx 64). **1000 tok/s is fysiek uitgesloten** — de gemeten streaming-
leesbandbreedte van 338,4 GB/s legt een harde ondergrens van 6,05 ms per
token bij ctx 0.

## Wat bewezen is

- **ERVF** (Exact-Reduction Virtual Fusion) — **1,936× sneller, 0/72
  numerieke verschillen**, breedte 16. Gerepliceerd van Qwen3-30B naar dit
  model.
- **v4-attentiekernel** — bitexact, −17,8%.
- **D1, accumulatie in routevolgorde** — run-to-run determinisme.
- **Geïntegreerd (E6)** — 41,98 → 37,49 ms, exact, VRAM ongewijzigd.
- **Geadopteerd (A1)** — ERVF + v4 + D1 staan default aan.
- **E1 fase 2.1 (2026-08-15)** — MoE-laag zonder één device→host-sync:
  routerkop, LRU-toewijzing en miss-staging (bulk-kopie 24,93 GB/s uit pinned
  host, DMA-pariteit) zijn kernels. −4,542 ms/token eager, pariteit vs
  bevroren A1-ids, capaciteitsinvariantie (56 ≡ 72), controle-arm faalde
  zoals vereist. Verifier 14/14 incl. bitexacte kernelspiegels. Rapport:
  `E1F21_DEVICE_ROUTING_REPORT_2026-08-15.md`.

## Wat weerlegd is — niet opnieuw proberen zonder nieuw idee

| idee | waarom dood |
|---|---|
| **Speculative decoding / MTP** | Langs drie onafhankelijke paden dicht: X1-ratio 1,0017, Z1-lineariteit R²=0,99986, K0/S13 route-unie boven pariteit. |
| **Gatherloze downflow** (E2 = NERVF-4) | −5,7 tot −7,4 ms per token. Strided host-reads halen 6,7 GB/s over PCIe tegen 85,9 vanaf device. |
| **GEMV die zelf van host leest** (M2) | 7,27 GB/s — marginaal. Daarom kopieert 2.1's staging-kernel bulk naar device (M1: 24,93 GB/s). |
| **Device-side routing via host-read** | Zelfde muur (V1). |
| **Byte-reductie / betere compressie** | Begrensd: halveren scheelt 34,2% (Y2-R1); OrbitANS-plafond 7,23% (O1). |
| **1000 tok/s** | Roofline sluit het uit. |

## De belangrijkste vondst van deze ronde

`_moe_cached` telde de zes routed experts op in **hit-dan-miss-volgorde** —
cachegeschiedenis bepaalde de optelvolgorde, en FP-optelling is niet
associatief. Vier fasen haalden hun exactheidspoort zonder dit te zien. De les
zit in werkregel 8: **bouw een controle-arm die moet falen.** In E1-2.1
verscheen dezelfde klasse fout als harnessbug (`enable_cache` resette de
device-LRU niet); de controle- en invariantiearmen vingen haar. *Cache-inhoud
en slot-staat zijn één invariant.*

## Twee ankers, niet door elkaar halen

- `reports/treesweep200/V35_GENERATION_ANCHOR.json` — hit-dan-miss-volgorde.
  Referentie voor elke meting van **vóór** 2026-08-15 A1.
- `reports/treesweep200/V36_DETERMINISTIC_ANCHOR.json` — routevolgorde.
  Referentie voor **al het werk hierna**.

Ze zijn **niet bit-vergelijkbaar**. Kies op de datum van je meting.

## De open richting: E1 fase 2.2 — graph-replay van de hele token

**Status: GEBOUWD, NOG NIET GEDRAAID.** Preregistratie is bevroren
(`E1F22_GRAPH_CAPTURE_PREREGISTRATION_2026-08-15.md`, poorten C/PAR/CTL/DET/
S1 ≥ 2,5 ms/VRAM). De code staat klaar maar is **ongemeten en ongetest**:

- `gpu_kernels.py`: `embed_gather_bf16`, `kv_append_fp8_dp`,
  `attn_decode_warp_fp8_gqa4_dp` (vaste grid (2,256), t/chunk op device,
  neutrale partials voor dode splits — combine slaat l≤0 al over),
  `argmax_part`/`argmax_final` (lage-index-ties), `pos_inc`.
- `runtime.py`: `graph_mode`-vlag, `_attention`-dp-tak, `setup_graph()`
  (pinned embed-kopie +0,656 GiB host, capture op eigen stream),
  `_step_body_graph()`, `step_graph(token_id)`, `ring_harvest(start, count)`.
  Graph-API is wel gesmoketest (capture→launch→correct, zie notebook).
- Te doen: runner `scripts/treesweep200/e1f22_graph_capture_ab.py` schrijven
  (4 armen: EGR/GRAPH/CTL/DET volgens de prereg), draaien, verifier
  (`e1f22_independent_verify.py`, nooit de runner importeren; kernelchecks:
  argmax vs cp.argmax incl. ties, gqa4_dp bitexact vs gqa4 over t=1..4096,
  embed_gather vs cupy-omzetting), rapport, registry-entry, notebook.
- Bekende risico's: event-hergebruik over 23 lagen in één capture
  (kill-criterium K1: fallback = single-stream capture); stale LRU-staat uit
  de capture-warmup is bewezen onschadelijk (INV). VRAM-poort < 64 MiB.

## Waar alles staat

| wat | waar |
|---|---|
| runtime | `src/moe_lab/lightningstream_nemotron/runtime.py` |
| NVFP4-kernels + ERVF | `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py` |
| overige GPU-kernels | `src/moe_lab/lightningstream_nemotron/gpu_kernels.py` |
| E-lijn (roofline) | `reports/treesweep200/` + `scripts/treesweep200/` |
| ERVF-replicatie | `reports/nervf_nemotron/` + `scripts/nervf_nemotron/` |
| registry met alle 33 experimenten | `reports/treesweep200/EXPERIMENT_REGISTRY.yaml` |
| eindrapport ERVF | `reports/nervf_nemotron/NERVF_NEMOTRON_FINAL_REPORT.md` |
| python | `./.venv-nemotron/Scripts/python.exe` |
| **nieuwe protected baseline** | `reports/lightningstream_nemotron/PROTECTED_80B_MANIFEST_AFTER_USER_COMMIT_2026-08-15.json` (gebouwd ná de eerste git-commit van de eigenaar; verifies voortaan hiertegen — de oude baseline dateert uit de pre-git periode en markeert .gitignore vals) |
