# Waar we staan — Nemotron 3.5 Lightning 30B-A3B NVFP4 op 8 GiB

Bijgewerkt: 2026-08-16 (na PRO V3 + model-identiteitsonderzoek) · lees dit vóór je iets aanraakt

## ⚠️ Modelidentiteit — lees dit eerst

**Alles hieronder gemerkt vóór 2026-08-16 is gemeten op `models/nemotron_3_5_lightning`,
wat bij nameting `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` blijkt te zijn (verkeerd
gedownload, misleidend hernoemd — zie `N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md`).
Het echte opdrachtdoel, `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, staat in
`models/nemotron_3_5_lightning_v35` en is dat wat `pro_research` default gebruikt.**
Bewijs: ankerpad `config.json.max_position_embeddings = 262144` (Nano-plafond)
vs `_v35`-pad `1048576` (Lightning-plafond); `A1_ADOPTION_PRECONDITION.json`
registreert `model_dir: nemotron_3_5_lightning`. Volledige ketting in
`RESEARCH_NOTEBOOK.md`, blok 2026-08-16. De kernelwinsten hieronder (ERVF,
v4-attentie, D1) zijn zeer aannemelijk overdraagbaar naar Lightning — N2R
adjudiceerde de tensor-vormen als "shape-identiek op 8 byte na", en V3's eigen
bitexacte pariteit op het échte Lightning-model bevestigt dat fysiek — maar de
tok/s-tabel hieronder is strikt genomen een Nano-tabel totdat de closed
`treesweep200`-lijn opnieuw draait met `LS_MODEL_DIR=nemotron_3_5_lightning_v35`.
`pro_research`'s V3-cijfers (sectie hieronder) zijn wél al op het juiste model.

## In één alinea

Een 30B MoE-model draait causaal op een 8 GiB laptop-GPU door de experts vanaf
host te streamen. Een token kostte **41,98 ms**; de geadopteerde stack kost
**37,49 ms** (bit-identiek, fysiek gemeten over 3 × 512 tokens). Daar bovenop
komt nu **E1 fase 2.1** (device-resident routing + device-LRU, alle poorten
PASS, verifier 14/14): in hetzelfde meetregime 41,540 → **36,998 ms per
token**, nog eens −4,542 ms, eveneens met pariteit. Nog niet als default
geadopteerd — opt-in via `rt.device_cache = True`.

## Hoeveel tok/s?

**Nano-tabel (closed treesweep200-lijn, vóór 2026-08-16 identiteitscorrectie):**

| regime | tok/s | bron |
|---|---:|---|
| E1-2.1-arm, contexts_max=4096 | **27,0** | E1F21 A/B (36,998 ms) |
| 512-token rollout, context groeit mee | **26,7** | E6 (zonder 2.1) |
| kort, ctx 0, vaste diepte | **29,5** | NERVF-3 |
| ctx 262100 | **19,6** | NERVF-3 (vóór 2.1; niet opnieuw gemeten) |
| *roofline-plafond van deze machine* | *165 (ctx0) / 119 (lang)* | Y-lijn |
| *plafond mét volledige graph-residentie (oracle)* | *~41,5 bij ctx 64* | E1 fase 1 |

**Lightning-tabel (echte doelmodel, `nemotron_3_5_lightning_v35`):**

| regime | tok/s | bron |
|---|---:|---|
| kale stack (vóór ERVF/D1/A1), ctx 0 | 27,743 | HANDOVER_TO_KIMI 2026-08-15 |
| kale stack, ctx 262100 | 18,424 | HANDOVER_TO_KIMI 2026-08-15 |
| device-cache eager (EGR), ctx ≤4096, smoke n=45 | 31,75–31,85 | PRO V3-G0S/G1B smoke, 2026-08-16 |
| + graph-safe residency (V3-G0S), smoke | 34,96 | 28,6063 ms, gain 2,8931 ms (+10,1%) |
| + selectieve ERVF, geen graph (V3-G1B), smoke | 35,51 | 28,158 ms, gain 3,3841 ms (+10,73%) |
| + beide fysiek geïntegreerd (V4), full 256×3, 765 samples | 41,13 | 24,3152 ms, gain 6,8634 ms (+22,0%) vs zelfde-sessie EGR (31,1786 ms) |
| + batched down_proj (V5: panel_scan+reduce_partials+accumulate), eager only, full 256×3 | (component) | 28,7823 ms, gain 3,1552 ms (+9,88%) vs eager BASE_A/B-midden |
| + batched up-proj ERVF-GEMV, eager only, full 256×3, apart van V5 gehouden | (component) | 26,759 ms, gain 1,7423 ms (+6,11%) bovenop V5 |
| + alle vijf geïntegreerd (V6), full 256×3, 765 samples | 47,37 | 21,1118 ms, gain 9,9855 ms (+32,1%) vs zelfde-sessie EGR (31,0973 ms) |
| + **per-laag cachecapaciteit erbij (budget-neutraal, geen VRAM-kost), full 256×3, 765 samples** | **47,41** | **21,0923 ms, gain 10,3366 ms (+32,9%) vs zelfde-sessie EGR (31,4289 ms)** |

V6 (2026-08-16) is het huidige record: device-resident routing + graph-safe
residency + selectieve ERVF (V4) + batched `panel_scan`/`reduce_partials`/
`weighted_accumulate_ind`/up-proj ERVF-GEMV in `_moe_dev` (V5, tweemaal
uitgebreid) — alle mechanismen tegelijk gevangen in één CUDA-graph, omdat
`_install_selective` (patcht `rt.k.mv_bf16`/`mv_fp8_tensor`, gebruikt in
attentie/Mamba) en `install_batched_moe_dev` (vervangt `rt._moe_dev`
volledig, alleen MoE) verschillende aanroeppunten raken en dus vrij te
combineren zijn. De accumulate-batching gebruikt bewust GEEN mechanische
kopie van het panel_scan/reduce_partials-patroon (dat zou een race-conditie
+ mogelijk gewijzigde FP-optelvolgorde geven) — een nieuwe kernel
reproduceert de exacte `s=0..5`-fmaf-volgorde uit één launch. De up-proj-
batching is wél een mechanische kopie (onafhankelijke output per slot, geen
race) maar dan van de zorgvuldig geverifieerde WIDTH-16-ERVF-reductiekernel
— de referentiekernel staat daarom letterlijk naast de batched versie in de
broncode om transcriptiefouten te vermijden. Alle correctheidspoorten groen
(bitexact vs EGR over 256×3, deterministisch, controle-arm wijkt af,
dot-graph bevat alle vijf kernelnamen, VRAM binnen budget). Onderweg vond de
VRAM-poort een echte bug (een overbodige extra `mirror`-buffer per laag,
~61,6 MB) — gevonden, begrepen, gefixt, opnieuw bitexact geverifieerd.
Zie `agents/RESEARCH_NOTEBOOK.md`, blokken "PRO V5 + V6", "weighted_accumulate_ind"
en "Up-proj ERVF-GEMV gebatcht", en `pro_research/results/PRO_V6_FULL_STACK.json`.

De GPU-roofline (165/119 tok/s) is hardware-eigenschap, niet modelafhankelijk,
en blijft dus gelden voor Lightning. De runtime draait nu op **28,7%** van het
ctx0-roofline (was 26,9% na accumulate-batching, 24,9% bij V4, 17% op de oude
Nano-lijn). Dat is de kern van de zaak: er is nog veel hoofdruimte, maar niet
oneindig veel. **Doel van deze sessie: 100 tok/s (60,6% van roofline) — nog
een factor 2,11× te gaan vanaf V6, niet uitgesloten, ver van bewezen. Er is
nog geen geïdentificeerd pad naar 100; de resterende bekende hefboom
(PCIe-gather-herstructurering van
`gather_down_sparse_ind`, buiten V5's scope gehouden) levert volgens de
ablatiemeting hooguit een paar ms/token, niet genoeg alleen. Componentafbraak
(2026-08-16) laat zien dat MoE **57,8%** van het token is — groter dan alleen
down_proj — dus verder zoeken binnen MoE (shared-expert-GEMV's, up-proj-
ERVF-GEMV, routing/cache-kernels) is de meest kansrijke volgende richting.**

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
| **Speculative decoding / MTP (Nano)** | Langs drie onafhankelijke paden dicht op de oude (Nano-)lijn: X1-ratio 1,0017, Z1-lineariteit R²=0,99986, K0/S13 route-unie boven pariteit. |
| **Speculative decoding / MTP (Lightning, S10)** | Heropend op het juiste model (echte draft-weights aanwezig) — acceptatiegraad haalde zijn poort (`A=2,114`), maar de route-unie over 5 tokens (`pro_research/diag_mtp_route_union.py`, 2026-08-16) is 19,88/128 experts per laag, 3,313× t.o.v. 6 voor één token. Ingevuld in het rapport z'n eigen rekensom: 57,51 ms/token speculatief vs. 54,28 ms/token niet-speculatief — **6,0% trager**. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16. |
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
