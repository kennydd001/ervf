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
oneindig veel.

**Alle winst tot nu toe (V4-V6, elke kernel-batching, capaciteitstuning)
blijft binnen batch=1 — en het 165 tok/s-plafond zelf is ONDER die aanname
berekend.** Niets binnen batch=1 kan daar ooit boven komen; 100 tok/s vraagt
60,6% van dat plafond, V6 zit op 28,7%. De binnen-batch=1-hefbomen zijn nu
grotendeels uitgeput (down_proj/up_proj/accumulate/panel_scan/reduce_partials
allemaal gebatcht; gather/down_masked geprobeerd en afgewezen op VRAM;
capaciteitstuning geïntegreerd en bevestigd near-optimaal voor deze
lagenkeuze). **Meest kansrijke, nog niet aangepakte richting: batch>1.**
Een eerste, goedkope meting (`pro_research/diag_cross_sequence_union.py`,
2026-08-16) bevestigt reëel potentieel: bij 16 gelijktijdige sequenties is
de gemiddelde expert-unie 63,9 van 128 per laag — 66,6% van de no-overlap-
baseline (96), dus 33% minder unieke PCIe-gebonden expert-loads voor
evenveel nuttige tokens, zonder de speculatieve "draft tax" die MTP deed
mislukken (elke sequentie is al echt opgevraagd, niets wordt weggegooid).
**Het mechanisme is daarna ook fysiek getest, niet alleen geteld** — eerst
op één laag (`proto_batch_moe_layer.py`: 2,89× sneller, bitexact), daarna
**op alle 23 MoE-lagen met fetch en compute apart gemeten**
(`proto_batch_moe_multilayer.py`): **bitexact op alle 23 lagen, 0
mismatches totaal**; fetch-winst wisselt per laag (1,42×-3,15×, consistent
met de eerder gemeten niet-uniforme lokaliteit per laag); compute-tijd
blijft vlak tussen naive/batched (geen straf voor batchen). **Opgeteld
over alle 23 daadwerkelijk gemeten lagen: 367,05 ms → 214,43 ms, 1,71×
sneller** — een preciezer, minder toevallig-gunstig getal dan de losse
laag. Dit is een meting van het kernmechanisme, geen projectie — maar dekt
nog steeds niet de volledige doorvoer (down_proj, shared expert, attentie,
Mamba, KV-cache, graph-capture, routing/argmax/norm-overhead); een
tok/s-claim optellen zou een aanname zijn (werkregel 7 verbiedt dat). De
runtime heeft nul batch-ondersteuning (elke buffer 1D) — de volledige
integratie is een meerdere-weken-herontwerp, niet gestart deze sessie, maar
het kernmechanisme is nu consistent bewezen correct én fysiek sneller over
de hele MoE-stack, niet alleen theorie of één gunstige laag. Zie
`agents/RESEARCH_NOTEBOOK.md` 2026-08-16.

**Houdt dat stand onder een warme, evoluerende cache, of was het cold-cache-
alleen?** Alle bovenstaande metingen waren één cold-cache-snapshot, met
opzet. `pro_research/diag_batch_warm_cache.py` (2026-08-16) test dit expliciet:
N=4 sequenties, T=40 opeenvolgende **echte** stappen, met de **echte**
productie-`cache_assign`-kernel (niet herïmplementeerd) — gedeelde cache
(unie-gevoed) tegen 4 onafhankelijke caches, zelfde budget elk. **27,6%
minder missers over de volle 40 stappen, 28,0% minder in het laatste kwart
(warm steady-state)** — het voordeel verdwijnt dus niet zodra de cache warmt,
al is het kleiner dan de cold-cache-unie-cijfers alleen deden vermoeden
(cross-sequentie-deling concurreert met temporele lokaliteit die elke
sequentie toch al gratis krijgt). Bijvangst: 1×72-slot gedeelde cache tegen
4×72-slot naive — ~4× minder VRAM voor hetzelfde budget per sequentie. Zie
`agents/RESEARCH_NOTEBOOK.md` 2026-08-16, blok "Warme-cache-dynamiek".

**Drie resterende risico's uit `BATCH_ARCHITECTURE_DESIGN.md` zijn nu ook
gemeten (2026-08-16), en het beeld is per saldo positief maar genuanceerder:**
1. **Staggered posities (continuous batching)** — unie krimpt licht
   (89,4%→91,4% van max), geen ineenstorting.
2. **VRAM per extra sequentie** — 60,16 MiB (Mamba-state domineert, niet
   KV-cache); ruim budget (N tot 30) buiten graph-capture om, 0 MiB binnen
   V6's volledige graph — het echte knelpunt is graph-capture zelf, niet
   batch>1's eigen kost.
3. **Eerste gecombineerde meting (up_proj + down_proj-deling tegelijk, één
   laag, N=8)** — bitexact, maar **1,209× (+20,9%), kleiner dan de
   afzonderlijke cijfers deden vermoeden**: down_proj se GEMV werd zelf
   trager door een grotere gedeelde mirror (geheugenlocaliteit), een
   interactie-effect dat pas zichtbaar werd toen beide mechanismen samen
   draaiden. Zie `agents/RESEARCH_NOTEBOOK.md` 2026-08-16 voor alle drie.

**N-schaling van de niet-gedeelde componenten, viermaal gecheckt, plus de
oorzaak gevonden (2026-08-16).** Naast attentie (bevestigd lineair) en Mamba
(gecorrigeerd, mild supra-lineair) zijn ook shared-expert (**bevestigd
lineair**, 0,85-0,92× ideaal) en lm_head (**nieuw risico, nooit eerder
genoemd**: 1,19-1,24× ideaal, groter dan Mamba se straf, op de duurste GEMV
van het model) gemeten. Patroon: hoe duurder de kernel per aanroep, hoe
slechter de schaling — bevestigd als **reëel GPU-SM-klokverval** (36%,
2685→~1710 MHz binnen ~1 seconde aanhoudende belasting,
`diag_lmhead_throttle_check.py`). **Geruststellend:** `clocks.mem` bleef
exact 9001 MHz, geen afwijking — dit project se roofline is
geheugenbandbreedte-gebonden, dus de kern-roofline (165 tok/s ctx0) en V6's
47,41 tok/s-record zijn **niet** bedreigd. Raakt alleen reken-zware kernels
specifiek. Zie `agents/RESEARCH_NOTEBOOK.md` 2026-08-16.

**EERSTE ECHTE END-TO-END N=2-METING (2026-08-16) — de belangrijkste
mijlpaal van de batch>1-lijn tot nu toe.** Alles hiervoor was component-
niveau (één MoE-laag, geïsoleerde kernels, read-only diagnostiek). 
`pro_research/proto_multi_seq_full_model.py` draait voor het eerst het
**echte, volledige 52-lagen model**, meerdere **echte** decode-stappen, voor
N=2 sequenties, met een **echt gemeten** aggregate tok/s-getal — via een
generiek state-wisselmechanisme (~30 dynamische buffers per sequentie
gewisseld vóór aanroepen van de ongewijzigde productie-`rt.step()`, gewichten
en MoE-device-cache blijven gedeeld). Een `pos`-bug (plain-int-rebinding
i.p.v. in-place-mutatie) gevonden en gefixt vóór meting. **Correctheidspoort:
bitexact onder volledige interleaving tegen onafhankelijke controleruns,
15/15 tokens, beide sequenties.** Resultaat (kale eager-configuratie, geen
graph/selectieve-ERVF/gebatchte-kernels, schoon voor een N=1-vs-N=2-
vergelijking): **N=1 solo 29,798 tok/s vs N=2 naive (nog GEEN expliciete
deel-logica, alleen incidenteel warm-cache-hergebruik) 31,411 tok/s
aggregate — 1,054× (+5,4%), reëel en positief.** Bevestigt dat het
mechanisme praktisch werkt; de expliciete unie-gevoede MoE-deling
(`proto_batch_moe_layer_combined.py`, al bitexact bewezen op één laag) nog
niet in deze staplus geïntegreerd — dat is de directe vervolgstap. Zie
`agents/RESEARCH_NOTEBOOK.md` 2026-08-16, blok "EERSTE ECHTE END-TO-END
METING".

**Vervolg, zelfde dag — de expliciete deling geïntegreerd: bitexact
correct, maar 12× TRAGER, een belangrijke eerlijke uitkomst.**
`pro_research/proto_multi_seq_moe_shared.py` bouwt de al bewezen unie-
gevoede MoE-deling in de echte staplus, alle 23 lagen. **Correctheidspoort
GESLAAGD** (bitexact tegen onafhankelijke `_moe_dev`-referentieruns, 12/12
tokens × 2 sequenties) — bevestigt en passant voor het eerst dat `gemv_into`
en productie se `gemv_ervf_indirect` bitexact gelijk zijn. **Timing eerste
versie: 2,655 tok/s aggregate, tegen 31,411 (naive) en 29,798 (solo) — een
12× regressie.** Oorzaak: de deling is puur in Python gebouwd (host-syncs en
kleine kernel-launches per sequentie/unie-expert per laag per stap) — exact
de overhead die productie se `_moe_dev` zorgvuldig vermijdt. **Eén
overduidelijke inefficiëntie gevonden en gefixt (zelfde dag): een numpy-
array werd onnodig naar cupy geconverteerd en dan element-voor-element
teruggelezen BINNEN een lus over `npanel` panelen — honderdduizenden
overbodige host-syncs. Na de fix, nog steeds bitexact: 9,469 tok/s (3,57×
sneller), nog 3,3× trager dan naive.** Het mechanisme zelf is niet fout
(bitexact bewezen); een naïeve Python-orkestratie ervan kost veel, en een
groot deel daarvan is vermijdbaar (bewezen) maar niet alles (het resterende
gat blijft). Bevestigt scherp waarom `BATCH_ARCHITECTURE_DESIGN.md` een
echte integratie als meerdere-weken CUDA-engineeringwerk scoopte. Zie
`agents/RESEARCH_NOTEBOOK.md`
2026-08-16, blok "Expliciete MoE-deling geïntegreerd in de echte staplus".

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
