# Waar we staan — Nemotron 3.5 Lightning 30B-A3B NVFP4 op 8 GiB

Bijgewerkt: 2026-08-16 (na PRO V3 + model-identiteitsonderzoek) · lees dit vóór je iets aanraakt

**Voor een eerlijke plafondanalyse en concrete routekaart naar 100 tok/s
(single-stream vs. aggregate, wat wél en niet bereikbaar is, en de precieze
resterende technische stappen): zie `agents/PATH_TO_100_TOKS.md`.**

## 🚩 Stand van zaken 2026-08-16, einde sessie: géén van beide routes haalt de 100 met de huidige kernels

- **single-stream**: record 47,41 tok/s; hard theoretisch maximum **~94 tok/s**
  (elke gemeten ms hoofdruimte opgeteld). Negen ingrepen gebouwd en gemeten,
  beste opbrengst −0,42 ms.
- **batch**: gemeten batchwinst op de ERVF-paden is **×1,64 bij N=4** — en
  **99,2% van het VRAM-verkeer gaat door een ERVF-kernel**. Projectie
  **~71 tok/s bij N=4, ~83 bij N=8**.
- **De reden is één samenhangend feit:** ERVF haalt bij N=1 al **247-266 GB/s =
  77% van het apparaatplafond**. Batching amortiseert gewichtslezingen en werkt
  alleen als je bandbreedte verspilde — V4-V6 zijn daar al mee gestopt.
- **K-tiling is geprobeerd en WEERLEGD** (×1,14 bij N=4 tegen ×1,64 met X uit
  global; L1 leverde het 16× hergebruik al). Het batchplafond staat daarmee vast.

### 🎯 De grootste post die nog open staat — en die is nieuw

| | GB/s |
|---|---:|
| apparaat, puur streamen | 345,9 |
| ERVF **geïsoleerd**, koud, N=1 | **248-267** (72-77%) |
| Mamba **in de lus** (892 MB / 5,168 ms) | **172,6** (50%) |

Dezelfde kernel, dezelfde shape — **in de lus maar 64% van zijn geïsoleerde
snelheid**. Als elke GEMV in de lus zijn geïsoleerde tempo haalde, kostte een
token 7,67 ms VRAM + 2,47 ms PCIe = **10,1 ms ≈ 99 tok/s**. We meten 21,24 ms.
**Bijna de helft van het token gaat verloren tussen "wat de kernel kan" en "wat
de kernel in de lus doet".** Hypotheses (ongemeten): L2-verdringing tussen lagen,
bandbreedteconcurrentie met `copy_stream`, thermische throttling. Dit is
orthogonaal aan de single-stream/batch-keuze — sluit je dit gat, dan profiteren
beide routes. **Dit is de eerstvolgende meting.**

## 🔎 De volledige rekening (2026-08-16, alles gemeten)

Voor het eerst is elke term van het budget nagemeten in plaats van geschat, en
sluiten de metingen op elkaar aan. Record blijft **47,41 tok/s (21,09 ms)**.

**De machinevloer, uit drie onafhankelijke metingen:**
- apparaat streamt **345,9 GB/s** (512 MiB, byte-geverifieerd)
- de dense-GEMV-kernel haalt koud **230-261 GB/s** (67-76%); de eerder gemeten
  336 was een **L2-artefact** (L2 = 32 MiB, Mamba's in_proj is 27,7 MB)
- PCIe Gen5 ×8 levert **25,9 GB/s** (byte-geverifieerd)

| | MB/token | vloer |
|---|---:|---:|
| VRAM (Mamba 892 + routed-up 387 + shared/gate 290 + attn 281 + lm_head 198) | 2048 | 8,22 ms |
| PCIe (routed down_proj, sparse) | ~64 | 2,47 ms |
| **serieel** | | **10,69 ms = 93,6 tok/s** |
| **volledig overlappend** | | **8,22 ms = 122 tok/s** |

**Waar de token heen gaat — GEMETEN IN DE GEVANGEN GRAPH** (marginale methode,
alle armen bitexact, drift 0,378 ms). Basis-midden **20,7722 ms = 48,14 tok/s**.

| component | gemeten | vloer | **hoofdruimte** | efficiëntie |
|---|---:|---:|---:|---:|
| **MoE** | 9,408 | 5,19 (2,72 VRAM + 2,47 PCIe) | **4,22** | — |
| rest (lm_head, norms, embed, argmax) | ~3,72 | ~0,9 | ~2,8 | — |
| **Mamba** | 5,168 | 3,582 | **1,586** | 69,3% |
| **attention** | 2,479 | 1,128 | **1,352** | **45,5%** ← minst efficiënt |
| som van de drie marginalen | 17,056 (82% van het token) | | | |

Totale hoofdruimte ~10 ms van de 20,77 ms → vloer rond **10,7 ms ≈ 93 tok/s**
bij seriële PCIe, wat de plafondrekening hierboven langs een tweede
onafhankelijke weg bevestigt.

De eerdere **eager** tabel is vervallen: die bevatte ~7,75 µs CPU-uitgiftetijd
per kernel-launch (MoE doet er ~414 per token) en wees `down_masked` ten
onrechte aan als slechtste pad — dat draait op 60% van zijn vloer en is prima.
Zie het correctieblok in `RESEARCH_NOTEBOOK.md`.

**Conclusie.** 100 tok/s = 10,0 ms ligt binnen de fysica maar vraagt twee dingen
tegelijk: de PCIe-gather grotendeels verstoppen (B3 werkt en is bitexact, maar
haalt nu 16,8%) én ~10 ms kernel-inefficiëntie wegwerken. De grootste post is
MoE (4,22 ms); het minst efficiënte pad is **attention op 45,5%** — waar
PV2-11 (Q/K/V one-launch) al een exacte kandidaat klaar heeft liggen die
uitsluitend op de driftpoort sneuvelde, en die poort haalt de huidige harness
ruim.

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

**Multi-sequentie-tabel (aggregate tok/s, Lightning, 2026-08-16):**

| opzet | aggregate tok/s | bron |
|---|---:|---|
| N=2 naïef-eager (gedeelde cap-72, geen graph) | 31,66 | proto_multi_seq_naive, robuust 40 stappen |
| N=2 expliciete deling, Python-orkestratie | 11,23 | proto_multi_seq_moe_shared |
| N=2 graph, private caches cap-24×2 — **bitexact PASS** | **36,86** | proto_multi_seq_graph_n2 (27,13 ms/token, 40 tokens) |
| N=2 graph, gedeelde cache cap-64, één stream — **bitexact PASS** | 33,52 | proto_multi_seq_graph_n2_shared (29,84 ms/token) |

NB: een eerdere N=2-graph-meting (23,59 tok/s "bitexact") is **ongeldig**
(staging-race → garbage==garbage-vergelijking + CuPy-pool-aliasing tussen de
graphs); zie het correctieblok bovenaan RESEARCH_NOTEBOOK.md 2026-08-16 voor
de drie bugfixes (staging-ring, CACHE_ATTRS-referenties, setup_graph
early-return). Verrassing die nog een meting verdient: shared < private
ondanks grotere cache (werkhypothese: LRU-thrash bij afwisselende
werkingssets; niet gemeten).

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
15/15 tokens, beide sequenties.** Resultaat bij 15 stappen (kale eager-
configuratie, geen graph/selectieve-ERVF/gebatchte-kernels, schoon voor een
N=1-vs-N=2-vergelijking): N=1 solo 29,798 tok/s vs N=2 naive (nog GEEN
expliciete deel-logica, alleen incidenteel warm-cache-hergebruik) 31,411
tok/s aggregate — 1,054× (+5,4%). **Robuustheidscontrole bij 40 stappen
(zelfde dag): krimpt naar het robuustere +2,05%** (31,656 tegen solo
31,020) — consistent met `diag_batch_warm_cache.py`'s eigen cold-vs-
steady-state-bevinding, geen tegenspraak, wel het cijfer om te citeren.
Bevestigt dat het mechanisme praktisch werkt; de expliciete unie-gevoede
MoE-deling (`proto_batch_moe_layer_combined.py`, al bitexact bewezen op
één laag) nog niet in deze staplus geïntegreerd — dat is de directe
vervolgstap. Zie `agents/RESEARCH_NOTEBOOK.md` 2026-08-16, blokken
"EERSTE ECHTE END-TO-END METING" en "Robuustheidscontrole van de
N=2-naive-baseline".

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
gat blijft). Geprofileerd per sectie: down_proj gather+masked+reduce was
**48,9%** van de resterende tijd. **Vervolg, zelfde dag: de al gebouwde
gebatchte V5/V6-kernels daadwerkelijk toegepast op de unie-dimensie, plus
numpy-vectorisatie van de resterende Python-lus — bitexact bij elke stap,
eindresultaat 11,12 tok/s (4,19× sneller dan de eerste werkende versie).**
Belangrijke fysieke les onderweg: masked/reduce/accumulate batchen hielp
sterk (reken-gebonden, launch-overhead was de kost); gather batchen hielp
nauwelijks (PCIe-bandbreedte-gebonden — zelfde bytes over de bus ongeacht
launch-aantal, zelfde klasse beperking als de eerder weerlegde
E2/NERVF-4-sporen). **Nog steeds 2,82× trager dan de naive baseline
(31,411)**, maar dat resterende gat is nu fysiek verklaard
(gather+up_proj-fetch ≈49%, beide bandbreedte-gebonden, dicht bij hun
vloer) — de volgende hefboom zou PCIe-overlap met rekenwerk zijn, geen
verdere kernel-batching. Bevestigt scherp waarom
`BATCH_ARCHITECTURE_DESIGN.md` een echte integratie als meerdere-weken
CUDA-engineeringwerk scoopte. Zie `agents/RESEARCH_NOTEBOOK.md`
2026-08-16, blok "Expliciete MoE-deling geïntegreerd in de echte staplus".

**N=4 naive baseline (zelfde dag)**: groeit het incidentele voordeel mee met
N, zoals losstaande diagnostiek suggereerde? **Verrassend: nee** — 31,215
tok/s aggregate tegen solo 29,820 (1,047×, +4,7%), vlak tot licht lager dan
N=2's +5,4%. Vermoedelijke oorzaak: vaste cache-capaciteit (72) geeft meer
onderlinge eviction bij groter N, wat de grotere theoretische overlap-kans
compenseert. **Vervolghypothese getoetst en VERWORPEN**: een met N
meeschalende cache (144 i.p.v. 72) herstelt dit niet — integendeel, **0,706×,
een echte regressie** (19,07 tok/s tegen solo 27,01). Eerst voorgestelde
verklaring (`cache_assign`'s lineaire eviction-scan wordt duurder bij
grotere cap) **direct getoetst met een geïsoleerde micro-benchmark en
WEERLEGD** — de kost per aanroep daalt licht met grotere cap, stijgt niet.
**De regressie zelf staat vast; de oorzaak is nog onbekend.**

**N=8 (zelfde dag): geen vlakke trend meer — een INSTORTING.** 7,521 tok/s
aggregate tegen solo 29,743 — **0,253×, 4× TRAGER dan één sequentie
alleen** (bitexact bevestigd). Samen met de twee andere cache-gerelateerde
regressies dezelfde dag (grotere capaciteit bij N=4: 0,706×; persistente
warme cache + unie-deling: 0,17×, 6,5× trager) tekent zich een coherent
beeld af: **een vaste of vergrote gedeelde cache wordt bij een bepaald
punt een knelpunt in plaats van een voordeel, en "gewoon N verhogen" is
géén pad naar hogere aggregate doorvoer** met het naive mechanisme. De
achterliggende oorzaak (waarom cache-gerelateerde wijzigingen zo groot en
consistent negatief uitpakken) is nog niet vastgesteld — sectieprofilering
van de warme-cache-regressie liet zien dat het effect **globaal** is (ook
zichtbaar in ongerelateerde Mamba-lagen), niet gelokaliseerd in de
cache-code zelf, en heeft de grens bereikt van wat sync-gebaseerde
profilering (zonder Nsight Compute/Systems) kan verklaren. Zie
`agents/RESEARCH_NOTEBOOK.md` 2026-08-16, blokken "N=4 naive baseline",
"Grotere cache bij groter N", "N=8 naive baseline" en alle vervolgen
direct erna.

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

**Status 2026-08-16: BEWEZEN IN PRODUCTIE via de pro_research-lijn.** De
machinerie (`setup_graph()`/`step_graph()`/`_step_body_graph()`, dp-kernels in
`gpu_kernels.py`) is gebouwd onder de bevroren preregistratie
`E1F22_GRAPH_CAPTURE_PREREGISTRATION_2026-08-15.md`, maar de treesweep200-eigen
gegate A/B is nooit gedraaid op het Nano-checkpoint — in plaats daarvan heeft
pro_research exact deze code bitexact in productie bewezen op het échte
Lightning-model (V4: 41,13 tok/s; V6-record: 47,41 tok/s;
`pro_research/results/PRO_V4_GRAPH_SELECTIVE.json`). De treesweep200-
vervolgstappen (eigen runner/verifier/rapport op Nano) zijn daarmee
**vervallen** — de machine draagt het bewijs al. Wat wél open staat: graph-
residency voor **multi-sequentie** (batch>1), zie
`pro_research/proto_multi_seq_graph_n2.py` en `PATH_TO_100_TOKS.md`.

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
