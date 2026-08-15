# Onderzoekslogboek

Eén blok per fase, **nieuwste bovenaan**. Schrijf hier ook wat er *niet* werkte
en waarom — dat is meestal het bruikbaarste deel. Formaat:

```
## <datum> — <fase> — <verdict in één zin>
**Vraag** · **Opzet** (armen, één variabele) · **Uitkomst** (getallen) ·
**Poorten** · **Wat dit sluit of opent** · **Artefacten**
```

---

## 2026-08-16 — PRO G2 — K-token epoch-graph: technisch gesloten, geen bug

**Vraag.** Kan de bestaande token-graph (`rt._graph`, een geïnstantieerde
`cudaGraphExec_t` via CuPy) K keer als child in één parent-graph gevangen
worden, zodat één host-launch K tokens vooruitbrengt?

**Uitkomst.** Nee, met de huidige aanpak. `pro_research/epoch_graph.py --mode
smoke` (ongewijzigd, al aanwezig, nooit eerder gedraaid) faalt voor k=2 én k=4
met `cudaErrorStreamCaptureUnsupported: operation not permitted when stream is
capturing`, bij `rt._graph.launch(stream)` binnen `stream.begin_capture()`.

**Waarom.** `cudaGraphLaunch()` — het aanroepen van een reeds
geïnstantieerde/uitvoerbare graph — is zelf geen capturable API-call. Om een
graph als child-node in een andere capture op te nemen moet je de
graph-**template** (`cudaGraph_t`, vóór instantiatie) doorgeven aan
`cudaGraphAddChildGraphNode()`, niet de uitvoerbare `cudaGraphExec_t` die
`.launch()` gebruikt. `runtime.py`'s `setup_graph()` bewaart alleen het
geïnstantieerde object (`s.end_capture()`); de template wordt niet apart
vastgehouden. Dit is een CUDA-API-beperking, geen CuPy-instelling of bug in
deze pack.

**Status: `technical_blocked`, poort `parent_graph_ids_exact` niet bereikt —
eerlijke sluiting per de eigen regel van de pack** ("Unsupported nested
capture is a valid technical closure"). Niet geforceerd, geen alternatief
mechanisme stilletjes gesubstitueerd.

**Wat dit open laat.** De K-token-amortisatie-hypothese zelf is niet weerlegd
— alleen déze implementatiestrategie. Een diepere vervolgstap (niet in deze
sessie gedaan) zou de graph-template apart moeten bewaren tijdens
`setup_graph()`'s eigen capture en `cudaGraphAddChildGraphNode` rechtstreeks
via CuPy's lage-niveau runtime-bindings aanroepen — een aparte,
substantiëlere CUDA-engineeringtaak, geen kleine reparatie.

**Artefacten.** `pro_research/results/PRO_G2_EPOCH_GRAPH.json` (status
`technical_blocked`).

---

## 2026-08-16 — PRO V4 — graph-safe + selectieve ERVF fysiek geïntegreerd: 41,13 tok/s, alle poorten groen

**Vraag.** V3-G0S (graph-residentie alleen, +10,1%) en V3-G1B (selectieve ERVF
alleen, +10,73%) zijn los gemeten en mogen niet worden opgeteld (Amdahl-
interactie onbekend). Wat levert één fysiek geïntegreerde arm echt op?

**Mechanisme.** `_step_body_graph()` roept `self._attention`/`self._mamba` aan,
die zelf via `self.k.mv_bf16`/`mv_fp8_tensor`/`mv_f32` dispatchen
(`runtime.py:401-474`). CUDA-graph-capture legt vast welke kernel op
capture-moment achter dat Python-attribuut zit. Dus: eerst
`selective_ervf_v3._install_selective(rt, dense)` draaien (herbindt die drie
attributen voor de vier bevroren winnende vormen), **dan pas** `rt.setup_graph()`
— de ERVF-kernels worden dan mee vastgelegd in de graph, terwijl K/V/router op
productiekernels blijven binnen dezelfde graph. Nieuwe runner:
`pro_research/graph_selective_v4.py`, preregistratie
`PRO_V4_PREREGISTRATION.md` (bevroren vóór meting).

**Opzet.** EGR (productie, device-cache, geen graph) vs GRAPH_SELECTIVE
(selectieve dispatch geïnstalleerd vóór capture) vs DET (twee rollouts) vs CTL
(`bad_pick=1`-sabotage herbinnen dezelfde graph, moet falen). Structurele
verificatie: `rt._graph.debug_dot_str()` moet `pro_gemv_bf16_ervf16` én
`pro_gemv_fp8_tensor_ervf16` bevatten — bewijst dat de ERVF-kernels echt in de
graph zitten, niet alleen tijdens de warmup zijn aangeroepen.

**Uitkomst (full, 3 prompts × 256 tokens, 765 getimede samples).**

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,1786 ms | 32,07 |
| **GRAPH_SELECTIVE** | **24,3152 ms** | **41,13** |

Winst: **6,8634 ms / 22,0%** — meer dan de grootste losse mechanisme-winst
(3,3841 ms) en dicht bij de naïeve som van beide losse smoke-winsten
(2,8931 + 3,3841 = 6,2772 ms), dus nagenoeg volledig additief met slechts een
kleine overlap-tax. GRAPH_SELECTIVE's eigen p50 (24,3152) ligt ook onder béíde
eerder apart gemeten mechanismen (28,6063 en 28,158 ms) — dat is het eerste
directe bewijs dat de twee winsten fysiek samen bestaan zonder elkaar op te
eten.

**Poorten.** `argmax_direct_tie` ✅ · `graph_dot_contains_ervf` ✅ (beide
kernelnamen aangetroffen) · `graph_selective_equals_egr` ✅ (bitexact op alle
drie prompts, 256 tokens) · `graph_selective_deterministic` ✅ ·
`bad_pick_control_diverges` ✅ (2 van 3 prompts wijken af zoals vereist — de
controle heeft dus onderscheidend vermogen) · `extra_vram_lt_64MiB` ✅ (4 MiB)
· `full_speed_gain_ge_2_5ms` ✅ · `full_samples_ge_500` ✅ (765).

**Wat dit betekent voor het 100 tok/s-doel.** 41,13 tok/s is 24,9% van het
ctx0-roofline-plafond (165 tok/s, hardware-eigenschap, modelonafhankelijk) —
op van ~17% (Nano-lijn) naar ~25%. Nog altijd een factor 2,4× te gaan tot 100.
Dit is wél het eerste fysiek geïntegreerde, onafhankelijk gepoorte resultaat op
het **juiste** doelmodel, en het bevestigt dat losse mechanismewinsten hier
grotendeels blijven optellen in plaats van elkaar te kannibaliseren — een
gunstig signaal voor het toevoegen van een derde/vierde mechanisme (K-token
epoch-graph, MTP) op dezelfde manier.

**Artefacten.** `pro_research/PRO_V4_PREREGISTRATION.md` ·
`pro_research/graph_selective_v4.py` ·
`pro_research/results/PRO_V4_GRAPH_SELECTIVE.json`.

---

## 2026-08-16 — PRO V3 anchor-onderzoek — verklaard: twee verschillende modellen, geen bug

**Vraag.** De gebruiker vroeg expliciet: V3-G0S/G1B (`pro_research`) draaien
bit-identiek EGR vs GRAPH_SAFE/SELECTIVE, maar beide wijken al bij token 1 af
van het bevroren `V36_DETERMINISTIC_ANCHOR.json`. Komt dat door
`nemotron_3_5_lightning_v35` versus het oudere ankerpad, of door een
runtime/model-identiteitswijziging?

**Antwoord: het ankerpad.** Twee fysiek verschillende checkpoints:

| | `models/nemotron_3_5_lightning` (het ankerpad) | `models/nemotron_3_5_lightning_v35` (pro_research default) |
|---|---|---|
| werkelijke identiteit | **NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4**, verkeerd gedownload en misleidend hernoemd | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, sha `6dbbd757…` |
| `max_position_embeddings` | 262 144 (Nano) | 1 048 576 (Lightning) |
| quantisatie | — | MIXED_PRECISION: experts+lm_head NVFP4, Mamba in/out FP8-per-tensor, attentie BF16 |
| bron | — | modelopt 0.44.0rc5, drieweg `quant_kind()` |

Bewijsketen: `reports/lightningstream_nemotron/N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md`
("De hele lijn draait op NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4") →
`N2R_V35_LAYOUT_REPORT_2026-08-14.md` (adjudicatie van de échte Lightning-download,
`models/nemotron_3_5_lightning_v35`) → `HANDOVER_TO_KIMI_2026-08-15.md`
("Model staat nu in `models/nemotron_3_5_lightning_v35`. Zet
`LS_MODEL_DIR=nemotron_3_5_lightning_v35` om ermee te draaien."). Bevestigd
lokaal: `config.json` van het ankerpad heeft `max_position_embeddings: 262144`
(Nano-plafond), het `_v35`-pad heeft `1048576` (Lightning-plafond).
`A1_ADOPTION_PRECONDITION.json.environment.model_dir` = `"nemotron_3_5_lightning"`
— A1/D1/E1-E6/NERVF-0..5 en `V36_DETERMINISTIC_ANCHOR.json` zijn dus **allemaal
gemeten op Nano**, ondanks dat de correctie al op 2026-08-14 bekend was. De
`scripts/treesweep200/*.py`-scripts defaulten nog steeds naar
`nemotron_3_5_lightning` (geen `LS_MODEL_DIR` override in die lijn) — de
"herhaal de meetketen op het juiste model"-stap uit N0R_CORRECTION is voor de
closed-namespace-lijn **nooit uitgevoerd**.

**Gevolg — geen bug, wel een scope-correctie.**
- V3's eigen interne vergelijkingen (EGR vs GRAPH_SAFE, BASE_A/SELECTIVE/BASE_B)
  zijn methodologisch geldig: alle armen laden hetzelfde model in hetzelfde
  proces. De 28,61 ms / 28,16 ms resultaten staan.
- De externe ankervergelijking in V3 is terecht `informative`, niet gating —
  precies zoals de V3-preregistratie het al voorzag.
- **Alles in dit bestand vóór 2026-08-16 (ERVF 1,936×, D1, A1, E1 fase 2.1,
  "37,49 ms/token", "26,7–29,5 tok/s") is gemeten op Nemotron-3-Nano, niet op
  Nemotron-3.5-Lightning.** `HANDOVER_TO_KIMI_2026-08-15.md` mat vóór A1/D1 al
  een kale Lightning-baseline (27,743 tok/s @ctx0, vóór ERVF/D1/A1) die al in de
  buurt zit van Nano's volledig-geoptimaliseerde 29,5 tok/s — Lightning's
  kleinere `lm_head` (NVFP4 i.p.v. BF16, 704→198 MB) en FP8-Mamba bewegen
  minder bytes per token. De kernelwinsten (ERVF, v4-attentie, D1) zijn
  architectuur-onafhankelijk aannemelijk overdraagbaar (N2R: "shape-identiek
  op 8 byte na"), en V3's eigen bitexacte pariteit op het echte Lightning-model
  bevestigt dat ze *fysiek* overdragen — maar de closed-namespace tok/s-tabel
  in `STATE_OF_THE_WORK.md` beschrijft strikt genomen Nano, niet het
  opdrachtdoel.

**Wat dit niet doet.** Geen enkele eerder gemeten winst wordt ingetrokken —
D1/A1/E1F21's eigen interne A/B's zijn ook allemaal single-model, dus intern
geldig. Het is puur een naamgevings-/scopefout die al één keer eerder werd
gedocumenteerd (N0R_CORRECTION) maar niet is doorgezet naar de
adoptiemetingen erna.

**Aanbeveling voor de volgende fase.** `pro_research` blijft op
`nemotron_3_5_lightning_v35` (correct) draaien — geen wijziging nodig. Wie ooit
weer in de closed `treesweep200`-lijn werkt, moet `LS_MODEL_DIR=nemotron_3_5_lightning_v35`
zetten, of nog beter: het ankerpad hernoemen zodat de mismatch niet blijft
terugkomen (`models/nemotron_3_5_lightning` → `models/nemotron_3_nano`, zodat de
naam niet meer liegt). Niet in deze sessie gedaan omdat schrijfrechten buiten
`agents/`+`pro_research/` niet zijn opgeëist.

**Artefacten.** `pro_research/PRO_V4_PREREGISTRATION.md` (model-identiteitsnoot
erin opgenomen) · bronnen: `N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md` ·
`N2R_V35_LAYOUT_REPORT_2026-08-14.md` · `HANDOVER_TO_KIMI_2026-08-15.md`.

---

## 2026-08-15 — E1 fase 2.2 — graph-replay GEBOUWD maar ongemeten (sessie gestopt op quota)

**Vraag.** Kan de hele token — embedding t/m argmax — als één CUDA-graph
replays worden nu het MoE-pad sync-vrij is?

**Status.** Preregistratie bevroren (`E1F22_GRAPH_CAPTURE_PREREGISTRATION`),
graph-API gesmoketest (capture → launch → correct over 5 replays), alle
kernels en runtime-methoden geschreven en op syntax gecontroleerd — maar de
A/B is **niet gedraaid**. Niets hieronder is een meting.

**Ontwerpkeuzes die de volgende agent niet opnieuw hoeft te bedenken.**
- `attn_decode_warp_fp8_gqa4_dp`: vaste grid (2,256); elke block schrijft áltijd
  zijn 4 partials — dode splits schrijven neutraal (m=-inf, l=0) — dus nooit
  stale data, en `attn_decode_combine` (vaste 1024 slots) slaat l≤0 al over.
  Zelfde optelvolgorde als eager → bitexact te verwachten, te bewijzen door
  de verifier.
- Embedding: tabel wordt bij `setup_graph()` naar pinned+mapped gekopieerd
  (+0,656 GiB host-RAM); `embed_gather_bf16` leest hem in-graph via tok_dev.
- Token-flow: argmax schrijft tok_dev aan het einde van replay N; embed_gather
  leest het aan het begin van replay N+1. Prompt-tokens staged de host met een
  stream-geordende 4-byte H2D (geen sync). Ids oogsten via pinned ringbuffer
  (`ring_harvest`); gegenereerde ids staan vanaf ring-index P-1.
- Kill-criteria staan in de prereg: K1 = event-fork weigert → single-stream
  fallback als aparte arm.

**Artefacten.** Prereg + code in `gpu_kernels.py`/`runtime.py` (zoek
"E1 fase 2.2"). Nog te schrijven: runner, verifier, rapport.

---

## 2026-08-15 — E1 fase 2.1 — device-resident routing werkt: −4,54 ms/token eager, alle poorten groen

**Vraag.** Kan de MoE-laag zonder één device→host-sync draaien (routerkop,
LRU, miss-staging als kernels), zodat graph-capture (fase 2.2) überhaupt kan —
en wat levert alleen dat al op?

**Opzet.** Eén variabele: `device_cache` aan/uit op de geadopteerde stack.
BASE = default, DEV = device routing+LRU (cap 72), INV = cap 56 moet dezelfde
tokens geven, CTL = `bad_pick`-sabotage moet falen. Pariteit tegen het
bevroren A1/V36-anker, 2 prompts × 64 tokens, contexts_max=4096. De
staging-kernel volgt het M1-patroon uit de microbench (bulk-read uit pinned
host = 24,93 GB/s, 96% van DMA), NIET de M2-variant (GEMV leest zelf van host:
7,27 GB/s — dood).

**Uitkomst.** p50 41,540 → **36,998 ms/token (−4,542 ms, −10,9%)**, pariteit
behouden. Dat is 51% van het 8,925-ms-budget uit fase 1; de rest is pure
launch-overhead voor fase 2.2. Verifier 14/14, inclusief bitexacte spiegels
(indirecte GEMV == directe ERVF; accumulate_ind == accumulate_into) en een
exacte Python-LRU-spiegel van `cache_assign`.

**Bug gevonden.** `enable_cache` resette `_dev_cache` niet → INV-arm draaide
cap-56-semantiek over vuile LRU-staat en faalde terecht. Fix: `_dev_cache = {}`
in `enable_cache`; INV+CTL opnieuw gedraaid met schone staat, beide groen.
Ook de verifier had zelf zo'n staat-desync (verse cache-buffers tegen oude
slot-tabellen) — zelfde les: *cache-inhoud en slot-staat zijn één invariant*.

**Poorten.** C1 ✅ · INV ✅ · CTL ✅ (schone attributie) · S1 ✅ (−4,542 ≥ 1,5)
· V1 ✅ (113 KiB analytisch).

**Wat dit opent.** Fase 2.2 (graph-capture van de hele token) heeft nu een
sync-vrij MoE-pad. Resterend capture-werk: embedding-gather, argmax,
pos-afhankelijke kernels (kv_write_fp8, attentie-splits) op device-pos.

**Artefacten.** `E1F21_DEVICE_ROUTING_AB.json` · `E1F21_INV_CTL_RERUN.json` ·
`e1f21_independent_verification.json` · rapport
`E1F21_DEVICE_ROUTING_REPORT_2026-08-15.md`.

---

## 2026-08-15 — A1 — de bewezen stack staat nu default aan, en het anker is opnieuw bevroren

**Vraag.** E6 mat dat ERVF + v4 + D1 sneller én exact is. Mag dat de default worden?

**Waarom niet meteen.** E6 vergeleek twee armen binnen één proces met dezelfde
cachegeschiedenis — precies het regime waarin de exactheid vóór D1 óók léék te
kloppen. Vier fasen (NERVF-3, NERVF-4, E4, S11) haalden hun pariteitspoort over
2×64 tokens terwijl de runtime niet deterministisch was. Een adoptie mag niet op
een blinde test rusten.

**Opzet.** Verander de **cachecapaciteit** (72 vs 56). Dat verandert het
hit/miss-patroon op vrijwel elke laag van elke token, dus de optelvolgorde, zonder
een gewicht/route/kernel aan te raken. Plus een **controle-arm die moet falen**:
dezelfde vergelijking zonder D1.

**Uitkomst.** Met D1: identiek over 2 × 256 tokens. Zonder D1: divergeert
(expository, token 224; narrative niet). De test heeft dus vermogen — maar smal,
en dat verklaart waarom de fout vier fasen lang onzichtbaar bleef.

**Poorten.** G-A1-CAP ✅ · G-A1-CTL ✅ (faalde zoals vereist) · G-A1B-DEFAULT ✅
(runtime zonder enige vlag reproduceert bit-identiek wat A1 mat) · G-A1B-FLAGS ✅.

**Wat dit opent.** Defaults om: `use_ervf=True`, `rt.attn=attention_fp8_gqa4`,
`deterministic_accum=True`. De attentiekernel wordt nu via `rt.attn` gekozen;
oude scripts die `rt.k.attention_fp8_gqa` overschrijven meten voortaan een
nul-verschil in plaats van stil iets verkeerds — bewust zo gekozen.

**Het anker.** V35 wordt niet meer gereproduceerd, divergentie al bij token 1.
Dat komt van D1, niet van v4 (E4 reproduceerde V35 wél). Het anker legde een
ordeafhankelijk artefact vast. Vooraf vastgelegde regel gevolgd: `V36_DETERMINISTIC_ANCHOR.json`
bevroren, V35 ongewijzigd bewaard, niet-vergelijkbaarheid opgeschreven.

**Artefacten.** `A1_ADOPTION_PREREGISTRATION_2026-08-15.md` ·
`A1_ADOPTION_REPORT_2026-08-15.md` · `A1_ADOPTION_PRECONDITION.json` ·
`A1B_ADOPTION_VERIFY.json` · `V36_DETERMINISTIC_ANCHOR.json`

---

## 2026-08-15 — E6 — geïntegreerd 41,98 → 37,49 ms per token, exact

**Opzet.** Drie armen `base_a / integrated / base_b`, 3 domeinen × 512 causale
tokens, D1 in **alle** armen (zonder D1 zijn twee armen niet eens vergelijkbaar).
Eén variabele: ERVF + v4-attention.

**Uitkomst.** +4,169 (expository) / +4,443 (narrative) / +4,858 (code) ms, elk
boven zijn eigen drift. Bit-identieke uitvoer. VRAM ongewijzigd.

**Poorten.** Exactheid ✅ · latency ✅ · VRAM ✅ · eindpoort ≥50 tok/s ❌ (26,7).

**Wat dit sluit.** Niets — maar het laat zien dat de resterende afstand tot 50
tok/s niet uit de gebouwde componenten kan komen. E2 is weerlegd en E1 fase 2 is
ongebouwd; dat zijn de twee posten die het plan ervoor had ingeboekt.

**Artefacten.** `E6_INTEGRATED_REPORT_2026-08-15.md` · `E6_INTEGRATED_RUN.json`

---

## 2026-08-15 — D1 — de runtime was niet deterministisch, en dat is nu opgelost

**Vondst.** `_moe_cached` accumuleert de zes routed experts in
**hit-dan-miss-volgorde**. Welke expert een hit is hangt van de LRU-staat af, dus
twee runs met andere cachegeschiedenis tellen in andere volgorde op. FP-optelling
is niet associatief → twee armen met **identieke configuratie** divergeerden over
512 tokens. Ontdekt doordat NERVF-5 `base_b` tegen `base_a` zette.

**Ingreep.** Rekenvolgorde en optelvolgorde gescheiden: rekenen blijft hit-eerst
(latencywinst blijft), bijdragen naar aparte slotbuffers, na de lus optellen in
routevolgorde `s = 0..5`. Kosten: `top_k × hidden` floats (64 KB), nul extra
kernels.

**Uitkomst.** `base_b` nu identiek aan `base_a` over 3 × 512 tokens; NERVF-5
slaagt alsnog op alle drie zijn poorten. ERVF-winst met D1 aan: +2,771 / +5,008 /
+5,395 ms.

**De les.** Vier eerdere exactheidspoorten waren waar voor hun eigen run maar
bewezen minder dan ze leken. Vandaar werkregel 8: bouw een controle-arm die moet
falen.

**Artefacten.** `D1_DETERMINISM_REPORT_2026-08-15.md` · `d1_determinism.json`

---

## 2026-08-15 — NERVF-0 t/m 5 — ERVF gerepliceerd op een tweede model

**Vraag.** Reproduceert de Qwen3-30B ERVF-doorbraak op een architecturaal andere
NVFP4 hybrid-Mamba MoE?

**Antwoord: ja.** 1,936× bitexact op het projectievlak, bij **dezelfde** gekozen
subwarp-breedte 16 die Qwen selecteerde, op een ander model, een andere
quantisatie (NVFP4 vs Q5/Q8) en een andere shape.

**Het mechanisme.** w-lane subwarps per rij, 256/w rijen per 256-thread block,
per lane aparte virtuele accumulatoren, en een reductie die de DAG van de
referentiekernel **exact** reconstrueert — de eerste butterfly-stap (offset 16)
wordt een lane-lokale optelling, offsets 8/4/2/1 blijven shuffles binnen de
subwarp, en de acht warp-sommen vouwen in registers in exact de volgorde
`((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

**Fout onderweg.** w=4 en w=8 gaven eerst 72/72 mismatches doordat ik de virtuele
accumulatoren *sequentieel* vouwde in plaats van in butterfly-volgorde. Na de fix
alle vier breedtes bitexact.

**NERVF-1 valstrik.** RAW_SCAN sprong van 9,77 naar 51,67 µs tussen runs: het
2,81 MiB record past in L2. Opgelost door alle armen door een 254 MiB pool van 95
replica's te cyclen → spreiding 0,1%. Elke bandbreedtemeting hierna doet dit.

**NERVF-4.** Weerlegd, zie E2.

**Poorten.** Doorbraakladder LEVEL 2 gehaald (≥1,35× exact). LEVEL 3 niet: het
volledige expertpad bevat de down-projectie, die ERVF niet raakt en waar NERVF-4
de voor de hand liggende route sloot. LEVEL 4 (≥35 tok/s) niet: 29,45.

**Niet geclaimd.** Geen nieuwheidsclaim — geen prior-art-audit, geen stock
llama.cpp-vergelijking, geen tweede GPU.

**Artefacten.** `NERVF_NEMOTRON_FINAL_REPORT.md` · verifier 66/66 in
`nervf_independent_verification.json`
