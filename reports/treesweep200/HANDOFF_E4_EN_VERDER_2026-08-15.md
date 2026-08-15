> **VERVANGEN 2026-08-15.** De actieve takenlijst staat nu in
> agents/TODO.md, met agents/STATE_OF_THE_WORK.md als startpunt en
> agents/RESEARCH_NOTEBOOK.md als logboek. Dit bestand blijft staan als
> historisch document -- vink hier niets meer af.

# HANDOFF — TreeSweep-200 / exact-efficiency track — stand 2026-08-15 (na E4-kernelwerk)

Voor de volgende agent. Bevat: missie, harde regels, wat klaar is, wat er
netjes half klaar ligt (E4), en het volledige restprogramma met startpunten.

## 0. Missie

50 tok/s (mijlpalen 75/100) bij lange context (tot 262100) op één 8 GiB
laptop-GPU, Nemotron **3.5 Lightning** 30B-A3B NVFP4, **exact** (geen
speculatie/pruning/low-rank — TreeSweep-as is gefalsifieerd door Z1:
tree-verifier kost lineair 16,5 ms/positie, plafond 45–61 tok/s; 200 tok/s is
dood, pack zelf zegt "door naar 50–100, geen drafter trainen").

Huidige productie-baseline (onveranderd deze sessie): **27,7 / 26,2 / 21,7 /
18,4 tok/s** bij ctx 0 / 32K / 131K / 262100. Streaming-roofline 330–338 GB/s.
Bytevloeren: 6,31 ms (ctx0) / 8,74 ms (262K) → plafonds 158,6 / 114,4 tok/s.
50 tok/s vraagt 31,6% (ctx0) resp. 42,8% (262K) van de roofline.

## 1. Harde werkeisen (niet onderhandelbaar)

- Schrijf ALLEEN in `reports|scripts|src|tests/treesweep200/` (pack-namespace),
  plus eigen-lijn `reports|scripts|src|tests/lightningstream_nemotron/` en
  `docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md`. ALLES daarbuiten read-only
  (andere agent = 80B/streamq5).
- Na elke fase:
  `.venv-nemotron/Scripts/python.exe scripts/lightningstream_nemotron/protected_manifest.py verify --baseline reports/lightningstream_nemotron/PROTECTED_80B_MANIFEST_BEFORE.json --out reports/lightningstream_nemotron/protected_verification_after_<fase>.json --label <fase>`
  Eis: 0 modified / 0 removed ("root digest ok: False" en honderden "added"
  zijn normaal — de treesweep200-namespace is niet in de allowlist maar
  "added" telt niet als overtreding).
- Vóór elke GPU-run: `nvidia-smi --query-compute-apps` (leeg = vrij). Nooit
  processen killen.
- Protocol per fase: preregistratie mét bevroren poorten → runner → aparte
  verifier (importeert de runner NOOIT; de runtime-library wél) → rapport (NL)
  met claim boundary → registry-update + `tools/validate_registry.py` →
  log-append. Poorten nooit verruimen na resultaten. Eén variabele per meting.
  Componentmetingen NOOIT opwaarderen naar tok/s.
- Interpreter overal: `.venv-nemotron/Scripts/python.exe`. Model:
  `LS_MODEL_DIR=nemotron_3_5_lightning_v35` voor de v35-runtime.

## 2. Klaar (onafhankelijk geverifieerd)

- **P0/E0** (identiteit + roofline-reproductie): alle poorten PASS, 8/8 claims
  gereproduceerd. Bestanden: `reports/treesweep200/P0_E0_PREREGISTRATION_*`,
  `P0_E0_REPORT_*`, `P0_IDENTITY_MANIFEST.json`,
  `E0_N1_N5_EVIDENCE_MANIFEST.json`, `E0_ROOFLINE_REPRODUCTION.json`;
  runners `scripts/treesweep200/p0e0_identity_roofline.py`,
  `e0_independent_verify.py`. Registry: P0/E0 = `gate_passed`, valideert OK.
- Modelconfig (gemeten): 52 lagen, hybride patroon met **6 attention-lagen**,
  n_heads=32, n_kv=2, head_dim=128, fp8-e4m3 KV, 128 routed experts, top-6,
  moe_inter 1856, hidden 2688.

## 3. E4 (attention-roofline, agent 21) — status: KERNELFASE KLAAR, IN-LUS OPEN

Rapport: `reports/treesweep200/E4_ATTENTION_ROOFLINE_REPORT_2026-08-15.md` —
lees dat eerst. Kern:

- Poorten F1/P1/C1 PASS; **S1 (≥100 GB/s) en S2 (≥169 GB/s) FAIL** — exact-fp32
  vloer geschat ~1,2–1,5 ms/laag @262K; S2 structureel onhaalbaar zonder
  fp16/tensor-cores (verworpen wegens exactheid).
- **Beste kandidaat v4** (`attn_decode_warp_fp8_gqa4` in
  `src/moe_lab/lightningstream_nemotron/gpu_kernels.py`, wrapper
  `attention_fp8_gqa4`): hardware-cvt-decode + double-buffered loads +
  2-posities-ILP. **Bitwise identiek aan v1** op alle 5 contexten × 3 seeds.
  2,304 vs 2,803 ms/laag @262144 (−17,8%), −33% @4K.
- Registry E4 staat nu op `dependency_blocked` → zet op `gate_failed`
  (poorten S1/S2) en noteer v4 als bevroren beste exacte kandidaat.
- **Te doen, in volgorde:**
  1. In-lus adoptiemeting (G-E4-T1): schrijf
     `scripts/treesweep200/e4_inloop.py` op het s14-patroon
     (`scripts/lightningstream_nemotron/s14_moe_layer_timeline.py`): subclass
     `LightningRuntime`, monkeypatch `rt.k.attention_fp8_gqa =
     rt.k.attention_fp8_gqa4` (signatures identiek), meet de
     attention-component @262100 met CUDA-events, en eis 64-token-pariteit met
     het v1-anker (`reports/lightningstream_nemotron/s5_baseline_generation.json`).
     Contextvulling: zie `n7b_cached_decode.py` (rt.pos = target-truc).
  2. Onafhankelijke verifier `e4_independent_verify.py` (rekent JSON na zonder
     de runner te importeren; mag `gpu_kernels` wél importeren voor bitwise-
     herbevestiging v4 vs v1).
  3. Registry + log + rapport aanvullen; protected-manifest opnieuw.

### Kernel-inventaris (allemaal geregistreerd in gpu_kernels.py)

v1 (productie), v2 (gerepareerd OOB-bug, trager), v3 (cvt+prefetch, bitwise
=v1, −10,5%), v4 (beste), v6 (f32x2+q-regs, trager), v7 (f32x2+shared-q,
~v4). Niet gebouwd (gedocumenteerd in rapport): transpose-reduce (31 i.p.v.
80 shuffles, geschat ~75 GB/s — onvoldoende voor S1).

### Valkuilen (hard geleerd)

- `pack/tools/validate_registry.py` draaien met `.venv-nemotron` python.
- fp8 random data: remap e4m3-NaN-patronen `(b & 0x7F)==0x7F → b &= 0xFE`,
  anders rel_l2 = NaN.
- Kernel-probes: geen float64-arrays aan float*-kernels voeren (gaf valse
  "f32x2 niet bitwise" — later weerlegd: f32x2 IS bitwise = fmaf).
- `redux.sync.add.f32` compileert niet op sm_120.
- In `p0e0_identity_roofline.py`: `is_bank()` moet `startswith("backbone.")`
  checken (MTP-experts delen substring `.mixer.experts.`).
- CuPy RawModule: geen CUDA-headers beschikbaar (CUDA_PATH ontbreekt) — pure
  inline-PTX-aanpak gebruiken zoals `e4m3x4_f32` in gpu_kernels.py.

## 4. Restprogramma (volgorde besloten, niet heropenen zonder nieuwe data)

### E2 — gatherloze downflow (agent 20) — GROOTSTE open hefboom
N2-gat: `gather_down_sparse` = 8,19 ms/token in-lus (4,3 GB/s vs 25,05
geïsoleerd); scan+gather = 67,2% van het down-pad. Pack-idee: ReLU²-output +
bitmask → index-carrying down-GEMV. Poorten: ≥80% gathertijd weg, ≥1,8×
down-pad, identieke output. Kandidaat: warp-ballot-iterator in een
`down_masked_into`-variant in `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py`.
Lees eerst agent 20 + de N2-evidence in E0_N1_N5_EVIDENCE_MANIFEST.json.

### E5 — GEMV-roofline (agent 22)
N5/Y2: kritieke GEMV 81,4 GB/s (4,2× onder roofline); bytehalvering bespaart
slechts 34,2% (31,6% vaste kost). Poort: ≥140 GB/s gewogen suite, geen shape
>5% trager.

### E1 — graph-resident token (agent 19) — lastigste
N1: CUDA-graph bovengrens 23,7% van een token (eager 36,714 → graph 28,023 ms,
bevroren routes). V1 heeft de host-read-variant gesloten (6,7 GB/s,
1,42× te duur); device-gestuurde expert-indirectie is open. Eerste poort:
N1-oracle binnen 5% reproduceren.

### E6 — integratie + E50 (agent 23)
≤20 ms/token @ctx0 (E50), 10.000 tokens, thermisch uur. Combineert
E1/E2/E4(v4)/E5. Pas starten als E2/E5/E1-resultaten er liggen.

### Mijn eigen S14/Y1-gaten (lightningstream-lijn, ook open)
MoE-lagen 27,7 ms stream-wand/token: up 6,55 + down_masked 8,39 + accum 1,0 +
route 3,5 (launch-gebonden) + shared 3,6 + **host_gap 4,7 ms GPU-idle**;
route-readback-sync = 6,66 ms/token @262K (Y1-oracle). Overlap/readback-
eliminatie is de grootste niet-kernel-hefboom maar raakt de host/device-
scheiding — behoort bij E1/E6.

## 5. Verwachte opbrengst als alles lukt (schatting, geen claim)

@262K: v4-attention −3,0 ms; E2-poort −6,6 ms (down); E5-poort ~−2 ms; E1
(graph-resident) −8,7 ms (23,7% van 36,7) — samen ~−20 ms van 54,3 ms →
~34 ms ≈ 29 tok/s. 50 tok/s @262K vereist bovendien host_gap+readback (~11 ms)
én Mamba/LM-head-winst; @ctx0 is 50 tok/s (20 ms) aannemelijker: 36,7 ms −
E1(8,7) − E2(6,6×~) − E5(~2) ≈ 19–20 ms. Eerlijk beeld: 50 @ctx0 kansrijk,
50 @262K vereist meer dan het huidige pack.

## 6. TODO List (volledig)

1. [done] Pack ingelezen; treesweep200-namespace; P0/E0 (rapport + verifier)
2. [done] E4 kernelfase: prereg, 5 kandidaten, v4 bitwise-exact −17,8%,
   S1/S2 FAIL gedocumenteerd
3. [DONE 2026-08-15, Claude] E4 in-lus adoptie (G-E4-T1) + onafhankelijke
   verifier (52/52) + registry-subblok `inloop_phase` + log-append.
   Uitkomst: attention 18,211 → 14,554 ms @262100 (−3,658, drift 0,544), token
   55,915 → 51,164 (−4,751, drift 3,014); alle diepten conclusief. G-E4-T1
   FAILED op zijn absolute drempel (14,554 > 6,0 ms) — niet verruimd. Arm-tegen-
   arm pariteit PASS (2×64 bit-identiek in v1/v4/v1); bitwise herbevestigd los
   van de runner (3/3). De S5-ankerclausule faalde op een **verouderd artefact**:
   dat anker is van 2026-08-14T20:02Z, vóór de v35-wissel om 20:52Z, dus van
   Nemotron 3 Nano. Nieuw anker bevroren: `V35_GENERATION_ANCHOR.json` (2×64).
   Rapport: `E4_INLOOP_REPORT_2026-08-15.md`. Adoptie van v4 als default bewust
   NIET omgezet — hoort bij E6.
4. [WEERLEGD 2026-08-15] E2 gatherloze downflow. **Hetzelfde experiment als
   NERVF-4**, één keer gemeten onder de strengere van de twee opzetten (3 armen,
   3 contextdiepten tot 262100, exact tegen het anker). Mechanisme werkt en is
   **exact**, maar is conclusief **slechter**: −5,70 / −7,56 / −7,38 ms per token;
   MoE-blok 0,72 tot 0,79×. Oorzaak: strided host-reads halen 6,7 GB/s over PCIe
   tegen 85,9 vanaf device (V1), dus de gather van 8,19 ms verdient zichzelf
   terug. Niet de grootste hefboom maar een gesloten deur.
   Rapport: `E2_GATHERLESS_DOWNFLOW_REPORT_2026-08-15.md`.
5. [DONE 2026-08-15] E5 GEMV-roofline — opgelost met **ERVF** i.p.v. een nieuwe
   kernel. Alle vier de shapes bitexact en sneller: routed_up 1,621x ·
   shared_up 1,784x · shared_down 1,646x · lm_head 1,676x; **gewogen 77,1 ->
   127,9 GB/s (1,660x)**. `weighted_suite_ge_140` **FAIL** (127,9, niet
   verruimd); `no_critical_shape_regression_gt_5pct` **PASS** (slechtste
   1,621x, elke shape verbetert); `integrated_token_improvement_ge_8pct`
   **PASS** (9,8% ctx0 / 8,1% 262100 uit NERVF-3).
   Rapport: `E5_GEMV_ROOFLINE_REPORT_2026-08-15.md`. (was: agent 22)
6. [FASE 1 DONE 2026-08-15] E1 graph-resident token. **G-E1-R1 gehaald**:
   N1-oracle gereproduceerd op 22,2% tegen 23,7% (afwijking 1,5pp, tolerantie
   5pp). Nieuw: **met ERVF aan stijgt het naar 27,0%** (8,925 ms) omdat de
   uitgifte-overhead per-launch is en niet meebeweegt met snellere kernels —
   ERVF en graph-residentie zijn complementair en ERVF **verhoogt** de
   opbrengst van E1. Gecombineerd plafond op ctx 64: 24,112 ms (~41,5 tok/s).
   Fase 2 (echte graph-resident token) blijft OPEN: vereist device-side
   routing, en V1 sloot de host-read-variant (6,7 vs 85,9 GB/s). Budget voor
   dat ontwerp: **8,9 ms/token**.
   Rapport: `E1_GRAPH_ORACLE_REPORT_2026-08-15.md`. (was: agent 19)
7. [DONE 2026-08-15, deelintegratie] E6. Fysieke A/B van wat gebouwd is —
   **ERVF + v4-attention + D1** — over 3 x 512 tokens: **41.980 -> 37.490 ms**
   gemiddeld, winst +4.169 / +4.443 / +4.858 ms per domein, alle conclusief,
   **bit-identieke uitvoer**, VRAM ongewijzigd. Eindpoort >=50 tok/s kort
   **NIET gehaald** (26.7 tok/s in dit regime): E2 is weerlegd en E1 fase 2 is
   ongebouwd (budget 8,9 ms/token).
   Rapport: `E6_INTEGRATED_REPORT_2026-08-15.md`. Resterend werk, daarna pas
   `agents/24_OPTIMIZED_TREE_RERUN.md` overwegen
8. [DOORLOPEND] Na elke fase: protected-manifest verify (0/0), registry-
   validatie, log-append, tussentijds rapport aan gebruiker.
   Laatst gedaan na E4-in-lus en na NERVF-1: 0 modified / 0 removed, registry
   valideert (30 experimenten).

9. [INGELAST 2026-08-15, Claude, op verzoek gebruiker] **NERVF — ERVF-replicatie
   op Nemotron** (nieuwe namespace `NERVF_NEMOTRON`, tweede-modelreplicatie van
   de bewezen Qwen-P7-doorbraak).
   - [DONE] Archiefinspectie: ERVF bestaat NIET in de Nemotron-runtime;
     `gemv_nvfp4_rows` is structureel exact de Qwen-vorm van vóór ERVF
     (1 block van 256 threads per rij, shared-memory reductie, `__syncthreads`).
     P7 staat in streamq5 en blijft read-only.
   - [DONE] NERVF-0 baseline-lock (`nervf0_baseline_lock.json`).
   - [DONE] NERVF-1 geometrie-audit. L2-defect uit de eerste ronde verholpen:
     alle armen cyclen nu door een pool van 95 replicas (254 MiB) boven de
     gemeten L2 van 32 MiB, waardoor `RAW_SCAN` van 5,3× spreiding naar 0,1%
     gaat. **Beide poorten open**: bandbreedte-efficiëntie 0,322 ≤ 0,40 en
     reductie+sync **46,1%** ≥ 25%. Armen: RAW 12,43 µs / 225,8 GB/s ·
     ROW_PATTERN 17,08 · DECODE_SCALE 20,80 · FULL_GEMV 38,58 / 72,7 GB/s.
     De instorting zit niet in geheugen of decode maar voor 46% in de
     reductie/sync — exact de Qwen-signatuur.
   - [DONE] NERVF-2 ERVF-microkernel. **Alle vier de breedtes bit-identiek**
     (0/72 mismatches elk, over 3 lagen × 3 experts × 4 activatieregimes × 2
     ReLU²-standen). Speedups: w=4 1,336× · w=8 1,686× · **w=16 1,936×** ·
     w=32 1,897×. **Primaire poort (1,35×) én sterke poort (1,75×) gehaald**,
     moonshot (2,0×) net niet. Gekozen breedte **16 — dezelfde als Qwen P7**.
     Onderweg gecorrigeerd: de lane-lokale stappen moeten in **butterfly**-
     volgorde, niet sequentieel; bij w=16 viel dat toevallig samen, bij w=4/w=8
     gaf het eerst 72/72 mismatches. Verifier 46/46.
     Rapport: `reports/nervf_nemotron/NERVF_1_2_REPORT_2026-08-15.md`.
   - [DONE] NERVF-3 routed-expert integratie. ERVF additief in
     `fused_nvfp4.py` achter `use_ervf` (default **uit**), vervangt
     `gemv_nvfp4_rows` overal: routed up, beide shared-projecties, NVFP4-Mamba,
     LM-kop. Drie armen base/ervf/base. **G-NERVF-3C exact: alle armen
     bit-identiek, ook tegen het bevroren V35-anker.** Tokenwinst conclusief op
     alle diepten: **ctx 0 −3,701 ms** (37,660 → 33,959; 26,55 → 29,45 tok/s),
     **131K −3,102**, **262100 −4,505** (55,640 → 51,135; 17,97 → 19,56 tok/s).
     G-NERVF-3P (≥1,35× op de MoE-component) **gefaald met 1,144×** — het
     geïnstrumenteerde venster omsluit router+shared+up+down+accum terwijl ERVF
     alleen de rij-GEMV vervangt, dus verdund per constructie; niet
     geherinterpreteerd. Verifier 66/66.
     Rapport: `reports/nervf_nemotron/NERVF_3_REPORT_2026-08-15.md`.
   - [DONE] NERVF-4 gatherless down — **WEERLEGD**. De gather weglaten en de
     masked GEMV rechtstreeks uit mapped host laten lezen is fors **trager**:
     MoE −5,99 / −8,45 / −8,42 ms en token −5,70 / −7,56 / −7,38 bij
     ctx 0 / 131K / 262100, alles ver boven de drift. G-NERVF-4P vroeg ≥6,55 ms
     winst. Exactheid bleef intact (identiek tussen armen en tegen het anker).
     Reden: V1 mat dezelfde GEMV op **6,7 GB/s** uit mapped host tegen 85,9
     vanaf device — het strided byte-per-thread-patroon overleeft PCIe niet,
     terwijl de gather warp-per-kolom met brede `uchar4`-loads leest. **De
     gather van 8,19 ms verdient zichzelf terug.** Dit sluit E2's eerste
     kandidaat (gather simpelweg weglaten); het sluit NIET een echte fusie die
     het **gecoalesceerde patroon** de GEMV in trekt zonder mirror — en deze
     meting laat precies zien waarom zo'n kernel dat patroon moet behouden.
     Meetnotitie: de eerste run gaf +3,0 ms door een stille `str.replace` die de
     armvlag niet zette (de armen herhaalden NERVF-3); artefact apart gezet als
     `nervf3r_ervf_replication_MISLABELED.json`, runner heeft nu asserts.
     Rapport: `reports/nervf_nemotron/NERVF_4_REPORT_2026-08-15.md`.
   - [DONE, GESTOPT OP STOPREGEL] NERVF-5 full model. 3 domeinen x 512 causale
     tokens. **G-NERVF-5C faalt, maar niet door ERVF: ook `base_b` wijkt af van
     `base_a`** terwijl beide `use_ervf=False` draaien. De opdracht schrijft dan
     voor te stoppen en te documenteren, en dat is gedaan.
     **Oorzaak, in `_moe_cached` en niet in ERVF:** `order = [hits] + [misses]`
     accumuleert de zes experts in hit-dan-miss-volgorde, niet in routevolgorde.
     Welke expert een hit is hangt af van de LRU-staat, dus twee runs met andere
     cachegeschiedenis tellen in andere volgorde op — en FP-optelling is niet
     associatief. Eerdere pariteitschecks liepen over 2x64 tokens en kwamen wel
     identiek uit; over 512 divergeert het. De eigenschap is dus **fragiel**, niet
     afwezig, en dat kwalificeert alle eerdere bit-identiek-claims in deze lijn
     en die van Kimi.
     Latencywinst blijft conclusief in alle drie de domeinen: **+1,789 /
     +4,605 / +4,330 ms** per token boven hun eigen drift. G-NERVF-5M (VRAM) OK.
     Rapport: `reports/nervf_nemotron/NERVF_5_REPORT_2026-08-15.md`.

10. [DONE 2026-08-15] **Deterministische accumulatievolgorde.**
    Vervang in `_moe_cached` de hit-dan-miss-accumulatie door: reken in
    hit-volgorde (die latencywinst blijft), maar bewaar de zes bijdragen apart en
    tel ze aan het eind in **routevolgorde** op — precies wat X1's `reduce_slots`
    al doet. Kosten ~64 KB per laag plus een reductiekernel. Pas daarna heeft elke
    exactheidspoort in deze lijn en in Kimi's E-lijn echte betekenis. Eigen
    preregistratie; niet binnen NERVF-5 geopend.
    **UITGEVOERD en geslaagd.** `rt.deterministic_accum` (default uit) scheidt
    rekenvolgorde van optelvolgorde: rekenen blijft hit-dan-miss (latencywinst
    behouden), elke bijdrage gaat naar zijn eigen slotbuffer, en na de lus wordt
    in routevolgorde opgeteld. Kosten 64 KB, nul extra kernels. Resultaat:
    **`base_b` nu identiek aan `base_a` over 3 x 512 tokens**, waarmee NERVF-5
    alsnog op alle drie zijn poorten slaagt: exactheid PASS, latency PASS
    (+2,771 / +5,008 / +5,395 ms, alle conclusief), VRAM PASS. ERVF is daarmee
    bewezen exact over 512-token rollouts, niet alleen over 2 x 64.
    Rapport: `reports/nervf_nemotron/D1_DETERMINISM_REPORT_2026-08-15.md`.
    Aanbeveling: default aanzetten zodra een fase het als basislijn gebruikt;
    default niet omgezet (productiebeslissing).
   Losse vondst uit 1: de kernel stageert `x` per block, dus **19,0 MiB
   activatieverkeer tegen 2,7 MiB gewichten (7,1×)**; 16 rijen per block snijdt
   dat 16×. Staat los van de reductiewinst.
   Rapport: `reports/nervf_nemotron/NERVF_0_1_REPORT_2026-08-15.md`.
