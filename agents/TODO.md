# TODO — de enige actieve takenlijst

Bijgewerkt: 2026-08-15 · vervangt `reports/treesweep200/HANDOFF_E4_EN_VERDER_2026-08-15.md`
(dat bestand blijft staan als historisch document, maar vink daar niets meer af)

Afvinkregel: zet `[DONE <datum>]`, één regel wat er gemeten is met het getal
erbij, en de bestandsnaam van het rapport. Een weerlegging is óók DONE.

---

## Afgerond

- [DONE 2026-08-15] **E0/P0** — pack ingelezen, treesweep200-namespace, identiteitsbaseline.
- [DONE 2026-08-15] **E4 kernelfase** — 5 kandidaten; v4 bitexact **−17,8%**.
- [DONE 2026-08-15] **E4 in-lus adoptie** (G-E4-T1) + onafhankelijke verifier 52/52.
  Nieuw anker `V35_GENERATION_ANCHOR.json` bevroren (de S5-baseline dateerde van
  vóór de v35-checkpointwissel). Rapport: `E4_INLOOP_REPORT_2026-08-15.md`.
- [WEERLEGD 2026-08-15] **E2 gatherloze downflow** — hetzelfde experiment als
  NERVF-4, één keer gemeten onder de strengere opzet (3 armen, 3 diepten tot
  262100). Exact, maar **−5,70 / −7,56 / −7,38 ms** per token; MoE-blok 0,72–0,79×.
  Oorzaak: strided host-reads 6,7 GB/s over PCIe tegen 85,9 vanaf device.
  Rapport: `E2_GATHERLESS_DOWNFLOW_REPORT_2026-08-15.md`.
- [DONE 2026-08-15] **E5 GEMV-roofline** — opgelost met ERVF i.p.v. een nieuwe
  kernel. Vier shapes bitexact én sneller: routed_up 1,621× · shared_up 1,784× ·
  shared_down 1,646× · lm_head 1,676×; **gewogen 77,1 → 127,9 GB/s (1,660×)**.
  `weighted_suite_ge_140` **FAIL** (127,9 — niet verruimd);
  `no_critical_shape_regression` **PASS**; `integrated_token_improvement_ge_8pct`
  **PASS**. Rapport: `E5_GEMV_ROOFLINE_REPORT_2026-08-15.md`.
- [FASE 1 DONE 2026-08-15] **E1 graph-resident token** — N1-oracle gereproduceerd
  op **22,2%** tegen 23,7% (afwijking 1,5pp, tolerantie 5pp). Met ERVF aan stijgt
  het naar **27,0% (8,925 ms)**: ERVF en graph-residentie zijn complementair.
  Gecombineerd plafond ctx 64: 24,112 ms (~41,5 tok/s).
  Rapport: `E1_GRAPH_ORACLE_REPORT_2026-08-15.md`. **Fase 2 staat hieronder open.**
- [DONE 2026-08-15] **E6 integratie** — fysieke A/B van wat gebouwd is (ERVF +
  v4 + D1) over 3 × 512 tokens: **41,980 → 37,490 ms**, winst +4,169 / +4,443 /
  +4,858 ms per domein, alle conclusief, **bit-identiek**, VRAM ongewijzigd.
  Eindpoort ≥50 tok/s **NIET gehaald** (26,7): E2 is weerlegd en E1 fase 2 is
  ongebouwd. Rapport: `E6_INTEGRATED_REPORT_2026-08-15.md`.
- [DONE 2026-08-15] **NERVF-0 t/m 5 — ERVF-replicatie op Nemotron.**
  - NERVF-0 baseline-lock ✅ · NERVF-1 geometrie-audit ✅ (bandbreedte-efficiëntie
    0,322 ≤ 0,40; reductie+sync 46,1% ≥ 25%)
  - NERVF-2 microkernel ✅ **alle vier breedtes bitexact (0/72 elk)**, w=16
    **1,936×**, 72,7 → 140,8 GB/s
  - NERVF-3 integratie ✅ exact tegen het anker, token −3,7 tot −4,5 ms
  - NERVF-4 gatherless ❌ **weerlegd** (= E2 hierboven)
  - NERVF-5 full model ⛔ gestopt op de eigen stopregel → leidde tot D1
  - Onafhankelijke verifier **66/66**. Eindrapport: `NERVF_NEMOTRON_FINAL_REPORT.md`.
- [DONE 2026-08-15] **D1 — deterministische accumulatievolgorde.** `_moe_cached`
  telde op in hit-dan-miss-volgorde, wat van de LRU-staat afhangt; twee armen met
  identieke configuratie divergeerden over 512 tokens. Opgelost door rekenvolgorde
  en optelvolgorde te scheiden. Kosten 64 KB, nul extra kernels. NERVF-5 slaagt
  daarmee alsnog op alle drie zijn poorten.
  Rapport: `D1_DETERMINISM_REPORT_2026-08-15.md`.
- [DONE 2026-08-15] **A1 — adoptie van de bewezen stack als default.**
  Preregistratie eerst. Harde poort: uitvoer identiek bij capacity 72 vs 56 met
  D1 ✅; **controle-arm zonder D1 divergeerde** (expository, token 224) — de test
  hád vermogen. Default nu: `use_ervf=True`, `rt.attn = attention_fp8_gqa4`,
  `deterministic_accum=True`, `gatherless_down=False`. Een runtime zonder enige
  vlag reproduceert bit-identiek wat A1 mat. Nieuw anker
  `V36_DETERMINISTIC_ANCHOR.json` bevroren; V35 blijft staan en is **niet**
  bit-vergelijkbaar. Rapport: `A1_ADOPTION_REPORT_2026-08-15.md`.

---

## Afgerond (pro_research, 2026-08-16)

- [DONE 2026-08-16] **Model-identiteitsonderzoek.** Uitgezocht waarom V3's
  eigen resultaten al bij token 1 afwijken van `V36_DETERMINISTIC_ANCHOR.json`.
  Antwoord: geen bug. `models/nemotron_3_5_lightning` (het ankerpad, gebruikt
  door de hele closed `treesweep200`-lijn: A1/D1/E1-E6/NERVF) is bij nameting
  `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, verkeerd gedownload en misleidend
  hernoemd. Het echte opdrachtdoel zit in `models/nemotron_3_5_lightning_v35`
  en is wat `pro_research` default gebruikt. Zie `RESEARCH_NOTEBOOK.md`
  2026-08-16 en `STATE_OF_THE_WORK.md` bovenaan.
- [DONE 2026-08-16] **PRO V4 — graph-safe + selectieve ERVF fysiek
  geïntegreerd.** Selectieve ERVF-dispatch geïnstalleerd op `rt.k` vóór
  `rt.setup_graph()` zodat de CUDA-graph de ERVF-kernels meevangt. Full-mode
  (256×3, 765 samples): **31,1786 → 24,3152 ms/token (+22,0%, 41,13 tok/s)**,
  bitexact, deterministisch, controle-arm wijkt af zoals vereist, VRAM +4 MiB.
  Nieuw record op het juiste model. Rapport/artefacten:
  `pro_research/PRO_V4_PREREGISTRATION.md`,
  `pro_research/results/PRO_V4_GRAPH_SELECTIVE.json`.

## Open

- [ ] **Down_proj-pijplijn optimaliseren in `_moe_dev` (hoogste prioriteit,
      BIJGEWERKT 2026-08-16 — eerste versie van dit item was deels fout, zie
      hieronder).** ~~Device-cache down_proj net als up_proj~~ — **onhaalbaar**:
      `DOWN_PANEL_BYTES` is 2,68 MB/expert (niet ~1 kB zoals eerst aangenomen),
      vol cachen bij cap 72×23 lagen kost ~4,4 GiB, GPU had tijdens V4 al 0 MiB
      vrij. Componentmeting (`diag_component_timing_v4.py`,
      `diag_down_subkernels_v4.py`) geeft een preciezer beeld: de hele
      down_proj-pijplijn (`panel_scan`+`gather_down_sparse_ind`+
      `down_masked_ind`+`reduce_partials`) kost **11,39 ms/token (29,9%)**,
      groter dan up_proj's eigen GEMV (5,00 ms/token). Twee onafhankelijke
      deelhefbomen, geen van beide alleen dominant:
      1. `gather_down_sparse_ind` (PCIe host-gemapte masked read, 4,74 ms/token,
         41,6% van de pijplijn) — mogelijk dezelfde klasse trage
         strided-host-toegang als E2/NERVF-4 al vond.
      2. `panel_scan`+`reduce_partials` samen (4,74 ms/token) — kleine
         device-only kernels, kosten wijzen op **launch-overhead** (552
         kernellaunches/token alleen al voor down_proj: 4 subkernels × 138
         expert-aanroepen). Fusie/batchen over de 6 experts per laag is een
         onafhankelijke hefboom.
      Voorzichtige bovengrens: zelfs een volledige eliminatie van de hele
      pijplijn zou V4 van ~24,3 naar ruw ~13-15 ms/token brengen (~65-75 tok/s,
      niet 100) — substantieel maar niet op zichzelf genoeg. Twee echte
      CUDA-engineeringtaken, niet gebouwd deze sessie (kritiek pad van een 30B
      productiemodel, verdient eigen preregistratie + bitexact-verificatie
      i.p.v. haast). Zie `RESEARCH_NOTEBOOK.md` 2026-08-16 voor de volledige
      onderbouwing en alle vier diagnostische scripts/JSONs.
      **Preregistratie klaar om uit te voeren:** `pro_research/PRO_V5_PREREGISTRATION.md`
      — dekt alleen de launch-overhead-helft (batchen van de 4 down_proj-
      subkernels over de 6 expert-slots per laag, architecturaal veilig
      bevestigd, geen kernelrekenkunde gewijzigd), niet de PCIe-kant. BASE_A/
      BATCHED/BASE_B/CTL-armen, bitexact-poort, controle-arm die moet falen.
      Nog niet gebouwd — vereist nieuwe CUDA-kernels (batched varianten), dus
      een aparte, zorgvuldige sessie i.p.v. haast.
- [ ] **Per-laag capaciteitstuning.** Zelfde diagnose: missrate is sterk
      niet-uniform over lagen (laag 1/3/6/51 missen 25-42%, de rest 6-14%).
      Bevestig eerst stabiliteit over meerdere prompts vóór er iets aan de
      capaciteit per laag verandert.
- [ ] **Push V4-resultaten** — `git add -f pro_research/PRO_V4_PREREGISTRATION.md
      pro_research/graph_selective_v4.py pro_research/results/PRO_V4_GRAPH_SELECTIVE.json`
      + de bijgewerkte `agents/`-bestanden, commit, push naar `pro-research`.
- [WEERLEGD-VOOR-DEZE-AANPAK 2026-08-16] **G2 — K-token epoch-graph.**
  `pro_research/epoch_graph.py --mode smoke` gedraaid: `technical_blocked`,
  `cudaErrorStreamCaptureUnsupported` — `cudaGraphLaunch()` op een reeds
  geïnstantieerde graph is zelf niet capturable; je hebt de graph-**template**
  + `cudaGraphAddChildGraphNode()` nodig, niet `.launch()` op de exec. Niet
  opnieuw proberen met dezelfde aanpak. Open vervolgpad: `setup_graph()`
  aanpassen om de capture-template apart te bewaren en lage-niveau
  CuPy-runtimebindings gebruiken voor `AddChildGraphNode` — een aparte,
  grotere CUDA-taak, niet gedaan in deze sessie. Rapport:
  `pro_research/results/PRO_G2_EPOCH_GRAPH.json`.
- [WEERLEGD 2026-08-16] **MTP speculatief decoderen (S10 stap 2).** Was al
      heropend voor Lightning en deels gemeten (`S10A_MTP_ACCEPTANCE_REPORT_2026-08-15.md`:
      acceptatiegraad `A=2,114`, poort G-S10-1 gehaald) vóórdat deze sessie
      begon — maar de prereg identificeerde zelf de beslissende onbekende
      term: de unie van experts over een 5-token-verificatiesweep. Gemeten
      (`pro_research/diag_mtp_route_union.py`, teacher-forced replay via de
      al bestaande `capture_routes`-API, geen bouw nodig): **19,88 van de 128
      experts per laag, 3,313× t.o.v. 6 voor één token.** Met dat getal in de
      eigen rekensom van het rapport: speculatief **57,51 ms/token vs.
      niet-speculatief 54,28 ms/token — 6,0% trager, niet sneller.** S10 stap
      2 sluit. Niet opnieuw proberen zonder een architecturaal ander idee
      (bv. minder routed experts per token, of experts met meer
      gedeeldheid tussen naburige tokens forceren — allebei kwaliteitsingrepen
      op het model, geen runtime-truc). Zie `RESEARCH_NOTEBOOK.md` 2026-08-16.
- [ ] **Closed `treesweep200`-lijn herhalen op het juiste model** —
      `LS_MODEL_DIR=nemotron_3_5_lightning_v35` zetten en A1/D1/E1-E6/NERVF
      opnieuw ijken, of het ankerpad expliciet hernoemen zodat de naam niet
      meer liegt. Niet in deze sessie gedaan (schrijfrechten buiten
      `agents/`+`pro_research/` niet opgeëist).
- [ ] **E1 fase 2 — de echte graph-resident token.**
  - [DONE 2026-08-15] **Fase 2.1 — device-resident routing + device-LRU
    (eager).** Alle vijf poorten PASS; p50 41,540 → **36,998 ms/token
    (−4,542 ms)**, pariteit behouden, verifier 14/14. Bugfix:
    `enable_cache` reset nu ook `_dev_cache`. Rapport:
    `E1F21_DEVICE_ROUTING_REPORT_2026-08-15.md`.
  - [GEBOUWD, ONGEMETEN 2026-08-15] **Fase 2.2 — CUDA-graph-replay van de
    volledige token.** Preregistratie BEVROZEN:
    `E1F22_GRAPH_CAPTURE_PREREGISTRATION_2026-08-15.md` (poorten PAR/CTL/DET/
    S1 ≥ 2,5 ms/VRAM < 64 MiB, kill-criteria K1–K3). Code staat klaar maar is
    **niet gedraaid**: dp-kernels in `gpu_kernels.py` (embed_gather_bf16,
    kv_append_fp8_dp, attn_decode_warp_fp8_gqa4_dp met vaste grid + neutrale
    partials, argmax_part/final, pos_inc) en `runtime.py` (graph_mode,
    `setup_graph()`, `_step_body_graph()`, `step_graph()`, `ring_harvest()`).
    Graph-API wél gesmoketest. **Volgende stappen, in volgorde:**
    1. Runner schrijven `scripts/treesweep200/e1f22_graph_capture_ab.py`
       (4 armen EGR/GRAPH/CTL/DET exact zoals de prereg; basis van
       `e1f21_device_routing_ab.py`; GPU-check eerst).
    2. Draaien. Bij capture-fout op de multi-stream fork → K1-fallback
       (single-stream) als eerlijk benoemde tweede arm.
    3. Verifier `e1f22_independent_verify.py`: herberekent poorten uit het
       JSON + eigen kernelchecks (argmax vs cp.argmax incl. gefabriceerde
       ties; gqa4_dp bitexact vs gqa4 voor t ∈ {1, 64, 511, 512, 513, 4096};
       embed_gather vs de cupy-omzetting). Importeert de runner NOOIT.
    4. Rapport + registry-entry + notebook + protected-manifest verify tegen
       de **nieuwe** baseline
       (`PROTECTED_80B_MANIFEST_AFTER_USER_COMMIT_2026-08-15.json`).
    5. Bij S1-PASS: overwegen graph als default te adopteren (aparte fase,
       eigen prereg).
  - Resterend budget na 2.1: ~4,4 ms launch-overhead (van 8,925 uit fase 1).
- [ ] **Langecontext-profiel van de geadopteerde stack** — E6 mat 3 × 512 tokens
      bij `contexts_max=4096`. De stack is nooit end-to-end gemeten op 128K/262K
      ná adoptie. NERVF-3 deed dat vóór D1. Nu óók mét device_cache meten.
- [ ] **Duurloop** — ≥10.000 causale tokens en één thermisch uur, om te toetsen of
      exactheid en winst standhouden buiten korte rollouts.
- [ ] **Prior-art-audit + stock llama.cpp-differentieel** — nodig vóór er ooit een
      nieuwheidsclaim over ERVF wordt gedaan. Nu wordt die claim expliciet
      **niet** gemaakt.

## Doorlopend

- [ ] Na elke fase: `protected_manifest.py verify` (**0 modified / 0 removed**)
      tegen `PROTECTED_80B_MANIFEST_AFTER_USER_COMMIT_2026-08-15.json`
      (nieuwe baseline, gebouwd ná de eerste git-commit van de eigenaar —
      de oude baseline is pre-git en markeert .gitignore vals),
      registry-entry bijwerken, rapport schrijven, hier afvinken, en één blok
      toevoegen aan `RESEARCH_NOTEBOOK.md`.
