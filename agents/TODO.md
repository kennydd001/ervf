# TODO — de enige actieve takenlijst

Bijgewerkt: 2026-08-15 · vervangt `reports/treesweep200/HANDOFF_E4_EN_VERDER_2026-08-15.md`
(dat bestand blijft staan als historisch document, maar vink daar niets meer af)

Afvinkregel: zet `[DONE <datum>]`, één regel wat er gemeten is met het getal
erbij, en de bestandsnaam van het rapport. Een weerlegging is óók DONE.

---

## Afgerond

- [WEERLEGD 2026-08-16] **PRO V12 async-harvest** — Kimi's bevroren prereg,
  ongewijzigd gedraaid op de V6-stack. QUEUED-K (K=2..32) en EVENT-STREAM-K
  (K=4,8,16) halen **43,1-44,7 tok/s** tegen SYNC **45,1** — geen enkele arm
  sneller, geen E50. Beslissend getal: pure Python-uitgifte kost
  **0,0128-0,0537 ms/token = 0,06-0,24% van een token**. Er is geen
  queue-starvation; host-scheduling is meetbaar nul als E50-hefboom. Alle armen
  bitexact (765 tokens × 8 armen). Maakt V12B/V12C overbodig vóór uitvoering.
  Weerlegt tevens PV2-20's eigen controle-arm (18,8-19,1 ms/token reproduceert
  niet binnen de volledige V6-rekenkunde) — open discrepantie, geen record.
  Resultaat: `pro_research/results/v12_async/PRO_V12_ASYNC_HARVEST.json`.
- [DONE 2026-08-16] **Drift-stabiele meetharness gevonden** — bijvangst van V12
  en het eigenlijke resultaat: `baseline_drift_le_1ms` haalt **0,108 ms**, waar
  PV2's hele campagne op 1,86-3,24 ms strandde. Recept: 128 tokens preheat,
  één gevangen runtime (niet hercompileren tussen armen), `_reset_exact_state()`
  dat state wist zonder graph-pointers te heralloceren, armen kort na elkaar in
  één proces. **Hierdoor is PV2-11 (Q/K/V one-launch: exact op 3/3 prompts,
  0,2387 ms onder baseline-midden, alléén op drift gesneuveld) opnieuw
  beslisbaar.** Zie "Open — eerstvolgend".
- [DONE 2026-08-16] **N=2 CUDA-graph multi-sequentie + echte staging-race-fix** —
  drie bugs gevonden vóór enige tok/s-claim (4-byte pinned staging-slot dat een
  gequeude H2D-copy te laat las → garbage==garbage "bitexact"; CACHE_ATTRS-
  referentie-aliasing tussen graphs; `setup_graph` early-return). `runtime.py`
  stageert prompttokens nu via een 256-slots pinned ring. Gemeten (beide
  bitexact): private caches cap-24 → **36,86 tok/s** aggregate; gedeelde cache
  cap-64 op één stream → **33,52**. Beide **onder** het single-stream record
  van 47,41 — geïnterleavede batch>1 in deze vorm is netto verlies, geen winst.
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
- [DONE 2026-08-16] **Multi-seq graph N=2 met PRIVATE caches — EERSTE RUN
  ONGELDIG (23,59 tok/s), HERMETING GELDIG: 36,86 tok/s bitexact.** Claude's
  ongedraaide prototype `proto_multi_seq_graph_n2.py` afgemaakt en gedraaid.
  Zijn open vraag beantwoord: V4-V6 gebruikten runtime.py's eigen
  `setup_graph/step_graph/_step_body_graph` — het "NOT YET RUN"-commentaar
  was stale (inmiddels bijgewerkt). **VRAM-fix vóór de run**: 2× cap-72
  (8,66 GiB) past niet in 8151 MiB → cap 24 per sequentie (numeriek
  invariant, E1F21-INV). De eerste run rapporteerde "fase 2 PASS, 23,59
  tok/s" maar was **ongeldig**: staging-race corrupteerde de prompts
  (garbage==garbage-vergelijking). Drie echte bugs gevonden en gefixt:
  (1) staging-race in `step_graph` → 256-slot pinned ring in runtime.py;
  (2) **CuPy-pool-aliasing**: `cache`/`_dev_cache` zaten niet in de state-
  snapshot, de pool gaf sequentie 1's cache exact dezelfde adressen als
  sequentie 0's vrijgekomen buffers → beide graphs schreven in gedeeld
  geheugen (fix: `CACHE_ATTRS` in de snapshot — refs vasthouden is
  correctheid bij pointer-gebonden graphs); (3) `setup_graph()` early-return
  bij `_graph is not None` → sequentie 1 kreeg nooit een eigen graph (fix:
  `rt._graph = None` vóór tweede capture). **Geldige hermeting, beide armen
  fase 2 écht bitexact (20/20 × 2, coherente prompts): private cap-24×2 =
  36,86 tok/s aggregate (27,13 ms/token); gedeelde cap-64 één-stream =
  33,52 tok/s (29,84 ms/token).** Beide boven naïef-eager N=2 (31,66) en
  ver boven expliciete-deling (11,23); ver onder solo V6 (47,41) — scaling
  0,78× solo voor 2× werk. Verrassing: shared < private ondanks grotere
  cache; werkhypothese LRU-thrash bij afwisselende werkingssets + enkele
  stream, nog NIET gemeten. Volgende stap: expert-unie-fetch ín de graph-
  loop (mechanisme elders bitexact bewezen, 1,71× over 23 lagen) of N=4
  met hetzelfde patroon. Zie RESEARCH_NOTEBOOK.md 2026-08-16 (bovenste
  twee blokken).
- [DONE 2026-08-16] **Hitrate-vervolgmeting: thrash-hypothese WEERLEGD —
  delen werkt juist; de enkele stream is de boosdoener.**
  `diag_n2_graph_cache_hitrates.py` las de device-tellers van
  `cache_assign` uit (integriteitsasserts: beide N=2-armen én de solo-armen
  reproduceerden de gated tokens exact). Solo: cap72 63,4% / cap64 62,9% /
  cap24 55,1%. Private N=2: 55,1% + 49,5%. **Shared N=2: 71,1% — hoger
  dan solo cap-64; cross-sequentie-lokaliteit is reëel.** Shared had ~1936
  missers minder (~209 ms gewonnen) maar was 108 ms trager → de enkele
  gedeelde stream serialiseert de cross-sequentie-PCIe/compute-overlap die
  private (eigen streams + eigen copy_streams) wél heeft (~317 ms verloren).
  "Gedeelde cache + twee streams" racet op de LRU-tabellen → structureel
  niet oplosbaar met losse per-sequentie-graphs → argument voor de
  gebatchte-graph-route (Path B, POST_V6_100TPS_PLAN.md).
- [DONE 2026-08-16] **PRO-MAX V2 (branch pro-max-v2, kennydd001/ervf) —
  post-V6 final-mile naar 50 tok/s: geen E50; thermische drift > het hele
  E50-gat.** Payload gereconstrueerd uit 7 base64-delen, zip-SHA256 + 19/19
  bestandsmanifest geverifieerd vóór uitvoering; `-Mode install` PASS;
  smoke en full gedraaid. **Full (≥500 samples/arm, definitief):**
  PV2-10 add+norm micro-bitexact maar **causal_parity FALSE bij 256 tokens**
  (smoke was true — D1-les: korte runs zien de fout niet) → terecht
  afgewezen, eerst debuggen vóór herindiening. PV2-11 Q/K/V one-launch:
  alle exactheidspoorten groen, +0,24 ms binnen regressiegrens, afgewezen
  ALLEEN op drift-poort (1,86 ms > 1,0) → **hermeten onder thermisch
  stabiele condities verdienen**. PV2-12 LM-head+argmax: exact maar echt
  trager (22,37 ms) → weg. PV2-13 finale met lege selectie: 46,48 tok/s,
  E50 false. PV2-20 child-graphs: bit-identiek mechanisme dat WERKT
  (cudaGraphAddChildGraphNode), geen speedup; nevenmeting: pure gequeue-de
  replay ~19,2-19,5 ms/token → ~1,6 ms/token Python/launch-overhead in de
  productie-21,09 ms. PV2-21: child-graph/conditional/TMA-symbolen
  aanwezig, mapped-host→SMEM TMA onbewezen. **Systematisch**: baseline
  drift 1,9-3,2 ms binnen één arm (base_a ~20,6-20,8 ms ≈ 48+ tok/s koud!)
  — thermische throttling is groter dan het hele E50-gat (1,09 ms) → Path
  A is zonder thermisch gestabiliseerde meting niet beslisbaar; poort niet
  verruimen, omgeving verbeteren (vaste clocks / steady-state / kortere
  A/B-interleave). Onafhankelijke verifier bevestigde elke status.
  Artefacten: `pro_research/results/pro_max_v2/PV2_*.json`,
  `PV2_FINAL_REPORT.md`, `PV2_VERIFICATION.json`. Zie RESEARCH_NOTEBOOK.md
  2026-08-16 bovenste blok.

## Open — eerstvolgend

- [ ] **HOOGSTE PRIORITEIT — gather-grid verkleinen zodat de overlap échte
      SM-ruimte krijgt.** B3 is gebouwd, bitexact en werkt (zie hieronder), maar
      verstopt maar **16,8%** van de 2,47 ms. Oorzaak is gemeten en structureel:
      `gather_down_sparse_ind` is een SM-side zero-copy kernel (moet wel — de
      selectie is data-afhankelijk), dus gather en `down_masked` **vechten om
      dezelfde SM's**. De launch is bovendien op de worst case gesized:
      **247 blokken waarvan er ~31 werk hebben** (164,7 kolommen + 90,1 panelen
      ≈ 247 warps van de 1972 gelanceerd). Een kleine statische grid (capture
      vereist statisch) met een grid-stride-lus over warps laat veel meer SM
      over voor `down_masked`. Goedkoop, raakt precies het mechanisme, en de
      A/B-harness staat er al (`overlap_v14_graph.py`, drift 0,32 ms).
      Als dit werkt is het pad naar 100 open; zo niet, dan is de PCIe-tijd
      alleen met de copy-engine te verstoppen en dat vraagt een ander
      transferpatroon.
- [DEELS-DONE 2026-08-16] **B3 — PCIe-gather overlappen met VRAM-werk
      (double-buffer expert fetch). Gebouwd, bitexact, in de graph −0,416 ms.**
      Eager was **+3,65 ms** (langzamer): `diag_event_op_cost` prijsde een kale
      `Event.record` op 0,285 µs maar het volledige cross-stream
      wait/record-patroon op ~183 µs per iteratie, en V14 doet er 138 per token
      — de eager-uitslag prijsde de scheduler, niet het idee. In de graph zijn
      fork/join statische randen en klapt het teken om: **−0,4158 ms, alle
      correctheidspoorten groen, capture accepteerde de multi-stream-topologie**.
      Eigen poort was ≥0,8 ms → `gate_failed`, niet verruimd.
      Oorspronkelijke onderbouwing: De rekening, met alleen gemeten getallen: het eerlijke
      kernelplafond is **249 GB/s** koud (`diag_gemv_width32.json`), dus de
      VRAM-vloer is 2048/249 = **8,22 ms**; de sparse down_proj rijdt over PCIe
      (25,9 GB/s gemeten) en kost daar **2,47 ms**.
      - serieel: 8,22 + 2,47 = 10,69 ms = **93,6 tok/s** → **100 onbereikbaar**
      - volledige overlap: 8,22 ms = **122 tok/s** → 100 haalbaar met 82%
        efficiëntie
      **100 tok/s staat of valt dus met het verstoppen van de PCIe-gather onder
      het VRAM-werk.** Concreet: de gather van expert-slot s+1 moet lopen
      terwijl de up-GEMV/masked-GEMV van slot s rekent (aparte stream +
      double-buffered mirror, ~2,7 MB extra per buffer), en de MoE-marginaal
      bevestigt de prijs: van MoE's 5,81 ms hoofdruimte is 2,47 ms zuivere
      PCIe-tijd die géén snellere kernel ooit weghaalt.
- [WEERLEGD 2026-08-16] **32-lane ERVF-GEMV (cacheline-breedte-hypothese)** —
      gebouwd en **bitexact op alle vier de echte shapes** (bij width 32 valt
      virtuele tid `lane + 32·vj` exact in referentiewarp `vj`, dus een gewone
      32-brede shuffle-reductie reproduceert de boom letterlijk), maar de
      snelheid is **neutraal**: 0,954 / 1,033 / 1,068 / 0,982×. 64 B versus
      128 B per instructie was niet de rem. `diag_gemv_width32.json`.
      Bijvangst: dezelfde run corrigeert het koude-GEMV-plafond van 209-229 naar
      **230-261 GB/s** — de eerdere meting liep deels op 795 MHz SM-klok
      (throttle), zichtbaar in haar eigen klokregistratie.
- [WEERLEGD 2026-08-16] **FP8-decode-LUT als bottleneck** — de rekenkundige
      LUT-vrije e4m3-decode is bitidentiek op alle 256 bytewaarden maar
      **25-27% langzamer** op alle vier de echte shapes. De kernel is
      bandbreedtegebonden met ALU over; een SMEM-lookup is goedkoper dan zes
      ALU-ops per byte. Deur dicht. `diag_fp8_lutfree_gemv.json`.
- [ ] **H-SCALE — down_proj-blokschalen residentie in VRAM. Gebouwd, bitexact,
      past in VRAM, maar −0,374 ms/token tegen een eigen poort van ≥0,5 ms →
      `gate_failed`, poort niet verruimd.** Marginale ontleding: plane-fetch
      kost +0,327 ms, bruto gather-besparing 0,701 ms. Twee wegen om er alsnog
      boven te komen (samen ≈ −0,54 ms): contiguë host-repack van de
      schaalvlakken, en schaalvlakken voor alle 128 experts residentie maken
      (875 MiB, vraagt up-capaciteit 72→68). Lagere prioriteit dan de GEMV.
      Oude beschrijving hieronder ter referentie: Grondslag: `diag_gather_pcie_ceiling.json` toont dat
      **52,2% van al het down-PCIe-verkeer FP8-blokschalen zijn** (90,1 actieve
      panelen × 2688 B tegen 164,7 kolommen × 1344 B), en hypothesearm v3 prijst
      het weghalen daarvan op **−1,380 ms/token gemeten**. Netto na extra
      missverkeer ≈ **−1,14 ms/token** → 21,09 → 19,95 ms ≈ **50,1 tok/s**.
      Geen rekenkundige wijziging: dezelfde schaalbytes, andere herkomst.
      Bouwstappen:
      1. per laag een device-buffer `scale_planes` van `cap × 311.808 B`
         (72 × 23 = 492,4 MiB; gemeten vrij tijdens de V12-full-run: 639 MiB);
      2. `cache_fetch` DMA't bij een miss óók het schaalvlak naar slot `s`;
      3. nieuwe gather-kernel zonder schaalpanelen (v3 staat al geschreven in
         `diag_gather_pcie_ceiling.py`);
      4. `gemv_down_masked_partial_ind` leest de schalen uit
         `scale_planes + slot*plane_bytes + panel*2688` in plaats van uit de
         mirror;
      5. poort: bitexact tegen de huidige V6-stack over 3 × 256 tokens, VRAM
         binnen budget, dan A/B in de V12-harness (drift 0,108 ms — beslisbaar).
- [ ] **PV2-11 (Q/K/V one-launch) hermeten in de V12-harness.** Was exact op
      3/3 prompts en 0,2387 ms onder het baseline-midden, en sneuvelde **alleen**
      op de driftpoort (1,8577 ms). De V12-harness haalt 0,108 ms drift, dus deze
      kandidaat is nu beslisbaar. Code staat in `pro_research/pro_max_v2/qkv_v8.py`.
- [ ] **PV2-10 (add + next-RMSNorm) late divergentie isoleren** — micro
      bitexact, maar causale pariteit faalt pas bij gegenereerd token 124 van
      één prompt. Kimi's `diag_addnorm_late_divergence.py` (branch
      `pro-v12-async`) staat klaar en is nog niet gedraaid. Niet heraanbieden
      als kandidaat vóór de oorzaak bekend is.

## Open

- [ ] **NIEUW, HOOGSTE PRIORITEIT 2026-08-16 — batch>1 cross-sequentie
      expert-unie meten (nog niet gebouwd, alleen een meting nodig).**
      Alle winst tot nu toe (V4-V6, elke kernel-batching, capaciteitstuning)
      blijft binnen **batch=1** — en het 165 tok/s-roofline-plafond zelf is
      ONDER die aanname berekend. Niets binnen batch=1 kan daar ooit boven
      komen; 100 tok/s vraagt 60,6% van dat plafond, V6 zit op 28,7%.
      Hypothese: bij **N sequenties gelijktijdig** hoeft een expert maar
      één keer per stap van host geladen te worden, niet één keer per
      sequentie — dat kan het aggregate-plafond zelf optillen (een ANDER,
      hoger plafond, geen schending van het bestaande). Bevestigd: de
      runtime heeft **nul** batch-ondersteuning (elke buffer 1D,
      single-sequence) — dit is geen kleine uitbreiding maar een
      meerdere-weken-herontwerp, met opzet niet gebouwd deze sessie.
      **[EERSTE METING DONE 2026-08-16] Cross-sequentie expert-unie
      gemeten** (`diag_cross_sequence_union.py`, 16 diverse prompts, 20
      stappen elk, `capture_routes`): bij N=16 gelijktijdige sequenties is
      de gemiddelde unie 63,9 van de 128 experts per laag — **66,6% van de
      no-overlap-baseline van 96**, dus 33% minder unieke PCIe-gebonden
      expert-loads nodig voor evenveel nuttige tokens. In tegenstelling tot
      MTP (gesloten, want de draft-kost woog niet op tegen de winst) is hier
      **geen speculatieve/weggegooide kost** — elke sequentie is een echt
      opgevraagde token, dus elke gedeelde expert is pure winst.
      **[MECHANISME FYSIEK GETEST 2026-08-16] `proto_batch_moe_layer.py`**
      — één echte laag, N=16 echte sequenties, cold-cache: NAIVE 96 fetches
      / 12,60 ms vs BATCHED 33 fetches (unie) / 4,36 ms — 2,89× sneller,
      bitexact, 0 mismatches tussen naive en batched output.
      **[UITGEBREID NAAR ALLE 23 LAGEN + COMPUTE APART GEMETEN 2026-08-16]
      `proto_batch_moe_multilayer.py`** — zelfde opzet over alle 23 MoE-
      lagen, fetch en compute apart getimed. **Bitexact op alle 23 lagen, 0
      mismatches totaal.** Fetch-winst wisselt per laag (1,42×-3,15×,
      consistent met eerder gemeten niet-uniforme lokaliteit per laag).
      Compute-tijd blijft vlak tussen naive/batched (geen straf voor
      batchen — alleen fetch profiteert, zoals verwacht). **Opgeteld over
      alle 23 gemeten lagen (niet geëxtrapoleerd): 367,05 ms → 214,43 ms,
      1,71× sneller** — een preciezer, minder toevallig-gunstig getal dan
      de 2,89× van de losse laag 24.
      **[DOWN_PROJ OOK GETEST 2026-08-16] `proto_batch_down_proj.py`** —
      down_proj is architecturaal ANDERS dan up_proj (gemaskeerd/sparse,
      niet gewoon dedupliceren op expert-id: twee sequenties met dezelfde
      expert kunnen andere niet-nul-kolommen nodig hebben). Juiste
      generalisatie gebouwd: unie van niet-nul-kolommen ophalen (boven-
      verzameling), elke sequentie rekent daarna met haar EIGEN masker
      tegen de gedeelde mirror. Echte post-ReLU2-activaties (echte
      up_proj-GEMV gedraaid, geen synthetische data). **Bitexact, 0
      mismatches, 1,91× sneller fetch (6,44→3,37 ms), 54,0% minder bytes
      over PCIe** (op één laag — nog niet over alle 23 herhaald zoals
      up_proj wel is). Claim-grens ongewijzigd streng: dekt nog steeds
      geen shared expert, attentie, Mamba, KV-cache, graph-capture,
      routing/argmax/norm-overhead. Geen doorvoerclaim. **Volgende stap is
      architectuurontwerp** (batch-
      dimensie op alle buffers, per-stap expert-unie-bepaling voor de
      volledige runtime) — een meerdere-weken-taak, niet gestart deze
      sessie, maar het kernmechanisme is nu consistent bewezen correct én
      fysiek sneller over de hele MoE-stack, niet toevallig op één laag.
      **[ARCHITECTUURONTWERP DONE 2026-08-16] `agents/BATCH_ARCHITECTURE_DESIGN.md`**
      — de stappen die een echte batch>1-integratie nodig heeft (routing-
      unie in de staplus, `cache_assign` voor N×top_k, batch-dimensie op
      alle buffers, graph met actief-masker voor continuous batching), plus
      de belangrijkste eerlijke waarschuwing: **attentie/Mamba/KV-cache
      hebben geen deel-mogelijkheid** (geen expert-selectie, dus geen PCIe-
      amortisatie zoals bij MoE) — de aggregate-winst zal dus kleiner zijn
      dan de MoE-alleen-cijfers suggereren. Bevat een expliciet-als-
      **rekensom-niet-meting** gelabelde grove bovengrens (~114 tok/s
      aggregate bij aanname van perfecte MoE-deling en ongewijzigde rest —
      niet gemeten, waarschijnlijk optimistisch, zelfde soort rekensom als
      S10's MTP-voorcalculatie die achteraf te optimistisch bleek).
      **[AANBEVOLEN METING GEDAAN 2026-08-16] `diag_attention_n_scaling.py`**
      — bestaande Q-GEMV N keer gedraaid tegen N echte activaties,
      N∈{1,2,4,8,16}: **94-97% van ideaal lineair, ms/sequentie nagenoeg
      constant.** Bevestigt de aanname: attentie schaalt ~lineair, geen
      launch-overhead-speling zoals MoE had.
      **[CORRECTIE 2026-08-16] `diag_mamba_n_scaling.py`** — het
      ontwerpdocument nam bij analogie aan dat Mamba hetzelfde zou doen als
      attentie; dat bleek **fout**. Mamba se `in_proj` (FP8-per-tensor-
      kernel, fysiek andere kernel) is **mild supra-lineair**: ms/sequentie
      stijgt van 0,177 ms (N=1) naar ~0,203-0,206 ms (N=8-16) — een reële
      ~15% straf bij grotere N, geen neutrale schaling. De ~114 tok/s-
      bovengrensrekensom nam "rest ongewijzigd" aan — dat klopt voor
      attentie maar niet voor Mamba, dus de werkelijke aggregate bovengrens
      ligt iets lager dan 114. Betekenis: een batch>1-integratie haalt zijn
      winst vrijwel uitsluitend uit MoE (57,8% van het token); de rest
      profiteert niet alleen niet mee, Mamba wordt er zelfs iets duurder
      van per sequentie. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16.
- [DONE 2026-08-16] **Down_proj-pijplijn optimaliseren in `_moe_dev`.**
      ~~Device-cache down_proj net als up_proj~~ — **onhaalbaar**:
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
      **[DONE 2026-08-16] Gebouwd, geverifieerd, geïntegreerd — V5 + V6.**
      `PRO_V5_PREREGISTRATION.md` uitgevoerd in drie stappen: (1) geïsoleerde
      kernel-unittest van `panel_scan_batched`/`reduce_partials_batched`,
      bitexact bij 6 sparsity-niveaus; (2) causale A/B/A/CTL in eager modus
      (`v5_batched_downproj_ab.py`) — bitexact, controle-arm wijkt af,
      **−2,2126 ms/token (−7,07%)**, alle poorten groen; (3) **V6**: alle drie
      mechanismen (device routing + graph-safe + selectieve ERVF + batched
      down_proj) tegelijk in één graph gevangen (`graph_v6_full_stack.py`,
      niet apart gepreregistreerd — volgt rechtstreeks uit V4/V5's eigen
      poorten). **Nieuw record: 22,6306 ms/token, 44,19 tok/s, +27,4% t.o.v.
      zelfde-sessie EGR**, alle poorten groen (bitexact, deterministisch,
      controle-arm, VRAM). Onderweg een echte VRAM-bug gevonden en gefixt
      (overbodige dubbele `mirror`-buffer, ~61,6 MB) — precies waarom de
      VRAM-poort bestaat. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "PRO V5
      + V6".
- [ ] **MoE is 57,8% van V6's token (12,07 ms van ~20,9-22,6 ms), niet
      alleen down_proj (6,51 ms daarvan).** Componentafbraak
      (`pro_research/diag_v6_component_breakdown.py --drive`, 2026-08-16):
      attentie 14,9% (3,10 ms), Mamba ~0% (ruis in deze apart-proces-meting),
      MoE 57,8%, lm_head+shared-expert samen 10,1% (2,10 ms, één methode
      `fused.gemv_into` bedient beide — niet los te meten zonder
      surgischer patchen). Onverklaarde rest binnen MoE (~5,56 ms na aftrek
      van down_proj): shared-expert-GEMV's, up-proj ERVF-GEMV, de batched
      panel_scan/reduce_partials-kernels zelf, `accumulate_indirect`,
      routing/cache-kernels — nog niet los gemeten.
      **[DONE 2026-08-16] `accumulate_indirect` batchen** — gebouwd met de
      juiste, veilige aanpak (niet de mechanische V5-kopie): nieuwe
      `weighted_accumulate_ind_batched`-kernel reproduceert de exacte
      `s=0..5`-fmaf-volgorde in één launch i.p.v. een parallelle/atomic
      reductie die de FP-optelvolgorde had kunnen veranderen (D1-les).
      Bitexact geverifieerd (geïsoleerd + causale A/B), **−3,1552 ms/token
      (−9,88%) eager**. V6 opnieuw gedraaid: 44,37 tok/s (was 44,19).
      Zie `RESEARCH_NOTEBOOK.md` 2026-08-16.
      **[DONE 2026-08-16] Up-proj ERVF-GEMV (`gemv_nvfp4_ervf_ind`)
      batchen** — zelfde veilige klasse als panel_scan/reduce_partials
      (onafhankelijke output per slot, geen race), referentiekernel
      letterlijk naast de batched versie gehouden om transcriptiefouten in
      de WIDTH-16-reductieboom te vermijden. Bitexact (geïsoleerd + causale
      A/B, apart van V5's eigen resultaat), **+1,7423 ms/token (+6,11%)**
      bovenop V5. **V6 opnieuw gedraaid: 47,37 tok/s** (was 44,37), 28,7%
      van roofline. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "Up-proj
      ERVF-GEMV gebatcht".
- [DONE, NIET GEADOPTEERD 2026-08-16] **`gather_down_sparse_ind`/
      `gemv_down_masked_partial_ind` batchen.** **Correctie op een eerdere
      "race condition"-diagnose in dit bestand: die was fout.** Vervolgtest
      met écht gevangen modeldata (i.p.v. de synthetische random data die
      eerst NaN gaf) bewees beide kernels **bitexact, nul NaN**
      (`verify_down_gather_batch_real_full.py`,
      `verify_gather_batch_real_full.py`). Geïntegreerd en fysiek getest:
      in isolatie een echte winst (+0,6826 ms/token, +2,56%, bitexacte
      causale A/B). Toch **niet** in V6 opgenomen: (1) vereist `top_k`
      onafhankelijke mirror-buffers i.p.v. één hergebruikte → ~387 MB extra
      VRAM tegen een budget van 64 MiB, VRAM-poort faalde en is niet
      verruimd; (2) bij volledige V6-integratie verdween de winst binnen
      ruis (47,36 vs 47,37 tok/s zonder). `moe_dev_batched.py` heeft de
      optie (`gather_kernels=`, default `None`) klaarstaan voor als VRAM
      ooit geen blokkade meer is. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16,
      blok "Correctie: gather/down_masked batchen bleek WÉL correct".
- [DONE 2026-08-16] **Per-laag capaciteitstuning — fysiek gemeten en
      geïntegreerd.** Hitrate-diagnose (`diag_per_layer_capacity.py`):
      budget-neutrale herverdeling (−20 op 6 laagste-miss lagen, +30 op 4
      hoogste-miss lagen) geeft −14,3% missers (5182→4443), hitrate
      85,6%→87,7%. Causale A/B (`v_capacity_realloc_ab.py`, productiekernels,
      zelfde precedent als A1): bitexact, controle-arm wijkt af, **+0,2362
      ms/token (+0,75%)**, alle poorten groen. Geïntegreerd in V6
      (`pro_research/layer_capacity.py`, budget-neutraal dus **geen VRAM-
      kost**, in tegenstelling tot de gather/down_masked-poging). **Nieuw
      klein record: 47,41 tok/s.** Nog open: de −20/+30-verdeling was een
      eerste gok op één prompt/rollout, niet verder geoptimaliseerd. Zie
      `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "Per-laag cachecapaciteit
      fysiek gemeten en geïntegreerd".
      Zie `RESEARCH_NOTEBOOK.md` 2026-08-16.
- [DONE 2026-08-16] **Batch>1: houdt fetch-deling stand onder een warme,
      evoluerende cache, of was het cold-cache-artefact?** Alle eerdere
      `proto_batch_*`-prototypes maten met opzet één cold-cache-snapshot.
      `pro_research/diag_batch_warm_cache.py`: N=4 sequenties, T=40
      opeenvolgende **echte** stappen op MoE-laag 24, echte productie-
      `cache_assign`/`alloc_device_cache`-kernels (geen herimplementatie).
      GEDEELD (1 cache cap=72, gevoed met de unie van 4 sequenties se ids/stap)
      vs NAIVE (4 onafhankelijke caches, elk cap=72 — wat 4 losse
      batch=1-instanties vandaag hebben). **Resultaat: 142 vs 196 missers over
      960 aanroepen — 27,6% minder missers, 28,0% in het laatste kwart
      (stap 30-40, "warm steady-state").** Het voordeel verdwijnt dus **niet**
      naarmate de cache warmt, al is het kleiner dan de eerdere cold-cache-
      unie-cijfers (bijv. 90,3% van no-overlap bij N=4) deden vermoeden —
      cross-sequentie-deling concurreert met temporele lokaliteit die de
      NAIVE-arm ook al gratis krijgt zodra warm. Bijvangst: GEDEELD gebruikt
      1×72-slot cache tegen NAIVE se 4×72-slot — ~4× minder VRAM voor
      hetzelfde per-sequentie-budget, apart van de missers-winst. Read-only
      diagnostiek, geen runtime-wijziging, geen tok/s-claim. Zie
      `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "Warme-cache-dynamiek".
- [DONE 2026-08-16] **Batch>1: overleeft de expert-unie continuous batching
      (staggered posities), of was "alle N op dezelfde stap" een gunstige
      aanname?** Sluit risico #3 uit `BATCH_ARCHITECTURE_DESIGN.md`.
      `pro_research/diag_staggered_position_union.py`: N=4 sequenties,
      MoE-laag 24, T=30 wall-clock-ticks vergeleken tussen LOCKSTEP (zelfde
      stap-index, zoals elke eerdere meting) en STAGGERED (vaste offsets
      0/7/15/23, elke sequentie op zijn eigen echte generatiediepte) —
      beide views uit dezelfde onderliggende echte trajectdata, dus de
      staggering is de enige variabele. **Resultaat: 89,4% (lockstep) vs
      91,4% (staggered) van max-unie (24) — slechts +1,9 procentpunt groter,
      geen ineenstorting.** Consistent met `diag_cross_sequence_union.py`'s
      eerdere 90,3% voor N=4 (andere prompts/laag) — geen toevalstreffer.
      Continuous batching vernietigt het deel-potentieel dus niet, verzwakt
      het licht. Volledige runtime-integratie blijft ongebouwd. Read-only
      diagnostiek, geen tok/s-claim. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16,
      blok "Staggered posities".
- [DONE 2026-08-16] **Batch>1: exacte VRAM-kost per extra sequentie —
      sluit risico #4 uit `BATCH_ARCHITECTURE_DESIGN.md` met een echt getal.**
      `pro_research/diag_batch_vram_cost.py`: host-aritmetiek die
      `runtime.py`'s eigen `_alloc_state`-formules natrekt voor KV-cache
      (FP8, 6 attentielagen) en Mamba ssm+conv-state (FP32, 23 lagen).
      **60,16 MiB per extra sequentie — en Mamba-state domineert (48,2 MiB),
      niet KV-cache (12,0 MiB)**, tegen de gebruikelijke transformer-
      intuïtie in (dit model heeft maar 6 van 52 lagen volledige attentie).
      Bij het eager+device-cache-bedrijfspunt (geen graph): **1.771 MiB
      vrij → ruimte voor 29 extra sequenties (N tot 30)**, echt gemeten via
      nvidia-smi. Bij volledige V6-graph-capture: **0 MiB vrij** (bekend
      feit uit de V4-preregistratie, hier hergebruikt, niet opnieuw
      gemeten). **Herkadreert het risico:** niet batch>1 se eigen
      VRAM-kost is het probleem (60 MiB/sequentie is klein) — de
      graph-capture-overhead zelf eet al het budget op, vóór batch>1 er
      iets bij vraagt. Een eager (niet-graph-resident) batch>1-integratie
      zou ruim budget hebben; een graph-resident integratie moet eerst de
      graph-capture-kost verlagen. Read-only diagnostiek/aritmetiek, geen
      tok/s-claim. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "VRAM-kost
      per extra sequentie".
- [DONE 2026-08-16] **Batch>1: eerste gecombineerde meting (up_proj +
      down_proj-deling tegelijk, één laag).** `proto_batch_moe_layer.py` en
      `proto_batch_down_proj.py` bewezen elk apart, maar down_proj se meting
      deed zijn eigen up_proj-stap nog naïef. `pro_research/proto_batch_moe_layer_combined.py`
      combineert beide echt, N=8. **Methodologiebug gevonden en gefixt
      vóórdat er iets gerapporteerd werd**: eerste versie timede de
      NAIVE-arm alleen als GEMV (fetch buiten het venster) tegen BATCHED se
      fetch+GEMV — oneerlijk, gaf een vals "0,888× verlies". Gefixt (beide
      armen dezelfde productie-`cache_fetch` incl. in het getimede venster).
      **Resultaat na fix, bitexact 0/48 mismatches: 15,582 → 12,890 ms,
      1,209× (+20,9%), 2,692 ms bespaard** — reëel maar **kleiner dan de
      losse metingen deden vermoeden**. Opmerkelijk: down_proj se
      masked-GEMV werd zelf LANGZAMER in de gecombineerde meting (7,037 ms)
      ondanks gelijke FLOP's per sequentie en een snellere fetch — vermoedelijk
      slechtere geheugenlocaliteit door de grotere unie-mirror, een reëel
      interactie-effect tussen de twee mechanismen dat pas zichtbaar wordt
      als ze samen draaien. **Les: twee apart bewezen winsten optellen is
      geen vervanging voor ze samen meten** (werkregel 6 in de praktijk).
      Read-only prototype, geen tok/s-claim. Zie `RESEARCH_NOTEBOOK.md`
      2026-08-16, blok "Eerste gecombineerde meting".
- [DONE 2026-08-16] **Shared-expert N-schaling — "triviaal"-aanname
      geverifieerd, dit keer bevestigd (niet gecorrigeerd zoals Mamba).**
      `pro_research/diag_shared_expert_n_scaling.py`: bestaande shared-
      expert-GEMV N keer gedraaid tegen N echte activaties, N∈{1,2,4,8,16}.
      **ms/sequentie nagenoeg vlak (0,0378→0,0345), verhouding tegen ideaal-
      lineair 0,85-0,92 (iets efficiënter dan lineair, geen straf).**
      Bevestigt `BATCH_ARCHITECTURE_DESIGN.md` stap 6's aanname. Read-only
      diagnostiek. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "Shared-
      expert-schaling".
- [DONE 2026-08-16] **NIEUW risico gevonden — lm_head N-schaling, nooit
      eerder genoemd in `BATCH_ARCHITECTURE_DESIGN.md`.**
      `pro_research/diag_lmhead_n_scaling.py`: lm_head is de duurste GEMV
      van het model (output=vocab, 1,15 ms/aanroep bij N=1), niet
      expert-geselecteerd dus in theorie "triviaal" net als attentie/shared-
      expert — maar bleek dat NIET te zijn. **ms/sequentie stijgt van 1,154
      (N=1) naar ~1,38-1,43 (N=2-16), verhouding tegen ideaal-lineair
      1,19-1,24 — een grotere straf dan Mamba se eigen ~15% bij N=8-16, op
      de duurste GEMV in het model.** Consistent over N=2/4/8/16, geen
      ruis. Betekent: lm_head hoort bij "duurder per sequentie bij grotere
      N" net als Mamba, niet bij "vlak" zoals attentie/shared-expert —
      maakt de al-gecorrigeerde ~114 tok/s-bovengrens nóg iets
      optimistischer dan gedacht (geen nieuw getal berekend). Read-only
      diagnostiek. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok
      "lm_head-schaling".
- [DONE 2026-08-16] **Synthese van de vier N-schalingstests — patroon
      gevonden, geen nieuwe GPU-tijd gebruikt.** Attentie/Mamba/shared-
      expert/lm_head se resultaten naast elkaar gelegd: verhouding-tegen-
      ideaal-lineair correleert monotoon met ms/aanroep bij N=1
      (shared-expert 0,0378ms, 0,85-0,91x; attentie 0,0961ms, 0,96-0,97x;
      Mamba 0,1767ms, 1,15-1,16x; lm_head 1,1537ms, 1,19-1,21x). **Hoe
      duurder de kernel, hoe slechter de schaling bij herhaalde back-to-
      back-aanroepen** — geen twee losse toevalstreffers (Mamba, lm_head)
      maar één onderliggend patroon, vermoedelijk klok-/stroomthrottling of
      geheugencontentie (oorzaak niet vastgesteld, geen in-run klokmeting
      gedaan). Betekent: elke voldoende dure kernel verdient dezelfde check
      vóór hij als "triviaal lineair" wordt aangenomen. Zie
      `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "Synthese van de vier
      N-schalingstests".
- [DONE 2026-08-16] **Oorzaak van de N-schalingsstraf gevonden: reëel
      GPU-klokverval (36%) onder aanhoudende belasting, fysiek bevestigd.**
      `pro_research/diag_lmhead_throttle_check.py`: nvidia-smi gepold
      (5 Hz) tijdens 6 seconden aanhoudend lm_head-werk (272 batches N=16).
      **SM-klok daalt van piek 2685 MHz naar een stabiel niveau van
      ~1710-1717 MHz binnen ~1 seconde — een daling van 36%**, temperatuur
      steeg gestaag, stroom bleef vlak, `pstate` bleef P4 (dus boost-
      klok-afbouw, geen pstate-crisis). Groot genoeg om de eerder gemeten
      15-24%-tijdstraffen voor Mamba/lm_head volledig te verklaren
      (ondersteunend bewijs, geen exacte reconstructie — de oorspronkelijke
      vier schalingstests deden geen gelijktijdige klokmeting). **Belangrijke
      geruststellende correctie, zelfde dag: `clocks.mem` ook gepold —
      blijft exact 9001 MHz, geen afwijking.** Dit project se roofline is
      geheugenbandbreedte-gebonden (338,4 GB/s), niet reken-gebonden — dus
      deze SM-klok-throttling bedreigt de kern-roofline (165 tok/s ctx0) en
      V6's 47,41 tok/s-record NIET. Raakt alleen reken-zware kernels
      specifiek (lm_head, Mamba), niet de PCIe/HBM-streaming-hefbomen die
      dit project tot nu toe domineerden. Zie `RESEARCH_NOTEBOOK.md`
      2026-08-16, blok "Oorzaak
      van de supra-lineaire straf gevonden".
- [DEELS-DONE 2026-08-16] **EERSTE ECHTE END-TO-END N=2-METING — volledig
      model, meerdere stappen, bitexact geverifieerd.** Alles hiervoor was
      één MoE-laag of een geïsoleerde kernel-test; dit is de eerste keer
      dat het **echte, volledige 52-lagen model** meerdere **echte**
      decode-stappen draait voor N=2 sequenties met een **echt gemeten**
      aggregate tok/s-getal. `pro_research/proto_multi_seq_full_model.py`:
      per-sequentie DYNAMISCHE toestand (~30 buffers uit `_alloc_state()`,
      generiek gevangen via `getattr`, niet handmatig herimplementeerd) per
      sequentie gewisseld met `setattr` vóór aanroepen van de
      **ongewijzigde, echte** `rt.step()`; gewichten en MoE-device-cache
      blijven gedeeld. **Bug gevonden en gefixt vóór meting**: `pos` is een
      plain Python int die `step()` REBINDT (niet muteert) — zonder
      terugschrijven na elke stap zou wisselen-en-terug de KV-cache-
      leesoffset corrumperen. **Correctheidspoort**: N=2 volledig
      geïnterleaved (wissel-stap-wissel-stap, de exacte fase-3-patroon, niet
      slechts één ongebroken sequentie) tegen onafhankelijke
      `rt.reset()`-controleruns — **bitexact, 15/15 tokens per sequentie,
      beide sequenties**. **Resultaat bij 15 stappen (kale E1-fase-2.1-
      configuratie, geen graph/selectieve-ERVF/gebatchte-kernels, bewust
      schoon voor een N=1-vs-N=2-vergelijking): N=1 solo 29,798 tok/s vs
      N=2 naive (GEEN expliciete deel-logica, alleen incidenteel
      warm-cache-hergebruik) 31,411 tok/s aggregate — 1,054× (+5,4%), reëel
      en positief.** Sluit de vraag of het state-managementmechanisme
      praktisch werkt: ja. **[ROBUUSTHEIDSCONTROLE BIJ 40 STAPPEN, ZELFDE
      DAG]**: het cijfer krimpt naar **+2,05%** (31,656 tegen solo 31,020)
      bij een langere, representatievere horizon — consistent met
      `diag_batch_warm_cache.py`'s eigen cold-vs-steady-state-bevinding,
      geen tegenspraak. **+2,05% is het robuustere cijfer om te citeren,
      niet +5,4%.** Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok
      "Robuustheidscontrole van de N=2-naive-baseline".
      **[VERVOLGD EN AFGEROND 2026-08-16]** zie direct hieronder.
- [DONE 2026-08-16] **N=4 naive baseline — groeit het incidentele voordeel
      mee met N? Verrassend: nee.** `pro_research/proto_multi_seq_full_model_n4.py`,
      zelfde geverifieerde mechanisme als N=2, nu N=4. **Bitexact, 15/15
      tokens × 4 sequenties.** N=4 naive aggregate: **31,215 tok/s (1,047×,
      +4,7%)** tegen solo 29,820 — **vlak tot licht LAGER dan N=2's +5,4%**,
      ondanks dat losstaande diagnostiek (`diag_batch_warm_cache.py`,
      `diag_cross_sequence_union.py`) groei met N suggereerde. Meest
      aannemelijke verklaring: vaste cache-capaciteit (72) bij groter N geeft
      meer onderlinge eviction/contentie, wat de grotere theoretische
      overlap-kans compenseert. Nuanceert "meer N = meer incidenteel
      voordeel" — geldt niet zomaar. Niet gemeten: of een met N meeschalende
      cache-capaciteit dit zou herstellen. Zie `RESEARCH_NOTEBOOK.md`
      2026-08-16, blok "N=4 naive baseline".
- [WEERLEGD 2026-08-16] **N=8 naive baseline — de dalende trend wordt een
      INSTORTING, niet een vlakke lijn.** `pro_research/proto_multi_seq_full_model_n8.py`,
      zelfde geverifieerde mechanisme, N=8, 30 stappen. **Bitexact, 30/30
      tokens × 8 sequenties.** **Resultaat: 7,521 tok/s aggregate tegen
      solo 29,743 — 0,253×, 4× TRAGER dan één sequentie alleen.** Geen
      geleidelijke verslechtering meer zoals N=2→N=4, een echte instorting.
      Sluit aan bij twee andere N-schaal-regressies dezelfde dag (grotere
      cache-capaciteit: 0,706×; persistente warme cache + unie-deling:
      0,17×) — een coherent beeld: het naive gedeelde-cache-mechanisme
      (cap=72 vast) schaalt niet voorbij kleine N; bij N=8 kunnen tot 48
      experts/stap nodig zijn (bijna de hele cache), met hoge omloop. Geen
      bug, een capaciteit-versus-vraag-mismatch. **Sluit definitief "gewoon
      N verhogen" als pad naar hogere aggregate doorvoer met de naive
      aanpak.**
      **[MYSTERIE OPGELOST, ZELFDE DAG]**: `pro_research/diag_n8_cache_hitrate.py`
      leest `_dev_cache`'s eigen hit/miss-tellers uit — **hitrate bij N=8
      is HOGER, niet lager, dan solo (77,1% tegen 69,7%).**
      Cache-thrashing-hypothese **verworpen** (`thrashing_hypothesis_supported:
      false`). De instorting heeft dus NIETS met cache-missers te maken.
      **Samen met de eerdere sectieprofilering (de warme-cache-vertraging
      was globaal, ook zichtbaar in Mamba-lagen) wijst dit definitief naar
      Python-orkestratie-/kernel-launch-overhead van het state-
      wisselmechanisme zelf** — elke sequentie doorloopt alle 52 lagen apart
      (geen enkele niet-MoE-laag deelt ooit een launch), en N1
      (`N1_N5_OWN_HYPOTHESES_REPORT_2026-08-15.md`) mat al dat 23,7% van
      één token kernel-uitgifte is, geen rekenwerk — dat vermenigvuldigt
      zich met N (bij N=8: ~12.480 Python-`setattr`-aanroepen alleen al per
      decode-stap). Een architecturale eigenschap van deze prototype-bouw
      (aparte Python-aanroepen per sequentie per laag), geen bug — precies
      wat een CUDA-graph-integratie (routekaart-item 2) zou oplossen.
      **[KRITIEKE TEGENCONTROLE, ZELFDE DAG]**: een onafhankelijke
      `time.perf_counter()`-herhaling gaf GEEN instorting (tegenspraak!) —
      grondig uitgezocht (drie reconstructiepogingen faalden de instorting
      te reproduceren, ondanks regel-voor-regel-vergelijking), uiteindelijk
      opgelost door het ORIGINELE bestand direct te instrumenteren (geen
      reconstructie): `cp.cuda.Event()` en `time.perf_counter()` komen
      binnen dezelfde run PERFECT overeen (ratio 1,0000) — **de instorting
      is fysiek reëel, geen meetartefact.** De kleinere puzzel (waarom de
      reconstructie het niet reproduceerde) blijft open, eerlijk
      ongeverklaard, maar raakt de hoofdconclusie niet. Zie
      `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "N=8 naive baseline" en beide
      vervolgen direct erna.
- [WEERLEGD 2026-08-16] **Grotere cache herstelt het N=4-voordeel niet —
      hypothese verworpen, echte regressie gevonden.**
      `pro_research/proto_multi_seq_full_model_n4_bigcache.py`: cap 72→144
      (2×, matcht N=4/N=2) voor de N=4-arm, solo-controle bewust op
      standaard 72. **Bug gevonden en gefixt vóór meting** (`rt.reset()`
      per ongeluk verward met `rt.enable_cache()` in de correctheidspoort —
      zou dynamische toestand laten lekken tussen ground-truth-sequenties).
      **Bitexact, 15/15 tokens × 4 sequenties, PASS.** **Resultaat: 19,071
      tok/s aggregate tegen solo 27,013 — 0,706×, een ECHTE REGRESSIE, geen
      herstel.** Eerst voorgestelde verklaring (`cache_assign`'s lineaire
      eviction-scan wordt duurder bij grotere cap) **direct getoetst en
      WEERLEGD** (zie volgende item) — de regressie zelf staat vast, de
      oorzaak is open. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok
      "Grotere cache bij groter N".
- [WEERLEGD 2026-08-16] **Verklaring voor de bigcache-regressie getoetst en
      verworpen — geïsoleerde `cache_assign`-micro-benchmark.**
      `pro_research/diag_cache_assign_scan_cost.py`: cap ∈ {72,144,288,576},
      elke aanroep een gegarandeerde volle-cache-eviction (worst case voor
      de lineaire scan), 200 herhalingen per cap. **Kost per aanroep STIJGT
      NIET met cap — daalt licht** (0,1012→0,0695 ms van cap 72 naar 576).
      **De eerder voorgestelde verklaring (lineaire scan wordt duurder) is
      dus fout** — de regressie in `proto_multi_seq_full_model_n4_bigcache.py`
      blijft een geldige meting, maar de werkelijke oorzaak is **nog
      onbekend**, niet wat eerst gerapporteerd werd. Correctie toegepast op
      de eerdere claim. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok
      "Correctie, direct erna".
- [DONE 2026-08-16] **`PATH_TO_100_TOKS.md` routekaart-item 1 begonnen:
      device-only unie-berekening voor up_proj-fetch geverifieerd — geen
      nieuwe CUDA-code nodig.** `pro_research/diag_device_only_union.py`:
      bewijst dat de bestaande, ongewijzigde `cache_assign`/`cache_fetch`-
      kernels een RUWE (ongededupliceerde) N×top_k-idlijst al correct
      dedupliceren binnen één aanroep (nagelezen in de kernelbroncode, niet
      aangenomen) — elimineert de host-sync die de huidige Python-unie
      (`cp.asnumpy`+`set`+`dict`) nodig heeft, voor de up_proj-fetch-stap
      specifiek. **Bug gevonden en gefixt vóór een geldige meting**:
      `cache_fetch` leest `dev["ids"]` specifiek (niet de losse `ids`-
      parameter die aan `cache_assign` werd gegeven) — eerste versie liet
      dat leeg, waardoor alles expert 0 ophaalde (12/12 mismatches).
      Gefixt door `dev["ids"]` direct te vullen, het productiepatroon
      volgend. **Na de fix: bitexact, 0/12 mismatches, dedup-patroon
      correct (9 van 12 uniek).** Nog niet geïntegreerd in
      `proto_multi_seq_moe_shared.py` (geïsoleerde mechanismecontrole
      eerst, zoals de discipline vereist) en dekt nog niet de down_proj-
      maskerunie (apart, groter probleem). Geen nieuwe tok/s-claim.
      **[GEÏNTEGREERD, ZELFDE DAG]**: het mechanisme ingebouwd in de echte
      staplus (`route_topk` schrijft direct in één platte buffer,
      `cache_assign`/`cache_fetch` op de ruwe lijst, geen host-`set()`/
      `dict()` meer voor de up_proj-unie). **Bitexact op de robuuste
      40-stappen-schaal, 40/40 tokens × 2 sequenties, PASS.** **Timing:
      10,898 tok/s — een kleine REGRESSIE tegenover de vorige 11,234, geen
      winst.** Vermoedelijke oorzaak: de fetch-buffer is nu altijd
      P-groot (worst case) i.p.v. u-groot (werkelijke unie-grootte), plus
      een verse `dev_union`-allocatie per laag/stap — de bespaarde
      host-syncs wegen daar niet tegenop. **Les: minder host-syncs is niet
      automatisch sneller** als er een nieuwe, even grote kost tegenover
      staat (zelfde klasse les als eerder bij gather-batching).
      **[TWEEDE POGING, ZELFDE DAG]**: de gediagnosticeerde oorzaak
      (worst-case-P-grote buffer) direct gefixt — `cache_assign` levert
      zelf al een gepakte, gededupliceerde expertlijst
      (`expert_of[:filled]`, geen nieuwe kernel nodig). Fetch nu naar een
      werkelijke-`u`-grote buffer. **Bitexact opnieuw bevestigd, timing
      10,894 tok/s — vrijwel identiek, ook geen verbetering.** De
      voorgestelde oorzaak was dus OOK fout. **`proto_multi_seq_moe_shared.py`
      teruggezet naar de host-unie-versie (11,234 tok/s, het beste
      geverifieerde getal)** — beide device-only-pogingen blijven
      gedocumenteerd als eerlijke, tweemaal-bevestigde nulresultaten, niet
      verborgen. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok
      "Routekaartstap 1 begonnen" en beide vervolgen direct erna.
- [WEERLEGD 2026-08-16] **Derde poging — unie-deling + warme persistente
      cache over stappen gecombineerd. Bitexact, maar 6,5× REGRESSIE, geen
      winst, oorzaak onbekend.** `pro_research/proto_multi_seq_moe_shared_warmcache.py`:
      één persistente per-laag-cache (cap=72) i.p.v. een verse cache per
      aanroep, gebouwd vóór de decode-lus en hergebruikt over 40 echte
      stappen — combineert `diag_batch_warm_cache.py`'s eerder bewezen
      27,6%-missersreductie met de unie-deling-pijplijn, voor het eerst
      samen. **Bitexact, 40/40 tokens × 2 sequenties, PASS.** **Timing:
      1,725 tok/s — 6,5× TRAGER dan de 11,234-baseline, tegenovergesteld
      aan wat verwacht werd** (minder missers zou sneller moeten zijn, niet
      trager). **Oorzaak NIET vastgesteld** — de voor de hand liggende
      verklaring (eviction-scan duurder bij cap=72) is al tweemaal deze
      sessie getoetst en weerlegd (`diag_cache_assign_scan_cost.py`), dus
      bewust NIET herhaald zonder nieuwe toetsing. Een derde eerlijk
      nulresultaat, ditmaal sterk negatief.
      **[SECTIEGEPROFILEERD, ZELFDE DAG]**: `cache_assign`/`cache_fetch`
      **definitief onschuldig bevonden** (samen ~3% van de tijd) — de
      eviction-scan-hypothese nu een DERDE keer uitgesloten. Verrassing:
      "routing + shared expert" (code identiek aan de snelle versie, raakt
      de persistente cache niet aan) domineert onverwacht met **54,1%**
      (tegen ~12-13% in de niet-warme versie). Aannemelijke maar NIET
      geverifieerde kandidaat: de grote permanente `codes`/`scales`-
      buffer-reservering (~190 MB/laag × 23) maakt CuPy se geheugenpool
      trager voor kleine tijdelijke allocaties.
      **[DIRECT GETOETST EN WEERLEGD, ZELFDE DAG]**:
      `pro_research/diag_alloc_pressure.py` — losstaande micro-benchmark,
      kleine allocaties met/zonder een ~4,33 GiB permanente reservering.
      **Kleine allocaties worden juist SNELLER met de reservering (0,0363
      → 0,0197 ms, 0,54×), het tegenovergestelde van de aanname.** Drie
      op elkaar volgende, elk getoetste verklaringen voor de 6,5×-
      regressie zijn nu weerlegd (eviction-scan tweemaal, geheugendruk
      eenmaal). **De werkelijke oorzaak blijft onbekend** — eerlijk als
      open vraag gelaten, geen vierde ongeverifieerde gok.
      **[NOG FIJNER GEPROFILEERD, ZELFDE DAG]**: sectie 1 opgesplitst in 7
      losse kernels (allemaal <2% samen) én, voor het eerst, ook Mamba/
      attentie-lagen geprofileerd (lagen buiten `shared_moe_layer`, nooit
      eerder gemeten). **Belangrijkste bevinding: de vertraging is NIET
      gelokaliseerd in de MoE-cache-code — hij verschijnt OOK in Mamba-
      verwerking (27,0%) en de triviale MoE-add-back-stap (35,9%, de
      grootste losse post), code die niets met de cache-wijziging te maken
      heeft.** Wijst op een globaal, doorlopend effect (mogelijk
      werkelijk meer totaal PCIe-werk, of systeembrede contentie), niet
      een bug in één sectie. **Eerlijke grens bereikt**: sync-gebaseerde
      profilering (het enige beschikbare gereedschap) kan bij zo'n ongelijk
      verdeelde vertraging geen betrouwbare sectie-toewijzing meer geven —
      verder onderzoek vraagt echt Nsight Compute/Systems, buiten bereik
      hier. Zie `RESEARCH_NOTEBOOK.md` 2026-08-16, blok "Derde poging" en
      alle vervolgen direct erna.
- [WEERLEGD-VOOR-DEZE-AANPAK, CORRECTHEID BEVESTIGD 2026-08-16] **Expliciete
      unie-gevoede MoE-deling geïntegreerd in de echte staplus — bitexact
      correct, maar 12× TRAGER, niet sneller.**
      `pro_research/proto_multi_seq_moe_shared.py` integreert de al
      bitexact-bewezen deling uit `proto_batch_moe_layer_combined.py` over
      alle 23 MoE-lagen, meerdere echte stappen. **Correctheidspoort
      GESLAAGD**: bitexact tegen onafhankelijke `_moe_dev`-referentieruns,
      12/12 tokens × 2 sequenties. Bijvangst: bevestigt voor het eerst dat
      `gemv_into` en productie se `gemv_ervf_indirect` bitexact gelijk zijn
      (nooit eerder getoetst — eerdere prototypes vergeleken nooit tegen
      `_moe_dev` zelf). **Timing: 2,655 tok/s aggregate — tegen 31,411
      (naive) en 29,798 (solo) — een 12× REGRESSIE, geen winst.** Oorzaak
      duidelijk: de deling is **puur in Python** gebouwd (host-syncs via
      `cp.asnumpy()`/`.get()` per sequentie/unie-expert per MoE-laag per
      stap, honderden kleine allocaties/launches over 23 lagen × 12
      stappen) — precies de overhead die productie se `_moe_dev`
      (device-only routing, geen host-sync, gepijplijnde copy-stream)
      zorgvuldig vermijdt. **Het mechanisme is niet fout (bitexact bewezen)
      — een naïeve Python-orkestratie ervan wel.** Bevestigt exact wat
      `BATCH_ARCHITECTURE_DESIGN.md` van meet af aan zei: een echte
      integratie vraagt **echt CUDA-engineeringwerk** (device-only
      unie-routing-kernel, gebatchte launches over de unie, geen
      Python-host-syncs in de hete lus), niet Python-orkestratie van al
      bewezen stukken. **Vervolg, zelfde dag**: één overduidelijke
      inefficiëntie gevonden (numpy-data onnodig naar cupy geconverteerd en
      dan element-voor-element teruggelezen BINNEN een lus over `npanel`
      panelen — honderdduizenden overbodige host-syncs totaal). Gefixt
      (puur numpy houden voor de hostzijde-berekening). **Nog steeds
      bitexact, timing 2,655 → 9,469 tok/s aggregate (3,57× sneller)** —
      nog steeds 3,3× trager dan de naive baseline (31,411), dus nog geen
      nettowinst, maar bevestigt dat de overhead grotendeels vermijdbaar is,
      niet fundamenteel. **Sectiegeprofileerd (zelfde dag, `PROFILE`-vlag,
      geen correctheidsrisico)**: down_proj gather+masked+reduce is **48,9%**
      van de resterende tijd — verreweg dominant (routing+shared 12,0%,
      up_proj-fetch 17,0%, up_proj-GEMV+panel_scan 8,9%, unie-masker 11,4%,
      accumuleren 1,8%). **Concrete vervolghefboom, nog niet toegepast**: de
      al gebouwde en geverifieerde gebatchte kernels uit V5/V6
      (`gather_down_sparse_ind_batched`, `gemv_down_masked_partial_ind_batched`,
      `reduce_partials_batched`) toepassen op de unie-over-sequenties-
      dimensie i.p.v. per-paar losse launches — vereist herstructurering
      naar hun samenhangende buffervorm, een afgebakende taak, geen vage
      "meer engineering". Schone hertiming zonder profiling: 9,692 tok/s.
      **[VOLTOOID, ZELFDE DAG]** de al gebouwde gebatchte V5/V6-kernels
      (`gather_down_sparse_ind_batched`, `gemv_down_masked_partial_ind_batched`,
      `reduce_partials_batched`, `weighted_accumulate_ind_batched`)
      daadwerkelijk toegepast op de unie-dimensie. **Bitexact bij elke
      stap.** masked/reduce/accumulate batchen: nauwelijks effect op
      zichzelf (9,469→9,789) maar onthulde gather als nieuwe dominante
      kost (37,4% na herprofilering). Gather óók batchen: 10,17 → schoon
      **10,72 tok/s — 4,04× sneller dan de eerste werkende versie.**
      **Belangrijke fysieke les**: masked/reduce/accumulate zijn reken-
      gebonden (launch-batching hielp sterk); gather is
      PCIe-bandbreedte-gebonden (batching hielp nauwelijks — zelfde
      bytes over de bus, ongeacht launch-aantal, zelfde klasse
      beperking als de eerder weerlegde E2/NERVF-4-sporen). **Nog steeds
      2,93× trager dan de naive baseline (31,411)** — dat gat is nu fysiek
      verklaard (gather+up_proj-fetch ≈49% van de tijd, beide bandbreedte-
      gebonden, waarschijnlijk dicht bij hun vloer) i.p.v. vaag. Volgende
      hefboom zou PCIe-overlap met rekenwerk zijn (zoals graph-residentie
      voor batch=1 al doet), niet verdere kernel-batching.
      **[LAATSTE STAP, ZELFDE DAG]** unie-nz-lijstberekening gevectoriseerd
      met numpy-bit-trucs i.p.v. een geneste Python-lus (puur CPU-overhead,
      geen GPU-semantiekwijziging). Bitexact bevestigd. **Eindstand: 11,12
      tok/s, 4,19× sneller dan de eerste werkende versie**, nog 2,82× trager
      dan naive — het PCIe-gebonden gat blijft de grens van wat
      launch-batching/Python-vectorisatie kunnen oplossen. Zie
      `RESEARCH_NOTEBOOK.md` 2026-08-16, blok
      "Expliciete MoE-deling geïntegreerd in de echte staplus".
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
      exactheid en winst standhouden buiten korte rollouts. **Context
      2026-08-16:** `diag_lmhead_throttle_check.py` mat een reëel 36%
      SM-klokverval binnen ~1 seconde aanhoudende belasting, maar
      `clocks.mem` bleef exact 9001 MHz — de kern-roofline (geheugen-
      bandbreedte-gebonden) en V6's 47,41 tok/s-record zijn dus **niet**
      bedreigd. Relevant blijft: reken-zware kernels (lm_head, Mamba)
      kunnen in kortlopende metingen (30 rondes) een mix van boost- en
      duurzame klok bevatten — de Duurloop-taak zou dat voor die specifieke
      kernels kunnen bevestigen, maar is geen kritieke blokkade meer voor
      de hoofd-tok/s-cijfers.
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
