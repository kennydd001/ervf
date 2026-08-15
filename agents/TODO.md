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

## Open

- [ ] **E1 fase 2 — de echte graph-resident token.**
  - [DONE 2026-08-15] **Fase 2.1 — device-resident routing + device-LRU
    (eager).** Alle vijf poorten PASS; p50 41,540 → **36,998 ms/token
    (−4,542 ms)**, pariteit behouden, verifier 14/14. Bugfix:
    `enable_cache` reset nu ook `_dev_cache`. Rapport:
    `E1F21_DEVICE_ROUTING_REPORT_2026-08-15.md`.
  - [ ] **Fase 2.2 — CUDA-graph-capture van de volledige token.** Budget dat
    over is: ~4,4 ms launch-overhead (van de 8,925 uit fase 1). MoE-pad is
    sync-vrij; nog capture-compatibel maken: embedding-gather uit mapped
    host-tabel, argmax over logits, pos op device (kv_write_fp8,
    attentie-splits/combine met vaste grid). **Preregistratie mét poorten
    vóór de eerste meting.**
- [ ] **Langecontext-profiel van de geadopteerde stack** — E6 mat 3 × 512 tokens
      bij `contexts_max=4096`. De stack is nooit end-to-end gemeten op 128K/262K
      ná adoptie. NERVF-3 deed dat vóór D1. Nu óók mét device_cache meten.
- [ ] **Duurloop** — ≥10.000 causale tokens en één thermisch uur, om te toetsen of
      exactheid en winst standhouden buiten korte rollouts.
- [ ] **Prior-art-audit + stock llama.cpp-differentieel** — nodig vóór er ooit een
      nieuwheidsclaim over ERVF wordt gedaan. Nu wordt die claim expliciet
      **niet** gemaakt.

## Doorlopend

- [ ] Na elke fase: `protected_manifest.py verify` (**0 modified / 0 removed**),
      registry-entry bijwerken, rapport schrijven, hier afvinken, en één blok
      toevoegen aan `RESEARCH_NOTEBOOK.md`.
