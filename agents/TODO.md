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
