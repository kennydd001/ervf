# E1 fase 2.1 — Device-resident routing + device-LRU-cache (eager)

**Datum:** 2026-08-15 · **Status:** ALLE POORTEN GESLAAGD · **Verdict verifier:**
VERIFIED (14/14, 0 failed)

Preregistratie (bevroren vóór de meting):
`E1F21_DEVICE_ROUTING_PREREGISTRATION_2026-08-15.md`
Runner: `scripts/treesweep200/e1f21_device_routing_ab.py` +
`e1f21_inv_ctl_rerun.py` (re-run INV/CTL, zie "Bug onderweg")
Resultaten: `E1F21_DEVICE_ROUTING_AB.json`, `E1F21_INV_CTL_RERUN.json`
Onafhankelijke verifier: `scripts/treesweep200/e1f21_independent_verify.py` →
`e1f21_independent_verification.json` (importeert de runner nooit)

## Wat dit is

E1 fase 1 mat een budget van **8,925 ms per token** aan host-issue-overhead
(met ERVF aan). Fase 2 haalt die er in twee stappen uit:

- **2.1 (dit rapport):** de MoE-laag draait zonder één device→host-sync —
  routerkop, LRU-toewijzing en miss-staging lopen als kernels op device;
  de host lancéert alleen nog. Nog steeds eager (elke launch apart).
- **2.2 (volgt):** CUDA-graph-capture van de hele tokenlus bovenop 2.1.

Ontwerpkeuze uit de microbench (`E1F2_ZEROCOPY_MICROBENCH.json`): NIET "GEMV
leest zelf van host" (M2: 7,27 GB/s, marginaal), maar een **staging-kernel die
bulk vanuit de pinned bank naar een device-cache-slot kopieert** (M1-patroon:
24,93 GB/s = 96% van de DMA-engine's 26,03 GB/s). De host-DMA blijft bestaan
voor de host-gedreven arm; de device-arm gebruikt `cache_fetch`.

## Gemeten (2 prompts × 64 tokens, contexts_max=4096, V36-anker)

| arm | p50 ms/token | mean ms/token | pariteit vs A1-ids |
|---|---|---|---|
| BASE (geadopteerde stack, device_cache uit) | 41,540 | 41,829 | ✅ beide prompts |
| DEV (device routing+LRU, capacity 72) | **36,998** | 37,037 | ✅ beide prompts |

**Winst: −4,542 ms per token (−10,9%)**, eager, zonder graph-capture. Dat is
51% van het 8,925-ms-budget; de rest zit in launches die alleen een graph kan
wegvangen (fase 2.2).

## Poorten (bevroren in de preregistratie)

| poort | eis | uitslag |
|---|---|---|
| G-E1F21-C1 | token-pariteit DEV vs bevroren A1-ids | **PASS** (2/2 prompts, in beide runs) |
| G-E1F21-INV | capaciteit 56 ≡ capaciteit 72 (hit/miss mag numeriek niets veranderen) | **PASS** (na fix, zie onder) |
| G-E1F21-CTL | controle-arm `bad_pick=1` MOET pariteit breken | **PASS** (breekt, schone attributie) |
| G-E1F21-S1 | p50 DEV ≤ p50 BASE − 1,5 ms | **PASS** (−4,542 ms) |
| G-E1F21-V1 | extra device-tabellen < 32 MiB | **PASS** (analytisch 115.664 B ≈ 113 KiB voor 23 lagen + contrib; pool-delta 0 B — poolgranulariteit verbergt sub-MiB-allocaties, vandaar de analytische onderschrijving) |

## Onafhankelijke verificatie (14/14)

Naast het herrekenen van elke poort uit de ruwe JSON's draaide de verifier
eigen kernelchecks op synthetische data, zonder de runner te importeren:

- `route_topk_f32` ids exact gelijk aan een NumPy-referentie; gewichten binnen
  8,1e-8 relatief.
- `cache_assign` exact gelijk aan een Python-LRU-spiegel over 60 stappen
  (slots, need, slot_of, expert_of, last_used, tick, filled).
- `cache_fetch` bytes exact: opgehaald slot == bank-expert, 10 rondes met
  evicties.
- `gemv_nvfp4_ervf_ind` **bitexact** gelijk aan de directe ERVF-GEMV op
  hetzelfde record (productie-afmetingen 2688×1856).
- `weighted_accumulate_ind` **bitexact** gelijk aan `accumulate_into`.

## Bug onderweg (eerlijkheidshalve, en omdat hij in het log staat)

De eerste A/B liet INV falen. Oorzaak: `enable_cache(56)` herbouwde de
host-cache maar liet `_dev_cache` (device-LRU-tabellen mét live LRU-staat)
staan — de INV-arm draaide capaciteit-56-semantiek over vuile 72-slot-staat.
Dit was een harnessbug, geen kernelfout. Fix in `runtime.enable_cache`:
`self._dev_cache = {}` (de tabellen zijn per capaciteit gedimensioneerd en
staatsdragend, dus ze hóren mee te resetten). INV en CTL zijn daarna opnieuw
gedraaid met schone staat (`E1F21_INV_CTL_RERUN.json`): beide slagen. BASE en
DEV uit de oorspronkelijke A/B stonden al met schone staat en zijn onaangeroerd.
De CTL-arm in de eerste run faalde óók, maar kon toen nog aan de vuile staat
worden toegeschreven; pas de re-run geeft de sabotage-attributie die werkregel
8 eist. De verifier legt beide vast.

## Waar de winst vandaan komt (interpretatie, geen aparte meting)

Het host-gedreven pad deed per MoE-laag een route-readback (sync) plus
host-zijdig LRU-beheer; over 23 MoE-lagen is dat ~46 syncs per token. Die
syncs verdwijnen; de kernels zijn bewust reken-identiek gehouden (bitexact
tegen hun host-gedreven spiegel, zie verifier). De overgebleven ~4,4 ms van
het budget zit in pure launch-overhead en de niet-MoE-hoststappen (embedding,
argmax, pos) — exact wat fase 2.2 met graph-capture aanpakt.

## Claim boundary

- Gemeten op 2 anker-prompts × 64 tokens bij `contexts_max=4096`. De winst is
  per token en context-onafhankelijk van aard (het is host-overhead), maar dat
  is een aannemelijke extrapolatie, **geen** langecontextmeting — die staat
  open in de gedeelde TODO.
- −4,542 ms/token is een **component-integratie**-meting van de eager stack;
  dit is nog geen tok/s-claim en zeker geen 50 tok/s. Bij ctx 64 impliceert
  het ~37,0 ms/token (≈27 tok/s in dit regime).
- De device-arm verandert geen enkel getal dat het model berekent (pariteit +
  bitexacte kernelspiegels); de claim is uitsluitend "minder host-wachten".
- `bad_pick`-sabotage en de capaciteitsinvariantie zijn de enige
  controle-strumenten; een regressie die routes én output identiek fout maakt
  zou hierdoorheen glippen — vandaar de bitexacte kernelchecks in de verifier.

## Verder

Protected-manifest na deze fase (`protected_verification_after_e1f21.json`):
**0 removed, 0 modified door deze fase**; de tool rapporteert één
`content_changed` op `.gitignore` (127 → 962 B) die dateert van de eerste
git-commit van de repo-eigenaar (2026-08-15 21:49 — de baseline stamt uit een
pre-git tijdstip, `git_before.branch = null`). Geen enkel 80B-onderzoeks-
artefact is geraakt. Behandeling van die baseline-verversing is bij de
eigenaar belegd; deze fase zelf is schoon.

Fase 2.2: CUDA-graph-capture van de volledige token (CuPy 14.1.1 heeft
`Stream.begin_capture` + `graphInstantiate/graphLaunch`). Nog capture-compatibel
te maken: embedding-gather vanuit mapped host-tabel, argmax over logits,
pos-afhankelijke kernels (kv_write_fp8, attentie-splits) op device-pos.
Preregistratie mét poorten vóór de eerste meting.
