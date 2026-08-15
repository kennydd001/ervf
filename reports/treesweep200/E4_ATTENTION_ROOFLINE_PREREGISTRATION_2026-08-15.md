# E4 — Attention Roofline Recovery — preregistratie (2026-08-15)

Bron: `info/NEMOTRON_TREESWEEP_200_ROOFLINE_V2_AGENT_PACK_2026-08-15/agents/21_ATTENTION_ROOFLINE_RECOVERY.md`.
Bevroren vóór elke meting van deze fase. Poorten worden na het zien van
resultaten niet verruimd.

## Geïmporteerde stand (ankers, niet opnieuw te bewijzen)

- N4: huidige fp8-GQA-decodekernel (`attn_decode_warp_fp8_gqa`, "v1") is
  byte-lineair: fit 21,48 ms/GB, intercept −0,033 ms, R² = 0,9964; effectief
  47,2 GB/s bij lange context — 7,2× onder de onafhankelijk gereproduceerde
  streaming-roofline (338,4 GB/s, P0/E0 deze sessie: 330,5 GB/s).
- v1 is compute-bound: ~80 warp-shuffles per positie (5-fase butterfly × 16
  query-heads per positie). Gemeten 3,07 ms/laag @262144.
- Config (gemeten uit `models/nemotron_3_5_lightning/config.json`): 52 lagen,
  hybride patroon met **6 attention-lagen** (`*`), n_heads=32, n_kv=2,
  head_dim=128, fp8-e4m3 KV (128 B/rij/cache → 256 B/positie/kv-head).
- Byte-volume per laag bij t: `bytes(t) = 2 caches × 2 kv-heads × t × 128 B`.
  Bij t = 262144: 134,22 MB/laag.
- In de broncode bestaat een ongeregistreerde, nooit gerunde kandidaat
  `attn_decode_warp_fp8_gqa2` ("v2"): twee lanes per query-head, lane-lokale
  64-dim dot + één `shfl_xor`, q in shared met stride 132. Deze wordt in deze
  fase voor het eerst geregistreerd, gecorrigeerd en gemeten.

## Contextset

`t ∈ {64, 4096, 32768, 131072, 262144}` (64 staat voor "ctx 0"; de kernel
vereist t ≥ 1 en korte context is niet het knelpunt).

## Metingen (allemaal componentniveau; nooit opwaarderen naar tok/s)

Standalone harness (geen model nodig): gefillde gefixeerde-seed random KV-caches
in fp8-e4m3-bytevorm, random q. CUDA-event timing, 20 reps, mediaan.

1. **Profiel-decompositie** op alle 5 contexten:
   - `raw_scan`: puur sequentieel lezen van K+V (bandbreedte-plafond van dit
     volume);
   - `addr_scan`: het exacte gqa2-adrespatroon (kv-head-basis + 128 B-rijen,
     hf-halften) zonder enige rekenkunde;
   - `qk_only`: gqa2 minus softmax en PV (dot + één shuffle, score-sink);
   - `qk_softmax`: plus online-softmax-update, zonder PV;
   - `full`: gqa2 ongemoeid;
   - `combine`: `attn_decode_combine` apart.
2. **v1-baseline** op alle 5 contexten (full-kernel + combine).
3. **Fit-reproductie**: v1-tijd vs bytes over de 5 contexten, lineaire fit.
4. **Correctheid v2 vs v1**: eind-`out` na combine, rel_l2 per context,
   3 seeds; bitwise determinisme over 2 runs van v2.
5. Alleen als G-E4-C1 en G-E4-S1 passen: adoptie in `runtime.py` achter
   env-schakelaar `LS_ATTN_KERNEL=gqa2` (default blijft v1) en in-lus meting
   met de bestaande meetrunner: attention-componenttijd bij 262100 en
   token-pariteit met het baseline-anker.

## Poorten (bevroren)

- **G-E4-F1** (fit): lineaire refit van v1 full-kernel-tijd vs bewogen bytes
  over de 5 contexten haalt R² ≥ 0,99.
- **G-E4-P1** (profiel): alle zes profielmetingen gerapporteerd op alle 5
  contexten; monotonie `raw_scan ≤ addr_scan` en `qk_only ≤ qk_softmax ≤ full`;
  de stage-deltas `(qk_only−addr_scan) + (qk_softmax−qk_only) +
  (full−qk_softmax) + addr_scan` sommeren tot ≥ 90% van `full` bij t=262144.
- **G-E4-C1** (correctheid): rel_l2(v2, v1) ≤ 3e-4 op alle 5 contexten × 3
  seeds; v2 bitwise identiek over 2 runs. (Motivatie 3e-4: identieke fp8-bytes,
  alleen sommatievolgorde verschilt; online-rescaling over 262144 stappen geeft
  ~√t·ε ≈ 3e-5 — de poort ligt 10× daarboven en 10× onder de fp8-kwantisatie-
  ruis van 2,45e-3.)
- **G-E4-S1** (eerste snelheidspoort): v2 full-pad (kernel + combine) ≥ 100
  GB/s effectief bij t=262144, dus ≤ 1,342 ms/laag.
- **G-E4-S2** (sterk): ≥ 169 GB/s bij t=262144, dus ≤ 0,794 ms/laag.
- **G-E4-T1** (in-lus, alleen bij adoptie): attention-component in de echte
  tokenlus ≤ 6,0 ms/token bij ctx 262100 (6 lagen × ≤1,0 ms), stretch ≤ 4,8 ms;
  én 64 gegenereerde tokens bit-identiek aan het v1-anker onder identieke
  prompt/cache-vulling.

## Beslisregels

- G-E4-C1 faalt → v2 niet adopteren; bug zoeken of v2 verlaten; geen
  tolerantie verruimen.
- G-E4-S1 faalt maar C1 slaagt → profiel gebruiken om de volgende kandidaat
  te kiezen (dubbele buffering, 4 posities/warp-iteratie, warp-specialisatie);
  elke nieuwe kandidaat doorloopt C1 opnieuw.
- G-E4-S2/T1 niet gehaald bij gehaalde S1 → adoptie alsnog toegestaan
  (strikt beter dan v1), maar de claim boundary noteert expliciet dat de
  sterke poort niet gehaald is.
- Claim boundary: alle getallen zijn componentmetingen op de attention-kernel;
  geen enkele wordt opgewaardeerd naar tok/s. Alleen G-E4-T1 meet in de echte
  lus en wordt als zodanig gerapporteerd.
