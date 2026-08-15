# NERVF_NEMOTRON — eindrapport van de ERVF-replicatie

Datum: 2026-08-15 · Namespace `NERVF_NEMOTRON` · append-only
Verifier: `nervf_independent_verification.json` — **66/66, VERIFIED**
Protected manifest na elke fase: **0 modified / 0 removed**

## De wetenschappelijke vraag, en het antwoord

> Kan Exact-Reduction Virtual Fusion dezelfde bitexacte reductie/occupancy-winst
> die op Qwen3-30B-A3B werd gemeten, reproduceren op een architecturaal andere
> moderne NVFP4 Nemotron hybrid-Mamba MoE?

**Ja.** 1,936× bitexact op het projectievlak, bij dezelfde gekozen subwarp-breedte
16 die Qwen selecteerde, op een ander model, een andere quantisatie (NVFP4 tegen
Q5/Q8) en een andere shape. Geïntegreerd levert het −3,7 tot −4,5 ms per token.

## Tabel

| | baseline | ERVF | delta |
|---|---:|---:|---:|
| raw scan GB/s (L2-koud) | 225,8 | — | referentie |
| kritieke GEMV GB/s | 72,7 | **140,8** | **+93,7%** |
| kritieke GEMV µs (1856×2688) | 38,58 | **19,93** | **1,936×** |
| MoE-blok ms/token @ctx0 | 23,711 | 21,060 | −2,651 |
| MoE-blok ms/token @262100 | 25,550 | 22,330 | −3,220 |
| token ms @ctx0 | 37,660 | **33,959** | **−3,701** |
| token ms @131072 | 46,780 | **43,678** | **−3,102** |
| token ms @262100 | 55,640 | **51,135** | **−4,505** |
| tok/s @ctx0 (binnen deze meting) | 26,55 | **29,45** | +10,9% |
| tok/s @262100 (binnen deze meting) | 17,97 | **19,56** | +8,8% |
| 512-token rollout p50, narrative | 41,487 | **36,882** | −4,605 |
| 512-token rollout p50, code | 41,049 | **36,719** | −4,330 |
| VRAM | ongewijzigd | ongewijzigd | 0 |
| numerieke verschillen (kernel) | — | **0 van 288** | bitexact |

## Fasen

| fase | uitkomst |
|---|---|
| **NERVF-0** baseline-lock | ✅ model, hashes, GPU, klokken, bevroren baseline |
| **NERVF-1** geometrie-audit | ✅ beide poorten: bandbreedte-efficiëntie **0,322** ≤ 0,40; reductie+sync **46,1%** ≥ 25% |
| **NERVF-2** microkernel | ✅ **alle vier de breedtes bitexact** (0/72 elk); w=16 **1,936×**; primair én sterk gehaald, moonshot 2,0× net niet |
| **NERVF-3** integratie | ✅ exact tegen het anker; token −3,7 tot −4,5 ms, alle diepten conclusief. Componentpoort 1,144× gefaald (verdund venster) |
| **NERVF-4** gatherless down | ❌ **weerlegd**: −6,0 tot −8,4 ms slechter. De gather van 8,19 ms verdient zichzelf terug |
| **NERVF-5** full model | ⛔ **gestopt op de eigen stopregel** — de productie-runtime blijkt niet run-to-run deterministisch |

## Doorbraakladder

- **LEVEL 2 gehaald** — ≥1,35× projectie, exact (1,936×).
- **LEVEL 3 niet gehaald** — het volledige expertpad bevat de down-projectie,
  die ERVF niet raakt en waar NERVF-4 de voor de hand liggende route sloot.
- **LEVEL 4 (≥35 tok/s) niet gehaald** — 29,45 tok/s bij ctx 0.

## De belangrijkste nevenvondst

NERVF-5 legde bloot dat `_moe_cached` de zes routed experts in
**hit-dan-miss-volgorde** accumuleert in plaats van in routevolgorde. Welke
expert een hit is hangt af van de LRU-staat, dus twee runs met een andere
cachegeschiedenis tellen in een andere volgorde op. Twee armen met **identieke
configuratie** divergeerden daardoor over 512 tokens.

Dat is geen ERVF-eigenschap — het geldt even hard met ERVF uit — maar het
kwalificeert elke bit-identiek-claim in deze lijn én in Kimi's E-lijn: die zijn
waar voor de runs waarin ze gemeten zijn (2 × 64 tokens), maar de eigenschap is
fragiel. Herstel is klein en precies beschreven: reken in hit-volgorde, bewaar de
zes bijdragen apart, tel op in routevolgorde — wat X1's `reduce_slots` al doet.
Dat is nu het hoogste openstaande punt op de lijst.

## Wat niet geclaimd wordt

Geen algemene nieuwheidsclaim: er is geen prior-art-audit, geen stock
llama.cpp-vergelijking en geen tweede GPU. Wat er ligt is een tweede
modelreplicatie op één GPU en één runtime. De winst mag niet worden opgeteld bij
attention-v4, graph of gatherless — elke combinatie vraagt een nieuwe A/B.

Alle percentages zijn microbenchmarks of in-lus componenten, behalve de
tokentijden en rollout-percentielen, die end-to-end wandtijd zijn. De
tok/s-omrekening geldt binnen de betreffende meting en is niet vergelijkbaar met
n7b's bevroren baseline.
