# S7 — GQA-grouped attention: kill the 16× KV re-read (preregistration)

Datum: 2026-08-14
Status: PREREGISTERED, voor elke S7-meting.
Motivatie: S6-breakdown op de masked runtime @262100: attention ≈ 37,7 ms
(6 × 6,275 ms/laag) van de 75,7 ms-token — de dominante term. De FP8-KV
leesvloer is 805 MB/token ≈ 3,2 ms. De huidige kernel
(`attn_decode_warp_fp8`) lancert per QUERY-head (grid 32 × splits); bij
GQA met 32 q-heads / 2 kv-heads lezen tot 16 q-head-blokken dezelfde K/V-rij
opnieuw → tot 16× HBM-amplificatie. **Dit is "wat er beweegt": 4,3 GB →
805 MB per token als elke byte één keer per laag beweegt.**

## Stap 1 — mechanismecheck (voor er iets gebouwd wordt)

Tijd `attn_decode_warp_fp8` op diepte 262.144 met grid heads = 32 (huidig) en
heads = 2 (één q-head per kv-groep; zelfde splits, zelfde chunk). Voorspelling
als HBM-geamplificeerd: ~16× tijdsverschil. Als het verschil klein is (<4×),
is de amplificatie-hypothese fout en stopt S7 hier als negatief.

## Stap 2 — GQA-gegroepeerde kernel (alleen als stap 1 ≥4× toont)

Design (bevroren): grid = (n_kv, splits); block 128 = 4 warps; warp bezit een
positie; K-rij (128 B FP8) wordt één keer per positie per warp gelezen en voor
alle 16 q-heads van de groep gebruikt; q-vectoren (16 × 128 f32 = 8 KB) in
shared; per head accumulatoren in registers (4 dims/lane × 16 heads); online
softmax per head identiek aan de huidige vorm (twee `__expf`, geen branches —
de N8-meting dat branch-skip langzamer is, wordt gerespecteerd). Partials in
de bestaande `[h][splits*4][head_dim]`-layout zodat `attn_decode_combine`
ongewijzigd blijft. Reductie-orde per head ongewijzigd (zelfde warp-shuffle
boom); de enige semantische delta is dat K/V-bytes één keer bewegen.

## Gates (bevroren)

Correctheid:
- G-S7-C1: greedy generatie 2×32 tokens identiek aan `s5_baseline_generation.json`.
- G-S7-C2: attention-output van de gegroepeerde kernel vs de huidige kernel op
  echte decode-stappen: rel_l2 ≤ 1e-6 per laag, 64 stappen, alle 6 lagen.
Prestatie (zelfde harness als S5, capacity 31):
- G-S7-P3: ctx 0 p50 ≥ 21 tok/s (geen regressie).
- G-S7-P1 (minimum): ctx 262100 p50 ≥ 18 tok/s (S5-stand: 13,678).
- G-S7-P2 (primary): ctx 262100 p50 ≥ 22 tok/s.
Poorten worden niet verruimd; een gefaalde poort wordt vastgelegd.

## Claim boundary

Mechanisme-meting en, bij bouw, gemeten decode op deze GPU/contexten met
bit-identieke generatie. De kernel is GQA-bewuste herlezing van bestaande
bytes — gepresenteerd als gemeten amplificatie-verwijdering, geen claim van
nieuwigheid van het GQA-principe an sich. Geen kwaliteits-/benchmark-claims.
