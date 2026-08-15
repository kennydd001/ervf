# S5 — column-selective down_proj: eindrapport

Datum: 2026-08-14
Verdict: **Correctheidspoorten allemaal gehaald; prestatiepoorten P1/P2 gefaald zoals gepregistreerd. Het mechanisme is exact en blijft bestaan — de premisse dat MoE-miss-PCIe de dominante term is bij 262K is daarentegen weerlegd.**
Terminal state: `s5_masked_path_exact_perf_gates_failed`

Preregistratie: `S5_COLUMN_SELECTIVE_DOWN_PROJ_PREREGISTRATION_2026-08-14.md`
(incl. R1-addendum, design A2) · input lock `s5_input_lock.json`
Runner: `s5_masked_decode.py` · Verifier: `s5_independent_verify.py`
**Onafhankelijke verificatie: 9/9.**

## Wat gebouwd is (design A2)

- down_proj ligt **panel-major** in de host-bank: 116 panels × 16 kolommen;
  per panel 2.688 scale-bytes + 16 kolommen à 1.344 B contiguous. Pure
  byte-permutatie van de checkpoint-waarden.
- Cache en staging houden alleen de **up-helft**; down wordt nooit H2D
  gekopieerd. Miss = 2,81 MB in plaats van 5,61 MB.
- Per expert-call: `panel_scan` (deterministische compactie van niet-nul
  kolommen) → `gather_down_sparse` (warp-per-kolom, uchar4, gemeten 25 GB/s
  van mapped host naar een device-mirror) → ongewijzigde masked GEMV op de
  mirror. Alleen niet-nul kolommen en actieve scale-blokken steken PCIe over.

## Poorten

| poort | vereist | gemeten | |
|---|---:|---:|:--:|
| C1 identieke generatie (2×32 tokens) | exact | exact | ✅ |
| C2 transpose exact, alle 2.944 records | 0 afwijking | 0 | ✅ |
| C3 per-call rel_l2, 2.208 echte calls | ≤ 1e-6 | ≤ 1,88e-07 | ✅ |
| P3 ctx0 ≥ 21 tok/s | 21 | 21,759 | ✅ |
| **P1 ctx262100 ≥ 15 tok/s** | 15 | **13,678** | ❌ |
| **P2 ctx262100 ≥ 18 tok/s** | 18 | **13,678** | ❌ |

Context-sweep (masked runtime): 21,759 / 20,854 / 16,661 / 13,678 tok/s bij
0 / 32K / 131K / 262100 — vs gereproduceerde baseline 22,062 / 20,147 /
16,686 / 13,143. Effect: **+4,1% bij 262K**, ~0% bij korte context.

## Waarom de byte-besparing niet doorslaat (en wat dat betekent)

De bespaarde bytes zijn reëel (miss 5,61 → ~3,25 MB effectief), maar de
token-tijd bij 262K wordt gedomineerd door iets anders: N8 mat attention al op
**39,3 ms per token bij 262K** tegenover een FP8-KV-leesvloer van 3,2 ms
(805 MB @ ~250 GB/s) — de attention-kernel zit ~12× boven zijn roofline,
terwijl de MoE-term deels overlapt. 75,6 ms totaal = ~39 attention + ~36 rest.
Een 40%-bezuiniging op de MoE-miss-term raakt dus maar enkele milliseconden.

**Bij ctx 0 is het effect ~0%** — daar was de transfer kennelijk al vrijwel
volledig verborgen achter compute. De vloer-analyse van de opdracht
(10,6 ms PCIe + 10,4 ms compute) beschrijft de SOM van vloeren; de runtime
realiseert die overlap deels al.

Bijkomende winst die WEL is gerealiseerd: de cache is 1,86 GiB kleiner
(up-only slots) → 1,4 GiB VRAM vrij bij 262K-configuratie. Dat is munitie
voor een capaciteitsverdubbeling (31 → 62 slots/laag), die de hitrate en dus
het aantal misses verlaagt — een eigen fase, één variabele.

## Methodologische notities

- Variant A (masked kernel leest mapped host met 1 byte/thread) is als
  component weerlegd vóór de gegate run: 1,78 GB/s. Microbench toonde
  uchar4-reads op 25,05 GB/s ≈ memcpy-piek → design A2. Verspreide 1.344
  B-copies via de copy engine: 0,16 GB/s → variant B niet uitgevoerd.
- Eerste verifier-run faalde op W3 met rel_l2 = 1,1e+03: een spy die vóór de
  kernel call capturede en een hergebruikte buffer bij referentie vasthield.
  Verifier-protocol-negatief (zie Intel R8A5-precedent): de verifier is
  hersteld (compute-then-snapshot, kopieën) en beoordeelde daarna dezelfde
  artefacten: 9/9.

## Claim boundary

Gemeten batch-1 decode van de masked kolom-selectieve runtime op deze GPU,
met bit-identieke generatie aan de bevroren baseline en exacte transpose.
De prestatiemetrieken gelden deze machine, deze prompts, deze contexten.
Geen kwaliteits-, benchmark- of cross-hardware-claims. De conclusie
"attention domineert 262K" steunt op N8's componentmeting en moet voor een
volgende fase op de huidige runtime worden hermeten (S6).

## Artefacten

- `s5_masked_decode.json` · `s5_transpose_check.json` · `s5_independent_verification.json` (9/9)
- `s5_baseline_generation.json` · `s5_mapped_read_microbench.py`-metingen
- bronnen: `fused_nvfp4.py` (panel_scan/gather/masked GEMV), `runtime.py` (panel-major bank, up-only cache)
EOF
