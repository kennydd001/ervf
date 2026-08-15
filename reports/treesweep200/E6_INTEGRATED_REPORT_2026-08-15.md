# E6 — geintegreerde fysieke run: exact, en 4,2 tot 4,9 ms per token

Datum: 2026-08-15 · Registry `TREESWEEP200`
Verdict: **De stack die daadwerkelijk gebouwd is — ERVF (breedte 16) + de v4-attentionkernel + deterministische accumulatie — levert geintegreerd 41.980 -> 37.490 ms per token over 512-token rollouts, met bit-identieke uitvoer. E6's eindpoort (>=50 tok/s kort) wordt niet gehaald: de twee grootste posten uit het plan, E1 fase 2 en E2, zijn ongebouwd.**
Terminal state: `e6_integrated_exact_gain_endgate_blocked_on_e1_e2`

## Wat er gecombineerd is

Geen componentpercentages opgeteld — dit is een **fysieke A/B** met drie armen
`base_a / integrated / base_b`, drie promptdomeinen, 512 causale tokens elk.

Beide armen draaien met `deterministic_accum` aan. Dat is een voorwaarde, geen
detail: zonder D1 hangt de optelvolgorde van de cachegeschiedenis af en zijn twee
armen niet eens onderling vergelijkbaar (NERVF-5).

De enige variabele is de stack: **ERVF** in plaats van `gemv_nvfp4_rows`, en
**`attention_fp8_gqa4`** in plaats van v1.

## Uitkomst

| domein | basis p50 | stack p50 | winst | drift | p95 | p99 |
|---|---:|---:|---:|---:|---|---|
| expository | 41.829 | **37.661** | **+4.169** | 2.486 | 44.516 -> 41.262 | 46.100 -> 43.144 |
| narrative | 42.110 | **37.667** | **+4.443** | 2.127 | 45.210 -> 41.457 | 47.371 -> 44.085 |
| code | 42.002 | **37.144** | **+4.858** | 1.019 | 45.861 -> 41.349 | 47.707 -> 42.792 |

| poort | uitslag |
|---|:--|
| **G-E6-C1** exactheid | OK — identiek over 3 x 512, beide armen deterministisch |
| **G-E6-P1** latency | OK — alle domeinen conclusief |
| **G-E6-M1** VRAM | OK — geen regressie |
| `exact_short_tok_s_ge_50` | GEFAALD — ~26.7 tok/s in dit rollout-regime |

## Waarom de eindpoort niet gehaald wordt, en wat er ligt

Het oorspronkelijke E6-plan combineerde **E1 + E2 + E4(v4) + E5**. Daarvan zijn
E4 en E5 gebouwd en zitten ze hierin. De andere twee niet:

- **E2** (gatherless downflow) is gemeten en **weerlegd** in zijn eenvoudige vorm:
  de gather weglaten kost 6,0 tot 8,4 ms extra, want het strided
  byte-per-thread-patroon van de masked GEMV haalt 6,7 GB/s over PCIe tegen 85,9
  vanaf device. De gather van 8,19 ms verdient zichzelf terug.
- **E1 fase 2** is ongebouwd. Fase 1 is wel af: het N1-oracle reproduceert op
  22,2 procent en stijgt **met ERVF aan naar 27,0 procent (8,9 ms)** — de
  uitgifte-overhead is per-launch en beweegt niet mee met snellere kernels, dus
  ERVF **vergroot** wat E1 kan opleveren.

Het budget voor E1 fase 2 staat daarmee vast op **8,9 ms per token**, en het
ontwerp moet dat verdienen zonder de miss-afhandeling duurder te maken dan E2
liet zien.

## Claim boundary

512-token causale rollouts op deze GPU bij capacity 72, drie domeinen, drie armen
zodat de herhaling de drift begrenst. Exactheid is een harde poort over elke
gegenereerde token en is gehaald. Latency is end-to-end wandtijd per token
inclusief synchronisatie; de context groeit tijdens de rollout, dus deze cijfers
zijn niet vergelijkbaar met n7b's bevroren vaste-diepte-baseline en de
tok/s-omrekening geldt alleen binnen dit regime. Dit is een geintegreerde run van
twee gebouwde componenten, geen optelsom van losse metingen.

## Artefacten

`scripts/treesweep200/e6_integrated_run.py` · `E6_INTEGRATED_RUN.json`
