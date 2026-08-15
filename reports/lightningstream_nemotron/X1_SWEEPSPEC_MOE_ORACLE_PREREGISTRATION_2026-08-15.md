# X1 — SweepSpec MoE-oracle: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Aanleiding: `NEMOTRON_EXACTFLOW_AGENT_PACK` agent 06 (SweepSpec, prioriteit 2) en
`LIGHTNINGSPEC_50` P1. Beide zetten de target-only blockverifier voorop; K0 wees
dezelfde term aan als beslissend.

## 1. De vraag

Een blockverifier verwerkt `B` kandidaatposities in **één** gewichts-sweep. De
MoE-term is 39,523 van de 54,277 ms per token en is daarmee de enige term die
beslist of dat kan. K0 mat de rauwe belasting (unie 19,88 bij B=5, 6,4% boven
pariteit) en kwam negatief uit; S12 mat dat de MoE-term **niet**
expert-load-gebonden is (per-expert marginalen 12,23 ms van de 39,5). Die twee
lezingen verschillen een factor twee in wat een sweep kost.

Deze fase meet dat verschil weg. Niet de hele verifier — alleen de dominante
term, want als die faalt is de rest verspilde bouw.

```
T_moe_seq(B)    = B opeenvolgende token-MoE-passes, het huidige pad
T_moe_sweep(B)  = expert-major over de unie, elk record één keer geladen
```

## 2. Wat er gebouwd wordt

Eén nieuwe kernelfamilie, `sweepspec.py`, buiten `runtime.py`:

- `gemm_nvfp4_rows_b` — de bestaande `gemv_nvfp4_rows` met `B`
  activatievectoren: het gewicht wordt **één keer** gelezen en tegen `B` vectoren
  vermenigvuldigd. Zelfde decode, zelfde volgorde van de acht FMA's per uchar4.
- `gemm_down_masked_b` — de bestaande `gemv_down_masked_partial` met `B`
  activatievectoren en de **unie**-panelmasker. Kolommen waar een node nul heeft
  dragen exact nul bij, dus de unie verandert de waarde niet.

De unie-mask komt uit de bestaande `panel_scan` op `Σ_b act_b`: de activaties
zijn ReLU², dus niet-negatief, dus de som is nul precies waar alle nodes nul
zijn. Geen nieuwe scanlogica.

Bijdragen worden per `(node, slot)` bewaard en pas daarna in **route-volgorde**
gesommeerd, zodat de optelvolgorde per node identiek blijft aan het sequentiële
pad. Expert-major groepering mag de uitvoer niet veranderen.

## 3. Poorten

- **G-X1-C1 — batched == gemv bij B=1.** `gemm_nvfp4_rows_b` met `B=1` levert
  **bit-identieke** uitvoer aan `gemv_nvfp4_rows`. Faalt dit, dan is de kernel
  fout en worden er geen tijden gerapporteerd.
- **G-X1-C2 — sweep == sequentieel.** De MoE-uitvoer van alle `B` nodes uit het
  sweep-pad is **bit-identiek** aan het sequentiële pad, over ≥ 20 echte lagen ×
  echte routes × echte activaties. Exacte targetsemantiek is een voorwaarde, geen
  afweging (`G-S10-C1`).
- **G-X1-P1 — de beslissende verhouding.**
  `T_moe_sweep(5) / T_moe_seq(5)` wordt gemeten. De drempel volgt uit de
  rondebudgetten van het pack, niet uit een verzonnen getal: bij 3,114
  uitgestoten tokens per ronde is het sweep-pad winst zodra
  `T_moe_sweep(5) < 3,114 × T_moe_seq(1)`, dus zodra de verhouding tot het
  sequentiële vijftal onder `3,114/5 = 0,6228` ligt.
- **G-X1-D1 — drift.** Gebracketeerde basislijnen, zoals S12-R1. Een verhouding
  telt alleen als het verschil tussen de armen groter is dan de lokale drift.

Poorten worden na het zien van het resultaat niet verruimd.

## 4. Meetopzet

`B ∈ {1,2,3,4,5}`. Echte hidden states en echte routes, opgenomen uit een echte
greedy generatie op de bevroren prompts, dus geen synthetische activaties. Alle
23 MoE-lagen. Warm-up, dan p50 over herhalingen, gebracketeerd tegen basislijnen.

De cache blijft het bestaande up-only pad bij capacity 72; `down` komt zoals nu
uit mapped host via de gather. Eén variabele: token-major versus expert-major.

## 5. Wat deze fase niet doet

Geen Mamba-blockverificatie, geen GQA-sweep, geen gebatchte LM-kop, geen
speculatieve lus, geen commit/rollback. Die zijn pas zinvol als de MoE-term
meewerkt. `runtime.py` wordt niet gewijzigd. Er wordt niets naar tok/s omgerekend:
dit is een componentmeting, en de ronde-conclusie volgt pas als álle termen
gemeten zijn.

## 6. Artefacten

`src/moe_lab/lightningstream_nemotron/sweepspec.py` ·
`scripts/lightningstream_nemotron/x1_sweepspec_moe_oracle.py` ·
`x1_sweepspec_moe_oracle.json` ·
`scripts/lightningstream_nemotron/x1_independent_verify.py` ·
`x1_independent_verification.json` · rapport met claim boundary.

## 7. Claim boundary van dit document

Geen meting, geen resultaat.
