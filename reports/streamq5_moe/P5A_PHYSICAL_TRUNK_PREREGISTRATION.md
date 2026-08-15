# STREAMQ5-MoE P5A — fysieke INT8-trunk preregistratie

Datum: 2026-08-12. Status: geen P5A-bank of timingoutput geopend.

## Hypothese

De echte INT8 attention/router/LM-head-projectieplane van Qwen3-30B-A3B kan
naast de fysieke expertcache en KV-reservering op de 8-GB-GPU verblijven en al
haar batch-1 GEMV's in mean <= 30 ms en p95 <= 35 ms uitvoeren.

Dit test de onbewezen trunkterm in `DATAPLANE_ONTLEDING_2026-08-12.md`; niet de
volledige attention- of decoderloop.

## Exacte matrixset

Voor ieder van 48 lagen:

- `q_proj [4096,2048]`, `k_proj [512,2048]`, `v_proj [512,2048]`;
- `o_proj [2048,4096]`;
- router `mlp.gate [128,2048]`.

Daarna `lm_head [151936,2048]`. Totaal 1.229.717.504 gewichten. Embedding blijft
host-resident en valt buiten de GPU-bank.

Alle matrices worden symmetrisch per rijgroep van 128 gekwantiseerd met
`qmax=127`; de maxabs-schaal wordt vóór dequantisatie naar BF16 afgerond,
identiek aan P0C. Codes zijn fysieke int8-bytes, schalen fysieke BF16-bits.

## Fasen en poorten

P5A-1 bouwt een immutable bank met recordoffsets, SHA-256 per bestand en exacte
code-/schaalaantallen. Een onafhankelijke verifier decodeert 15 vooraf
gedetermineerde records en vergelijkt codes en BF16-schalen opnieuw met de
brongewichten. Alle 15 moeten exact zijn.

P5A-2 voert een Q8 group-128 CUDA-GEMV uit. Per iteratie lopen alle vijf
laagmatrices voor 48 lagen plus de LM-head; q/k/v worden echt berekend,
`o_proj` gebruikt het q-resultaat als 4096-vector en zijn output voedt de
volgende laag/router. Dit is een projectieplane, geen attentionsemantiek.

- smoke: 3 iteraties; validation: 120; test: 360, eenmalig na validation-pass;
- 20 warmups voor beslissende splits;
- host-wand- en CUDA-eventtijd worden bewaard;
- sampled correctheid op 15 records: `max_abs <= 0.02`, `relative_l2 <= 1e-4`;
- bankcodes, schalen, offsets en bytes exact;
- co-resident: expertcache 4.977.623.040 bytes, fysieke trunkbank,
  KV 402.653.184 bytes, minimaal 384 MiB scratch;
- validation/test host mean <= 30 ms, p95 <= 35 ms;
- alle outputs en timings eindig;
- event/host p50-ratio in `[0.90,1.05]`.

## Beslissingsbetekenis

Een pass bewijst fysieke residentie en GEMV-wandtijd van alle echte
attention/router/headgewichten. Samen met P4A geeft dit een gemeten
expert+projectiebudget; RMSNorm, RoPE, attention score/softmax/value-reductie,
KV-lezen/schrijven, residuals, echte routertop-k/gewichten, sampling en
autoregressieve feedback blijven nog open. Een fail falsificeert de
roofline-gebaseerde trunkprojectie voor deze kernel.
