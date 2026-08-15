# RSIV-MoE P1B — V2 1.024→128 long-prefixscreen

**Verdict: `long_prefix_screen_negative_v2`.**

Ook na 1.024 prompttokens haalt dezelfde validation→testdiscipline de rank-32/page-faultgates niet. Promptlengte verklaart P1A dus niet.

## Bevroren validationkandidaat

- Rank: `32`.
- Threshold: `0.1`.
- Selectietype: `diagnostic_validation_failure`.
- Validation double-fast: `0.087%`.
- Validation cold reduction: `1.001×`.
- Test double-fast: `0.434%`.
- Test cold reduction: `1.007×`.

## Ruimste diagnostiek na lock

Rank 128 / threshold 0,10 bereikt op test `1.866%` double-fast en `1.033×` koude-bytereductie.

## Controles

- Capture: `True`.
- Rank/count/bound/full-rankprojectie: `True`.
- Vereiste upstream operatoridentiteit: `True`.
- Extra long-prefix FP32-extreemcheck (diagnostisch): `False`.

## Claimgrens

Dit blijft een teacher-state rank/page-faultscreen. Kwaliteit, packed runtime, SSD-latency en snelheid zijn niet gemeten; geen P1B-uitkomst is op zichzelf Eureka.

## Volgende actie

Houd V2-P2 gesloten en preregistreer de hogere-E-rankcensus op Qwen3-30B-A3B-Base vóór de checkpointdownload.
