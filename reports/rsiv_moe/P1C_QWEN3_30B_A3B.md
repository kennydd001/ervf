# RSIV-MoE P1C — Qwen3-30B-A3B hogere-E rankscreen

**Verdict: `falsified_rank_working_set`.**

Zelfs rank 32 bij de ruimste geregistreerde threshold blijft op validation én test onder 80%; hard-falsificatieregel 1 is nu gereproduceerd op V2 en Qwen3.

## Vergrendelde kandidaat

- Rank/threshold: `4` / `0.001`.
- Validation: `0.000%` double-fast, `1.000×` cold-byte reductie.
- Test: `0.000%` double-fast, `1.000×` cold-byte reductie.

## Hard-falsificatiediagnostiek

- Rank 32 / threshold 0,10: validation `0.000%`, test `1.742%`.
- Rank 128 / threshold 0,10 test: `5.762%`, `1.108×`.

## Controles en claimgrens

Alle vereiste controls: `True`.
Dit is een teacher-state rank/page-faultscreen. Kwaliteit, echte cold I/O, latency en decode-snelheid zijn niet gemeten; dit is geen Eureka.

## Volgende actie

Sluit RSIV_MOE_V1 en zoek alleen onder een nieuwe, onafhankelijk vooraf geregistreerde hypothese verder.
