# FLEQ-MoE P1 — Qwen expert-streamed GSQ-smoke

**Verdict: `smoke_negative`. P2 geautoriseerd: `False`.**

## Kernresultaat

De officiële, hash-gepinde GSQ-codebookoperator draait deterministisch en ruim binnen het laptopgeheugen. De inhoudelijke 2-bitgate faalt echter hard: geen van de zestien vooraf geselecteerde experts verbetert op de ongeziene context tegenover zijn GPTQ-initialisatie.

| Laag | Vergelijking | Experts verbeterd | Gemiddelde verbetering | Gem. held-out gewogen relatieve MSE |
|---:|---|---:|---:|---:|
| 0 | 2-bit GSQ vs GPTQ | 0/8 | -28.90% | 0.1262 → 0.1557 |
| 0 | ternary GSQ vs RTN | 8/8 | 10.25% | 0.3115 → 0.2795 |
| 47 | 2-bit GSQ vs GPTQ | 0/8 | -105.84% | 0.0374 → 0.0590 |
| 47 | ternary GSQ vs RTN | 8/8 | 19.78% | 0.3475 → 0.2810 |

Laag 0 heeft 0/8 en laag 47 0/8 2-bitverbeteringen; vereist was minstens 6/8 per laag met minimaal 20% aggregate verbetering en zonder p95-regressie. Ternary verbetert RTN lokaal, vooral in laag 47, maar blijft absoluut veel onnauwkeuriger dan 2-bit GPTQ en mag de gefaalde primaire gate niet vervangen.

## Opslaggrens

2-bit codes plus BF16-group128-scales kosten analytisch 2,125 bpp en missen dus al de uiteindelijke ≤2,0-bpp-gate wanneer metadata wordt meegerekend. Ternary heeft een ideale cardinaliteitsbound van 1,710 bpp inclusief die scales, maar een gewone 2-bit pack kost eveneens 2,125 bpp; echte entropycoding en directe kernels zijn niet gebouwd.

## Controles en claimgrens

Alle verplichte uitvoeringscontroles: `True`. BF16-fallbacks zijn bit-exact, beide determinismeherhalingen sluiten, alle codes/scales/outputs zijn eindig en alle artifacthashes sluiten. Full-depth CE, benchmarkkwaliteit, rollouts, bitpacked artifactgrootte en runtime zijn niet gemeten. Dit is geen Eureka.

Volgens de preregistratie blijft P2 geblokkeerd. De bestaande GSQ-PTQ-reproductielijn sluit als `smoke_negative`; expert-trajectory-QAT mag alleen via een nieuwe, afzonderlijk gemotiveerde near-miss-preregistratie worden geopend.
