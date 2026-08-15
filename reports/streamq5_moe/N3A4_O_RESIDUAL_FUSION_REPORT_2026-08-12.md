# N3A4 — O-projectie→residual-fusie

Datum: 2026-08-12. Status: **bitexact, snelheidshypothese gesloten**.

## Uitkomst

De fused write bewaart beide BF16-grenzen exact, maar versnelt de fysieke
48-laagse O→residual-component niet materieel.

| Metriek | baseline | fused | ratio | validationpoort |
|---|---:|---:|---:|---:|
| mean | 2,7049 ms | 2,7073 ms | 1,0009 | diagnostisch |
| p50 | 2,6918 ms | 2,6902 ms | 0,9994 | ≤0,98 |
| p95 | 2,7800 ms | 2,7930 ms | 1,0047 | diagnostisch |

De p50-poort faalde; de afgesloten testhelft is daarom niet geopend.

## Exactheid

Alle 98.304 uiteindelijke FP32-state-elementen—48 lagen × 2.048—waren
bit-voor-bit gelijk, eindig en hadden een maximale absolute afwijking van nul.
De kandidaat voert in lane 0 nog steeds eerst `round_bf16(projected)` en daarna
`round_bf16(residual + projected)` uit.

## Interpretatie

Per laag leest de O-projectie 2.048 × 4.096 fysieke Q8-gewichten. Daartegenover
bespaart de fusie één kleine residual-launch en circa 8 KiB projected-write plus
8 KiB read. De gewichtenstroom en dotproducten domineren de componenttijd; het
weglaten van de scratchroundtrip verandert de gemeten p50 praktisch niet en
verslechtert p95 licht.

Hiermee is deze zelfstandige exact-fused write gesloten. Een grotere fusion die
ook attention-values of een andere O-kernelgeometrie omvat is een andere
hypothese en wordt niet door N3A4 uitgesloten.

Auditspoor:

- `N3A4_O_RESIDUAL_FUSION_PREREGISTRATION.md`;
- `scripts/streamq5_moe/run_n3a4_o_residual_fusion.py`;
- `n3a4_o_residual_fusion.json`;
- `scripts/streamq5_moe/verify_n3a4_o_residual_fusion.py`;
- `n3a4_o_residual_fusion_verification.json`.

Claimgrens: fysieke residentiële O→residual-component; geen volledige decoder,
kwaliteit, cross-GPU of SOTA-claim.
