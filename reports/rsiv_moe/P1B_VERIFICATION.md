# RSIV-MoE P1B onafhankelijke verificatie

**Verdict: `p1b_verification_passed_with_declared_warning`.**

- Controles: 3905.
- Geslaagd: 3905.
- Fouten: 0.
- Gedeclareerde waarschuwingen: 1.

De audit bevestigt de 1.024→128-splits, hashes, validationselectie, beide grids, koude-byteboekhouding, rank/count/opslagbounds en raw capturevormen. Het resultaat blijft `long_prefix_screen_negative_v2`; kwaliteit en runtime zijn niet gemeten.

## Waarschuwing

The extra long-prefix FP32 maximum-absolute diagnostic exceeds the old sample-extreme tolerance; P1B Control Addendum 001 keeps it outside the preregistered required controls while preserving all values.
