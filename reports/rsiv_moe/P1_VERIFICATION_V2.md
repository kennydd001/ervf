# RSIV-MoE P1 onafhankelijke verificatie

**Verdict: `p1_verification_passed_with_declared_warning`.**

- Controles: 7744.
- Geslaagd: 7744.
- Fouten: 0.
- Gedeclareerde waarschuwingen: 1.

De verifier herberekent hashes, validationselectie, grididentiteiten, koude-byte-reciproken, rank/count/opslagbounds, operatorimagetoleranties en raw capturevormen. Het bevestigde empirische besluit is `screen_negative_v2`; er volgt geen runtime- of Eureka-claim.

## Waarschuwing

BF16 z regrouping is not bit-exact across GEMM batch shapes; Control Addendum 001 correctly keeps this diagnostic outside required gates.
