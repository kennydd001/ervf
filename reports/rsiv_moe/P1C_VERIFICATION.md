# RSIV-MoE P1C onafhankelijke verificatie

**Verdict: `p1c_verification_passed_with_declared_warning`.**

- Controles: 4429.
- Geslaagd: 4429.
- Fouten: 0.
- Gedeclareerde waarschuwingen: 1.

Bevestigd onderzoeksverdict: `falsified_rank_working_set`. De audit controleert checkpointidentiteit, capturehashes en -vormen, validation→test-slot, beide grids, rankgroei, opslagbounds, selectieregel en de hard-falsificatielogica.

## Waarschuwing

Verifierpoging 001 gebruikte atol=0,002 voor de som van naar BF16 teruggecastte routergewichten. Dat was strenger dan de analytische BF16-eenheidsafronding eps/2=0,00390625. Addendum 001 vervangt alleen deze extra verifiercontrole; captures, selectie, grids, gates en onderzoeksresultaat zijn ongewijzigd.
