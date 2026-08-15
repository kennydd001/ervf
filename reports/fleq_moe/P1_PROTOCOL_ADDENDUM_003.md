# P1 protocoladdendum 003 — immutable ternary-herhaalstate

Datum: 2026-08-11

## Probleem

De eerste uitvoering na addendum 002 produceerde geldige ternary-codebooks,
maar de verplichte herhaling van laag 0, expert 46 was niet bit-exact. Audit
toonde aan dat de officiële GSQ-quantizerconstructor de aangeleverde RTN-
initialisatietensors in-place normaliseert. Daardoor begon de tweede
optimalisatie niet uit dezelfde state als de eerste.

## Correctie vóór de overige experts

`optimize_gsq_expert` en `hard_gsq_initialization` geven voortaan clones van
gewicht en scales aan de upstream constructor. De caller-owned initialisatie
blijft daarmee immutable. Dataset, geselecteerde experts, seed, optimizer,
schedule, epochs, batchgrootte en gates zijn ongewijzigd.

De ongeldige uitvoering is bewaard als:

- `reports/runs/fleq_moe/p1_ternary_attempt_004/layer_00_expert_046.safetensors`
  (`5afb6789c8652d2258e91806d3a623cd4dda0e44da2fad854d8ee669d4534e6d`)
- `reports/fleq_moe/p1_ternary_experts_attempt_004/layer_00_expert_046.json`
  (`38e091c3d78f1befdbf2dbb07a86a489862b963e96f6296eabe61afb49af718d`)

Deze uitvoering telt niet mee. De gecorrigeerde herhaling sluit bit-exact voor
zowel alle harde gewichten als alle 60 losswaarden. Pas daarna zijn de overige
vijftien ternary-diagnostieken uitgevoerd.
