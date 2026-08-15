# P8A preregistratie — Projection-Adaptive ERVF

Datum: 2026-08-12

## Hypothese

De globaal gekozen ERVF-breedte 16 kan lokale optima verbergen. Kies daarom op
de validation-run afzonderlijk een breedte uit `{8, 16, 32}` voor Q8
`q/k/v/o/router/head`, Q5 `gate_up` en Q5 `down`. Bevries die keuze vóór de
testmeting.

## Primaire grens

- Alle gekozen projecties zijn bitexact gelijk aan ERVF-16.
- De adaptieve volledige Q8-projectieplane én Q5-expertplane halen op de
  ongeopende testmeting `p50_ratio <= 0.97` en `p95_ratio <= 0.97` tegenover
  globaal ERVF-16.
- Validation: 5 warmups en 30 iteraties; test: 10 warmups en 120 iteraties.

Een mislukking is een geldig negatief resultaat en wordt niet post-hoc
hergedefinieerd.

