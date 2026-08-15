# P8B preregistratie — Scale-Broadcast ERVF

Datum: 2026-08-12

## Hypothese

Bij Q5 ERVF-16 verwerken de zestien lanes van een subwarp pakken uit dezelfde
128-gewichtenschaalgroep. De ongewijzigde BF16-schaalbits kunnen daarom door
lane 0 worden geladen en met een width-16 shuffle worden verspreid, zonder de
rekenvolgorde of afronding te wijzigen.

## Primaire grens

- Alle Q5 `gate/up/down`-uitvoer over 48 lagen is bitexact gelijk aan gewone
  ERVF-16.
- De volledige Q5-plane haalt op de testmeting `p50_ratio <= 0.97` en
  `p95_ratio <= 0.97`.
- Validation: 5 warmups en 30 iteraties; test: 10 warmups en 120 iteraties.

Geen bitexactheid betekent onmiddellijke falsificatie, ongeacht snelheid.

