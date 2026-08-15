# P8A2 preregistratie — Q5-width-8, Q8-width-16 end-to-end

Datum: 2026-08-12. Status bij vastlegging: geen P8A2-output geopend.

## Hypothese

P8A selecteerde op de volledige 48-laagse Q5-plane breedte 8 voor zowel
`gate/up` als `down`, met een bitexacte geïsoleerde testwinst. De Q8-selectie
faalde en blijft daarom ongewijzigd op de bewezen ERVF-breedte 16.

## Vergrendelde wijziging

- Q8 `q/k/v/o/router/head`: ERVF-16, identiek aan P7C.
- Q5 `gate/up/down`: ERVF-8.
- Alle codes, schalen, MAC-volgorde per virtuele thread, reductieboom,
  cache/transfers, router, attention/KV, sampling en evaluator blijven gelijk.

## Gates

1. Smoke, validation, test en 512-token rollout moeten alle bestaande P7C-
   kwaliteits-, residentie- en latentieplafonds opnieuw halen.
2. CE-arrays, voorspellingen, misses, KV-digests en rollouttokens moeten exact
   gelijk zijn aan P7C.
3. Validation en test moeten beide minstens 2% winnen op mean en p95 tegenover
   P7C (`ratio <= 0.98`).
4. De rollout moet niet regresseren: mean- en p95-ratio `<= 1.00`.

Een isolated-pass maar end-to-end-fail sluit deze kandidaat als niet robuust.

