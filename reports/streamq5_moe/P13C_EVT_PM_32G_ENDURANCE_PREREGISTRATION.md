# P13C preregistratie — EVT-PM in de 32-GiB 10K-decoder

Datum: 2026-08-12. P13B isolated test is geopend en geslaagd; geen P13C-output.

## Hypothese

Integreer uitsluitend P13B `attention_scores_evt8`, exact softmax-
materialiseren en `attention_values_materialized` in P12R2. Pinned-Window
Streaming, Q8-staging, ERVF-16, banken, router, cache en KV-layout blijven
ongewijzigd.

## Gates

1. Echte 32-GiB Job-limit, 4K-KV en 10.000 tokens zoals P12R2.
2. Prediction-digest, volledige missreeks en 4K-KV-digest exact gelijk aan
   P12R2; alle attention-isolatie was reeds bitexact in P13B.
3. Gehele run: mean `<= 100 ms`, p95 `<= 150 ms`, p99 `< 110 ms`, `>= 10 tok/s`.
4. Thermiek wordt contextgepaard gemeten: posities 16–1015 van cyclus 3 versus
   dezelfde posities van cyclus 1; mean en p95 maximaal 110%.
5. Peak commit `<= 32 GiB`, system-pagefilegroei `<= 256 MiB`.

De contextgepaarde thermische grens vervangt de P12-vergelijking van ongelijke
contextposities; de volledige-run-latentiegates blijven ongewijzigd.

