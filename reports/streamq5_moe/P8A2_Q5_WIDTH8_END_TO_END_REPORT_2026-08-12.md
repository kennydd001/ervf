# P8A2 Q5-width-8 — end-to-end-besluit

Datum: 2026-08-12

## Uitkomst

Smoke slaagde met dezelfde fysieke residentie als P7C. Validation bleef exact:
de next-token-CE (`1,9038253266041674`), prediction-digest, volledige missreeks
en KV-contextdata zijn identiek aan P7C.

De vooraf vastgelegde snelheidsgrens faalde:

| Metriek | P7C ERVF-16 | P8A2 Q5-8 | Ratio | Vereist |
|---|---:|---:|---:|---:|
| mean | 33,7137 ms | 33,9171 ms | 1,00603 | <= 0,98 |
| p95 | 45,4395 ms | 45,3789 ms | 0,99867 | <= 0,98 |

Daarom is de ongeopende test/rollout conform preregistratie niet uitgevoerd.
De losse Q5-plane-winst generaliseert niet naar de volledige decoder. ERVF-16
blijft de geselecteerde implementatie.

