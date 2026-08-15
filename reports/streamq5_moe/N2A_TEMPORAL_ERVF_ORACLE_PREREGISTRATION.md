# N2A preregistratie — Temporal ERVF weight-reuse oracle

Datum: 2026-08-12. Output ongeopend.

## Hypothese

Voor S onafhankelijke tokenactivaties kan één ERVF-kernel iedere Q8/Q5-weight
éénmaal laden en daarmee S aparte virtuele accumulatorbomen bijwerken. Iedere
tokenboom behoudt dezelfde operatorvolgorde als S afzonderlijke ERVF-16-calls.

## Test

- S = 2, 4 en 8; vaste NumPy-seed `120821`;
- volledige fysieke Q8-plane en de bestaande 48×8-expert Q5-plane;
- Q5 is de optimistische zelfde-expertsetcontrol; echte route-unies worden
  afzonderlijk gerapporteerd en mogen niet uit deze timing worden weggelaten in
  een speculative claim;
- correctness tegen S afzonderlijke ERVF-16-calls;
- validation: 5 warmups, 30 metingen per S en plane;
- S=4 opent test alleen als beide planes bitexact zijn en de gezamenlijke
  kandidaat maximaal 80% van vier afzonderlijke calls kost;
- test: 10 warmups, 120 metingen; gezamenlijke ratio maximaal 75%, p95 maximaal
  80%.

Een pass bewijst alleen target-kernel weightreuse. Hij bewijst geen acceptance,
causale blockattention, echte route-unionkosten of end-to-end speculative tok/s.
