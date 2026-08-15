# S6 — component breakdown at 262K on the masked runtime (preregistration)

Datum: 2026-08-14
Status: PREREGISTERED, voor elke meting.
Aard: diagnostische meetfase — **geen performance-poorten, geen tok/s-claims
buiten de volledig gemeten token**. Doel: de 73,1 ms @262100 van S5
toeschrijven aan componenten door meting, zodat de volgende hypothese op data
staat in plaats van op N8's (inmiddels verouderde) breakdown.

## Meting (bevroren)

Op de huidige masked runtime, cache 31 slots/laag, FP8 KV, synthetische
KV-populatie op diepte 262.100 + 32 warm-up-stappen met gevarieerde tokens
(zelfde methodiek als n7b/s5):

- per componentklasse, afzonderlijk getimed op diepte (p50, 20 reps):
  rmsnorm ×1, mamba ×1, attention ×1 @262100, router ×1,
  MoE ×1 hit-pad (zelfde token herhaald → alle hits),
  MoE ×1 gemengd (gevarieerde tokens → natuurlijke hit/miss),
  shared expert, LM head;
- volledige token gemeten op dezelfde diepte;
- unattributed = full − Σ(parts) als getal gerapporteerd, NOOIT benoemd
  (projectregel na de "glue"-fout);
- dezelfde breakdown ook op ctx 0 als controlegroep.

## Gates

- G-S6-1: unattributed is gerapporteerd en |unattributed| ≤ 20% van full,
  anders is de breakdown onvolledig en wordt de oorzaak gemeten (niet
  benoemd) vóór er een hypothese uit voortkomt.
- Geen performance-gates; dit is een census van kosten, geen verbetering.

## Claim boundary

Kostenverdeling per component op deze runtime, deze GPU, deze diepte, met
synthetisch gevulde KV (decode-stap-kosten op diepte, geen echte generatie
naar diepte). Geen tok/s-doelen, geen kwaliteitsclaims, geen projecties.
