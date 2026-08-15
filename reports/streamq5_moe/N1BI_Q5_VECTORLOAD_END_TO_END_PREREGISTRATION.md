# N1BI — Q5-vectorload end-to-end preregistratie

Vastgelegd na de geïsoleerde N1B-componentpass en vóór deze integratiemeting.

## Hypothese

De bitexacte `aligned32x2`-Q5-load uit N1B verlaagt de volledige P13-decodetijd,
niet alleen de resident Q5-plane.

## Ontwerp

- Eén P13-runtime wordt geladen en één general-cache wordt geactiveerd.
- Na een vast 7-tokenprompt worden 128 fysieke decodes gemeten.
- Per token worden dezelfde toestand, KV-positie, router-LRU en invoertoken eerst
  gesnapshot. Baseline en kandidaat starten elk vanaf die snapshot; alleen de
  Q5-kernelfuncties wisselen.
- Volgorde is ABBA: even tokens baseline→candidate, oneven tokens
  candidate→baseline. Alleen de vooraf gekozen canonieke baseline-uitkomst
  voedt het volgende token.
- De eerste 16 paren zijn warmup en worden niet in timing gebruikt.
- Invoer komt uit de P7-testrollout. Q8, aandacht, cachekopieën en alle overige
  P13-semantiek blijven gelijk.

## Poorten

1. 128 voorspellingen, missenaantallen en KV-digests zijn exact gelijk.
2. Beide paden eindigen met dezelfde LRU-cachetoestand.
3. Na warmup is kandidaat/baseline mean `<=0.98`, p50 `<=0.98` en p95 `<=1.00`.

## Claimgrens

Dit is een gepaarde 128-token integratiemeting binnen één geladen runtime. Geen
10K-endurance-, andere-GPU-, andere-model-, kwaliteit- of SOTA-claim.
