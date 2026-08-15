# N3A3 — concat-QKV end-to-end preregistratie

Datum: 2026-08-12. Vastgelegd vóór de N3A3-runtime of output is geopend.

## Hypothese

De in N3A2 geselecteerde, bitexacte `concat_qkv`-kernel verlaagt de volledige
P13 EVT-PM-decodetijd wanneer uitsluitend de drie afzonderlijke Q/K/V
ERVF-16-launches per laag worden vervangen door één logisch aaneengesloten
ERVF-16-grid.

## Bevroren runtime en wijziging

- Eén fysiek geladen P13 EVT-PM-runtime met mapped/pinned expertbank.
- Exact dezelfde Q8-bank, Q5-kernels, EVT-PM-attention, router, cachepolicy,
  H2D-kopieën, normen en BF16-grenzen als baseline.
- Kandidaat wijzigt alleen Q/K/V-dispatch: drie grids van 256+32+32 blocks
  worden één grid van 320 blocks. De rekenvolgorde binnen iedere outputrij en
  alle Q/K/V-uitgangen zijn identiek.
- N3A2 moet `overall_pass=true` en `selected=concat_qkv` bevatten.

## Workload en same-runtime-pairing

- Verzegeld P7-testprompt, één geactiveerd `general`-cachedomein.
- 128 gepaarde fysieke decode-tokens; eerste 16 paren zijn timingwarmup.
- Voor ieder token starten baseline en kandidaat uit dezelfde router-LRU- en
  tellersnapshot.
- ABBA: even stappen baseline→kandidaat, oneven kandidaat→baseline.
- Een derde, ongetimede canonieke baseline bepaalt token en toestand voor de
  volgende stap; meetvolgorde kan de rollout niet sturen.

## Exactheidspoorten

Voor alle 128 paren moeten exact gelijk zijn:

1. next-token-prediction;
2. fysiek aantal expertcachemisses;
3. volledige KV-digest tot de huidige positie;
4. dynamische LRU-cachetoestand;
5. SHA256 van alle 151.936 FP32-logits;
6. SHA256 van de uiteindelijke 2.048-element state.

## Timingpoorten

Over de 112 paren na warmup:

- candidate/baseline mean `<=0,98`;
- p50 `<=0,98`;
- p95 `<=1,00`.

Alle exacte en timingpoorten moeten slagen voor `overall_pass`. Er wordt niet
post-hoc hertuned.

## Claimgrens

Dit is een gepaarde 128-token same-runtime-integratie op één GPU en één domein.
Geen 10K-endurance, tweede model/GPU, energie-, nieuwheids- of SOTA-claim.
